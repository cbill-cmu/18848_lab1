"""Record one radar scene to data/<condition>.npz.

Run this on the Linux machine connected to the radar, from this folder:

    python data_stream.py static_static
    python data_stream.py static_moving
    python data_stream.py moving_static
    python data_stream.py moving_moving

The corner reflector stays at 2 m for every condition. A 3 second countdown
runs before the radar starts, then frames are recorded for 8 seconds.
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from queue import Empty

import numpy as np
import yaml

import xwr

ROOT = Path(__file__).resolve().parent
CONDITIONS = (
    "static_static",
    "static_moving",
    "moving_static",
    "moving_moving",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one radar scene.")
    parser.add_argument(
        "condition",
        choices=CONDITIONS,
        help="Which of the four lab scenes this recording is.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Radar and capture-card configuration.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="How long to record after the countdown.",
    )
    parser.add_argument(
        "--countdown",
        type=float,
        default=3.0,
        help="Seconds to wait after you press Enter, before the radar starts.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if "radar" not in cfg or "capture" not in cfg:
        raise SystemExit(f"{path} must contain 'radar' and 'capture' sections.")
    return cfg


def print_metrics(awr: xwr.XWRSystem) -> None:
    cfg = awr.config
    print("Chirp check")
    print(f"  frame rate          {awr.fps:.1f} fps")
    print(f"  range bins          {cfg.adc_samples}")
    print(f"  Doppler bins        {cfg.frame_length}")
    print(f"  bandwidth           {cfg.bandwidth:.1f} MHz")
    print(f"  range resolution    {cfg.range_resolution * 100:.2f} cm")
    print(f"  maximum range       {cfg.max_range:.2f} m")
    print(f"  Doppler resolution  {cfg.doppler_resolution:.4f} m/s")
    print(f"  maximum speed       {cfg.max_doppler:.2f} m/s")
    print(f"  raw frame shape     {cfg.raw_shape}")


def countdown(seconds: float) -> None:
    print("Get into position. The reflector stays at 2 m.")
    remaining = int(np.ceil(seconds))
    for step in range(remaining, 0, -1):
        print(f"  starting in {step}")
        time.sleep(1)


def take_frame(
    frame, raw_shape: tuple[int, ...], frames: list[np.ndarray], timestamps: list[float]
) -> bool:
    """Copy one complete frame. Return True when the frame is skipped."""
    if not frame.complete:
        return True
    sample = np.frombuffer(frame.data, dtype=np.int16)
    expected = int(np.prod(raw_shape))
    if sample.size != expected:
        return True
    frames.append(sample.reshape(raw_shape).copy())
    timestamps.append(frame.timestamp)
    return False


def collect(awr: xwr.XWRSystem, seconds: float) -> tuple[list[np.ndarray], list[float], int]:
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    skipped = 0
    raw_shape = awr.config.raw_shape
    queue = awr.qstream(numpy=False)
    deadline = time.perf_counter() + seconds
    try:
        while time.perf_counter() < deadline:
            try:
                frame = queue.get(timeout=0.2)
            except Empty:
                continue
            if frame is None:
                break
            skipped += take_frame(frame, raw_shape, frames, timestamps)
            if len(frames) % 10 == 0 and len(frames) > 0:
                print(f"  {len(frames)} frames")
    except KeyboardInterrupt:
        print("Stopping early. Saving frames collected so far.")
    finally:
        awr.stop()

    while True:
        try:
            frame = queue.get(timeout=2.0)
        except Empty:
            break
        if frame is None:
            break
        skipped += take_frame(frame, raw_shape, frames, timestamps)
    return frames, timestamps, skipped


def output_path(condition: str) -> Path:
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    path = folder / f"{condition}.npz"
    if path.exists():
        stamp = datetime.now().strftime("%H%M%S")
        path = folder / f"{condition}_{stamp}.npz"
    return path


def save(
    path: Path,
    frames: list[np.ndarray],
    timestamps: list[float],
    cfg: dict,
) -> None:
    np.savez(
        path,
        frames=np.stack(frames),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        config=np.array(yaml.safe_dump(cfg)),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(args.config)

    print(f"Scene: {args.condition}")
    print(f"Config: {args.config}")
    input("Press Enter when the lane is clear and you are ready. ")
    countdown(args.countdown)

    awr = xwr.XWRSystem(**cfg)
    print_metrics(awr)
    print(f"Recording for {args.seconds:.0f} seconds.")
    frames, timestamps, skipped = collect(awr, args.seconds)
    if not frames:
        raise SystemExit(
            f"No complete frames saved ({skipped} incomplete). "
            "Check Ethernet IP 192.168.33.30, the USB port, and power."
        )

    path = output_path(args.condition)
    save(path, frames, timestamps, cfg)
    print(f"Saved {len(frames)} frames to {path}")
    print(f"Frame shape {frames[0].shape}, skipped {skipped} incomplete frames.")


if __name__ == "__main__":
    main()
