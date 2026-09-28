"""Process captured frames and save the three lab figures for each scene.

The capture card stores samples in IIQQ order. This script converts them with
xwr.rsp.iq_from_iiqq, then runs a standalone range-Doppler, azimuth-elevation,
and cell-averaging CFAR pipeline for the AWR1843AOP.

Run on the machine that has the recordings:

    python process_data.py

Figures are written to figures/. Each recording also gets a figures/<name>.md
file with the chirp scale, the chosen frame, every frame's peak, and the
CFAR points. Frames are not averaged.

To plot a specific frame, pass its index from that markdown table:

    python process_data.py --frame static_moving=20 --frame moving_static=15
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from xwr.rsp import iq_from_iiqq

ROOT = Path(__file__).resolve().parent
C = 299792458.0
NUM_TX = 3
ANGLE_BINS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process recorded radar scenes.")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    parser.add_argument(
        "--threshold",
        type=float,
        default=8.0,
        help="CA-CFAR threshold as a linear power ratio.",
    )
    parser.add_argument(
        "--frame",
        action="append",
        default=[],
        metavar="NAME=INDEX",
        help="Plot this frame instead of the automatic choice. Repeatable.",
    )
    return parser.parse_args()


def frame_choices(items: list[str]) -> dict[str, int]:
    chosen: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--frame must look like static_moving=20, not {item!r}")
        name, raw = item.split("=", 1)
        chosen[name] = int(raw)
    return chosen


def load_recording(path: Path) -> tuple[np.ndarray, dict]:
    archive = np.load(path, allow_pickle=True)
    frames = archive["frames"]
    stored = archive["config"]
    text = stored.item() if getattr(stored, "shape", ()) == () else str(stored)
    cfg = yaml.safe_load(text)
    return frames, cfg["radar"]


def chirp_metrics(radar: dict) -> dict:
    """Same formulas as xwr.XWRConfig, without opening the radar."""
    sample_us = radar["adc_samples"] / radar["sample_rate"] * 1e3
    bandwidth_mhz = radar["freq_slope"] * sample_us
    range_res = C / (2 * bandwidth_mhz * 1e6)
    offset_us = radar["adc_start_time"] + sample_us / 2
    center_hz = radar["frequency"] * 1e9 + radar["freq_slope"] * offset_us * 1e6
    wavelength = C / center_hz
    chirp_us = (radar["idle_time"] + radar["ramp_end_time"]) * NUM_TX
    n_doppler = radar["frame_length"]
    max_speed = wavelength / (4 * chirp_us * 1e-6)
    speed_res = wavelength / (2 * n_doppler * chirp_us * 1e-6)
    return {
        "range_res": range_res,
        "max_range": range_res * radar["adc_samples"],
        "speed_res": speed_res,
        "max_speed": max_speed,
    }


def range_doppler(iq: np.ndarray) -> np.ndarray:
    """Complex cube, shape (doppler, tx, rx, range)."""
    spectrum = np.fft.fft(iq, axis=-1)
    return np.fft.fftshift(np.fft.fft(spectrum, axis=0), axes=0)


def angle_spectrum(rd: np.ndarray) -> np.ndarray:
    """Zero-pad the 4x3 AOP array to 32x32. Shape (doppler, el, az, range)."""
    virtual = np.swapaxes(rd, 1, 2)
    n_el, n_az = virtual.shape[1], virtual.shape[2]
    virtual = np.pad(
        virtual,
        ((0, 0), (0, ANGLE_BINS - n_el), (0, ANGLE_BINS - n_az), (0, 0)),
    )
    return np.fft.fftshift(np.fft.fft2(virtual, axes=(1, 2)), axes=(1, 2))


def angle_axis(n: int) -> np.ndarray:
    """Bin angle in radians for a half-wavelength array."""
    sine = np.clip(np.linspace(-1.0, 1.0, n), -1.0, 1.0)
    return np.arcsin(sine)


def ca_cfar(
    power: np.ndarray,
    guard: int = 4,
    train: int = 8,
    threshold: float = 8.0,
    min_range: int = 8,
) -> np.ndarray:
    """Cell-averaging CFAR on a (doppler, range) power image."""
    n_doppler, n_range = power.shape
    detections = np.zeros(power.shape, dtype=bool)
    radius = guard + train
    guard_slice = slice(train, train + 2 * guard + 1)
    n_train = (2 * radius + 1) ** 2 - (2 * guard + 1) ** 2
    # The training ring has to sit fully on the range axis. Doppler wraps.
    start = max(min_range, radius)
    stop = n_range - start
    for doppler in range(n_doppler):
        d_index = (np.arange(doppler - radius, doppler + radius + 1)) % n_doppler
        for rng in range(start, stop):
            r_index = np.arange(rng - radius, rng + radius + 1)
            window = power[np.ix_(d_index, r_index)]
            train_cells = window.copy()
            train_cells[guard_slice, guard_slice] = 0
            noise = train_cells.sum() / n_train
            if power[doppler, rng] > threshold * max(noise, 1e-12):
                detections[doppler, rng] = True
    return detections


def local_peaks(detections: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Keep a detection only when it is the brightest cell in a 3x3 neighborhood."""
    kept = np.zeros_like(detections)
    n_doppler, n_range = power.shape
    ys, xs = np.nonzero(detections)
    for doppler, rng in zip(ys, xs):
        d_index = (np.arange(doppler - 1, doppler + 2)) % n_doppler
        r_index = np.arange(max(0, rng - 1), min(n_range, rng + 2))
        if power[doppler, rng] >= power[np.ix_(d_index, r_index)].max():
            kept[doppler, rng] = True
    return kept


def points_from_peaks(
    ang: np.ndarray,
    detections: np.ndarray,
    metrics: dict,
) -> np.ndarray:
    """Cartesian points, columns x, y, z, velocity."""
    el_axis = angle_axis(ang.shape[1])
    az_axis = angle_axis(ang.shape[2])
    n_doppler = ang.shape[0]
    rows = []
    for doppler, rng in zip(*np.nonzero(detections)):
        el_i, az_i = np.unravel_index(
            np.argmax(np.abs(ang[doppler, :, :, rng])),
            ang.shape[1:3],
        )
        el = el_axis[el_i]
        az = az_axis[az_i]
        if abs(np.degrees(el)) > 50 or abs(np.degrees(az)) > 80:
            continue
        distance = rng * metrics["range_res"]
        speed = (doppler - n_doppler / 2) * metrics["speed_res"]
        rows.append((
            distance * np.cos(-az) * np.cos(el),
            distance * np.sin(-az) * np.cos(el),
            distance * np.sin(el),
            speed,
        ))
    if not rows:
        return np.zeros((0, 4))
    return np.asarray(rows)


def peak_at(power: np.ndarray, metrics: dict, min_range: int = 8) -> tuple[float, float, float]:
    """Strongest bin outside the coupling region. Returns range, speed, power."""
    usable = power.copy()
    usable[:, :min_range] = 0
    doppler, rng = np.unravel_index(np.argmax(usable), usable.shape)
    distance = rng * metrics["range_res"]
    speed = (doppler - power.shape[0] / 2) * metrics["speed_res"]
    return distance, speed, float(usable[doppler, rng])


def moving_peak(power: np.ndarray, metrics: dict, min_speed: float = 0.3) -> tuple[float, float, float]:
    """Strongest bin whose speed is outside the zero-Doppler clutter."""
    usable = power.copy()
    usable[:, :8] = 0
    half = int(np.ceil(min_speed / metrics["speed_res"]))
    center = power.shape[0] // 2
    usable[center - half : center + half + 1, :] = 0
    if usable.max() <= 0:
        return 0.0, 0.0, 0.0
    doppler, rng = np.unravel_index(np.argmax(usable), usable.shape)
    return (
        rng * metrics["range_res"],
        (doppler - power.shape[0] / 2) * metrics["speed_res"],
        float(usable[doppler, rng]),
    )


def show_scale(image: np.ndarray) -> np.ndarray:
    view = np.log10(image + 1e-12)
    cap = np.percentile(view, 99)
    return np.clip(view, None, cap)


def save_figure(
    path: Path,
    power: np.ndarray,
    ang: np.ndarray,
    cloud: np.ndarray,
    metrics: dict,
    title: str,
) -> None:
    n_doppler, n_range = power.shape
    extent_rd = [0, metrics["max_range"], -metrics["max_speed"], metrics["max_speed"]]
    az = np.degrees(angle_axis(ang.shape[2]))
    ra = np.max(np.abs(ang), axis=(0, 1))
    extent_ra = [0, metrics["max_range"], az[0], az[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].imshow(
        show_scale(power),
        origin="lower",
        aspect="auto",
        extent=extent_rd,
        cmap="viridis",
    )
    axes[0].set_title("Range-Doppler")
    axes[0].set_xlabel("Range (m)")
    axes[0].set_ylabel("Speed (m/s)")

    axes[1].imshow(
        show_scale(ra),
        origin="lower",
        aspect="auto",
        extent=extent_ra,
        cmap="viridis",
    )
    axes[1].set_title("Range-azimuth")
    axes[1].set_xlabel("Range (m)")
    axes[1].set_ylabel("Azimuth (deg)")

    if len(cloud):
        colors = axes[2].scatter(
            cloud[:, 0], cloud[:, 1], c=cloud[:, 3], cmap="coolwarm", s=18,
            vmin=-metrics["max_speed"], vmax=metrics["max_speed"],
        )
        fig.colorbar(colors, ax=axes[2], label="Speed (m/s)", fraction=0.046)
    axes[2].set_title("Bird's eye")
    axes[2].set_xlabel("Forward (m)")
    axes[2].set_ylabel("Lateral (m)")
    axes[2].set_xlim(0, metrics["max_range"])
    axes[2].set_ylim(-metrics["max_range"] / 2, metrics["max_range"] / 2)
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def choose_frame(powers: list[np.ndarray], metrics: dict, condition: str) -> int:
    if condition.startswith("static_static"):
        ranges = [peak_at(power, metrics)[0] for power in powers]
        return int(np.argmin(np.abs(np.asarray(ranges) - 2.0)))
    scores = [moving_peak(power, metrics)[2] for power in powers]
    return int(np.argmax(scores))


def verdict(condition: str, still: tuple[float, float], mover: tuple[float, float, float], noise: float) -> str:
    key = "_".join(condition.split("_")[:2])
    still_range, still_speed = still
    mover_range, mover_speed, mover_power = mover
    mover_visible = mover_power > 30 * noise
    reflector_ok = 1.2 <= still_range <= 3.2 and abs(still_speed) < 0.25
    if key == "static_static":
        if reflector_ok and not mover_visible:
            return "keep: reflector is near 2 m and the scene is still"
        return "redo: static scene should be a still peak near 2 m"
    if key == "static_moving":
        if reflector_ok and mover_visible:
            return f"keep: still reflector plus a mover at {mover_range:.1f} m, {mover_speed:.2f} m/s"
        return "redo: need the reflector at 0 m/s and a person off zero Doppler"
    if key == "moving_static":
        if abs(still_speed) > 0.3:
            return "keep: the brightest return left zero Doppler with the radar"
        return "redo: the radar move did not show up; the peak is still at 0 m/s"
    if key == "moving_moving":
        if abs(still_speed) > 0.3 and mover_visible and abs(mover_speed - still_speed) > 0.35:
            return "keep: two different speeds are visible"
        return "redo: need the reflector and the person at two different speeds"
    return "check the figure"


def write_results(
    path: Path,
    recording: Path,
    n_frames: int,
    raw_shape: tuple[int, ...],
    metrics: dict,
    index: int,
    powers: list[np.ndarray],
    cloud: np.ndarray,
    still: tuple[float, float],
    mover: tuple[float, float, float],
    noise: float,
    note: str,
) -> None:
    """One Markdown file of the numbers behind a scene's figure."""
    lines = [
        f"# {recording.stem}",
        "",
        f"Plotted frame **{index}** of {n_frames}. Frames are not averaged.",
        f"Raw shape `{raw_shape}`.",
        "",
        f"Check: {note}",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Range resolution | {metrics['range_res'] * 100:.2f} cm |",
        f"| Maximum range | {metrics['max_range']:.2f} m |",
        f"| Speed resolution | {metrics['speed_res']:.4f} m/s |",
        f"| Maximum speed | {metrics['max_speed']:.2f} m/s |",
        f"| Strongest peak | {still[0]:.2f} m at {still[1]:.2f} m/s |",
        f"| Off-zero peak | {mover[0]:.2f} m at {mover[1]:.2f} m/s |",
        f"| Median power | {noise:.3e} |",
        f"| CFAR points | {len(cloud)} |",
        "",
        "## Every frame",
        "",
        "The plotted frame is marked. `peak` is the brightest cell. "
        "`off-zero` is the brightest cell at least 0.3 m/s away from zero speed.",
        "",
        "| Frame | Peak range (m) | Peak speed (m/s) | Off-zero range (m) | Off-zero speed (m/s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, power in enumerate(powers):
        peak_range, peak_speed, _ = peak_at(power, metrics)
        off_range, off_speed, _ = moving_peak(power, metrics)
        mark = " ← plotted" if i == index else ""
        lines.append(
            f"| {i}{mark} | {peak_range:.2f} | {peak_speed:.2f} | "
            f"{off_range:.2f} | {off_speed:.2f} |"
        )
    lines.extend([
        "",
        "## CFAR points in the plotted frame",
        "",
        "| Forward (m) | Lateral (m) | Up (m) | Speed (m/s) |",
        "| --- | --- | --- | --- |",
    ])
    if len(cloud) == 0:
        lines.append("| — | — | — | — |")
    for point in cloud:
        lines.append(
            f"| {point[0]:.2f} | {point[1]:.2f} | {point[2]:.2f} | {point[3]:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def process_file(
    path: Path, out_dir: Path, threshold: float, chosen: int | None
) -> None:
    frames, radar = load_recording(path)
    metrics = chirp_metrics(radar)
    print(f"\n{path.name}: {len(frames)} frames, raw shape {frames.shape}")
    print(
        f"  resolution {metrics['range_res'] * 100:.2f} cm, "
        f"max range {metrics['max_range']:.2f} m, "
        f"max speed {metrics['max_speed']:.2f} m/s"
    )

    powers = []
    cubes = []
    for frame in frames:
        iq = iq_from_iiqq(frame)
        rd = range_doppler(iq)
        cubes.append(rd)
        powers.append(np.sum(np.abs(rd) ** 2, axis=(1, 2)))

    condition = path.stem
    if chosen is None:
        index = choose_frame(powers, metrics, condition)
    elif not 0 <= chosen < len(powers):
        raise SystemExit(f"{condition}: frame {chosen} is outside 0..{len(powers) - 1}")
    else:
        index = chosen
    power = powers[index]
    ang = angle_spectrum(cubes[index])
    detections = local_peaks(ca_cfar(power, threshold=threshold), power)
    cloud = points_from_peaks(ang, detections, metrics)

    still = peak_at(power, metrics)[:2]
    mover = moving_peak(power, metrics)
    noise = float(np.median(power[:, 8:]))
    note = verdict(condition, still, mover, noise)
    print(f"  frame {index}: strongest peak {still[0]:.2f} m at {still[1]:.2f} m/s")
    print(f"  off-zero peak {mover[0]:.2f} m at {mover[1]:.2f} m/s")
    print(f"  CFAR points {len(cloud)}")
    print(f"  {note}")

    figure = out_dir / f"{condition}.png"
    report = out_dir / f"{condition}.md"
    write_results(
        report, path, len(frames), frames.shape, metrics, index,
        powers, cloud, still, mover, noise, note,
    )
    save_figure(
        figure,
        power,
        ang,
        cloud,
        metrics,
        f"{condition}  frame {index}  peak {still[0]:.2f} m, {still[1]:.2f} m/s",
    )
    print(f"  saved {figure}")
    print(f"  saved {report}")


def main() -> None:
    args = parse_args()
    recordings = sorted(args.data.glob("*.npz"))
    if not recordings:
        raise SystemExit(f"No recordings in {args.data}")
    args.out.mkdir(exist_ok=True)
    chosen = frame_choices(args.frame)
    for path in recordings:
        process_file(path, args.out, args.threshold, chosen.get(path.stem))


if __name__ == "__main__":
    main()
