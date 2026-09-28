# 18-848 Lab 1

Radar capture and processing for an AWR1843AOP with a DCA1000EVM. The chirp is 10 fps, 128 range bins, and 128 Doppler bins, covering about 5.55 m and ±2.05 m/s.

## Layout

| Path | What it is |
| --- | --- |
| `config.yaml` | Chirp and capture-card settings |
| `data_stream.py` | Records one scene to `data/<scene>.npz` |
| `process_data.py` | Range-Doppler, angle, CFAR, and plots for each recording |
| `data/` | Raw frames |
| `figures/` | One `.png` and one `.md` per scene |
| `xwr/` | Capture library used by the scripts |

## Setup

On the Linux machine connected to the radar:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install xwr pyyaml matplotlib
```

Set the Ethernet interface to `192.168.33.30/24` and raise the receive buffer, as described in the [xwr user guide](https://radarml.github.io/xwr/usage/).

## Record

Leave the corner reflector at 2 m. Each command counts down 3 seconds, then records for 8 seconds.

```bash
python data_stream.py static_static
python data_stream.py static_moving
python data_stream.py moving_static
python data_stream.py moving_moving
```

## Process

```bash
python process_data.py
```

This writes `figures/<scene>.png` and `figures/<scene>.md`. The markdown file lists every frame's peak range and speed. Frames are not averaged.

To plot a chosen frame, use the index from that table:

```bash
python process_data.py --frame static_moving=20
```

To compare the local range and speed equations, and one angle cube, with `xwr`:

```bash
python process_data.py --check
```

`process_data.py` calls `xwr` only for `iq_from_iiqq`, which unpacks the capture card's IIQQ byte order. The FFTs and CFAR are local NumPy.
