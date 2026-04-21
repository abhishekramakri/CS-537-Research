# CS 537 — Bandwidth-Efficient Keyword Spotting

Compares five feature-level transmission strategies for server-side keyword spotting on the Google Speech Commands dataset. Each approach differs in what gets sent over the network (raw audio, MFCCs, or a learned embedding) and when.

## Approaches

| | Name | What's transmitted | Bandwidth |
|---|---|---|---|
| A1 | Raw waveform | 16-bit PCM audio | ~32,000 bytes |
| A2 | Fixed MFCC | 13 MFCC coefficients | ~5,252 bytes |
| A3 | Event-triggered | MFCCs, only when voice detected | varies |
| A4 | Dynamic MFCC | Variable-resolution MFCCs | varies |
| A5 | Learned embedding | Compressed neural embedding | 64–512 bytes |

## Project Structure

```
data/download.py        download Google Speech Commands v2 (~2.3 GB)
models/cnn.py           CNN classifier shared by A1–A4
models/encoder.py       encoder + classifier for A5
train.py                train all models, saves weights to checkpoints/
approaches/             one file per approach — extract() and deserialize()
device.py               device side: extract payload, send over socket, record results
server.py               server side: receive payload, run model, return prediction
evaluate.py             load results/, compute metrics, generate Pareto plots
```

## Setup

```bash
pip install -r requirements.txt
git lfs install        # needed to pull model weights
python data/download.py
```

## Running

**Training** (already done, weights in `checkpoints/` via LFS):
```bash
/usr/bin/python3 train.py
/usr/bin/python3 train.py --embedding-dim 16
/usr/bin/python3 train.py --embedding-dim 32
/usr/bin/python3 train.py --embedding-dim 128
```

**Experiments** — run server on one machine, device on the other:
```bash
# server machine
/usr/bin/python3 server.py --approach a2

# device machine
/usr/bin/python3 device.py --approach a2 --host <server-ip>
```

Approach-specific flags:
- A4: `--resolution [high|medium|low]`
- A5: `--embedding-dim [16|32|64|128]`
- A3: `--vad-threshold 0.02` (optional, default 0.02)
- Add `--num-samples N` to run on a subset for quick testing

**Evaluation** (after running all approaches):
```bash
/usr/bin/python3 evaluate.py
```
Outputs a metrics table to terminal and saves plots to `plots/`.

## Notes

- Models are trained on Python 3.9 (`/usr/bin/python3`) — the package installs live there, not in the default `python3` on this machine
- `data/SpeechCommands/` and result JSONs are gitignored — dataset must be downloaded locally
- Model weights are tracked via Git LFS
