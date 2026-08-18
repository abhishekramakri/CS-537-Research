# Bandwidth-Efficient Keyword Spotting

This project studies a basic tradeoff in edge/server speech systems: a small device (phone, smart speaker, sensor) hears a 1-second audio clip and needs the cloud to tell it what word was said. The device could send the raw audio, or it could do some of the work locally and send something smaller. The question is how much bandwidth you can save by preprocessing on the device, and how much accuracy you give up for it.

We test this on keyword spotting (KWS): classifying a 1-second clip into one of 35 spoken words from the Google Speech Commands v2 dataset (plus background noise, labeled "silence"). Five approaches (A1 through A5) each choose a different point on the bandwidth/accuracy tradeoff, by changing what gets computed on the device versus the server and what actually goes over the wire.

## The setup

Two machines talk over a raw TCP socket (`server.py` and `device.py`). The "device" reads a test clip, runs whatever local processing its approach calls for, and sends the result as a byte payload with a 4-byte length header. The "server" reads the payload, reconstructs a tensor, runs it through a trained model, and sends back a 4-byte class index. Round-trip time and payload size are logged for every sample, so we get real (not simulated) latency and bandwidth numbers.

All five approaches are graded on the same 35-way classification task, so the comparison is apples to apples: same test set, same 16kHz audio, same class labels, just different device-side processing and different wire formats.

## The five approaches

**A1: Raw waveform.** The device does nothing but re-quantize the float32 waveform to 16-bit PCM and send all 16,000 samples (`approaches/a1_raw.py`). The server does the MFCC extraction itself before running the model. This is the baseline: maximum bandwidth (~32,000 bytes), zero device-side compute, and it represents the "dumb device, smart cloud" end of the spectrum.

**A2: Fixed MFCC.** The device computes 13 MFCCs (the classic feature set from the speech recognition literature) locally and sends those instead of raw audio (`approaches/a2_fixed_mfcc.py`). This is a huge bandwidth cut (~5,252 bytes, about 6x smaller than A1) for a small amount of on-device compute, and it is the standard baseline that A3, A4, and A5 are all variations on.

**A3: Event-triggered transmission.** Same 13-coefficient MFCC as A2, but before extracting or sending anything, the device runs a voice activity detector (Silero VAD) on the clip (`approaches/a3_event.py`). If no speech is detected, nothing is sent at all, the payload is `None` and the device just records a "skipped" result. The idea: most of what a listening device hears is silence or background noise, and there is no reason to pay for a round trip if there is nothing to classify. On the keywords-only test set every clip actually has speech in it, so A3 looks identical to A2. Its advantage only shows up in "mixed" mode (see below), where hundreds of pure background-noise chunks are mixed in and A3 correctly suppresses transmission for most of them.

**A4: Dynamic-resolution MFCC.** Same idea as A2, but the resolution of the MFCC (number of coefficients and how many time frames, controlled by the hop length) is chosen at run time (`approaches/a4_dynamic.py`). Three fixed presets: `high` (40 coefficients, 10ms hop), `medium` (20 coefficients, 20ms hop), `low` (13 coefficients, 40ms hop). Coarser hop length means fewer time frames, so both axes shrink the payload. A4 exists to show what happens when you deliberately throw away resolution for bandwidth, giving a 3-point curve to compare against A5's continuous one. Since the server doesn't know in advance which resolution a given payload uses, A4 prepends an 8-byte header (`n_mfcc`, `n_frames`) so the server can reshape the tensor correctly.

**A5: Learned embedding.** Instead of a hand-designed feature like MFCC, the device runs a small CNN encoder locally and sends its output, a compressed embedding vector, directly (`approaches/a5_embedding.py`, `models/encoder.py`). The encoder and the server-side classifier are trained jointly, end to end, so the embedding is optimized specifically for this classification task rather than being a general-purpose audio representation. The embedding dimension (16/32/64/128) is a direct bandwidth knob: at dim 16 the payload is just 64 bytes, about 500x smaller than raw audio. This is the "push more intelligence to the edge" end of the spectrum, and it's the approach most likely to win on the accuracy-per-byte tradeoff because it learns exactly what's worth keeping instead of relying on a generic transform like MFCC.

## Models

Two model families, defined in `models/`:

- `models/cnn.py` has `KWSModel`, a compact CNN (3 conv blocks, global average pooling, then a small classifier head) shared by A1 through A4. It always takes an MFCC-shaped tensor `(batch, 1, n_mfcc, n_frames)` regardless of how many coefficients or frames there are, since global average pooling makes the spatial size irrelevant.
- `models/encoder.py` has `Encoder` (device-side, MFCC in, embedding out) and `EmbeddingClassifier` (server-side, embedding in, class logits out), used only by A5. These are trained together as one pipeline (`EncoderClassifierPipeline` in `train.py`) so gradients flow from the classification loss all the way back through the encoder, which is what makes the embedding task-specific.

`train.py` trains `KWSModel` once (covers A1-A4, since they all feed the same model at inference time regardless of feature resolution) and trains one encoder/classifier pair per embedding dimension. Checkpoints are saved to `checkpoints/` and tracked with Git LFS since they're binary weight files.

## Two test modes

Every approach can be run in two modes, producing separate result files:

- **Keywords-only**: the 35-class test split from Speech Commands, nothing else. This is the fair accuracy comparison, every approach sees the same clips and every clip has an actual word in it.
- **Mixed** (`--mixed` flag): the same keyword clips plus roughly 400 chunks of pure background noise from the dataset's `_background_noise_` folder, labeled "silence." This is where A3 actually differentiates itself. On keywords-only data A3 has to transmit every clip (there's always speech), so it looks the same as A2. With background noise mixed in, A3 correctly detects that the noise chunks aren't speech and skips transmission for most of them, which is the entire point of event-triggered sending. Evaluation counts a skipped transmission as "correct" when the true label is silence (correct suppression) and as "wrong" when the true label was an actual keyword (a missed detection).

## Evaluation

`evaluate.py` reads every JSON file in `results/`, computes per-approach metrics (accuracy, macro F1, average payload bytes, average device-side compute time, average round-trip latency, and transmission rate), prints a comparison table, and generates three plots in `plots/`:

- `accuracy_vs_bandwidth.png`: every approach/config plotted as accuracy against average bytes sent.
- `accuracy_vs_latency.png`: same idea but against measured round-trip time.
- `pareto_curves.png`: A4's three resolution presets connected as a line, A5's four embedding dimensions connected as another line, with A1/A2/A3 plotted as fixed reference points. This is the main result plot, it directly compares the shape of A4's discrete accuracy/bandwidth curve against A5's, and shows where the fixed baselines fall relative to both.

## Project layout

```
data/download.py        downloads Google Speech Commands v2 (~2.3 GB)
models/cnn.py            KWSModel, shared classifier for A1-A4
models/encoder.py        Encoder + EmbeddingClassifier for A5
train.py                 trains KWSModel and all four A5 encoder/classifier pairs
approaches/               one file per approach, each exposing extract() and deserialize()
device.py                device side: runs extract(), sends payload, records RTT and result
server.py                server side: receives payload, runs deserialize() + model, replies
evaluate.py               reads results/, computes metrics, writes plots/
```

## Running it

Setup:
```bash
pip install -r requirements.txt
git lfs install        # needed to pull model weights
python data/download.py
```

Training (already done for this repo, weights live in `checkpoints/` via LFS, only needed if retraining):
```bash
/usr/bin/python3 train.py
/usr/bin/python3 train.py --embedding-dim 16
/usr/bin/python3 train.py --embedding-dim 32
/usr/bin/python3 train.py --embedding-dim 128
```

A single experiment runs the server on one machine and the device on another:
```bash
# server machine
/usr/bin/python3 server.py --approach a2

# device machine
/usr/bin/python3 device.py --approach a2 --host <server-ip>
```

Approach-specific flags:
- A4: `--resolution [high|medium|low]`
- A5: `--embedding-dim [16|32|64|128]`
- A3: `--vad-threshold 0.5` (optional, Silero VAD speech probability threshold from 0.0 to 1.0)
- `--num-samples N` limits the run to a subset, useful for a quick smoke test

Full two-machine sweep, what actually produces the results used in evaluation. On the server machine, run each in its own terminal (or kill and restart between runs, they all bind port 9999):
```bash
/usr/bin/python3 server.py --approach a1
/usr/bin/python3 server.py --approach a2   # also serves a3, same model and feature format
/usr/bin/python3 server.py --approach a4
/usr/bin/python3 server.py --approach a5 --embedding-dim 16
/usr/bin/python3 server.py --approach a5 --embedding-dim 32
/usr/bin/python3 server.py --approach a5 --embedding-dim 64
/usr/bin/python3 server.py --approach a5 --embedding-dim 128
```

On the device machine (replace `<server-ip>`), start the matching server first each time:
```bash
# keywords-only
/usr/bin/python3 device.py --approach a1 --host <server-ip>
/usr/bin/python3 device.py --approach a2 --host <server-ip>
/usr/bin/python3 device.py --approach a3 --host <server-ip>
/usr/bin/python3 device.py --approach a4 --resolution high   --host <server-ip>
/usr/bin/python3 device.py --approach a4 --resolution medium --host <server-ip>
/usr/bin/python3 device.py --approach a4 --resolution low    --host <server-ip>
/usr/bin/python3 device.py --approach a5 --embedding-dim 16  --host <server-ip>
/usr/bin/python3 device.py --approach a5 --embedding-dim 32  --host <server-ip>
/usr/bin/python3 device.py --approach a5 --embedding-dim 64  --host <server-ip>
/usr/bin/python3 device.py --approach a5 --embedding-dim 128 --host <server-ip>

# mixed mode, same servers, just add --mixed to each device command
/usr/bin/python3 device.py --approach a1 --host <server-ip> --mixed
/usr/bin/python3 device.py --approach a2 --host <server-ip> --mixed
/usr/bin/python3 device.py --approach a3 --host <server-ip> --mixed
# ... same pattern for a4 and a5
```

Results land in `results/` as `<tag>.json` (for example `a2.json`, `a4_medium.json`, `a3_mixed.json`). Copy the whole `results/` folder onto one machine before evaluating.

Evaluation, after all approaches have been run:
```bash
/usr/bin/python3 evaluate.py
```
Prints the metrics table and writes the three plots to `plots/`.

## Notes

- Models were trained on Python 3.9 (`/usr/bin/python3`); the package installs live there, not in the default `python3` on this machine.
- `data/SpeechCommands/` and the result JSONs are gitignored, the dataset has to be downloaded locally and results regenerated locally.
- Model weights are tracked via Git LFS.
