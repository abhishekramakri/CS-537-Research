import socket
import struct
import time
import json
import os
import glob
import argparse
import torch
import torchaudio

from approaches import a1_raw, a2_fixed_mfcc, a3_event, a4_dynamic, a5_embedding
from models.encoder import Encoder
from train import SpeechCommandsDataset, get_mfcc_transform, LABELS

RESULTS_DIR = "results"
BG_NOISE_DIR = os.path.join("data", "SpeechCommands", "speech_commands_v0.02", "_background_noise_")


def load_background_noise():
    samples = []
    for wav_path in glob.glob(os.path.join(BG_NOISE_DIR, "*.wav")):
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(0, keepdim=True)  # stereo → mono
        n_chunks = waveform.shape[-1] // 16000
        for j in range(n_chunks):
            samples.append((waveform[:, j * 16000:(j + 1) * 16000], "silence"))
    return samples


def recv_exact(sock, n):
    """Read exactly n bytes from socket, blocking until all arrive."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server closed connection unexpectedly")
        buf += chunk
    return buf


def send_payload(host, port, payload: bytes) -> tuple[int, float]:
    """
    Sends payload to server and waits for prediction.
    Returns (predicted_class_index, round_trip_time_seconds).
    Protocol: [4-byte payload length][payload] → server → [4-byte class index]
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))

        header = struct.pack("!I", len(payload))
        t0 = time.perf_counter()
        sock.sendall(header + payload)

        response = recv_exact(sock, 4)
        rtt = time.perf_counter() - t0

    predicted_idx = struct.unpack("!I", response)[0]
    return predicted_idx, rtt


def run(approach, host, port, num_samples, resolution, embedding_dim, vad_threshold, mixed):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    test_ds = torchaudio.datasets.SPEECHCOMMANDS("data", subset="testing", download=False)

    encoder = None
    if approach == "a5":
        encoder = Encoder(embedding_dim=embedding_dim)
        ckpt = os.path.join("checkpoints", f"encoder_{embedding_dim}.pt")
        encoder.load_state_dict(torch.load(ckpt, map_location="cpu"))
        encoder.eval()

    n = min(num_samples, len(test_ds)) if num_samples else len(test_ds)

    # build sample list: (waveform, true_label)
    samples = []
    for i in range(n):
        waveform, _, true_label, *_ = test_ds[i]
        if waveform.shape[-1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[-1]))
        else:
            waveform = waveform[:, :16000]
        samples.append((waveform, true_label))

    if mixed:
        bg_samples = load_background_noise()
        samples += bg_samples
        print(f"Mixed mode: {n} keyword samples + {len(bg_samples)} background noise chunks")

    print(f"Running approach {approach} against {host}:{port} on {len(samples)} samples")

    records = []
    for i, (waveform, true_label) in enumerate(samples):
        if approach == "a1":
            payload = a1_raw.extract(waveform)
        elif approach == "a2":
            payload = a2_fixed_mfcc.extract(waveform)
        elif approach == "a3":
            payload = a3_event.extract(waveform, threshold=vad_threshold)
        elif approach == "a4":
            payload = a4_dynamic.extract(waveform, resolution=resolution)
        elif approach == "a5":
            payload = a5_embedding.extract(waveform, encoder)

        if payload is None:
            records.append({
                "true": true_label,
                "predicted": None,
                "bytes": 0,
                "rtt": None,
                "transmitted": False,
            })
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(samples)}")
            continue

        predicted_idx, rtt = send_payload(host, port, payload)

        records.append({
            "true": true_label,
            "predicted": LABELS[predicted_idx],
            "bytes": len(payload),
            "rtt": rtt,
            "transmitted": True,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(samples)}")

    tag = approach if approach != "a4" else f"a4_{resolution}"
    tag = tag if approach != "a5" else f"a5_dim{embedding_dim}"
    if mixed:
        tag += "_mixed"
    out_path = os.path.join(RESULTS_DIR, f"{tag}.json")

    with open(out_path, "w") as f:
        json.dump(records, f)

    print(f"Saved {len(records)} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach",       required=True, choices=["a1","a2","a3","a4","a5"])
    parser.add_argument("--host",           default="localhost")
    parser.add_argument("--port",           type=int, default=9999)
    parser.add_argument("--num-samples",    type=int, default=None, help="Limit samples for quick testing")
    parser.add_argument("--resolution",     default="high", choices=["high","medium","low"], help="A4 only")
    parser.add_argument("--embedding-dim",  type=int, default=64,  help="A5 only")
    parser.add_argument("--vad-threshold",  type=float, default=15.0, help="A3 only")
    parser.add_argument("--mixed",          action="store_true", help="Add background noise chunks labeled 'silence'")
    args = parser.parse_args()

    run(args.approach, args.host, args.port, args.num_samples,
        args.resolution, args.embedding_dim, args.vad_threshold, args.mixed)
