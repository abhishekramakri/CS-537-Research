import socket
import struct
import time
import json
import os
import argparse
import torch
import torchaudio

from approaches import a1_raw, a2_fixed_mfcc, a3_event, a4_dynamic, a5_embedding
from models.encoder import Encoder
from train import SpeechCommandsDataset, get_mfcc_transform, LABELS

RESULTS_DIR = "results"


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


def run(approach, host, port, num_samples, resolution, embedding_dim, vad_threshold):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    test_ds = torchaudio.datasets.SPEECHCOMMANDS("data", subset="testing", download=False)

    encoder = None
    if approach == "a5":
        encoder = Encoder(embedding_dim=embedding_dim)
        ckpt = os.path.join("checkpoints", f"encoder_{embedding_dim}.pt")
        encoder.load_state_dict(torch.load(ckpt, map_location="cpu"))
        encoder.eval()

    records = []
    n = min(num_samples, len(test_ds)) if num_samples else len(test_ds)

    print(f"Running approach {approach} against {host}:{port} on {n} samples")

    for i in range(n):
        waveform, _, true_label, *_ = test_ds[i]

        # pad/trim to 1 second
        if waveform.shape[-1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[-1]))
        else:
            waveform = waveform[:, :16000]

        # --- extract payload based on approach ---
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

        # A3 returns None when VAD doesn't trigger — skip transmission
        if payload is None:
            records.append({
                "true": true_label,
                "predicted": None,
                "bytes": 0,
                "rtt": None,
                "transmitted": False,
            })
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
            print(f"  {i + 1}/{n}")

    tag = approach if approach != "a4" else f"a4_{resolution}"
    tag = tag if approach != "a5" else f"a5_dim{embedding_dim}"
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
    parser.add_argument("--vad-threshold",  type=float, default=0.02, help="A3 only")
    args = parser.parse_args()

    run(args.approach, args.host, args.port, args.num_samples,
        args.resolution, args.embedding_dim, args.vad_threshold)
