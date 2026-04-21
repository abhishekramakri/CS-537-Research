import socket
import struct
import argparse
import torch

from approaches import a1_raw, a2_fixed_mfcc, a3_event, a4_dynamic, a5_embedding
from models.cnn import KWSModel
from models.encoder import EmbeddingClassifier


def recv_exact(sock, n):
    """Read exactly n bytes from socket, blocking until all arrive."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client closed connection unexpectedly")
        buf += chunk
    return buf


def load_kws_model(device):
    model = KWSModel(n_mfcc=40, n_classes=35).to(device)
    model.load_state_dict(torch.load("checkpoints/kws_model.pt", map_location=device))
    model.eval()
    return model


def load_embedding_classifier(embedding_dim, device):
    classifier = EmbeddingClassifier(embedding_dim=embedding_dim, n_classes=35).to(device)
    classifier.load_state_dict(
        torch.load(f"checkpoints/embedding_classifier_{embedding_dim}.pt", map_location=device)
    )
    classifier.eval()
    return classifier


def handle_client(conn, approach, model, device):
    """Receives one payload, runs inference, sends back predicted class index."""
    length_bytes = recv_exact(conn, 4)
    payload_length = struct.unpack("!I", length_bytes)[0]
    payload = recv_exact(conn, payload_length)

    # deserialize payload into a tensor based on approach
    if approach == "a1":
        tensor = a1_raw.deserialize(payload)
    elif approach == "a2":
        tensor = a2_fixed_mfcc.deserialize(payload)
    elif approach == "a3":
        tensor = a3_event.deserialize(payload)
    elif approach == "a4":
        tensor = a4_dynamic.deserialize(payload)
    elif approach == "a5":
        tensor = a5_embedding.deserialize(payload, model.embedding_dim)

    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)

    predicted_idx = logits.argmax(dim=1).item()
    conn.sendall(struct.pack("!I", predicted_idx))


def run(approach, port, embedding_dim):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    if approach == "a5":
        model = load_embedding_classifier(embedding_dim, device)
        # attach embedding_dim so handle_client can pass it to deserialize
        model.embedding_dim = embedding_dim
    else:
        model = load_kws_model(device)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen()
        print(f"Server listening on port {port} — approach {approach}")

        while True:
            conn, addr = server.accept()
            with conn:
                try:
                    handle_client(conn, approach, model, device)
                except Exception as e:
                    print(f"Error handling client {addr}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach",      required=True, choices=["a1","a2","a3","a4","a5"])
    parser.add_argument("--port",          type=int, default=9999)
    parser.add_argument("--embedding-dim", type=int, default=64, help="A5 only")
    args = parser.parse_args()

    run(args.approach, args.port, args.embedding_dim)
