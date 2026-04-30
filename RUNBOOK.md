# Experiment Runbook

Server: `node0` — `10.10.1.1`  
Device: `node1`

Run each server command first, then the matching device command. Kill the server (Ctrl+C) between runs since all use port 9999.

---

## A1 — Raw Waveform

**node0:**
```bash
python3 server.py --approach a1
```
**node1:**
```bash
python3 device.py --approach a1 --host 10.10.1.1
```

---

## A2 — Fixed MFCC

**node0:**
```bash
python3 server.py --approach a2
```
**node1:**
```bash
python3 device.py --approach a2 --host 10.10.1.1
```

---

## A3 — Event-Triggered

**node0:**
```bash
python3 server.py --approach a3
```
**node1:**
```bash
python3 device.py --approach a3 --host 10.10.1.1
python3 device.py --approach a3 --host 10.10.1.1 --mixed
```

---

## A4 — Dynamic MFCC (3 resolutions)

**node0:**
```bash
python3 server.py --approach a4
```
**node1:**
```bash
python3 device.py --approach a4 --host 10.10.1.1 --resolution high
python3 device.py --approach a4 --host 10.10.1.1 --resolution medium
python3 device.py --approach a4 --host 10.10.1.1 --resolution low
```

---

## A5 — Learned Embedding (4 dims)

**node0 + node1 — repeat for each embedding dim:**

```bash
# dim 16
python3 server.py --approach a5 --embedding-dim 16    # node0
python3 device.py --approach a5 --host 10.10.1.1 --embedding-dim 16   # node1

# dim 32
python3 server.py --approach a5 --embedding-dim 32    # node0
python3 device.py --approach a5 --host 10.10.1.1 --embedding-dim 32   # node1

# dim 64
python3 server.py --approach a5 --embedding-dim 64    # node0
python3 device.py --approach a5 --host 10.10.1.1 --embedding-dim 64   # node1

# dim 128
python3 server.py --approach a5 --embedding-dim 128   # node0
python3 device.py --approach a5 --host 10.10.1.1 --embedding-dim 128  # node1
```

---

## Evaluate

After all runs, run evaluate on node1:

```bash
python3 evaluate.py
```

Plots saved to `plots/`.

---

## Copy Results to Local Machine

Run these **on your local machine**:

```bash
scp -r adhaw2@hp076.utah.cloudlab.us:~/CS-537-Research/results/ ./results
scp -r adhaw2@hp076.utah.cloudlab.us:~/CS-537-Research/plots/ ./plots
```
