# =========================
# ALL IMPORTS (COMPLETE)
# =========================
import os
import time
import warnings
import numpy as np
import multiprocessing

import mne
import antropy as ant

from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from scipy.signal import welch, hilbert, butter, filtfilt

# Optional (for future extensions / type hints)
from typing import List, Tuple, Optional

# Suppress warnings/log spam
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")


# @title
# ============================================================
# 🧩 DATA PIPELINE (PREPROCESSING & GRAPH BUILDING) - FINAL
# Eliminates Label Leakage, Time Jumps, and Restores True Frequency Graphs
# ============================================================
import time
import os
import numpy as np
import mne
import antropy as ant
from scipy.signal import welch, hilbert, butter, filtfilt
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

mne.set_log_level("WARNING")



from pathlib import Path

# 🔑 UNIQUE ID (FIXES COLLISIONS)
def make_unique_id(edf_path):
    return Path(edf_path).stem


# 📦 SCAN OUTPUT DIR FOR COMPLETED FILES
def get_completed_files(output_dir):
    files = list(Path(output_dir).glob("*_nodes.npy"))
    completed = set()

    for f in files:
        base = f.name.replace("_nodes.npy", "")

        adj = f.with_name(base + "_adj.npy")
        lbl = f.with_name(base + "_labels.npy")
        msk = f.with_name(base + "_mask.npy")

        if adj.exists() and lbl.exists() and msk.exists():
            completed.add(base)

    return completed
# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "data_root":      "/content/drive/MyDrive/v2.0.5/edf/train",
    "output_dir":     "/content/drive/MyDrive/TUH_EEG_SEIZURE_v2.0/graph_output",

    "sfreq":          256,
    "epoch_duration": 4.0,
    "epoch_overlap":  0.25,

    "preictal_sec":   300,
    "min_gap_sec":    60,

    "freq_bands": {
        "delta": (0.5,  4),
        "theta": (4,    8),
        "alpha": (8,   13),
        "beta":  (13,  30),
        "gamma": (30,  40),
    },

    "seq_len": 10,
    "n_jobs":  min(8, os.cpu_count() or 1)
}

# Standard TUH TCP Montage (16 channels)
BIPOLAR_PAIRS = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2')
]

# =========================================================
# 1. REFERENCING & ALIGNMENT
# =========================================================
def align_and_bipolar(raw):
    """Creates bipolar channels safely without average reference."""
    data   = raw.get_data()
    ch_map = {ch: data[i] for i, ch in enumerate(raw.ch_names)}

    bipolar_data, mask = [], []
    for ch1, ch2 in BIPOLAR_PAIRS:
        if ch1 in ch_map and ch2 in ch_map:
            bipolar_data.append(ch_map[ch1] - ch_map[ch2])
            mask.append(1) # Channel present
        else:
            bipolar_data.append(np.zeros(data.shape[1]))
            mask.append(0) # Channel missing

    bipolar_data = np.array(bipolar_data)
    info = mne.create_info(
        ch_names=[f"{a}-{b}" for a, b in BIPOLAR_PAIRS],
        sfreq=raw.info["sfreq"],
        ch_types="eeg"
    )
    new_raw = mne.io.RawArray(bipolar_data, info, verbose=False)

    return new_raw, np.array(mask, dtype=np.float32)

# =========================================================
# 2. LABELS & EPOCHING
# =========================================================
def get_label(t, seizures):
    for sz_start, sz_end in seizures:
        if sz_start <= t <= sz_end:
            return 2  # Ictal
        if sz_start - CONFIG["preictal_sec"] <= t < sz_start:
            return 1  # Preictal
        if sz_end < t < sz_end + CONFIG["min_gap_sec"]:
            return None  # Postictal gap
    return 0  # Interictal

def create_epochs(raw, seizures):
    data  = raw.get_data()
    times = raw.times
    sfreq = raw.info["sfreq"]

    win  = int(CONFIG["epoch_duration"] * sfreq)
    step = int(win * (1 - CONFIG["epoch_overlap"]))

    epochs, labels = [], []

    for i in range(0, data.shape[1] - win + 1, step):
        seg   = data[:, i:i+win]
        t     = times[i + win // 2]
        lbl   = get_label(t, seizures)

        # Keep None labels so we can track gaps later
        epochs.append(seg)
        labels.append(lbl)

    return epochs, labels

# =========================================================
# 3. FEATURE EXTRACTION & TRUE FREQUENCY ADJACENCY
# =========================================================
def compute_node_features(epoch, identity_matrix):
    feats = []
    sfreq = CONFIG["sfreq"]

    # 5 Frequency Bands (PSD)
    freqs, psd = welch(epoch, sfreq, nperseg=sfreq*2)
    for band, (low, high) in CONFIG["freq_bands"].items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = psd[:, idx].mean(axis=1)
        feats.append(band_power)

    # Non-linear: Sample Entropy
    entropy = np.array([ant.sample_entropy(ch) for ch in epoch])
    feats.append(entropy)

    # Hjorth Parameters
    var_0 = np.var(epoch, axis=1)
    diff_1 = np.diff(epoch, axis=1)
    var_1 = np.var(diff_1, axis=1)
    diff_2 = np.diff(diff_1, axis=1)
    var_2 = np.var(diff_2, axis=1)

    activity = var_0
    mobility = np.sqrt(var_1 / (var_0 + 1e-6))
    complexity = np.sqrt(var_2 / (var_1 + 1e-6)) / (mobility + 1e-6)

    feats.extend([activity, mobility, complexity])

    feats = np.array(feats).T
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
    feats = np.concatenate([feats, identity_matrix], axis=1)
    return feats.astype(np.float32)

def compute_adjacency(epoch):
    """✅ FIXED: True band-specific filtering before Hilbert PLV"""
    sfreq = CONFIG["sfreq"]
    nyq = 0.5 * sfreq
    adj_bands = []

    for band_name, (low, high) in CONFIG["freq_bands"].items():
        # 1. Bandpass filter for the specific frequency
        b, a = butter(4, [low / nyq, high / nyq], btype='band')
        filtered_epoch = filtfilt(b, a, epoch, axis=1)

        # 2. Extract Phase via Hilbert
        analytic = hilbert(filtered_epoch, axis=1)
        phase = np.unwrap(np.angle(analytic), axis=1)

        # 3. Calculate PLV
        plv = np.zeros((16, 16), dtype=np.float32)
        for i in range(16):
            for j in range(i, 16):
                phase_diff = phase[i] - phase[j]
                val = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv[i, j] = plv[j, i] = val

        np.fill_diagonal(plv, 1.0)
        adj_bands.append(plv)

    return np.array(adj_bands, dtype=np.float32) # [5, 16, 16]

def _compute_epoch_pair(epoch, identity):
    nodes = compute_node_features(epoch, identity)
    adj   = compute_adjacency(epoch)
    return nodes, adj

# =========================================================
# 4. SEQUENCE BUILDING
# =========================================================
def build_sequences(epochs, labels):
    seq_len = CONFIG["seq_len"]
    if len(epochs) < seq_len:
        return None, "too_few_epochs"

    identity = np.eye(16, dtype=np.float32)

    valid_epochs = [(i, ep, lbl) for i, (ep, lbl) in enumerate(zip(epochs, labels)) if lbl is not None]
    if not valid_epochs:
        return None, "all_epochs_invalid"

    results = Parallel(n_jobs=CONFIG["n_jobs"], prefer="threads")(
        delayed(_compute_epoch_pair)(ep, identity) for _, ep, _ in valid_epochs
    )

    node_map, adj_map = {}, {}
    for (orig_idx, _, _), res in zip(valid_epochs, results):
        node_map[orig_idx] = res[0]
        adj_map[orig_idx]  = res[1]

    nodes_all, adj_all, seq_labels = [], [], []

    for i in range(len(epochs) - seq_len + 1):
        seq_lbls = labels[i:i + seq_len]

        if any(l is None for l in seq_lbls):
            continue

        nodes_all.append([node_map[i + j] for j in range(seq_len)])
        adj_all.append([adj_map[i + j]    for j in range(seq_len)])

        # Center Labeling
        center_idx = (seq_len - 1) // 2
        seq_labels.append(seq_lbls[center_idx])

    if len(nodes_all) == 0:
         return None, "all_sequences_bridged_gaps"

    return (
        np.array(nodes_all,  dtype=np.float32),
        np.array(adj_all,    dtype=np.float32),
        np.array(seq_labels, dtype=np.int64),
    ), None

# =========================================================
# 5. ORCHESTRATION & SAVING
# =========================================================
def process_one_file(edf_path, seizures):
    start = time.time()
    uid = make_unique_id(edf_path)
    out_prefix = os.path.join(CONFIG["output_dir"], uid)


    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

# 🔥 FAST CHECK BEFORE HEAVY OPS
        def normalize(ch):
          return ch.upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "").strip()

        available = {normalize(ch): ch for ch in raw.ch_names}
        valid_count = 0
        for ch1, ch2 in BIPOLAR_PAIRS:
          if normalize(ch1) in available and normalize(ch2) in available:
            valid_count += 1


        if valid_count < 8:
          return "error_too_many_dead_channels"

        # ONLY NOW load + filter
        raw.load_data()
        raw.filter(0.5, 40.0, fir_design='firwin', verbose=False)
        raw.notch_filter(60.0, verbose=False)
        new_raw, ch_mask = align_and_bipolar(raw)

        epochs, labels = create_epochs(new_raw, seizures)

        if len(epochs) == 0:
            return "empty"

        seq_data, err = build_sequences(epochs, labels)
        if seq_data is None:
            return err

        nodes, adj, seq_labels = seq_data
        print(f"💾 Saving: {out_prefix}")
        np.save(f"{out_prefix}_nodes.npy", nodes)
        np.save(f"{out_prefix}_adj.npy", adj)
        np.save(f"{out_prefix}_labels.npy", seq_labels)
        np.save(f"{out_prefix}_mask.npy", ch_mask)

        return "done", time.time() - start

    except Exception as e:
        print(f"❌ {uid}: {e}")
        return "error", time.time() - start

def load_seizures_from_csv(csv_path):
    seizures = []
    try:
        with open(csv_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                # Skip header
                try:
                    start = float(parts[1])
                    end   = float(parts[2])
                    label = parts[3].strip().lower()
                    if label not in ('bckg', 'label', ''):
                        seizures.append((start, end))
                except:
                    continue
    except Exception as e:
        print(f"⚠️ Error reading {csv_path}: {e}")
    return seizures
# =========================================================
# 6. DATASET DRIVER (AUTO SCAN + SAFE REPROCESS)
# =========================================================
def run():
    total_start = time.time()
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    edfs = list(Path(CONFIG["data_root"]).rglob("*.edf"))

    print(f"🎯 Total target EDFs: {len(edfs)}\n")

    skip_done = 0
    skip_empty = 0
    error = 0
    processed = 0

    out = Path(CONFIG["output_dir"])
    completed = get_completed_files(CONFIG["output_dir"])
    print(f"✅ Already completed: {len(completed)} files\n")
    for i, edf in enumerate(edfs, 1):
        uid = make_unique_id(str(edf))
        print(f"[{i}/{len(edfs)}] {uid}")
        out_prefix = os.path.join(CONFIG["output_dir"], uid)
        if os.path.exists(f"{out_prefix}_nodes.npy") and \
          os.path.exists(f"{out_prefix}_adj.npy") and \
          os.path.exists(f"{out_prefix}_labels.npy") and \
          os.path.exists(f"{out_prefix}_mask.npy"):
          print("⏭️ Skipping (already processed)")
          skip_done += 1
          continue

        ann = edf.with_suffix(".csv_bi")
        if ann.exists():
          ann_path = str(ann)
        else:
          print("⚠️ No annotation file → skipping")
          skip_empty += 1
          continue
        seizures = load_seizures_from_csv(ann_path)
        if len(seizures) == 0:
          print("🚫 No seizures → skipping")
          skip_empty += 1
          continue


        start = time.time()
        result = process_one_file(str(edf), seizures)
        t = time.time() - start
        print(f"⏱ {t:.2f}s | result: {result}")



        if result == "done":
            processed += 1
        elif result == "empty":
            skip_empty += 1
        else:
            error += 1

    # ---------------- SUMMARY ----------------
    print("\n================ FINAL SUMMARY ================")
    print(f"✅ Processed: {processed}")
    print(f"⏭️ Skipped (already done): {skip_done}")
    print(f"🚫 Skipped (empty): {skip_empty}")
    print(f"❌ Errors: {error}")
    print(f"⏱ TOTAL TIME: {time.time() - total_start:.2f}s")
    print("=============================================")


if __name__ == "__main__":
    run()
