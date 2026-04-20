import numpy as np
import torch
import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from person23_graph_encoder import UnifiedGraphEncoder
from person4_transformer import TransformerSSL
from Person5 import SeizureClassifier
from analysis import run_analysis, temporally_smooth_predictions, apply_refractory_period

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR  = "/content/drive/MyDrive/TUH_EEG_SEIZURE_v2.0/graph_output"
CKPT_PATH = "/content/drive/MyDrive/version.0/checkpoints/best_model.pth"

STEP_SECONDS    = 3.0       # each window = 3 seconds
FUTURE_WINDOW   = 100       # 5 minutes ahead (100 × 3s)
ICTAL_THRESHOLD = 0.45


# ── 1. Load data ────────────────────────────────────────────────
files = os.listdir(DATA_DIR)[10:20]
base  = sorted([f for f in files if f.endswith("_nodes.npy")])[0].replace("_nodes.npy", "")

nodes  = np.load(f"{DATA_DIR}/{base}_nodes.npy")
adj    = np.load(f"{DATA_DIR}/{base}_adj.npy")
labels = np.load(f"{DATA_DIR}/{base}_labels.npy")

try:
    ch_mask = np.load(f"{DATA_DIR}/{base}_mask.npy")
except FileNotFoundError:
    ch_mask = np.ones(16, dtype=np.float32)

nodes_t   = torch.tensor(nodes,   dtype=torch.float32).to(DEVICE)
adj_t     = torch.tensor(adj,     dtype=torch.float32).to(DEVICE)
ch_mask_t = torch.tensor(ch_mask, dtype=torch.float32).to(DEVICE)
labels_t  = torch.tensor(labels,  dtype=torch.long).to(DEVICE)


# ── 2. Initialize & load models ─────────────────────────────────
p23 = UnifiedGraphEncoder().to(DEVICE)
p4  = TransformerSSL().to(DEVICE)
p5  = SeizureClassifier().to(DEVICE)

ckpt = torch.load(CKPT_PATH,map_location=DEVICE, weights_only=False)
p23.load_state_dict(ckpt["p23"], strict=False)
p4.load_state_dict(ckpt["p4"],   strict=False)
ckpt_p5 = ckpt["p5"]

# ❌ Remove ALL temporal_head weights
filtered_state_dict = {
    k: v for k, v in ckpt_p5.items()
    if not k.startswith("temporal_head")
}

# ✅ Load remaining pretrained weights
missing, unexpected = p5.load_state_dict(filtered_state_dict, strict=False)

print("✅ Loaded pretrained weights (excluding temporal_head)")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
best_tau = ckpt.get("best_tau", 0.45)
print(f"✅ Loaded best_model.pth  |  best_tau={best_tau:.3f}")

p23.eval(); p4.eval(); p5.eval()


# ── 3. Forward pass ─────────────────────────────────────────────
with torch.no_grad():
    out_p23, spatial_nodes, band_weights = p23(nodes_t, adj_t, ch_mask_t)
    cls, tokens                          = p4(out_p23)
    logits, temporal_logits              = p5(cls, tokens)

    # Raw softmax probabilities
    probs    = torch.softmax(logits, dim=1).cpu().numpy()   # [N, 3]
   # 🚨 FIX NaNs
    if np.isnan(probs).any():
      print(f"⚠️ NaNs detected in {base}, fixing...")
      probs = np.nan_to_num(probs, nan=1e-6)
      probs = probs / probs.sum(axis=1, keepdims=True) 
    true_np  = labels_t.cpu().numpy()


# ── 4. FIX: Temporal smoothing + refractory period ──────────────
smoothed_preds = temporally_smooth_predictions(probs, best_tau,
                                               window=5, min_consecutive=3)
smoothed_preds = apply_refractory_period(smoothed_preds, refractory_windows=30)
preds_np = smoothed_preds

# ── 5. Window-level metrics ─────────────────────────────────────
print("\n── Window-Level Metrics ──")
print(classification_report(true_np, smoothed_preds,
      target_names=["interictal", "preictal", "ictal"], zero_division=0))

print(f"Preictal F1  : {f1_score(true_np, smoothed_preds, labels=[1], average='macro', zero_division=0):.4f}")
print(f"Pred counts  : {np.bincount(smoothed_preds, minlength=3)}")
print(f"True counts  : {np.bincount(true_np,        minlength=3)}")


# ── 6. FIX: False Alarm Rate (clinical definition) ──────────────
total_hours   = (len(true_np) * STEP_SECONDS) / 3600
alerts        = (smoothed_preds == 1) | (smoothed_preds == 2)

future_seizure = np.zeros(len(true_np), dtype=bool)
for i in range(len(true_np)):
    end_idx = min(i + FUTURE_WINDOW, len(true_np))
    future_seizure[i] = np.any(true_np[i:end_idx] == 2)
false_alarms  = (alerts & ~future_seizure).sum()
far_per_hour  = false_alarms / (total_hours + 1e-8)
print(f"\n── False Alarm Rate ──")
print(f"FAR per hour : {far_per_hour:.2f}")


# ── 7. Full clinical analysis (smoothed preds passed through) ───
print("\n── Clinical Analysis ──")
analysis_results = run_analysis(
    preds=smoothed_preds,
    p5=p5,
    cls=cls,
    spatial_nodes=spatial_nodes,
    band_weights=band_weights,
    labels=labels_t,
    best_tau=best_tau,
    window_size_sec=STEP_SECONDS
)

print("\n✅ Pipeline complete. Analysis ready for frontend.")
print(f"   Sensitivity  : {analysis_results['clinical_metrics']['sensitivity']}")
print(f"   FPR/hour     : {analysis_results['clinical_metrics']['fpr_per_hour']:.2f}")
