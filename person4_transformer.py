# ============================================================
# 🧠 TRANSFORMER — MULTI-OBJECTIVE SSL (PERSON 4 FIXED)
# ============================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from person23_graph_encoder import UnifiedGraphEncoder

# ── DIRS ──
GRAPH_DIR  = "/content/drive/MyDrive/TUH_EEG_SEZIURE_v2.0/graph_output"
OUTPUT_DIR = "/content/drive/MyDrive/version.0/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CONSTANTS ──
D_MODEL     = 512
NUM_WINDOWS = 10
SEQ_LEN     = 11
NUM_HEADS   = 8
NUM_LAYERS  = 6
SSL_EPOCHS  = 15
BATCH_SIZE  = 32
GRAD_ACCUM  = 4
DROPOUT     = 0.1
FFN_DIM     = 4 * D_MODEL
SSL_LR      = 1e-4
NUM_WORKERS = 2
MAX_PAIRS = 1000


class JointSSLEEGDataset(Dataset):
    def __init__(self, graph_dir: str, max_pairs: int = None):
        self.samples = []
        self.by_patient = {}
        files = sorted(glob(os.path.join(graph_dir, "*_nodes.npy")))

        for f in files:
            stem = os.path.basename(f).replace("_nodes.npy", "")
            parts = stem.split("_")
            pid = parts[0]

            arr = np.load(f, mmap_mode="r")
            if arr.ndim == 3:
                entry = (f, -1, pid)
                self.samples.append(entry)
                self.by_patient.setdefault(pid, []).append(entry)
            elif arr.ndim == 4:
                for row in range(arr.shape[0]):
                    entry = (f, row, pid)
                    self.samples.append(entry)
                    self.by_patient.setdefault(pid, []).append(entry)

        if max_pairs and len(self.samples) > max_pairs:
            self.samples = self.samples[:max_pairs]

    def _load_data(self, fpath, row):
        raw = np.load(fpath)
        nodes_raw = torch.tensor(raw[row] if row != -1 else raw, dtype=torch.float32)

        nodes = torch.zeros((NUM_WINDOWS, 16, 25))
        actual_windows = min(nodes_raw.shape[0], NUM_WINDOWS)
        actual_feats   = min(nodes_raw.shape[2], 25)
        nodes[:actual_windows, :, :actual_feats] = nodes_raw[:actual_windows, :, :actual_feats]
        nodes[:, :, 9:25] = torch.eye(16).unsqueeze(0).expand(NUM_WINDOWS, -1, -1)

        adj_path = fpath.replace("_nodes.npy", "_adj.npy")
        mask_path = fpath.replace("_nodes.npy", "_mask.npy")

        if os.path.exists(adj_path):
            adj_raw = np.load(adj_path)
            adj = torch.tensor(adj_raw[row] if row != -1 else adj_raw, dtype=torch.float32)
        else:
            adj = torch.eye(16).unsqueeze(0).unsqueeze(0).expand(NUM_WINDOWS, 5, -1, -1).clone()

        if os.path.exists(mask_path):
            mask = torch.tensor(np.load(mask_path), dtype=torch.float32)
        else:
            mask = torch.ones(16)

        return nodes, adj, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f, r, pid = self.samples[idx]
        n, a, m = self._load_data(f, r)

        shuf_idx = torch.randperm(NUM_WINDOWS)
        n_s, a_s = n[shuf_idx], a[shuf_idx]

        # Hierarchical negative: prefer same patient, different row
        patient_entries = self.by_patient.get(pid, [])
        neg_entry = None
        for _ in range(10):
            cand = random.choice(patient_entries)
            if cand[1] != r or cand[0] != f:
                neg_entry = cand
                break
        if neg_entry is None:
            # fallback: different patient
            other_pid = random.choice([p for p in self.by_patient if p != pid] or [pid])
            neg_entry = random.choice(self.by_patient[other_pid])

        n_n, a_n, m_n = self._load_data(neg_entry[0], neg_entry[1])

        return (n, a, m), (n_s, a_s, m), (n_n, a_n, m_n)


# ============================================================
# MODEL
# ============================================================
class TransformerSSL(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NUM_HEADS,
            dim_feedforward=FFN_DIM, dropout=DROPOUT,
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)

        self.order_head   = nn.Sequential(nn.Linear(D_MODEL, 128), nn.GELU(), nn.Linear(128, 2))
        self.pred_head_fwd = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.proj_head    = nn.Sequential(nn.Linear(D_MODEL, 256), nn.GELU(), nn.Linear(256, 128))

        self.log_sigma_order    = nn.Parameter(torch.zeros(1))
        self.log_sigma_pred     = nn.Parameter(torch.zeros(1))
        self.log_sigma_contrast = nn.Parameter(torch.zeros(1))

    def encode(self, x):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        enc = self.transformer(x)
        return self.norm(enc[:, 0, :]), enc[:, 1:, :]

    def forward(self, x):
        return self.encode(x)

    def ssl_loss(self, seq, shuffled, neg_seq, tau=0.07):
        B = seq.shape[0]
        device = seq.device

        cls_orig,  tok_orig = self.encode(seq)
        cls_shuffle, _      = self.encode(shuffled)
        cls_neg, _          = self.encode(neg_seq)

        # 1. Order prediction
        order_input  = torch.cat([cls_orig, cls_shuffle], dim=0)
        order_labels = torch.cat([
            torch.ones(B,  device=device, dtype=torch.long),
            torch.zeros(B, device=device, dtype=torch.long)
        ], dim=0)
        L_order = F.cross_entropy(self.order_head(order_input), order_labels)

        # 2. Future prediction
        L_pred = F.mse_loss(
            self.pred_head_fwd(tok_orig[:, :-1, :]),
            tok_orig[:, 1:, :].detach()
        )

        # 3. Contrastive (NT-Xent)
        z_orig  = F.normalize(self.proj_head(cls_orig),  dim=-1)
        z_neg   = F.normalize(self.proj_head(cls_neg),   dim=-1)
        cls_orig2, _ = self.encode(seq)
        z_orig2 = F.normalize(self.proj_head(cls_orig2), dim=-1)

        pos_sim = (z_orig * z_orig2).sum(dim=-1, keepdim=True) / tau
        neg_sim = (z_orig @ z_neg.T) / tau
        logits  = torch.cat([pos_sim, neg_sim], dim=-1)
        L_contrast = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=device))

        # 4. Uncertainty weighting
        s1 = torch.exp(-self.log_sigma_order)
        s2 = torch.exp(-self.log_sigma_pred)
        s3 = torch.exp(-self.log_sigma_contrast)
        L_total = (
            s1 * L_order    + self.log_sigma_order    +
            s2 * L_pred     + self.log_sigma_pred     +
            s3 * L_contrast + self.log_sigma_contrast
        )
        return L_total, {
            "order":    L_order.item(),
            "pred":     L_pred.item(),
            "contrast": L_contrast.item()
        }


# ============================================================
# PRETRAINING LOOP
# ============================================================
def ssl_pretrain():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    dataset = JointSSLEEGDataset(GRAPH_DIR, max_pairs=MAX_PAIRS)
    print(f"Dataset size: {len(dataset)} samples")

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=use_amp,
        persistent_workers=(NUM_WORKERS > 0), drop_last=True
    )

    p23 = UnifiedGraphEncoder().to(device)
    p4  = TransformerSSL().to(device)

    optimizer = torch.optim.AdamW(
        list(p23.parameters()) + list(p4.parameters()),
        lr=SSL_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SSL_EPOCHS, eta_min=1e-5
    )

    best_loss = float("inf")

    for ep in range(SSL_EPOCHS):
        p23.train(); p4.train()
        total_loss = 0.0
        n_steps    = 0
        current_tau = max(0.07, 0.5 - (0.43 * (ep / SSL_EPOCHS)))
        optimizer.zero_grad()

        for step, (orig, shuf, neg) in enumerate(loader):
            n_o, a_o, m_o = [x.to(device, non_blocking=True) for x in orig]
            n_s, a_s, m_s = [x.to(device, non_blocking=True) for x in shuf]
            n_n, a_n, m_n = [x.to(device, non_blocking=True) for x in neg]

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                seq_emb,  _, _ = p23(n_o, a_o, m_o)
                shuf_emb, _, _ = p23(n_s, a_s, m_s)
                neg_emb,  _, _ = p23(n_n, a_n, m_n)

                seq_emb  = torch.nan_to_num(seq_emb,  nan=0.0)
                shuf_emb = torch.nan_to_num(shuf_emb, nan=0.0)
                neg_emb  = torch.nan_to_num(neg_emb,  nan=0.0)

                loss, info = p4.ssl_loss(seq_emb, shuf_emb, neg_emb, tau=current_tau)

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            loss_scaled = loss / GRAD_ACCUM
            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(p23.parameters()) + list(p4.parameters()), 1.0
                )
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            n_steps    += 1

        scheduler.step()
        avg = total_loss / max(n_steps, 1)
        print(f"Epoch {ep+1:2d} | loss={avg:.4f} | tau={current_tau:.3f} | "
              f"order={info['order']:.4f} pred={info['pred']:.4f} contrast={info['contrast']:.4f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(p4.state_dict(),  os.path.join(OUTPUT_DIR, "transformer_ssl_pretrained.pth"))
            torch.save(p23.state_dict(), os.path.join(OUTPUT_DIR, "p23_ssl_pretrained.pth"))
            print(f"  ★ best saved")


if __name__ == "__main__":
    ssl_pretrain()