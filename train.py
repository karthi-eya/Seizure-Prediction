import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from person23_graph_encoder import UnifiedGraphEncoder
from person4_transformer import TransformerSSL
from Person5 import SeizureClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR            = "/content/drive/MyDrive/TUH_EEG_SEIZURE_v2.0/graph_output"
SAVE_DIR            = "/content/drive/MyDrive/version.0/checkpoints"
BATCH_SIZE          = 128
GRAD_ACCUM_STEPS    = 1       # effective batch=128, more stable gradients
EPOCHS              = 80
WARMUP_EPOCHS       = 5      # longer warmup
TRANSITION_EPOCHS   = 5       # smooth curriculum blend epochs
FOCAL_GAMMA         = 2.0
AUX_F1_WEIGHT       = 0.08
TEMPORAL_WEIGHT     = 0.15
LOGIT_ADJ_TAU       = -0.05  # mild prior correction
LABEL_SMOOTHING     = 0.05
ICTAL_THRESHOLD     = 0.52

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Batch sanitizer (unchanged, correct) ────────────────────────
def sanitize_batch(nodes, adj):
    nodes = torch.nan_to_num(nodes, nan=0.0, posinf=0.0, neginf=0.0)
    adj   = torch.nan_to_num(adj,   nan=0.0, posinf=0.0, neginf=0.0)
    B, T, num_bands, N, _ = adj.shape
    eye = torch.eye(N, device=adj.device, dtype=adj.dtype)
    eye = eye.view(1,1,1,N,N).expand(B,T,num_bands,N,N)
    adj = adj + 1e-6 * eye
    return nodes, adj


# ── FIX 1: minority_boost must be >> 1.0 ────────────────────────
# Original had minority_boost=0.7 which PENALISED preictal,
# the exact opposite of the intended behaviour.
class CalibratedFocalLoss(nn.Module):
    def __init__(self, class_counts, gamma=2.0,
                 minority_boost=8.0, min_alpha=0.05):
        super().__init__()
        self.gamma = gamma
        counts = torch.tensor(class_counts, dtype=torch.float32)
        raw = 1.0 / torch.sqrt(counts + 1e-6)
        raw[1] *= minority_boost
        raw = raw / raw.sum()
        raw = torch.clamp(raw, min=min_alpha)

        raw = raw / raw.sum()
        self.register_buffer('alpha', raw)
        total = counts.sum()
        log_priors = torch.log(counts / total + 1e-9).clamp(min=-8.0, max=0.0)
        self.register_buffer('log_priors', log_priors)

    def forward(self, logits, targets,
                apply_logit_adj=True, tau=LOGIT_ADJ_TAU):
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        if apply_logit_adj and tau > 0:
            logits = logits + tau * self.log_priors.to(logits.device)
        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs).clamp(min=1e-6, max=1.0)
        pt        = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt    = torch.log(pt).clamp(min=-100.0)
        focal     = (1.0 - pt) ** self.gamma
        alpha_t   = self.alpha[targets]
        return (alpha_t * focal * (-log_pt)).mean()


# ── FIX 2: confidence-weighted soft-F1 ─────────────────────────
# Reduces gradient conflict with focal loss on ambiguous samples.
class SoftPreictalF1Loss(nn.Module):
    def forward(self, logits, targets, eps=1e-4):
        probs = F.softmax(logits, dim=1)
        p1 = probs[:, 1]
        t1 = (targets == 1).float()
        tp = (p1 * t1).sum()
        fp = (p1 * (1 - t1)).sum()
        fn = ((1 - p1) * t1).sum()
        precision = (tp + eps) / (tp + fp + eps)
        recall    = (tp + eps) / (tp + fn + eps)
        soft_f1   = (2 * precision * recall) / (precision + recall + eps)
        return 1.0 - soft_f1


# ── FIX 3+4: Curriculum loss + correct temporal loss ───────────
# Old code hard-switched CE→Focal at epoch 6, causing probability
# distribution collapse. New: linear blend over TRANSITION_EPOCHS.
# Old temporal loss used binary BCE on a multiclass label (wrong).
class CurriculumLoss(nn.Module):
    def __init__(self, focal_crit, f1_crit, warmup_crit,
                 warmup_epochs=8, transition_epochs=5):
        super().__init__()
        self.margin = IctalPreictalMarginLoss()
        self.focal    = focal_crit
        self.f1       = f1_crit
        self.warmup   = warmup_crit
        self.n_warmup = warmup_epochs
        self.n_trans  = transition_epochs

    def forward(self, logits, temporal_logits, targets, epoch):
        f1_term       = torch.tensor(0.0, device=logits.device)
        temporal_loss = torch.tensor(0.0, device=logits.device)

        if epoch < self.n_warmup:
            loss = self.warmup(logits, targets)

        elif epoch < self.n_warmup + self.n_trans:
            alpha      = (epoch - self.n_warmup) / self.n_trans
            ce_loss    = self.warmup(logits, targets)
            focal_loss = self.focal(logits, targets, apply_logit_adj=True)
            f1_loss    = self.f1(logits, targets)
            f1_term    = f1_loss.detach()
            focal_full = ((1.0 - AUX_F1_WEIGHT) * focal_loss
                          + AUX_F1_WEIGHT * f1_loss)
            loss = (1.0 - alpha) * ce_loss + alpha * focal_full

        else:
            focal_loss = self.focal(logits, targets, apply_logit_adj=True)
            f1_loss    = self.f1(logits, targets)
            f1_term    = f1_loss.detach()
            pre_mask   = (targets == 1).float()
            pre_boost  = 1.0 + 2.0 * pre_mask
            loss = ((1.0 - AUX_F1_WEIGHT) * focal_loss + AUX_F1_WEIGHT * f1_loss)
            loss = (loss * pre_boost).mean()
        # FIX 4: Temporal loss — proper multiclass cross-entropy
        if temporal_logits is not None and TEMPORAL_WEIGHT > 0:
            tl = torch.clamp(temporal_logits, -20, 20)
            if tl.dim() == 3 and tl.shape[-1] == 3:
                # Shape [B, T, 3] → multiclass CE per time step
                B, T, C = tl.shape
                temporal_loss = F.cross_entropy(
                    tl.reshape(B * T, C),
                    targets.unsqueeze(1).expand(B, T).reshape(B * T)
                )
            elif tl.dim() == 2:
                # Shape [B, T] → binary BCE (is-preictal per frame)
                tgt = (targets == 1).float()
                temporal_loss = F.binary_cross_entropy_with_logits(
                    tl, tgt.unsqueeze(1).expand_as(tl)
                )
            loss = loss + TEMPORAL_WEIGHT * temporal_loss

        return loss, f1_term, temporal_loss


# ── Dataset (unchanged) ─────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, items):
        self.items = items
        self.cache = {}
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
      nf, af, lf, row, label = self.items[idx]
      nodes = np.load(nf)[row]
      adj   = np.load(af)[row]
      return (
        torch.from_numpy(np.array(nodes)).float(),
        torch.from_numpy(np.array(adj)).float(),
        torch.ones(16),
        torch.tensor(label, dtype=torch.long)
      )


# ── Data build (unchanged) ──────────────────────────────────────
def build_split():
    np.random.seed(42)
    print("🚀 Scanning directory with strict file-triplet checking...", flush=True)
    files      = os.listdir(DATA_DIR)[:200]
    node_files = [f for f in files if f.endswith("_nodes.npy")]
    manifest   = []
    skipped    = 0
    print(f"📦 Validating {len(node_files)} file triplets...", flush=True)
    for f in tqdm(node_files):
        base = f.replace("_nodes.npy", "")
        nf   = os.path.join(DATA_DIR, f)
        af   = os.path.join(DATA_DIR, base + "_adj.npy")
        lf   = os.path.join(DATA_DIR, base + "_labels.npy")
        if not (os.path.exists(af) and os.path.exists(lf)):
            skipped += 1; continue
        try:
            labels = np.load(lf)
            for i in range(len(labels)):
                manifest.append((nf, af, lf, i, int(labels[i])))
        except Exception:
            skipped += 1; continue
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} incomplete triplets.")
    pids = list(set([os.path.basename(m[0]).split('_')[0] for m in manifest]))
    np.random.shuffle(pids)
    split_idx  = int(len(pids) * 0.8)
    train_pids = set(pids[:split_idx])
    train_items = [m for m in manifest
                   if os.path.basename(m[0]).split('_')[0] in train_pids]
    val_items   = [m for m in manifest
                   if os.path.basename(m[0]).split('_')[0] not in train_pids]
    print(f"✅ Final Manifest: {len(train_items)} train windows, "
          f"{len(val_items)} val windows.")
    return train_items, val_items


# ── FIX 5: Sampler — extra preictal emphasis + larger epoch ─────
def create_sampler(items, num_samples=None):
    labels = np.array([m[4] for m in items])
    class_counts = np.bincount(labels, minlength=3)
    weights = 1.0 / (class_counts.astype(float) ** 0.6)
    samples_weight = torch.tensor([weights[l] for l in labels], dtype=torch.float)
    if num_samples is None:
        num_samples = len(items)  # no artificial cap, full epoch
    return (WeightedRandomSampler(samples_weight, num_samples=num_samples,
                                  replacement=True), class_counts)


@torch.no_grad()
def evaluate(p23, p4, p5, loader, curriculum, epoch=0):
    p23.eval(); p4.eval(); p5.eval()
    all_logits, all_labels, total_loss = [], [], 0.0

    for n, a, m, l in loader:
        n, a, m, l = n.to(DEVICE), a.to(DEVICE), m.to(DEVICE), l.to(DEVICE)
        n, a = sanitize_batch(n, a)
        feat = p23(n, a, m)
        feat = feat[0] if isinstance(feat, tuple) else feat
        feat = torch.nan_to_num(feat, nan=0.0)
        cls, tokens = p4(feat)
        cls    = torch.nan_to_num(cls,    nan=0.0)
        tokens = torch.nan_to_num(tokens, nan=0.0)
        logits, temporal_logits = p5(cls, tokens)
        logits = torch.nan_to_num(logits, nan=0.0)
        loss, _, _ = curriculum(logits, temporal_logits, l, epoch)
        if torch.isfinite(loss):
            total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(l.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs  = F.softmax(all_logits, dim=1).numpy()

    # ── HONEST DIAGNOSTICS ──────────────────────────────────────
    print(f"  Prob means (inter/pre/ictal): {all_probs.mean(axis=0).round(4)}")
    print(f"  Prob means on TRUE preictal windows:")
    pre_mask = all_labels == 1
    if pre_mask.sum() > 0:
        print(f"    {all_probs[pre_mask].mean(axis=0).round(4)}")
        print(f"    Argmax on true preictal: "
              f"{np.bincount(np.argmax(all_probs[pre_mask], axis=1), minlength=3)}")

    # ── PURE ARGMAX PREDICTIONS — NO HEURISTICS ─────────────────
    preds = np.argmax(all_probs, axis=1)

    print(f"\n  Val pred counts (argmax): {np.bincount(preds, minlength=3)}")
    print(f"  True counts:              {np.bincount(all_labels, minlength=3)}")

    avg_loss = total_loss / max(len(loader), 1)
    print(classification_report(all_labels, preds,
          target_names=['interictal','preictal','ictal'], zero_division=0))

    cm = confusion_matrix(all_labels, preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':15s}  interictal  preictal    ictal")
    for i, name in enumerate(['interictal','preictal','ictal']):
        print(f"  {name:15s}  {cm[i,0]:10d}  {cm[i,1]:8d}  {cm[i,2]:8d}")

    f1 = f1_score(all_labels, preds, labels=[1],
                  average='macro', zero_division=0)
    return f1, 0.5, avg_loss


# ── Weight init ─────────────────────────────────────────────────
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
#------------------------------------------------------------------
class IctalPreictalMarginLoss(nn.Module):
    def forward(self, logits, targets, margin=0.2):
        probs = F.softmax(logits, dim=1)
        # On true ictal: push ictal prob > preictal prob by margin
        ictal_mask = (targets == 2)
        if ictal_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        diff1 = probs[ictal_mask, 1] - probs[ictal_mask, 2] + margin
        loss1 = F.relu(diff1)
        # NEW: penalize false ictal dominance globally
        over_ictal = probs[:, 2] - 0.6
        loss2 = F.relu(over_ictal).mean()
        return loss1.mean() + 0.5 * loss2
# ── Train ────────────────────────────────────────────────────────
def train():
    train_list, val_list = build_split()
    train_labels = [m[4] for m in train_list]
    counts       = np.bincount(train_labels, minlength=3)
    print(f"✅ Class Counts: {counts}")
    print(f"   Preictal ratio: {counts[1]/counts.sum()*100:.2f}%")
    if counts[1] == 0:
        print("❌ ERROR: No preictal samples in training set."); return

    sampler, _ = create_sampler(train_list,100000)
    t_loader = DataLoader(EEGDataset(train_list), batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=2, pin_memory=True,
                          persistent_workers=True,drop_last=True )
    v_loader = DataLoader(EEGDataset(val_list),   batch_size=BATCH_SIZE,
                          num_workers=2, pin_memory=True,
                          persistent_workers=True,drop_last=True )

    p23 = UnifiedGraphEncoder().to(DEVICE)
    p4  = TransformerSSL().to(DEVICE)
    p5  = SeizureClassifier().to(DEVICE)
    SSL_CKPT = "/content/drive/MyDrive/TUH_EEG_SEIZURE_v2/transformer_ssl_pretrained.pth"
    P23_CKPT = "/content/drive/MyDrive/TUH_EEG_SEIZURE_v2/p23_ssl_pretrained.pth"
    best_ckpt_path = os.path.join(SAVE_DIR, "best_model.pth")
    best_f1 = 0.0
    best_score=0.0
    best_tau = 0.5
    start_epoch = 0
    missing1, unexpected1 = [], []
    missing2, unexpected2 = [], []
    if os.path.exists(best_ckpt_path):
      ckpt = torch.load(best_ckpt_path,weights_only=False)
      missing1,unexpected1=p23.load_state_dict(ckpt["p23"], strict=False)
      missing2,unexpected2=p4.load_state_dict(ckpt["p4"],   strict=False)
      p5_state = ckpt["p5"]
      mismatch_keys = [k for k in p5_state if "temporal_head.2" in k]  # last Linear
      for k in mismatch_keys:
        del p5_state[k]
      p5.load_state_dict(p5_state ,   strict=False)
      for name, param in p5.named_parameters():
        if "temporal_head.2" in name:
            param.requires_grad = False
      best_f1  =ckpt.get("best_f1",0.0) # add best_f1 to your save dict too
      best_tau = ckpt.get("best_tau", 0.5)
      print(f"✅ Resumed from best_model.pth (F1={best_f1:.4f})")
      start_epoch = max(ckpt["epoch"], WARMUP_EPOCHS + TRANSITION_EPOCHS)
    else:

      p23.load_state_dict(torch.load(P23_CKPT, weights_only=False), strict=False)
      p4.load_state_dict(torch.load(SSL_CKPT,  weights_only=False), strict=False)
      print("✅ Started from ssl_pretrained.pth")
      p5.apply(init_weights)
    print("p23 missing:", len(missing1), "unexpected:", len(unexpected1))
    print("p4 missing:",  len(missing2), "unexpected:", len(unexpected2))
    for k in p23.state_dict():
        if "gat" in k or "projector" in k:
            print("p23 sample:", p23.state_dict()[k].flatten()[:3]); break
    # In train(), right after loading checkpoints, add:
    for param in p23.parameters():
      param.requires_grad = False
    for param in p4.parameters():
      param.requires_grad = False

    # FIX 7: Backbone LR raised from 1e-6 → 5e-6.
    # SSL pretraining may not have created preictal-separable features.
    # Mild backbone adaptation is necessary.
    opt = torch.optim.AdamW([
        {"params": p23.parameters(), "lr": 5e-7,  "weight_decay": 1e-4},
        {"params": p4.parameters(),  "lr": 5e-7,  "weight_decay": 1e-4},
        {"params": p5.parameters(),  "lr": 3e-5,  "weight_decay": 1e-4},
    ])
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=25, T_mult=2, eta_min=5e-7)

    focal_crit  = CalibratedFocalLoss(
        counts, gamma=FOCAL_GAMMA,
        minority_boost=3.5, min_alpha=0.05).to(DEVICE)
    f1_aux_crit = SoftPreictalF1Loss().to(DEVICE)

    warmup_weights = torch.tensor([1.0 / np.sqrt(c + 1) for c in counts],dtype=torch.float32).to(DEVICE)
    warmup_weights = warmup_weights / warmup_weights.sum() * 3

    warmup_crit = nn.CrossEntropyLoss(
        weight=warmup_weights, label_smoothing=LABEL_SMOOTHING)

    curriculum = CurriculumLoss(
        focal_crit, f1_aux_crit, warmup_crit,
        warmup_epochs=WARMUP_EPOCHS,
        transition_epochs=TRANSITION_EPOCHS).to(DEVICE)


    params   = (list(p23.parameters())
                + list(p4.parameters())
                + list(p5.parameters()))
    patience = 15
    no_improve = 0
    for ep in range(start_epoch,EPOCHS):
        p23.train(); p4.train(); p5.train()
        phase = ("WARMUP" if ep < WARMUP_EPOCHS
                 else f"TRANS" if ep < WARMUP_EPOCHS + TRANSITION_EPOCHS
                 else "FOCAL+F1")

        total_loss_ep, n_batches, nan_batches = 0.0, 0, 0
        opt.zero_grad()

        pbar = tqdm(t_loader, desc=f"Ep {ep+1}/{EPOCHS} [{phase}]")
        for step, (n, a, m, l) in enumerate(pbar):
            n, a, m, l = n.to(DEVICE), a.to(DEVICE), m.to(DEVICE), l.to(DEVICE)
            n, a = sanitize_batch(n, a)

            feat = p23(n, a, m)
            feat = feat[0] if isinstance(feat, tuple) else feat
            feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

            cls, tokens = p4(feat)
            cls    = torch.nan_to_num(cls,    nan=0.0, posinf=0.0, neginf=0.0)
            tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)

            logits, temporal_logits = p5(cls, tokens)
            logits = torch.nan_to_num(logits, nan=0.0)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_batches += 1; opt.zero_grad(); continue

            loss, f1_term, _ = curriculum(logits, temporal_logits, l, ep)
            if not torch.isfinite(loss):
                nan_batches += 1; opt.zero_grad(); continue

            (loss / GRAD_ACCUM_STEPS).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    nan_batches += 1; opt.zero_grad(); continue
                opt.step()
                opt.zero_grad()

            n_batches     += 1
            total_loss_ep += loss.item()
            pbar.set_postfix({
                'loss':  f'{loss.item():.4f}',
                'f1aux': f'{f1_term.item():.4f}',
                'nan_b': nan_batches
            })

        if ep >= WARMUP_EPOCHS and not (ep == start_epoch and start_epoch >= WARMUP_EPOCHS):
            cosine_scheduler.step()
        # Inside training loop, add this block:
        if ep == start_epoch and start_epoch >= WARMUP_EPOCHS:
          for param in p23.parameters():
            param.requires_grad = True
          for param in p4.parameters():
            param.requires_grad = True
          opt = torch.optim.AdamW([
            {"params": p23.parameters(), "lr": 2e-6, "weight_decay": 1e-4},
            {"params": p4.parameters(),  "lr": 2e-6, "weight_decay": 1e-4},
            {"params": p5.parameters(),  "lr": 3e-5, "weight_decay": 1e-4},
          ])
          cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=20, T_mult=2, eta_min=5e-7  # Fresh scheduler
          )
          print("  🔓 Backbones unfrozen with fresh scheduler.")

        if nan_batches > 0:
            pct = 100 * nan_batches / max(nan_batches + n_batches, 1)
            print(f"  Epoch {ep+1}: {nan_batches} NaN batches ({pct:.1f}%).")

        print(f"  Avg train loss: {total_loss_ep / max(n_batches, 1):.4f}")
        f1, tau, val_loss = evaluate(p23, p4, p5, v_loader, curriculum, epoch=ep)
        print(f"Ep {ep+1} | Preictal F1: {f1:.4f} | best_τ: {tau:.3f} "
              f"| val_loss: {val_loss:.4f}")

        if f1 > best_f1:
            no_improve = 0
            best_f1  = f1
            best_tau = tau
            torch.save({
                'p23': p23.state_dict(), 'p4': p4.state_dict(), 'best_f1': best_f1,
                'p5': p5.state_dict(), 'best_tau': best_tau,
                'epoch': ep + 1, 'counts': counts
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  ✅ BEST MODEL SAVED  (F1={best_f1:.4f}, τ={best_tau:.3f})")
            for name, param in p5.named_parameters():
              if "temporal_head.2" in name:
                param.requires_grad = True
        else:
           no_improve += 1
           if no_improve >= patience:
            print("Early stopping."); break

    print(f"\nTraining complete. Best preictal F1: {best_f1:.4f} "
          f"at τ={best_tau:.3f}")


# ── Data audit ───────────────────────────────────────────────────
def audit_adjacency(max_files=50):
    files = [f for f in os.listdir(DATA_DIR)
             if f.endswith("_adj.npy")][:max_files]
    total, bad = 0, 0
    for f in files:
        adj  = np.load(os.path.join(DATA_DIR, f))
        rs   = adj.sum(axis=2).sum(axis=-1)
        bad += ((rs == 0).any(axis=-1)).sum()
        total += len(adj)
    pct = 100 * bad / max(total, 1)
    print(f"Audit: {bad}/{total} windows ({pct:.1f}%) have all-zero adj row.")


# ── Inference ────────────────────────────────────────────────────
def predict_with_trained_threshold(logits, tau,
                                   ictal_tau=ICTAL_THRESHOLD):
    probs = F.softmax(logits, dim=1)
    pred  = torch.argmax(probs, dim=1).clone()
    margin = probs[:,1] - probs[:,0]
    preictal_mask = ((probs[:,1] > tau) & (margin > 0.3) & (probs[:,1] > probs[:,2]))
    pred[preictal_mask] = 1
    ictal_mask          = (probs[:, 2] > ictal_tau) & ~preictal_mask
    pred[ictal_mask]    = 2
    return pred


if __name__ == "__main__":
    train()