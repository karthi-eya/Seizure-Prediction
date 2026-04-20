import numpy as np
import torch


# ── FIX: Temporal Smoothing ─────────────────────────────────────
def temporally_smooth_predictions(probs, best_tau, window=5, min_consecutive=3):
    """
    Only flag preictal if sustained across consecutive windows.
    Eliminates isolated false positives from single noisy frames.
    """
    preds = np.argmax(probs, axis=1)
    preictal_binary = (probs[:, 1] > best_tau).astype(float)
    smoothed_score  = np.convolve(preictal_binary, np.ones(window) / window, mode='same')

    for i in range(len(preds)):
        if smoothed_score[i] >= (min_consecutive / window):
            preds[i] = 1
        elif preds[i] == 1:
            preds[i] = 0  # revert isolated preictal
    return preds


# ── FIX: Refractory Period ──────────────────────────────────────
def apply_refractory_period(preds, refractory_windows=30):
    """
    After ictal detection suppress preictal alerts for refractory_windows.
    Post-ictal brain cannot be pre-ictal — prevents post-ictal false alarms.
    """
    preds = preds.copy()
    refractory_counter = 0
    for i in range(len(preds)):
        if refractory_counter > 0:
            if preds[i] == 1:
                preds[i] = 0  # force interictal during refractory
            refractory_counter -= 1
        if preds[i] == 2:
            refractory_counter = refractory_windows
    return preds


# ── FIX: Event-Level Clinical Metrics ──────────────────────────
def event_level_metrics(preds, labels, window_size_sec=3.0,
                         min_lead_time_windows=5, max_lead_time_windows=75):
    """
    Clinical metric — what actually matters for a wearable/implant:
      TP = preictal alert fired between min and max windows before ictal onset
      FP = alert fired with no upcoming seizure in that horizon
      FN = ictal onset with no prior alert
    """
    ictal_starts = np.where(np.diff((labels == 2).astype(int)) == 1)[0]

    tp, fn = 0, 0
    valid_preictal_mask = np.zeros(len(labels), dtype=bool)

    for onset in ictal_starts:
        w_start = max(0, onset - max_lead_time_windows)
        w_end   = max(0, onset - min_lead_time_windows)
        valid_preictal_mask[w_start:w_end] = True
        if np.any(preds[w_start:w_end] == 1):
            tp += 1
        else:
            fn += 1

    fp = int(np.sum((preds == 1) & ~valid_preictal_mask & (labels != 1)))

    sensitivity    = tp / max(tp + fn, 1)
    total_hours    = (len(labels) * window_size_sec) / 3600
    fpr_per_hour   = fp / max(total_hours, 1e-8)

    print(f"  Event-level Sensitivity : {sensitivity:.3f}")
    print(f"  False Positive Rate     : {fpr_per_hour:.2f}/hour")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    return sensitivity, fpr_per_hour


# ── Main Analysis ───────────────────────────────────────────────
def run_analysis(p5, cls, spatial_nodes, band_weights, tokens=None, labels=None,
                 best_tau=0.45, window_size_sec=3.0,preds=None):
    output = p5.predict_full(cls, tokens=tokens, spatial_nodes=spatial_nodes,
                             band_weights=band_weights)
    
    risk_score   = output["preictal_risk"]
    lower, upper = output["confidence_interval"]
    trajectory   = output["trajectory"]
    if trajectory is not None:
      traj_np = trajectory.detach().cpu().numpy()
    else:
      traj_np = None
    drivers      = output["driver_channels"]
    bands        = output["band_contributions"]
    chan_scores  = output["channel_scores"]

    risk_np  = risk_score.detach().cpu().numpy().flatten()
    lower_np = lower.detach().cpu().numpy().flatten()
    upper_np = upper.detach().cpu().numpy().flatten()
    bands_np = bands.detach().cpu().numpy()      if bands       is not None else None
    chan_np  = chan_scores.detach().cpu().numpy() if chan_scores is not None else None
    if preds is None:  # fallback only
        proxy_probs = np.stack([1.0 - risk_np, risk_np, np.zeros_like(risk_np)], axis=1)
        preds = temporally_smooth_predictions(proxy_probs, best_tau)
        preds = apply_refractory_period(preds)
    smoothed_preds = preds
    # ── FIX: Apply temporal smoothing + refractory on risk probs ──
    # Build a fake probs array from risk_score for smoothing
    # (assumes risk_score is preictal probability per window)
    risk_flat = risk_np
    # Construct 3-class proxy probs for smoothing pipeline
    

    #smoothed_preds = temporally_smooth_predictions(proxy_probs, best_tau)
    #smoothed_preds = apply_refractory_period(smoothed_preds)

    # ── Event-level metrics (only if labels provided) ─────────────
    if labels is not None:
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        print("\n── Clinical Event-Level Metrics ──")
        sensitivity, fpr_per_hour = event_level_metrics(
            smoothed_preds, labels_np, window_size_sec=window_size_sec
        )
    else:
        sensitivity, fpr_per_hour = None, None

    result = {
        # Risk graph (MC Dropout backed)
        "risk_plot": {
            "x":     list(range(len(risk_np))),
            "y":     risk_np.tolist(),
            "lower": lower_np.tolist(),
            "upper": upper_np.tolist(),
            "gt":    labels.cpu().numpy().tolist()
                     if labels is not None else None
        },

        # Trajectory graph
        "trajectory": {
            "x":   list(range(traj_np.shape[1])),
            "y":   traj_np.mean(axis=0).tolist(),
            "std": traj_np.std(axis=0).tolist()
        }if traj_np is not None else None,
        

        # Clinically grounded attributions
        "clinical_drivers": {
            "top_channels":  drivers,
            "channel_scores": chan_np.tolist()  if chan_np  is not None else None,
            "band_weights":   bands_np.tolist() if bands_np is not None else None
        },

        # FIX: Clinical metrics surfaced to frontend
        "clinical_metrics": {
            "sensitivity":   sensitivity,
            "fpr_per_hour":  fpr_per_hour,
            "smoothed_preds": smoothed_preds.tolist()
        }
    }
    return result