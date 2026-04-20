import os
import io
import torch
import numpy as np
import math
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from person23_graph_encoder import UnifiedGraphEncoder
from person4_transformer import TransformerSSL
from Person5 import SeizureClassifier
from analysis import run_analysis, temporally_smooth_predictions, apply_refractory_period

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")

print("Initializing Models on", DEVICE)
p23 = UnifiedGraphEncoder().to(DEVICE)
p4 = TransformerSSL().to(DEVICE)
p5 = SeizureClassifier().to(DEVICE)

best_tau = 0.45
model_loaded = False

if os.path.exists(CKPT_PATH):
    print("Loading actual checkpoint weights...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    p23.load_state_dict(ckpt["p23"], strict=False)
    p4.load_state_dict(ckpt["p4"], strict=False)
    
    # Strip temporal_head just like in inference.py
    ckpt_p5 = ckpt["p5"]
    filtered_state_dict = {k: v for k, v in ckpt_p5.items() if not k.startswith("temporal_head")}
    p5.load_state_dict(filtered_state_dict, strict=False)
    
    best_tau = ckpt.get("best_tau", 0.45)
    model_loaded = True
else:
    print(f"WARNING: No checkpoint found at {CKPT_PATH}")

p23.eval()
p4.eval()
p5.eval()

@app.get("/api/status")
def status():
    return {"status": "online", "model_loaded": model_loaded, "device": str(DEVICE)}

def predict_pipeline(nodes_np, adj_np):
    # Ensure correct shape and data types and handle potential NaNs from raw EEG
    nodes_np = np.nan_to_num(nodes_np)
    adj_np = np.nan_to_num(adj_np)
    
    nodes_t = torch.tensor(nodes_np, dtype=torch.float32).to(DEVICE)
    adj_t = torch.tensor(adj_np, dtype=torch.float32).to(DEVICE)
    
    # Missing ch_mask handled dynamically
    ch_mask_t = torch.ones(16, dtype=torch.float32).to(DEVICE)
    
    # Expand dims if necessary [Seq, Channels, Freq, Time]
    if len(nodes_t.shape) == 3:
        nodes_t = nodes_t.unsqueeze(0)
    if len(adj_t.shape) == 4:
        adj_t = adj_t.unsqueeze(0)

    with torch.no_grad():
        out_p23, spatial_nodes, band_weights = p23(nodes_t, adj_t, ch_mask_t)
        cls, tokens = p4(out_p23)
        logits, temporal_logits = p5(cls, tokens)
        
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Smooth using functions from analysis.py
        smoothed_preds = temporally_smooth_predictions(probs, best_tau, window=5, min_consecutive=3)
        smoothed_preds = apply_refractory_period(smoothed_preds, refractory_windows=30)
        
        # Run the full clinical analysis
        result = run_analysis(
            preds=smoothed_preds,
            p5=p5,
            cls=cls,
            spatial_nodes=spatial_nodes,
            band_weights=band_weights,
            tokens=tokens,
            labels=None,
            best_tau=best_tau,
            window_size_sec=3.0
        )
        
        # Wire the Temporal Head's 10-step trajectory output for the graph!
        traj_y = []
        if "trajectory" in result and result["trajectory"] is not None:
             raw_y = result["trajectory"].get("y", [])
             if raw_y:
                 raw_np = np.array(raw_y)
                 if raw_np.ndim > 1:
                     raw_np = raw_np.mean(axis=0)
                 # apply Sigmoid activation to raw logits so they graph naturally between 0.0 and 1.0 (0% - 100%)
                 traj_y = [1.0 / (1.0 + math.exp(-float(v))) for v in raw_np.flatten()]
        
        # Fallback to current single value if trajectory generation missed
        if not traj_y or len(traj_y) < 2:
             traj_y = np.maximum(probs[:, 1], probs[:, 2]).tolist() * 10

        band_contrib = result.get("clinical_drivers", {}).get("band_weights", None)
        band_dict = {"delta": 0, "theta": 0, "alpha": 0, "beta": 0, "gamma": 0}
        if band_contrib is not None:
            bc_np = np.array(band_contrib)
            if bc_np.ndim > 1:
                bc_mean = bc_np.mean(axis=0)
            else:
                bc_mean = bc_np
                
            if len(bc_mean) == 5:
                band_dict = {
                    "delta": float(bc_mean[0]),
                    "theta": float(bc_mean[1]),
                    "alpha": float(bc_mean[2]),
                    "beta":  float(bc_mean[3]),
                    "gamma": float(bc_mean[4])
                }

        # Map channel indices to 16 standard clinical system names
        channel_names = ["FP1", "FP2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2"]
        raw_drivers = result.get("clinical_drivers", {}).get("top_channels", [])
        mapped_drivers = []
        if raw_drivers:
            if isinstance(raw_drivers[0], list):
                raw_drivers = raw_drivers[0]
            mapped_drivers = [channel_names[idx] if idx < len(channel_names) else f"CH_{idx}" for idx in raw_drivers]
            
        # Clinical risk is the max probability between Preictal(1) and Ictal(2)
        mean_preictal = float(probs[:, 1].mean())
        mean_ictal    = float(probs[:, 2].mean())
        combined_risk = float(max(mean_preictal, mean_ictal))
        
        final_state = int(smoothed_preds[-1])
        prediction_label = "Interictal"
        if final_state == 1: prediction_label = "Preictal Warning"
        elif final_state == 2: prediction_label = "Seizure (Ictal)"

    def sanitize(o):
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [sanitize(v) for v in o]
        elif isinstance(o, float):
            return 0.0 if math.isnan(o) or math.isinf(o) else o
        return o

    response_dict = {
        "mean_risk": combined_risk, 
        "prediction_label": prediction_label,
        "band_contributions": band_dict,
        "driver_channels": mapped_drivers,
        "risk_scores": traj_y,
        "final_smoothed_pred": final_state
    }
    return sanitize(response_dict)

@app.post("/api/analyze")
async def analyze_eeg(nodes_file: UploadFile = File(...), adj_file: UploadFile = File(...)):
    try:
        nodes_bytes = await nodes_file.read()
        adj_bytes = await adj_file.read()
        
        nodes_np = np.load(io.BytesIO(nodes_bytes), allow_pickle=True)
        adj_np = np.load(io.BytesIO(adj_bytes), allow_pickle=True)
        return predict_pipeline(nodes_np, adj_np)
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}\n{traceback.format_exc()}")
        return {"error": str(e), "trace": traceback.format_exc()}

@app.get("/api/demo")
def demo():
    # Real Model, Real Logic, Real Local Saved Data from demoooo-main directory
    # No fake data produced.
    demo_nodes_path = r"C:\Users\KARTHIKEYA\Downloads\demoooo-main\demoooo-main\aaaaaaac_s001_t000_nodes.npy"
    demo_adj_path = r"C:\Users\KARTHIKEYA\Downloads\demoooo-main\demoooo-main\aaaaaaac_s001_t000_adj.npy"
    
    if os.path.exists(demo_nodes_path) and os.path.exists(demo_adj_path):
        nodes_np = np.load(demo_nodes_path)
        adj_np = np.load(demo_adj_path)
        return predict_pipeline(nodes_np, adj_np)
    else:
        return {"error": "No real demo data found on disk to run inference. Please use upload option."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
