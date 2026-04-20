import torch
import torch.nn as nn
import torch.nn.functional as F

class SeizureClassifier(nn.Module):
    def __init__(self, in_dim=512, spatial_dim=128, num_channels=16):
        super().__init__()
        
        self.num_channels = num_channels

        # ── 1. Residual Feature Projection ──
        # Adds a "Skip Connection" to prevent vanishing gradients
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512), # Added for training stability
            nn.GELU(),
            nn.Dropout(0.4)      # Increased for better generalization
        )

        # ── 2. Risk Classifier (Multi-Head Architecture) ──
        # Using a deeper, regularized bottleneck for patent-level accuracy
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)    # 0: Interictal, 1: Preictal, 2: Ictal
        )

        # ── 3. Clinical Decoders (The "Expo" Features) ──
        
        # Trajectory: Predicts the next 10 seconds of risk levels
        self.temporal_head = nn.Sequential(
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Linear(128, 10)
        )

        # Spatial Attention: Identifies which electrode is the "Driver"
        self.spatial_attention = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, cls,tokens=None):
        # 🧪 PATENT TIP: Residual Connection
        # We add the original 'cls' back to the projected features 
        # to ensure the model doesn't "forget" the Transformer context.
        proj = self.feature_proj(cls)
        # Assuming in_dim is 512, we can add them directly
        combined = proj + cls if proj.shape == cls.shape else proj
        logits = self.classifier(combined)
        temporal_logits = None
        if tokens is not None:
          temporal_logits = self.temporal_head(tokens).squeeze(-1)  # [B,10]
        return logits, temporal_logits

    def predict_proba(self, cls,tokens=None):
      logits, _ = self.forward(cls,tokens)
      return torch.softmax(logits, dim=-1)

    def predict_full(self, cls,tokens=None, spatial_nodes=None, band_weights=None, runs=20):
        """
        Runs Monte Carlo (MC) Dropout to calculate 'Uncertainty'.
        A seizure prediction with high uncertainty = 'False Alarm' warning.
        """
        self.eval()

        # ✅ Enable Dropout during inference for MC Sampling
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        # ── 1. Confidence Intervals via MC Dropout ──
        with torch.no_grad():
            mc_probs = torch.stack([self.predict_proba(cls,tokens) for _ in range(runs)])

        # Mean Risk of Preictal (Index 1) for your 30-second lead goal
        mean_preictal_risk = mc_probs.mean(dim=0)[:, 1]
        mean_ictal_risk     = mc_probs.mean(dim=0)[:, 2]
        
        # Calculate Variance as Uncertainty
        uncertainty = mc_probs[:, :, 1].std(dim=0)

        # ✅ Restore eval mode
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

        # ── 2. Interpretability (XAI) for Judges ──
        with torch.no_grad():
            
            if tokens is not None:
              trajectory = self.temporal_head(tokens).squeeze(-1)
            else:
              trajectory = None
            
            drivers, channel_scores = None, None
            if spatial_nodes is not None:
                # spatial_nodes: [B, T, 16, 128]
                pooled_spatial = spatial_nodes.mean(dim=1) 
                attn_logits = self.spatial_attention(pooled_spatial).squeeze(-1)
                channel_scores = torch.softmax(attn_logits, dim=-1)
                drivers = torch.topk(channel_scores, 5, dim=1).indices.tolist()

        return {
            "confidence_interval": (mean_preictal_risk - 1.645 * uncertainty,mean_preictal_risk + 1.645 * uncertainty),
            "preictal_risk":       mean_preictal_risk,
            "ictal_risk":          mean_ictal_risk,
            "uncertainty":         uncertainty,
            "trajectory":          trajectory,
            "driver_channels":     drivers,
            "channel_scores":      channel_scores,
            "band_contributions":  band_weights
        }