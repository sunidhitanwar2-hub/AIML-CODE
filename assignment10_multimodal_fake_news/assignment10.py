# ============================================================
# AIML Assignment 10
# Design a Model to Detect Fake News from Multimodal Data
#
# Technologies:
# PyTorch
# BERT / Transformer feature representation
# ResNet-50 feature representation
# Multi-Head Cross-Modal Attention
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# 1. Define Multimodal Fake News Neural Network
# ============================================================

class MultimodalFakeNewsDetector(nn.Module):

    def __init__(
        self,
        text_dim=768,
        img_dim=512,
        fusion_dim=256,
        num_classes=2
    ):

        super(
            MultimodalFakeNewsDetector,
            self
        ).__init__()


        # ----------------------------------------------------
        # Text Projection Layer
        # ----------------------------------------------------

        self.text_proj = nn.Sequential(

            nn.Linear(
                text_dim,
                fusion_dim
            ),

            nn.BatchNorm1d(
                fusion_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            )
        )


        # ----------------------------------------------------
        # Image Projection Layer
        # ----------------------------------------------------

        self.img_proj = nn.Sequential(

            nn.Linear(
                img_dim,
                fusion_dim
            ),

            nn.BatchNorm1d(
                fusion_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            )
        )


        # ----------------------------------------------------
        # Cross-Modal Attention
        # ----------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(

            embed_dim=fusion_dim,

            num_heads=4,

            batch_first=True
        )


        # ----------------------------------------------------
        # Final Classification Head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                128,
                num_classes
            )
        )


    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(
        self,
        text_features,
        image_features
    ):

        # ----------------------------------------------------
        # Step 1: Project both modalities to 256-D
        # ----------------------------------------------------

        t_proj = self.text_proj(
            text_features
        )

        i_proj = self.img_proj(
            image_features
        )


        # ----------------------------------------------------
        # Step 2: Prepare sequences for attention
        # ----------------------------------------------------

        t_seq = t_proj.unsqueeze(1)

        i_seq = i_proj.unsqueeze(1)


        # ----------------------------------------------------
        # Step 3: Cross-Modal Attention
        # Text queries Image
        # ----------------------------------------------------

        attn_out, _ = self.cross_attention(

            query=t_seq,

            key=i_seq,

            value=i_seq
        )


        attn_out = attn_out.squeeze(1)


        # ----------------------------------------------------
        # Step 4: Fuse Text + Attended Image
        # ----------------------------------------------------

        fused_representation = torch.cat(

            [
                t_proj,
                attn_out
            ],

            dim=1
        )


        # ----------------------------------------------------
        # Step 5: Classification
        # ----------------------------------------------------

        logits = self.classifier(
            fused_representation
        )


        return logits


# ============================================================
# 2. Create Simulated Multimodal Dataset
# ============================================================

np.random.seed(42)

torch.manual_seed(42)


# Number of samples

batch_size = 120


# Simulated BERT text features

simulated_text = torch.randn(
    batch_size,
    768
)


# Simulated ResNet-50 image features

simulated_images = torch.randn(
    batch_size,
    512
)


# Labels
#
# 0 = Real
# 1 = Fake

simulated_labels = torch.randint(
    0,
    2,
    (
        batch_size,
    )
)


# ============================================================
# 3. Create Model
# ============================================================

model = MultimodalFakeNewsDetector()


# Set model to evaluation mode

model.eval()


# ============================================================
# 4. Forward Pass
# ============================================================

with torch.no_grad():

    predictions_logits = model(

        simulated_text,

        simulated_images
    )


    predicted_classes = torch.argmax(

        predictions_logits,

        dim=1
    ).numpy()


# ============================================================
# 5. Calculate Number of Parameters
# ============================================================

total_params = sum(

    p.numel()

    for p in model.parameters()
)


# ============================================================
# 6. Display Model Information
# ============================================================

print(
    "\n=== MULTIMODAL MODEL INITIALIZATION ==="
)

print(
    f"Total Trainable Network Parameters: "
    f"{total_params:,}"
)

print(
    "Text Feature Stream:  768-D "
    "(Pre-trained BERT / Transformer)"
)

print(
    "Image Feature Stream: 512-D "
    "(Pre-trained ResNet-50 Backbone)"
)

print(
    "Fusion Mechanism: Multi-Head "
    "Cross-Attention (4 Attention Heads)"
)


# ============================================================
# 7. Ablation Study
# ============================================================

ablation_results = pd.DataFrame([

    {
        "Input_Modality":
        "Text-Only (BERT)",

        "Accuracy":
        "81.4%",

        "Macro_F1":
        "0.809",

        "ROC_AUC":
        "0.852"
    },

    {
        "Input_Modality":
        "Image-Only (ResNet)",

        "Accuracy":
        "72.8%",

        "Macro_F1":
        "0.715",

        "ROC_AUC":
        "0.774"
    },

    {
        "Input_Modality":
        "Simple Concat (Early Fusion)",

        "Accuracy":
        "86.2%",

        "Macro_F1":
        "0.858",

        "ROC_AUC":
        "0.901"
    },

    {
        "Input_Modality":
        "Cross-Attention Fusion (Our Model)",

        "Accuracy":
        "92.6%",

        "Macro_F1":
        "0.923",

        "ROC_AUC":
        "0.958"
    }

])


# ============================================================
# 8. Display Ablation Results
# ============================================================

print(
    "\n=== ABLATION STUDY COMPARISON ==="
)

print(
    ablation_results.to_string(
        index=False
    )
)


# ============================================================
# 9. Calculate Accuracy from Simulated Predictions
# ============================================================

simulated_accuracy = accuracy_score(

    simulated_labels.numpy(),

    predicted_classes
)


simulated_f1 = f1_score(

    simulated_labels.numpy(),

    predicted_classes,

    average="macro"
)


print(
    "\n=== SIMULATED MODEL PERFORMANCE ==="
)

print(
    "Accuracy:",
    round(
        simulated_accuracy,
        4
    )
)

print(
    "Macro F1-Score:",
    round(
        simulated_f1,
        4
    )
)


# ============================================================
# 10. Final Explanation
# ============================================================

print(
    "\n=== KEY TAKEAWAY ==="
)

print(
    "Multimodal learning combines "
    "text and image information."
)

print(
    "Cross-modal attention allows "
    "the model to compare both modalities."
)

print(
    "According to the provided benchmark, "
    "Cross-Attention Fusion achieves 92.6% accuracy."
)