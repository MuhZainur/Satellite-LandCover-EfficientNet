"""Trains EfficientNet-B0 on EuroSAT Sentinel-2 tiles and exports to PyTorch and ONNX.
Usage:
    python train.py --epochs 5 --limit 150
"""
from __future__ import annotations

import argparse
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.classifier import build_model, evaluate, train
from src.dataset_loader import get_loaders

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=5, help="jumlah epoch training")
ap.add_argument("--limit", type=int, default=150, help="jumlah sampel per kelas")
ap.add_argument("--batch", type=int, default=32, help="batch size")
ap.add_argument("--out", default="model.pt", help="output path bobot model")
ap.add_argument("--onnx", default="model.onnx", help="output path model ONNX")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Computing device: {dev.upper()}")

t0 = time.time()
train_dl, val_dl, test_dl, classes = get_loaders(batch=a.batch, limit_per_class=a.limit)
print(f"[INFO] Dataset loaded: {len(classes)} classes | Train batches: {len(train_dl)} | Val batches: {len(val_dl)} | Test batches: {len(test_dl)}")

model = build_model(len(classes), pretrained=True)
best_state = train(model, train_dl, val_dl, a.epochs, dev)
model.load_state_dict(best_state)

# Save PyTorch checkpoint
torch.save({"state": model.state_dict(), "classes": classes}, a.out)
print(f"[SAVED] PyTorch model checkpoint saved to: {a.out}")

# Evaluate on test set
yt, yp = evaluate(model, test_dl, dev)
acc = (yt == yp).mean()
print(f"\n--- Final Test Set Evaluation (Overall Accuracy: {acc:.1%}) ---")
print(classification_report(yt, yp, target_names=classes, digits=3, zero_division=0))

# Confusion Matrix
plt.figure(figsize=(9, 8))
cm = confusion_matrix(yt, yp)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes)
plt.title(f"EuroSAT Sentinel-2 Confusion Matrix (Test Acc: {acc:.1%})", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Land Cover", fontsize=10)
plt.ylabel("True Ground Truth", fontsize=10)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print(f"[SAVED] Confusion matrix saved to: confusion_matrix.png")

# Export to ONNX for edge deployment
try:
    dummy_input = torch.randn(1, 3, 128, 128).to(dev)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        a.onnx,
        input_names=["satellite_tile"],
        output_names=["landcover_logits"],
        dynamic_axes={"satellite_tile": {0: "batch_size"}, "landcover_logits": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"[SAVED] Model successfully exported to ONNX: {a.onnx} ({os.path.getsize(a.onnx)/(1024*1024):.2f} MB)")
except Exception as e:
    print(f"[WARN] ONNX export skipped: {e}")

t1 = time.time()
print(f"[DONE] Complete pipeline finished in {t1 - t0:.2f}s!")

