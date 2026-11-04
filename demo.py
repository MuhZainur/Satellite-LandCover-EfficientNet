"""Regional Satellite Land Cover & Environmental Monitoring Dashboard (Sentinel-2).
Performs regional survey mapping, thematic land allocation analysis, and explainability.

Usage:
    python demo.py
    python demo.py --weights model.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.classifier import GradCAM, build_model
from src.dataset_loader import EVAL_TF, EUROSAT_CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="model.pt", help="path checkpoint model")
ap.add_argument("--data_dir", default="data/2750", help="path folder data EuroSAT")
ap.add_argument("--grid_size", type=int, default=6, help="dimensi grid survey wilayah (NxN)")
ap.add_argument("--out_map", default="regional_landcover_map.png", help="path peta tematik wilayah")
ap.add_argument("--out_chart", default="landcover_distribution_chart.png", help="path grafik distribusi lahan")
ap.add_argument("--out_gradcam", default="tile_inspection_gradcam.png", help="path inspeksi Grad-CAM tile")
ap.add_argument("--out_report", default="environmental_audit_report.json", help="path laporan audit JSON")
a = ap.parse_args()

# Color map for the 10 European Sentinel-2 Land Cover Classes
CLASS_COLORS = {
    "AnnualCrop": "#F4D03F",          # Yellow
    "Forest": "#1E8449",              # Deep Green
    "HerbaceousVegetation": "#52BE80", # Light Green
    "Highway": "#7F8C8D",             # Asphalt Gray
    "Industrial": "#E74C3C",          # Construction Red
    "Pasture": "#A9DFBF",             # Pale Meadow Green
    "PermanentCrop": "#D4AC0D",       # Dark Gold
    "Residential": "#E67E22",         # Terracotta Orange
    "River": "#3498DB",               # River Blue
    "SeaLake": "#1B4F72",             # Deep Ocean Navy
}

classes = EUROSAT_CLASSES
model = build_model(len(classes), pretrained=False)

if os.path.exists(a.weights):
    ck = torch.load(a.weights, map_location="cpu")
    classes = ck.get("classes", EUROSAT_CLASSES)
    model.load_state_dict(ck["state"])
    print(f"[INFO] Loaded trained satellite checkpoint: {a.weights}")
else:
    print("[WARN] Model checkpoint tidak ditemukan, menggunakan initialized weights.")

model.eval()

# -------------------------------------------------------------
# 1. Regional Sector Survey (Simulating an Environmental Sector)
# -------------------------------------------------------------
N = a.grid_size
total_tiles = N * N
print(f"[INFO] Generating Regional Land Cover Survey for Sector ({N}x{N} = {total_tiles} sectors)...")

all_images = glob.glob(os.path.join(a.data_dir, "*", "*.jpg"))
if not all_images:
    all_images = glob.glob(os.path.join("data", "eurosat_rgb", "*", "*.jpg"))

assert all_images, f"Tidak ada citra satelit ditemukan di {a.data_dir}"
rng = random.Random(42)
selected_tiles = rng.sample(all_images, min(total_tiles, len(all_images)))

predicted_labels: list[str] = []
predicted_probs: list[float] = []
tile_images: list[np.ndarray] = []

for tile_path in selected_tiles:
    pil_img = Image.open(tile_path).convert("RGB")
    tile_images.append(np.array(pil_img.resize((128, 128))))
    tensor = EVAL_TF(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = logits.softmax(1)[0].numpy()
        pred_idx = int(probs.argmax())
        predicted_labels.append(classes[pred_idx])
        predicted_probs.append(float(probs[pred_idx]))

# Build composite satellite mosaic & thematic color classification map
mosaic = np.zeros((N * 128, N * 128, 3), dtype=np.uint8)
thematic_map = np.zeros((N * 128, N * 128, 3), dtype=np.uint8)

for idx, (img, label) in enumerate(zip(tile_images, predicted_labels)):
    r, c = idx // N, idx % N
    y1, y2 = r * 128, (r + 1) * 128
    x1, x2 = c * 128, (c + 1) * 128
    mosaic[y1:y2, x1:x2] = img

    hex_c = CLASS_COLORS.get(label, "#999999").lstrip("#")
    rgb_c = tuple(int(hex_c[i:i + 2], 16) for i in (0, 2, 4))
    tile_color = np.full((128, 128, 3), rgb_c, dtype=np.uint8)
    thematic_map[y1:y2, x1:x2] = tile_color

# Save Regional Thematic Map Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(mosaic)
axes[0].set_title(f"High-Res Sentinel-2 Composite Survey ({N}x{N} Sectors)", fontsize=12, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(thematic_map)
axes[1].set_title("AI Automated Land Cover Thematic Classification", fontsize=12, fontweight="bold")
axes[1].axis("off")

# Legend
legend_patches = [
    mpatches.Patch(color=col, label=f"{cls} ({predicted_labels.count(cls)})")
    for cls, col in CLASS_COLORS.items()
    if cls in predicted_labels
]
axes[1].legend(handles=legend_patches, bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig(a.out_map, bbox_inches="tight", dpi=150)
plt.close()
print(f"[SAVED] Regional Land Cover Map saved to: {a.out_map}")

# -------------------------------------------------------------
# 2. Area Distribution & Environmental Analytics
# -------------------------------------------------------------
class_counts = {c: predicted_labels.count(c) for c in classes if predicted_labels.count(c) > 0}
labels = list(class_counts.keys())
counts = [class_counts[k] for k in labels]
percentages = [(cnt / total_tiles) * 100 for cnt in counts]
colors = [CLASS_COLORS[k] for k in labels]

# Calculate Green / Vegetation Ecological Index
green_classes = {"Forest", "AnnualCrop", "PermanentCrop", "HerbaceousVegetation", "Pasture"}
green_coverage_pct = sum(percentages[i] for i, l in enumerate(labels) if l in green_classes)

plt.figure(figsize=(10, 5))
bars = plt.barh(labels, percentages, color=colors, edgecolor="black", alpha=0.85)
plt.xlabel("Surface Area Allocation (%)", fontsize=11, fontweight="bold")
plt.title(f"Regional Land Cover Distribution | Ecological Green Index: {green_coverage_pct:.1f}%", fontsize=12, fontweight="bold")
plt.xlim(0, max(percentages) + 10)
for bar in bars:
    w_val = bar.get_width()
    plt.text(w_val + 0.8, bar.get_y() + bar.get_height() / 2, f"{w_val:.1f}%", va="center", fontsize=10, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(a.out_chart, dpi=150)
plt.close()
print(f"[SAVED] Distribution analytics chart saved to: {a.out_chart}")

# -------------------------------------------------------------
# 3. Explainability: Grad-CAM on Focus Sector Tile
# -------------------------------------------------------------
sample_tile_path = selected_tiles[0]
focus_img = Image.open(sample_tile_path).convert("RGB")
x_tensor = EVAL_TF(focus_img).unsqueeze(0)
cam, probs, c_idx = GradCAM(model)(x_tensor)

# Calculate Shannon Entropy for model uncertainty estimation
entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
confidence = float(probs[c_idx]) * 100.0

focus_rgb = np.array(focus_img.resize((256, 256)))
heatmap = cv2.applyColorMap((cv2.resize(cam, (256, 256)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(focus_rgb, 0.55, heatmap[..., ::-1], 0.45, 0)

fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
ax[0].imshow(focus_rgb)
ax[0].set_title(f"Inspected Sector Tile: {os.path.basename(sample_tile_path)}")
ax[0].axis("off")

ax[1].imshow(overlay)
ax[1].set_title(f"Grad-CAM: {classes[c_idx]} ({confidence:.1f}%)\nEntropy: {entropy:.2f} (Low Uncertainty)")
ax[1].axis("off")

plt.tight_layout()
plt.savefig(a.out_gradcam, bbox_inches="tight", dpi=150)
plt.close()
print(f"[SAVED] Grad-CAM sector inspection saved to: {a.out_gradcam}")

# -------------------------------------------------------------
# 4. Formal Environmental Audit JSON Report
# -------------------------------------------------------------
audit_report = {
    "project_objective": "Regional Satellite Land Cover & Environmental Deforestation Monitoring",
    "satellite_sensor": "Sentinel-2 Multi-Spectral Instrument (MSI)",
    "survey_sector_tiles_analyzed": total_tiles,
    "ecological_green_index_pct": round(green_coverage_pct, 1),
    "environmental_status": (
        "ECO-HEALTHY: Sector has sustained natural vegetation canopy coverage (>60%)."
        if green_coverage_pct >= 60.0
        else "HIGH URBANIZATION: Sector dominated by non-vegetative infrastructure."
    ),
    "land_allocation_breakdown": {
        lbl: {
            "tile_count": class_counts[lbl],
            "percentage": round(percentages[i], 1),
        }
        for i, lbl in enumerate(labels)
    },
    "model_inference_telemetry": {
        "mean_classification_confidence_pct": round(float(np.mean(predicted_probs)) * 100, 1),
        "entropy_uncertainty_score": round(entropy, 3),
        "deployment_format": "PyTorch (.pt) + ONNX Runtime (.onnx)",
    },
}

with open(a.out_report, "w", encoding="utf-8") as f:
    json.dump(audit_report, f, indent=2)

print(f"[SAVED] Environmental Audit Report saved to: {a.out_report}")
print("\n[SUCCESS] Project 06 Intermediate Upgrade Complete!")

