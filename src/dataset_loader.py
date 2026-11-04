"""EuroSAT Sentinel-2 Satellite Dataset Loader & Stratified Sampler."""
from __future__ import annotations

import os
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

SIZE = 128  # 64px asli di-upsample agar feature map akhir EfficientNet kaya konteks
_norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
EVAL_TF = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), _norm])
TRAIN_TF = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    _norm
])

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


def get_loaders(
    root: str = "data",
    batch: int = 32,
    limit_per_class: int = 150,
    seed: int = 42
):
    """Loads Sentinel-2 EuroSAT RGB tiles with balanced stratified sampling."""
    cand_dirs = [
        os.path.join(root, "2750"),
        os.path.join(root, "eurosat_rgb"),
    ]
    target_dir = next((d for d in cand_dirs if os.path.isdir(d)), None)
    assert target_dir is not None, f"Folder EuroSAT tidak ditemukan di {root}"

    train_base = datasets.ImageFolder(target_dir, transform=TRAIN_TF)
    eval_base = datasets.ImageFolder(target_dir, transform=EVAL_TF)
    classes = train_base.classes

    # Stratified balanced sampling per class
    class_indices = defaultdict(list)
    for idx, (_, class_idx) in enumerate(train_base.samples):
        class_indices[class_idx].append(idx)

    rng = torch.Generator().manual_seed(seed)
    train_ids, val_ids, test_ids = [], [], []

    for c_idx in sorted(class_indices.keys()):
        all_ids = class_indices[c_idx]
        perm = torch.randperm(len(all_ids), generator=rng).tolist()
        sampled = [all_ids[i] for i in perm[:min(limit_per_class, len(all_ids))]]
        
        n = len(sampled)
        n_train = max(1, int(0.75 * n))
        n_val = max(1, int(0.15 * n))

        train_ids.extend(sampled[:n_train])
        val_ids.extend(sampled[n_train:n_train + n_val])
        test_ids.extend(sampled[n_train + n_val:])

    train_dl = DataLoader(Subset(train_base, train_ids), batch_size=batch, shuffle=True)
    val_dl = DataLoader(Subset(eval_base, val_ids), batch_size=batch, shuffle=False)
    test_dl = DataLoader(Subset(eval_base, test_ids), batch_size=batch, shuffle=False)

    return train_dl, val_dl, test_dl, classes

