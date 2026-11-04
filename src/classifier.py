"""EfficientNet-B0 transfer learning: model, training loop, evaluasi, dan Grad-CAM."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models


def build_model(n_classes: int = 10, pretrained: bool = True) -> torch.nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, n_classes)
    return m


def evaluate(model: torch.nn.Module, dl, dev: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred)."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.append(model(x.to(dev)).argmax(1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train(model: torch.nn.Module, train_dl, val_dl, epochs: int, dev: str, lr: float = 1e-3):
    """Fine-tune seluruh jaringan. Return state_dict dengan akurasi val terbaik."""
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr)
    best_acc, best = 0.0, None
    for ep in range(epochs):
        model.train()
        for x, y in train_dl:
            loss = F.cross_entropy(model(x.to(dev)), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        yt, yp = evaluate(model, val_dl, dev)
        acc = float((yt == yp).mean())
        print(f"epoch {ep + 1}/{epochs}  val acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best = acc, copy.deepcopy(model.state_dict())
    return best


class GradCAM:
    """Grad-CAM pada blok konvolusi terakhir (model.features[-1])."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model, self.act, self.grad = model, None, None
        model.features[-1].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out) -> None:
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, int]:
        """x: (1,3,H,W). Return (cam 0-1, probabilitas per kelas, indeks kelas prediksi)."""
        self.model.eval()
        self.model.zero_grad()
        out = self.model(x)
        c = int(out.argmax(1))
        out[0, c].backward()
        w = self.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1))[0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), out.softmax(1)[0].detach().cpu().numpy(), c
