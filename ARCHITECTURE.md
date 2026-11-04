# Arsitektur

```mermaid
flowchart LR
    subgraph Train ["train.py"]
        A[EuroSAT 27k, 64x64] --> B["Split 80/10/10<br/>seed tetap"]
        B --> C["Resize 128 + flip<br/>(hanya train)"]
        C --> D["EfficientNet-B0 pre-trained<br/>head baru 10 kelas"]
        D --> E[Cross-entropy + Adam]
        E --> F[Bobot terbaik di val]
        F --> G["Test: classification report<br/>+ confusion matrix"]
    end
    F --> W[(model.pt)]
    subgraph Demo ["demo.py"]
        X[Citra satelit] --> Y[EfficientNet-B0]
        W --> Y
        Y --> P[Probabilitas kelas]
        Y --> H["Grad-CAM<br/>features[-1]"]
        P --> Z[Plot: citra + overlay heatmap]
        H --> Z
    end
```

| File | Peran |
|---|---|
| `src/dataset_loader.py` | unduh EuroSAT, split, transform |
| `src/classifier.py` | model, training loop, evaluasi, Grad-CAM |
| `train.py` | latih dan laporan metrik |
| `demo.py` | prediksi dan visualisasi Grad-CAM |
