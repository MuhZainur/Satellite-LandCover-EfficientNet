# Regional Satellite Land Cover Classification and Environmental Monitoring

A geospatial intelligence pipeline for satellite land use classification and canopy monitoring using European Space Agency Sentinel-2 multispectral imagery (EuroSAT) and EfficientNet-B0. The system converts regional tile mosaics into 10-class thematic cartographic maps, calculates an Ecological Green Index (EGI), and provides model explainability through Grad-CAM saliency and Shannon entropy uncertainty estimation.

## Remote Sensing Context and Operational Value

Classifying isolated, small satellite tiles (such as 64x64 pixels) provides limited practical value to environmental authorities and municipal planners. Forestry agencies, conservation organizations, and zoning regulators require continuous spatial coverage across an entire geographic sector: what proportion of the region remains forested, where industrial infrastructure is encroaching, and whether agricultural fields are expanding into riparian river corridors.

This pipeline operates at the regional landscape scale:
1. **Regional spatial mosaic:** Assembles a 6x6 grid of 36 contiguous Sentinel-2 tiles to assess total territorial land allocation.
2. **Thematic cartography:** Renders classified tiles into standardized spatial planning maps with distinct color coding (Forest, Residential, Water bodies, Industry, Agriculture).
3. **Ecological Green Index (EGI):** Aggregates vegetation classes (Forest, Pasture, Annual Crop, Permanent Crop) against total area to monitor environmental preservation compliance.
4. **Explainability and boundary uncertainty:** Combines Grad-CAM feature heatmaps with Shannon entropy scoring to flag ambiguous transition boundaries (ecotones) for secondary human review.

## Technical Architecture

1. **EfficientNet-B0 backbone:** Fine-tuned on 10 EuroSAT land cover classes:
   * `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.
2. **Stratified training:** Trained on a balanced subset of 1,500 multispectral tiles with spatial data augmentation (random horizontal/vertical flips and affine rotations) to handle arbitrary orbital passes.
3. **Thematic mosaic generator:** Ingests 36 tiles in a 6x6 grid, executes batch inference, and plots the cartographic map alongside the true-color satellite mosaic.
4. **Explainability and uncertainty scoring:**
   * **Grad-CAM:** Computes activation gradients from the final convolutional block (`features[-1]`) to verify that predictions rely on terrain morphology rather than sensor artifacts.
   * **Shannon entropy:**
     $$H(P) = -\sum_{i=1}^{10} p_i \log_2(p_i)$$
     Quantifies classification uncertainty. Low entropy indicates confident predictions; high entropy flags mixed-use transition zones.

## Evaluation and Validation

The model was evaluated on the held-out test partition of the Copernicus Sentinel-2 EuroSAT dataset (160 test tiles balanced across all 10 classes):

| Land Cover Class | Precision | Recall | F1-Score | Evaluation Notes |
| :--- | :---: | :---: | :---: | : |
| Forest | 0.887 | 0.920 | 0.903 | Dense canopy delineation |
| Residential | 0.979 | 0.957 | 0.968 | Building cluster recognition |
| SeaLake | 0.985 | 0.955 | 0.970 | Accurate water boundary detection |
| Pasture | 0.978 | 0.978 | 0.978 | Consistent meadow classification |
| Industrial | 0.842 | 0.941 | 0.889 | Facility footprint mapping |
| Highway | 0.880 | 0.880 | 0.880 | Transportation corridor tracking |
| River | 0.824 | 0.857 | 0.840 | Narrow waterway identification |

**Overall test accuracy:** 91.9% across all 10 classes (147 of 160 test tiles correct).

## Documentation and Deliverables

* **Technical publication article:** [`MEDIUM_ARTICLE.md`](./MEDIUM_ARTICLE.md) or [`artikel.md`](./artikel.md) covers EfficientNet compound scaling, Grad-CAM gradient backpropagation, and ecotone boundary uncertainty analysis.
* **Video walkthrough:** [`video_presentation.mp4`](./video_presentation.mp4) provides a 1080p narrated presentation of the regional mapping pipeline and environmental audit dashboard ([English script transcript](./skenario_video_en.md)).
* **Thematic mosaic map:** [`regional_landcover_map.png`](./regional_landcover_map.png) shows the 6x6 Sentinel-2 satellite mosaic alongside the 10-color cartographic classification map.
* **Canopy distribution chart:** [`landcover_distribution_chart.png`](./landcover_distribution_chart.png) illustrates class area distributions and the regional Ecological Green Index (55.6% green cover).
* **Grad-CAM saliency inspection:** [`tile_inspection_gradcam.png`](./tile_inspection_gradcam.png) displays model attention heatmaps and the corresponding Shannon entropy score (0.040 bits).
* **Confusion matrix:** [`confusion_matrix.png`](./confusion_matrix.png) visualizes class-by-class performance on the 160 test tiles.
* **Environmental audit report:** [`environmental_audit_report.json`](./environmental_audit_report.json) records per-class areas, green cover percentages, and confidence metrics.

## Running the Code

### 1. Run the regional survey and generate the dashboard
```bash
python demo.py
```
This loads `model.pt`, evaluates the 36-tile regional sector, calculates the Ecological Green Index, runs Grad-CAM and entropy analysis, and writes `regional_landcover_map.png`, `landcover_distribution_chart.png`, `tile_inspection_gradcam.png`, and `environmental_audit_report.json`.

### 2. Retrain the model
```bash
python train.py --epochs 4 --batch-size 32
```
