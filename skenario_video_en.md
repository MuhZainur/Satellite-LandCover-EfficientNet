# Video Presentation Scenario (English): Land Cover Classification from Sentinel-2 Satellite Imagery

> **Voice:** Natural Conversational English Narration (ElevenLabs / Azure Neural Speech)
> **Total Duration:** 142.84s (~2.38 min, 4286 frames @ 30 FPS)
> **Format:** 1920x1080 Full HD, Remotion MP4

---

### S01: Land Cover Classification from Sentinel-2 Imagery
- **Duration:** 13.20s audio + 0.5s pause = 13.70s (411 frames)
- **On-Screen Title:** `Land Cover Classification from Sentinel-2 Imagery`
- **On-Screen Subtitle:** `Mapping Earth terrain categories using European Space Agency satellite data`
- **Narration (Spoken):**
  > "Seen from orbit, landscapes resemble patches of geometric patterns. In this project, a deep learning model learns to identify land cover categories from Sentinel-2 satellite image tiles."

### S02: EuroSAT: 27,000 Tiles, 10 Classes, 64x64 Pixels
- **Duration:** 22.32s audio + 0.5s pause = 22.82s (685 frames)
- **On-Screen Title:** `EuroSAT: 27,000 Tiles, 10 Classes, 64x64 Pixels`
- **On-Screen Subtitle:** `Forests, rivers, highways, residential, industrial, and agricultural crop classes`
- **Narration (Spoken):**
  > "The data comes from EuroSAT, captured by the European Space Agency's Sentinel-2 satellite. It contains twenty-seven thousand tiles at sixty-four by sixty-four pixel resolution across ten land cover classes, including forests, rivers, highways, residential areas, industrial complexes, and various agricultural crops."

### S03: 1,500 Balanced Tiles: 1,120 Train, 220 Validation, 160 Test
- **Duration:** 19.70s audio + 0.5s pause = 20.20s (606 frames)
- **On-Screen Title:** `1,500 Balanced Tiles: 1,120 Train, 220 Validation, 160 Test`
- **On-Screen Subtitle:** `EfficientNet-B0 fine-tuned from ImageNet representations (150 tiles per class)`
- **Narration (Spoken):**
  > "The architecture is EfficientNet-B0, pre-trained on ImageNet and fine-tuned for ten land cover categories. I sampled one hundred fifty balanced tiles per class, creating a split of one thousand one hundred twenty training tiles, two hundred twenty validation tiles, and one hundred sixty test tiles."

### S04: Grad-CAM: Highlighting Influential Terrain Features
- **Duration:** 19.85s audio + 0.5s pause = 20.35s (610 frames)
- **On-Screen Title:** `Grad-CAM: Highlighting Influential Terrain Features`
- **On-Screen Subtitle:** `Convolutional gradient activation heatmaps provide spatial decision transparency`
- **Narration (Spoken):**
  > "To ensure the model is not acting as a complete black box, I integrated Grad-CAM. By calculating gradients from the final convolutional layer, it generates a spatial activation heatmap over the satellite tile. The highlighted red zones show exactly which terrain features drove the model's classification."

### S05: Test Accuracy: 91.9% (147 out of 160 Correct)
- **Duration:** 25.15s audio + 0.5s pause = 25.65s (770 frames)
- **On-Screen Title:** `Test Accuracy: 91.9% (147 out of 160 Correct)`
- **On-Screen Subtitle:** `100% precision on industrial, permanent crop, and water; 2 confuse grass with crop`
- **Narration (Spoken):**
  > "On the one hundred sixty test tiles, the model classified one hundred forty-seven correctly, achieving an overall test accuracy of ninety-one point nine percent. Industrial sites, permanent crops, and bodies of water were predicted with one hundred percent precision. The most common confusion occurred when herbaceous vegetation was predicted as permanent crop, accounting for two misclassified tiles."

### S06: 36 Random Tiles From Dataset | Ecological Green Index: 55.6%
- **Duration:** 22.63s audio + 0.5s pause = 23.13s (694 frames)
- **On-Screen Title:** `36 Random Tiles From Dataset | Ecological Green Index: 55.6%`
- **On-Screen Subtitle:** `Synthesized 6x6 illustrative grid (Not a real continuous geographical survey map)`
- **Narration (Spoken):**
  > "To demonstrate a regional monitoring dashboard, I assembled thirty-six random tiles from the dataset into a six-by-six grid. This is not a contiguous geographical map, but an illustrative dashboard sample. Out of the thirty-six tiles, twenty belong to vegetation classes, yielding an Ecological Green Index of fifty-five point six percent."

### S07: Next Steps: Temporal Multi-Year Deforestation Detection
- **Duration:** 16.49s audio + 0.5s pause = 16.99s (510 frames)
- **On-Screen Title:** `Next Steps: Temporal Multi-Year Deforestation Detection`
- **On-Screen Subtitle:** `Expand beyond European topography to diverse tropical and arid biomes`
- **Narration (Spoken):**
  > "Current limitations include that temporal change detection across seasons has not yet been implemented, and the dataset is sampled exclusively from European topography. Expanding to diverse global biomes is the next engineering milestone. Thank you for watching."

