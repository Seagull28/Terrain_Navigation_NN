# Autonomous Terrain Relative Navigation (TRN) System

## Overview

This project implements an Autonomous Terrain Relative Navigation (TRN) and Safe Landing Analysis System using YOLOv8-based crater detection, spatial reasoning, landing suitability analysis, and pseudo-3D terrain reconstruction.

The system detects craters from planetary surface imagery, estimates crater geometry and depth, evaluates safe landing regions, and reconstructs a synthetic 3D terrain surface for autonomous landing analysis.

---

# Features

- YOLOv8-based crater detection
- Crater depth estimation
- Spatial reasoning between craters
- Crater distance analysis
- Safe landing site selection
- Landing suitability heatmap generation
- Pseudo-3D terrain reconstruction
- Automatic output folder generation with timestamps
- Research-oriented modular architecture

---

# System Pipeline

```text
Input Terrain Image
        ↓
YOLOv8 Crater Detection
        ↓
Crater Geometry Extraction
        ↓
Depth Estimation
        ↓
Spatial Reasoning
        ↓
Landing Suitability Analysis
        ↓
Best Landing Point Selection
        ↓
3D Terrain Reconstruction
```

---

# Project Structure

```text
TRN/
│
├── data/
│   ├── TRN/
│   └── reference_preprocessed/
│
├── outputs/
│   ├── sample/
│   │    ├── localization.png
│   │    ├── distances.png
│   │    ├── heatmap.png
│   │    └── terrain_3d.png
│   │
│   └── 2026-05-16_14-40-00/
│        ├── localization.png
│        ├── distances.png
│        ├── heatmap.png
│        └── terrain_3d.png
│
├── src/
│   ├── Crater.py
│   ├── NeuralNetwork.py
│   ├── Preprocessor.py
│   ├── TerrainNavigator.py
│   └── LandingSystem.py
│
├── best.pt
├── crater.yaml
├── Main.py
├── train.py
└── README.md
```

---

# Core Technologies

- Python
- YOLOv8
- Ultralytics
- NumPy
- Pillow (PIL)
- Matplotlib

---

# Dataset

Dataset used:

Martian & Lunar Crater Detection Dataset (Kaggle)

Dataset Link:
https://www.kaggle.com/datasets/lincolnzh/martianlunar-crater-detection-dataset

Dataset Configuration:

```yaml
train: craters/train/images
val: craters/valid/images

nc: 1
names: ["crater"]
```

---

# Model Details

## Detection Model

- Model: YOLOv8
- Task: Crater Detection
- Output:
  - Bounding boxes
  - Confidence scores
  - Crater center points

---

# Depth Estimation

Crater depth is estimated empirically using:

```python
Depth = 0.15 × Diameter
```

This produces approximate crater depressions for terrain reconstruction.

---

# Landing Site Selection Logic

The landing system evaluates:

- Minimum crater clearance
- Average global crater clearance
- Local crater density

Landing score:

```python
score = (
    3.0 * min_clearance
    + 1.5 * avg_clearance
    - 4.0 * density
)
```

The highest scoring region is selected as the safest landing point.

---

# Output Visualizations

## 1. Crater Localization

Detected craters with estimated depth labels.

```markdown
![Localization](outputs/localization.png)
```

---

## 2. Crater Distance Analysis

Distances between detected craters.

```markdown
![Distances](outputs/distances.png)
```

---

## 3. Landing Suitability Heatmap

Heatmap showing safe and unsafe landing regions.

- Red → safer regions
- Blue → hazardous crater regions
- Black X → optimal landing point

```markdown
![Heatmap](outputs/heatmap.png)
```

---

## 4. 3D Terrain Reconstruction

Pseudo-3D terrain generated using crater depth estimation.

- Blue depressions → deeper craters
- Elevated terrain → safer landing regions
- Red X → selected landing site

```markdown
![3D Terrain](outputs/terrain_3d.png)
```

---

# Running the Project

## Install dependencies

```bash
pip install ultralytics numpy pillow matplotlib
```

---

## Run the system

```bash
py Main.py
```

---

# Output Structure

Each run automatically creates a timestamped output folder.

Example:

```text
outputs/
   └── 2026-05-16_14-40-00/
         ├── localization.png
         ├── distances.png
         ├── heatmap.png
         └── terrain_3d.png
```

---
# Input Image
<br> <img width="512" height="512" alt="Scene1" src="https://github.com/user-attachments/assets/461407a6-8112-401d-90ab-29d4ad5efe39" /><br>


# Output Images
Localization.png
<br> <img width="512" height="512" alt="localization" src="https://github.com/user-attachments/assets/c20803c8-8541-4550-ae5a-32b74b050626" /><br>
Distances.png
<br> <img width="512" height="512" alt="distances" src="https://github.com/user-attachments/assets/7adb6b97-7b21-466c-ba85-c2b9643ae7d8" /><br>
Heatmap.png
<br> <img width="512" height="512" alt="heatmap" src="https://github.com/user-attachments/assets/b56e6fc4-5782-412e-bf78-30a6842d8e06" /><br>
terrain_3d.png
<br> <img width="512" height="512" alt="terrain_3d" src="https://github.com/user-attachments/assets/a6fddb3c-673a-4083-a8dc-8aeda6f230b8" /><br>


# Research Contributions

- Terrain-relative navigation using visual perception
- Autonomous safe landing analysis
- Spatial crater reasoning
- Terrain reconstruction from detected crater geometry
- Research-oriented modular navigation pipeline

---

# Future Improvements

## Phase 1
- Multi-class terrain detection
- Rock and obstacle detection

## Phase 2
- Stereo depth estimation
- Real elevation mapping

## Phase 3
- SLAM integration
- Multi-frame terrain tracking

## Phase 4
- UAV autonomous path planning
- Real-time navigation support

---

# Applications

- Planetary landing systems
- Mars/Lunar terrain analysis
- UAV landing zone analysis
- Autonomous navigation research
- Aerospace AI systems

---

# License

This project is intended for academic and research purposes.
