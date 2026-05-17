# Autonomous Terrain Relative Navigation (TRN) System

> End-to-end computer vision pipeline for autonomous spacecraft descent — detects lunar/planetary surface craters via a fine-tuned YOLO model and dynamically evaluates safe landing zones through multi-factor topographic scoring.

---

## Pipeline Architecture

```
[Raw Descent Frame]
        │
        ▼
[Neural Detector — YOLO]
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
[Euclidean Distance               [Landing Hazard Assessment]
 Triangulation]                               │
        │                                      ▼
        ▼                         [Coarse-to-Fine Scoring Search]
[Constellation                                │
 Tracking Graph]                              ▼
                              [Optimal Landing Point Selection]
```

1. **Neural Hazard Recognition** — `NeuralDetector` extracts crater boundaries, centroids $(c_x, c_y)$, diameters, and confidence scores via YOLO inference with NMS post-filtering.
2. **Depth Estimation** — Translates 2D bounding geometry into physical hazard depth estimates: $d = 0.15 \cdot \text{diameter} + 0.02 \cdot \sqrt{\text{diameter}}$
3. **Circular Density Matrix** — Vectorized coordinate meshes build spatial risk fields using true circular crater footprints.
4. **Adaptive Suitability Scoring** — Dual-pass coarse→fine grid search scores every candidate point across five normalised terms (clearance, centrality, density, smoothness, slope) within a 40 px safety buffer.
5. **Telemetry Visualisation** — Produces 3D terrain reconstructions, suitability heatmaps, density maps, and triangulation graphs.

---

## Visual Outputs

| File | Description |
|---|---|
| `terrain_3d.png` | Gaussian depression surface with 3D location pin at landing point |
| `heatmap.png` | Safety score field (red = hazardous → green = optimal) with annotated landing marker |
| `density_map.png` | Circular crater footprint overlay showing hazard cluster density |
| `distances.png` | Euclidean linkage graph between top-5 confidence crater centroids |
| `localization.png` | Raw descent image with detected craters and landing point marked |
| `report.txt` | Structured metrics: landing point, score, confidence statistics |

---

## Repository Structure

```
Terrain_Navigation_NN/
├── Main.py                     # Pipeline entry point
├── requirements.txt            # Dependency manifest
├── best.pt                     # Fine-tuned YOLO weights
├── data/
│   └── TRN/
│       ├── ReferenceMap.ppm    # Reference terrain map
│       └── Scene4.ppm          # Sample descent image
└── src/
    ├── __init__.py
    ├── Crater.py               # Crater data class (centerpoint, diameter, depth, radius)
    ├── ExperimentLogger.py     # Append-mode text report writer
    ├── LandingSystem.py        # Scoring functions and coarse-to-fine optimizer
    ├── NeuralNetwork.py        # YOLO inference, NMS, depth estimation
    ├── Preprocessor.py         # Centerpoint extraction utilities
    └── TerrainNavigator.py     # Pipeline orchestrator and visualisation engine
```

---

## Installation

**Requirements:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/Seagull28/Terrain_Navigation_NN.git
cd Terrain_Navigation_NN

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python Main.py
```

Outputs are saved automatically to a timestamped folder:

```
outputs/
└── 2026-05-17_21-14-00/
    ├── report.txt
    ├── heatmap.png
    ├── terrain_3d.png
    ├── density_map.png
    ├── distances.png
    └── localization.png
```

---
# Input Image
<br> <img width="512" height="512" alt="Scene1" src="https://github.com/user-attachments/assets/461407a6-8112-401d-90ab-29d4ad5efe39" /><br>


# Output Images
Localization.png
<br><img width="512" height="512" alt="localization" src="https://github.com/user-attachments/assets/6a01e6b9-f6c1-4e1d-8db7-dd9bf060d3cf" /><br>
Distances.png
<br><img width="512" height="512" alt="distances" src="https://github.com/user-attachments/assets/8dac864c-b0b6-4962-8b90-06c98d27af76" /><br>
Heatmap.png
<br><img width="2031" height="1907" alt="heatmap" src="https://github.com/user-attachments/assets/f9057460-0ec4-44c2-bf40-44d8438908f4" /><br>
terrain_3d.png
<br><img width="3149" height="2683" alt="terrain_3d" src="https://github.com/user-attachments/assets/782f31d1-955c-4c45-b176-123fd243cd1c" /><br>
density_map.png
<br><img width="2022" height="1950" alt="density_map" src="https://github.com/user-attachments/assets/c2901ae2-6899-4753-a1d8-88a07123bc3a" /><br>


## Sample Output

```
📁 Saving outputs to: outputs/2026-05-17_21-14-00
🧠 NN detected craters: 14
🧠 NN filtered craters: 12

⚠️  DETECTION ANALYSIS
----------------------
Small Craters        : 0
Low Confidence (<0.8): 5

🎯 LANDING ANALYSIS
----------------------
Best Landing Point         : (270, 280)
Landing Score              : 2.9076
Distance from nearest rim  : 54.21 px

PERFORMANCE METRICS
----------------------
Total Craters Detected: 12
Average Confidence: 0.818
Maximum Confidence: 0.885
Minimum Confidence: 0.707
```

---

## Landing Scoring Function

The suitability score at each candidate point is computed as:

$$\text{Score} = 5.0 \cdot \hat{d}_{min} + 3.0 \cdot C_{central} - 4.0 \cdot \hat{\rho} - 2.0 \cdot \hat{s} - 2.0 \cdot \hat{g}$$

| Term | Description |
|---|---|
| $\hat{d}_{min}$ | Normalised clearance from nearest crater rim |
| $C_{central}$ | Centrality reward — peaks at image centre, 0 at corners |
| $\hat{\rho}$ | Normalised local crater density within 80 px radius |
| $\hat{s}$ | Normalised terrain smoothness (depth-weighted influence) |
| $\hat{g}$ | Normalised terrain slope (finite-difference gradient magnitude) |

All terms are scaled to $[0, 1]$ before weighting so no single factor dominates.

---

## Dependencies

```
ultralytics    # YOLO object detection framework
numpy          # Vectorised grid computations
matplotlib     # 3D rendering and heatmap visualisation
pillow         # Image drawing and annotation
```

Install via:

```bash
pip install ultralytics numpy matplotlib pillow
```

---

##  License & Citation

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. You are free to modify, distribute, and build upon this software for academic, research, or personal applications provided the original copyright notice remains intact.

### Citation
If you use this autonomous navigation pipeline, the neural crater detection framework, or the topographical safety estimation algorithms in your research or academic publications, please cite the repository as follows:

```bibtex
@misc{trn_navigation_nn_2026,
  author       = {Thanujha Yadav},
  title        = {Autonomous Terrain Relative Navigation (TRN) System using Neural Crater Detection},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{[https://github.com/Seagull28/Terrain_Navigation_NN](https://github.com/Seagull28/Terrain_Navigation_NN)}}
}


