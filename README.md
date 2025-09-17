# Optical Flow Cluster Analyzer (OFCA)

A desktop GUI for **human movement analysis** that combines **YOLOv8 person detection**, **Lucas–Kanade optical flow**, and **unsupervised clustering** to characterize motion patterns and movement quality (e.g., *fluidity*, *consistency*, origin of motion). Built with **PyQt6** and **OpenCV**.

> This README documents the refactored app found in: `Human movement analysis/ofca_project/`  
> A legacy single-file prototype also exists at: `Human movement analysis/Human origin movement analysis.py`.

---

## ✨ Key Features

- **Video/Webcam ingestion** with frame-by-frame processing.
- **Person detection** via **Ultralytics YOLOv8** (`HumanDetector`).
- **Optical flow** via **Lucas–Kanade** (`OpticalFlowProcessor`).
- **Clustering** of flow vectors with **K-Means / DBSCAN / Agglomerative / OPTICS** (see `analysis/worker.py`).
- **Quality metrics**:
  - Hopkins statistic (clusterability proxy)
  - Magnitude/consistency summaries
  - Silhouette, Davies–Bouldin, Calinski–Harabasz indices
- **Interactive GUI** (PyQt6):
  - Live video view, dashboards, and result tabs (`ui/tabs.py`)
  - Validation dialog to compare automatic vs ground-truth labels
  - Export of reports/CSV (see `io/reporting.py`)
- **Visualization helpers**: motion trails, heatmaps, cluster centroids (`utils/visuals.py`).
- **Threaded analysis worker** to keep the UI responsive (`analysis/worker.py`).

---

## 📦 Project Structure (refactored)

```
ofca_project/
├─ main.py                  # Entry point (PyQt6 app)
├─ requirements.txt         # Minimal runtime deps for ofca_project
├─ ofca/
│  ├─ app.py                # QMainWindow: OpticalFlowClusterAnalyzer
│  ├─ analysis/
│  │  ├─ movement_quality.py  # MovementQualityAnalyzer (RF, stats, thresholds)
│  │  ├─ fluid_motion.py      # FluidMotionAnalyzer (Hopkins-based fluidity)
│  │  └─ worker.py            # QThread for clustering + metrics
│  ├─ vision/
│  │  ├─ detector.py          # YOLOv8 person detection
│  │  └─ optical_flow.py      # Lucas–Kanade optical flow pipeline
│  ├─ ui/
│  │  ├─ dialogs.py           # Validation results & helpers
│  │  ├─ tabs.py              # Video / Dashboard / Results tabs
│  │  └─ style.py             # Fonts and style helpers
│  ├─ utils/
│  │  ├─ metrics.py           # Hopkins, indices, helpers
│  │  ├─ visuals.py           # Trails, heatmaps, centroids
│  │  └─ tables.py            # QTable helpers
│  └─ _legacy/
│     └─ Human origin movement analyzer.py  # backup of the old monolith
└─ (optional) yolov8n.pt     # If present, used for detection; otherwise auto-download
```

The repository root also contains:
- `Human movement analysis/requirements.txt` — a **larger** dependency set used during experimentation (includes `ultralytics`).
- `Human movement analysis/Human origin movement analysis.py` — the **legacy single-file UI**.

---

## 🛠️ Requirements

- Python **3.10+** recommended
- OS: Windows / macOS / Linux
- GPU optional (YOLOv8 will use CUDA if available)
- **Important**: The refactored app uses **PyQt6**. Make sure your environment matches.

### Minimal install (recommended for the refactored app)

```bash
cd "Human movement analysis/ofca_project"
python -m venv .venv
# On macOS/Linux
. .venv/bin/activate
# On Windows
# .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install ultralytics  # YOLOv8 (if not already in requirements)
```

> If you get import errors for `PyQt6` but your `requirements.txt` pins **PyQt5**, update it to `PyQt6==6.6.*` (or compatible) because the code imports from `PyQt6`.

---

## ▶️ Run

```bash
cd "Human origin movement analysis.py"
```

The GUI will open:
1. Choose a **video file** (or enable **webcam**).
2. Select a **detection model size** (n/s/m/l) if applicable.
3. Start processing to see optical flow vectors, clusters, and quality metrics update live.

---

## 🔬 How It Works (Pipeline)

1. **Detection** – `vision/detector.py`
   - Loads a YOLOv8 model (e.g., `yolov8n.pt`).
   - Detects person bounding boxes per frame.

2. **Tracking & Flow** – `vision/optical_flow.py`
   - Samples feature points (Shi–Tomasi).
   - Computes point trajectories with Lucas–Kanade optical flow pyramids.
   - Emits per-frame **flow_points** (dx, dy, positions) and derived metrics.

3. **Clustering & Metrics** – `analysis/worker.py`
   - Stacks flow vectors across a window; runs KMeans/DBSCAN/Agglomerative/OPTICS.
   - Computes **silhouette**, **DB**, **CH** indices, and **Hopkins** (via `utils/metrics.py`).

4. **Movement Quality** – `analysis/movement_quality.py` & `analysis/fluid_motion.py`
   - Applies thresholds for **fluidity**, **magnitude**, **consistency**.
   - Optionally trains a **RandomForest** on labeled sessions (for auto-labeling).
   - Tracks motion origins and proof frames.

5. **UI** – `ofca/app.py` + `ui/tabs.py`
   - Video tab (preview), Dashboard tab (metrics), Result tab (clusters & validation).
   - Export reports and compare with **ground-truth CSV** (see `io/reporting.py`, `ui/dialogs.py`).

---

## 📁 Data & File Formats

- **Input**: Any video readable by OpenCV (mp4, avi, mov) or a Webcam stream.
- **Ground truth** (optional): A CSV/JSON file mapping **frame ranges → labels**.
- **Exports**: CSV summaries (metrics per window), optional PNGs of snapshots/heatmaps.

---

## ⚙️ Configuration Tips

- **Model weights**: If `yolov8n.pt` is in the project folder, it will be used; otherwise, Ultralytics will download weights.
- **Performance**: Environment variable `OMP_NUM_THREADS=1` is set to avoid KMeans warnings and oversubscription.
- **Sampling**: Adjust max corners, LK window size, and pyramid levels in `vision/optical_flow.py`.
- **Clustering**: For noisy scenes, try **DBSCAN** (tune `eps`, `min_samples`). For compact motions, **KMeans** works well.

---

## 🧪 Reproducible Experiments

- Use a fixed random seed in metrics/clustering for reproducibility when benchmarking.
- Save **per-frame** features to CSV and compute aggregate metrics offline for ablation.

---

## 🚧 Known Issues / To‑Dos

- **PyQt version**: Some files import `PyQt6` while an older `requirements.txt` pins `PyQt5`.  
  **Action**: Standardize on **PyQt6** in `ofca_project/requirements.txt`.
- **Large assets**: Model weights (`*.pt`) and videos **should not** be committed to git.  
  **Action**: Add them to `.gitignore` or provide download instructions.
- **Docs**: Add screenshots/GIFs of the GUI to improve the README.
- **Tests**: Add unit tests for `utils.metrics` and `vision` modules.

---

## 🧰 Development

```bash
# Lint
pip install ruff black
ruff check ofca
black ofca

# Run in dev
python main.py
```

---

## 🤝 Contributing

PRs are welcome! If you add a new clustering method or metric:
- Place algorithms in `ofca/analysis/` or `ofca/utils/`.
- Expose controls in `ui/tabs.py`.
- Update this README and the in‑app help.

---

---

## ✍️ Citation

If you use **OFCA** in academic work:

```
@software{ofca_2025,
  author  = {{Abdullaj Al Foysal}},
  title   = {{Optical Flow Cluster Analyzer (OFCA)}},
  year    = {{2025}},
  url     = {{https://github.com/<Foysal440>/Origin-of-Human-Movement}}
}
```

---

##  Acknowledgments

- Built with **OpenCV**, **PyQt6**, **scikit‑learn**, and **Ultralytics YOLOv8**.
- Thanks to collaborators and reviewers for feedback on movement quality metrics and validation UI.

---
