# vkist-facerec
VKIST Face Recognition System

## Download weights

Download from Google Drive (faster) or GitHub Releases:

- Google Drive: https://drive.google.com/drive/folders/1Nb4V75i_BX00RT0HQNAIcgsKmYLQpn9p?usp=sharing
- GitHub Releases: https://github.com/itvkist/vkist-facerec/releases/tag/v1.0.0

Extract each archive into its respective folder:

```bash
unzip weights-backend.zip -d backend/
unzip weights-frontend.zip -d frontend/
```

---

## File structure

```
vkist-facerec/
|
|-- backend/
|    |-- app/
|    |    |-- accessories_classification/
|    |    |    |-- shuffle0_0_epoch_47.pth          # Accessories classifier weights
|    |    |
|    |    |-- arcface/
|    |    |    |-- ms1mv3_arcface_r50_fp16/
|    |    |    |    |-- backbone_ir50_ms1m_epoch120.pth  # ArcFace backbone weights
|    |    |    |-- backbone.py                      # ArcFace model definition
|    |    |
|    |    |-- vision/
|    |    |    |-- detect_RFB_640/
|    |    |    |    |-- version-RFB-640.pth         # Face detector weights
|    |    |    |    |-- voc-model-labels.txt        # Face detector labels
|    |    |    |-- <model source files>             # SSD/RFB detector source
|    |    |
|    |    |-- __init__.py
|    |
|    |-- deep3d/
|    |    |-- BFM/                                  # Basel Face Model data files
|    |    |-- checkpoints/                          # Deep3D model weights
|    |    |-- models/                               # 3D reconstruction model definitions
|    |    |-- options/                              # Training/test option parsers
|    |    |-- util/                                 # Preprocessing and visualization utilities
|    |    |-- masked-face.jpg                       # Test input for test_deep3d.py
|    |    |-- unmasked-face.jpg                     # Test output from test_deep3d.py
|    |
|    |-- face_dream/
|    |    |-- dream.py                              # DREAM pose-invariant embedding correction
|    |    |-- checkpoint_512.pth                    # DREAM model weights
|    |
|    |-- static/                                    # CSS, JS, and image assets
|    |-- templates/                                 # Jinja2 HTML templates
|    |-- images/                                    # Captured face images (runtime)
|    |-- indexes/                                   # hnswlib index files (runtime)
|    |-- instance/                                  # SQLite database (runtime)
|    |
|    |-- app.py                                     # Main backend server (aiohttp, port 5002)
|    |-- create_app.py                              # Database models and app factory
|    |-- test_deep3d.py                             # Test the Deep3D mask-removal pipeline
|    |-- export_data.py                             # Export attendance records to Excel by month
|    |-- requirements.txt
|
|-- frontend/
|    |-- utils/
|    |    |-- service/
|    |    |    |-- weights/                         # TFLite model weights
|    |    |    |-- TFLiteFaceDetector.py            # Lightweight face detector (TFLite)
|    |    |    |-- TFLiteFaceAlignment.py           # Face alignment model (TFLite)
|    |    |-- functions.py                          # Shared utility functions
|    |
|    |-- static/                                    # CSS, JS, and image assets
|    |-- templates/
|    |    |-- index.html                            # Main attendance monitoring page
|    |
|    |-- app.py                                     # Frontend streaming server (Flask, port 5000)
|    |-- requirements.txt
```

---

## Backend

### Setup

```bash
cd backend
conda create -n vkist-facerec-backend python=3.10
conda activate vkist-facerec-backend
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

> nvdiffrast requires Microsoft Visual C++ Build Tools on Windows.

### Verify Deep3D

Before running the backend, verify that the mask-removal pipeline works correctly:

```bash
python test_deep3d.py
```

This reads `deep3d/masked-face.jpg` and writes the result to `deep3d/unmasked-face.jpg`.

### Run

```bash
python app.py
```

Access the backend at: http://localhost:5002

> To use the system, open the backend, create a user account on the registration page, and note down your secret key. Authentication is required to access all private pages.

---

## Frontend

### Setup

```bash
cd frontend
conda create -n vkist-facerec-frontend python=3.10
conda activate vkist-facerec-frontend
pip install -r requirements.txt
```

### Configure

Open `frontend/app.py` and replace the `SECRET_KEY` value with the secret key from your backend account. Add any RTSP camera URLs to `RTSP_URLS` if needed — webcams are detected automatically.

### Run

```bash
python app.py
```

Access the frontend at: http://localhost:5000

---

## Export data

To export monthly attendance records to an Excel spreadsheet:

```bash
cd backend
pip install pandas openpyxl
python export_data.py
```
