# PathoFocus - WSI Diagnostic System

A web-based diagnostic interface for Whole Slide Image (WSI) analysis with AI-powered predictions for hepatocellular carcinoma (HCC). PathoFocus provides tumor grading, fibrosis staging, and microvascular invasion (MVI) detection with case-based retrieval capabilities.

## Features

### AI Diagnosis
- **Tumor Grading**: G1 (Well), G2 (Moderate), G3 (Poor) differentiation classification
- **Fibrosis Staging**: F0-F1 (None/Mild), F2-F3 (Moderate), F4 (Cirrhosis) staging
- **MVI Detection**: Microvascular invasion presence/absence prediction
- Confidence scores and probability distributions for all predictions

### Slide Viewer
- WSI thumbnail display with zoom/pan controls (Ctrl+scroll, drag)
- Fullscreen viewing mode
- Drag-and-drop file loading
- Support for .svs, .ndpi, .tiff, .png, .jpg formats

### Case Retrieval
- Embedding-based similar case retrieval
- Multiple retrieval strategies: Balanced, Morphology, Diagnosis
- Side-by-side slide comparison with diagnostic matching indicators

### Database Browser
- Label distribution visualization (pie charts)
- Searchable slide database with filtering by Grade, Fibrosis, and MVI
- Click-to-load slides from database

## Project Structure

```
pathofocus/
├── pathofocus_server.py      # FastAPI backend server
├── pathofocus_frontend.html  # Single-file web frontend
├── thumbnail_cache/          # Cached WSI thumbnails (auto-created)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### 1. Clone and Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. OpenSlide (for WSI thumbnail generation)

**Ubuntu/Debian:**
```bash
sudo apt-get install openslide-tools
pip install openslide-python
```

**macOS:**
```bash
brew install openslide
pip install openslide-python
```

**Windows:**
Download OpenSlide binaries from https://openslide.org/download/ and add to PATH.

## Usage

### Demo Mode (No data required)

```bash
python pathofocus_server.py --demo
```

This starts the server with mock data for UI testing and development.

### Production Mode

```bash
python pathofocus_server.py --production
```

Before running production mode, update the paths in `ProductionConfig` class:

```python
class ProductionConfig:
    def __init__(self):
        # Update these paths to match your environment
        self.annotation_root = Path("../hcc/data/preprocessed/wsi")  # patchinfo.json files
        self.feature_cache_path = Path("../hcc/data/preprocessed/wsi/tmp/curriculum/patch_feat_1000")  # .pt features
        self.wsi_roots = {
            "13": Path("../../WSI_HCC/WSI_HCC_2013"),
            "15": Path("../../WSI_HCC/WSI_HCC_2015"),
            "16": Path("../../WSI_HCC/WSI_HCC_2016"),
        }
        self.model_checkpoint = "../hcc/lightning_logs/three_stage_curriculum_bs8_lr5e-05_new/best_three_stage_epoch32_val_mvi_auc0.994.ckpt"
```

### Server Options

```bash
python pathofocus_server.py --demo --port 8000 --host 0.0.0.0
python pathofocus_server.py --production --port 8080
```

### Accessing the Frontend

1. Start the server (demo or production mode)
2. Open `pathofocus_frontend.html` in a web browser
3. Or serve it via any static file server

The frontend connects to `http://localhost:8000` by default. To change, modify `API_BASE` in the HTML file.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and mode |
| `/api/health` | GET | Health check |
| `/api/analyze` | POST | Analyze a WSI slide |
| `/api/retrieve` | POST | Retrieve similar cases |
| `/api/database/stats` | GET | Get label distribution statistics |
| `/api/database/slides` | GET | List all slides in database |
| `/api/thumbnail/{slidename}` | GET | Get WSI thumbnail image |
| `/api/region/{slidename}` | GET | Get specific WSI region |

### Example API Calls

```bash
# Analyze a slide
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"filepath": "HCC2023-001.ndpi", "use_cache": true}'

# Retrieve similar cases
curl -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"slidename": "HCC2023-001", "top_k": 5, "strategy": "balanced"}'

# Get database stats
curl http://localhost:8000/api/database/stats

# Get slide thumbnail
curl http://localhost:8000/api/thumbnail/HCC2023-001?size=1024
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `+` | Zoom in |
| `-` | Zoom out |
| `0` | Fit to view |
| `F` | Toggle fullscreen |
| `Esc` | Close modal/exit fullscreen |
| `Ctrl+Scroll` | Zoom in/out |

## System Requirements

- Python 3.8+
- CUDA-capable GPU (for production mode with model inference)
- 8GB+ RAM recommended
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Dependencies

### Core
- FastAPI - Web framework
- Uvicorn - ASGI server
- NumPy - Numerical computing
- Pydantic - Data validation

### Production Mode (additional)
- PyTorch - Deep learning
- OpenSlide - WSI file handling
- timm - Vision models
- Pillow - Image processing

## Troubleshooting

### "OpenSlide not available" warning
Install OpenSlide system library and openslide-python package. Thumbnails will show as placeholders without it.

### CORS errors in browser
The server allows all origins by default. If issues persist, check that the frontend `API_BASE` matches your server URL.

### Slides not loading in production
1. Verify WSI file paths in `ProductionConfig.wsi_roots`
2. Check that `.pt` feature files exist in `feature_cache_path`
3. Ensure model checkpoint path is correct

### Database shows 0 slides
- In demo mode: Mock database auto-generates slides
- In production: Check that feature cache directory contains `.pt` files

## License

[Your License Here]

## Citation

If you use PathoFocus in your research, please cite:

```bibtex
@software{pathofocus2026,
  title={PathoFocus: WSI Diagnostic System for HCC},
  year={2026}
}
```
