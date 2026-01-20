# PathoFocus - WSI Diagnostic & Retrieval System

A modern, clinical-grade interface for whole slide image analysis with multi-task 
AI diagnosis (Tumor Grading, Fibrosis Staging, MVI Detection) and similar case retrieval.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Backend (Demo Mode)
```bash
python pathofocus_backend.py
# API will be available at http://localhost:8000
```

### 3. Open the Frontend
Simply open `pathofocus_frontend.html` in your browser.
The UI works in demo mode with mock data even without the backend.

## 📁 Project Structure

```
pathofocus/
├── pathofocus_backend.py       # FastAPI backend (demo mode)
├── pathofocus_production.py    # Production adapter for real WSI system
├── pathofocus_frontend.html    # Complete UI (single-file SPA)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Production Deployment

### Connect to Your WSI System

1. Edit `pathofocus_production.py` and update the paths in `ProductionConfig`:
```python
class ProductionConfig:
    def __init__(self):
        self.dataroot = Path("/your/wsi/directory")
        self.feature_cache_path = Path("/your/feature/cache")
        self.tumor_encoder_path = "/path/to/tumor_encoder"
        self.fibrosis_encoder_path = "/path/to/fibrosis_encoder"
        self.model_checkpoint = "/path/to/model.ckpt"
```

2. Replace imports in the backend to use production classes:
```python
from pathofocus_production import ProductionDiagnosticSystem, ProductionConfig

config = ProductionConfig()
system = ProductionDiagnosticSystem(config)
```

3. Run the production server:
```bash
uvicorn pathofocus_production:create_production_app --host 0.0.0.0 --port 8000
```

## 🎯 Features

### AI Diagnosis Panel
- **Tumor Grading**: G1 (Well) / G2 (Moderate) / G3 (Poor)
- **Fibrosis Staging**: F0-F1 / F2-F3 / F4
- **MVI Detection**: Absent / Present with visual alert

### Similar Case Retrieval
- **Balanced**: Combined morphological and diagnostic similarity
- **Morphology**: Pure embedding-based visual similarity
- **Diagnosis**: Prioritizes cases with matching diagnoses

### Comparison Tools
- **Split-View Comparison**: Side-by-side slide viewing
- **Sync-Lock**: Synchronized panning between slides
- **Attention Heatmap**: Visualize AI attention regions

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+H` | Toggle heatmap overlay |
| `Esc` | Close comparison modal |

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Analyze a WSI slide |
| `/api/retrieve` | POST | Retrieve similar slides |
| `/api/database/stats` | GET | Get database statistics |
| `/api/database/slides` | GET | List all indexed slides |
| `/api/compare` | POST | Compare two slides |

### Example API Call
```python
import requests

# Analyze a slide
response = requests.post("http://localhost:8000/api/analyze", json={
    "file_path": "/path/to/slide.svs",
    "use_cache": True
})
print(response.json())
```

## 🎨 Customization

### Color Scheme
Edit CSS variables in `pathofocus_frontend.html`:
```css
:root {
    --accent-primary: #6366f1;    /* Main accent */
    --accent-success: #10b981;    /* Positive indicators */
    --accent-danger: #ef4444;     /* Alerts and MVI+ */
    --bg-primary: #0f0f13;        /* Background */
}
```

### Add New Tasks
1. Add prediction card in HTML
2. Update `updateDiagnosisPanel()` in JavaScript
3. Add corresponding API fields in backend

## 📋 Requirements

- Python 3.9+
- CUDA-capable GPU (for production)
- Modern browser (Chrome, Firefox, Edge)

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

Built on top of the WSI Diagnostic System with UNI encoders and 
multi-task curriculum learning for HCC analysis.
