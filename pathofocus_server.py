"""
PathoFocus Backend - Unified Demo/Production Server
====================================================

Usage:
  Demo mode:       python pathofocus_server.py --demo
  Production mode: python pathofocus_server.py --production

Before running production mode, update the paths in ProductionConfig below.
"""

import os
import sys
import json
import asyncio
import argparse
import io
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn


# ============================================================================
# CONFIGURATION
# ============================================================================

class ProductionConfig:
    """UPDATE THESE PATHS TO MATCH YOUR ENVIRONMENT"""
    def __init__(self):
        # === REQUIRED: Update these paths ===
        # Where *_patch_info.json files are located
        self.annotation_root = Path("../hcc/data/preprocessed_wsi")

        # Where .pt feature files are stored  
        self.feature_cache_path = Path("../hcc/data/preprocessed_wsi/tmp_curriculum_patch_feat_1000")

        # Where actual WSI files (.ndpi, .svs) are stored
        # This maps year prefixes to folders
        self.wsi_roots = {
            "13": Path("../../WSI_HCC/WSI_HCC_2013"),
            "15": Path("../../WSI_HCC/WSI_HCC_2015"),
            "16": Path("../../WSI_HCC/WSI_HCC_2016"),
        }

        # Thumbnail cache directory
        self.thumbnail_cache = Path("./thumbnail_cache")
        self.thumbnail_cache.mkdir(exist_ok=True)

        # Model paths
        self.tumor_encoder_path = "../hcc/lightning_logs/tumor/128_5e-05_64/"
        self.fibrosis_encoder_path = "../hcc/lightning_logs/fibrosis/256_5e-05_128/"
        self.model_checkpoint = "../hcc/lightning_logs/three_stage_curriculum/bs8_lr5e-05_new/best_three_stage_epoch=32_val_mvi_auc=0.994.ckpt"

        # Other settings
        self.model_id = "hf-hub:MahmoodLab/uni"
        self.patch_size = 1024
        self.num_patches = 128
        self.img_size = 224
        self.device = "cuda"
        self.embedding_dim = 1024


# ============================================================================
# WSI THUMBNAIL GENERATOR
# ============================================================================

class WSIThumbnailGenerator:
    """Generate and cache thumbnails from WSI files"""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.openslide = None
        self._init_openslide()

    def _init_openslide(self):
        try:
            import openslide
            self.openslide = openslide
            print("✓ OpenSlide loaded for thumbnail generation")
        except ImportError:
            print("⚠ OpenSlide not available - thumbnails will be placeholders")
            self.openslide = None

    def get_wsi_path(self, slide_name: str) -> Optional[Path]:
        """Find the WSI file for a given slide name"""
        year_prefix = slide_name.split('-')[0]

        if year_prefix in self.config.wsi_roots:
            base_dir = self.config.wsi_roots[year_prefix]

            # Try different extensions
            for ext in ['.ndpi', '.svs', '.tiff', '.tif']:
                wsi_path = base_dir / f"{slide_name}{ext}"
                if wsi_path.exists():
                    return wsi_path

        # Try all directories if year prefix didn't match
        for base_dir in self.config.wsi_roots.values():
            if base_dir.exists():
                for ext in ['.ndpi', '.svs', '.tiff', '.tif']:
                    wsi_path = base_dir / f"{slide_name}{ext}"
                    if wsi_path.exists():
                        return wsi_path

        return None

    def get_thumbnail(self, slide_name: str, max_size: int = 1024) -> Optional[bytes]:
        """Get thumbnail as PNG bytes, using cache if available"""
        cache_path = self.config.thumbnail_cache / f"{slide_name}_{max_size}.png"

        # Return cached thumbnail if exists
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return f.read()

        # Generate thumbnail
        if self.openslide is None:
            return None

        wsi_path = self.get_wsi_path(slide_name)
        if wsi_path is None:
            print(f"  WSI not found for {slide_name}")
            return None

        try:
            slide = self.openslide.OpenSlide(str(wsi_path))

            # Get dimensions and calculate thumbnail size
            w, h = slide.dimensions
            scale = min(max_size / w, max_size / h)
            thumb_size = (int(w * scale), int(h * scale))

            # Generate thumbnail
            thumbnail = slide.get_thumbnail(thumb_size)
            slide.close()

            # Save to cache
            thumbnail.save(cache_path, 'PNG')

            # Return as bytes
            img_bytes = io.BytesIO()
            thumbnail.save(img_bytes, 'PNG')
            img_bytes.seek(0)
            return img_bytes.getvalue()

        except Exception as e:
            print(f"  Error generating thumbnail for {slide_name}: {e}")
            return None

    def get_region(self, slide_name: str, x: int, y: int, 
                   width: int = 512, height: int = 512, level: int = 0) -> Optional[bytes]:
        """Get a specific region from WSI"""
        if self.openslide is None:
            return None

        wsi_path = self.get_wsi_path(slide_name)
        if wsi_path is None:
            return None

        try:
            slide = self.openslide.OpenSlide(str(wsi_path))
            region = slide.read_region((x, y), level, (width, height)).convert('RGB')
            slide.close()

            img_bytes = io.BytesIO()
            region.save(img_bytes, 'JPEG', quality=85)
            img_bytes.seek(0)
            return img_bytes.getvalue()

        except Exception as e:
            print(f"  Error getting region for {slide_name}: {e}")
            return None


# ============================================================================
# MOCK SYSTEM (Demo Mode)
# ============================================================================

class MockSlideAnalysis:
    def __init__(self, slide_name: str):
        self.slide_name = slide_name
        self.slide_type = np.random.choice(['tumor_dominant', 'fibrosis_dominant'])
        self.source = 'validated'
        self.embedding = np.random.randn(1024).astype(np.float32)
        self.num_patches = np.random.randint(50, 200)

        self.grade = self._make_prediction(3, ['G1 (Well)', 'G2 (Moderate)', 'G3 (Poor)'])
        self.fibrosis = self._make_prediction(3, ['F0-F1 (None/Mild)', 'F2-F3 (Moderate)', 'F4 (Cirrhosis)'])
        self.mvi = self._make_prediction(2, ['Absent', 'Present'])

        self.coords = [(np.random.randint(0, 50000), np.random.randint(0, 50000)) 
                       for _ in range(min(20, self.num_patches))]
        self.attention_weights = np.random.dirichlet(np.ones(len(self.coords))).tolist()

    def _make_prediction(self, n, names):
        dist = np.random.dirichlet(np.ones(n) * 0.5).tolist()
        pred = int(np.argmax(dist))
        return {
            "distribution": dist,
            "predicted_class": pred,  # underscore
            "confidence": float(max(dist)),
            "class_names": names,      # no underscore
            "ground_truth": np.random.choice([0, 1, 2, None])[:n]
        }

    def to_dict(self):
        return {
            'slide_name': self.slide_name,
            'slide_type': self.slide_type,
            'source': self.source,
            'grade': self.grade,
            'fibrosis': self.fibrosis,
            'mvi': self.mvi,
            'num_patches': self.num_patches,
            'coords': self.coords,
            'attention_weights': self.attention_weights
        }


class MockDatabase:
    def __init__(self):
        names = [f'HCC_{y}_{str(i).zfill(3)}' for y in [2022, 2023, 2024] for i in range(1, 5)]
        self.slides = {name: MockSlideAnalysis(name) for name in names}
        self.embeddings = np.array([s.embedding for s in self.slides.values()])
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
        self.slide_names = list(self.slides.keys())

    def search(self, query_embedding, top_k=5, strategy='balanced'):
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        similarities = np.dot(self.embeddings, query_norm)
        top_idx = np.argsort(similarities)[::-1][:top_k]
        return [{
            'slide_name': self.slide_names[i],
            'similarity': float(similarities[i]),
            'analysis': self.slides[self.slide_names[i]].to_dict()
        } for i in top_idx]


# ============================================================================
# GROUND TRUTH LOADER
# ============================================================================

def load_ground_truth_from_json(annotation_root: Path, slide_name: str) -> Dict:
    """Load ground truth labels from *_patch_info.json files"""
    possible_paths = [
        annotation_root / f"{slide_name}_patch_info.json",
        annotation_root / f"{slide_name}.json",
    ]

    json_path = None
    for p in possible_paths:
        if p.exists():
            json_path = p
            break

    if json_path is None:
        return {}

    try:
        with open(json_path) as f:
            data = json.load(f)

        result = {}

        if 'Tumor' in data:
            elem = data['Tumor']
            if 'stage' in elem:
                stages = [s for s in elem['stage'] if s != 'others']
                if 'G3' in stages:
                    result['grade'] = 2
                elif 'G2' in stages:
                    result['grade'] = 1
                elif 'G1' in stages:
                    result['grade'] = 0

        result['mvi'] = 1 if 'MVI' in data else 0

        if 'Fibrosis' in data:
            elem = data['Fibrosis']
            if 'stage' in elem:
                fib_stages = elem['stage']
                if 'S4' in fib_stages:
                    result['fibrosis'] = 2
                elif any(s in fib_stages for s in ['S2', 'S3']):
                    result['fibrosis'] = 1
                else:
                    result['fibrosis'] = 0

        return result
    except Exception as e:
        return {}


# ============================================================================
# PRODUCTION SYSTEM
# ============================================================================

class ProductionSystem:
    """Production system with thumbnail support"""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self._system = None
        self._database = None
        self._cache: Dict[str, Any] = {}
        self._gt_cache: Dict[str, Dict] = {}
        self.thumbnail_gen = WSIThumbnailGenerator(config)

    def _init_system(self):
        if self._system is not None:
            return

        print("Loading WSI Diagnostic System...")

        sys.path.append("../hcc")

        from WSI_Retrieval_System_VALIDATED_ALL_YEARS import (
            WSIDiagnosticSystem, Config, RetrievalDatabase
        )

        original_config = Config()
        original_config.dataroot = self.config.annotation_root
        original_config.feature_cache_path = self.config.feature_cache_path
        original_config.tumor_encoder_path = self.config.tumor_encoder_path
        original_config.fibrosis_encoder_path = self.config.fibrosis_encoder_path

        self._system = WSIDiagnosticSystem(
            model_checkpoint=self.config.model_checkpoint,
            config=original_config
        )

        self._database = RetrievalDatabase(self.config.embedding_dim)
        self._build_database()

        print(f"✓ System ready with {len(self._cache)} slides in database")

    def _build_database(self):
        """Build database from cached features"""
        print("Building database from cached features...")
        analyses = {}
        cache_dir = self.config.feature_cache_path

        if cache_dir.exists():
            slide_names = [f.stem for f in cache_dir.glob("*.pt")]
            print(f"Found {len(slide_names)} cached slides")

            for name in slide_names:
                try:
                    analysis = self._system.analyzer.analyze(
                        wsi_path=Path("dummy"),
                        slide_name=name,
                        use_cache=True
                    )
                    if analysis:
                        gt = load_ground_truth_from_json(self.config.annotation_root, name)
                        self._gt_cache[name] = gt
                        analyses[name] = analysis
                        self._cache[name] = analysis
                except Exception as e:
                    print(f"  Warning: {name} failed: {e}")

        if analyses:
            self._database.build(analyses)
            gt_count = sum(1 for gt in self._gt_cache.values() if gt)
            print(f"Ground truth loaded for {gt_count} slides")
            print(f"Database built with {len(analyses)} slides")

    def _to_dict(self, analysis) -> Dict:
        """Convert SlideAnalysis to frontend format"""
        def pred_to_dict(pred, gt_value=None):
            dist = pred.distribution
            if hasattr(dist, 'tolist'):
                dist = dist.tolist()
            return {
                'distribution': list(dist),
                'predicted_class': pred.predicted_class,
                'confidence': pred.confidence,
                'class_names': pred.class_names,
                'ground_truth': gt_value if gt_value is not None else pred.ground_truth
            }

        coords = []
        if hasattr(analysis, 'coords') and analysis.coords:
            coords = analysis.coords[:30]

        gt = self._gt_cache.get(analysis.slide_name, {})

        return {
            'slide_name': analysis.slide_name,
            'slide_type': analysis.slide_type,
            'source': analysis.source,
            'grade': pred_to_dict(analysis.grade, gt.get('grade')),
            'fibrosis': pred_to_dict(analysis.fibrosis, gt.get('fibrosis')),
            'mvi': pred_to_dict(analysis.mvi, gt.get('mvi')),
            'num_patches': analysis.num_patches,
            'coords': coords,
            'attention_weights': [1.0/max(1, len(coords))] * len(coords)
        }

    def analyze(self, file_path: str, use_cache: bool = True) -> Dict:
        self._init_system()
        slide_name = Path(file_path).stem

        if use_cache and slide_name in self._cache:
            return {'status': 'complete', 'analysis': self._to_dict(self._cache[slide_name])}

        try:
            analysis = self._system.analyzer.analyze(
                wsi_path=Path(file_path),
                slide_name=slide_name,
                use_cache=use_cache
            )
            if analysis:
                gt = load_ground_truth_from_json(self.config.annotation_root, slide_name)
                self._gt_cache[slide_name] = gt
                self._cache[slide_name] = analysis
                return {'status': 'complete', 'analysis': self._to_dict(analysis)}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

        return {'status': 'error', 'message': 'Analysis failed'}

    def retrieve(self, slide_name: str, top_k: int = 5, strategy: str = 'balanced') -> Dict:
        self._init_system()

        if slide_name not in self._cache:
            return {'error': f'{slide_name} not found in cache'}

        query = self._cache[slide_name]

        from WSI_Retrieval_System_VALIDATED_ALL_YEARS import RetrievalStrategy
        strategy_map = {
            'balanced': RetrievalStrategy.BALANCED,
            'embedding': RetrievalStrategy.EMBEDDING,
            'morphology': RetrievalStrategy.EMBEDDING,
        }
        db_strategy = strategy_map.get(strategy, RetrievalStrategy.BALANCED)

        results = self._database.retrieve(query, top_k, db_strategy)

        return {
            'query_slide': slide_name,
            'strategy': strategy,
            'results': [{
                'slide_name': r.slide_name,
                'similarity': r.similarity_score,
                'analysis': self._to_dict(self._cache[r.slide_name]) 
                    if r.slide_name in self._cache else None
            } for r in results if r.slide_name in self._cache]
        }

    def get_stats(self) -> Dict:
        self._init_system()

        grade_counts = [0, 0, 0]
        fibrosis_counts = [0, 0, 0]
        mvi_counts = [0, 0]

        for analysis in self._cache.values():
            grade_counts[analysis.grade.predicted_class] += 1
            fibrosis_counts[analysis.fibrosis.predicted_class] += 1
            mvi_counts[analysis.mvi.predicted_class] += 1

        return {
            'total_slides': len(self._cache),
            'database_ready': self._database is not None,
            'grade_distribution': {'G1': grade_counts[0], 'G2': grade_counts[1], 'G3': grade_counts[2]},
            'fibrosis_distribution': {'F0-F1': fibrosis_counts[0], 'F2-F3': fibrosis_counts[1], 'F4': fibrosis_counts[2]},
            'mvi_distribution': {'Negative': mvi_counts[0], 'Positive': mvi_counts[1]}
        }

    def get_all_slides(self) -> List[Dict]:
        self._init_system()
        return [
            {
                "slide_name": name,
                "slidename": name,
                "grade": analysis.grade.class_names[analysis.grade.predicted_class],
                "fibrosis": analysis.fibrosis.class_names[analysis.fibrosis.predicted_class],
                "mvi": analysis.mvi.class_names[analysis.mvi.predicted_class],
                "slide_type": analysis.slide_type
            }
            for name, analysis in self._cache.items()
        ]

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="PathoFocus API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

system = None
database = None
analysis_cache = {}
thumbnail_gen = None


class AnalyzeRequest(BaseModel):
    file_path: str
    use_cache: bool = True

class RetrieveRequest(BaseModel):
    slide_name: str
    top_k: int = 5
    strategy: str = 'balanced'


@app.get("/")
async def root():
    mode = "production" if isinstance(system, ProductionSystem) else "demo"
    return {"message": "PathoFocus API", "mode": mode}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "cached": len(analysis_cache)}


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    global analysis_cache
    slide_name = Path(request.file_path).stem

    if isinstance(system, ProductionSystem):
        return system.analyze(request.file_path, request.use_cache)

    if request.use_cache and slide_name in analysis_cache:
        return {'status': 'complete', 'analysis': analysis_cache[slide_name].to_dict()}

    await asyncio.sleep(0.3)
    analysis = MockSlideAnalysis(slide_name)
    analysis_cache[slide_name] = analysis
    return {'status': 'complete', 'analysis': analysis.to_dict()}


@app.post("/api/retrieve")
async def retrieve(request: RetrieveRequest):
    global analysis_cache

    if isinstance(system, ProductionSystem):
        return system.retrieve(request.slide_name, request.top_k, request.strategy)

    if request.slide_name not in analysis_cache:
        analysis_cache[request.slide_name] = MockSlideAnalysis(request.slide_name)

    query = analysis_cache[request.slide_name]
    results = database.search(query.embedding, request.top_k, request.strategy)
    return {'query_slide': request.slide_name, 'strategy': request.strategy, 'results': results}


@app.get("/api/database/stats")
async def stats():
    if isinstance(system, ProductionSystem):
        return system.get_stats()
    return {
        'total_slides': len(database.slides) if database else 0,
        'mode': 'demo',
        'grade_distribution': {'G1': 4, 'G2': 5, 'G3': 3},
        'fibrosis_distribution': {'F0-F1': 6, 'F2-F3': 4, 'F4': 2},
        'mvi_distribution': {'Negative': 7, 'Positive': 5}
    }

@app.get("/api/database/slides")
async def list_slides():
    try:
        if isinstance(system, ProductionSystem):
            system._init_system()  # Ensure system is initialized
            slides = system.get_all_slides()
            print(f"Returning {len(slides)} slides from production database")
            return slides
        
        # Demo mode
        slides = database.slides if database else {}
        result = [
            {
                'slide_name': n,
                'slidename': n,
                'grade': s.grade['class_names'][s.grade['predicted_class']],
                'fibrosis': s.fibrosis['class_names'][s.fibrosis['predicted_class']],
                'mvi': s.mvi['class_names'][s.mvi['predicted_class']],
                'slide_type': s.slide_type
            }
            for n, s in slides.items()
        ]
        print(f"Returning {len(result)} slides from demo database")
        return result
    except Exception as e:
        print(f"Error in list_slides: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# THUMBNAIL ENDPOINTS
# ============================================================================

@app.get("/api/thumbnail/{slide_name}")
async def get_thumbnail(slide_name: str, size: int = 1024):
    """Get WSI thumbnail image"""
    if isinstance(system, ProductionSystem):
        img_bytes = system.thumbnail_gen.get_thumbnail(slide_name, size)
        if img_bytes:
            return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

    # Return 404 if no thumbnail available
    raise HTTPException(status_code=404, detail=f"Thumbnail not found for {slide_name}")


@app.get("/api/region/{slide_name}")
async def get_region(slide_name: str, x: int = 0, y: int = 0, 
                     width: int = 512, height: int = 512, level: int = 0):
    """Get specific region from WSI"""
    if isinstance(system, ProductionSystem):
        img_bytes = system.thumbnail_gen.get_region(slide_name, x, y, width, height, level)
        if img_bytes:
            return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail=f"Region not found for {slide_name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global system, database, analysis_cache, thumbnail_gen

    parser = argparse.ArgumentParser(description='PathoFocus Server')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--production', action='store_true')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    if args.production:
        print("=" * 60)
        print("  PATHOFOCUS - PRODUCTION MODE (with thumbnails)")
        print("=" * 60)
        config = ProductionConfig()
        system = ProductionSystem(config)
        system._init_system()
    else:
        print("=" * 60)
        print("  PATHOFOCUS - DEMO MODE")
        print("=" * 60)
        system = None
        database = MockDatabase()
        print(f"  Mock database: {len(database.slides)} slides")

    print(f"  Server: http://{args.host}:{args.port}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
