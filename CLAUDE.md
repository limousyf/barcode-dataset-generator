# Claude Code Context - Barcode Dataset Generator

## Project Overview

This is a standalone client application for generating training and testing datasets for barcode detection, segmentation, and classification. It connects to the **barcodes.dev API** to generate barcodes and apply degradation effects, then assembles them into properly formatted datasets.

**Supported output formats:**
- **YOLO** - For training object detection/segmentation models
- **Testplan** - JSON sidecar format for decoder testing

### Relationship to barcodes.dev

This project is an **API client** of the barcodes.dev barcode generation service:

- **API Server**: `barcodes.dev` (Flask app at `/Users/francis/Documents/barcodes.dev`)
- **This Client**: Calls the API to generate barcodes, handles dataset assembly locally

The server provides:
- Barcode generation with metadata (coordinates, regions)
- Image degradation with coordinate transformation tracking
- Multiple barcode symbologies (80+ types)

This client provides:
- Multi-format dataset structure creation (YOLO, Testplan)
- Batch processing and progress tracking
- Background image embedding
- Train/val/test splitting
- Degradation parameter sweeps for systematic testing
- Label file generation (detection, segmentation, classification)

## API Environments

The client can connect to two environments:

| Environment | URL | API Key Required |
|-------------|-----|------------------|
| **Local** | `http://localhost:5001` | No |
| **Production** | `https://barcodes.dev` | Yes |

### Setting the Environment

**Option 1: Environment Variables (Recommended)**
```bash
# Local development (no API key needed)
export BARCODE_API_URL=http://localhost:5001

# Production (API key required)
export BARCODE_API_URL=https://barcodes.dev
export BARCODE_API_KEY=your-api-key-here
```

**Option 2: `.env.local` file (git-ignored)**
```bash
BARCODE_API_URL=http://localhost:5001
BARCODE_API_KEY=your-api-key-here
```

**Option 3: CLI Argument**
```bash
python -m src.dataset_generator --api-url https://barcodes.dev --api-key $BARCODE_API_KEY ...
```

### Getting an API Key

1. Create an account at https://barcodes.dev
2. Navigate to Account → API Keys
3. Generate a new key with appropriate permissions
4. Store securely (environment variable or secrets manager)

## API Endpoints Used

### Primary Endpoints (barcodes.dev API)

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `POST /api/v2/barcode/generate` | POST | API Key | Generate barcode + optional degradation with metadata |
| `GET /api/v2/barcode/symbologies` | GET | None | List supported symbologies by category |
| `GET /api/v2/barcode/families` | GET | None | List symbology family groupings |
| `GET /api/v2/degrade/presets` | GET | None | List available degradation presets |

### Authentication Header

Authenticated endpoints require the `X-API-Key` header:
```
X-API-Key: your-api-key-here
```

### Degradation Config Format

The API expects degradation configs with categories as **lists** of transformation objects:

```json
{
  "geometry": [
    {"type": "y_axis_rotation", "angle_degrees": 15}
  ],
  "damage": [
    {"type": "motion_blur", "intensity": 2.0, "direction": 45},
    {"type": "scratches", "count": 3, "severity": 0.4}
  ],
  "materials": [
    {"type": "transparent_overlay", "opacity": 0.8}
  ]
}
```

**Available transformation types:**

| Category | Types |
|----------|-------|
| `geometry` | `y_axis_rotation`, `x_axis_rotation`, `z_axis_rotation`, `cylindrical`, `flexible_wrinkle` |
| `damage` | `motion_blur`, `fading`, `scratches`, `ink_bleeding`, `broken_bars`, `low_ink`, `white_noise`, `glare`, `water_droplets`, `stains`, `smudges`, `partial_removal`, `low_light`, `overexposure` |
| `materials` | `metallic_reflection`, `transparent_overlay` |

### API Response Format

**Generate Barcode Response:**
```json
{
  "success": true,
  "image": "base64-encoded-png",
  "format": "PNG",
  "degradation_applied": true,
  "transformations": [
    {"type": "y_axis_rotation", "category": "geometry", "success": true, "parameters": {...}}
  ],
  "metadata": {
    "original_size": {"width": 400, "height": 150},
    "regions": {
      "barcode_only": {
        "polygon": [[10, 5], [390, 5], [390, 120], [10, 120]],
        "bbox": [10, 5, 390, 120]
      },
      "text": {
        "polygon": [[10, 125], [390, 125], [390, 145], [10, 145]],
        "present": true
      }
    }
  }
}
```

## Architecture

```
barcode-dataset-generator/
├── CLAUDE.md                    # This file
├── README.md                    # User documentation
├── requirements.txt             # Python dependencies
├── config.yaml                  # Default configuration
├── .env.local                   # Local environment (git-ignored)
│
├── src/
│   ├── __init__.py
│   ├── api_client.py            # HTTP client for barcodes.dev API
│   ├── dataset_generator.py     # Main dataset generation logic + CLI
│   ├── background_manager.py    # Background image handling
│   ├── label_generator.py       # YOLO annotation generation (legacy)
│   ├── config.py                # Configuration management
│   ├── utils.py                 # Helper functions
│   │
│   └── formats/                 # Output format handlers
│       ├── __init__.py          # Format registry
│       ├── base.py              # OutputFormat ABC + AnnotationData
│       ├── yolo.py              # YOLO format handler
│       └── testplan.py          # Testplan JSON format handler
│
├── tests/
│   ├── test_api_client.py
│   ├── test_label_generator.py
│   └── test_utils.py
│
└── docs/
    └── YOLO_DATASET_GUIDE.md
```

## Key Components

### 1. API Client (`src/api_client.py`)
- HTTP client wrapper for barcodes.dev API
- Handles API key authentication
- Retry logic with exponential backoff
- Base64 image decoding
- BarcodeResult dataclass with region accessors

### 2. Dataset Generator (`src/dataset_generator.py`)
- Orchestrates the generation pipeline
- Manages symbology selection and sampling
- Handles train/val/test splitting
- Supports degradation sweeps for systematic testing
- Progress reporting with tqdm
- Parallel request handling

### 3. Format Handlers (`src/formats/`)
- **YOLOFormat**: YOLO detection/segmentation annotations
- **TestplanFormat**: JSON sidecar files for decoder testing
- Extensible base class for adding new formats

### 4. Background Manager (`src/background_manager.py`)
- Loads background images from folder
- Resizes and crops backgrounds
- Places barcodes with collision detection
- Transforms metadata coordinates after placement

## CLI Arguments

```
python -m src.dataset_generator [OPTIONS]

Required:
  --output, -o PATH        Output directory for dataset

Samples:
  --samples, -n N          Samples per symbology (default: 100)
  --single, -1             Generate single sample (quick test mode)

Symbology Selection:
  --symbologies LIST       Specific symbologies (code128, qr, upca, etc.)
  --categories LIST        Categories (linear, 2d, stacked, postal, popular)
  --families LIST          Families (code128, ean_upc, qr, pdf417, etc.)

Output Format:
  --output-format, -f      Format: yolo, testplan (default: yolo)
  --task TYPE              detection, segmentation, or classification
  --label-mode MODE        symbology, category, family, or binary

Degradation:
  --degrade                Enable random degradation effects
  --degrade-prob FLOAT     Degradation probability (default: 0.5)
  --degrade-preset NAME    Use API degradation preset
  --degrade-config FILE    Load degradation config from JSON file
  --degrade-sweep TYPE MIN MAX STEPS
                           Sweep a degradation parameter

Background:
  --backgrounds PATH       Background images folder
  --barcodes-per-image     Range like "1-3" for multi-barcode images

Splitting:
  --split RATIO            Train/val/test split (default: 80/10/10)
  --no-split               Disable splitting (flat directory)

Other:
  --format FORMAT          Image format: png or jpg (default: png)
  --api-url URL            API server URL
  --api-key KEY            API key
  --workers N              Parallel workers (default: 4)
  --verbose, -v            Enable verbose logging
```

## Degradation Sweep Types

For systematic decoder testing with `--degrade-sweep TYPE MIN MAX STEPS`:

| Category | Sweep Type | Description | Range |
|----------|------------|-------------|-------|
| Geometry | `rotation_y` | Y-axis rotation | -180° to 180° |
| | `rotation_x` | X-axis rotation | -90° to 90° |
| | `rotation_z` | Z-axis rotation | -90° to 90° |
| | `cylindrical` | Cylindrical warp radius | 10 to 500 |
| | `wrinkle` | Wrinkle depth | 0.05 to 0.5 |
| Damage | `blur` | Motion blur intensity | 0.5 to 10.0 |
| | `fading` | Contrast reduction | 0.1 to 0.9 |
| | `scratches` | Scratch severity | 0.1 to 1.0 |
| | `ink_bleeding` | Ink bleed intensity | 0.1 to 1.0 |
| | `broken_bars` | Bar break intensity | 0.5 to 1.0 |
| | `low_ink` | Low ink effect | 0.3 to 1.0 |
| | `white_noise` | Noise intensity | 0.4 to 1.0 |
| | `glare` | Glare intensity | 0.1 to 1.0 |
| | `water_droplets` | Water effect | 0.1 to 1.0 |
| | `stains` | Stain intensity | 0.1 to 1.0 |
| | `smudges` | Smudge intensity | 0.1 to 1.0 |
| | `partial_removal` | Label removal | 0.1 to 0.6 |
| | `low_light` | Darkness level | 0.1 to 0.9 |
| | `overexposure` | Brightness level | 0.1 to 0.9 |
| Materials | `metallic` | Reflection specularity | 0.1 to 1.0 |
| | `transparent` | Overlay opacity | 0.1 to 0.9 |

## Example Usage

```bash
# Quick single-sample test
python -m src.dataset_generator -o ./test -1 --symbologies code128

# Generate YOLO detection dataset
python -m src.dataset_generator -o ./yolo-dataset \
    --samples 100 --symbologies code128 qr \
    --task detection --degrade

# Generate testplan for decoder testing (flat structure)
python -m src.dataset_generator -o ./decoder-tests \
    --samples 50 --symbologies code128 \
    --output-format testplan --no-split

# Rotation sweep for systematic testing
python -m src.dataset_generator -o ./rotation-test \
    --samples 10 --symbologies code128 \
    --output-format testplan --no-split \
    --degrade-sweep rotation_y -30 30 10

# Blur sweep
python -m src.dataset_generator -o ./blur-test \
    --samples 10 --symbologies code128 \
    --output-format testplan --no-split \
    --degrade-sweep blur 0.5 5.0 10
```

## Output Formats

### YOLO Format
```
dataset/
├── train/images/
├── train/labels/
├── val/images/
├── val/labels/
├── test/images/
├── test/labels/
├── data.yaml
├── classes.txt
└── class_mapping.json
```

### Testplan Format
```
dataset/
├── images/
│   └── code128_000001.png
├── labels/
│   └── code128_000001.json    # JSON sidecar
└── manifest.json              # Dataset summary
```

## Error Handling

### API Connection Errors
- Retry with exponential backoff (configurable attempts)
- Clear error messages with troubleshooting hints

### Generation Failures
- Log failed symbologies
- Continue with remaining samples
- Report success/failure statistics

## Troubleshooting

### API Server Not Running
```
Error: Connection refused to http://localhost:5001
Solution: Start the barcodes.dev server:
  cd /Users/francis/Documents/barcodes.dev
  PORT=5001 python3 run.py
```

### API Key Required
```
Error: API key required
Solution: Set BARCODE_API_KEY environment variable or use --api-key
```

---

**Last Updated:** 2025-12-21
**Version:** 0.2.0
**API Compatibility:** barcodes.dev v2.0+
