# Claude Code Context - Barcode Dataset Generator

## Project Overview

This is a standalone client application for generating YOLO training datasets for barcode detection, segmentation, and classification. It connects to the **barcodes.dev API** to generate barcodes and apply degradation effects, then assembles them into properly formatted YOLO datasets.

### Relationship to barcodes.dev

This project is an **API client** of the barcodes.dev barcode generation service:

- **API Server**: `barcodes.dev` (Flask app at `/Users/francis/Documents/barcodes.dev`)
- **This Client**: Calls the API to generate barcodes, handles dataset assembly locally

The server provides:
- Barcode generation with metadata (coordinates, regions)
- Image degradation with coordinate transformation tracking
- Multiple barcode symbologies (80+ types)

This client provides:
- YOLO dataset structure creation
- Batch processing and progress tracking
- Background image embedding
- Train/val/test splitting
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

**Option 2: Config File (`config.yaml`)**
```yaml
api:
  base_url: https://barcodes.dev
  # api_key: your-api-key-here  # Better to use env var
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
| `/api/v2/test/generate-and-degrade` | POST | API Key | Generate barcode + apply degradation with metadata |
| `/api/v2/degrade/presets` | GET | API Key | List available degradation presets |
| `/api/v1/barcode/generate` | POST | None | Generate barcode without degradation (legacy) |

### Authentication Header

All v2 endpoints require the `X-API-Key` header:
```
X-API-Key: your-api-key-here
```

### Request/Response Format

**Generate and Degrade Request:**
```json
{
  "barcode_type": "code128",
  "text": "SAMPLE123",
  "degradation": {
    "geometry": {
      "rotation": {"angle": 15},
      "perspective": {"x_tilt": 10}
    },
    "materials": {
      "noise": {"intensity": 0.3}
    }
  }
}
```

**Response with Metadata:**
```json
{
  "success": true,
  "image": "base64-encoded-png",
  "metadata": {
    "original_size": {"width": 400, "height": 150},
    "regions": {
      "barcode_only": {
        "polygon": [[10, 5], [390, 5], [390, 120], [10, 120]],
        "bbox": [10, 5, 390, 120]
      },
      "text_region": {
        "polygon": [[10, 125], [390, 125], [390, 145], [10, 145]]
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
├── setup.py                     # Package installation
├── config.yaml                  # Default configuration
│
├── src/
│   ├── __init__.py
│   ├── api_client.py            # HTTP client for barcodes.dev API
│   ├── dataset_generator.py     # Main dataset generation logic
│   ├── background_manager.py    # Background image handling
│   ├── label_generator.py       # YOLO annotation generation
│   ├── config.py                # Configuration management
│   └── utils.py                 # Helper functions
│
├── tests/
│   ├── test_api_client.py
│   ├── test_dataset_generator.py
│   └── test_label_generator.py
│
├── examples/
│   └── generate_code128_dataset.py
│
└── docs/
    └── api_reference.md
```

## Key Components to Implement

### 1. API Client (`src/api_client.py`)
- HTTP client wrapper for barcodes.dev API
- Handles authentication (if needed in future)
- Retry logic and error handling
- Base64 image decoding
- Metadata extraction

### 2. Dataset Generator (`src/dataset_generator.py`)
- Orchestrates the generation pipeline
- Manages symbology selection and sampling
- Handles train/val/test splitting
- Progress reporting
- Parallel request handling (optional)

### 3. Label Generator (`src/label_generator.py`)
- Converts metadata polygons to YOLO format
- Supports detection (bbox) and segmentation (polygon)
- Handles coordinate normalization
- Classification folder structure

### 4. Background Manager (`src/background_manager.py`)
- Loads background images from folder
- Resizes and crops backgrounds
- Places barcodes with collision detection
- Transforms metadata coordinates after placement

## Configuration

### Environment Variables
```bash
BARCODE_API_URL=http://localhost:5001    # API server URL
BARCODE_API_TIMEOUT=30                    # Request timeout in seconds
```

### Config File (`config.yaml`)
```yaml
api:
  base_url: http://localhost:5001
  timeout: 30
  retry_attempts: 3

generation:
  default_samples_per_class: 100
  default_image_format: png
  default_split: "80/10/10"

degradation:
  default_probability: 0.5
  presets:
    - light
    - moderate
    - heavy

backgrounds:
  default_folder: ~/backgrounds
  target_size: [640, 640]
```

## Supported Features

### Task Types
1. **detection**: Bounding box annotations (YOLO format)
2. **segmentation**: Polygon annotations (YOLO segmentation format)
3. **classification**: Folder-based structure (ImageNet style)

### Label Modes
1. **symbology**: Fine-grained (code128, qr, upca, etc.)
2. **category**: Coarse (linear, 2d, stacked, postal)
3. **family**: Mid-level (code128_family, ean_upc_family, etc.)
4. **binary**: Single "barcode" class

### Barcode Categories
- **linear**: Code 128, Code 39, UPC, EAN, ITF, etc. (40+ types)
- **2d**: QR, DataMatrix, Aztec, MaxiCode, etc. (10+ types)
- **stacked**: PDF417, MicroPDF417, Codablock F, etc.
- **postal**: USPS, Royal Mail, Australia Post, etc.
- **composite**: GS1 composite symbologies
- **popular**: Top 11 most common symbologies

## Development Setup

### Prerequisites
- Python 3.9+
- Running instance of barcodes.dev API (local or remote)

### Installation
```bash
cd /Users/francis/Documents/barcode-dataset-generator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Tests
```bash
pytest tests/ -v
```

### Example Usage
```bash
# Generate small test dataset
python -m src.dataset_generator \
    --output ~/datasets/test-dataset \
    --samples 10 \
    --symbologies code128 qr \
    --task segmentation \
    --api-url http://localhost:5001

# Generate production dataset
python -m src.dataset_generator \
    --output ~/datasets/barcode-detection \
    --samples 1000 \
    --categories linear 2d \
    --task detection \
    --degrade \
    --degrade-prob 0.6 \
    --backgrounds ~/backgrounds \
    --split 80/10/10
```

## CLI Arguments (Target Interface)

```
--output PATH          Output directory for dataset
--samples N            Samples per symbology (default: 100)
--symbologies LIST     Specific symbologies to include
--categories LIST      Barcode categories (linear, 2d, etc.)
--families LIST        Barcode families (code128, ean_upc, etc.)
--task TYPE            detection, segmentation, or classification
--label-mode MODE      symbology, category, family, or binary
--degrade              Enable degradation effects
--degrade-prob FLOAT   Probability of degradation (0.0-1.0)
--backgrounds PATH     Folder with background images
--barcodes-per-image   Range like "1-3" for background embedding
--split RATIO          Train/val/test split like "80/10/10"
--format FORMAT        png or jpg
--api-url URL          API server URL (default: http://localhost:5001)
--workers N            Parallel workers for API requests
--config PATH          Path to config.yaml
```

## Error Handling

### API Connection Errors
- Retry with exponential backoff
- Fall back to cached responses if available
- Clear error messages with troubleshooting hints

### Generation Failures
- Log failed symbologies
- Continue with remaining samples
- Report success/failure statistics

## Performance Considerations

### Parallel Processing
- Use `concurrent.futures` for parallel API requests
- Configurable worker count (default: 4)
- Rate limiting to avoid overwhelming API

### Caching
- Cache degradation presets
- Optional caching of generated barcodes for reproducibility

## Future Enhancements

1. **Web UI**: Simple Flask/Streamlit interface for configuration
2. **Resume Support**: Continue interrupted generation runs
3. **Cloud Storage**: Direct upload to S3/GCS
4. **Distributed Generation**: Multiple workers across machines
5. **Dataset Validation**: Verify generated datasets before training

## Troubleshooting

### API Server Not Running
```
Error: Connection refused to http://localhost:5001
Solution: Start the barcodes.dev server:
  cd /Users/francis/Documents/barcodes.dev
  PORT=5001 python3 run.py
```

### Missing Symbology
```
Error: Unknown symbology 'xyz'
Solution: Check supported symbologies:
  curl http://localhost:5001/api/v1/barcode/symbologies
```

### Large Dataset Memory Issues
```
Solution: Reduce --workers or process in batches with --batch-size
```

---

**Last Updated:** 2025-12-18
**Version:** 0.1.0 (Initial Setup)
**API Compatibility:** barcodes.dev v1.7.0+
