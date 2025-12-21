# Barcode Dataset Generator

A standalone tool for generating training and testing datasets for barcode detection, segmentation, and classification. Uses the [barcodes.dev](https://barcodes.dev) API for barcode generation and degradation.

## Features

- Generate datasets for **80+ barcode symbologies**
- **Multiple output formats**: YOLO, Testplan (decoder testing)
- Support for **detection**, **segmentation**, and **classification** tasks
- Realistic **image degradation** (rotation, noise, perspective, blur)
- **Background embedding** for realistic training scenarios
- **Metadata-driven coordinates** for pixel-perfect annotations
- Configurable **train/val/test splits**
- **Parallel processing** for faster generation

## Installation

```bash
git clone https://github.com/limousyf/barcode-dataset-generator.git
cd barcode-dataset-generator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Prerequisites

1. **Get an API key** from [barcodes.dev](https://barcodes.dev/account/api-keys)
2. Set your API key as an environment variable:
   ```bash
   export BARCODE_API_KEY=your-api-key-here
   ```

## Quick Start

```bash
# Generate a detection dataset (YOLO format)
python -m src.dataset_generator \
    --output ./my-dataset \
    --samples 50 \
    --symbologies code128 qr datamatrix \
    --task detection \
    --degrade \
    --split 80/10/10

# Generate a testplan dataset for decoder testing
python -m src.dataset_generator \
    --output ./decoder-tests \
    --samples 100 \
    --symbologies code128 ean13 \
    --output-format testplan \
    --no-split \
    --degrade --degrade-prob 0.8

# Generate dataset with backgrounds
python -m src.dataset_generator \
    --output ./realistic-dataset \
    --samples 200 \
    --categories linear 2d \
    --task segmentation \
    --backgrounds ~/backgrounds \
    --degrade-prob 0.6
```

## Output Formats

### YOLO Format (default)

Standard format for training object detection and segmentation models.

```
my-dataset/
├── train/
│   ├── images/
│   │   ├── code128_00001.png
│   │   └── ...
│   └── labels/
│       ├── code128_00001.txt    # YOLO annotation
│       └── ...
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml          # Dataset configuration
├── classes.txt        # Class names
└── class_mapping.json # Symbology to class ID mapping
```

For detailed YOLO dataset information, see [docs/YOLO_DATASET_GUIDE.md](docs/YOLO_DATASET_GUIDE.md).

### Testplan Format

JSON sidecar format for barcode decoder testing with comprehensive metadata.

```
decoder-tests/
├── images/
│   ├── code128_000001.png
│   └── ...
├── labels/
│   ├── code128_000001.json    # JSON sidecar
│   └── ...
└── manifest.json              # Dataset summary
```

Each JSON sidecar contains:
- Barcode metadata (symbology, encoded value, orientation)
- Region polygons (full, barcode_only, text_label)
- Bounding boxes
- Degradation info

Example JSON:
```json
{
  "schema_version": "1.0.0",
  "image": { "filename": "code128_000001.png", "width": 640, "height": 640 },
  "barcode": {
    "symbology": "code128",
    "encoded_value": "ABC123",
    "printed_text": "ABC123",
    "orientation": "top-left"
  },
  "regions": {
    "full": { "polygon": [[...]], "vertex_count": 4 },
    "barcode_only": { "polygon": [[...]], "bbox": [x1, y1, x2, y2] },
    "text_label": { "polygon": [[...]] }
  },
  "generation": { "degradation_applied": true, "transformations": ["rotation", "noise"] }
}
```

## Usage

```
python -m src.dataset_generator [OPTIONS]

Options:
  --output PATH              Output directory for dataset (required)
  --samples N                Samples per symbology [default: 100]
  --symbologies LIST         Specific symbologies (code128, qr, upca, etc.)
  --categories LIST          Categories (linear, 2d, stacked, postal, popular)
  --families LIST            Families (code128, ean_upc, qr, pdf417, etc.)
  --output-format FORMAT     Output format: yolo, testplan [default: yolo]
  --task TYPE                detection, segmentation, or classification
  --label-mode MODE          symbology, category, family, or binary
  --degrade                  Enable degradation effects
  --degrade-prob FLOAT       Degradation probability [default: 0.5]
  --backgrounds PATH         Background images folder
  --barcodes-per-image RANGE Range like "1-3" for multi-barcode images
  --split RATIO              Train/val/test split [default: 80/10/10]
  --no-split                 Disable splitting (flat directory structure)
  --format FORMAT            Image format: png or jpg [default: png]
  --api-url URL              API server URL (or set BARCODE_API_URL env var)
  --api-key KEY              API key (or set BARCODE_API_KEY env var)
  --workers N                Parallel workers [default: 4]
  --verbose                  Enable verbose logging
  --help                     Show this message and exit
```

## Configuration

Set your API credentials via environment variables (recommended):

```bash
export BARCODE_API_KEY=your-api-key-here
export BARCODE_API_URL=https://barcodes.dev  # optional, this is the default
```

Or create a `.env.local` file (git-ignored):

```bash
BARCODE_API_KEY=your-api-key-here
BARCODE_API_URL=http://localhost:5001  # for local development
```

Or use a `config.yaml` file for custom defaults:

```yaml
api:
  base_url: https://barcodes.dev
  timeout: 30

generation:
  default_samples_per_class: 100
  default_split: "80/10/10"

degradation:
  default_probability: 0.5
```

## Supported Barcode Categories

| Category | Examples |
|----------|----------|
| **linear** | Code 128, Code 39, UPC-A, EAN-13, ITF-14 |
| **2d** | QR Code, Data Matrix, Aztec, PDF417 |
| **stacked** | PDF417, MicroPDF417, Codablock F |
| **postal** | USPS IMb, Royal Mail, Australia Post |
| **composite** | GS1 Composite symbols |
| **popular** | Top 11 most common symbologies |

## License

MIT License
