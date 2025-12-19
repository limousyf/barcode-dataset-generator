# Barcode Dataset Generator

A standalone tool for generating YOLO training datasets for barcode detection, segmentation, and classification. Uses the [barcodes.dev](https://github.com/limousyf/barcodes.dev) API for barcode generation and degradation.

## Features

- Generate datasets for **80+ barcode symbologies**
- Support for **detection**, **segmentation**, and **classification** tasks
- Realistic **image degradation** (rotation, noise, perspective, blur)
- **Background embedding** for realistic training scenarios
- **Metadata-driven coordinates** for pixel-perfect annotations
- Configurable **train/val/test splits**
- **Parallel processing** for faster generation

## Installation

```bash
git clone https://github.com/your-repo/barcode-dataset-generator.git
cd barcode-dataset-generator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Prerequisites

This tool requires a running instance of the barcodes.dev API:

```bash
# In the barcodes.dev directory
cd /path/to/barcodes.dev
PORT=5001 python3 run.py
```

## Quick Start

```bash
# Generate a small test dataset
python -m src.dataset_generator \
    --output ./my-dataset \
    --samples 50 \
    --symbologies code128 qr datamatrix \
    --task detection \
    --degrade \
    --split 80/10/10

# Generate segmentation dataset with backgrounds
python -m src.dataset_generator \
    --output ./segmentation-dataset \
    --samples 200 \
    --categories linear 2d \
    --task segmentation \
    --backgrounds ~/backgrounds \
    --barcodes-per-image 1-3 \
    --degrade-prob 0.6
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
  --task TYPE                detection, segmentation, or classification
  --label-mode MODE          symbology, category, family, or binary
  --degrade                  Enable degradation effects
  --degrade-prob FLOAT       Degradation probability [default: 0.5]
  --backgrounds PATH         Background images folder
  --barcodes-per-image RANGE Range like "1-3" for multi-barcode images
  --split RATIO              Train/val/test split [default: 80/10/10]
  --format FORMAT            png or jpg [default: png]
  --api-url URL              API server URL [default: http://localhost:5001]
  --workers N                Parallel workers [default: 4]
  --help                     Show this message and exit
```

## Output Structure

### Detection/Segmentation
```
my-dataset/
├── train/
│   ├── images/
│   │   ├── code128_00001.png
│   │   └── ...
│   └── labels/
│       ├── code128_00001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
├── classes.txt
└── class_mapping.json
```

### Classification
```
my-dataset/
├── train/
│   ├── code128/
│   │   ├── 00001.png
│   │   └── ...
│   └── qr/
│       └── ...
├── val/
└── test/
```

## Configuration

Create a `config.yaml` file for custom defaults:

```yaml
api:
  base_url: http://localhost:5001
  timeout: 30

generation:
  default_samples_per_class: 100
  default_split: "80/10/10"

degradation:
  default_probability: 0.5
```

## License

MIT License
