# YOLO Barcode Dataset Generator - Complete Guide

A comprehensive tool for generating YOLO-format training datasets for barcode **detection**, **classification**, and **instance segmentation** across multiple symbology types.

> **Note**: This is an API client that connects to the barcodes.dev service for barcode generation and degradation. See [API Configuration](#api-configuration) for setup.

## Overview

This tool generates synthetic barcode images with YOLO-format annotations for training three types of models:
- **Object Detection**: Locating and classifying barcodes with bounding boxes
- **Instance Segmentation**: Precise polygon masks around barcode modules
- **Classification**: Recognizing barcode types without localization

It supports 80+ barcode symbologies across 6 major categories, with optional degradation effects to simulate real-world scanning conditions.

### Key Features

- **Multi-category support**: Linear, 2D, Stacked, Postal, Composite, and Popular barcodes
- **80+ symbologies**: From common types (UPC-A, QR Code, PDF417) to specialized formats (DAFT, MaxiCode, UPNQR, Telepen, Pharmacode)
- **Popular codes category**: Pre-curated selection of the 11 most commonly-used barcode types for focused training
- **Three task types**: Detection (bounding boxes), Segmentation (precise polygons), and Classification (barcode type recognition)
- **Precise segmentation**: Pixel-accurate polygons around barcode modules (bars/squares/dots), excluding quiet zones and padding
- **Text inclusion control**: Option to include or exclude human-readable text in segmentation polygons
- **Rotation-aware segmentation**: Accurate polygon generation for rotated and transformed barcodes
- **Intelligent data generation**: Each symbology receives appropriate sample data (numeric for ITF, URLs for QR codes, etc.)
- **Automatic train/val/test splitting**: Built-in dataset splitting with configurable ratios (e.g., 70/20/10)
- **Stratified splitting**: Ensures every symbology has balanced representation across all splits for robust evaluation
- **Optional degradation**: Realistic effects including blur, noise, compression, lighting variations, perspective distortion, and rotation
- **YOLO-ready output**: Normalized bounding boxes or segmentation polygons in standard YOLO format
- **Metadata-driven coordinates**: Pixel-perfect annotation accuracy using coordinate tracking through transformations
- **Complete dataset structure**: Images, labels, class mappings, and YAML configuration
- **Flexible label modes**: Choose between symbology-level, category-level, family-level, or binary labeling

## API Configuration

This tool requires access to the barcodes.dev API for barcode generation.

### Local Development (No API Key Required)

```bash
# Start the local server (in barcodes.dev directory)
cd /path/to/barcodes.dev
PORT=5001 python3 run.py

# Generate dataset using local API
python -m src.dataset_generator \
    --output ./my-dataset \
    --api-url http://localhost:5001 \
    ...
```

### Production (API Key Required)

```bash
# Set environment variables
export BARCODE_API_URL=https://barcodes.dev
export BARCODE_API_KEY=your-api-key-here

# Generate dataset
python -m src.dataset_generator \
    --output ./my-dataset \
    ...
```

Or use CLI arguments:

```bash
python -m src.dataset_generator \
    --output ./my-dataset \
    --api-url https://barcodes.dev \
    --api-key your-api-key-here \
    ...
```

## Label Modes

The generator supports four labeling modes to match different training objectives:

### 1. Symbology Mode (Default)

**Each symbology is a separate class** for fine-grained classification.

- **Classes**: One per symbology (e.g., `code39`, `code128`, `upca`, `qr`, `datamatrix`)
- **Use Case**: Precise symbology identification
- **Example**: 11 classes for `popular` category, 40+ for `linear`, 80+ for `--all`
- **Training Goal**: Detect AND classify exact barcode type
- **Inference Speed**: Medium (~15ms)

```bash
# Default behavior - no flag needed
--label-mode symbology
```

### 2. Category Mode

**Group symbologies by category** for coarse classification.

- **Classes**: One per category (e.g., `linear`, `2d`, `stacked`, `postal`, `composite`)
- **Use Case**: High-level barcode classification
- **Example**: 3 classes for `--categories linear 2d stacked`
- **Training Goal**: Detect AND classify barcode category
- **Inference Speed**: Fast (~10ms)

```bash
--label-mode category
```

**Example Output:**
- All linear barcodes (Code 39, Code 128, UPC, EAN) → class 0 `linear`
- All 2D barcodes (QR, DataMatrix, Aztec) → class 1 `2d`
- All stacked barcodes (PDF417, MicroPDF) → class 2 `stacked`

### 3. Family Mode

**Group symbologies by family** for mid-level classification.

- **Classes**: One per family (e.g., `code128`, `ean_upc`, `qr`, `pdf417`)
- **Use Case**: Group related barcode variants
- **Example**: 5 classes for Code 128 family, EAN/UPC family, QR family, PDF417 family, Code 39 family
- **Training Goal**: Detect AND classify barcode family
- **Inference Speed**: Fast (~10ms)

```bash
--label-mode family
```

**Example Output:**
- Code 128, GS1-128, EAN-14, NVE18, SSCC18 → class 0 `code128`
- EAN-13, EAN-8, UPC-A, UPC-E, ISBN → class 1 `ean_upc`
- QR, MicroQR, rMQR, UPNQR → class 2 `qr`
- PDF417, PDF417 Compact, MicroPDF417 → class 3 `pdf417`

**Why use family mode?**
- Groups barcode variants that share common characteristics
- More specific than category (linear/2d) but less granular than individual symbologies
- Perfect for applications that need to distinguish between Code 128 vs EAN/UPC but don't need to differentiate EAN-13 vs EAN-8

### 4. Binary Mode

**Single "barcode" class** for pure detection without classification.

- **Classes**: 1 class (`barcode`)
- **Use Case**: "Is there a barcode?" detection only
- **Example**: All symbologies map to single class
- **Training Goal**: Detect presence of ANY barcode
- **Inference Speed**: Fastest (~5ms)

```bash
--label-mode binary
```

**Example Output:**
- Code 39 → class 0 `barcode`
- QR Code → class 0 `barcode`
- PDF417 → class 0 `barcode`
- (All symbologies get same label)

### Choosing the Right Mode

| Requirement | Recommended Mode | Classes | Accuracy | Speed |
|-------------|------------------|---------|----------|-------|
| Need exact symbology (e.g., UPC-A vs EAN-13) | symbology | 10-80 | Highest | Medium |
| Need barcode variant groups (Code 128 vs EAN/UPC) | family | 5-15 | High | Fast |
| Need barcode category (linear vs 2D vs stacked) | category | 3-6 | High | Fast |
| Just need to find barcodes (no classification) | binary | 1 | N/A | Fastest |
| Training a general-purpose detector | binary | 1 | N/A | Fastest |
| Sorting/routing by barcode family | family | 5-15 | High | Fast |
| Compliance/validation (must verify exact type) | symbology | 10-80 | Highest | Medium |

## Task Types

The generator supports three task types for different training objectives:

### 1. Detection (Default)

**Bounding box annotations** for object detection models.

- **Output Format**: `<class_id> <x_center> <y_center> <width> <height>`
- **Use Case**: Locate and classify barcodes in images
- **Annotation**: Rectangular bounding boxes around entire barcode (including quiet zones)
- **Training Goal**: Fast barcode detection and classification
- **Model Type**: YOLOv8 Detection (yolov8n.pt, yolov8s.pt, etc.)
- **Inference Speed**: Fast (~5-15ms per image)

```bash
# Default behavior - no flag needed
--task detection
```

**Example annotation:**
```
0 0.500000 0.500000 0.800000 0.300000
```

### 2. Segmentation

**Precise polygon annotations** for instance segmentation models.

- **Output Format**: `<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...`
- **Use Case**: Pixel-accurate localization of barcode modules
- **Annotation**: Polygons tracing the exact outline of barcode content (bars, squares, dots)
- **Excludes**: Quiet zones, padding, and background
- **Includes** (optional): Human-readable text below/above barcode
- **Training Goal**: Precise segmentation for cropping, restoration, quality assessment
- **Model Type**: YOLOv8 Segmentation (yolov8n-seg.pt, yolov8s-seg.pt, etc.)
- **Inference Speed**: Medium (~10-25ms per image)

```bash
--task segmentation
```

**Example annotation:**
```
0 0.123456 0.234567 0.156789 0.234567 0.156789 0.765432 0.123456 0.765432
```

#### Segmentation Features

**Precise Module Detection:**
- **Linear barcodes**: Polygon wraps around the black bars only
- **2D barcodes**: Polygon wraps around the matrix modules (squares/dots)
- **Stacked barcodes**: Polygon encompasses all stacked rows
- **Excludes quiet zones**: Focuses on actual barcode content

**Text Inclusion Options:**
```bash
# Barcode modules only (default)
--task segmentation

# Include human-readable text in polygon
--task segmentation --include-text
```

**Rotation Support:**
- Works correctly with rotated barcodes (any angle)
- Polygon vertices ordered clockwise for consistency
- Handles background embedding with rotation

### 3. Classification

**Folder-based structure** for barcode type recognition models.

- **Output Format**: Folder structure `train/class_name/image.jpg` (no label files)
- **Use Case**: Recognize barcode type without localization
- **Annotation**: None - folder name determines the class
- **Training Goal**: "What type of barcode is this?" classification
- **Model Type**: Image classification models (ResNet, EfficientNet, ViT, etc.) or YOLOv8-cls
- **Inference Speed**: Very Fast (~2-8ms per image)

```bash
--task classification
```

**Example directory structure:**
```
dataset/
├── train/
│   ├── code39/
│   │   ├── code39_00000.png
│   │   └── ...
│   ├── code128/
│   │   ├── code128_00000.png
│   │   └── ...
│   └── qr/
│       ├── qr_00000.png
│       └── ...
└── val/
    └── ...
```

### Choosing the Right Task Type

| Requirement | Recommended Task | Annotation | Use Case |
|-------------|------------------|------------|----------|
| Find barcodes in images | Detection | Bounding box | General detection, real-time scanning |
| Identify barcode type only | Classification | None (folder) | Type recognition, pre-cropped images |
| Need exact barcode boundaries | Segmentation | Polygon | Precise cropping, alignment correction |
| Rotate/transform barcodes | Segmentation | Polygon | Image rectification, perspective correction |
| Assess barcode quality | Segmentation | Polygon | Quality control, damage detection |
| Extract barcode only | Segmentation | Polygon | Clean barcode extraction for processing |
| Multiple barcodes in scene | Detection | Bounding box | Warehouse/retail environments |
| Damaged barcode restoration | Segmentation | Polygon | Repair/enhancement pipelines |
| Two-stage pipeline (detect→classify) | Classification | None (folder) | Fast type verification after detection |
| Standard classification models | Classification | None (folder) | ResNet, EfficientNet, ViT compatibility |

## Symbology Filtering

The generator provides three powerful ways to select which symbologies to include in your dataset:

### Filter Types

#### 1. Categories (`--categories`)
Select broad groups of barcodes by category:

```bash
--categories linear 2d stacked postal composite popular
```

Available categories:
- `linear`: 1D barcodes (Code 128, UPC, EAN, Code 39, etc.) - 40 symbologies
- `2d`: 2D matrix codes (QR, DataMatrix, Aztec, etc.) - 10 symbologies
- `stacked`: Stacked/composite linear codes (PDF417, etc.) - 9 symbologies
- `postal`: Postal service barcodes (USPS, Royal Mail, etc.) - 10 symbologies
- `composite`: GS1 composite symbologies - 9 symbologies
- `popular`: 11 most commonly-used types

#### 2. Symbologies (`--symbologies`)
Select specific individual symbology types:

```bash
--symbologies code128 qr upca datamatrix
```

Examples: `code128`, `qr`, `upca`, `ean13`, `pdf417`, `aztec`, `code39`, `datamatrix`, `itf14`, etc.

#### 3. Families (`--families`)
Select groups of related barcode variants:

```bash
--families code128 ean_upc qr
```

Available families:
- `code128`: Code 128, GS1-128, EAN-14, NVE18, SSCC18 (5 symbologies)
- `ean_upc`: EAN-13/8, UPC-A/E, ISBN, EAN-2/5 (9 symbologies)
- `code39`: Code 39, Extended Code 39, Code 32, HIBC 39, VIN, LOGMARS (6 symbologies)
- `code2of5`: Standard 2of5, IATA, Industrial, Interleaved, Logic, ITF, ITF-14 (7 symbologies)
- `qr`: QR Code, MicroQR, rMQR, UPNQR (4 symbologies)
- `pdf417`: PDF417, PDF417 Compact, MicroPDF417 (3 symbologies)
- `usps`: Intelligent Mail, Postnet, Planet (3 symbologies)
- And many more...

### Combining Filters

**Filters are additive and automatically deduplicated:**

```bash
# Select all linear barcodes AND add QR code
--categories linear --symbologies qr

# Select Code 128 family AND add specific DataMatrix
--families code128 --symbologies datamatrix

# Select Code 2 of 5 family AND ITF (ITF counted only once)
--families code2of5 --symbologies itf

# Complex combination
--categories 2d --families code128 ean_upc --symbologies itf14
```

### Default Behavior

If no filters are specified, defaults to `--categories linear 2d` (50+ symbologies).

## Supported Symbologies

### Linear Barcodes (40 types)
- **Code 128 Family**: Code 128, GS1-128, EAN-14, NVE18, SSCC18
- **EAN/UPC Family**: EAN-13, EAN-8, EAN-2, EAN-5, UPC-A, UPC-E, ISBN (10/13)
- **Code 39 Family**: Code 39, Extended Code 39, Code 32, HIBC 39, VIN, LOGMARS
- **Code 93**: Code 93
- **Code 11**: Code 11
- **Code 2 of 5 Family**: Standard, IATA, Industrial, Interleaved, Logic, ITF, ITF-14
- **Plessey Family**: Plessey, MSI Plessey
- **Telepen**: Telepen Alpha, Telepen Numeric
- **Others**: Codabar, Pharmacode, PZN, Channel Code
- **GS1 DataBar**: Omnidirectional, Limited, Expanded

### 2D Barcodes (10 types)
- **QR Code Variants**: QR Code, MicroQR, rMQR (Rectangular Micro QR), UPN QR
- **DataMatrix**: DataMatrix
- **Aztec**: Aztec Code
- **Others**: MaxiCode, Code One, Grid Matrix, DotCode, Han Xin

### Stacked Barcodes (9 types)
- **PDF417 Variants**: PDF417, PDF417 Compact, MicroPDF417
- **Code Variants**: Codablock-F, Code 16K, Code 49
- **GS1 DataBar Stacked**: Stacked, Stacked Omnidirectional, Expanded Stacked

### Postal Codes (10 types)
- **USPS**: Postnet, Planet, Intelligent Mail (IMB)
- **International**: Royal Mail 4-State, Australia Post, Japan Post, Netherlands KIX, Korea Post
- **Deutsche Post**: DPLEIT, DPIDENT
- **DAFT**: DAFT Code

### Composite Symbologies (9 types)
- **GS1-128**: GS1-128 with Composite Component
- **EAN/UPC**: EAN-CC, UPC-A-CC, UPC-E-CC
- **GS1 DataBar**: Omnidirectional-CC, Limited-CC, Expanded-CC, Stacked-CC, Omnidirectional Stacked-CC, Expanded Stacked-CC

### Popular Codes (11 types)
A curated selection of the most commonly-used barcode symbologies in retail, logistics, and general commerce:
- **Code 39**: Alphanumeric barcode widely used in automotive and healthcare
- **Code 128**: High-density linear barcode, standard for shipping labels
- **UPC-A**: Universal Product Code used in retail (North America)
- **UPC-E**: Compressed version of UPC-A for small packages
- **EAN-13**: European Article Number, retail standard globally
- **EAN-8**: Compact version of EAN-13 for small products
- **QR Code**: 2D matrix barcode for URLs, payments, and general data
- **DataMatrix**: 2D barcode for small items, electronics marking
- **PDF417**: Stacked barcode for licenses, shipping, and identification
- **ITF-14**: Interleaved 2 of 5, used for carton/case identification
- **Aztec Code**: 2D barcode for transport tickets and documents

**Use Case**: Perfect for training general-purpose barcode detectors without the overhead of 80+ symbologies. Covers ~90% of real-world barcode scanning scenarios.

## Command-Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--output` | `-o` | string | *required* | Output directory for the dataset |
| `--samples` | `-n` | int | 100 | Number of samples to generate per symbology |
| `--categories` | `-c` | list | none | Barcode categories to include (linear, 2d, stacked, postal, composite, popular) |
| `--symbologies` | | list | none | Specific symbologies to include (e.g., code128, qr, upca) |
| `--families` | | list | none | Barcode families to include (e.g., code128, ean_upc, qr) |
| `--all` | `-a` | flag | false | Include all barcode categories (shortcut for all 6 categories) |
| `--label-mode` | | string | symbology | Label mode: `symbology`, `category`, `family`, or `binary` |
| `--task` | | string | detection | Task type: `detection` (bounding boxes), `segmentation` (polygons), or `classification` (type recognition) |
| `--include-text` | | flag | false | Include human-readable text in segmentation polygons (only for `--task segmentation`) |
| `--degrade` | `-d` | flag | false | Enable degradation effects |
| `--degrade-prob` | `-p` | float | 0.5 | Probability of degradation (0.0-1.0) |
| `--split` | | string | none | Split ratio for train/val/test (e.g., "70/20/10") |
| `--no-stratify` | | flag | false | Disable stratified splitting (use random split instead) |
| `--format` | | string | png | Output image format: png or jpeg |
| `--use-backgrounds` | | flag | false | Enable background embedding mode (place barcodes in real scenes) |
| `--backgrounds` | | string | none | Path to folder containing background images (required with --use-backgrounds) |
| `--barcodes-per-image` | | string | "1-2" | Number of barcodes per composite image (e.g., "0-3", "1-5") |
| `--barcode-scale` | | string | "0.1-0.3" | Scale range for barcodes relative to image size (e.g., "0.1-0.5") |
| `--api-url` | | string | http://localhost:5001 | API server URL |
| `--api-key` | | string | none | API key for v2 endpoints (or set BARCODE_API_KEY) |
| `--workers` | | int | 4 | Number of parallel workers for API requests |

## Usage Examples

### Quick Start

```bash
# Generate a small test dataset (local API)
python -m src.dataset_generator \
    --output ./test-dataset \
    --samples 10 \
    --symbologies code128 qr \
    --task detection \
    --split 80/10/10

# Generate with degradation
python -m src.dataset_generator \
    --output ./degraded-dataset \
    --samples 100 \
    --categories popular \
    --degrade \
    --degrade-prob 0.5 \
    --split 80/10/10
```

### Detection Dataset

```bash
python -m src.dataset_generator \
    --output ./detection-dataset \
    --samples 500 \
    --categories linear 2d \
    --task detection \
    --degrade \
    --degrade-prob 0.6 \
    --split 70/20/10
```

### Segmentation Dataset

```bash
python -m src.dataset_generator \
    --output ./segmentation-dataset \
    --samples 300 \
    --categories popular \
    --task segmentation \
    --degrade \
    --degrade-prob 0.4 \
    --split 80/10/10
```

### Classification Dataset

```bash
python -m src.dataset_generator \
    --output ./classification-dataset \
    --samples 500 \
    --categories popular \
    --task classification \
    --label-mode symbology \
    --degrade \
    --degrade-prob 0.3 \
    --split 80/10/10
```

### With Background Embedding

```bash
python -m src.dataset_generator \
    --output ./realistic-dataset \
    --samples 200 \
    --categories popular \
    --task detection \
    --use-backgrounds \
    --backgrounds ~/backgrounds \
    --barcodes-per-image 1-3 \
    --barcode-scale 0.1-0.4 \
    --degrade \
    --degrade-prob 0.3 \
    --split 80/10/10
```

### Using Production API

```bash
export BARCODE_API_KEY=your-api-key-here

python -m src.dataset_generator \
    --output ./production-dataset \
    --samples 1000 \
    --categories linear 2d \
    --task segmentation \
    --api-url https://barcodes.dev \
    --degrade \
    --split 80/10/10
```

## Output Structure

### Detection/Segmentation with Split

```
dataset/
├── train/
│   ├── images/
│   │   ├── code128_00000.png
│   │   └── ...
│   └── labels/
│       ├── code128_00000.txt
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

### Classification with Split

```
dataset/
├── train/
│   ├── code39/
│   │   ├── 00001.png
│   │   └── ...
│   └── code128/
│       └── ...
├── val/
│   ├── code39/
│   └── code128/
└── test/
    └── ...
```

## Training with YOLOv8

### Detection

```bash
yolo train data=/path/to/data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### Segmentation

```bash
yolo segment train data=/path/to/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
```

### Classification

```bash
yolo classify train data=/path/to/dataset model=yolov8n-cls.pt epochs=100 imgsz=224
```

## Troubleshooting

### API Connection Issues

```
Error: Connection refused to http://localhost:5001
Solution: Ensure the barcodes.dev server is running:
  cd /path/to/barcodes.dev
  PORT=5001 python3 run.py
```

### API Key Issues

```
Error: API key required for barcodes.dev v2 endpoints
Solution: Set the BARCODE_API_KEY environment variable or use --api-key
```

### Generation Failures

Some symbologies may fail to generate due to strict data format requirements. The generator will continue with other symbologies and report failures in the summary.

## Version History

Based on barcodes.dev YOLO Dataset Generator v1.7.0
- Detection, segmentation, and classification support
- Four label modes: symbology, category, family, binary
- Advanced filtering: categories, symbologies, families
- Degradation presets and custom configurations
- Background embedding for realistic scenes
- Stratified train/val/test splitting
- Metadata-driven pixel-perfect annotations
