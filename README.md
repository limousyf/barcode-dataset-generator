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
  --single                   Generate a single sample (quick test mode)
  --symbologies LIST         Specific symbologies (code128, qr, upca, etc.)
  --categories LIST          Categories (linear, 2d, stacked, postal, popular)
  --families LIST            Families (code128, ean_upc, qr, pdf417, etc.)
  --output-format FORMAT     Output format: yolo, testplan [default: yolo]
  --task TYPE                detection, segmentation, or classification
  --label-mode MODE          symbology, category, family, or binary
  --degrade                  Enable random degradation effects
  --degrade-prob FLOAT       Degradation probability [default: 0.5]
  --degrade-preset NAME      Use API degradation preset
  --degrade-config FILE      Load degradation config from JSON file
  --degrade-sweep TYPE MIN MAX STEPS
                             Sweep a degradation parameter (see below)
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

## Degradation Options

The generator supports three ways to apply degradation effects:

### Random Degradation

Apply random degradation with configurable probability:

```bash
python -m src.dataset_generator -o ./dataset --symbologies code128 \
    --degrade --degrade-prob 0.7
```

### Parameter Sweeps

Generate systematic test sets with controlled degradation levels using `--degrade-sweep`:

```bash
# Syntax: --degrade-sweep TYPE MIN MAX STEPS

# Y-axis rotation from -30° to +30° in 5 steps
python -m src.dataset_generator -o ./rotation-test --symbologies code128 \
    --samples 5 --output-format testplan --no-split \
    --degrade-sweep rotation_y -30 30 5

# Motion blur from light (0.5) to heavy (5.0) in 10 steps
python -m src.dataset_generator -o ./blur-test --symbologies code128 \
    --samples 10 --output-format testplan --no-split \
    --degrade-sweep blur 0.5 5.0 10

# Fading/contrast reduction sweep
python -m src.dataset_generator -o ./fading-test --symbologies code128 \
    --samples 5 --output-format testplan --no-split \
    --degrade-sweep fading 0.1 0.5 5
```

**Available sweep types:**

| Category | Sweep Type | Description | Valid Range |
|----------|------------|-------------|-------------|
| **Geometry** | `rotation_y` | Y-axis rotation (horizontal tilt) | -180° to 180° |
| | `rotation_x` | X-axis rotation (vertical tilt) | -90° to 90° |
| | `rotation_z` | Z-axis rotation (in-plane) | -90° to 90° |
| | `cylindrical` | Cylindrical surface warp radius | 10 to 500 |
| | `wrinkle` | Flexible surface wrinkle depth | 0.05 to 0.5 |
| **Damage** | `blur` / `motion_blur` | Motion blur intensity | 0.5 to 10.0 |
| | `fading` | Contrast reduction | 0.1 to 0.9 |
| | `scratches` | Scratch severity | 0.1 to 1.0 |
| | `ink_bleeding` | Ink bleed/spread intensity | 0.1 to 1.0 |
| | `broken_bars` | Bar break intensity | 0.5 to 1.0 |
| | `low_ink` | Low ink/faded print effect | 0.3 to 1.0 |
| | `white_noise` | Random noise intensity | 0.4 to 1.0 |
| | `glare` | Glare/hotspot intensity | 0.1 to 1.0 |
| | `water_droplets` | Water droplet effect intensity | 0.1 to 1.0 |
| | `stains` | Stain/dirt intensity | 0.1 to 1.0 |
| | `smudges` | Smudge/fingerprint intensity | 0.1 to 1.0 |
| | `partial_removal` | Label removal coverage | 0.1 to 0.6 |
| | `low_light` | Darkness level | 0.1 to 0.9 |
| | `overexposure` | Brightness/washout level | 0.1 to 0.9 |
| **Materials** | `metallic` | Metallic reflection specularity | 0.1 to 1.0 |
| | `transparent` | Transparent overlay opacity | 0.1 to 0.9 |

### Custom Configuration

Load a custom degradation config from a JSON file:

```bash
python -m src.dataset_generator -o ./custom-test --symbologies code128 \
    --samples 20 --degrade-config my-degradation.json
```

Example `my-degradation.json`:
```json
{
  "geometry": [
    {"type": "y_axis_rotation", "angle_degrees": 15},
    {"type": "cylindrical", "radius": 50, "axis": "vertical", "wrap_angle": 180}
  ],
  "damage": [
    {"type": "motion_blur", "intensity": 2.0, "direction": 45},
    {"type": "scratches", "count": 3, "severity": 0.4},
    {"type": "stains", "stain_count": 2, "intensity": 0.5}
  ],
  "materials": [
    {"type": "transparent_overlay", "opacity": 0.8}
  ]
}
```

You can also provide a list of configs to cycle through:
```json
[
  {"damage": [{"type": "fading", "contrast_reduction": 0.2}]},
  {"damage": [{"type": "fading", "contrast_reduction": 0.4}]},
  {"damage": [{"type": "fading", "contrast_reduction": 0.6}]}
]
```

**Available transformation types for custom configs:**

| Category | Type | Key Parameters |
|----------|------|----------------|
| `geometry` | `y_axis_rotation` | `angle_degrees` |
| | `x_axis_rotation` | `angle_degrees` |
| | `z_axis_rotation` | `angle_degrees` |
| | `cylindrical` | `radius`, `axis`, `wrap_angle` |
| | `flexible_wrinkle` | `fold_count`, `depth`, `direction` |
| `damage` | `motion_blur` | `intensity`, `direction` |
| | `fading` | `contrast_reduction`, `pattern` |
| | `scratches` | `count`, `severity` |
| | `ink_bleeding` | `intensity`, `spread_radius` |
| | `broken_bars` | `intensity`, `streak_count` |
| | `low_ink` | `intensity`, `speckle_density` |
| | `white_noise` | `intensity`, `noise_density` |
| | `glare` | `intensity`, `size`, `light_direction` |
| | `water_droplets` | `intensity`, `droplet_count` |
| | `stains` | `intensity`, `stain_count`, `stain_type` |
| | `smudges` | `intensity`, `smudge_count`, `smudge_type` |
| | `partial_removal` | `coverage`, `removal_count` |
| | `low_light` | `darkness`, `uneven` |
| | `overexposure` | `brightness`, `clip_highlights` |
| `materials` | `metallic_reflection` | `specularity`, `roughness` |
| | `transparent_overlay` | `opacity`, `refraction_index` |

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
