# Paired Format Guide - Image Restoration Training

Generate paired datasets of degraded/sharp barcode images for training image restoration models (deblurring, denoising, super-resolution).

## Overview

The **paired format** creates matching pairs of:
- **Input images**: Degraded barcodes (blur, noise, distortion, etc.)
- **Target images**: Clean/sharp barcodes (ground truth)

This is the standard format for supervised training of image restoration networks like U-Net, ESRGAN, NAFNet, and other encoder-decoder architectures.

### Use Cases

- **Deblurring models**: Train networks to remove motion blur from barcode images
- **Denoising models**: Train networks to clean noisy/grainy barcode scans
- **Super-resolution**: Train networks to enhance low-quality barcode images
- **General restoration**: Train multi-purpose restoration models for barcode recovery
- **Domain-specific fine-tuning**: Adapt general restoration models to barcode images

## Quick Start

```bash
# Generate 500 paired samples with motion blur (intensity 0.5-10)
python -m src.dataset_generator \
    --output ./paired-Code128-blur-500 \
    --samples 500 \
    --symbologies code128 \
    --output-format paired \
    --degrade-sweep blur 0.5 10.0 20 \
    --split 80/10/10

# Generate paired samples with random degradation
python -m src.dataset_generator \
    --output ./paired-mixed-degradation \
    --samples 200 \
    --symbologies code128 qr datamatrix \
    --output-format paired \
    --degrade --degrade-prob 1.0 \
    --split 80/10/10
```

## Requirements

**Degradation is required** for the paired format. You must use one of:

| Option | Description |
|--------|-------------|
| `--degrade` | Enable random degradation effects |
| `--degrade-sweep TYPE MIN MAX STEPS` | Systematic parameter sweep |
| `--degrade-preset NAME` | Use a predefined degradation preset |
| `--degrade-config FILE` | Load custom degradation from JSON file |

If you run `--output-format paired` without any degradation option, you'll get an error:

```
Error: --output-format paired requires degradation to be enabled
Use --degrade, --degrade-sweep, --degrade-config, or --degrade-preset
```

## Output Structure

### With Train/Val/Test Splits (default)

```
paired-Code128-blur-500/
├── manifest.json           # Dataset summary and file listing
├── input/                  # Degraded images (model input)
│   ├── train/
│   │   ├── code128_000001.png
│   │   ├── code128_000002.png
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── target/                 # Sharp images (ground truth)
│   ├── train/
│   │   ├── code128_000001.png
│   │   ├── code128_000002.png
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── metadata/               # JSON sidecar files with degradation info
    ├── train/
    │   ├── code128_000001.json
    │   ├── code128_000002.json
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

### Flat Structure (with `--no-split`)

```
paired-Code128-blur-500/
├── manifest.json
├── input/
│   ├── code128_000001.png
│   ├── code128_000002.png
│   └── ...
├── target/
│   ├── code128_000001.png
│   ├── code128_000002.png
│   └── ...
└── metadata/
    ├── code128_000001.json
    ├── code128_000002.json
    └── ...
```

## Metadata JSON Schema

Each image pair has a corresponding JSON sidecar file in the `metadata/` directory containing information about the degradation applied.

### Example: `metadata/train/code128_000001.json`

```json
{
  "schema_version": "1.0.0",
  "image": {
    "input_filename": "input/train/code128_000001.png",
    "target_filename": "target/train/code128_000001.png",
    "width": 400,
    "height": 150
  },
  "barcode": {
    "symbology": "code128",
    "encoded_value": "ABC-12345-XYZ"
  },
  "degradation": {
    "applied": true,
    "transformations": [
      {
        "type": "motion_blur",
        "parameters": {
          "intensity": 2.5,
          "direction": 45
        }
      }
    ]
  }
}
```

### Schema Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | JSON schema version for compatibility |
| `image.input_filename` | string | Relative path to degraded image |
| `image.target_filename` | string | Relative path to sharp image |
| `image.width` | int | Image width in pixels |
| `image.height` | int | Image height in pixels |
| `barcode.symbology` | string | Barcode type (e.g., "code128", "qr") |
| `barcode.encoded_value` | string | Data encoded in the barcode |
| `degradation.applied` | bool | Whether degradation was applied |
| `degradation.transformations` | array | List of transformations with parameters |

## Manifest File

The `manifest.json` at the dataset root provides a summary and complete file listing:

```json
{
  "schema_version": "1.0.0",
  "format": "paired",
  "created_at": "2024-12-23T10:30:00Z",
  "dataset": {
    "total_pairs": 500,
    "failed": 0,
    "splits_enabled": true,
    "splits": {
      "train": 400,
      "val": 50,
      "test": 50
    }
  },
  "symbologies": ["code128"],
  "files": [
    {
      "input": "input/train/code128_000001.png",
      "target": "target/train/code128_000001.png",
      "metadata": "metadata/train/code128_000001.json",
      "split": "train",
      "symbology": "code128"
    }
  ]
}
```

## Usage Examples

### Deblurring Dataset with Blur Sweep

Generate a dataset with systematically increasing blur intensity for training deblurring models:

```bash
python -m src.dataset_generator \
    --output ./deblur-dataset \
    --samples 1000 \
    --symbologies code128 \
    --output-format paired \
    --degrade-sweep blur 0.5 10.0 20 \
    --split 80/10/10
```

This creates samples with blur intensity values: 0.5, 1.0, 1.5, 2.0, ..., 10.0 (20 steps).

### Multi-Symbology Restoration Dataset

Train a model that works across different barcode types:

```bash
python -m src.dataset_generator \
    --output ./multi-barcode-restoration \
    --samples 200 \
    --categories popular \
    --output-format paired \
    --degrade --degrade-prob 1.0 \
    --split 80/10/10
```

### Specific Degradation Types

#### Motion Blur Only
```bash
python -m src.dataset_generator \
    --output ./paired-blur \
    --samples 500 \
    --symbologies code128 \
    --output-format paired \
    --degrade-sweep blur 1.0 8.0 16
```

#### Noise Only
```bash
python -m src.dataset_generator \
    --output ./paired-noise \
    --samples 500 \
    --symbologies code128 \
    --output-format paired \
    --degrade-sweep white_noise 0.3 0.8 10
```

#### Low Light Conditions
```bash
python -m src.dataset_generator \
    --output ./paired-lowlight \
    --samples 500 \
    --symbologies code128 \
    --output-format paired \
    --degrade-sweep low_light 0.2 0.7 10
```

#### Custom Degradation Config
```bash
python -m src.dataset_generator \
    --output ./paired-custom \
    --samples 200 \
    --symbologies code128 qr \
    --output-format paired \
    --degrade-config my-degradation.json
```

Example `my-degradation.json`:
```json
{
  "damage": [
    {"type": "motion_blur", "intensity": 3.0, "direction": 45},
    {"type": "white_noise", "intensity": 0.3}
  ]
}
```

### With Background Images

Embed barcodes on realistic backgrounds for more challenging restoration:

```bash
python -m src.dataset_generator \
    --output ./paired-with-backgrounds \
    --samples 300 \
    --symbologies code128 qr \
    --output-format paired \
    --backgrounds ~/ml-backgrounds \
    --barcode-scale 0.1-0.3 \
    --degrade-sweep blur 1.0 6.0 12 \
    --split 80/10/10
```

### Flat Structure (No Splits)

For custom train/val splitting or when using all data:

```bash
python -m src.dataset_generator \
    --output ./paired-flat \
    --samples 100 \
    --symbologies code128 \
    --output-format paired \
    --degrade-sweep blur 1.0 5.0 10 \
    --no-split
```

## Loading in PyTorch

Example PyTorch Dataset class for loading paired data:

```python
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PairedBarcodeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform or transforms.ToTensor()

        # Load manifest
        with open(self.root / 'manifest.json') as f:
            manifest = json.load(f)

        # Filter files by split
        self.pairs = [
            f for f in manifest['files']
            if f['split'] == split
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load images
        input_img = Image.open(self.root / pair['input']).convert('RGB')
        target_img = Image.open(self.root / pair['target']).convert('RGB')

        # Load metadata
        with open(self.root / pair['metadata']) as f:
            metadata = json.load(f)

        # Apply transforms
        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)

        return {
            'input': input_tensor,
            'target': target_tensor,
            'metadata': metadata
        }

# Usage
train_dataset = PairedBarcodeDataset('./paired-Code128-blur-500', split='train')
val_dataset = PairedBarcodeDataset('./paired-Code128-blur-500', split='val')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

## Training Tips

### Loss Functions

For barcode restoration, consider these loss functions:

```python
import torch.nn as nn
import torch.nn.functional as F

# L1 Loss - good for sharp edges in barcodes
l1_loss = nn.L1Loss()

# SSIM Loss - preserves structural patterns
from pytorch_msssim import ssim
ssim_loss = lambda pred, target: 1 - ssim(pred, target, data_range=1.0)

# Combined loss
def restoration_loss(pred, target):
    l1 = l1_loss(pred, target)
    structural = ssim_loss(pred, target)
    return l1 + 0.1 * structural
```

### Data Augmentation

Augment during training (apply same transform to both input and target):

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Paired augmentation (same transform for input and target)
def paired_augment(input_img, target_img):
    transform = A.Compose([
        A.RandomCrop(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, p=0.3),
    ])

    # Use same random seed for both
    result_input = transform(image=input_img)
    result_target = transform(image=target_img, replay=result_input['replay'])

    return result_input['image'], result_target['image']
```

### Model Architectures

Recommended architectures for barcode restoration:

| Architecture | Strengths | Training Time |
|-------------|-----------|---------------|
| **U-Net** | Simple, effective for localized degradation | Fast |
| **NAFNet** | State-of-art, efficient | Medium |
| **Restormer** | Excellent for global degradation | Slow |
| **ESRGAN** | Good for super-resolution | Medium |

## Available Degradation Types for Sweeps

These sweep types work well with the paired format:

| Category | Type | Description | Typical Range |
|----------|------|-------------|---------------|
| **Blur** | `blur` | Motion blur | 0.5 - 10.0 |
| **Noise** | `white_noise` | Random noise | 0.3 - 0.8 |
| **Lighting** | `low_light` | Darkness | 0.2 - 0.7 |
| | `overexposure` | Brightness | 0.2 - 0.7 |
| | `glare` | Hotspots | 0.2 - 0.8 |
| **Damage** | `fading` | Contrast loss | 0.1 - 0.5 |
| | `scratches` | Line damage | 0.2 - 0.8 |
| | `stains` | Dirt marks | 0.2 - 0.8 |
| **Geometry** | `rotation_z` | In-plane rotation | -15 - 15 |
| | `wrinkle` | Surface deformation | 0.1 - 0.4 |

See the main [README](../README.md) for the complete list of degradation options.

## Naming Convention

Recommended naming format for paired datasets:

```
paired-{Symbology}-{DegradationType}-{SampleCount}
```

Examples:
- `paired-Code128-blur-500`
- `paired-QR-noise-1000`
- `paired-Mixed-random-2000`
- `paired-Popular-lowlight-300`

## Comparison with Other Formats

| Feature | Paired | YOLO | Testplan |
|---------|--------|------|----------|
| **Purpose** | Image restoration training | Object detection/segmentation | Decoder testing |
| **Outputs per sample** | 2 images (input + target) | 1 image + label | 1 image + JSON |
| **Degradation required** | Yes | Optional | Optional |
| **Bounding boxes** | No | Yes | Yes |
| **Polygons** | No | Yes (segmentation) | Yes |
| **Train/Val/Test splits** | Yes | Yes | Yes |

## Troubleshooting

### "paired requires degradation to be enabled"

You must specify a degradation method:

```bash
# Add one of these options:
--degrade                          # Random degradation
--degrade-sweep blur 1.0 5.0 10    # Parameter sweep
--degrade-preset moderate          # API preset
--degrade-config config.json       # Custom config
```

### Images look identical in input/target

Check that degradation is being applied:
1. Verify `--degrade-prob` is not 0 (default is 0.5)
2. Use `--degrade-prob 1.0` to ensure all samples are degraded
3. Check the metadata JSON files for `"degradation.applied": true`

### Out of memory with large datasets

For large datasets, process in batches:

```bash
# Generate 10k samples in batches of 1000
for i in {1..10}; do
    python -m src.dataset_generator \
        --output "./batch-$i" \
        --samples 1000 \
        --output-format paired \
        --degrade-sweep blur 1.0 5.0 20
done
```

Then merge the batches programmatically.

## Version History

- **v1.0.0** (2024-12): Initial paired format release
  - Input/target/metadata structure
  - JSON sidecar files with degradation info
  - Manifest file with dataset summary
  - Train/val/test split support
