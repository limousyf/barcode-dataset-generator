"""
Paired format output handler for image restoration training.

Generates paired degraded/sharp images for supervised training of
image restoration models (deblurring, denoising, etc.).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import OutputFormat, AnnotationData


class PairedFormat(OutputFormat):
    """Paired format for image restoration training.

    Generates matching pairs of degraded (input) and sharp (target) images
    along with metadata JSON files describing the degradation applied.

    Output structure (flat, default):
        output_folder/
        ├── manifest.json
        ├── input/
        │   └── code128_000001.png
        ├── target/
        │   └── code128_000001.png
        └── metadata/
            └── code128_000001.json

    Output structure (with splits):
        output_folder/
        ├── manifest.json
        ├── input/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── target/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── metadata/
            ├── train/
            ├── val/
            └── test/
    """

    name = "paired"
    description = "Paired degraded/sharp images for restoration training"
    file_extension = ".json"

    # JSON schema version for forward compatibility
    SCHEMA_VERSION = "1.0.0"

    # Paired format doesn't use traditional detection/segmentation tasks
    supports_detection = False
    supports_segmentation = False
    supports_classification = False
    supports_split = True

    def __init__(self, output_folder: Path, config: Optional[Dict] = None):
        super().__init__(output_folder, config)
        self.enable_split = False
        self.generated_files: List[Dict] = []  # Track for manifest

    def setup_directories(
        self,
        task: str,
        class_names: List[str],
        enable_split: bool = True
    ) -> None:
        """Create paired directory structure with input/target/metadata folders."""
        self.enable_split = enable_split
        self.output_folder.mkdir(parents=True, exist_ok=True)

        if self.enable_split:
            for split in ["train", "val", "test"]:
                (self.output_folder / "input" / split).mkdir(parents=True, exist_ok=True)
                (self.output_folder / "target" / split).mkdir(parents=True, exist_ok=True)
                (self.output_folder / "metadata" / split).mkdir(parents=True, exist_ok=True)
        else:
            # Flat structure
            (self.output_folder / "input").mkdir(parents=True, exist_ok=True)
            (self.output_folder / "target").mkdir(parents=True, exist_ok=True)
            (self.output_folder / "metadata").mkdir(parents=True, exist_ok=True)

    def save_annotation(
        self,
        data: AnnotationData,
        split: str,
        task: str
    ) -> Path:
        """Save paired images and metadata."""
        # Determine output directories
        if self.enable_split and split:
            input_dir = self.output_folder / "input" / split
            target_dir = self.output_folder / "target" / split
            metadata_dir = self.output_folder / "metadata" / split
        else:
            input_dir = self.output_folder / "input"
            target_dir = self.output_folder / "target"
            metadata_dir = self.output_folder / "metadata"

        # Save degraded image (input)
        input_path = input_dir / data.image_filename
        data.image.save(input_path)

        # Save sharp image (target)
        target_path = target_dir / data.image_filename
        if data.target_image:
            data.target_image.save(target_path)
        else:
            # Fallback: save same image if no target provided
            data.image.save(target_path)

        # Create metadata JSON
        json_filename = Path(data.image_filename).stem + ".json"
        json_path = metadata_dir / json_filename

        metadata = self._build_metadata(data, input_path, target_path)

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Track for manifest
        self.generated_files.append({
            "input": str(input_path.relative_to(self.output_folder)),
            "target": str(target_path.relative_to(self.output_folder)),
            "metadata": str(json_path.relative_to(self.output_folder)),
            "split": split if self.enable_split else None,
            "symbology": data.symbology,
        })

        return json_path

    def _build_metadata(
        self,
        data: AnnotationData,
        input_path: Path,
        target_path: Path
    ) -> Dict[str, Any]:
        """Build metadata JSON structure for a paired sample."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "image": {
                "input_filename": str(input_path.relative_to(self.output_folder)),
                "target_filename": str(target_path.relative_to(self.output_folder)),
                "width": data.image_size[0],
                "height": data.image_size[1],
            },
            "barcode": {
                "symbology": data.symbology,
                "encoded_value": data.encoded_value,
            },
            "degradation": {
                "applied": data.degradation_applied,
                "transformations": data.transformations or [],
            },
        }

    def finalize(self, task: str, stats: Dict[str, Any]) -> None:
        """Write manifest.json with dataset summary."""
        manifest: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "format": "paired",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dataset": {
                "total_pairs": stats.get("generated", 0),
                "failed": stats.get("failed", 0),
                "splits_enabled": self.enable_split,
            },
            "symbologies": list(stats.get("by_symbology", {}).keys()),
            "files": self.generated_files,
        }

        # Add split counts if enabled
        if self.enable_split:
            manifest["dataset"]["splits"] = {
                "train": stats.get("train", 0),
                "val": stats.get("val", 0),
                "test": stats.get("test", 0),
            }

        with open(self.output_folder / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
