"""
Testplan format output handler for decoder testing.

Generates sidecar JSON files with comprehensive barcode metadata
for testing barcode decoder accuracy and robustness.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import OutputFormat, AnnotationData


class TestplanFormat(OutputFormat):
    """Testplan format for barcode decoder testing.

    Generates one JSON sidecar file per image with comprehensive metadata
    including region polygons, encoded values, and orientation information.

    Output structure (flat, default):
        output_folder/
        ├── manifest.json
        ├── image_001.png
        ├── image_001.json
        ├── image_002.png
        └── image_002.json

    Output structure (with splits):
        output_folder/
        ├── manifest.json
        ├── train/
        │   ├── image_001.png
        │   └── image_001.json
        ├── val/
        └── test/
    """

    name = "testplan"
    description = "Decoder testing format with sidecar JSON annotations"
    file_extension = ".json"

    # JSON schema version for forward compatibility
    SCHEMA_VERSION = "1.0.0"

    supports_detection = True
    supports_segmentation = True
    supports_classification = True
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
        """Create testplan directory structure."""
        self.enable_split = enable_split
        self.output_folder.mkdir(parents=True, exist_ok=True)

        if self.enable_split:
            for split in ["train", "val", "test"]:
                (self.output_folder / split).mkdir(parents=True, exist_ok=True)
        # else: single flat folder (default)

    def save_annotation(
        self,
        data: AnnotationData,
        split: str,
        task: str
    ) -> Path:
        """Save testplan format annotation (sidecar JSON)."""
        # Determine output directory
        if self.enable_split and split:
            output_dir = self.output_folder / split
        else:
            output_dir = self.output_folder

        # Save image
        img_path = output_dir / data.image_filename
        data.image.save(img_path)

        # Create sidecar JSON (same name, .json extension)
        json_filename = Path(data.image_filename).stem + ".json"
        json_path = output_dir / json_filename

        # Build annotation
        annotation = self._build_annotation(data, task)

        with open(json_path, "w") as f:
            json.dump(annotation, f, indent=2)

        # Track for manifest
        self.generated_files.append({
            "image": str(img_path.relative_to(self.output_folder)),
            "annotation": str(json_path.relative_to(self.output_folder)),
            "split": split if self.enable_split else None,
            "symbology": data.symbology,
        })

        return json_path

    def _build_annotation(self, data: AnnotationData, task: str) -> Dict[str, Any]:
        """Build testplan annotation structure."""
        annotation: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "image": {
                "filename": data.image_filename,
                "width": data.image_size[0],
                "height": data.image_size[1],
            },
            "barcode": {
                "symbology": data.symbology,
                "encoded_value": data.encoded_value,
                "printed_text": data.printed_text,
                "orientation": data.orientation,
            },
            "regions": {},
            "generation": {
                "degradation_applied": data.degradation_applied,
                "transformations": data.transformations or [],
            },
        }

        # Add region polygons
        if data.full_region_polygon:
            annotation["regions"]["full"] = self._build_region(
                data.full_region_polygon,
                "Complete barcode area including quiet zones",
            )

        if data.barcode_only_polygon:
            region = self._build_region(
                data.barcode_only_polygon,
                "Barcode modules only (no quiet zones or text)",
            )
            # Add bounding box for convenience
            if data.barcode_only_bbox:
                region["bbox"] = data.barcode_only_bbox
            else:
                region["bbox"] = self.bbox_from_polygon(data.barcode_only_polygon)
            annotation["regions"]["barcode_only"] = region

        if data.text_region_polygon:
            annotation["regions"]["text_label"] = self._build_region(
                data.text_region_polygon,
                "Human-readable text region below barcode",
            )

        return annotation

    def _build_region(
        self,
        polygon: List[List[int]],
        description: str
    ) -> Dict[str, Any]:
        """Build a region object for the annotation."""
        return {
            "description": description,
            "polygon": polygon,
            "vertex_count": len(polygon),
        }

    def finalize(self, task: str, stats: Dict[str, Any]) -> None:
        """Write manifest.json with dataset summary."""
        manifest: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "format": "testplan",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dataset": {
                "total_images": stats.get("generated", 0),
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
