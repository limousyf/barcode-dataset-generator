"""Tests for testplan format handler."""

import json
import pytest
from pathlib import Path
from PIL import Image

from src.formats.testplan import TestplanFormat
from src.formats.base import AnnotationData


class TestTestplanFormat:
    """Tests for TestplanFormat handler."""

    def test_setup_directories_no_split(self, tmp_path):
        """Test flat directory structure without splits."""
        handler = TestplanFormat(tmp_path)
        handler.setup_directories("detection", ["code128", "qr"], enable_split=False)

        assert tmp_path.exists()
        assert (tmp_path / "images").exists()
        assert (tmp_path / "labels").exists()
        assert not (tmp_path / "train").exists()
        assert not (tmp_path / "val").exists()

    def test_setup_directories_with_split(self, tmp_path):
        """Test directory structure with train/val/test splits."""
        handler = TestplanFormat(tmp_path)
        handler.setup_directories("detection", ["code128", "qr"], enable_split=True)

        for split in ["train", "val", "test"]:
            assert (tmp_path / split / "images").exists()
            assert (tmp_path / split / "labels").exists()

    def test_save_annotation_creates_sidecar_json(self, tmp_path):
        """Test that save_annotation creates image and JSON sidecar."""
        handler = TestplanFormat(tmp_path)
        handler.setup_directories("detection", ["code128"], enable_split=False)

        # Create test image
        img = Image.new('RGB', (400, 150), color='white')

        # Create annotation data
        data = AnnotationData(
            image=img,
            image_filename="code128_000001.png",
            image_size=(400, 150),
            symbology="code128",
            encoded_value="ABC123",
            printed_text="ABC123",
            barcode_only_polygon=[[10, 5], [390, 5], [390, 120], [10, 120]],
            barcode_only_bbox=[10, 5, 390, 120],
            text_region_polygon=[[10, 125], [390, 125], [390, 145], [10, 145]],
            full_region_polygon=[[0, 0], [400, 0], [400, 150], [0, 150]],
            orientation="top-left",
            class_id=0,
            class_name="code128",
            degradation_applied=False,
            transformations=[],
        )

        # Save annotation
        json_path = handler.save_annotation(data, "", "detection")

        # Verify files exist in separate folders
        assert (tmp_path / "images" / "code128_000001.png").exists()
        assert (tmp_path / "labels" / "code128_000001.json").exists()

        # Verify JSON content
        with open(json_path) as f:
            annotation = json.load(f)

        assert annotation["schema_version"] == "1.0.0"
        assert annotation["barcode"]["symbology"] == "code128"
        assert annotation["barcode"]["encoded_value"] == "ABC123"
        assert annotation["barcode"]["printed_text"] == "ABC123"
        assert annotation["barcode"]["orientation"] == "top-left"

        assert "barcode_only" in annotation["regions"]
        assert annotation["regions"]["barcode_only"]["polygon"] == [
            [10, 5], [390, 5], [390, 120], [10, 120]
        ]
        assert annotation["regions"]["barcode_only"]["bbox"] == [10, 5, 390, 120]

        assert "text_label" in annotation["regions"]
        assert "full" in annotation["regions"]

    def test_finalize_creates_manifest(self, tmp_path):
        """Test that finalize creates manifest.json."""
        handler = TestplanFormat(tmp_path)
        handler.setup_directories("detection", ["code128"], enable_split=False)

        # Simulate some generated files
        handler.generated_files = [
            {"image": "code128_000001.png", "annotation": "code128_000001.json",
             "split": None, "symbology": "code128"},
            {"image": "code128_000002.png", "annotation": "code128_000002.json",
             "split": None, "symbology": "code128"},
        ]

        stats = {
            "generated": 2,
            "failed": 0,
            "by_symbology": {"code128": {"generated": 2, "failed": 0}},
        }

        handler.finalize("detection", stats)

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["format"] == "testplan"
        assert manifest["dataset"]["total_images"] == 2
        assert manifest["dataset"]["failed"] == 0
        assert manifest["dataset"]["splits_enabled"] is False
        assert len(manifest["files"]) == 2
        assert "code128" in manifest["symbologies"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
