"""Tests for label generator."""

import pytest

from src.label_generator import LabelGenerator


class TestLabelGenerator:
    """Tests for LabelGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = LabelGenerator()

    def test_generate_detection_label(self):
        """Test detection label generation."""
        bbox = (100, 50, 300, 150)  # x_min, y_min, x_max, y_max
        image_size = (640, 480)  # width, height

        label = self.generator.generate_detection_label(
            class_id=0,
            bbox=bbox,
            image_size=image_size
        )

        # Parse label
        parts = label.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class_id

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Check normalized values
        assert 0 <= x_center <= 1
        assert 0 <= y_center <= 1
        assert 0 <= width <= 1
        assert 0 <= height <= 1

        # Check calculations
        expected_x_center = (100 + 300) / 2 / 640
        expected_y_center = (50 + 150) / 2 / 480
        expected_width = (300 - 100) / 640
        expected_height = (150 - 50) / 480

        assert abs(x_center - expected_x_center) < 0.0001
        assert abs(y_center - expected_y_center) < 0.0001
        assert abs(width - expected_width) < 0.0001
        assert abs(height - expected_height) < 0.0001

    def test_generate_segmentation_label(self):
        """Test segmentation label generation."""
        polygon = [(10, 10), (100, 10), (100, 50), (10, 50)]
        image_size = (200, 100)

        label = self.generator.generate_segmentation_label(
            class_id=1,
            polygon=polygon,
            image_size=image_size
        )

        parts = label.split()
        assert parts[0] == "1"  # class_id

        # Should have class_id + 8 coordinate values (4 points x 2)
        assert len(parts) == 9

        # Check normalized coordinates
        coords = [float(p) for p in parts[1:]]
        for coord in coords:
            assert 0 <= coord <= 1

    def test_generate_segmentation_label_clamps_values(self):
        """Test that segmentation labels clamp out-of-bounds coordinates."""
        # Polygon with points outside image bounds
        polygon = [(-10, -10), (250, -10), (250, 150), (-10, 150)]
        image_size = (200, 100)

        label = self.generator.generate_segmentation_label(
            class_id=0,
            polygon=polygon,
            image_size=image_size
        )

        parts = label.split()
        coords = [float(p) for p in parts[1:]]

        # All coordinates should be clamped to [0, 1]
        for coord in coords:
            assert 0 <= coord <= 1

    def test_bbox_from_polygon(self):
        """Test bounding box derivation from polygon."""
        polygon = [(10, 20), (100, 15), (95, 80), (15, 85)]

        bbox = self.generator.bbox_from_polygon(polygon)

        assert bbox == (10, 15, 100, 85)

    def test_bbox_from_polygon_single_point(self):
        """Test bbox from degenerate polygon."""
        polygon = [(50, 50)]

        bbox = self.generator.bbox_from_polygon(polygon)

        assert bbox == (50, 50, 50, 50)

    def test_detection_label_with_different_classes(self):
        """Test detection labels with various class IDs."""
        bbox = (0, 0, 100, 100)
        image_size = (100, 100)

        for class_id in [0, 5, 10, 99]:
            label = self.generator.generate_detection_label(
                class_id=class_id,
                bbox=bbox,
                image_size=image_size
            )
            assert label.startswith(f"{class_id} ")

    def test_detection_label_precision(self):
        """Test that labels have proper decimal precision."""
        bbox = (33, 33, 66, 66)
        image_size = (100, 100)

        label = self.generator.generate_detection_label(
            class_id=0,
            bbox=bbox,
            image_size=image_size
        )

        parts = label.split()
        # Check that coordinates have 6 decimal places
        for part in parts[1:]:
            assert "." in part
            decimals = len(part.split(".")[1])
            assert decimals == 6
