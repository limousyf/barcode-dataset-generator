"""
YOLO label/annotation generation.

Converts barcode metadata (polygons, bounding boxes) into YOLO format annotations.

.. deprecated::
    This module is deprecated in favor of the formats.yolo module.
    Use ``from src.formats import YOLOFormat`` instead.
"""

import warnings
from typing import List, Tuple


class LabelGenerator:
    """Generates YOLO format annotations from barcode metadata.

    .. deprecated::
        This class is deprecated. Use :class:`src.formats.yolo.YOLOFormat` instead.
    """

    def __init__(self):
        warnings.warn(
            "LabelGenerator is deprecated. Use src.formats.yolo.YOLOFormat instead.",
            DeprecationWarning,
            stacklevel=2
        )

    def generate_detection_label(
        self,
        class_id: int,
        bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, int]
    ) -> str:
        """
        Generate YOLO detection annotation.

        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All coordinates normalized to [0, 1].

        Args:
            class_id: Class index
            bbox: Bounding box (x_min, y_min, x_max, y_max) in pixels
            image_size: Image dimensions (width, height)

        Returns:
            YOLO format annotation string
        """
        x_min, y_min, x_max, y_max = bbox
        img_width, img_height = image_size

        # Calculate center and dimensions
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def generate_segmentation_label(
        self,
        class_id: int,
        polygon: List[Tuple[int, int]],
        image_size: Tuple[int, int]
    ) -> str:
        """
        Generate YOLO segmentation annotation.

        YOLO format: <class_id> <x1> <y1> <x2> <y2> ...
        All coordinates normalized to [0, 1].

        Args:
            class_id: Class index
            polygon: List of (x, y) points in pixels
            image_size: Image dimensions (width, height)

        Returns:
            YOLO format annotation string
        """
        img_width, img_height = image_size

        # Normalize polygon coordinates
        normalized_points = []
        for x, y in polygon:
            norm_x = x / img_width
            norm_y = y / img_height
            # Clamp to valid range
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            normalized_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

        return f"{class_id} " + " ".join(normalized_points)

    def bbox_from_polygon(
        self,
        polygon: List[Tuple[int, int]]
    ) -> Tuple[int, int, int, int]:
        """
        Derive bounding box from polygon points.

        Args:
            polygon: List of (x, y) points

        Returns:
            Bounding box (x_min, y_min, x_max, y_max)
        """
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))
