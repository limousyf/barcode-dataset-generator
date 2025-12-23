"""
Abstract base class for output format handlers.

This module defines the interface that all output format handlers must implement,
enabling support for multiple annotation formats (YOLO, testplan, COCO, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


@dataclass
class AnnotationData:
    """Container for annotation data from API result.

    This dataclass holds all the information needed by format handlers
    to generate annotations in their respective formats.
    """

    # Image data
    image: Image.Image
    image_filename: str
    image_size: Tuple[int, int]

    # Barcode identification
    symbology: str
    encoded_value: str  # The data encoded in barcode
    printed_text: Optional[str] = None  # Human-readable text (may differ or be None)

    # Region polygons (4 vertices each as [[x,y], ...])
    barcode_only_polygon: Optional[List[List[int]]] = None
    barcode_only_bbox: Optional[List[int]] = None  # [x_min, y_min, x_max, y_max]
    text_region_polygon: Optional[List[List[int]]] = None
    full_region_polygon: Optional[List[List[int]]] = None  # Includes quiet zones

    # Orientation: which corner is top-left after any rotation
    orientation: str = "top-left"

    # Classification info
    class_id: Optional[int] = None
    class_name: Optional[str] = None

    # Generation metadata
    degradation_applied: bool = False
    transformations: List[str] = field(default_factory=list)

    # Paired format: sharp version of the image for restoration training
    target_image: Optional[Image.Image] = None


class OutputFormat(ABC):
    """Abstract base class for output format handlers.

    Subclasses implement specific output formats (YOLO, testplan, COCO, etc.)
    by overriding the abstract methods.
    """

    # Format metadata (override in subclasses)
    name: str = "base"
    description: str = "Base format"
    file_extension: str = ".txt"

    # Capabilities (override in subclasses)
    supports_detection: bool = True
    supports_segmentation: bool = True
    supports_classification: bool = True
    supports_split: bool = True  # train/val/test splitting

    def __init__(self, output_folder: Path, config: Optional[Dict] = None):
        """Initialize format handler.

        Args:
            output_folder: Root directory for output files
            config: Optional format-specific configuration
        """
        self.output_folder = Path(output_folder)
        self.config = config or {}
        self.class_mapping: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

    def set_class_mapping(self, class_to_id: Dict[str, int]) -> None:
        """Set class ID mapping.

        Args:
            class_to_id: Mapping from class names to integer IDs
        """
        self.class_mapping = class_to_id
        self.id_to_class = {v: k for k, v in class_to_id.items()}

    @abstractmethod
    def setup_directories(
        self,
        task: str,
        class_names: List[str],
        enable_split: bool = True
    ) -> None:
        """Create output directory structure.

        Args:
            task: Task type (detection, segmentation, classification)
            class_names: List of class names for classification folders
            enable_split: Whether to create train/val/test subdirectories
        """
        pass

    @abstractmethod
    def save_annotation(
        self,
        data: AnnotationData,
        split: str,
        task: str
    ) -> Path:
        """Save annotation for a single sample.

        Args:
            data: Annotation data container
            split: Split name (train, val, test, or empty string)
            task: Task type

        Returns:
            Path to saved annotation file
        """
        pass

    @abstractmethod
    def finalize(self, task: str, stats: Dict[str, Any]) -> None:
        """Called after all samples generated.

        Use this to write manifest files, configuration files,
        or any other dataset-level metadata.

        Args:
            task: Task type
            stats: Generation statistics dictionary
        """
        pass

    def validate_task(self, task: str) -> bool:
        """Check if this format supports the given task.

        Args:
            task: Task type to validate

        Returns:
            True if supported, False otherwise
        """
        if task == "detection":
            return self.supports_detection
        elif task == "segmentation":
            return self.supports_segmentation
        elif task == "classification":
            return self.supports_classification
        return False

    @staticmethod
    def bbox_from_polygon(polygon: List[List[int]]) -> List[int]:
        """Derive bounding box from polygon points.

        Args:
            polygon: List of [x, y] points

        Returns:
            Bounding box as [x_min, y_min, x_max, y_max]
        """
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return [min(xs), min(ys), max(xs), max(ys)]
