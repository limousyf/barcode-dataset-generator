"""
Background image management for composite dataset generation.

Handles loading, caching, and placement of barcodes on background images.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


class BackgroundManager:
    """Manages background images for barcode embedding."""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, backgrounds_folder: Optional[Path] = None):
        """
        Initialize background manager.

        Args:
            backgrounds_folder: Path to folder containing background images
        """
        self.backgrounds_folder = Path(backgrounds_folder) if backgrounds_folder else None
        self.background_files: List[Path] = []

        if self.backgrounds_folder:
            self._load_background_list()

    def _load_background_list(self) -> None:
        """Scan folder recursively and load list of background images."""
        if not self.backgrounds_folder or not self.backgrounds_folder.exists():
            return

        # Search recursively for all image files
        self.background_files = [
            f for f in self.backgrounds_folder.rglob('*')
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not self.background_files:
            raise ValueError(f"No background images found in {self.backgrounds_folder} (searched recursively)")

    def get_random_background(
        self,
        size: Tuple[int, int] = (640, 640)
    ) -> Image.Image:
        """
        Get a random background image, resized to target size.

        Args:
            size: Target size (width, height)

        Returns:
            PIL Image of the background
        """
        if not self.background_files:
            # Return solid color if no backgrounds available
            return Image.new('RGB', size, (255, 255, 255))

        bg_path = random.choice(self.background_files)
        bg = Image.open(bg_path).convert('RGB')

        # Resize to cover target size, then crop
        bg = self._resize_and_crop(bg, size)

        return bg

    def _resize_and_crop(
        self,
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize image to cover target size, then center crop."""
        target_w, target_h = target_size
        orig_w, orig_h = image.size

        # Calculate scale to cover target
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        image = image.crop((left, top, left + target_w, top + target_h))

        return image

    def place_barcode(
        self,
        background: Image.Image,
        barcode: Image.Image,
        existing_bboxes: List[Tuple[int, int, int, int]],
        max_attempts: int = 50
    ) -> Optional[Tuple[int, int]]:
        """
        Find position to place barcode without overlapping existing barcodes.

        Args:
            background: Background image
            barcode: Barcode image to place
            existing_bboxes: List of existing bounding boxes
            max_attempts: Maximum placement attempts

        Returns:
            (x, y) position or None if placement failed
        """
        bg_w, bg_h = background.size
        bc_w, bc_h = barcode.size
        margin = 20

        for _ in range(max_attempts):
            x = random.randint(margin, max(margin, bg_w - bc_w - margin))
            y = random.randint(margin, max(margin, bg_h - bc_h - margin))

            new_bbox = (x, y, x + bc_w, y + bc_h)

            if not self._check_overlap(new_bbox, existing_bboxes):
                return (x, y)

        return None

    def _check_overlap(
        self,
        bbox: Tuple[int, int, int, int],
        existing: List[Tuple[int, int, int, int]]
    ) -> bool:
        """Check if bbox overlaps with any existing bbox."""
        x1, y1, x2, y2 = bbox

        for ex1, ey1, ex2, ey2 in existing:
            if not (x2 < ex1 or x1 > ex2 or y2 < ey1 or y1 > ey2):
                return True

        return False
