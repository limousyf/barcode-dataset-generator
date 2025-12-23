"""
Main dataset generation orchestrator.

This module coordinates barcode generation, degradation, placement,
and annotation creation to produce datasets in various formats.
"""

import argparse
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from .api_client import BarcodeAPIClient, BarcodeResult, APIError
from .formats import get_format, OutputFormat, AnnotationData
from .background_manager import BackgroundManager
from .config import Config
from .utils import (
    generate_sample_data,
    get_symbologies_for_category,
    get_symbologies_for_family,
    parse_split_ratio,
    parse_barcodes_per_image,
)

logger = logging.getLogger(__name__)


def load_degradation_config(config_path: str) -> List[Dict[str, Any]]:
    """Load degradation config from JSON file.

    The file can contain either a single config or a list of configs.
    """
    with open(config_path) as f:
        config = json.load(f)

    # Normalize to list
    if isinstance(config, list):
        return config
    return [config]


def build_sweep_configs(sweep_type: str, min_val: float, max_val: float, steps: int) -> List[Dict[str, Any]]:
    """Build degradation configs for parameter sweep.

    The API expects degradation configs with categories as lists of transformation objects.
    Each transformation must have a 'type' key matching the API schema.

    Supported sweep types:

    GEOMETRY:
    - rotation_y: Y-axis rotation angle (-180 to 180 degrees)
    - rotation_x: X-axis rotation angle (-90 to 90 degrees)
    - rotation_z: Z-axis rotation angle (-90 to 90 degrees)
    - cylindrical: Cylindrical warp radius (10-500)
    - wrinkle: Flexible wrinkle depth (0.05-0.5)

    DAMAGE:
    - blur/motion_blur: Motion blur intensity (0.5-10.0)
    - fading: Contrast reduction (0.1-0.9)
    - scratches: Scratch severity (0.1-1.0)
    - ink_bleeding: Ink bleed intensity (0.1-1.0)
    - broken_bars: Bar break intensity (0.5-1.0)
    - low_ink: Low ink effect intensity (0.3-1.0)
    - white_noise: Noise intensity (0.4-1.0)
    - glare: Glare/hotspot intensity (0.1-1.0)
    - water_droplets: Water droplet intensity (0.1-1.0)
    - stains: Stain intensity (0.1-1.0)
    - smudges: Smudge intensity (0.1-1.0)
    - partial_removal: Label removal coverage (0.1-0.6)
    - low_light: Darkness level (0.1-0.9)
    - overexposure: Brightness/washout level (0.1-0.9)

    MATERIALS:
    - metallic: Metallic reflection specularity (0.1-1.0)
    - transparent: Transparent overlay opacity (0.1-0.9)
    """
    configs = []
    step_size = (max_val - min_val) / max(steps - 1, 1) if steps > 1 else 0

    for i in range(steps):
        value = min_val + (i * step_size)

        # Build config in API's expected format
        # Categories must be LISTS of transformation objects with 'type' key

        # === GEOMETRY transforms ===
        if sweep_type == "rotation_y":
            config = {"geometry": [{"type": "y_axis_rotation", "angle_degrees": value}]}
        elif sweep_type == "rotation_x":
            config = {"geometry": [{"type": "x_axis_rotation", "angle_degrees": value}]}
        elif sweep_type == "rotation_z":
            config = {"geometry": [{"type": "z_axis_rotation", "angle_degrees": value}]}
        elif sweep_type == "cylindrical":
            config = {"geometry": [{"type": "cylindrical", "radius": value, "axis": "vertical", "wrap_angle": 180}]}
        elif sweep_type == "wrinkle":
            config = {"geometry": [{"type": "flexible_wrinkle", "depth": value, "fold_count": 3, "direction": "random"}]}

        # === DAMAGE transforms ===
        elif sweep_type in ("blur", "motion_blur"):
            config = {"damage": [{"type": "motion_blur", "intensity": value, "direction": 0}]}
        elif sweep_type == "fading":
            config = {"damage": [{"type": "fading", "contrast_reduction": value, "pattern": "uniform"}]}
        elif sweep_type == "scratches":
            config = {"damage": [{"type": "scratches", "severity": value, "count": 3}]}
        elif sweep_type == "ink_bleeding":
            config = {"damage": [{"type": "ink_bleeding", "intensity": value}]}
        elif sweep_type == "broken_bars":
            config = {"damage": [{"type": "broken_bars", "intensity": value}]}
        elif sweep_type == "low_ink":
            config = {"damage": [{"type": "low_ink", "intensity": value}]}
        elif sweep_type == "white_noise":
            config = {"damage": [{"type": "white_noise", "intensity": value}]}
        elif sweep_type == "glare":
            config = {"damage": [{"type": "glare", "intensity": value}]}
        elif sweep_type == "water_droplets":
            config = {"damage": [{"type": "water_droplets", "intensity": value}]}
        elif sweep_type == "stains":
            config = {"damage": [{"type": "stains", "intensity": value}]}
        elif sweep_type == "smudges":
            config = {"damage": [{"type": "smudges", "intensity": value}]}
        elif sweep_type == "partial_removal":
            config = {"damage": [{"type": "partial_removal", "coverage": value}]}
        elif sweep_type == "low_light":
            config = {"damage": [{"type": "low_light", "darkness": value}]}
        elif sweep_type == "overexposure":
            config = {"damage": [{"type": "overexposure", "brightness": value}]}

        # === MATERIAL transforms ===
        elif sweep_type == "metallic":
            config = {"materials": [{"type": "metallic_reflection", "specularity": value}]}
        elif sweep_type == "transparent":
            config = {"materials": [{"type": "transparent_overlay", "opacity": value}]}

        else:
            supported = (
                "rotation_y, rotation_x, rotation_z, cylindrical, wrinkle, "
                "blur, motion_blur, fading, scratches, ink_bleeding, broken_bars, low_ink, "
                "white_noise, glare, water_droplets, stains, smudges, partial_removal, "
                "low_light, overexposure, metallic, transparent"
            )
            raise ValueError(f"Unknown sweep type: {sweep_type}. Supported: {supported}")

        configs.append(config)

    return configs


def fetch_preset_config(client: "BarcodeAPIClient", preset_name: str) -> Dict[str, Any]:
    """Fetch degradation preset from API."""
    presets = client.get_degradation_presets()

    if not presets or "presets" not in presets:
        raise ValueError("Failed to fetch degradation presets from API")

    if preset_name not in presets["presets"]:
        available = ", ".join(presets["presets"].keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")

    return presets["presets"][preset_name]["degradation"]


class DatasetGenerator:
    """Generates datasets using barcodes.dev API in various formats."""

    def __init__(
        self,
        output_folder: Path,
        api_client: BarcodeAPIClient,
        config: Optional[Config] = None,
        output_format: str = "yolo"
    ):
        """
        Initialize dataset generator.

        Args:
            output_folder: Output directory for dataset
            api_client: Configured API client
            config: Optional configuration object
            output_format: Output format (yolo, testplan, etc.)
        """
        self.output_folder = Path(output_folder)
        self.api_client = api_client
        self.config = config or Config()
        self.output_format_name = output_format
        self.background_manager = None
        self.barcode_scale_range: Tuple[float, float] = (0.3, 0.6)  # Default 30-60% of bg width

        # Initialize format handler
        format_class = get_format(output_format)
        self.format_handler: OutputFormat = format_class(self.output_folder)

        # Class mapping for labels
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

    def set_backgrounds(self, backgrounds_folder: Optional[Path]) -> None:
        """Configure background image embedding."""
        if backgrounds_folder:
            self.background_manager = BackgroundManager(backgrounds_folder)

    def set_barcode_scale_range(self, scale_range: Tuple[float, float]) -> None:
        """Set the scale range for barcode size when embedding on backgrounds.

        Args:
            scale_range: Tuple of (min_scale, max_scale) as fraction of background width.
                        e.g., (0.1, 0.3) means 10%-30% of background width.
        """
        self.barcode_scale_range = scale_range

    def generate(
        self,
        symbologies: List[str],
        samples_per_class: int,
        task: str = "detection",
        label_mode: str = "symbology",
        enable_degradation: bool = False,
        degradation_prob: float = 0.5,
        degradation_configs: Optional[List[Dict]] = None,
        split_ratio: str = "80/10/10",
        image_format: str = "png",
        barcodes_per_image: str = "1",
        workers: int = 4,
        enable_split: bool = True
    ) -> Dict:
        """
        Generate complete dataset.

        Args:
            symbologies: List of symbology types to generate
            samples_per_class: Number of samples per symbology
            task: Task type (detection, segmentation, classification)
            label_mode: Label mode (symbology, category, family, binary)
            enable_degradation: Whether to apply degradation effects
            degradation_prob: Probability of degradation per sample
            degradation_configs: List of specific degradation configs to cycle through.
                If provided, overrides random degradation.
            split_ratio: Train/val/test split ratio (e.g., "80/10/10")
            image_format: Output image format (png, jpg)
            barcodes_per_image: Number of barcodes per image (e.g., "1" or "1-3")
            workers: Number of parallel workers
            enable_split: Whether to create train/val/test splits

        Returns:
            Statistics dictionary with generation results
        """
        # Store degradation configs for use in sample generation
        self._degradation_configs = degradation_configs
        self._degradation_index = 0
        # Build class mapping based on label mode
        self._build_class_mapping(symbologies, label_mode)

        # Setup format handler with class mapping
        self.format_handler.set_class_mapping(self.class_to_id)
        self.format_handler.setup_directories(
            task=task,
            class_names=list(self.id_to_class.values()),
            enable_split=enable_split
        )

        # Parse split ratio
        train_ratio, val_ratio, test_ratio = parse_split_ratio(split_ratio)

        # Parse barcodes per image
        min_barcodes, max_barcodes = parse_barcodes_per_image(barcodes_per_image)

        # Calculate total samples
        total_samples = len(symbologies) * samples_per_class

        # Statistics tracking
        stats = {
            "total_requested": total_samples,
            "generated": 0,
            "failed": 0,
            "train": 0,
            "val": 0,
            "test": 0,
            "by_symbology": {},
        }

        # Generate samples
        logger.info(f"Generating {total_samples} samples for {len(symbologies)} symbologies...")

        sample_idx = 0
        samples_to_generate = []

        # Build list of (symbology, sample_num) pairs
        for symbology in symbologies:
            stats["by_symbology"][symbology] = {"generated": 0, "failed": 0}
            for i in range(samples_per_class):
                samples_to_generate.append((symbology, i))

        # Shuffle for random split assignment
        random.shuffle(samples_to_generate)

        # Process with progress bar
        with tqdm(total=len(samples_to_generate), desc="Generating dataset") as pbar:
            if workers > 1:
                # Parallel generation
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {}
                    for idx, (symbology, sample_num) in enumerate(samples_to_generate):
                        # Determine split
                        split = self._get_split(idx, len(samples_to_generate),
                                              train_ratio, val_ratio)

                        future = executor.submit(
                            self._generate_and_save_sample,
                            symbology=symbology,
                            sample_idx=idx,
                            split=split,
                            task=task,
                            label_mode=label_mode,
                            enable_degradation=enable_degradation,
                            degradation_prob=degradation_prob,
                            image_format=image_format,
                            min_barcodes=min_barcodes,
                            max_barcodes=max_barcodes,
                        )
                        futures[future] = (symbology, sample_num)

                    for future in as_completed(futures):
                        symbology, sample_num = futures[future]
                        try:
                            success = future.result()
                            if success:
                                stats["generated"] += 1
                                stats["by_symbology"][symbology]["generated"] += 1
                                stats[success] += 1  # success is the split name
                            else:
                                stats["failed"] += 1
                                stats["by_symbology"][symbology]["failed"] += 1
                        except Exception as e:
                            logger.error(f"Error generating {symbology}: {e}")
                            stats["failed"] += 1
                            stats["by_symbology"][symbology]["failed"] += 1
                        pbar.update(1)
            else:
                # Sequential generation
                for idx, (symbology, sample_num) in enumerate(samples_to_generate):
                    split = self._get_split(idx, len(samples_to_generate),
                                          train_ratio, val_ratio)
                    try:
                        success = self._generate_and_save_sample(
                            symbology=symbology,
                            sample_idx=idx,
                            split=split,
                            task=task,
                            label_mode=label_mode,
                            enable_degradation=enable_degradation,
                            degradation_prob=degradation_prob,
                            image_format=image_format,
                            min_barcodes=min_barcodes,
                            max_barcodes=max_barcodes,
                        )
                        if success:
                            stats["generated"] += 1
                            stats["by_symbology"][symbology]["generated"] += 1
                            stats[success] += 1
                        else:
                            stats["failed"] += 1
                            stats["by_symbology"][symbology]["failed"] += 1
                    except Exception as e:
                        logger.error(f"Error generating {symbology}: {e}")
                        stats["failed"] += 1
                        stats["by_symbology"][symbology]["failed"] += 1
                    pbar.update(1)

        # Finalize dataset (write manifest/config files)
        self.format_handler.finalize(task, stats)

        logger.info(f"Dataset generation complete: {stats['generated']}/{stats['total_requested']} samples")
        return stats

    def _build_class_mapping(self, symbologies: List[str], label_mode: str) -> None:
        """Build class ID mapping based on label mode."""
        self.class_to_id = {}
        self.id_to_class = {}

        if label_mode == "binary":
            # Single class
            self.class_to_id["barcode"] = 0
            self.id_to_class[0] = "barcode"
        elif label_mode == "category":
            # Categories: linear, 2d, stacked, postal
            categories = ["linear", "2d", "stacked", "postal", "composite"]
            for idx, cat in enumerate(categories):
                self.class_to_id[cat] = idx
                self.id_to_class[idx] = cat
        elif label_mode == "family":
            # Use family groupings
            from .utils import SYMBOLOGY_FAMILIES
            families = sorted(SYMBOLOGY_FAMILIES.keys())
            for idx, family in enumerate(families):
                self.class_to_id[family] = idx
                self.id_to_class[idx] = family
        else:
            # Default: symbology mode - each symbology is its own class
            for idx, sym in enumerate(sorted(symbologies)):
                self.class_to_id[sym] = idx
                self.id_to_class[idx] = sym

    def _get_class_id(self, symbology: str, label_mode: str) -> int:
        """Get class ID for a symbology based on label mode."""
        if label_mode == "binary":
            return 0
        elif label_mode == "category":
            # Determine category from symbology
            from .utils import SYMBOLOGY_CATEGORIES
            for cat, syms in SYMBOLOGY_CATEGORIES.items():
                if symbology in syms:
                    return self.class_to_id.get(cat, 0)
            return 0  # Default
        elif label_mode == "family":
            from .utils import SYMBOLOGY_FAMILIES
            for family, syms in SYMBOLOGY_FAMILIES.items():
                if symbology in syms:
                    return self.class_to_id.get(family, 0)
            return 0
        else:
            return self.class_to_id.get(symbology, 0)

    def _setup_directories(self, task: str) -> None:
        """Create output directory structure."""
        self.output_folder.mkdir(parents=True, exist_ok=True)

        if task == "classification":
            # Classification: train/class_name/, val/class_name/
            for split in ["train", "val", "test"]:
                for class_name in self.id_to_class.values():
                    (self.output_folder / split / class_name).mkdir(parents=True, exist_ok=True)
        else:
            # Detection/Segmentation: images/train, labels/train, etc.
            for split in ["train", "val", "test"]:
                (self.output_folder / "images" / split).mkdir(parents=True, exist_ok=True)
                (self.output_folder / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _get_split(self, idx: int, total: int, train_ratio: float, val_ratio: float) -> str:
        """Determine which split (train/val/test) a sample belongs to."""
        position = idx / total
        if position < train_ratio:
            return "train"
        elif position < train_ratio + val_ratio:
            return "val"
        else:
            return "test"

    def _generate_and_save_sample(
        self,
        symbology: str,
        sample_idx: int,
        split: str,
        task: str,
        label_mode: str,
        enable_degradation: bool,
        degradation_prob: float,
        image_format: str,
        min_barcodes: int,
        max_barcodes: int,
    ) -> Optional[str]:
        """
        Generate a single sample and save to disk.

        Returns:
            Split name on success, None on failure
        """
        # Build degradation config
        degradation = None
        if self._degradation_configs:
            # Use specific degradation configs (cycle through list)
            degradation = self._degradation_configs[self._degradation_index % len(self._degradation_configs)]
            self._degradation_index += 1
        elif enable_degradation and random.random() < degradation_prob:
            # Random degradation
            degradation = self._random_degradation()

        # Generate barcode via API
        sample_data = generate_sample_data(symbology)

        try:
            result = self.api_client.generate_barcode(
                barcode_type=symbology,
                text=sample_data,
                degradation=degradation,
                image_format=image_format.upper()
            )
        except APIError as e:
            logger.warning(f"Failed to generate {symbology}: {e}")
            return None

        if not result:
            return None

        # Get class ID and name
        class_id = self._get_class_id(symbology, label_mode)
        class_name = self.id_to_class.get(class_id, symbology)

        # Generate filename
        filename = f"{symbology}_{sample_idx:06d}.{image_format}"

        # Handle background embedding if configured
        if self.background_manager and task != "classification":
            num_barcodes = random.randint(min_barcodes, max_barcodes)
            final_image, adjusted_result = self._embed_on_background(
                result, num_barcodes
            )
        else:
            final_image = result.image
            adjusted_result = result

        # Build AnnotationData for format handler
        annotation_data = AnnotationData(
            image=final_image,
            image_filename=filename,
            image_size=final_image.size,
            symbology=symbology,
            encoded_value=adjusted_result.input_text,
            printed_text=adjusted_result.input_text,  # May differ for some symbologies
            barcode_only_polygon=adjusted_result.barcode_polygon,
            barcode_only_bbox=adjusted_result.barcode_bbox,
            text_region_polygon=adjusted_result.text_region_polygon,
            full_region_polygon=adjusted_result.full_region_polygon,
            orientation="top-left",  # TODO: Calculate from transformations
            class_id=class_id,
            class_name=class_name,
            degradation_applied=adjusted_result.degradation_applied,
            transformations=adjusted_result.transformations,
        )

        # Save using format handler
        self.format_handler.save_annotation(annotation_data, split, task)

        return split

    def _random_degradation(self) -> Dict:
        """Generate random degradation configuration.

        Returns a config in the API's expected format with categories as lists
        of transformation objects, each with a 'type' key.
        """
        degradation = {}

        # Build geometry transforms list
        geometry_transforms = []
        if random.random() < 0.5:
            geometry_transforms.append({
                "type": "y_axis_rotation",
                "angle_degrees": random.uniform(-15, 15)
            })
        if random.random() < 0.3:
            geometry_transforms.append({
                "type": "combined_rotation",
                "x_angle": random.uniform(-10, 10),
                "y_angle": random.uniform(-10, 10),
                "z_angle": 0
            })
        if geometry_transforms:
            degradation["geometry"] = geometry_transforms

        # Build damage transforms list
        damage_transforms = []
        if random.random() < 0.4:
            damage_transforms.append({
                "type": "motion_blur",
                "intensity": random.uniform(0.5, 2.0),
                "direction": random.uniform(0, 360)
            })
        if random.random() < 0.3:
            damage_transforms.append({
                "type": "fading",
                "contrast_reduction": random.uniform(0.1, 0.3),
                "pattern": "uniform"
            })
        if damage_transforms:
            degradation["damage"] = damage_transforms

        return degradation

    def _embed_on_background(
        self,
        result: BarcodeResult,
        num_barcodes: int = 1
    ) -> Tuple[Image.Image, BarcodeResult]:
        """Embed barcode(s) on background image.

        Scales the barcode to a random size within barcode_scale_range,
        then places it on a random background without additional rotation.

        Returns:
            Tuple of (background image, adjusted BarcodeResult with scaled and offset coordinates)
        """
        target_size = self.config.backgrounds.target_size
        background = self.background_manager.get_random_background(target_size)
        bg_width, bg_height = background.size

        existing_bboxes = []

        # For now, just place the first barcode
        # TODO: Support multiple barcodes per image
        barcode_img = result.image

        # Scale barcode to random size within range (as fraction of background width)
        scale = random.uniform(*self.barcode_scale_range)
        new_barcode_width = int(bg_width * scale)
        aspect_ratio = barcode_img.height / barcode_img.width
        new_barcode_height = int(new_barcode_width * aspect_ratio)

        # Calculate actual scale factor for coordinate transformation
        actual_scale = new_barcode_width / barcode_img.width

        # Resize barcode image
        barcode_resized = barcode_img.resize(
            (new_barcode_width, new_barcode_height),
            Image.Resampling.LANCZOS
        )

        # Find placement position for scaled barcode
        position = self.background_manager.place_barcode(
            background, barcode_resized, existing_bboxes
        )

        if position:
            x, y = position
            background.paste(barcode_resized, (x, y))

            # Create adjusted metadata with scaled and offset coordinates
            adjusted_metadata = dict(result.metadata)
            adjusted_regions = {}

            # Helper function to scale and offset a polygon
            def scale_and_offset_polygon(polygon):
                return [
                    [int(p[0] * actual_scale) + x, int(p[1] * actual_scale) + y]
                    for p in polygon
                ]

            # Scale and offset barcode_only region
            if result.barcode_polygon:
                adjusted_polygon = scale_and_offset_polygon(result.barcode_polygon)
                # Calculate bbox from polygon if not provided by API
                if result.barcode_bbox:
                    adjusted_bbox = [
                        int(result.barcode_bbox[0] * actual_scale) + x,
                        int(result.barcode_bbox[1] * actual_scale) + y,
                        int(result.barcode_bbox[2] * actual_scale) + x,
                        int(result.barcode_bbox[3] * actual_scale) + y,
                    ]
                else:
                    # Derive bbox from polygon
                    xs = [p[0] for p in adjusted_polygon]
                    ys = [p[1] for p in adjusted_polygon]
                    adjusted_bbox = [min(xs), min(ys), max(xs), max(ys)]

                adjusted_regions["barcode_only"] = {
                    "polygon": adjusted_polygon,
                    "bbox": adjusted_bbox
                }

            # Scale and offset text region (use "text" key to match API response format)
            if result.text_region_polygon:
                adjusted_regions["text"] = {
                    "polygon": scale_and_offset_polygon(result.text_region_polygon),
                    "present": True
                }

            # Scale and offset full_region
            if result.full_region_polygon:
                adjusted_regions["full"] = {
                    "polygon": scale_and_offset_polygon(result.full_region_polygon)
                }

            adjusted_metadata["regions"] = adjusted_regions

            # Update image size in metadata to reflect background size
            adjusted_metadata["image_size"] = {
                "width": bg_width,
                "height": bg_height
            }

            # Create new BarcodeResult with adjusted metadata
            adjusted_result = BarcodeResult(
                image=background,
                metadata=adjusted_metadata,
                format=result.format,
                degradation_applied=result.degradation_applied,
                transformations=result.transformations,
                input_text=result.input_text,
            )

            return background, adjusted_result

        # If placement failed, return original
        return background, result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate barcode datasets for ML training and testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate YOLO dataset (default)
  python -m src.dataset_generator -o ./test-dataset -n 10 \\
      --symbologies code128 qr --api-key YOUR_API_KEY

  # Generate detection dataset with degradation
  python -m src.dataset_generator -o ./barcode-detection -n 100 \\
      --categories linear 2d --task detection --degrade

  # Generate segmentation dataset with backgrounds
  python -m src.dataset_generator -o ./barcode-seg -n 500 \\
      --symbologies code128 qr upca --task segmentation \\
      --backgrounds ~/backgrounds --barcodes-per-image 1-3

  # Generate testplan dataset for decoder testing (flat structure)
  python -m src.dataset_generator -o ./decoder-tests -n 50 \\
      --symbologies code128 qr upca --output-format testplan --no-split

  # Generate testplan with train/val/test splits
  python -m src.dataset_generator -o ./decoder-tests -n 100 \\
      --symbologies code128 qr --output-format testplan --split 80/10/10
"""
    )
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--samples", "-n", type=int, default=100, help="Samples per class")
    parser.add_argument("--single", "-1", action="store_true",
                        help="Generate a single sample (quick test mode)")
    parser.add_argument("--symbologies", nargs="+", help="Symbologies to include")
    parser.add_argument("--categories", nargs="+", help="Categories to include (linear, 2d, etc.)")
    parser.add_argument("--families", nargs="+", help="Families to include (code128, ean_upc, etc.)")
    parser.add_argument("--output-format", "-f", choices=["yolo", "testplan"],
                        default="yolo", help="Output annotation format")
    parser.add_argument("--task", choices=["detection", "segmentation", "classification"],
                        default="detection", help="Task type")
    parser.add_argument("--label-mode", choices=["symbology", "category", "family", "binary"],
                        default="symbology", help="Label mode")
    parser.add_argument("--degrade", action="store_true", help="Enable random degradation effects")
    parser.add_argument("--degrade-prob", type=float, default=0.5,
                        help="Degradation probability (0.0-1.0)")
    parser.add_argument("--degrade-preset",
                        help="Use API degradation preset (e.g., cylindrical_surface, flexible_packaging)")
    parser.add_argument("--degrade-config",
                        help="Load degradation config from JSON file")
    parser.add_argument("--degrade-sweep", nargs=4, metavar=("TYPE", "MIN", "MAX", "STEPS"),
                        help="Sweep degradation parameter (e.g., blur 0.1 0.5 5)")
    parser.add_argument("--backgrounds", help="Background images folder")
    parser.add_argument("--barcode-scale", default="0.3-0.6",
                        help="Scale range for barcode size as fraction of background width (e.g., '0.1-0.3')")
    parser.add_argument("--barcodes-per-image", default="1",
                        help="Barcodes per image (e.g., '1' or '1-3')")
    parser.add_argument("--split", default="80/10/10", help="Train/val/test split ratio")
    parser.add_argument("--no-split", action="store_true",
                        help="Disable train/val/test splitting (flat directory)")
    parser.add_argument("--format", choices=["png", "jpg"], default="png", help="Image format")
    parser.add_argument("--api-url",
                        help="API server URL (or set BARCODE_API_URL env var, default: https://barcodes.dev)")
    parser.add_argument("--api-key", help="API key (or set BARCODE_API_KEY env var)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load config
    config = Config()
    if args.config:
        config = Config.from_yaml(Path(args.config))
    config.merge_env()

    # Get API configuration
    api_key = args.api_key or os.environ.get("BARCODE_API_KEY") or config.api.api_key
    api_url = args.api_url or os.environ.get("BARCODE_API_URL") or config.api.base_url

    if not api_key:
        print("Error: API key required. Set --api-key or BARCODE_API_KEY environment variable.")
        print("Get your API key at: https://barcodes.dev/account/api-keys")
        sys.exit(1)

    # Create API client
    client = BarcodeAPIClient(
        base_url=api_url,
        api_key=api_key,
        timeout=config.api.timeout,
        retry_attempts=config.api.retry_attempts
    )

    # Resolve symbologies
    symbologies = []
    if args.symbologies:
        symbologies.extend(args.symbologies)
    if args.categories:
        for cat in args.categories:
            symbologies.extend(get_symbologies_for_category(cat))
    if args.families:
        for family in args.families:
            symbologies.extend(get_symbologies_for_family(family))

    # Remove duplicates while preserving order
    symbologies = list(dict.fromkeys(symbologies))

    if not symbologies:
        print("Error: No symbologies specified. Use --symbologies, --categories, or --families")
        sys.exit(1)

    # Handle --single mode
    if args.single:
        samples_per_class = 1
        enable_split = False
        # Use only first symbology in single mode
        symbologies = symbologies[:1]
        print(f"Single sample mode: generating 1 {symbologies[0]} barcode")
        print(f"  Format: {args.output_format}")
        print(f"  Output: {args.output}")
        print()
    else:
        samples_per_class = args.samples
        enable_split = not args.no_split
        print(f"Generating dataset with {len(symbologies)} symbologies:")
        print(f"  Symbologies: {', '.join(symbologies[:5])}{'...' if len(symbologies) > 5 else ''}")
        print(f"  Samples per class: {samples_per_class}")
        print(f"  Format: {args.output_format}")
        print(f"  Task: {args.task}")
        print(f"  Output: {args.output}")
        print()

    # Create generator
    generator = DatasetGenerator(
        output_folder=Path(args.output),
        api_client=client,
        config=config,
        output_format=args.output_format
    )

    # Set backgrounds if provided
    if args.backgrounds:
        generator.set_backgrounds(Path(args.backgrounds))

        # Parse barcode-scale range
        try:
            parts = args.barcode_scale.split('-')
            if len(parts) != 2:
                print("Error: --barcode-scale must be in format 'min-max' (e.g., '0.1-0.3')")
                sys.exit(1)
            min_scale, max_scale = float(parts[0]), float(parts[1])
            if min_scale <= 0 or max_scale > 1.0 or min_scale > max_scale:
                print("Error: Invalid barcode-scale range. Must be 0.0-1.0 with min <= max")
                sys.exit(1)
            generator.set_barcode_scale_range((min_scale, max_scale))
        except ValueError:
            print("Error: --barcode-scale values must be numbers (e.g., '0.1-0.3')")
            sys.exit(1)

    # Build degradation configs if specified
    degradation_configs = None
    if args.degrade_config:
        # Load from JSON file
        degradation_configs = load_degradation_config(args.degrade_config)
        print(f"  Degradation: {len(degradation_configs)} config(s) from {args.degrade_config}")
    elif args.degrade_preset:
        # Fetch from API preset
        try:
            preset_config = fetch_preset_config(client, args.degrade_preset)
            degradation_configs = [preset_config]
            print(f"  Degradation: preset '{args.degrade_preset}'")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.degrade_sweep:
        # Build sweep configs
        sweep_type, min_val, max_val, steps = args.degrade_sweep
        try:
            min_float = float(min_val)
        except ValueError:
            print(f"Error: --degrade-sweep MIN value '{min_val}' is not a valid number")
            sys.exit(1)
        try:
            max_float = float(max_val)
        except ValueError:
            print(f"Error: --degrade-sweep MAX value '{max_val}' is not a valid number")
            sys.exit(1)
        try:
            steps_int = int(steps)
        except ValueError:
            print(f"Error: --degrade-sweep STEPS value '{steps}' is not a valid integer")
            sys.exit(1)
        try:
            degradation_configs = build_sweep_configs(
                sweep_type, min_float, max_float, steps_int
            )
            print(f"  Degradation: sweep {sweep_type} from {min_val} to {max_val} in {steps} steps")
        except ValueError as e:
            print(f"Error: --degrade-sweep: {e}")
            sys.exit(1)

    # Generate dataset
    try:
        stats = generator.generate(
            symbologies=symbologies,
            samples_per_class=samples_per_class,
            task=args.task,
            label_mode=args.label_mode,
            enable_degradation=args.degrade,
            degradation_prob=args.degrade_prob,
            degradation_configs=degradation_configs,
            split_ratio=args.split,
            image_format=args.format,
            barcodes_per_image=args.barcodes_per_image,
            workers=args.workers,
            enable_split=enable_split
        )

        print("\nGeneration complete!")
        print(f"  Total generated: {stats['generated']}/{stats['total_requested']}")
        print(f"  Failed: {stats['failed']}")
        if enable_split:
            print(f"  Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
        print(f"\nDataset saved to: {args.output}")
        if args.output_format == "yolo":
            print(f"Use data.yaml for YOLO training: {Path(args.output) / 'data.yaml'}")
        elif args.output_format == "testplan":
            print(f"See manifest.json for dataset summary: {Path(args.output) / 'manifest.json'}")

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Generation failed")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
