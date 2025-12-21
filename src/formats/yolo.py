"""
YOLO format output handler.

Generates YOLO-compatible annotations for detection and segmentation tasks,
and folder-based structure for classification tasks.
"""

from pathlib import Path
from typing import Any, Dict, List

from .base import OutputFormat, AnnotationData


class YOLOFormat(OutputFormat):
    """YOLO format handler for detection, segmentation, and classification.

    Output structure for detection/segmentation:
        output_folder/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/

    Output structure for classification:
        output_folder/
        ├── data.yaml
        ├── train/
        │   ├── class_name_1/
        │   └── class_name_2/
        ├── val/
        └── test/
    """

    name = "yolo"
    description = "YOLO format (Ultralytics compatible)"
    file_extension = ".txt"

    supports_detection = True
    supports_segmentation = True
    supports_classification = True
    supports_split = True

    def setup_directories(
        self,
        task: str,
        class_names: List[str],
        enable_split: bool = True
    ) -> None:
        """Create YOLO directory structure."""
        self.output_folder.mkdir(parents=True, exist_ok=True)

        splits = ["train", "val", "test"] if enable_split else [""]

        if task == "classification":
            # Classification: train/class_name/, val/class_name/, etc.
            for split in splits:
                for class_name in class_names:
                    if split:
                        path = self.output_folder / split / class_name
                    else:
                        path = self.output_folder / class_name
                    path.mkdir(parents=True, exist_ok=True)
        else:
            # Detection/Segmentation: images/train, labels/train, etc.
            for split in splits:
                if split:
                    (self.output_folder / "images" / split).mkdir(
                        parents=True, exist_ok=True
                    )
                    (self.output_folder / "labels" / split).mkdir(
                        parents=True, exist_ok=True
                    )
                else:
                    (self.output_folder / "images").mkdir(
                        parents=True, exist_ok=True
                    )
                    (self.output_folder / "labels").mkdir(
                        parents=True, exist_ok=True
                    )

    def save_annotation(
        self,
        data: AnnotationData,
        split: str,
        task: str
    ) -> Path:
        """Save YOLO format annotation."""
        filename_base = Path(data.image_filename).stem

        if task == "classification":
            # Save image to class folder
            if split:
                img_path = (
                    self.output_folder / split / data.class_name / data.image_filename
                )
            else:
                img_path = self.output_folder / data.class_name / data.image_filename
            data.image.save(img_path)
            return img_path

        # Detection/Segmentation
        if split:
            img_path = self.output_folder / "images" / split / data.image_filename
            label_path = self.output_folder / "labels" / split / f"{filename_base}.txt"
        else:
            img_path = self.output_folder / "images" / data.image_filename
            label_path = self.output_folder / "labels" / f"{filename_base}.txt"

        data.image.save(img_path)

        # Generate label content
        label_content = self._generate_label(data, task)
        with open(label_path, "w") as f:
            f.write(label_content)

        return label_path

    def _generate_label(self, data: AnnotationData, task: str) -> str:
        """Generate YOLO format label string.

        Detection format: <class_id> <x_center> <y_center> <width> <height>
        Segmentation format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

        All coordinates are normalized to [0, 1].
        """
        class_id = data.class_id if data.class_id is not None else 0
        img_w, img_h = data.image_size

        if task == "segmentation" and data.barcode_only_polygon:
            # Segmentation: class_id x1 y1 x2 y2 ...
            points = []
            for x, y in data.barcode_only_polygon:
                norm_x = max(0.0, min(1.0, x / img_w))
                norm_y = max(0.0, min(1.0, y / img_h))
                points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
            return f"{class_id} " + " ".join(points)

        # Detection: class_id x_center y_center width height
        bbox = data.barcode_only_bbox
        if not bbox and data.barcode_only_polygon:
            bbox = self.bbox_from_polygon(data.barcode_only_polygon)

        if bbox:
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2.0 / img_w
            y_center = (y_min + y_max) / 2.0 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h
            return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        return ""

    def finalize(self, task: str, stats: Dict[str, Any]) -> None:
        """Write YOLO data.yaml configuration file."""
        # Determine paths based on task
        if task == "classification":
            train_path = "train"
            val_path = "val"
            test_path = "test"
        else:
            train_path = "images/train"
            val_path = "images/val"
            test_path = "images/test"

        yaml_content = f"""# YOLO Dataset Configuration
# Generated by barcode-dataset-generator

path: {self.output_folder.absolute()}
train: {train_path}
val: {val_path}
test: {test_path}

nc: {len(self.class_mapping)}
names: {list(self.id_to_class.values())}
"""
        with open(self.output_folder / "data.yaml", "w") as f:
            f.write(yaml_content)
