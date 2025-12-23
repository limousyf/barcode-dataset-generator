"""
Output format registry and factory.

This module provides a registry of available output formats and a factory
function to instantiate them by name.

Usage:
    from src.formats import get_format, FORMATS

    # Get format class by name
    format_class = get_format("yolo")
    handler = format_class(output_folder)

    # List available formats
    print(list(FORMATS.keys()))  # ['yolo', 'testplan']
"""

from typing import Dict, Type

from .base import OutputFormat, AnnotationData
from .yolo import YOLOFormat
from .testplan import TestplanFormat
from .paired import PairedFormat


# Format registry: name -> format class
FORMATS: Dict[str, Type[OutputFormat]] = {
    "yolo": YOLOFormat,
    "testplan": TestplanFormat,
    "paired": PairedFormat,
}


def get_format(name: str) -> Type[OutputFormat]:
    """Get format class by name.

    Args:
        name: Format name (e.g., 'yolo', 'testplan')

    Returns:
        Format class (not instance)

    Raises:
        ValueError: If format name is not registered
    """
    if name not in FORMATS:
        available = ", ".join(FORMATS.keys())
        raise ValueError(f"Unknown format: '{name}'. Available formats: {available}")
    return FORMATS[name]


def register_format(name: str, format_class: Type[OutputFormat]) -> None:
    """Register a new output format.

    Args:
        name: Format name to register
        format_class: Format class (must inherit from OutputFormat)
    """
    if not issubclass(format_class, OutputFormat):
        raise TypeError(f"{format_class} must inherit from OutputFormat")
    FORMATS[name] = format_class


def list_formats() -> Dict[str, str]:
    """List available formats with descriptions.

    Returns:
        Dictionary mapping format names to descriptions
    """
    return {name: cls.description for name, cls in FORMATS.items()}


__all__ = [
    "OutputFormat",
    "AnnotationData",
    "YOLOFormat",
    "TestplanFormat",
    "PairedFormat",
    "FORMATS",
    "get_format",
    "register_format",
    "list_formats",
]
