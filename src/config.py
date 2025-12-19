"""
Configuration management for dataset generator.

Handles loading from YAML files and environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class APIConfig:
    """API connection configuration."""
    base_url: str = "http://localhost:5001"
    api_key: Optional[str] = None  # Required for v2 endpoints on barcodes.dev
    timeout: int = 30
    retry_attempts: int = 3

    @property
    def is_local(self) -> bool:
        """Check if connecting to local development server."""
        return "localhost" in self.base_url or "127.0.0.1" in self.base_url

    @property
    def is_production(self) -> bool:
        """Check if connecting to production barcodes.dev."""
        return "barcodes.dev" in self.base_url


@dataclass
class GenerationConfig:
    """Dataset generation defaults."""
    default_samples_per_class: int = 100
    default_image_format: str = "png"
    default_split: str = "80/10/10"
    default_task: str = "detection"
    default_label_mode: str = "symbology"


@dataclass
class DegradationConfig:
    """Degradation settings."""
    default_probability: float = 0.5
    presets: List[str] = field(default_factory=lambda: ["light", "moderate", "heavy"])


@dataclass
class BackgroundConfig:
    """Background embedding settings."""
    default_folder: Optional[str] = None
    target_size: Tuple[int, int] = (640, 640)
    default_barcodes_per_image: str = "1"


@dataclass
class Config:
    """Main configuration container."""
    api: APIConfig = field(default_factory=APIConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    backgrounds: BackgroundConfig = field(default_factory=BackgroundConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        if "api" in data:
            config.api = APIConfig(**data["api"])
        if "generation" in data:
            config.generation = GenerationConfig(**data["generation"])
        if "degradation" in data:
            config.degradation = DegradationConfig(**data["degradation"])
        if "backgrounds" in data:
            config.backgrounds = BackgroundConfig(**data["backgrounds"])

        return config

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()

        if url := os.environ.get("BARCODE_API_URL"):
            config.api.base_url = url
        if api_key := os.environ.get("BARCODE_API_KEY"):
            config.api.api_key = api_key
        if timeout := os.environ.get("BARCODE_API_TIMEOUT"):
            config.api.timeout = int(timeout)

        return config

    def merge_env(self) -> "Config":
        """Merge environment variables into existing config."""
        if url := os.environ.get("BARCODE_API_URL"):
            self.api.base_url = url
        if api_key := os.environ.get("BARCODE_API_KEY"):
            self.api.api_key = api_key
        if timeout := os.environ.get("BARCODE_API_TIMEOUT"):
            self.api.timeout = int(timeout)
        return self

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # API key required for production
        if self.api.is_production and not self.api.api_key:
            errors.append(
                "API key required for barcodes.dev. "
                "Set BARCODE_API_KEY environment variable or api.api_key in config."
            )

        return errors
