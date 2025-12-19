"""
HTTP client for barcodes.dev API.

This module handles all communication with the barcode generation API,
including request formatting, response parsing, and error handling.
"""

import base64
import io
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API errors."""
    pass


class AuthenticationError(APIError):
    """Raised when API key is missing or invalid."""
    pass


class ConnectionError(APIError):
    """Raised when unable to connect to API."""
    pass


@dataclass
class BarcodeResult:
    """Result from barcode generation API."""
    image: Image.Image
    metadata: Dict[str, Any]
    format: str = "png"


class BarcodeAPIClient:
    """Client for barcodes.dev barcode generation API."""

    # Known API environments
    ENVIRONMENTS = {
        "local": "http://localhost:5001",
        "production": "https://barcodes.dev",
    }

    def __init__(
        self,
        base_url: str = "http://localhost:5001",
        api_key: Optional[str] = None,
        timeout: int = 30,
        retry_attempts: int = 3
    ):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the barcodes.dev API (or 'local'/'production')
            api_key: API key for v2 endpoints (required for production)
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
        """
        # Allow shorthand environment names
        if base_url in self.ENVIRONMENTS:
            base_url = self.ENVIRONMENTS[base_url]

        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.session = requests.Session()

        # Set up default headers
        self._setup_headers()

    def _setup_headers(self) -> None:
        """Configure session headers including authentication."""
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        # Add API key authentication if provided
        if self.api_key:
            self.session.headers["X-API-Key"] = self.api_key

    @property
    def is_authenticated(self) -> bool:
        """Check if API key is configured."""
        return self.api_key is not None

    @property
    def is_production(self) -> bool:
        """Check if connected to production."""
        return "barcodes.dev" in self.base_url

    def set_api_key(self, api_key: str) -> None:
        """
        Set or update the API key.

        Args:
            api_key: The API key for authentication
        """
        self.api_key = api_key
        self.session.headers["X-API-Key"] = api_key

    def generate_barcode(
        self,
        barcode_type: str,
        text: str,
        degradation: Optional[Dict] = None
    ) -> Optional[BarcodeResult]:
        """
        Generate a barcode with optional degradation.

        Uses v2 API endpoint which requires authentication.

        Args:
            barcode_type: Symbology type (code128, qr, upca, etc.)
            text: Data to encode in barcode
            degradation: Optional degradation configuration

        Returns:
            BarcodeResult with image and metadata, or None on failure

        Raises:
            AuthenticationError: If API key not set for production
            APIError: If API request fails
        """
        # Check authentication for production
        if self.is_production and not self.is_authenticated:
            raise AuthenticationError(
                "API key required for barcodes.dev v2 endpoints. "
                "Set api_key when creating client or use set_api_key()."
            )

        # TODO: Implement API call
        raise NotImplementedError("API client not yet implemented")

    def get_degradation_presets(self) -> List[Dict]:
        """
        Get available degradation presets from API.

        Returns:
            List of preset configurations
        """
        # TODO: Implement API call
        raise NotImplementedError("API client not yet implemented")

    def get_supported_symbologies(self) -> List[str]:
        """
        Get list of supported barcode symbologies.

        Returns:
            List of symbology names
        """
        # TODO: Implement API call
        raise NotImplementedError("API client not yet implemented")

    def health_check(self) -> bool:
        """
        Check if API server is reachable.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _decode_image(self, base64_data: str) -> Image.Image:
        """Decode base64 image data to PIL Image."""
        image_data = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_data))
