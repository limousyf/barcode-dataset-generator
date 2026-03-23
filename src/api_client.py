"""
HTTP client for barcodes.dev API.

This module handles all communication with the barcode generation API,
including request formatting, response parsing, and error handling.
"""

import base64
import io
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Composite symbologies that require the /api/v2/composite endpoint
COMPOSITE_SYMBOLOGIES = {
    'gs1_128_cc', 'eanx_cc', 'upca_cc', 'upce_cc',
    'dbar_omn_cc', 'dbar_ltd_cc', 'dbar_exp_cc',
    'dbar_stk_cc', 'dbar_omnstk_cc', 'dbar_expstk_cc',
    # Aliases used in this project
    'ean13_cc', 'ean8_cc', 'gs1_databar_cc',
    'gs1_databar_expanded_cc', 'gs1_databar_stacked_cc',
}

# Map aliases to API symbology names
COMPOSITE_SYMBOLOGY_MAP = {
    'ean13_cc': 'eanx_cc',
    'ean8_cc': 'eanx_cc',
    'gs1_databar_cc': 'dbar_omn_cc',
    'gs1_databar_expanded_cc': 'dbar_exp_cc',
    'gs1_databar_stacked_cc': 'dbar_stk_cc',
}


def is_composite_symbology(symbology: str) -> bool:
    """Check if a symbology is a composite type requiring special handling."""
    return symbology.lower() in COMPOSITE_SYMBOLOGIES


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
    degradation_applied: bool = False
    transformations: List[str] = field(default_factory=list)
    input_text: str = ""  # The original text encoded in the barcode

    @property
    def regions(self) -> Dict[str, Any]:
        """Get region metadata (polygons, bboxes)."""
        return self.metadata.get("regions", {})

    @property
    def barcode_polygon(self) -> Optional[List[List[int]]]:
        """Get barcode region polygon coordinates."""
        barcode_region = self.regions.get("barcode_only", {})
        return barcode_region.get("polygon")

    @property
    def barcode_bbox(self) -> Optional[List[int]]:
        """Get barcode bounding box [x_min, y_min, x_max, y_max]."""
        barcode_region = self.regions.get("barcode_only", {})
        return barcode_region.get("bbox")

    @property
    def text_region_polygon(self) -> Optional[List[List[int]]]:
        """Get text label region polygon coordinates."""
        # API returns 'text' with 'polygon' and 'present' fields
        text_region = self.regions.get("text", {})
        if not text_region.get("present", False):
            return None
        return text_region.get("polygon")

    @property
    def full_region_polygon(self) -> Optional[List[List[int]]]:
        """Get full barcode region including quiet zones."""
        # Try multiple possible key names
        full_region = self.regions.get("full", self.regions.get("full_region", {}))
        return full_region.get("polygon")

    @property
    def image_size(self) -> tuple:
        """Get image dimensions (width, height)."""
        return self.image.size


class BarcodeAPIClient:
    """Client for barcodes.dev barcode generation API."""

    # Known API environments
    ENVIRONMENTS = {
        "local": "http://localhost:5001",
        "production": "https://barcodes.dev",
    }

    def __init__(
        self,
        base_url: str = "https://barcodes.dev",
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
        degradation: Optional[Dict] = None,
        image_format: str = "PNG"
    ) -> Optional[BarcodeResult]:
        """
        Generate a barcode with optional degradation.

        Uses v2 API endpoint which requires authentication for production.
        Automatically routes composite symbologies to the correct endpoint.

        Args:
            barcode_type: Symbology type (code128, qr, upca, gs1_128_cc, etc.)
            text: Data to encode. For composite barcodes, use format:
                  "primary_data|composite_data" (pipe-separated)
            degradation: Optional degradation configuration
            image_format: Output format (PNG, JPEG, WEBP)

        Returns:
            BarcodeResult with image and metadata, or None on failure

        Raises:
            AuthenticationError: If API key not set for production
            APIError: If API request fails
        """
        # Route composite symbologies to the composite endpoint
        if is_composite_symbology(barcode_type):
            # Parse primary and composite data from text
            if "|" in text:
                primary_data, composite_data = text.split("|", 1)
            else:
                # If no separator, use text as primary and generate composite
                primary_data = text
                composite_data = self._generate_default_composite_data()

            return self.generate_composite_barcode(
                composite_type=barcode_type,
                primary_data=primary_data,
                composite_data=composite_data,
                degradation=degradation,
                image_format=image_format
            )

        # Check authentication for production
        if self.is_production and not self.is_authenticated:
            raise AuthenticationError(
                "API key required for barcodes.dev v2 endpoints. "
                "Set api_key when creating client or use set_api_key()."
            )

        # Build request payload
        payload = {
            "type": barcode_type,
            "text": text,
            "format": image_format.upper(),
            "include_metadata": True,
        }

        if degradation:
            payload["degradation"] = degradation

        # Make request with retry
        response = self._request_with_retry(
            "POST",
            "/api/v2/barcode/generate",
            json=payload
        )

        if not response:
            return None

        data = response.json()

        if not data.get("success", False):
            error_msg = data.get("error", "Unknown error")
            logger.error(f"Barcode generation failed: {error_msg}")
            raise APIError(f"Barcode generation failed: {error_msg}")

        # Decode image
        image = self._decode_image(data["image"])

        return BarcodeResult(
            image=image,
            metadata=data.get("metadata", {}),
            format=data.get("format", "PNG").lower(),
            degradation_applied=data.get("degradation_applied", False),
            transformations=data.get("transformations", []),
            input_text=text,
        )

    def _generate_default_composite_data(self) -> str:
        """Generate default 2D composite component data."""
        import random
        batch = random.randint(100, 999)
        # Expiry date format: YYMMDD
        expiry = f"26{random.randint(1, 12):02d}{random.randint(1, 28):02d}"
        return f"[10]BATCH{batch}[17]{expiry}"

    def generate_composite_barcode(
        self,
        composite_type: str,
        primary_data: str,
        composite_data: str,
        degradation: Optional[Dict] = None,
        image_format: str = "PNG"
    ) -> Optional[BarcodeResult]:
        """
        Generate a composite barcode with optional degradation.

        Composite barcodes combine a linear barcode with a 2D component.

        Args:
            composite_type: Composite symbology (gs1_128_cc, eanx_cc, etc.)
            primary_data: Data for the linear barcode component
            composite_data: Data for the 2D composite component
            degradation: Optional degradation configuration
            image_format: Output format (PNG, JPEG, WEBP)

        Returns:
            BarcodeResult with image and metadata, or None on failure
        """
        if self.is_production and not self.is_authenticated:
            raise AuthenticationError(
                "API key required for barcodes.dev v2 endpoints."
            )

        # Map aliases to API symbology names
        api_type = COMPOSITE_SYMBOLOGY_MAP.get(composite_type.lower(), composite_type.lower())

        payload = {
            "composite_type": api_type,
            "primary_data": primary_data,
            "text": composite_data,
            "format": image_format.upper(),
        }

        if degradation:
            payload["degradation"] = degradation

        response = self._request_with_retry(
            "POST",
            "/api/v2/composite",
            json=payload
        )

        if not response:
            return None

        data = response.json()

        if not data.get("success", False):
            error_msg = data.get("error", "Unknown error")
            logger.error(f"Composite barcode generation failed: {error_msg}")
            raise APIError(f"Composite barcode generation failed: {error_msg}")

        # Composite endpoint returns 'data' instead of 'image'
        image_data = data.get("data") or data.get("image")
        if not image_data:
            raise APIError("No image data in composite response")
        image = self._decode_image(image_data)

        # Store both primary and composite data in input_text
        combined_text = f"{primary_data}|{composite_data}"

        return BarcodeResult(
            image=image,
            metadata=data.get("metadata", {}),
            format=data.get("format", "PNG").lower(),
            degradation_applied=data.get("degradation_applied", False),
            transformations=data.get("transformations", []),
            input_text=combined_text,
        )

    def get_degradation_presets(self) -> Dict[str, Any]:
        """
        Get available degradation presets from API.

        Returns:
            Dictionary of preset configurations
        """
        response = self._request_with_retry("GET", "/api/v2/presets")

        if not response:
            return {}

        return response.json()

    def get_symbologies(self) -> Dict[str, List[str]]:
        """
        Get supported barcode symbologies grouped by category.

        Returns:
            Dictionary with categories as keys and lists of symbology names
        """
        response = self._request_with_retry("GET", "/api/v2/barcode/symbologies")

        if not response:
            return {}

        data = response.json()
        return data.get("categories", {})

    def get_families(self) -> Dict[str, List[str]]:
        """
        Get barcode family groupings.

        Returns:
            Dictionary with family names as keys and lists of symbology names
        """
        response = self._request_with_retry("GET", "/api/v2/barcode/families")

        if not response:
            return {}

        data = response.json()
        return data.get("families", {})

    def get_all_symbologies(self) -> List[str]:
        """
        Get flat list of all supported symbology names.

        Returns:
            List of symbology names
        """
        categories = self.get_symbologies()
        all_symbologies = set()
        for symbology_list in categories.values():
            all_symbologies.update(symbology_list)
        return sorted(all_symbologies)

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

    def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with retry and exponential backoff.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests

        Returns:
            Response object or None on failure
        """
        url = f"{self.base_url}{endpoint}"
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs
                )

                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                # Check for auth errors
                if response.status_code == 401:
                    raise AuthenticationError(
                        "Invalid or missing API key. "
                        "Check your BARCODE_API_KEY or --api-key."
                    )

                # Check for server errors (retry)
                if response.status_code >= 500:
                    logger.warning(
                        f"Server error {response.status_code}, "
                        f"attempt {attempt + 1}/{self.retry_attempts}"
                    )
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

                # Success or client error (don't retry)
                response.raise_for_status()
                return response

            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(
                    f"Connection error, attempt {attempt + 1}/{self.retry_attempts}: {e}"
                )
                time.sleep(2 ** attempt)

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(
                    f"Timeout, attempt {attempt + 1}/{self.retry_attempts}: {e}"
                )
                time.sleep(2 ** attempt)

            except requests.exceptions.HTTPError as e:
                # Client errors (4xx except 429) - don't retry
                if e.response is not None and 400 <= e.response.status_code < 500:
                    raise APIError(f"API error: {e.response.status_code} - {e.response.text}")
                last_error = e
                time.sleep(2 ** attempt)

        # All retries exhausted
        if last_error:
            raise ConnectionError(
                f"Failed to connect to {self.base_url} after {self.retry_attempts} attempts: {last_error}"
            )
        return None

    def _decode_image(self, base64_data: str) -> Image.Image:
        """Decode base64 image data to PIL Image."""
        image_data = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_data))
