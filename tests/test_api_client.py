"""Tests for API client."""

import base64
import io
import pytest
import responses
from PIL import Image

from src.api_client import (
    BarcodeAPIClient,
    BarcodeResult,
    APIError,
    AuthenticationError,
    ConnectionError,
)


def create_test_image_base64() -> str:
    """Create a simple test image and return base64 encoded."""
    img = Image.new('RGB', (100, 50), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class TestBarcodeAPIClient:
    """Tests for BarcodeAPIClient."""

    def test_init_default(self):
        """Test default initialization."""
        client = BarcodeAPIClient()
        assert client.base_url == "http://localhost:5001"
        assert client.api_key is None
        assert client.timeout == 30
        assert client.retry_attempts == 3

    def test_init_with_environment_shorthand(self):
        """Test initialization with environment shorthand."""
        client = BarcodeAPIClient("local")
        assert client.base_url == "http://localhost:5001"

        client = BarcodeAPIClient("production")
        assert client.base_url == "https://barcodes.dev"

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = BarcodeAPIClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.is_authenticated
        assert client.session.headers["X-API-Key"] == "test-key"

    def test_is_production(self):
        """Test production detection."""
        local_client = BarcodeAPIClient("http://localhost:5001")
        assert not local_client.is_production

        prod_client = BarcodeAPIClient("https://barcodes.dev")
        assert prod_client.is_production

    def test_set_api_key(self):
        """Test setting API key after initialization."""
        client = BarcodeAPIClient()
        assert not client.is_authenticated

        client.set_api_key("new-key")
        assert client.is_authenticated
        assert client.api_key == "new-key"

    @responses.activate
    def test_health_check_success(self):
        """Test successful health check."""
        responses.add(
            responses.GET,
            "http://localhost:5001/health",
            status=200
        )

        client = BarcodeAPIClient()
        assert client.health_check() is True

    @responses.activate
    def test_health_check_failure(self):
        """Test failed health check."""
        responses.add(
            responses.GET,
            "http://localhost:5001/health",
            status=500
        )

        client = BarcodeAPIClient()
        assert client.health_check() is False

    @responses.activate
    def test_get_symbologies(self):
        """Test fetching symbologies."""
        responses.add(
            responses.GET,
            "http://localhost:5001/api/v2/barcode/symbologies",
            json={
                "categories": {
                    "linear": ["code128", "code39", "ean13"],
                    "2d": ["qr", "datamatrix"],
                }
            },
            status=200
        )

        client = BarcodeAPIClient(api_key="test")
        symbologies = client.get_symbologies()

        assert "linear" in symbologies
        assert "2d" in symbologies
        assert "code128" in symbologies["linear"]
        assert "qr" in symbologies["2d"]

    @responses.activate
    def test_get_families(self):
        """Test fetching families."""
        responses.add(
            responses.GET,
            "http://localhost:5001/api/v2/barcode/families",
            json={
                "families": {
                    "code128": ["code128", "gs1_128"],
                    "ean_upc": ["ean13", "upca"],
                }
            },
            status=200
        )

        client = BarcodeAPIClient(api_key="test")
        families = client.get_families()

        assert "code128" in families
        assert "ean_upc" in families

    @responses.activate
    def test_get_degradation_presets(self):
        """Test fetching degradation presets."""
        responses.add(
            responses.GET,
            "http://localhost:5001/api/v2/presets",
            json={
                "presets": {
                    "light": {"damage": {"scratches": 0.1}},
                    "heavy": {"damage": {"scratches": 0.5}},
                }
            },
            status=200
        )

        client = BarcodeAPIClient(api_key="test")
        presets = client.get_degradation_presets()

        assert "presets" in presets

    @responses.activate
    def test_generate_barcode_success(self):
        """Test successful barcode generation."""
        test_image = create_test_image_base64()

        responses.add(
            responses.POST,
            "http://localhost:5001/api/v2/barcode/generate",
            json={
                "success": True,
                "image": test_image,
                "format": "PNG",
                "degradation_applied": False,
                "transformations": [],
                "metadata": {
                    "original_size": {"width": 100, "height": 50},
                    "regions": {
                        "barcode_only": {
                            "polygon": [[10, 5], [90, 5], [90, 45], [10, 45]],
                            "bbox": [10, 5, 90, 45]
                        }
                    }
                }
            },
            status=200
        )

        client = BarcodeAPIClient(api_key="test")
        result = client.generate_barcode("code128", "TEST123")

        assert result is not None
        assert isinstance(result, BarcodeResult)
        assert result.image.size == (100, 50)
        assert result.format == "png"
        assert result.barcode_bbox == [10, 5, 90, 45]
        assert result.barcode_polygon is not None

    @responses.activate
    def test_generate_barcode_with_degradation(self):
        """Test barcode generation with degradation."""
        test_image = create_test_image_base64()

        responses.add(
            responses.POST,
            "http://localhost:5001/api/v2/barcode/generate",
            json={
                "success": True,
                "image": test_image,
                "format": "PNG",
                "degradation_applied": True,
                "transformations": ["rotation", "noise"],
                "metadata": {
                    "regions": {
                        "barcode_only": {
                            "polygon": [[15, 10], [85, 8], [87, 42], [13, 44]],
                            "bbox": [13, 8, 87, 44]
                        }
                    }
                }
            },
            status=200
        )

        client = BarcodeAPIClient(api_key="test")
        degradation = {
            "geometry": {"rotation": {"angle": 5}},
            "materials": {"noise": {"intensity": 0.2}}
        }
        result = client.generate_barcode("code128", "TEST123", degradation=degradation)

        assert result is not None
        assert result.degradation_applied is True
        assert "rotation" in result.transformations
        assert "noise" in result.transformations

    @responses.activate
    def test_generate_barcode_api_error(self):
        """Test barcode generation API error."""
        responses.add(
            responses.POST,
            "http://localhost:5001/api/v2/barcode/generate",
            json={
                "success": False,
                "error": "Invalid barcode type"
            },
            status=200
        )

        client = BarcodeAPIClient(api_key="test")

        with pytest.raises(APIError) as exc_info:
            client.generate_barcode("invalid_type", "TEST123")

        assert "Invalid barcode type" in str(exc_info.value)

    def test_generate_barcode_requires_auth_for_production(self):
        """Test that production requires API key."""
        client = BarcodeAPIClient("https://barcodes.dev")

        with pytest.raises(AuthenticationError):
            client.generate_barcode("code128", "TEST123")

    @responses.activate
    def test_generate_barcode_auth_error(self):
        """Test authentication error handling."""
        responses.add(
            responses.POST,
            "http://localhost:5001/api/v2/barcode/generate",
            status=401
        )

        client = BarcodeAPIClient(api_key="invalid-key")

        with pytest.raises(AuthenticationError):
            client.generate_barcode("code128", "TEST123")


class TestBarcodeResult:
    """Tests for BarcodeResult dataclass."""

    def test_barcode_result_properties(self):
        """Test BarcodeResult property accessors."""
        img = Image.new('RGB', (200, 100), color='white')
        metadata = {
            "regions": {
                "barcode_only": {
                    "polygon": [[10, 10], [190, 10], [190, 90], [10, 90]],
                    "bbox": [10, 10, 190, 90]
                }
            }
        }

        result = BarcodeResult(
            image=img,
            metadata=metadata,
            format="png",
            degradation_applied=True,
            transformations=["rotation"]
        )

        assert result.image_size == (200, 100)
        assert result.barcode_bbox == [10, 10, 190, 90]
        assert result.barcode_polygon == [[10, 10], [190, 10], [190, 90], [10, 90]]
        assert result.regions == metadata["regions"]

    def test_barcode_result_empty_metadata(self):
        """Test BarcodeResult with empty metadata."""
        img = Image.new('RGB', (100, 50), color='white')
        result = BarcodeResult(image=img, metadata={})

        assert result.barcode_bbox is None
        assert result.barcode_polygon is None
        assert result.regions == {}
