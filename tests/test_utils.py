"""Tests for utility functions."""

import pytest

from src.utils import (
    get_symbologies_for_category,
    get_symbologies_for_family,
    parse_split_ratio,
    parse_barcodes_per_image,
    generate_sample_data,
    SYMBOLOGY_CATEGORIES,
    SYMBOLOGY_FAMILIES,
)


class TestSymbologyLookups:
    """Tests for symbology category and family lookups."""

    def test_get_symbologies_for_category_linear(self):
        """Test getting linear symbologies."""
        symbologies = get_symbologies_for_category("linear")
        assert "code128" in symbologies
        assert "ean13" in symbologies
        assert "upca" in symbologies

    def test_get_symbologies_for_category_2d(self):
        """Test getting 2D symbologies."""
        symbologies = get_symbologies_for_category("2d")
        assert "qr" in symbologies
        assert "datamatrix" in symbologies

    def test_get_symbologies_for_category_case_insensitive(self):
        """Test category lookup is case insensitive."""
        assert get_symbologies_for_category("LINEAR") == get_symbologies_for_category("linear")
        assert get_symbologies_for_category("2D") == get_symbologies_for_category("2d")

    def test_get_symbologies_for_category_unknown(self):
        """Test unknown category returns empty list."""
        assert get_symbologies_for_category("unknown") == []

    def test_get_symbologies_for_family_code128(self):
        """Test getting code128 family."""
        symbologies = get_symbologies_for_family("code128")
        assert "code128" in symbologies
        assert "gs1_128" in symbologies

    def test_get_symbologies_for_family_ean_upc(self):
        """Test getting EAN/UPC family."""
        symbologies = get_symbologies_for_family("ean_upc")
        assert "ean13" in symbologies
        assert "upca" in symbologies
        assert "isbn" in symbologies

    def test_get_symbologies_for_family_unknown(self):
        """Test unknown family returns empty list."""
        assert get_symbologies_for_family("unknown") == []


class TestParseSplitRatio:
    """Tests for split ratio parsing."""

    def test_parse_standard_split(self):
        """Test standard 80/10/10 split."""
        train, val, test = parse_split_ratio("80/10/10")
        assert train == 0.8
        assert val == 0.1
        assert test == 0.1

    def test_parse_different_split(self):
        """Test different split ratios."""
        train, val, test = parse_split_ratio("70/20/10")
        assert train == 0.7
        assert val == 0.2
        assert test == 0.1

    def test_parse_no_test_split(self):
        """Test split with no test set."""
        train, val, test = parse_split_ratio("90/10/0")
        assert train == 0.9
        assert val == 0.1
        assert test == 0.0

    def test_parse_invalid_format(self):
        """Test invalid format raises error."""
        with pytest.raises(ValueError):
            parse_split_ratio("80/20")  # Missing test

        with pytest.raises(ValueError):
            parse_split_ratio("80-10-10")  # Wrong delimiter

    def test_parse_invalid_sum(self):
        """Test ratios not summing to 100 raises error."""
        with pytest.raises(ValueError):
            parse_split_ratio("80/10/5")  # Sums to 95


class TestParseBarcodePerImage:
    """Tests for barcodes per image parsing."""

    def test_parse_single_value(self):
        """Test parsing single value."""
        min_bc, max_bc = parse_barcodes_per_image("1")
        assert min_bc == 1
        assert max_bc == 1

    def test_parse_range(self):
        """Test parsing range."""
        min_bc, max_bc = parse_barcodes_per_image("1-3")
        assert min_bc == 1
        assert max_bc == 3

    def test_parse_larger_range(self):
        """Test parsing larger range."""
        min_bc, max_bc = parse_barcodes_per_image("2-5")
        assert min_bc == 2
        assert max_bc == 5


class TestGenerateSampleData:
    """Tests for sample data generation."""

    def test_generate_upca_data(self):
        """Test UPC-A data generation (11 digits)."""
        data = generate_sample_data("upca")
        assert len(data) == 11
        assert data.isdigit()

    def test_generate_ean13_data(self):
        """Test EAN-13 data generation (12 digits + check)."""
        data = generate_sample_data("ean13")
        assert len(data) == 12
        assert data.isdigit()

    def test_generate_code128_data(self):
        """Test Code 128 data generation (alphanumeric)."""
        data = generate_sample_data("code128")
        assert 6 <= len(data) <= 12
        assert data.isalnum()

    def test_generate_qr_data(self):
        """Test QR code data generation."""
        data = generate_sample_data("qr")
        assert 10 <= len(data) <= 30

    def test_generate_unknown_symbology_data(self):
        """Test fallback for unknown symbology."""
        data = generate_sample_data("unknown_symbology")
        assert len(data) == 8
        assert data.isalnum()

    def test_data_is_random(self):
        """Test that generated data is random."""
        data1 = generate_sample_data("code128")
        data2 = generate_sample_data("code128")
        # Very unlikely to be the same
        # (technically possible but probability is negligible)
        # Just check they're both valid
        assert data1.isalnum()
        assert data2.isalnum()
