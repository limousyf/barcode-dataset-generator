"""
Utility functions for dataset generation.
"""

import random
import string
from typing import List, Tuple


# Symbology category mappings (mirror from barcodes.dev)
SYMBOLOGY_CATEGORIES = {
    "linear": [
        "code128", "code39", "code93", "codabar", "itf", "itf14",
        "upca", "upce", "ean13", "ean8", "ean2", "ean5",
        "isbn", "issn", "ismn", "plessey", "msi", "telepen",
        "pharmacode", "code11", "code32", "code49",
    ],
    "2d": [
        "qr", "microqr", "rmqr", "datamatrix", "aztec", "maxicode",
        "pdf417", "micropdf417", "dotcode", "hanxin", "gridmatrix",
    ],
    "stacked": [
        "pdf417", "pdf417comp", "micropdf417", "codablockf", "code16k",
        "code49",
    ],
    "postal": [
        "usps_imd", "usps_postnet", "usps_planet", "usps_onecode",
        "royalmail", "kix", "japanpost", "australiapost", "mailmark",
    ],
    "composite": [
        "gs1_128_cc", "eanx_cc", "upca_cc", "upce_cc",
        "dbar_omn_cc", "dbar_ltd_cc", "dbar_exp_cc",
        "dbar_stk_cc", "dbar_omnstk_cc", "dbar_expstk_cc",
    ],
    "popular": [
        "code128", "code39", "upca", "upce", "ean13", "ean8",
        "qr", "datamatrix", "pdf417", "itf14", "aztec",
    ],
}

# Family groupings
SYMBOLOGY_FAMILIES = {
    "code128": ["code128", "gs1_128", "ean14", "nve18", "sscc18"],
    "ean_upc": ["ean13", "ean8", "upca", "upce", "isbn", "issn", "ean2", "ean5"],
    "code39": ["code39", "code39ext", "code32", "hibc39", "vin", "logmars"],
    "qr": ["qr", "microqr", "rmqr", "upnqr"],
    "pdf417": ["pdf417", "pdf417comp", "micropdf417"],
    "datamatrix": ["datamatrix", "datamatrixrect"],
}


def get_symbologies_for_category(category: str) -> List[str]:
    """Get list of symbologies for a category."""
    return SYMBOLOGY_CATEGORIES.get(category.lower(), [])


def get_symbologies_for_family(family: str) -> List[str]:
    """Get list of symbologies for a family."""
    return SYMBOLOGY_FAMILIES.get(family.lower(), [])


def parse_split_ratio(ratio_str: str) -> Tuple[float, float, float]:
    """
    Parse split ratio string like '80/10/10' into floats.

    Returns:
        Tuple of (train, val, test) ratios as decimals
    """
    parts = ratio_str.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid split ratio: {ratio_str}, expected format: '80/10/10'")

    train, val, test = [float(p) / 100 for p in parts]

    if abs(train + val + test - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 100, got: {ratio_str}")

    return (train, val, test)


def parse_barcodes_per_image(range_str: str) -> Tuple[int, int]:
    """
    Parse barcodes per image range like '1-3' or '2'.

    Returns:
        Tuple of (min, max) barcodes
    """
    if "-" in range_str:
        parts = range_str.split("-")
        return (int(parts[0]), int(parts[1]))
    else:
        n = int(range_str)
        return (n, n)


def _calculate_check_digit(digits: str) -> str:
    """Calculate GS1 check digit for GTIN-8/12/13/14."""
    # Pad to 17 digits (max GTIN length - 1)
    padded = digits.zfill(17)
    total = 0
    for i, digit in enumerate(padded):
        weight = 3 if i % 2 == 0 else 1
        total += int(digit) * weight
    check = (10 - (total % 10)) % 10
    return str(check)


def _generate_gtin14() -> str:
    """Generate a valid GTIN-14 with correct check digit."""
    base = "".join(random.choices(string.digits, k=13))
    check = _calculate_check_digit(base)
    return base + check


def _generate_gtin13() -> str:
    """Generate a valid GTIN-13/EAN-13 with correct check digit."""
    base = "".join(random.choices(string.digits, k=12))
    check = _calculate_check_digit(base)
    return base + check


def _generate_composite_data() -> str:
    """Generate GS1 AI format data for 2D composite component."""
    batch = random.randint(100, 999)
    # Expiry date format: YYMMDD
    expiry = f"26{random.randint(1, 12):02d}{random.randint(1, 28):02d}"
    return f"[10]BATCH{batch}[17]{expiry}"


def generate_sample_data(symbology: str) -> str:
    """
    Generate appropriate sample data for a symbology.

    Args:
        symbology: Barcode type

    Returns:
        Sample data string. For composite symbologies, returns
        "primary_data|composite_data" format.
    """
    symbology_lower = symbology.lower()

    # Composite symbologies - return "primary|composite" format
    if symbology_lower in ["gs1_128_cc"]:
        # GS1-128 Composite: GS1 AI format for primary (AI 01 requires 14 digits)
        gtin = _generate_gtin14()
        primary = f"[01]{gtin}"
        composite = _generate_composite_data()
        return f"{primary}|{composite}"

    if symbology_lower in ["eanx_cc", "ean13_cc"]:
        # EAN Composite: 12 digits (Zint calculates check digit)
        primary = "".join(random.choices(string.digits, k=12))
        composite = _generate_composite_data()
        return f"{primary}|{composite}"

    if symbology_lower in ["ean8_cc"]:
        # EAN-8 Composite: 7 digits
        primary = "".join(random.choices(string.digits, k=7))
        composite = _generate_composite_data()
        return f"{primary}|{composite}"

    if symbology_lower in ["upca_cc"]:
        # UPC-A Composite: 11 digits
        primary = "".join(random.choices(string.digits, k=11))
        composite = _generate_composite_data()
        return f"{primary}|{composite}"

    if symbology_lower in ["upce_cc"]:
        # UPC-E Composite: 7 digits starting with 0
        primary = f"0{random.randint(100000, 999999)}"
        composite = _generate_composite_data()
        return f"{primary}|{composite}"

    if symbology_lower in ["dbar_omn_cc", "gs1_databar_cc"]:
        # DataBar Omnidirectional Composite (14-digit GTIN)
        primary = _generate_gtin14()
        composite = _generate_composite_data()
        return f"{primary}|{composite}"

    if symbology_lower in ["dbar_ltd_cc"]:
        # DataBar Limited Composite (first digit 0 or 1, 13 total)
        first_digit = random.choice(["0", "1"])
        primary = first_digit + "".join(random.choices(string.digits, k=12))
        composite = _generate_composite_data()
        return f"{primary}|{composite}"

    if symbology_lower in ["dbar_exp_cc", "gs1_databar_expanded_cc"]:
        # DataBar Expanded Composite (AI 01 requires 14 digits)
        gtin = _generate_gtin14()
        primary = f"[01]{gtin}"
        composite = _generate_composite_data()
        return f"{primary}|{composite}"

    if symbology_lower in ["dbar_stk_cc", "dbar_omnstk_cc", "gs1_databar_stacked_cc"]:
        # DataBar Stacked variants (14-digit GTIN)
        primary = _generate_gtin14()
        composite = _generate_composite_data()
        return f"{primary}|{composite}"

    if symbology_lower in ["dbar_expstk_cc"]:
        # DataBar Expanded Stacked Composite (AI 01 format)
        gtin = _generate_gtin14()
        primary = f"[01]{gtin}"
        composite = _generate_composite_data()
        return f"{primary}|{composite}"

    # Numeric-only symbologies
    if symbology_lower in ["upca", "ean13", "ean8", "itf", "itf14", "postnet"]:
        length = {
            "upca": 11,
            "ean13": 12,
            "ean8": 7,
            "itf": 10,
            "itf14": 13,
        }.get(symbology_lower, 10)
        return "".join(random.choices(string.digits, k=length))

    # Alphanumeric
    if symbology_lower in ["code128", "code39", "code93"]:
        length = random.randint(6, 12)
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

    # QR/DataMatrix can handle more data
    if symbology_lower in ["qr", "datamatrix", "aztec"]:
        length = random.randint(10, 30)
        chars = string.ascii_letters + string.digits + " -_."
        return "".join(random.choices(chars, k=length))

    # Default
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
