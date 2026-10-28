
import json
import re
import difflib
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Explicit path fixes TesseractNotFoundError on Windows.
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

EXPECTED_CLASSES = [
    "User",
    "Admin",
    "Student",
    "Rating",
    "Notification",
    "Payment",
    "Transaction",
    "Wallet",
    "DigitalCoin",
    "Listing",
    "Product",
    "Service",
]

CLASS_ALIASES = {
    "user": "User",
    "users": "User",
    "admin": "Admin",
    "administrator": "Admin",
    "student": "Student",
    "students": "Student",
    "rating": "Rating",
    "ratings": "Rating",
    "notification": "Notification",
    "notifications": "Notification",
    "payment": "Payment",
    "payments": "Payment",
    "transaction": "Transaction",
    "transactions": "Transaction",
    "wallet": "Wallet",
    "wallets": "Wallet",
    "digitalcoin": "DigitalCoin",
    "digitalcoins": "DigitalCoin",
    "listing": "Listing",
    "listings": "Listing",
    "product": "Product",
    "products": "Product",
    "service": "Service",
    "services": "Service",
}

RELATIONSHIP_WORDS = {
    "extends", "implements", "association", "composition",
    "aggregation", "inheritance", "dependency",
}

# Set True to save the processed image and OCR overlay.
SAVE_DIAGNOSTICS = True


# ---------------------------------------------------------
# IMAGE LOADING AND PREPROCESSING
# ---------------------------------------------------------

def load_image(image_path):
    """Load an image, including paths containing Windows characters."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"UML image not found: {path}")

    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def prepare_image(image):
    """Enlarge small UML images and create a high-contrast OCR image."""
    height, width = image.shape[:2]

    # Upscale small screenshots so class names are easier to recognize.
    scale = 2.0 if max(height, width) < 2200 else 1.0

    if scale != 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast without requiring the original image to be pure white.
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)

    # Light denoising helps OCR on compressed screenshots.
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Adaptive thresholding handles uneven backgrounds.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return image, gray, binary


# ---------------------------------------------------------
# TEXT NORMALIZATION AND CLASS-NAME DETECTION
# ---------------------------------------------------------

def normalize_text(text):
    """Normalize OCR text for comparison."""
    text = str(text or "").strip().lower()
    text = text.replace("0", "o")
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def normalize_class_name(text):
    """Return the canonical class name if text matches a known alias."""
    cleaned = normalize_text(text)

    if cleaned in CLASS_ALIASES:
        return CLASS_ALIASES[cleaned]

    # Match common OCR mistakes only when the match is sufficiently close.
    candidates = list(CLASS_ALIASES.keys())
    matches = difflib.get_close_matches(
        cleaned,
        candidates,
        n=1,
        cutoff=0.84,
    )

    if matches:
        return CLASS_ALIASES[matches[0]]

    return None


def group_ocr_words(data):
    """Group positioned OCR words into text lines."""
    lines = {}

    count = len(data["text"])

    for i in range(count):
        word = str(data["text"][i]).strip()

        if not word:
            continue

        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1

        if confidence < 0:
            continue

        key = (
            data["page_num"][i],
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i],
        )

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        if key not in lines:
            lines[key] = {
                "words": [],
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "confidence": [],
            }

        line = lines[key]
        line["words"].append((x, word))
        line["x1"] = min(line["x1"], x)
        line["y1"] = min(line["y1"], y)
        line["x2"] = max(line["x2"], x + w)
        line["y2"] = max(line["y2"], y + h)
        line["confidence"].append(confidence)

    result = []

    for line in lines.values():
        line["words"].sort(key=lambda item: item[0])
        line["text"] = " ".join(word for _, word in line["words"])
        line["x_center"] = (line["x1"] + line["x2"]) // 2
        line["y_center"] = (line["y1"] + line["y2"]) // 2
        result.append(line)

    result.sort(key=lambda item: (item["y_center"], item["x1"]))
    return result


def detect_heading(line_text):
    """
    Detect a class heading from a complete OCR line.

    Exact aliases are preferred. Fuzzy matching is used only as a fallback.
    """
    text = str(line_text or "").strip()

    # Strip common UML heading decoration and punctuation.
    text = re.sub(r"^[<+~#\-*\s]+", "", text)
    text = re.sub(r"[>+~#\-*\s:;]+$", "", text)

    # A heading should be short; avoid matching member descriptions.
    if len(text.split()) > 3 or len(text) > 35:
        return None

    # First try the whole line.
    result = normalize_class_name(text)
    if result:
        return result

    # Then try individual tokens, allowing OCR to add punctuation.
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", text)

    if len(tokens) == 1:
        return normalize_class_name(tokens[0])

    # Handle cases like "Digital Coin" or "Digital_Coin".
    compact = "".join(tokens)
    return normalize_class_name(compact)


def find_class_headings(lines):
    """Find recognized class headings and avoid duplicate detections."""
    headings = []

    for line in lines:
        class_name = detect_heading(line["text"])

        if class_name is None:
            continue

        headings.append({
            "name": class_name,
            "x": line["x_center"],
            "y": line["y_center"],
            "x1": line["x1"],
            "x2": line["x2"],
            "y1": line["y1"],
            "y2": line["y2"],
            "text": line["text"],
        })

    # Remove duplicate OCR detections of the same heading at nearly the
    # same position, keeping the first detection.
    unique = []

    for heading in headings:
        duplicate = any(
            old["name"] == heading["name"]
            and abs(old["x"] - heading["x"]) < 30
            and abs(old["y"] - heading["y"]) < 30
            for old in unique
        )

        if not duplicate:
            unique.append(heading)

    unique.sort(key=lambda item: (item["y"], item["x"]))
    return unique


# ---------------------------------------------------------
# ASSIGN OCR LINES TO CLASSES
# ---------------------------------------------------------

def cluster_columns(headings, image_width):
    """
    Group headings into approximate visual columns.

    This is a geometric heuristic, not full UML rectangle detection.
    """
    if not headings:
        return []

    threshold = max(100, int(image_width * 0.12))
    centers = []

    for heading in sorted(headings, key=lambda item: item["x"]):
        chosen = None

        for center in centers:
            if abs(heading["x"] - center["x"]) <= threshold:
                chosen = center
                break

        if chosen is None:
            chosen = {"x": heading["x"], "headings": []}
            centers.append(chosen)

        chosen["headings"].append(heading)
        chosen["x"] = sum(
            item["x"] for item in chosen["headings"]
        ) // len(chosen["headings"])

    for column_index, column in enumerate(centers):
        for heading in column["headings"]:
            heading["column"] = column_index

    return centers


def clean_member(text):
    """Clean an OCR line before adding it as a class member."""
    text = str(text or "").strip()

    # Remove bullets and common UML visibility symbols at the start.
    text = re.sub(r"^[\s•·|]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" |")

    if not text:
        return None

    if normalize_class_name(text):
        return None

    normalized = normalize_text(text)

    if normalized in RELATIONSHIP_WORDS:
        return None

    # Ignore obvious page artifacts.
    if len(normalized) < 2:
        return None

    return text


def assign_lines_to_classes(lines, headings, image_width):
    """Assign text lines to the closest heading in the same column."""
    if not headings:
        return []

    columns = cluster_columns(headings, image_width)

    for column in columns:
        column["headings"].sort(key=lambda item: item["y"])

    # Prepare each class record.
    records = {}

    for heading in headings:
        records[heading["name"]] = {
            "name": heading["name"],
            "attributes": [],
            "methods": [],
            "relationships": [],
            "_heading": heading,
            "_members": [],
        }

    for line in lines:
        # Do not assign a heading line as a member.
        if detect_heading(line["text"]) is not None:
            continue

        # Choose the column whose center is closest to the line.
        column = min(
            columns,
            key=lambda c: abs(c["x"] - line["x_center"]),
        )

        column_headings = column["headings"]
        eligible = []

        for index, heading in enumerate(column_headings):
            if line["y_center"] <= heading["y2"]:
                continue

            # Stop at the next heading in this column.
            if index + 1 < len(column_headings):
                next_heading = column_headings[index + 1]
                if line["y_center"] >= next_heading["y1"]:
                    continue

            eligible.append(heading)

        if not eligible:
            continue

        # Use the most recent heading above this line.
        heading = max(eligible, key=lambda item: item["y"])
        records[heading["name"]]["_members"].append(line)

    # Convert member text into a simple model.
    for record in records.values():
        for line in record.pop("_members"):
            text = clean_member(line["text"])

            if not text:
                continue

            lower = text.lower()

            if any(word in lower for word in (
                "inherits", "extends", "implements",
                "association", "aggregation", "composition",
            )):
                record["relationships"].append(text)
                continue

            # UML method signatures normally contain parentheses.
            if "(" in text and ")" in text:
                record["methods"].append(text)
            else:
                record["attributes"].append(text)

        for field in ("attributes", "methods", "relationships"):
            record[field] = list(dict.fromkeys(record[field]))

    # Return only classes actually detected in the image.
    ordered_names = [h["name"] for h in headings]
    seen = set()
    components = []

    for name in ordered_names:
        if name not in seen:
            components.append(records[name])
            seen.add(name)

    return components


# ---------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------

def save_diagnostics(image, lines, headings, output_dir):
    """Save the OCR image and an overlay showing detected headings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / "uml_detection_debug.png"
    cv2.imwrite(str(image_path), image)

    overlay = image.copy()

    for line in lines:
        cv2.rectangle(
            overlay,
            (line["x1"], line["y1"]),
            (line["x2"], line["y2"]),
            (0, 180, 0),
            1,
        )

    for heading in headings:
        cv2.rectangle(
            overlay,
            (heading["x1"], heading["y1"]),
            (heading["x2"], heading["y2"]),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            overlay,
            heading["name"],
            (heading["x1"], max(20, heading["y1"] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    overlay_path = output_dir / "uml_ocr_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "processed_image": str(image_path),
        "ocr_overlay": str(overlay_path),
    }


# ---------------------------------------------------------
# MAIN EXTRACTION FUNCTION
# ---------------------------------------------------------

def extract_uml_model(image_path, show_diagnostics=False):
    """
    Extract a UML model from an image.

    Returns:
        {
            "model": {
                "components": [...],
                "missing_expected_classes": [...],
                "review_required": bool,
                "warning": str
            },
            "diagnostics": {...},
            "ocr_text": str
        }
    """
    # Verify the OCR executable before starting.
    if not Path(TESSERACT_PATH).is_file():
        raise FileNotFoundError(
            "Tesseract executable was not found at: "
            + TESSERACT_PATH
        )

    original = load_image(image_path)
    working, gray, binary = prepare_image(original)

    # Try multiple OCR modes. PSM 11 is good for scattered text;
    # PSM 6 is useful for structured UML diagrams.
    all_lines = []
    ocr_texts = []
    errors = []

    for mode, ocr_image in ((11, binary), (6, gray)):
        try:
            data = pytesseract.image_to_data(
                ocr_image,
                config=f"--oem 3 --psm {mode}",
                output_type=Output.DICT,
            )

            lines = group_ocr_words(data)
            all_lines.append(lines)

            text = pytesseract.image_to_string(
                ocr_image,
                config=f"--oem 3 --psm {mode}",
            )
            ocr_texts.append(text)

        except Exception as exc:
            errors.append(f"PSM {mode}: {exc}")

    # Prefer the OCR pass that recognized more text lines.
    lines = max(all_lines, key=len) if all_lines else []
    ocr_text = max(ocr_texts, key=len) if ocr_texts else ""

    headings = find_class_headings(lines)

    components = assign_lines_to_classes(
        lines,
        headings,
        working.shape[1],
    )

    detected_names = [item["name"] for item in components]
    missing = [
        name for name in EXPECTED_CLASSES
        if name not in detected_names
    ]

    warning = None

    if not components:
        warning = (
            "No UML class headings were recognized. "
            "Check the OCR overlay and OCR text. "
            "Make sure class names are readable in the image."
        )
    elif missing:
        warning = (
            "Some expected classes were not detected: "
            + ", ".join(missing)
            + ". Check the OCR overlay and review the extracted model."
        )

    model = {
        "components": components,
        "missing_expected_classes": missing,
        "review_required": bool(missing) or not bool(components),
        "warning": warning,
    }

    diagnostics = {
        "image_path": str(image_path),
        "tesseract_path": TESSERACT_PATH,
        "ocr_line_count": len(lines),
        "detected_class_count": len(components),
        "detected_class_names": detected_names,
        "missing_expected_classes": missing,
        "errors": errors,
    }

    if show_diagnostics or SAVE_DIAGNOSTICS:
        output_dir = Path(image_path).resolve().parent
        diagnostics.update(
            save_diagnostics(
                working,
                lines,
                headings,
                output_dir,
            )
        )

    return {
        "model": model,
        "diagnostics": diagnostics,
        "ocr_text": ocr_text,
    }


# ---------------------------------------------------------
# COMMAND-LINE TEST
# ---------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract UML classes from an image."
    )
    parser.add_argument("image", help="Path to UML image")
    parser.add_argument(
        "--output",
        default="extracted_uml_model.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    result = extract_uml_model(
        args.image,
        show_diagnostics=True,
    )

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result["diagnostics"], indent=2))
    print("\nExtracted model:")
    print(json.dumps(result["model"], indent=2, ensure_ascii=False))
    print(f"\nSaved JSON to: {output_path.resolve()}")
