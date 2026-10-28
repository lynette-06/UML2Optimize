
from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
import traceback
from pathlib import Path
from typing import Any

import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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

# ============================================================
# OPTIONAL IMPORTS
# ============================================================

try:
    import pytesseract
    import cv2
    import numpy as np
    from PIL import Image
    from pytesseract import Output

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    OCR_IMPORT_ERROR = None
except Exception as exc:
    pytesseract = None
    cv2 = None
    np = None
    Image = None
    Output = None
    OCR_IMPORT_ERROR = str(exc)

try:
    from uml_image_extractor import extract_uml_model
    EXTRACTOR_IMPORT_ERROR = None
except Exception as exc:
    extract_uml_model = None
    EXTRACTOR_IMPORT_ERROR = str(exc)

try:
    from compiler.code_generator import generate_code
    CODEGEN_IMPORT_ERROR = None
except Exception as exc:
    generate_code = None
    CODEGEN_IMPORT_ERROR = str(exc)

try:
    from compiler.parser import parse_model
    PARSER_IMPORT_ERROR = None
except Exception as exc:
    parse_model = None
    PARSER_IMPORT_ERROR = str(exc)

try:
    from optimizer.optimizer import optimize_code
    OPTIMIZER_IMPORT_ERROR = None
except Exception as exc:
    optimize_code = None
    OPTIMIZER_IMPORT_ERROR = str(exc)

try:
    from ml_model.predict import predict_optimization
    ML_IMPORT_ERROR = None
except Exception as exc:
    predict_optimization = None
    ML_IMPORT_ERROR = str(exc)

try:
    from actual_code_benchmark import benchmark_actual_code
    BENCHMARK_IMPORT_ERROR = None
except Exception as exc:
    benchmark_actual_code = None
    BENCHMARK_IMPORT_ERROR = str(exc)


# ============================================================
# MODEL AND SESSION-STATE HELPERS
# ============================================================

EMPTY_MODEL = {
    "components": [],
    "missing_expected_classes": EXPECTED_CLASSES.copy(),
    "review_required": True,
    "warning": "Upload a UML image to start.",
}


def normalize_text(value: Any) -> str:
    """Normalize OCR text for comparison with known class names."""
    text = str(value or "").strip().lower()
    text = text.replace("0", "o")
    return "".join(ch for ch in text if ch.isalnum())


def class_from_text(text: str) -> str | None:
    """Recognize a known class name, allowing minor OCR spelling errors."""
    cleaned = normalize_text(text)

    if cleaned in CLASS_ALIASES:
        return CLASS_ALIASES[cleaned]

    if not cleaned:
        return None

    candidates = difflib.get_close_matches(
        cleaned,
        list(CLASS_ALIASES.keys()),
        n=1,
        cutoff=0.86,
    )

    if candidates:
        return CLASS_ALIASES[candidates[0]]

    return None


def ensure_model(value: Any) -> dict:
    """Normalize extractor output to the model format used by this app."""
    if not isinstance(value, dict):
        value = {}

    # Support extractor output shaped like:
    # {"model": {...}, "diagnostics": {...}, "ocr_text": "..."}
    if isinstance(value.get("model"), dict):
        value = value["model"]

    if isinstance(value, list):
        value = {"components": value}

    components = value.get("components", [])

    if not isinstance(components, list):
        components = []

    normalized_components = []

    for component in components:
        if not isinstance(component, dict):
            continue

        name = str(
            component.get("name", component.get("class_name", "Unknown"))
        ).strip()

        def as_list(field):
            items = component.get(field, [])

            if isinstance(items, str):
                items = items.splitlines()

            if not isinstance(items, list):
                return []

            return [
                str(item).strip()
                for item in items
                if str(item).strip()
            ]

        normalized_components.append({
            "name": name or "Unknown",
            "attributes": as_list("attributes"),
            "methods": as_list("methods"),
            "relationships": as_list("relationships"),
        })

    result = dict(value)
    result["components"] = normalized_components

    names = {item["name"] for item in normalized_components}

    result["missing_expected_classes"] = [
        name for name in EXPECTED_CLASSES if name not in names
    ]
    result["review_required"] = (
        not bool(normalized_components)
        or bool(result["missing_expected_classes"])
    )

    result.setdefault("warning", None)

    return result


def count_attributes(model: dict) -> int:
    return sum(
        len(item.get("attributes", []))
        for item in model.get("components", [])
    )


def count_methods(model: dict) -> int:
    return sum(
        len(item.get("methods", []))
        for item in model.get("components", [])
    )


def commit_model(model: dict) -> None:
    """
    Save the model and change the JSON editor's key.

    Changing the key makes Streamlit build a fresh editor using the latest
    model instead of retaining stale text from a previous extraction.
    """
    st.session_state["uml_model"] = ensure_model(model)
    st.session_state["model_revision"] = (
        st.session_state.get("model_revision", 0) + 1
    )

    # Previous generated results no longer correspond to the edited model.
    st.session_state["generated_code"] = ""
    st.session_state["optimized_code"] = ""
    st.session_state["benchmark_results"] = None
    st.session_state["benchmark_error"] = None
    st.session_state["ml_prediction"] = None
    st.session_state["ml_prediction_error"] = None


def safe_compile(code: str) -> tuple[bool, str]:
    """Check whether generated text is valid Python syntax."""
    try:
        compile(code, "<generated_code>", "exec")
        return True, "Python syntax check passed."
    except SyntaxError as exc:
        return False, f"Syntax error on line {exc.lineno}: {exc.msg}"
    except Exception as exc:
        return False, str(exc)


# ============================================================
# INDEPENDENT OCR FALLBACK
# ============================================================

def fallback_ocr_extract(image_path: str) -> tuple[dict, dict, str]:
    """
    Perform OCR independently of uml_image_extractor.py.

    This attempts to recover class headings when the main extractor fails.
    It is a heuristic and does not fully detect UML rectangle boundaries.
    """
    if OCR_IMPORT_ERROR:
        raise RuntimeError(
            f"OCR dependencies could not be imported: {OCR_IMPORT_ERROR}"
        )

    if not Path(TESSERACT_PATH).is_file():
        raise FileNotFoundError(
            f"Tesseract executable not found: {TESSERACT_PATH}"
        )

    raw_image = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not open image: {image_path}")

    height, width = image.shape[:2]

    if max(height, width) < 2200:
        image = cv2.resize(
            image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Try multiple page-segmentation modes.
    candidates = []
    raw_texts = []

    for mode, source in ((11, binary), (6, gray), (11, gray)):
        config = f"--oem 3 --psm {mode}"

        text = pytesseract.image_to_string(source, config=config)
        raw_texts.append(text)

        data = pytesseract.image_to_data(
            source,
            config=config,
            output_type=Output.DICT,
        )

        grouped = {}

        for i, word in enumerate(data["text"]):
            word = str(word).strip()

            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                confidence = -1

            if not word or confidence < 0:
                continue

            key = (
                data["page_num"][i],
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i],
            )

            x = int(data["left"][i])
            y = int(data["top"][i])
            word_width = int(data["width"][i])
            word_height = int(data["height"][i])

            if key not in grouped:
                grouped[key] = {
                    "words": [],
                    "x1": x,
                    "y1": y,
                    "x2": x + word_width,
                    "y2": y + word_height,
                }

            row = grouped[key]
            row["words"].append((x, word))
            row["x1"] = min(row["x1"], x)
            row["y1"] = min(row["y1"], y)
            row["x2"] = max(row["x2"], x + word_width)
            row["y2"] = max(row["y2"], y + word_height)

        lines = []

        for row in grouped.values():
            row["words"].sort(key=lambda pair: pair[0])
            row["text"] = " ".join(word for _, word in row["words"])
            row["xc"] = (row["x1"] + row["x2"]) // 2
            row["yc"] = (row["y1"] + row["y2"]) // 2
            lines.append(row)

        lines.sort(key=lambda row: (row["yc"], row["x1"]))
        candidates.append((lines, text))

    lines, _ = max(candidates, key=lambda item: len(item[0]))
    raw_text = max(raw_texts, key=len) if raw_texts else ""

    # Find class headings.
    headings = []

    for line in lines:
        text = line["text"].strip().strip("|:;<>+-* ")

        # A class heading should normally be a short text line.
        if len(text) > 36:
            continue

        class_name = class_from_text(text)

        if class_name is None:
            # Sometimes OCR places a class name beside a short token.
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text):
                class_name = class_from_text(token)
                if class_name:
                    break

        if class_name is None:
            continue

        duplicate = any(
            old["name"] == class_name
            and abs(old["x"] - line["xc"]) < 35
            and abs(old["y"] - line["yc"]) < 35
            for old in headings
        )

        if not duplicate:
            headings.append({
                "name": class_name,
                "x": line["xc"],
                "y": line["yc"],
                "x1": line["x1"],
                "x2": line["x2"],
                "y1": line["y1"],
                "y2": line["y2"],
            })

    # Group headings by approximate horizontal columns.
    columns = []
    column_gap = max(100, int(image.shape[1] * 0.12))

    for heading in sorted(headings, key=lambda item: item["x"]):
        column = next(
            (
                item for item in columns
                if abs(item["x"] - heading["x"]) <= column_gap
            ),
            None,
        )

        if column is None:
            column = {"x": heading["x"], "headings": []}
            columns.append(column)

        column["headings"].append(heading)
        column["x"] = sum(
            item["x"] for item in column["headings"]
        ) // len(column["headings"])

    for column in columns:
        column["headings"].sort(key=lambda item: item["y"])

    records = {}

    for heading in headings:
        records.setdefault(
            heading["name"],
            {
                "name": heading["name"],
                "attributes": [],
                "methods": [],
                "relationships": [],
            },
        )

    # Assign lines to the nearest heading above them within an approximate
    # column. This is not a replacement for actual UML box detection.
    for line in lines:
        line_text = line["text"].strip().strip("|:;<>+-* ")

        if class_from_text(line_text):
            continue

        if not columns:
            continue

        column = min(
            columns,
            key=lambda item: abs(item["x"] - line["xc"]),
        )

        eligible = []

        for index, heading in enumerate(column["headings"]):
            if line["yc"] <= heading["y2"]:
                continue

            if (
                index + 1 < len(column["headings"])
                and line["yc"] >= column["headings"][index + 1]["y1"]
            ):
                continue

            eligible.append(heading)

        if not eligible:
            continue

        heading = max(eligible, key=lambda item: item["y"])

        member = " ".join(line["text"].split()).strip(" |•·")

        if len(normalize_text(member)) < 2:
            continue

        record = records[heading["name"]]
        lower = member.lower()

        if any(
            word in lower
            for word in (
                "extends",
                "implements",
                "association",
                "composition",
                "aggregation",
                "inherits",
            )
        ):
            record["relationships"].append(member)
        elif "(" in member and ")" in member:
            record["methods"].append(member)
        else:
            record["attributes"].append(member)

    components = []
    seen = set()

    for heading in sorted(headings, key=lambda item: (item["y"], item["x"])):
        name = heading["name"]

        if name in seen:
            continue

        seen.add(name)

        for field in ("attributes", "methods", "relationships"):
            records[name][field] = list(
                dict.fromkeys(records[name][field])
            )

        components.append(records[name])

    found_names = [component["name"] for component in components]
    missing = [
        name for name in EXPECTED_CLASSES if name not in found_names
    ]

    model = {
        "components": components,
        "missing_expected_classes": missing,
        "review_required": bool(missing) or not bool(components),
        "warning": (
            None if components
            else "OCR ran, but no known class headings were recognized."
        ),
    }

    # Save a visual overlay showing detected class headings.
    overlay = image.copy()

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
        )

    overlay_path = (
        Path(image_path).resolve().parent
        / "uml_ocr_fallback_overlay.png"
    )
    cv2.imwrite(str(overlay_path), overlay)

    diagnostics = {
        "method": "independent OCR fallback",
        "ocr_line_count": len(lines),
        "detected_class_count": len(components),
        "detected_class_names": found_names,
        "missing_expected_classes": missing,
        "overlay": str(overlay_path),
    }

    return model, diagnostics, raw_text


# ============================================================
# CODE GENERATION AND OPTIMIZATION HELPERS
# ============================================================

def generate_python(model: dict) -> str:
    """Use the project's generator, optionally passing through its parser."""
    if generate_code is None:
        raise RuntimeError(
            "Code generator could not be imported: "
            f"{CODEGEN_IMPORT_ERROR}"
        )

    # First try the existing parser workflow, if available.
    if parse_model is not None:
        model_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
            ) as file:
                json.dump(model, file, indent=2, ensure_ascii=False)
                model_path = file.name

            parsed = parse_model(model_path)

            if parsed is not None:
                try:
                    output = generate_code(parsed)
                    if isinstance(output, str):
                        return output
                except Exception:
                    pass

        except Exception:
            pass

        finally:
            if model_path:
                try:
                    os.unlink(model_path)
                except OSError:
                    pass

    # Fall back to passing the dictionary directly to the generator.
    output = generate_code(model)

    if not isinstance(output, str):
        output = str(output)

    return output


def save_code_download(label: str, code: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=code,
        file_name=filename,
        mime="text/x-python",
    )


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="UML2Optimize",
    page_icon="🧩",
    layout="wide",
)

st.title("UML2Optimize")
st.caption(
    "Extract UML classes, review the model, generate Python code, "
    "and inspect optimization results."
)

# Initialize session state before rendering tabs.
if "uml_model" not in st.session_state:
    st.session_state["uml_model"] = EMPTY_MODEL.copy()

if "model_revision" not in st.session_state:
    st.session_state["model_revision"] = 0

if "ocr_diagnostics" not in st.session_state:
    st.session_state["ocr_diagnostics"] = {}

if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""

if "generated_code" not in st.session_state:
    st.session_state["generated_code"] = ""

if "optimized_code" not in st.session_state:
    st.session_state["optimized_code"] = ""

if "benchmark_results" not in st.session_state:
    st.session_state["benchmark_results"] = None

if "benchmark_error" not in st.session_state:
    st.session_state["benchmark_error"] = None

if "ml_prediction" not in st.session_state:
    st.session_state["ml_prediction"] = None

if "ml_prediction_error" not in st.session_state:
    st.session_state["ml_prediction_error"] = None


upload_tab, review_tab, optimize_tab = st.tabs([
    "1. Upload UML",
    "2. Review Model",
    "3. Optimize & Results",
])


# ============================================================
# TAB 1 — UPLOAD UML
# ============================================================

with upload_tab:
    st.subheader("Upload your UML class diagram")

    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG, JPEG, or BMP UML image",
        type=["png", "jpg", "jpeg", "bmp"],
        key="uml_image_upload",
    )

    if OCR_IMPORT_ERROR:
        st.error(f"OCR libraries could not be imported: {OCR_IMPORT_ERROR}")

    if not Path(TESSERACT_PATH).is_file():
        st.error(f"Tesseract executable not found: {TESSERACT_PATH}")
    else:
        st.caption(f"Tesseract path: {TESSERACT_PATH}")

    if EXTRACTOR_IMPORT_ERROR:
        st.warning(
            "The main UML extractor could not be imported. "
            "The app will try its independent OCR fallback. "
            f"Details: {EXTRACTOR_IMPORT_ERROR}"
        )

    if uploaded_file is not None:
        try:
            if Image is not None:
                preview = Image.open(uploaded_file)
                st.image(
                    preview,
                    caption="Uploaded UML diagram",
                    use_container_width=True,
                )
        except Exception as exc:
            st.warning(f"Could not display image preview: {exc}")

        if st.button(
            "Extract UML Classes",
            type="primary",
            use_container_width=True,
            key="extract_uml_button",
        ):
            temp_path = None

            with st.spinner("Processing UML image..."):
                try:
                    suffix = Path(uploaded_file.name).suffix or ".png"

                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        delete=False,
                        suffix=suffix,
                    ) as temporary_file:
                        temporary_file.write(uploaded_file.getvalue())
                        temp_path = temporary_file.name

                    model = ensure_model({"components": []})
                    diagnostics = {}
                    ocr_text = ""
                    extractor_error = None

                    # First try the project's main extractor.
                    if extract_uml_model is not None:
                        try:
                            result = extract_uml_model(
                                temp_path,
                                show_diagnostics=True,
                            )

                            if isinstance(result, dict):
                                candidate = result.get("model", result)
                                model = ensure_model(candidate)

                                diagnostic_value = result.get(
                                    "diagnostics", {}
                                )
                                if isinstance(diagnostic_value, dict):
                                    diagnostics.update(diagnostic_value)

                                ocr_text = str(result.get("ocr_text", ""))

                            else:
                                model = ensure_model(result)

                        except Exception as exc:
                            extractor_error = (
                                f"{type(exc).__name__}: {exc}"
                            )

                    # If no classes were found, try independent OCR.
                    if not model.get("components"):
                        try:
                            (
                                fallback_model,
                                fallback_diagnostics,
                                fallback_text,
                            ) = fallback_ocr_extract(temp_path)

                            diagnostics.update(fallback_diagnostics)
                            diagnostics["fallback_used"] = True

                            if not ocr_text:
                                ocr_text = fallback_text

                            if fallback_model.get("components"):
                                model = ensure_model(fallback_model)
                            else:
                                diagnostics["fallback_note"] = (
                                    "The fallback OCR found no recognized "
                                    "class headings. Check the raw OCR text "
                                    "and diagnostic overlay."
                                )

                        except Exception as exc:
                            diagnostics["fallback_error"] = (
                                f"{type(exc).__name__}: {exc}"
                            )

                    if extractor_error:
                        diagnostics["main_extractor_error"] = extractor_error

                    diagnostics["final_class_count"] = len(
                        model.get("components", [])
                    )
                    diagnostics["final_class_names"] = [
                        item["name"] for item in model.get("components", [])
                    ]
                    diagnostics["final_attribute_count"] = count_attributes(model)
                    diagnostics["final_method_count"] = count_methods(model)

                    # This increments the editor revision so the Review Model
                    # tab will show the newly extracted model, not old JSON.
                    commit_model(model)

                    st.session_state["ocr_diagnostics"] = diagnostics
                    st.session_state["ocr_text"] = ocr_text

                except Exception as exc:
                    st.error(
                        f"Extraction failed: {type(exc).__name__}: {exc}"
                    )
                    st.code(traceback.format_exc())

                finally:
                    if temp_path:
                        try:
                            os.unlink(temp_path)
                        except OSError:
                            pass

    # Display latest state.
    model = ensure_model(st.session_state["uml_model"])
    components = model.get("components", [])

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Classes detected", len(components))
    col2.metric("Attributes detected", count_attributes(model))
    col3.metric("Methods detected", count_methods(model))

    if components:
        st.success(
            f"Loaded {len(components)} classes. Review them in the "
            "Review Model tab before generating code."
        )
        st.write(
            "Detected class names:",
            ", ".join(item["name"] for item in components),
        )
    else:
        st.warning(
            "No classes are available yet. Check the diagnostics and raw "
            "OCR text below, or add classes manually in Review Model."
        )

    if model.get("warning"):
        st.info(str(model["warning"]))

    if st.session_state["ocr_diagnostics"]:
        with st.expander(
            "Extraction diagnostics",
            expanded=not bool(components),
        ):
            st.json(st.session_state["ocr_diagnostics"])

    if st.session_state["ocr_text"]:
        with st.expander(
            "Raw OCR text",
            expanded=not bool(components),
        ):
            st.text_area(
                "Text recognized from your image",
                value=st.session_state["ocr_text"],
                height=260,
                key="raw_ocr_text_box",
            )


# ============================================================
# TAB 2 — REVIEW MODEL
# ============================================================

with review_tab:
    st.subheader("Review and edit the extracted UML model")

    model = ensure_model(st.session_state["uml_model"])

    # Changing the key whenever the model changes prevents a stale JSON
    # text area from hiding freshly extracted or manually added classes.
    editor_key = f"model_json_editor_{st.session_state['model_revision']}"

    edited_json = st.text_area(
        "Model JSON — edit class names, attributes, methods, and relationships",
        value=json.dumps(model, indent=2, ensure_ascii=False),
        height=420,
        key=editor_key,
    )

    col_save, col_starter = st.columns(2)

    with col_save:
        if st.button(
            "Validate and Save Model",
            type="primary",
            use_container_width=True,
            key="validate_save_model",
        ):
            try:
                parsed_json = json.loads(edited_json)

                if not isinstance(parsed_json, dict):
                    raise ValueError("The JSON root must be an object.")

                if not isinstance(parsed_json.get("components", []), list):
                    raise ValueError(
                        'The model must contain a "components" list.'
                    )

                parsed_model = ensure_model(parsed_json)
                commit_model(parsed_model)

                st.success(
                    f"Saved {len(parsed_model['components'])} classes, "
                    f"{count_attributes(parsed_model)} attributes, and "
                    f"{count_methods(parsed_model)} methods."
                )
                st.rerun()

            except Exception as exc:
                st.error(f"Could not save model JSON: {exc}")

    with col_starter:
        if st.button(
            "Load 12-Class Starter Model",
            use_container_width=True,
            key="load_starter_model",
        ):
            starter = {
                "components": [
                    {
                        "name": name,
                        "attributes": [],
                        "methods": [],
                        "relationships": [],
                    }
                    for name in EXPECTED_CLASSES
                ],
                "missing_expected_classes": [],
                "review_required": True,
                "warning": (
                    "Starter class names only. Fill in the actual attributes "
                    "and methods from your UML diagram."
                ),
            }

            commit_model(starter)
            st.success("Starter model loaded.")
            st.rerun()

    st.divider()
    st.markdown("### Add or update a class manually")

    with st.form("manual_class_form", clear_on_submit=True):
        col_name, col_attributes = st.columns([1, 2])

        with col_name:
            new_class_name = st.text_input(
                "Class name",
                placeholder="e.g. User",
            )

        with col_attributes:
            new_attributes = st.text_area(
                "Attributes — one per line",
                placeholder="user_id: int\nname: str",
                height=100,
            )

        new_methods = st.text_area(
            "Methods — one per line",
            placeholder="login()\nupdate_profile(name)",
            height=90,
        )

        new_relationships = st.text_area(
            "Relationships (optional)",
            placeholder="Student inherits from User",
            height=70,
        )

        add_class_button = st.form_submit_button(
            "Add or Update Class",
            use_container_width=True,
        )

        if add_class_button:
            if not new_class_name.strip():
                st.error("Please enter a class name.")
            else:
                current_model = ensure_model(st.session_state["uml_model"])
                current_components = current_model["components"]

                name_to_save = new_class_name.strip()

                new_record = {
                    "name": name_to_save,
                    "attributes": [
                        line.strip()
                        for line in new_attributes.splitlines()
                        if line.strip()
                    ],
                    "methods": [
                        line.strip()
                        for line in new_methods.splitlines()
                        if line.strip()
                    ],
                    "relationships": [
                        line.strip()
                        for line in new_relationships.splitlines()
                        if line.strip()
                    ],
                }

                existing = next(
                    (
                        component
                        for component in current_components
                        if component["name"].casefold()
                        == name_to_save.casefold()
                    ),
                    None,
                )

                if existing is None:
                    current_components.append(new_record)
                else:
                    existing.update(new_record)

                commit_model(current_model)
                st.success(f"Class '{name_to_save}' saved.")
                st.rerun()

    # Always show the current committed model.
    model = ensure_model(st.session_state["uml_model"])
    components = model.get("components", [])

    st.divider()
    st.markdown("### Current model summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Classes", len(components))
    col2.metric("Attributes", count_attributes(model))
    col3.metric("Methods", count_methods(model))

    if components:
        for component in components:
            with st.expander(
                f"{component['name']} — "
                f"{len(component['attributes'])} attributes, "
                f"{len(component['methods'])} methods"
            ):
                st.write("**Attributes**")
                st.write(component["attributes"] or "None")

                st.write("**Methods**")
                st.write(component["methods"] or "None")

                st.write("**Relationships**")
                st.write(component["relationships"] or "None")

        st.download_button(
            "Download UML Model JSON",
            data=json.dumps(model, indent=2, ensure_ascii=False),
            file_name="extracted_uml_model.json",
            mime="application/json",
            key="download_model_json",
        )
    else:
        st.info(
            "The model is empty. Try extraction in Upload UML, "
            "or add your first class above."
        )


# ============================================================
# TAB 3 — GENERATE, OPTIMIZE, AND REVIEW RESULTS
# ============================================================

with optimize_tab:
    st.subheader("Generate and optimize Python code")

    model = ensure_model(st.session_state["uml_model"])
    components = model.get("components", [])

    st.write(
        f"**Current model:** {len(components)} classes, "
        f"{count_attributes(model)} attributes, "
        f"{count_methods(model)} methods."
    )

    if not components:
        st.warning(
            "The UML model is empty. Extract classes or add classes "
            "in the Review Model tab before generating code."
        )

    if CODEGEN_IMPORT_ERROR:
        st.error(
            "Code generator could not be imported: "
            f"{CODEGEN_IMPORT_ERROR}"
        )

    if st.button(
        "Generate, Optimize and Benchmark",
        type="primary",
        use_container_width=True,
        key="run_optimization",
    ):
        if not components:
            st.error(
                "There are no classes to generate. Add or extract classes first."
            )

        elif generate_code is None:
            st.error(
                f"Code generator is unavailable: {CODEGEN_IMPORT_ERROR}"
            )

        else:
            with st.spinner("Generating and optimizing Python code..."):
                try:
                    original_code = generate_python(model)
                    st.session_state["generated_code"] = original_code

                    original_ok, original_message = safe_compile(original_code)
                    st.session_state["original_syntax"] = {
                        "ok": original_ok,
                        "message": original_message,
                    }

                    # Run AST optimizer if available.
                    if optimize_code is not None:
                        try:
                            result = optimize_code(original_code)
                            optimized_code = (
                                result if isinstance(result, str)
                                else str(result)
                            )
                        except Exception as exc:
                            optimized_code = original_code
                            st.warning(
                                "Optimization failed; the original code "
                                "will be displayed instead. "
                                f"{type(exc).__name__}: {exc}"
                            )
                    else:
                        optimized_code = original_code
                        st.warning(
                            "Optimizer unavailable: "
                            f"{OPTIMIZER_IMPORT_ERROR}"
                        )

                    st.session_state["optimized_code"] = optimized_code

                    optimized_ok, optimized_message = safe_compile(
                        optimized_code
                    )
                    st.session_state["optimized_syntax"] = {
                        "ok": optimized_ok,
                        "message": optimized_message,
                    }

                    # Optional ML prediction.
                    st.session_state["ml_prediction"] = None
                    st.session_state["ml_prediction_error"] = None

                    if predict_optimization is not None:
                        try:
                            st.session_state["ml_prediction"] = (
                                predict_optimization(original_code)
                            )
                        except Exception as exc:
                            st.session_state["ml_prediction_error"] = (
                                f"{type(exc).__name__}: {exc}"
                            )
                    else:
                        st.session_state["ml_prediction_error"] = (
                            ML_IMPORT_ERROR
                            or "Prediction module is unavailable."
                        )

                    # Optional benchmark; unsupported code is handled safely.
                    st.session_state["benchmark_results"] = None
                    st.session_state["benchmark_error"] = None

                    if benchmark_actual_code is not None:
                        original_path = Path.cwd() / "generated_code.py"
                        optimized_path = Path.cwd() / "optimized_code.py"

                        try:
                            original_path.write_text(
                                original_code,
                                encoding="utf-8",
                            )
                            optimized_path.write_text(
                                optimized_code,
                                encoding="utf-8",
                            )

                            st.session_state["benchmark_results"] = (
                                benchmark_actual_code(
                                    str(original_path),
                                    str(optimized_path),
                                )
                            )

                        except Exception as exc:
                            st.session_state["benchmark_error"] = (
                                f"{type(exc).__name__}: {exc}"
                            )

                    else:
                        st.session_state["benchmark_error"] = (
                            BENCHMARK_IMPORT_ERROR
                            or "Benchmark module is unavailable."
                        )

                except Exception as exc:
                    st.error(
                        f"Code generation failed: {type(exc).__name__}: {exc}"
                    )
                    st.code(traceback.format_exc())

    original_code = st.session_state.get("generated_code", "")
    optimized_code = st.session_state.get("optimized_code", "")

    if original_code:
        st.divider()
        st.markdown("### Python syntax checks")

        col_original, col_optimized = st.columns(2)
        original_syntax = st.session_state.get("original_syntax", {})
        optimized_syntax = st.session_state.get("optimized_syntax", {})

        col_original.write(
            ("✅ " if original_syntax.get("ok") else "❌ ")
            + original_syntax.get("message", "Not checked")
        )

        col_optimized.write(
            ("✅ " if optimized_syntax.get("ok") else "❌ ")
            + optimized_syntax.get("message", "Not checked")
        )

        col_original, col_optimized = st.columns(2)

        with col_original:
            st.markdown("### Generated code")
            st.code(original_code, language="python")
            save_code_download(
                "Download generated_code.py",
                original_code,
                "generated_code.py",
            )

        with col_optimized:
            st.markdown("### Optimized code")
            st.code(optimized_code, language="python")
            save_code_download(
                "Download optimized_code.py",
                optimized_code,
                "optimized_code.py",
            )

        with st.expander("View code differences"):
            diff = difflib.unified_diff(
                original_code.splitlines(),
                optimized_code.splitlines(),
                fromfile="generated_code.py",
                tofile="optimized_code.py",
                lineterm="",
            )
            st.code(
                "\n".join(diff) or "No changes made by the optimizer.",
                language="diff",
            )

        st.markdown("### ML optimization prediction")

        prediction = st.session_state.get("ml_prediction")

        if prediction is not None:
            if isinstance(prediction, dict):
                st.json(prediction)
            else:
                st.write(prediction)

        elif st.session_state.get("ml_prediction_error"):
            st.info(
                "ML prediction is unavailable: "
                + st.session_state["ml_prediction_error"]
            )

        else:
            st.info("No ML prediction is available.")

        st.markdown("### Actual-code benchmark")

        benchmark = st.session_state.get("benchmark_results")

        if benchmark is not None:
            st.json(benchmark)

        elif st.session_state.get("benchmark_error"):
            st.info(
                "Benchmark could not run for this generated code: "
                + st.session_state["benchmark_error"]
            )

        else:
            st.info(
                "No benchmark results are available yet. The benchmark "
                "may only support certain arithmetic functions."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UML2Optimize — inspect the OCR output and correct the model before "
    "generating production code. OCR class-member assignment is heuristic "
    "and should be reviewed against the original UML diagram."
)
