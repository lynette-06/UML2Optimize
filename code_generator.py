
import ast
import keyword
import re
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

# This function makes the optimization/benchmark pipeline testable
# even when the UML diagram contains only class skeletons.
DEFAULT_DEMO_NAME = "calculate_demo"
DEFAULT_DEMO_EXPRESSION = "x + y + 0"


# ============================================================
# PYTHON IDENTIFIER AND TYPE HELPERS
# ============================================================

TYPE_MAP = {
    "string": "str",
    "str": "str",
    "integer": "int",
    "int": "int",
    "long": "int",
    "float": "float",
    "double": "float",
    "decimal": "float",
    "boolean": "bool",
    "bool": "bool",
    "char": "str",
    "character": "str",
    "date": "str",
    "datetime": "str",
    "void": "None",
    "none": "None",
    "object": "Any",
    "any": "Any",
}


def safe_identifier(value: Any, fallback: str = "item") -> str:
    """Convert a name into a valid Python identifier."""
    text = str(value or "").strip()
    text = re.sub(r"^[+\-#~\s]+", "", text)
    text = re.sub(r"\W", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    if not text:
        text = fallback

    if text[0].isdigit():
        text = "_" + text

    if keyword.iskeyword(text):
        text += "_"

    return text


def safe_class_name(value: Any) -> str:
    """Convert a UML class name to a Python class name."""
    name = safe_identifier(value, "GeneratedClass")
    return name[0].upper() + name[1:]


def python_type(value: Any) -> str:
    """Convert common UML types to Python type annotations."""
    if value is None:
        return "Any"

    text = str(value).strip()

    if not text:
        return "Any"

    normalized = text.lower()

    if normalized in TYPE_MAP:
        return TYPE_MAP[normalized]

    collection = re.fullmatch(
        r"(?:List|Array|Collection)\s*[<\[]\s*(.+?)\s*[>\]]",
        text,
        flags=re.IGNORECASE,
    )

    if collection:
        return f"list[{python_type(collection.group(1))}]"

    if re.fullmatch(r"[A-Za-z_]\w*", text):
        return safe_identifier(text, "Any")

    return "Any"


# ============================================================
# ATTRIBUTE NORMALIZATION
# ============================================================

def parse_attribute(attribute: Any) -> dict:
    """
    Supports dictionaries:
        {"name": "balance", "type": "float"}

    And strings:
        "balance: float"
        "- balance: float"
        "balance"
    """
    if isinstance(attribute, dict):
        name = (
            attribute.get("name")
            or attribute.get("attribute_name")
            or attribute.get("id")
            or "attribute"
        )

        data_type = (
            attribute.get("type")
            or attribute.get("data_type")
            or attribute.get("datatype")
            or "Any"
        )

    elif isinstance(attribute, str):
        text = attribute.strip()
        text = re.sub(r"^[+\-#~]\s*", "", text)
        text = text.rstrip(";").strip()

        if ":" in text:
            name, data_type = text.split(":", 1)
            name = name.strip()
            data_type = data_type.strip() or "Any"
        else:
            name = text.split("=", 1)[0].strip()
            data_type = "Any"

    else:
        name = "attribute"
        data_type = "Any"

    name = str(name).split("=", 1)[0].strip()

    return {
        "name": safe_identifier(name, "attribute"),
        "type": python_type(data_type),
    }


# ============================================================
# METHOD NORMALIZATION
# ============================================================

def parse_parameter(parameter: Any, index: int) -> dict:
    """Normalize a method parameter represented by text or a dictionary."""
    if isinstance(parameter, dict):
        name = (
            parameter.get("name")
            or parameter.get("parameter_name")
            or f"arg{index}"
        )
        data_type = (
            parameter.get("type")
            or parameter.get("data_type")
            or "Any"
        )

    elif isinstance(parameter, str):
        text = parameter.strip()

        if ":" in text:
            name, data_type = text.split(":", 1)
        else:
            name, data_type = text, "Any"

    else:
        name, data_type = f"arg{index}", "Any"

    return {
        "name": safe_identifier(name, f"arg{index}"),
        "type": python_type(data_type),
    }


def parse_method(method: Any) -> dict:
    """
    Supports dictionary methods and UML strings such as:
        "+ login(username: String): Boolean"
        "login()"
        "update_profile"
    """
    if isinstance(method, dict):
        name = (
            method.get("name")
            or method.get("method_name")
            or method.get("operation")
            or "generated_method"
        )

        parameters = (
            method.get("parameters")
            or method.get("params")
            or method.get("arguments")
            or []
        )

        return_type = (
            method.get("return_type")
            or method.get("returnType")
            or method.get("returns")
            or "Any"
        )

    elif isinstance(method, str):
        text = method.strip()
        text = re.sub(r"^[+\-#~]\s*", "", text)
        text = text.rstrip(";").strip()

        match = re.match(
            r"^\s*([^(]+?)\s*\((.*?)\)\s*(?::\s*(.+))?$",
            text,
        )

        if match:
            name = match.group(1).strip()
            parameter_text = match.group(2).strip()
            return_type = (match.group(3) or "Any").strip()

            parameters = [
                item.strip()
                for item in parameter_text.split(",")
                if item.strip()
            ]
        else:
            name = text.split(":", 1)[0].strip()
            parameters = []
            return_type = "Any"

    else:
        name = "generated_method"
        parameters = []
        return_type = "Any"

    name = str(name).strip()

    if "(" in name:
        name = name.split("(", 1)[0].strip()

    if isinstance(parameters, str):
        parameters = [
            item.strip()
            for item in parameters.split(",")
            if item.strip()
        ]

    parsed_parameters = []

    if isinstance(parameters, (list, tuple)):
        for index, parameter in enumerate(parameters, start=1):
            parsed = parse_parameter(parameter, index)

            if parsed["name"] != "self":
                parsed_parameters.append(parsed)

    return {
        "name": safe_identifier(name, "generated_method"),
        "parameters": parsed_parameters,
        "return_type": python_type(return_type),
    }


# ============================================================
# SAFE ARITHMETIC DEMO VALIDATION
# ============================================================

ALLOWED_EXPRESSION_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.UAdd,
    ast.USub,
)


def validate_arithmetic_expression(expression: str) -> str:
    """
    Permit only basic arithmetic using the demo inputs x and y.

    Function calls, attribute access, imports, and other executable
    expressions are intentionally rejected.
    """
    tree = ast.parse(expression, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_EXPRESSION_NODES):
            raise ValueError(
                "The demo expression may contain only basic arithmetic."
            )

        if isinstance(node, ast.Name) and node.id not in {"x", "y"}:
            raise ValueError(
                "The demo expression may use only variables x and y."
            )

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError(
                    "The demo expression may contain only numeric constants."
                )

    return ast.unparse(tree)


# ============================================================
# OPTIONAL OPTIMIZATION DEMO
# ============================================================

def build_demo_function(model: dict) -> list[str]:
    """
    Always generate a simple arithmetic demo by default.

    The function provides a consistent target for the project's optimizer
    and safe arithmetic benchmark. It does not represent UML business logic.
    """
    config = model.get("optimization_demo", {})

    if not isinstance(config, dict):
        config = {}

    # If a caller explicitly sets enabled=False, suppress the demo.
    # The actual-code benchmark will then have no demo to compare.
    if config.get("enabled") is False:
        return []

    function_name = safe_identifier(
        config.get("function_name", DEFAULT_DEMO_NAME),
        DEFAULT_DEMO_NAME,
    )

    expression = str(
        config.get("expression", DEFAULT_DEMO_EXPRESSION)
    )

    safe_expression = validate_arithmetic_expression(expression)

    return [
        "",
        "",
        f"def {function_name}(x: float, y: float) -> float:",
        '    """Arithmetic demo used to test code optimization."""',
        f"    return {safe_expression}",
        "",
    ]


# ============================================================
# MAIN CODE GENERATOR
# ============================================================

def generate_code(model: dict) -> str:
    """
    Generate Python class skeletons from the UML model.

    Attributes and methods may be strings or dictionaries.

    Also generates an arithmetic demonstration function by default so
    the optimizer and benchmark have a common function to compare.
    """
    if not isinstance(model, dict):
        raise TypeError("The UML model must be a dictionary.")

    components = model.get("components", [])

    if not isinstance(components, list):
        raise TypeError('model["components"] must be a list.')

    output = [
        '"""Python skeletons generated by UML2Optimize."""',
        "",
        "from typing import Any",
        "",
        "",
    ]

    used_class_names = set()

    for component in components:
        if not isinstance(component, dict):
            continue

        raw_name = (
            component.get("name")
            or component.get("class_name")
            or "GeneratedClass"
        )

        class_name = safe_class_name(raw_name)

        if class_name in used_class_names:
            continue

        used_class_names.add(class_name)

        raw_attributes = component.get("attributes", [])
        raw_methods = component.get("methods", [])
        relationships = component.get("relationships", [])

        if not isinstance(raw_attributes, list):
            raw_attributes = []

        if not isinstance(raw_methods, list):
            raw_methods = []

        if not isinstance(relationships, list):
            relationships = []

        attributes = []
        seen_attributes = set()

        for raw_attribute in raw_attributes:
            attribute = parse_attribute(raw_attribute)

            if attribute["name"] in seen_attributes:
                continue

            seen_attributes.add(attribute["name"])
            attributes.append(attribute)

        methods = []
        seen_methods = set()

        for raw_method in raw_methods:
            method = parse_method(raw_method)

            if method["name"] in seen_methods:
                continue

            seen_methods.add(method["name"])
            methods.append(method)

        output.append(f"class {class_name}:")
        output.append(f'    """Generated skeleton for {class_name}."""')
        output.append("")

        has_body = False

        for attribute in attributes:
            output.append(
                f"    {attribute['name']}: {attribute['type']}"
            )
            has_body = True

        if attributes and methods:
            output.append("")

        for index, method in enumerate(methods):
            if index:
                output.append("")

            parameters = ["self"]

            for parameter in method["parameters"]:
                parameters.append(
                    f"{parameter['name']}: {parameter['type']}"
                )

            signature = ", ".join(parameters)

            output.append(
                f"    def {method['name']}({signature}) "
                f"-> {method['return_type']}:"
            )
            output.append(
                f'        """TODO: Implement {method["name"]}."""'
            )
            output.append("        raise NotImplementedError")

            has_body = True

        if relationships:
            if has_body:
                output.append("")

            for relationship in relationships:
                text = " ".join(str(relationship).split())
                output.append(f"    # Relationship: {text}")

            has_body = True

        if not has_body:
            output.append("    pass")

        output.extend(["", ""])

    # Add a demonstration function, even when the UML contains only classes.
    output.extend(build_demo_function(model))

    if not used_class_names and not model.get("optimization_demo"):
        output.insert(
            1,
            "# No UML class definitions were supplied.",
        )

    return "\n".join(output).rstrip() + "\n"
