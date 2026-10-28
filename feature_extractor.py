
import ast


def extract_features(source_code):
    """
    Extract numerical features from Python source code
    for machine-learning-based optimization.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as error:
        raise ValueError(f"Invalid Python code: {error}") from error

    features = {
        "source_lines": len(source_code.splitlines()),
        "source_characters": len(source_code),
        "function_count": 0,
        "class_count": 0,
        "loop_count": 0,
        "conditional_count": 0,
        "arithmetic_operations": 0,
        "constant_operations": 0,
        "function_calls": 0,
    }

    arithmetic_types = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            features["function_count"] += 1

        elif isinstance(node, ast.ClassDef):
            features["class_count"] += 1

        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            features["loop_count"] += 1

        elif isinstance(node, ast.If):
            features["conditional_count"] += 1

        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, arithmetic_types
        ):
            features["arithmetic_operations"] += 1

            if (
                isinstance(node.left, ast.Constant)
                and isinstance(node.right, ast.Constant)
                and type(node.left.value) in (int, float)
                and type(node.right.value) in (int, float)
            ):
                features["constant_operations"] += 1

        elif isinstance(node, ast.Call):
            features["function_calls"] += 1

    return features
