
import ast
import inspect
import time
import statistics
from pathlib import Path


# Only allow simple arithmetic functions.
# Generated code containing imports, file access, arbitrary calls,
# loops, or other unsupported operations is rejected.
ALLOWED_NODES = (
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
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


def load_safe_functions(file_path):
    """Load supported top-level arithmetic functions without executing
    the original file or its classes, imports, and other statements."""
    source = Path(file_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    functions = {}

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue

        if node.decorator_list or node.returns is not None:
            continue

        args = node.args
        if (
            args.posonlyargs
            or args.vararg
            or args.kwonlyargs
            or args.kw_defaults
            or args.kwarg
            or args.defaults
            or len(args.args) > 1
        ):
            continue

        # Limit the accepted function body to simple statements.
        if not node.body or any(
            not isinstance(statement, (ast.Assign, ast.Return))
            for statement in node.body
        ):
            continue

        if any(
            isinstance(child, ast.Constant)
            and isinstance(child.value, (str, bytes))
            and len(child.value) > 1000
            for child in ast.walk(node)
        ):
            continue

        if not all(
            isinstance(child, ALLOWED_NODES)
            for child in ast.walk(node)
        ):
            continue

        # Reject assignments to anything other than simple variable names.
        if any(
            isinstance(target, (ast.Attribute, ast.Subscript))
            for child in ast.walk(node)
            if isinstance(child, ast.Assign)
            for target in child.targets
        ):
            continue

        # Compile only the validated function, never the complete file.
        isolated_module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(isolated_module)

        namespace = {}
        try:
            exec(compile(isolated_module, str(file_path), "exec"), namespace)
            functions[node.name] = namespace[node.name]
        except Exception:
            continue

    return functions


def benchmark_actual_code(
    original_path="generated_code.py",
    optimized_path="optimized_code.py",
    iterations=10_000,
    repeats=5,
):
    """Benchmark the same supported function from both generated files."""
    original_functions = load_safe_functions(original_path)
    optimized_functions = load_safe_functions(optimized_path)

    common_names = sorted(
        set(original_functions) & set(optimized_functions)
    )

    if not common_names:
        raise ValueError(
            "No matching safe arithmetic functions were found in both files."
        )

    results = {}

    for name in common_names:
        original = original_functions[name]
        optimized = optimized_functions[name]

        original_arg_count = len(inspect.signature(original).parameters)
        optimized_arg_count = len(inspect.signature(optimized).parameters)

        if original_arg_count != optimized_arg_count:
            continue

        if original_arg_count == 0:
            original_call = lambda value: original()
            optimized_call = lambda value: optimized()
            test_inputs = [0]
        else:
            original_call = original
            optimized_call = optimized
            test_inputs = [0, 1, 5, 10, 50, 99]

        # Correctness checks happen before timing.
        for value in test_inputs:
            try:
                old_result = original_call(value)
                new_result = optimized_call(value)
            except Exception as exc:
                raise ValueError(
                    f"Function '{name}' failed for input {value}: {exc}"
                ) from exc

            if old_result != new_result:
                raise ValueError(
                    f"Correctness check failed for '{name}' "
                    f"at input {value}: {old_result} != {new_result}"
                )

        original_times = []
        optimized_times = []

        # Warm up both implementations.
        for value in test_inputs:
            original_call(value)
            optimized_call(value)

        # Alternate measurement order to reduce order bias.
        for repeat in range(repeats):
            for func, destination in (
                ((original_call, original_times),
                 (optimized_call, optimized_times))
                if repeat % 2 == 0
                else
                ((optimized_call, optimized_times),
                 (original_call, original_times))
            ):
                start = time.perf_counter()
                for _ in range(iterations):
                    for value in test_inputs:
                        func(value)
                destination.append(time.perf_counter() - start)

        original_median = statistics.median(original_times)
        optimized_median = statistics.median(optimized_times)

        improvement = (
            (original_median - optimized_median)
            / original_median * 100
            if original_median > 0
            else 0.0
        )

        results[name] = {
            "test_inputs": test_inputs,
            "correctness": "PASS",
            "iterations": iterations,
            "repeats": repeats,
            "original_median": original_median,
            "optimized_median": optimized_median,
            "improvement_percent": improvement,
        }

    if not results:
        raise ValueError(
            "Matching functions were found, but their signatures "
            "were not supported."
        )

    return results
