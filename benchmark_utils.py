
import statistics
import time


def benchmark_functions(
    original_func,
    optimized_func,
    test_inputs=None,
    iterations=10_000,
    repeats=7,
):
    """Check correctness and compare execution times."""

    if test_inputs is None:
        test_inputs = [0, 1, 5, 10, 50, 99]
    else:
        test_inputs = list(test_inputs)

    if not test_inputs:
        raise ValueError("test_inputs must not be empty.")

    if iterations < 1 or repeats < 1:
        raise ValueError("Iterations and repeats must be positive.")

    # 1. Verify correctness.
    correctness = []

    for value in test_inputs:
        original_result = original_func(value)
        optimized_result = optimized_func(value)

        correctness.append({
            "input": value,
            "original_result": original_result,
            "optimized_result": optimized_result,
            "passed": original_result == optimized_result,
        })

    all_correct = all(item["passed"] for item in correctness)

    if not all_correct:
        return {
            "correctness": correctness,
            "all_correct": False,
            "original_median": None,
            "optimized_median": None,
            "improvement_percent": None,
            "original_samples": [],
            "optimized_samples": [],
            "iterations": iterations,
            "repeats": repeats,
        }

    # 2. Warm up both functions.
    for value in test_inputs:
        original_func(value)
        optimized_func(value)

    original_times = []
    optimized_times = []

    # 3. Measure execution time.
    for repeat in range(repeats):
        value = test_inputs[repeat % len(test_inputs)]

        functions = [
            ("original", original_func),
            ("optimized", optimized_func),
        ]

        # Alternate measurement order.
        if repeat % 2:
            functions.reverse()

        measured = {}

        for name, function in functions:
            start = time.perf_counter()

            for _ in range(iterations):
                function(value)

            measured[name] = time.perf_counter() - start

        original_times.append(measured["original"])
        optimized_times.append(measured["optimized"])

    original_median = statistics.median(original_times)
    optimized_median = statistics.median(optimized_times)

    improvement_percent = (
        (original_median - optimized_median)
        / original_median * 100
        if original_median > 0
        else 0.0
    )

    return {
        "correctness": correctness,
        "all_correct": True,
        "original_median": original_median,
        "optimized_median": optimized_median,
        "improvement_percent": improvement_percent,
        "original_samples": original_times,
        "optimized_samples": optimized_times,
        "iterations": iterations,
        "repeats": repeats,
    }
