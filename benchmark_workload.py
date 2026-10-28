
import statistics
import time
from pathlib import Path

from optimizer.optimizer import optimize_code


REPEATS = 9
WARMUP_ITERATIONS = 2_000
ITERATIONS = 20_000

PROJECT_DIR = Path(__file__).resolve().parent
ORIGINAL_OUTPUT = PROJECT_DIR / "benchmark_original.py"
OPTIMIZED_OUTPUT = PROJECT_DIR / "benchmark_optimized.py"


# This is the ONE source program used for both versions.
SOURCE_CODE = '''
def calculate(value):
    total = 0

    for i in range(1, 11):
        total += (value * 2 + 10 - 4) // 2
        total += (i * 3 + 12 - 6) // 3
        total += (value + 20) * 2 - 10
        total += (i + 8) * 4 - 16

    return total
'''


def load_function(source_code, filename):
    """Compile source code and return its calculate() function."""
    namespace = {}

    compiled = compile(source_code, filename, "exec")
    exec(compiled, namespace)

    function = namespace.get("calculate")

    if not callable(function):
        raise ValueError(f"{filename} does not define calculate().")

    return function


def measure(function):
    """Measure a batch of calls with different input values."""
    start = time.perf_counter()

    for i in range(ITERATIONS):
        function(i % 100)

    return time.perf_counter() - start


def main():
    print("=" * 68)
    print("UML2Optimize - Automatic Optimizer Benchmark")
    print("=" * 68)

    # Stage 1: Run the actual project optimizer.
    print("\n1. Running UML2Optimize constant folding...")

    optimized_source = optimize_code(SOURCE_CODE)

    ORIGINAL_OUTPUT.write_text(SOURCE_CODE, encoding="utf-8")
    OPTIMIZED_OUTPUT.write_text(optimized_source, encoding="utf-8")

    print(f"Original source saved:  {ORIGINAL_OUTPUT.name}")
    print(f"Optimized source saved: {OPTIMIZED_OUTPUT.name}")

    # Stage 2: Load both versions.
    original = load_function(SOURCE_CODE, "<original-workload>")
    optimized = load_function(optimized_source, "<optimized-workload>")

    print("\n2. Correctness checks")
    print("-" * 68)

    for value in (0, 1, 10, 50, 99):
        original_result = original(value)
        optimized_result = optimized(value)

        print(
            f"Input {value:>2}: original={original_result}, "
            f"optimized={optimized_result}"
        )

        if original_result != optimized_result:
            print("FAIL: Original and optimized results differ.")
            raise SystemExit(1)

    print("PASS: All correctness checks passed.")

    # Stage 3: Show the code transformation.
    print("\n3. Actual optimized source")
    print("-" * 68)
    print(optimized_source)

    # Stage 4: Warm up both implementations.
    for i in range(WARMUP_ITERATIONS):
        original(i % 100)
        optimized(i % 100)

    original_times = []
    optimized_times = []

    print("\n4. Repeated performance measurements")
    print("-" * 68)
    print(f"Calls per measurement: {ITERATIONS:,}")
    print(f"Measurement pairs:     {REPEATS}")
    print()

    # Alternate the measurement order.
    for repeat in range(REPEATS):
        if repeat % 2 == 0:
            original_time = measure(original)
            optimized_time = measure(optimized)
        else:
            optimized_time = measure(optimized)
            original_time = measure(original)

        original_times.append(original_time)
        optimized_times.append(optimized_time)

        change = (
            (original_time - optimized_time) / original_time * 100
            if original_time > 0
            else 0.0
        )

        print(
            f"Pair {repeat + 1}: "
            f"original={original_time:.4f}s, "
            f"optimized={optimized_time:.4f}s, "
            f"change={change:+.2f}%"
        )

    # Stage 5: Summarize results.
    original_median = statistics.median(original_times)
    optimized_median = statistics.median(optimized_times)

    improvement = (
        (original_median - optimized_median)
        / original_median
        * 100
        if original_median > 0
        else 0.0
    )

    print("\n5. Benchmark summary")
    print("-" * 68)
    print(f"Original median:   {original_median:.6f} seconds")
    print(f"Optimized median:  {optimized_median:.6f} seconds")
    print(f"Median change:     {improvement:+.2f}%")

    print("\nInterpretation")
    print("-" * 68)

    if improvement > 0:
        print(
            f"The automatically optimized version was {improvement:.2f}% "
            "faster by median time in this experiment."
        )
    elif improvement < 0:
        print(
            f"The automatically optimized version was {abs(improvement):.2f}% "
            "slower by median time in this experiment."
        )
    else:
        print("Both versions had the same median execution time.")

    print(
        "These measurements apply to this workload and machine. "
        "They do not establish that all programs will become faster."
    )


if __name__ == "__main__":
    main()
