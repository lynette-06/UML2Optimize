
import statistics
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ORIGINAL_FILE = PROJECT_DIR / "generated_code.py"
OPTIMIZED_FILE = PROJECT_DIR / "optimized_code.py"

REPEATS =  nine = 9
WARMUP_ITERATIONS = 1_000
BATCH_SIZE = 100_000


def load_function(file_path):
    """Load calculate_total() from a generated Python file."""
    if not file_path.is_file():
        raise FileNotFoundError(
            f"{file_path.name} not found. Run python app.py first."
        )

    source = file_path.read_text(encoding="utf-8")
    namespace = {}

    exec(compile(source, str(file_path), "exec"), namespace)

    function = namespace.get("calculate_total")
    if not callable(function):
        raise ValueError(
            f"{file_path.name} does not define calculate_total()."
        )

    return function


def measure_batch(function):
    """Measure the time required for a batch of function calls."""
    start = time.perf_counter()

    for _ in range(BATCH_SIZE):
        function()

    return time.perf_counter() - start


def main():
    print("=" * 64)
    print("UML2Optimize - Repeated Performance Benchmark")
    print("=" * 64)

    try:
        original = load_function(ORIGINAL_FILE)
        optimized = load_function(OPTIMIZED_FILE)
    except (OSError, ValueError, SyntaxError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)

    # Check that the two programs behave identically.
    original_result = original()
    optimized_result = optimized()

    print("\n1. Correctness")
    print("-" * 64)
    print(f"Original result:  {original_result}")
    print(f"Optimized result: {optimized_result}")

    if original_result != optimized_result:
        print("FAIL: Results differ.")
        raise SystemExit(1)

    print("PASS: Both functions return the same result.")

    # Warm up both functions.
    for _ in range(WARMUP_ITERATIONS):
        original()
        optimized()

    original_times = []
    optimized_times = []

    print("\n2. Repeated measurements")
    print("-" * 64)
    print(f"Calls per measurement: {BATCH_SIZE:,}")
    print(f"Measurement pairs:     {REPEATS}")
    print()

    # Alternate the order to reduce systematic ordering bias.
    for repeat in range(REPEATS):
        if repeat % 2 == 0:
            original_time = measure_batch(original)
            optimized_time = measure_batch(optimized)
        else:
            optimized_time = measure_batch(optimized)
            original_time = measure_batch(original)

        original_times.append(original_time)
        optimized_times.append(optimized_time)

        change = (
            (original_time - optimized_time) / original_time * 100
            if original_time > 0
            else 0.0
        )

        print(
            f"Pair {repeat + 1}: "
            f"original={original_time:.6f}s, "
            f"optimized={optimized_time:.6f}s, "
            f"change={change:+.2f}%"
        )

    original_median = statistics.median(original_times)
    optimized_median = statistics.median(optimized_times)

    # Positive means faster; negative means slower.
    improvement = (
        (original_median - optimized_median) / original_median * 100
        if original_median > 0
        else 0.0
    )

    print("\n3. Summary")
    print("-" * 64)
    print(f"Original median:       {original_median:.6f} seconds")
    print(f"Optimized median:      {optimized_median:.6f} seconds")
    print(f"Median time change:    {improvement:+.2f}%")
    print(
        f"Original timing range: "
        f"{min(original_times):.6f}–{max(original_times):.6f}s"
    )
    print(
        f"Optimized timing range: "
        f"{min(optimized_times):.6f}–{max(optimized_times):.6f}s"
    )

    print("\n4. Interpretation")
    print("-" * 64)

    if improvement > 0:
        print(
            f"The optimized version was {improvement:.2f}% faster "
            "by median time in this experiment."
        )
    elif improvement < 0:
        print(
            f"The optimized version was {abs(improvement):.2f}% slower "
            "by median time in this experiment."
        )
    else:
        print("Both versions had the same median time.")

    print(
        "These results apply only to this small function. "
        "Timing differences can reflect interpreter and system noise."
    )
    print(
        "The experiment measures runtime; it does not measure whether "
        "the ML model correctly predicts optimization benefits."
    )


if __name__ == "__main__":
    main()
