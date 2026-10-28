
import statistics
import time

from optimizer.optimizer import optimize_code


REPEATS = 9
WARMUP_CALLS = 2_000
ITERATIONS = 20_000

SOURCE_CODE = """
def calculate(x):
    return ((x + 0) * 1) + ((10 * 20) + 5)
"""


def load_function(source):
    namespace = {}
    exec(compile(source, "<benchmark>", "exec"), namespace)
    return namespace["calculate"]


def measure(function):
    start = time.perf_counter()

    for i in range(ITERATIONS):
        function(i % 100)

    return time.perf_counter() - start


def main():
    print("=" * 65)
    print("UML2Optimize - Algebraic Optimization Benchmark")
    print("=" * 65)

    # 1. Run the actual project optimizer.
    optimized_source = optimize_code(SOURCE_CODE)

    print("\n1. Original source:")
    print(SOURCE_CODE.strip())

    print("\n2. Optimized source:")
    print(optimized_source)

    # 2. Confirm that the optimizer changed the source.
    if optimized_source.strip() == SOURCE_CODE.strip():
        print("\nWARNING: The optimizer did not change the source.")
    else:
        print("\nPASS: The optimizer transformed the source.")

    original = load_function(SOURCE_CODE)
    optimized = load_function(optimized_source)

    # 3. Check correctness.
    print("\n3. Correctness checks")
    print("-" * 65)

    for x in (0, 1, 5, 10, 50, 99):
        original_result = original(x)
        optimized_result = optimized(x)

        print(
            f"x={x:>2}: original={original_result}, "
            f"optimized={optimized_result}"
        )

        if original_result != optimized_result:
            raise AssertionError(
                f"Results differ for input {x}: "
                f"{original_result} != {optimized_result}"
            )

    print("PASS: All correctness checks passed.")

    # 4. Warm up both implementations.
    for i in range(WARMUP_CALLS):
        original(i % 100)
        optimized(i % 100)

    # 5. Measure both implementations, alternating order.
    original_times = []
    optimized_times = []

    print("\n4. Performance measurements")
    print("-" * 65)

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
            f"original={original_time:.6f}s, "
            f"optimized={optimized_time:.6f}s, "
            f"change={change:+.2f}%"
        )

    # 6. Summarize using medians to reduce sensitivity to outliers.
    original_median = statistics.median(original_times)
    optimized_median = statistics.median(optimized_times)

    improvement = (
        (original_median - optimized_median) / original_median * 100
        if original_median > 0
        else 0.0
    )

    print("\n5. Summary")
    print("-" * 65)
    print(f"Original median:  {original_median:.6f} seconds")
    print(f"Optimized median: {optimized_median:.6f} seconds")
    print(f"Median change:    {improvement:+.2f}%")

    print(
        "\nNote: Python may optimize constant expressions itself. "
        "This benchmark tests this specific function on this machine; "
        "it does not establish a general performance improvement."
    )


if __name__ == "__main__":
    main()
