
from benchmark_utils import benchmark_functions


def original_calculate(value):
    return ((value + 0) * 1) + ((10 * 20) + 5)


def optimized_calculate(value):
    return value + 205


def main():
    results = benchmark_functions(
        original_func=original_calculate,
        optimized_func=optimized_calculate,
        test_inputs=[0, 1, 5, 10, 50, 99],
        iterations=10_000,
        repeats=5,
    )

    print("=" * 55)
    print("UML2Optimize - Benchmark Utility Test")
    print("=" * 55)

    for result in results["correctness"]:
        status = "PASS" if result["passed"] else "FAIL"

        print(
            f"Input {result['input']:>2}: "
            f"original={result['original_result']}, "
            f"optimized={result['optimized_result']} "
            f"[{status}]"
        )

    if not results["all_correct"]:
        print("\nFAIL: Correctness checks failed.")
        raise SystemExit(1)

    print("\nPASS: All correctness checks passed.")
    print(f"Iterations per repeat: {results['iterations']:,}")
    print(f"Repeats: {results['repeats']}")

    print(
        f"\nOriginal median: "
        f"{results['original_median']:.6f} seconds"
    )
    print(
        f"Optimized median: "
        f"{results['optimized_median']:.6f} seconds"
    )
    print(
        f"Measured improvement: "
        f"{results['improvement_percent']:+.2f}%"
    )

    print(
        "\nNote: This is a small demonstration benchmark. "
        "Results may vary between runs and machines."
    )


if __name__ == "__main__":
    main()
