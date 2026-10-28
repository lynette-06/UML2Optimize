
import random
import sys
import timeit
import statistics
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from optimizer.optimizer import optimize_code
from ml_model.feature_extractor import extract_features

OUTPUT_FILE = Path(__file__).resolve().parent / "benchmark_data.csv"

FEATURE_NAMES = [
    "source_lines",
    "source_characters",
    "function_count",
    "class_count",
    "loop_count",
    "conditional_count",
    "arithmetic_operations",
    "constant_operations",
    "function_calls",
]


def make_program(seed):
    """Generate a reproducible arithmetic benchmark program."""
    rng = random.Random(seed)
    expressions = []

    for _ in range(rng.randint(3, 12)):
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        c = rng.randint(1, 20)
        expressions.append(f"        total += ({a} * {b}) + {c}")

    source = (
        "def calculate(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        + "\n".join(expressions)
        + "\n    return total\n"
    )

    compile(source, "<generated-program>", "exec")
    return source


def load_function(source):
    namespace = {}
    exec(compile(source, "<benchmark>", "exec"), namespace)
    return namespace["calculate"]


def measure(function, n):
    """Return median time per call across seven batches."""
    samples = timeit.repeat(
        lambda: function(n),
        number=5,
        repeat=7,
    )
    return statistics.median(samples) / 5


def main():
    records = []
    number_of_programs = 100
    loop_iterations = 5000
    inconclusive_threshold = 0.05

    print("=" * 60)
    print("UML2Optimize - Improved Benchmark Data Collection")
    print("=" * 60)

    for seed in range(number_of_programs):
        original_code = make_program(seed)
        optimized_code = optimize_code(original_code)

        original_function = load_function(original_code)
        optimized_function = load_function(optimized_code)

        original_result = original_function(loop_iterations)
        optimized_result = optimized_function(loop_iterations)

        if original_result != optimized_result:
            raise AssertionError(
                f"Incorrect optimization for program {seed}"
            )

        # Warm up both versions.
        original_function(loop_iterations)
        optimized_function(loop_iterations)

        # Alternate measurement order to reduce order-related bias.
        original_times = []
        optimized_times = []

        for round_number in range(7):
            if round_number % 2 == 0:
                original_times.append(
                    measure(original_function, loop_iterations)
                )
                optimized_times.append(
                    measure(optimized_function, loop_iterations)
                )
            else:
                optimized_times.append(
                    measure(optimized_function, loop_iterations)
                )
                original_times.append(
                    measure(original_function, loop_iterations)
                )

        original_time = statistics.median(original_times)
        optimized_time = statistics.median(optimized_times)

        # Positive means the optimized version was faster.
        speedup_fraction = (
            (original_time - optimized_time) / original_time
            if original_time > 0
            else 0.0
        )

        # Exclude results within +/- 5% to avoid forcing noisy labels.
        if speedup_fraction > inconclusive_threshold:
            beneficial = 1
        elif speedup_fraction < -inconclusive_threshold:
            beneficial = 0
        else:
            beneficial = -1  # Inconclusive; exclude from classification.

        features = extract_features(original_code)

        records.append({
            **features,
            "original_time_seconds": original_time,
            "optimized_time_seconds": optimized_time,
            "speedup_fraction": speedup_fraction,
            "beneficial": beneficial,
            "program_id": seed,
        })

        print(
            f"Program {seed + 1:03d}/{number_of_programs} | "
            f"Speedup: {speedup_fraction:+.1%} | "
            f"Label: {beneficial}"
        )

    df = pd.DataFrame(records)
    df = df[
        FEATURE_NAMES
        + [
            "original_time_seconds",
            "optimized_time_seconds",
            "speedup_fraction",
            "beneficial",
            "program_id",
        ]
    ]

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("Collection complete!")
    print("=" * 60)
    print(f"Programs measured: {len(df)}")
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print("\nLabel counts (-1 = inconclusive):")
    print(df["beneficial"].value_counts().sort_index())
    print(f"\nInconclusive programs: {(df['beneficial'] == -1).sum()}")
    print("\nMedian original time:")
    print(f"{df['original_time_seconds'].median():.8f} seconds")
    print("\nMedian optimized time:")
    print(f"{df['optimized_time_seconds'].median():.8f} seconds")
    print(
        "\nNote: The dataset still uses a limited family of generated "
        "programs. The results are experimental, not proof of general "
        "optimization performance."
    )


if __name__ == "__main__":
    main()
