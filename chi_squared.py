import numpy as np
from scipy import stats
from decimal import Decimal
from datetime import datetime, date, time, timedelta
from uuid import uuid4
from fractions import Fraction
from collections import namedtuple, defaultdict
from enum import Enum


def generate_seq_strings():
    return [hash(f"{i}") for i in range(10000)]


def generate_int():
    return [hash(i) for i in range(10000)]


def generate_random_strings():
    sample_size = 100000

    return [
        hash(
            "".join(
                np.random.choice(
                    list(
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    ),
                    np.random.randint(5, 20),
                )
            )
        )
        for _ in range(sample_size)
    ]


def chi_squared_test(hash_values, num_bins=256, confidence_level=0.95):
    """
    Performs a chi-squared test on hash values to check for uniform distribution.

    Args:
        hash_values: List of hash values to test
        num_bins: Number of bins to divide the hash values into
        confidence_level: Confidence level for the test (default 0.95)

    Returns:
        tuple: (is_uniform, p_value, chi_squared_stat)
    """
    # Count occurrences in each bin
    observed = np.zeros(num_bins)
    for value in hash_values:
        bin_index = value % num_bins
        observed[bin_index] += 1

    # Expected count for uniform distribution
    expected = len(hash_values) / num_bins

    # Calculate chi-squared statistic
    chi_squared_stat = np.sum((observed - expected) ** 2 / expected)

    # Calculate p-value (degrees of freedom = num_bins - 1)
    p_value = 1 - stats.chi2.cdf(chi_squared_stat, num_bins - 1)

    # Test if distribution is uniform at given confidence level
    is_uniform = p_value > (1 - confidence_level)

    return is_uniform, p_value, chi_squared_stat


def main():
    sample_size = 1_000_000

    # 1. Special Numbers
    special_numbers = [
        hash(x)
        for x in [
            float("inf"),
            float("-inf"),
            float("nan"),
            complex(1, 2),
            Fraction(22, 7),  # rational numbers
            0,
            -0,
            +0,
        ]
        * (sample_size // 8)
    ]

    # 2. Date/Time Objects
    datetime_hashes = [
        hash(datetime.now() + timedelta(seconds=i)) for i in range(sample_size)
    ]
    date_hashes = [
        hash(date(2024, 1, 1) + timedelta(days=i)) for i in range(sample_size)
    ]

    # 3. Collections and Custom Types
    Point = namedtuple("Point", ["x", "y"])
    collection_hashes = [
        hash(
            (
                tuple([i, i + 1, i + 2]),  # tuples
                frozenset([i, i + 1, i + 2]),  # frozensets
                Point(x=i, y=i + 1),  # named tuples
            )
        )
        for i in range(sample_size)
    ]

    # 4. UUIDs and Unique Identifiers
    uuid_hashes = [hash(uuid4()) for _ in range(sample_size)]

    # 5. Mixed-type Nested Structures
    nested_hashes = [
        hash((f"str{i}", i, float(i) / 100, (i, i + 1), frozenset([i, i + 1])))
        for i in range(sample_size)
    ]

    # 6. Unicode and Special Characters
    unicode_hashes = [
        hash(
            "".join(
                [
                    chr(i % 0x10FFFF)  # All Unicode characters
                    for _ in range(5)
                ]
            )
        )
        for i in range(sample_size)
    ]

    # 7. Empty/None Values
    empty_hashes = [
        hash(x) for x in [None, "", tuple(), frozenset()] * (sample_size // 4)
    ]

    # Test all types with different bin sizes
    test_cases = {
        "Special Numbers": special_numbers,
        "DateTime": datetime_hashes,
        "Date": date_hashes,
        "Collections": collection_hashes,
        "UUIDs": uuid_hashes,
        "Nested Structures": nested_hashes,
        "Unicode": unicode_hashes,
        "Empty/None": empty_hashes,
    }

    bin_sizes = [2**8, 2**10, 2**12, 2**14, 2**16]

    for num_bins in bin_sizes:
        print(f"\n=== Testing with {num_bins} bins (2^{int(np.log2(num_bins))}) ===")

        for test_name, test_data in test_cases.items():
            print(f"\n{test_name} Test:")
            is_uniform, p_value, chi_squared = chi_squared_test(
                test_data, num_bins=num_bins
            )
            print(f"Is uniform: {is_uniform}")
            print(f"P-value: {p_value:.6f}")
            print(f"Chi-squared statistic: {chi_squared:.2f}")


if __name__ == "__main__":
    main()
