import numpy as np
from scipy import stats
from decimal import Decimal


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

    # Generate different types of floating point numbers
    float_sequential = [hash(float(i / 100)) for i in range(sample_size)]
    float_random = [hash(np.random.random()) for _ in range(sample_size)]
    float_scientific = [hash(float(f"{i}e-10")) for i in range(sample_size)]

    # Generate Decimal numbers
    decimal_sequential = [hash(Decimal(str(i / 100))) for i in range(sample_size)]
    decimal_random = [
        hash(Decimal(str(np.random.random()))) for _ in range(sample_size)
    ]
    decimal_scientific = [hash(Decimal(f"{i}e-10")) for i in range(sample_size)]

    # Test with different bin sizes
    bin_sizes = [2**8, 2**10, 2**12, 2**14, 2**16]

    for num_bins in bin_sizes:
        print(f"\n=== Testing with {num_bins} bins (2^{int(np.log2(num_bins))}) ===")

        print("\nSequential Float Test:")
        is_uniform, p_value, chi_squared = chi_squared_test(
            float_sequential, num_bins=num_bins
        )
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")

        print("\nRandom Float Test:")
        is_uniform, p_value, chi_squared = chi_squared_test(
            float_random, num_bins=num_bins
        )
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")

        print("\nScientific Notation Float Test:")
        is_uniform, p_value, chi_squared = chi_squared_test(
            float_scientific, num_bins=num_bins
        )
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")

        print("\nSequential Decimal Test:")
        is_uniform, p_value, chi_squared = chi_squared_test(
            decimal_sequential, num_bins=num_bins
        )
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")

        print("\nRandom Decimal Test:")
        is_uniform, p_value, chi_squared = chi_squared_test(
            decimal_random, num_bins=num_bins
        )
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")

        print("\nScientific Notation Decimal Test:")
        is_uniform, p_value, chi_squared = chi_squared_test(
            decimal_scientific, num_bins=num_bins
        )
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")


if __name__ == "__main__":
    main()
