import numpy as np
from scipy import stats


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
    # Generate more test data
    sample_size = 1_000_000  # Increase sample size

    int_values = [hash(i) for i in range(sample_size)]
    seq_strings = [hash(f"test{i}") for i in range(sample_size)]
    random_strings = [
        hash("".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 10)))
        for _ in range(sample_size)
    ]

    # Test with larger but still manageable bin sizes
    bin_sizes = [2**8, 2**10, 2**12, 2**14, 2**16]  # Up to 65,536 bins

    for num_bins in bin_sizes:
        print(f"\n=== Testing with {num_bins} bins (2^{int(np.log2(num_bins))}) ===")

        print("\nInteger Hash Test:")
        is_uniform, p_value, chi_squared = chi_squared_test(
            int_values, num_bins=num_bins
        )
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")

        print("\nSequential String Hash Test:")
        is_uniform, p_value, chi_squared = chi_squared_test(
            seq_strings, num_bins=num_bins
        )
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")

        print("\nRandom String Hash Test:")
        is_uniform, p_value, chi_squared = chi_squared_test(
            random_strings, num_bins=num_bins
        )
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")

    # Try different confidence levels
    confidence_levels = [0.90, 0.95, 0.99]
    for conf_level in confidence_levels:
        print(f"\n=== Testing with confidence level {conf_level} ===")
        is_uniform, p_value, chi_squared = chi_squared_test(
            random_strings, confidence_level=conf_level
        )
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")


if __name__ == "__main__":
    main()
