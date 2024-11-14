import numpy as np
from scipy import stats


def generate_seq_strings():
    return [hash(f"{i}") for i in range(10000)]


def generate_int():
    return [hash(i) for i in range(10000)]


def generate_random_strings():
    return [
        hash("".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 10)))
        for _ in range(10000)  # Random 10-letter strings
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

    # Generate some test hash values (replace with your actual hash function)
    for values in [generate_int(), generate_random_strings(), generate_seq_strings()]:
        is_uniform, p_value, chi_squared = chi_squared_test(values)
        print("Chi-squared test results:")
        print(f"Is uniform: {is_uniform}")
        print(f"P-value: {p_value:.6f}")
        print(f"Chi-squared statistic: {chi_squared:.2f}")


if __name__ == "__main__":
    main()
