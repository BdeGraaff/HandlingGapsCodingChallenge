from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def median_absolute_deviation(a: np.ndarray) -> float:
    """Compute the median absolute deviation (MAD)."""
    return np.median(np.abs(a - np.median(a)))


def estimated_standard_deviation(a: np.ndarray, scale_factor=1.4826) -> float:
    """Compute an estimated standard deviation using the median absolute deviation (MAD).
    Args:
        a: 1d array of values.
        scale_factor: Distribution dependend scale factor.
            Defaults to 1.4826 for normally distributed data which is calculated by the reciprocal
            of the quantile function '1.0/(sqrt(2)*erfinv((3/4)1))' enforcing the 3*sigma rule.
    Returns:
        The estimated standard deviation.
    """
    return scale_factor * median_absolute_deviation(a)


def hampel_filter(
    samples: np.ndarray, window_size: int = 11, nsigma: float = 3
) -> np.ndarray:
    """Apply hampel filter to replace outliers by the local median.
    For more details refer to:
    Hancong Liu, Sirish Shah, Wei Jiang,
    On-line outlier detection and data cleaning,
    Computers & Chemical Engineering,
    Volume 28, Issue 9,
    2004,
    Pages 1635-1647,
    ISSN 0098-1354,
    https://doi.org/10.1016/j.compchemeng.2004.01.009
    Args:
        samples: 1d array of sampels.
        window_size: Size of the moving window. Must be odd.
        nsigma: Number of standard deviations used to identify outliers.
    Raises:
        ValueError: If window size is even.
    Returns:
        The filtered samples.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd!")

    padded_samples = np.pad(
        samples, window_size // 2, "median", stat_length=window_size // 2
    )

    centered_windows = sliding_window_view(padded_samples, window_size)

    filtered_samples = np.copy(samples)
    for i, (sample, window) in enumerate(zip(samples, centered_windows)):

        local_median = np.median(window)
        local_standard_deviation = estimated_standard_deviation(window)
        threshold = nsigma * local_standard_deviation

        if np.abs(sample - local_median) > threshold:
            filtered_samples[i] = local_median

    return filtered_samples
