/*
 * ARGUS fast_math — C hot-path math for EMA, rolling z-score, weighted mid-price.
 *
 * Compile:
 *   Linux/Mac:  gcc -O3 -shared -fPIC -o fast_math.so fast_math.c -lm
 *   Windows:    gcc -O3 -shared -o fast_math.dll fast_math.c
 *
 * All functions operate on contiguous double arrays passed via ctypes.
 */

#include <math.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/*
 * Exponential Moving Average.
 *
 * prices:  input array of length `len`
 * len:     number of elements
 * period:  EMA period (smoothing = 2/(period+1))
 * output:  pre-allocated output array of length `len`
 */
EXPORT void exponential_moving_average(
    const double *prices,
    int len,
    int period,
    double *output
) {
    if (len <= 0 || period <= 0) return;

    double alpha = 2.0 / ((double)period + 1.0);

    /* Seed with first value */
    output[0] = prices[0];
    for (int i = 1; i < len; i++) {
        output[i] = alpha * prices[i] + (1.0 - alpha) * output[i - 1];
    }
}

/*
 * Rolling z-score.
 *
 * values:  input array of length `len`
 * len:     number of elements
 * window:  lookback window for mean/std computation
 * output:  pre-allocated output array of length `len`
 *          First (window-1) elements are set to 0.0
 */
EXPORT void rolling_zscore(
    const double *values,
    int len,
    int window,
    double *output
) {
    if (len <= 0 || window <= 1) {
        for (int i = 0; i < len; i++) output[i] = 0.0;
        return;
    }

    /* Set leading NaN-equivalent to 0 */
    for (int i = 0; i < window - 1 && i < len; i++) {
        output[i] = 0.0;
    }

    for (int i = window - 1; i < len; i++) {
        /* Compute mean over window */
        double sum = 0.0;
        for (int j = i - window + 1; j <= i; j++) {
            sum += values[j];
        }
        double mean = sum / (double)window;

        /* Compute sample std dev */
        double var_sum = 0.0;
        for (int j = i - window + 1; j <= i; j++) {
            double diff = values[j] - mean;
            var_sum += diff * diff;
        }
        double std_dev = sqrt(var_sum / (double)(window - 1));

        if (std_dev < 1e-15) {
            output[i] = 0.0;
        } else {
            output[i] = (values[i] - mean) / std_dev;
        }
    }
}

/*
 * Volume-Weighted Mid Price.
 *
 * bids, asks:       best bid/ask price arrays
 * bid_sizes, ask_sizes: corresponding size arrays
 * len:              number of elements
 * output:           pre-allocated output array
 *
 * Formula: wmid = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
 * This weights toward the side with more size (the price is "pulled" toward
 * the heavier book side).
 */
EXPORT void weighted_mid_price(
    const double *bids,
    const double *asks,
    const double *bid_sizes,
    const double *ask_sizes,
    int len,
    double *output
) {
    for (int i = 0; i < len; i++) {
        double total_size = bid_sizes[i] + ask_sizes[i];
        if (total_size < 1e-15) {
            output[i] = (bids[i] + asks[i]) / 2.0;
        } else {
            output[i] = (bids[i] * ask_sizes[i] + asks[i] * bid_sizes[i]) / total_size;
        }
    }
}
