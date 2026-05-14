// ARGUS Zig WASM Module — browser-side dashboard computations.
//
// Compiles to WebAssembly for high-performance client-side calculations.
// Functions exported:
//   - compute_drawdown: maximum drawdown from equity curve
//   - compute_sharpe: Sharpe ratio from returns
//   - compute_sortino: Sortino ratio (downside deviation only)
//   - compute_calmar: Calmar ratio (return / max drawdown)
//
// Build:
//   zig build-lib -target wasm32-freestanding -O ReleaseFast compute.zig
//   # Or use build.zig: zig build

const std = @import("std");

/// Compute maximum drawdown from an equity curve.
/// Returns the max drawdown as a positive fraction (e.g. 0.15 = 15% drawdown).
export fn compute_drawdown(equity: [*]const f64, len: usize) f64 {
    if (len < 2) return 0.0;

    var peak: f64 = equity[0];
    var max_dd: f64 = 0.0;

    var i: usize = 1;
    while (i < len) : (i += 1) {
        if (equity[i] > peak) {
            peak = equity[i];
        }
        if (peak > 1e-15) {
            const dd = (peak - equity[i]) / peak;
            if (dd > max_dd) {
                max_dd = dd;
            }
        }
    }
    return max_dd;
}

/// Compute annualized Sharpe ratio.
/// returns: array of period returns
/// risk_free_rate: per-period risk-free rate
export fn compute_sharpe(returns: [*]const f64, len: usize, risk_free_rate: f64) f64 {
    if (len < 2) return 0.0;

    // Mean excess return
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        sum += returns[i] - risk_free_rate;
    }
    const mean = sum / @as(f64, @floatFromInt(len));

    // Standard deviation
    var sq_sum: f64 = 0.0;
    i = 0;
    while (i < len) : (i += 1) {
        const diff = (returns[i] - risk_free_rate) - mean;
        sq_sum += diff * diff;
    }
    const variance = sq_sum / @as(f64, @floatFromInt(len - 1));
    const std_dev = @sqrt(variance);

    if (std_dev < 1e-15) return 0.0;
    return mean / std_dev;
}

/// Compute Sortino ratio (uses downside deviation instead of full std dev).
export fn compute_sortino(returns: [*]const f64, len: usize, risk_free_rate: f64) f64 {
    if (len < 2) return 0.0;

    // Mean excess return
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        sum += returns[i] - risk_free_rate;
    }
    const mean = sum / @as(f64, @floatFromInt(len));

    // Downside deviation (only negative excess returns)
    var down_sq_sum: f64 = 0.0;
    var down_count: usize = 0;
    i = 0;
    while (i < len) : (i += 1) {
        const excess = returns[i] - risk_free_rate;
        if (excess < 0) {
            down_sq_sum += excess * excess;
            down_count += 1;
        }
    }

    if (down_count == 0) {
        // No downside — return large positive if mean > 0
        if (mean > 0) return 99.99;
        return 0.0;
    }

    const down_variance = down_sq_sum / @as(f64, @floatFromInt(down_count));
    const down_dev = @sqrt(down_variance);

    if (down_dev < 1e-15) return 0.0;
    return mean / down_dev;
}

/// Compute Calmar ratio: annualized return / max drawdown.
export fn compute_calmar(annualized_return: f64, max_drawdown: f64) f64 {
    if (max_drawdown < 1e-15) {
        if (annualized_return > 0) return 99.99;
        return 0.0;
    }
    return annualized_return / max_drawdown;
}
