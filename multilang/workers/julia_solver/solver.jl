#=
ARGUS Julia Optimization Solver — portfolio optimization.

Functions:
  1. Mean-variance portfolio optimization (gradient descent)
  2. Risk parity allocation
  3. Kelly-optimal allocation

Protocol: JSON on stdin/stdout.
  Input:  {"command": "<name>", "data": {...}}
  Output: {"ok": true, "result": {...}} | {"ok": false, "error": "..."}

Usage:
    julia solver.jl < input.json
=#

using LinearAlgebra

"""
    mean_variance_optimize(returns_matrix, risk_aversion)

Mean-variance optimization via gradient descent.
Returns optimal portfolio weights.
"""
function mean_variance_optimize(returns_matrix::Matrix{Float64}, risk_aversion::Float64)
    n_assets = size(returns_matrix, 2)
    mu = vec(mean(returns_matrix, dims=1))
    cov_matrix = cov(returns_matrix)

    # Gradient descent for: max(w'μ - λ/2 * w'Σw) s.t. sum(w)=1, w>=0
    w = fill(1.0 / n_assets, n_assets)
    lr = 0.01

    for iter in 1:1000
        grad = mu - risk_aversion * cov_matrix * w
        w .+= lr * grad
        # Project onto simplex: clamp negatives, renormalize
        w .= max.(w, 0.0)
        s = sum(w)
        if s > 1e-15
            w ./= s
        else
            w .= 1.0 / n_assets
        end
    end

    return w
end

"""
    risk_parity(cov_matrix)

Risk parity allocation: each asset contributes equally to portfolio risk.
"""
function risk_parity(cov_matrix::Matrix{Float64})
    n = size(cov_matrix, 1)
    w = fill(1.0 / n, n)

    for iter in 1:500
        sigma_w = cov_matrix * w
        port_vol = sqrt(dot(w, sigma_w))
        if port_vol < 1e-15
            break
        end
        # Marginal risk contribution
        mrc = sigma_w ./ port_vol
        # Target: equal risk contribution
        rc = w .* mrc
        target_rc = port_vol / n

        # Update: scale inversely to risk contribution
        for i in 1:n
            if mrc[i] > 1e-15
                w[i] = target_rc / mrc[i]
            end
        end
        s = sum(w)
        if s > 1e-15
            w ./= s
        end
    end

    return w
end

"""
    kelly_optimal(win_rates, payoff_ratios, correlation_matrix)

Multi-asset Kelly criterion with correlation adjustment.
"""
function kelly_optimal(win_rates::Vector{Float64}, payoff_ratios::Vector{Float64},
                       corr_matrix::Matrix{Float64})
    n = length(win_rates)
    fractions = zeros(n)

    for i in 1:n
        if payoff_ratios[i] > 1e-15
            # Basic Kelly: f = p - q/b
            kelly_raw = win_rates[i] - (1.0 - win_rates[i]) / payoff_ratios[i]
            fractions[i] = clamp(kelly_raw, 0.0, 1.0)
        end
    end

    # Correlation adjustment: reduce fractions for correlated assets
    for i in 1:n
        correlation_penalty = 0.0
        for j in 1:n
            if i != j
                correlation_penalty += abs(corr_matrix[i, j]) * fractions[j]
            end
        end
        if n > 1
            avg_penalty = correlation_penalty / (n - 1)
            fractions[i] *= max(0.0, 1.0 - avg_penalty * 0.5)
        end
    end

    # Normalize if total > 1
    total = sum(fractions)
    if total > 1.0
        fractions ./= total
    end

    return fractions
end

# Main: read JSON from stdin, dispatch, write result to stdout
# (Actual JSON parsing with JSON3 package omitted — see bridge.py)
println("""{"ok": true, "result": {"status": "julia_solver_ready"}}""")
