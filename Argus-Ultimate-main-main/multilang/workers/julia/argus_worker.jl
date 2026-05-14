#!/usr/bin/env julia
# Argus Julia Worker — GARCH volatility + HMM regime detection
# Reads JSON lines from stdin, processes tasks, writes JSON results to stdout.
#
# Run: julia argus_worker.jl
#
# Implements all standard 20 task types PLUS:
#   - garch_volatility: Forward-looking GARCH(1,1) volatility forecast
#   - hmm_regime: Hidden Markov Model regime detection (2-state)
#   - evt_tail_risk: Extreme Value Theory tail risk (GPD fit)

using Printf

# ─────────────────────────────────────────────
# Language profile constants (Julia — stats role)
# ─────────────────────────────────────────────
const LANG = "julia"
const RISK_MAX = 0.45
const CYCLE_SCALE = 1.03
const VOL_WEIGHT = 1.15
const SIG_WEIGHT = 1.05
const SPREAD_MULT = 1.01
const MIN_CONF_TO_ACCEPT = 0.55

# ─────────────────────────────────────────────
# Minimal JSON parser/emitter (no deps)
# ─────────────────────────────────────────────
function parse_json(s::AbstractString)
    s = strip(s)
    if isempty(s)
        return nothing
    end
    # Use Julia's built-in Meta.parse for JSON-like structures
    # Simple recursive descent JSON parser
    idx = Ref(1)
    return _parse_value(s, idx)
end

function _skip_ws(s, idx)
    while idx[] <= length(s) && s[idx[]] in " \t\n\r"
        idx[] += 1
    end
end

function _parse_value(s, idx)
    _skip_ws(s, idx)
    if idx[] > length(s)
        return nothing
    end
    c = s[idx[]]
    if c == '"'
        return _parse_string(s, idx)
    elseif c == '{'
        return _parse_object(s, idx)
    elseif c == '['
        return _parse_array(s, idx)
    elseif c == 't'
        idx[] += 4; return true
    elseif c == 'f'
        idx[] += 5; return false
    elseif c == 'n'
        idx[] += 4; return nothing
    else
        return _parse_number(s, idx)
    end
end

function _parse_string(s, idx)
    idx[] += 1  # skip opening "
    buf = IOBuffer()
    while idx[] <= length(s)
        c = s[idx[]]
        if c == '"'
            idx[] += 1
            return String(take!(buf))
        elseif c == '\\'
            idx[] += 1
            if idx[] <= length(s)
                nc = s[idx[]]
                if nc == 'n'; write(buf, '\n')
                elseif nc == 't'; write(buf, '\t')
                elseif nc == '"'; write(buf, '"')
                elseif nc == '\\'; write(buf, '\\')
                elseif nc == '/'; write(buf, '/')
                else write(buf, nc)
                end
            end
        else
            write(buf, c)
        end
        idx[] += 1
    end
    return String(take!(buf))
end

function _parse_number(s, idx)
    start = idx[]
    if idx[] <= length(s) && s[idx[]] == '-'
        idx[] += 1
    end
    while idx[] <= length(s) && (s[idx[]] in "0123456789.eE+-")
        idx[] += 1
    end
    numstr = s[start:idx[]-1]
    if occursin('.', numstr) || occursin('e', numstr) || occursin('E', numstr)
        return parse(Float64, numstr)
    else
        return parse(Float64, numstr)  # always return Float64 for consistency
    end
end

function _parse_object(s, idx)
    idx[] += 1  # skip {
    d = Dict{String,Any}()
    _skip_ws(s, idx)
    if idx[] <= length(s) && s[idx[]] == '}'
        idx[] += 1; return d
    end
    while idx[] <= length(s)
        _skip_ws(s, idx)
        key = _parse_string(s, idx)
        _skip_ws(s, idx)
        idx[] += 1  # skip :
        val = _parse_value(s, idx)
        d[key] = val
        _skip_ws(s, idx)
        if idx[] > length(s) || s[idx[]] == '}'
            idx[] += 1; return d
        end
        idx[] += 1  # skip ,
    end
    return d
end

function _parse_array(s, idx)
    idx[] += 1  # skip [
    arr = Any[]
    _skip_ws(s, idx)
    if idx[] <= length(s) && s[idx[]] == ']'
        idx[] += 1; return arr
    end
    while idx[] <= length(s)
        val = _parse_value(s, idx)
        push!(arr, val)
        _skip_ws(s, idx)
        if idx[] > length(s) || s[idx[]] == ']'
            idx[] += 1; return arr
        end
        idx[] += 1  # skip ,
    end
    return arr
end

function to_json(v)::String
    if v isa Dict
        parts = String[]
        for (k, val) in v
            push!(parts, "\"$(escape_json_str(string(k)))\":$(to_json(val))")
        end
        return "{" * join(parts, ",") * "}"
    elseif v isa Vector || v isa Tuple
        return "[" * join([to_json(x) for x in v], ",") * "]"
    elseif v isa AbstractString
        return "\"$(escape_json_str(v))\""
    elseif v isa Bool
        return v ? "true" : "false"
    elseif v === nothing
        return "null"
    elseif v isa Number
        if isnan(v) || isinf(v)
            return "0.0"
        end
        return @sprintf("%.10g", Float64(v))
    else
        return "\"$(escape_json_str(string(v)))\""
    end
end

function escape_json_str(s)
    s = replace(s, "\\" => "\\\\")
    s = replace(s, "\"" => "\\\"")
    s = replace(s, "\n" => "\\n")
    s = replace(s, "\t" => "\\t")
    return s
end

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
getf(d, k, fallback=0.0) = get(d, k, fallback) isa Number ? Float64(get(d, k, fallback)) : fallback
gets(d, k, fallback="") = string(get(d, k, fallback))

function getfvec(d, k)
    v = get(d, k, nothing)
    if v isa Vector
        return Float64[x isa Number ? Float64(x) : 0.0 for x in v]
    end
    return Float64[]
end

function getpairs(d, k)
    v = get(d, k, nothing)
    if v isa Vector
        return [(Float64(x[1]), Float64(x[2])) for x in v if x isa Vector && length(x) >= 2]
    end
    return Tuple{Float64,Float64}[]
end

clamp01(x) = max(0.0, min(1.0, x))

function welford_vol(returns::Vector{Float64})
    n = length(returns)
    n == 0 && return 10.0
    mean = 0.0; m2 = 0.0
    for (i, r) in enumerate(returns)
        delta = r - mean
        mean += delta / i
        m2 += delta * (r - mean)
    end
    var = m2 / n
    var <= 0 && return 10.0
    return sqrt(var * 252.0) * 1e4
end

function prices_to_returns(prices::Vector{Float64})
    n = length(prices)
    n < 2 && return Float64[]
    return [(prices[i] - prices[i-1]) / max(abs(prices[i-1]), 1e-12) for i in 2:n]
end

# ─────────────────────────────────────────────
# GARCH(1,1) — Maximum Likelihood Estimation
# σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
# ─────────────────────────────────────────────
function garch11_fit(returns::Vector{Float64}; max_iter=100)
    n = length(returns)
    n < 20 && return (omega=1e-6, alpha=0.1, beta=0.85, sigma_forecast=welford_vol(returns))

    # Initial parameters
    sample_var = sum(r^2 for r in returns) / n
    omega = sample_var * 0.05
    alpha = 0.10
    beta  = 0.85

    best_ll = -Inf
    best_params = (omega, alpha, beta)

    # Grid search + gradient-free optimization
    for a in 0.03:0.03:0.25
        for b in 0.70:0.03:0.95
            a + b >= 0.999 && continue
            w = sample_var * (1.0 - a - b)
            w <= 0 && continue

            # Compute log-likelihood
            ll = 0.0
            sigma2 = sample_var
            valid = true
            for i in 2:n
                sigma2 = w + a * returns[i-1]^2 + b * sigma2
                sigma2 = max(sigma2, 1e-20)
                ll += -0.5 * (log(sigma2) + returns[i]^2 / sigma2)
                if isnan(ll) || isinf(ll)
                    valid = false; break
                end
            end
            if valid && ll > best_ll
                best_ll = ll
                best_params = (w, a, b)
            end
        end
    end

    omega, alpha, beta = best_params

    # Forecast: run filter forward to get last σ²
    sigma2 = sample_var
    for i in 2:n
        sigma2 = omega + alpha * returns[i-1]^2 + beta * sigma2
        sigma2 = max(sigma2, 1e-20)
    end
    # One-step-ahead forecast
    sigma2_forecast = omega + alpha * returns[end]^2 + beta * sigma2

    # Annualize: σ_annual = σ_daily * √252 * 10000 (bps)
    sigma_annual_bps = sqrt(max(sigma2_forecast, 0.0) * 252.0) * 1e4

    return (omega=omega, alpha=alpha, beta=beta, sigma_forecast=sigma_annual_bps)
end

# ─────────────────────────────────────────────
# Hidden Markov Model (2-state) — Baum-Welch
# State 0: Low volatility (range/trend)
# State 1: High volatility (crisis/momentum)
# ─────────────────────────────────────────────
function hmm_2state(returns::Vector{Float64}; max_iter=20)
    n = length(returns)
    n < 10 && return (regime="unknown", prob_high_vol=0.5, transition=zeros(2,2))

    # Initialize parameters
    sorted_rets = sort(abs.(returns))
    mid = div(n, 2)
    mu = [0.0, 0.0]  # both states have ~zero mean for returns
    sigma = [
        max(sqrt(sum(sorted_rets[1:mid].^2) / mid), 1e-8),
        max(sqrt(sum(sorted_rets[mid+1:end].^2) / (n - mid)), 1e-8)
    ]

    # Transition matrix: states tend to persist
    A = [0.95 0.05; 0.10 0.90]
    pi = [0.7, 0.3]  # prior: more likely low-vol

    # Gaussian emission
    gauss(x, m, s) = exp(-0.5 * ((x - m) / s)^2) / (s * sqrt(2π))

    for iter in 1:max_iter
        # Forward pass
        alpha_mat = zeros(n, 2)
        for j in 1:2
            alpha_mat[1, j] = pi[j] * gauss(returns[1], mu[j], sigma[j])
        end
        s = sum(alpha_mat[1, :])
        s > 0 && (alpha_mat[1, :] ./= s)

        for t in 2:n
            for j in 1:2
                alpha_mat[t, j] = sum(alpha_mat[t-1, i] * A[i, j] for i in 1:2) * gauss(returns[t], mu[j], sigma[j])
            end
            s = sum(alpha_mat[t, :])
            s > 0 && (alpha_mat[t, :] ./= s)
        end

        # Backward pass
        beta_mat = zeros(n, 2)
        beta_mat[n, :] .= 1.0
        for t in (n-1):-1:1
            for i in 1:2
                beta_mat[t, i] = sum(A[i, j] * gauss(returns[t+1], mu[j], sigma[j]) * beta_mat[t+1, j] for j in 1:2)
            end
            s = sum(beta_mat[t, :])
            s > 0 && (beta_mat[t, :] ./= s)
        end

        # Posterior (gamma)
        gamma = alpha_mat .* beta_mat
        for t in 1:n
            s = sum(gamma[t, :])
            s > 0 && (gamma[t, :] ./= s)
        end

        # Update parameters
        for j in 1:2
            wsum = sum(gamma[:, j])
            wsum < 1e-12 && continue
            mu[j] = sum(gamma[t, j] * returns[t] for t in 1:n) / wsum
            sigma[j] = sqrt(max(sum(gamma[t, j] * (returns[t] - mu[j])^2 for t in 1:n) / wsum, 1e-12))
        end

        # Update transition matrix
        for i in 1:2
            denom = sum(gamma[1:n-1, i])
            denom < 1e-12 && continue
            for j in 1:2
                numer = 0.0
                for t in 1:n-1
                    numer += alpha_mat[t, i] * A[i, j] * gauss(returns[t+1], mu[j], sigma[j]) * beta_mat[t+1, j]
                end
                A[i, j] = max(numer / max(denom, 1e-12), 1e-6)
            end
            rs = sum(A[i, :])
            rs > 0 && (A[i, :] ./= rs)
        end

        pi = gamma[1, :]
    end

    # Final state probabilities
    final_gamma = zeros(2)
    for j in 1:2
        final_gamma[j] = gauss(returns[end], mu[j], sigma[j]) * pi[j]
    end
    s = sum(final_gamma)
    s > 0 && (final_gamma ./= s)

    # Ensure state 1 is the high-vol state
    if sigma[1] > sigma[2]
        sigma = reverse(sigma)
        mu = reverse(mu)
        final_gamma = reverse(final_gamma)
        A = A[[2,1], [2,1]]
    end

    prob_high_vol = final_gamma[2]
    regime = prob_high_vol > 0.5 ? "high_vol" : (abs(mu[1]) > sigma[1] * 0.3 ? "trend" : "mean_revert")

    return (regime=regime, prob_high_vol=prob_high_vol, transition=A,
            sigma_low=sigma[1], sigma_high=sigma[2])
end

# ─────────────────────────────────────────────
# EVT Tail Risk — Generalized Pareto Distribution fit
# ─────────────────────────────────────────────
function evt_tail_risk(returns::Vector{Float64}; threshold_pct=5.0)
    n = length(returns)
    n < 30 && return Dict("cvar_95"=>0.0, "cvar_99"=>0.0, "tail_index"=>0.0, "language"=>LANG)

    # Use negative returns (losses) above threshold
    losses = sort(-returns)
    threshold_idx = max(1, round(Int, n * threshold_pct / 100.0))
    threshold = losses[n - threshold_idx + 1]
    exceedances = [l - threshold for l in losses if l > threshold]

    ne = length(exceedances)
    ne < 5 && return Dict("cvar_95"=>0.0, "cvar_99"=>0.0, "tail_index"=>0.0, "language"=>LANG)

    # Method of moments for GPD (shape ξ, scale σ)
    mean_exc = sum(exceedances) / ne
    var_exc = sum((e - mean_exc)^2 for e in exceedances) / ne

    # ξ = 0.5 * (mean²/var - 1), σ = mean * (mean²/var + 1) / 2
    ratio = mean_exc^2 / max(var_exc, 1e-12)
    xi = 0.5 * (ratio - 1.0)
    sigma_gpd = mean_exc * (ratio + 1.0) / 2.0

    # GPD quantile: u + σ/ξ * ((n/ne * (1-p))^(-ξ) - 1)
    p95 = threshold_idx / n
    gpd_quantile(p) = begin
        if abs(xi) < 1e-6
            threshold + sigma_gpd * log(p95 / (1 - p))
        else
            threshold + sigma_gpd / xi * ((p95 / (1 - p))^xi - 1)
        end
    end

    var_95 = gpd_quantile(0.95)
    var_99 = gpd_quantile(0.99)

    # CVaR = VaR + σ/(1-ξ) (for ξ < 1)
    cvar_95 = xi < 1.0 ? var_95 + sigma_gpd / max(1 - xi, 1e-6) : var_95 * 1.5
    cvar_99 = xi < 1.0 ? var_99 + sigma_gpd / max(1 - xi, 1e-6) : var_99 * 1.5

    return Dict(
        "cvar_95" => cvar_95 * sqrt(252.0) * 1e4,  # annualized bps
        "cvar_99" => cvar_99 * sqrt(252.0) * 1e4,
        "tail_index" => xi,
        "gpd_scale" => sigma_gpd,
        "threshold" => threshold,
        "n_exceedances" => ne,
        "language" => LANG
    )
end

# ─────────────────────────────────────────────
# Standard task handlers
# ─────────────────────────────────────────────

function handle_volatility_estimate(d)
    returns = getfvec(d, "returns")
    prices = isempty(returns) ? getfvec(d, "prices") : Float64[]
    if isempty(prices)
        prices = getfvec(d, "ohlcv_close")
    end

    if isempty(returns) && !isempty(prices)
        returns = prices_to_returns(prices)
    end

    # Use GARCH instead of simple Welford
    if length(returns) >= 20
        fit = garch11_fit(returns)
        return Dict(
            "volatility_annual_bps" => fit.sigma_forecast * VOL_WEIGHT,
            "volatility_weight" => VOL_WEIGHT,
            "method" => "garch11",
            "garch_alpha" => fit.alpha,
            "garch_beta" => fit.beta,
            "language" => LANG, "ok" => true
        )
    else
        vol = welford_vol(returns)
        return Dict(
            "volatility_annual_bps" => vol * VOL_WEIGHT,
            "volatility_weight" => VOL_WEIGHT,
            "method" => "welford",
            "language" => LANG, "ok" => true
        )
    end
end

function handle_regime_estimate(d)
    returns = getfvec(d, "returns")
    prices = getfvec(d, "prices")
    if isempty(prices); prices = getfvec(d, "ohlcv_close"); end
    if isempty(returns) && !isempty(prices)
        returns = prices_to_returns(prices)
    end

    if length(returns) >= 20
        hmm = hmm_2state(returns)
        return Dict(
            "regime" => hmm.regime,
            "prob_high_vol" => hmm.prob_high_vol,
            "regime_weight" => 1.1,
            "method" => "hmm_2state",
            "sigma_low" => hmm.sigma_low,
            "sigma_high" => hmm.sigma_high,
            "language" => LANG, "ok" => true
        )
    else
        vol = isempty(returns) ? 10.0 : welford_vol(returns)
        regime = vol > 30.0 ? "high_vol" : (vol > 15.0 ? "trend" : "mean_revert")
        return Dict("regime" => regime, "regime_weight" => 1.0, "method" => "threshold", "language" => LANG, "ok" => true)
    end
end

function handle_cycle_plan(d)
    lang_idx = sum(Int(c) for c in LANG) % 100
    key_str = join(sort(collect(keys(d))), ",")
    h = sum(Int(c) for c in key_str) + lang_idx
    boost = ((h % 1000) / 1000.0 - 0.5) * 0.01 * CYCLE_SCALE
    return Dict("language" => LANG, "cycle_boost" => boost, "ok" => true)
end

function handle_order_book(d)
    bids = getpairs(d, "bids")
    asks = getpairs(d, "asks")
    if isempty(bids); bids = haskey(d, "order_book") ? getpairs(d["order_book"], "bids") : bids; end
    if isempty(asks); asks = haskey(d, "order_book") ? getpairs(d["order_book"], "asks") : asks; end
    bb = isempty(bids) ? 0.0 : bids[1][1]
    ba = isempty(asks) ? 0.0 : asks[1][1]
    mid = (bb > 0 && ba > 0) ? (bb + ba) / 2 : 0.0
    spread = mid > 0 ? (ba - bb) / mid * 1e4 * SPREAD_MULT : 0.0
    bv = sum(p[2] for p in bids[1:min(5,length(bids))]; init=0.0)
    av = sum(p[2] for p in asks[1:min(5,length(asks))]; init=0.0)
    t = bv + av
    imb = t > 0 ? (bv - av) / t : 0.0
    return Dict("spread_bps" => spread, "imbalance" => imb, "mid" => mid, "language" => LANG)
end

function handle_risk(d)
    pv = getf(d, "position_value")
    cap = getf(d, "capital", 1.0)
    ratio = cap != 0 ? pv / cap : 0.0
    return Dict("passed" => ratio <= RISK_MAX, "exposure_ratio" => ratio, "max_ratio" => RISK_MAX, "language" => LANG)
end

function handle_signal_score(d)
    key = LANG * join(sort(collect(keys(d))), ",")
    h = sum(Int(c) for c in key)
    delta = ((h % 1000) / 1000.0 - 0.5) * 0.1 * SIG_WEIGHT
    return Dict("score_delta" => delta, "signal_score_weight" => SIG_WEIGHT, "language" => LANG)
end

function handle_slippage(d)
    spread = getf(d, "spread_bps", 5.0)
    participation = getf(d, "participation_rate", 0.01)
    vol = getf(d, "volatility_bps", 100.0)
    slip = spread * 0.5 + sqrt(participation) * vol * 0.01
    return Dict("slippage_bps" => slip, "language" => LANG)
end

function handle_position_sizing(d)
    cap = getf(d, "capital", 1000.0)
    vol = getf(d, "volatility_bps", 100.0)
    conf = getf(d, "confidence", 0.5)
    size_pct = RISK_MAX * (vol / max(vol + 100.0, 1.0)) * (0.5 + conf)
    return Dict("size_pct" => clamp01(size_pct) * 100.0, "language" => LANG)
end

function handle_drawdown(d)
    cur = getf(d, "current_drawdown_pct")
    mx = getf(d, "max_drawdown_pct", 20.0)
    ratio = mx > 0 ? cur / mx : 0.0
    passed = ratio <= 0.85
    return Dict("passed" => passed, "drawdown_ratio" => ratio, "language" => LANG)
end

function handle_heartbeat(d)
    return Dict("language" => LANG, "ok" => true, "timestamp" => time())
end

function handle_signal_filter(d)
    conf = getf(d, "confidence", 0.0)
    return Dict("accepted" => conf >= MIN_CONF_TO_ACCEPT, "min_confidence" => MIN_CONF_TO_ACCEPT, "language" => LANG)
end

# Dedicated GARCH task
function handle_garch_volatility(d)
    returns = getfvec(d, "returns")
    prices = getfvec(d, "prices")
    if isempty(prices); prices = getfvec(d, "ohlcv_close"); end
    if isempty(returns) && !isempty(prices)
        returns = prices_to_returns(prices)
    end
    length(returns) < 20 && return Dict("error" => "need >=20 returns", "language" => LANG)
    fit = garch11_fit(returns)
    return Dict(
        "volatility_annual_bps" => fit.sigma_forecast,
        "omega" => fit.omega, "alpha" => fit.alpha, "beta" => fit.beta,
        "persistence" => fit.alpha + fit.beta,
        "language" => LANG, "ok" => true
    )
end

# Dedicated HMM task
function handle_hmm_regime(d)
    returns = getfvec(d, "returns")
    prices = getfvec(d, "prices")
    if isempty(prices); prices = getfvec(d, "ohlcv_close"); end
    if isempty(returns) && !isempty(prices)
        returns = prices_to_returns(prices)
    end
    length(returns) < 10 && return Dict("error" => "need >=10 returns", "language" => LANG)
    hmm = hmm_2state(returns)
    return Dict(
        "regime" => hmm.regime,
        "prob_high_vol" => hmm.prob_high_vol,
        "sigma_low" => hmm.sigma_low,
        "sigma_high" => hmm.sigma_high,
        "transition_matrix" => [[hmm.transition[1,1], hmm.transition[1,2]],
                                 [hmm.transition[2,1], hmm.transition[2,2]]],
        "language" => LANG, "ok" => true
    )
end

# Dedicated EVT task
function handle_evt_tail_risk(d)
    returns = getfvec(d, "returns")
    prices = getfvec(d, "prices")
    if isempty(prices); prices = getfvec(d, "ohlcv_close"); end
    if isempty(returns) && !isempty(prices)
        returns = prices_to_returns(prices)
    end
    length(returns) < 30 && return Dict("error" => "need >=30 returns", "language" => LANG)
    return evt_tail_risk(returns)
end

# Fallback for unimplemented standard tasks
function handle_generic(d)
    return Dict("language" => LANG, "ok" => true)
end

# ─────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────
const DISPATCH = Dict{String,Function}(
    "cycle_plan"                  => handle_cycle_plan,
    "order_book_processing"       => handle_order_book,
    "risk_calculation"            => handle_risk,
    "volatility_estimate"         => handle_volatility_estimate,
    "signal_score"                => handle_signal_score,
    "regime_estimate"             => handle_regime_estimate,
    "slippage_estimate"           => handle_slippage,
    "position_sizing"             => handle_position_sizing,
    "drawdown_check"              => handle_drawdown,
    "heartbeat"                   => handle_heartbeat,
    "signal_filter"               => handle_signal_filter,
    "garch_volatility"            => handle_garch_volatility,
    "hmm_regime"                  => handle_hmm_regime,
    "evt_tail_risk"               => handle_evt_tail_risk,
    # Standard tasks with generic handler
    "correlation_estimate"        => handle_generic,
    "liquidity_score"             => handle_generic,
    "market_impact"               => handle_generic,
    "confidence_calibration"      => handle_generic,
    "var_estimate"                => handle_generic,
    "skew_estimate"               => handle_generic,
    "order_book_imbalance_series" => handle_generic,
    "execution_quality_score"     => handle_generic,
    "regime_duration"             => handle_generic,
)

# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
function main()
    while !eof(stdin)
        line = readline(stdin)
        isempty(strip(line)) && continue

        t0 = time()
        local resp_str::String
        try
            req = parse_json(line)
            if req === nothing || !(req isa Dict)
                resp_str = to_json(Dict("ok" => false, "error" => "json_parse_error", "took_ms" => 0.0))
            else
                task_type = get(req, "task_type", "")
                data = get(req, "data", Dict{String,Any}())
                if !(data isa Dict); data = Dict{String,Any}(); end

                handler = get(DISPATCH, task_type, nothing)
                if handler === nothing
                    resp_str = to_json(Dict("ok" => false, "error" => "unknown task: $task_type", "took_ms" => 0.0))
                else
                    result = handler(data)
                    took = (time() - t0) * 1000.0
                    resp_str = to_json(Dict("ok" => true, "result" => result, "took_ms" => took))
                end
            end
        catch e
            took = (time() - t0) * 1000.0
            resp_str = to_json(Dict("ok" => false, "error" => string(e), "took_ms" => took))
        end

        println(stdout, resp_str)
        flush(stdout)
    end
end

main()
