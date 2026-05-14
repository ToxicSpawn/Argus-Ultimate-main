# frozen_string_literal: true
# Ruby 3.4 — Argus multilang worker
# Run: C:\Ruby34-x64\bin\ruby.exe ruby_worker.rb
# Protocol: one JSON line in -> one JSON line out, loop until EOF

require 'json'
require 'digest'

$stdout.sync = true

LANGUAGE    = "ruby"
RISK_MAX    = 0.46
CYCLE_SCALE = 1.0
VOL_W       = 0.98
SIG_W       = 1.0
SPREAD      = 1.0

def sha256_int(str)
  hex = Digest::SHA256.hexdigest(str.to_s)
  hex[0, 16].to_i(16)
end

def dispatch(task_type, data, t0)
  case task_type

  when "cycle_plan"
    sorted = data.sort_by { |k, _| k.to_s }.map { |k, v| "#{k}:#{v}" }.join(",")
    h = sha256_int(sorted)
    base = ((h % 200) - 100) / 10_000.0
    cash    = data["cash"]&.to_f    || 0.0
    signals = data["signals"]&.to_f || 0.0
    tilt = (cash - 0.5) * 0.002 + signals * 0.001
    boost = (base + tilt) * CYCLE_SCALE
    boost = boost.clamp(-0.015, 0.015)
    { language: LANGUAGE, cycle_boost: boost.round(6), ok: true }

  when "order_book_processing"
    bids = Array(data["bids"]).map { |b| [b[0].to_f, b[1].to_f] }.sort_by { |p, _| -p }
    asks = Array(data["asks"]).map { |a| [a[0].to_f, a[1].to_f] }.sort_by { |p, _| p }
    best_bid = bids.first&.first || 0.0
    best_ask = asks.first&.first || 0.0
    mid = (best_bid + best_ask) / 2.0
    spread_bps = mid > 0 ? ((best_ask - best_bid) / mid * 10_000 * SPREAD).round(4) : 0.0
    top_bids = bids.first(5)
    top_asks = asks.first(5)
    bid_vol = top_bids.sum { |_, v| v }
    ask_vol = top_asks.sum { |_, v| v }
    total_vol = bid_vol + ask_vol
    imbalance = total_vol > 0 ? ((bid_vol - ask_vol) / total_vol).round(6) : 0.0
    { spread_bps: spread_bps, imbalance: imbalance, mid: mid.round(6), language: LANGUAGE }

  when "risk_calculation"
    position_value = data["position_value"]&.to_f || 0.0
    capital        = data["capital"]&.to_f        || 1.0
    capital = 1.0 if capital == 0.0
    exposure_ratio = position_value / capital
    passed = exposure_ratio <= RISK_MAX
    { passed: passed, exposure_ratio: exposure_ratio.round(6), max_ratio: RISK_MAX, language: LANGUAGE }

  when "volatility_estimate"
    prices = Array(data["prices"]).map(&:to_f)
    n = prices.length
    if n < 2
      return { volatility_annual_bps: 0.0, language: LANGUAGE, ok: true }
    end
    # Welford online variance on log-returns
    count = 0
    mean  = 0.0
    m2    = 0.0
    prices.each_cons(2) do |a, b|
      next if a <= 0 || b <= 0
      r = Math.log(b / a)
      count += 1
      delta  = r - mean
      mean  += delta / count
      m2    += delta * (r - mean)
    end
    var = count > 1 ? m2 / (count - 1) : 0.0
    vol = Math.sqrt(var * 252) * 10_000 * VOL_W
    { volatility_annual_bps: vol.round(4), volatility_weight: VOL_W, language: LANGUAGE, ok: true }

  when "signal_score"
    payload = data.sort_by { |k, _| k.to_s }.map { |k, v| "#{k}:#{v}" }.join(",")
    h = sha256_int(payload)
    delta = ((h % 100) - 50) / 5_000.0 * SIG_W
    { score_delta: delta.round(8), signal_score_weight: SIG_W, language: LANGUAGE, ok: true }

  when "regime_estimate"
    prices = Array(data["prices"]).map(&:to_f)
    n = prices.length
    if n < 2
      return { regime: "unknown", confidence: 0.0, language: LANGUAGE, ok: true }
    end
    returns = prices.each_cons(2).map { |a, b| a > 0 ? Math.log(b / a) : 0.0 }
    mean_r = returns.sum / returns.size
    variance = returns.sum { |r| (r - mean_r)**2 } / [returns.size - 1, 1].max
    vol_annual = Math.sqrt(variance * 252)
    regime = if vol_annual > 0.3 then "high_vol"
             elsif vol_annual > 0.15 then "normal"
             else "low_vol"
             end
    confidence = [1.0 - (vol_annual * 2.0), 0.1].max.clamp(0.0, 1.0)
    { regime: regime, confidence: confidence.round(4), regime_weight: 1.0, language: LANGUAGE, ok: true }

  when "slippage_estimate"
    half_spread   = data["half_spread_bps"]&.to_f   || 1.0
    participation = data["participation_rate"]&.to_f || 0.01
    slippage = half_spread * SPREAD * (1 + participation * 10)
    { slippage_bps: slippage.round(4), language: LANGUAGE, ok: true }

  when "position_sizing"
    vol_bps  = data["vol_bps"]&.to_f       || 100.0
    conf     = data["confidence"]&.to_f    || 0.5
    max_risk = data["max_risk_pct"]&.to_f  || 0.1
    size_pct = [RISK_MAX, max_risk * (vol_bps / 10.0) * (0.5 + conf)].min
    size_pct = size_pct.clamp(0.0, RISK_MAX)
    capital  = data["capital"]&.to_f || 100_000.0
    size_abs = size_pct * capital
    { size_pct: size_pct.round(6), size_abs: size_abs.round(2), language: LANGUAGE, ok: true }

  when "drawdown_check"
    peak    = data["peak_value"]&.to_f    || 1.0
    current = data["current_value"]&.to_f || 1.0
    max_dd  = data["max_drawdown"]&.to_f  || 0.2
    peak = 1.0 if peak == 0.0
    current_drawdown = (peak - current) / peak
    passed = current_drawdown <= max_dd * RISK_MAX
    { passed: passed, current_drawdown_pct: (current_drawdown * 100).round(4), language: LANGUAGE, ok: true }

  when "correlation_estimate"
    xs = Array(data["series_a"]).map(&:to_f)
    ys = Array(data["series_b"]).map(&:to_f)
    n  = [xs.size, ys.size].min
    if n < 2
      return { correlation: 0.0, language: LANGUAGE, ok: true }
    end
    xs = xs.first(n)
    ys = ys.first(n)
    mean_x = xs.sum / n
    mean_y = ys.sum / n
    num = xs.zip(ys).sum { |x, y| (x - mean_x) * (y - mean_y) }
    den_x = Math.sqrt(xs.sum { |x| (x - mean_x)**2 })
    den_y = Math.sqrt(ys.sum { |y| (y - mean_y)**2 })
    denom = den_x * den_y
    corr  = denom > 0 ? (num / denom).clamp(-1.0, 1.0) : 0.0
    { correlation: corr.round(6), language: LANGUAGE, ok: true }

  when "liquidity_score"
    bids = Array(data["bids"]).map { |b| [b[0].to_f, b[1].to_f] }.sort_by { |p, _| -p }
    asks = Array(data["asks"]).map { |a| [a[0].to_f, a[1].to_f] }.sort_by { |p, _| p }
    top5_vol = (bids.first(5) + asks.first(5)).sum { |_, v| v }
    score = [1.0, top5_vol / 100.0].min
    best_bid = bids.first&.first || 0.0
    best_ask = asks.first&.first || 0.0
    mid = (best_bid + best_ask) / 2.0
    depth_bps = mid > 0 ? ((best_ask - best_bid) / mid * 10_000).round(4) : 0.0
    { liquidity_score: score.round(6), depth_bps: depth_bps, language: LANGUAGE, ok: true }

  when "market_impact"
    qty  = data["quantity"]&.to_f  || 0.0
    adv  = data["adv"]&.to_f       || 1.0
    vol  = data["volatility"]&.to_f || 0.01
    adv  = 1.0 if adv == 0.0
    impact = 10.0 * Math.sqrt(qty / adv) * vol * 10_000
    { impact_bps: impact.round(4), language: LANGUAGE, ok: true }

  when "signal_filter"
    conf   = data["confidence"]&.to_f || 0.0
    regime = data["regime"]&.to_s     || "unknown"
    vol    = data["volatility"]&.to_f || 0.0
    accept = conf >= 0.5 && (regime != "high_vol" || vol < 0.02)
    reason = if conf < 0.5 then "low_confidence"
             elsif regime == "high_vol" && vol >= 0.02 then "high_vol_regime"
             else "passed"
             end
    { accept: accept, filter_reason: reason, language: LANGUAGE, ok: true }

  when "confidence_calibration"
    confidences = Array(data["confidences"]).map(&:to_f)
    win_rate    = data["win_rate"]&.to_f || 0.5
    avg_conf    = confidences.empty? ? 0.5 : confidences.sum / confidences.size
    calibrated  = 0.5 * avg_conf + 0.5 * win_rate
    { calibrated_confidence: calibrated.round(6), language: LANGUAGE, ok: true }

  when "heartbeat"
    { ok: true, latency_ms: 0.0, language: LANGUAGE, cycle_id: data["cycle_id"] || 0 }

  when "var_estimate"
    returns = Array(data["returns"]).map(&:to_f).sort
    n = returns.size
    if n == 0
      return { var_pct: 0.0, cvar_pct: 0.0, language: LANGUAGE, ok: true }
    end
    conf_level = data["confidence_level"]&.to_f || 0.95
    idx = [(n * (1.0 - conf_level)).floor, 0].max
    var_pct  = -returns[idx]
    cvar_arr = returns[0..idx]
    cvar_pct = cvar_arr.empty? ? 0.0 : -(cvar_arr.sum / cvar_arr.size)
    { var_pct: var_pct.round(6), cvar_pct: cvar_pct.round(6), language: LANGUAGE, ok: true }

  when "skew_estimate"
    vals = Array(data["returns"]).map(&:to_f)
    n = vals.size
    if n < 3
      return { skew: 0.0, language: LANGUAGE, ok: true }
    end
    mean  = vals.sum / n
    m2    = vals.sum { |v| (v - mean)**2 } / n
    m3    = vals.sum { |v| (v - mean)**3 } / n
    std   = Math.sqrt(m2)
    skew  = std > 0 ? m3 / (std**3) : 0.0
    { skew: skew.round(6), language: LANGUAGE, ok: true }

  when "order_book_imbalance_series"
    bids = Array(data["bids"]).map { |b| [b[0].to_f, b[1].to_f] }.sort_by { |p, _| -p }
    asks = Array(data["asks"]).map { |a| [a[0].to_f, a[1].to_f] }.sort_by { |p, _| p }
    bid_vol = bids.first(5).sum { |_, v| v }
    ask_vol = asks.first(5).sum { |_, v| v }
    total   = bid_vol + ask_vol
    imbalance = total > 0 ? (bid_vol - ask_vol) / total : 0.0
    series = [imbalance.round(6)]
    trend  = imbalance > 0.1 ? "bid_heavy" : imbalance < -0.1 ? "ask_heavy" : "balanced"
    { imbalance_series: series, trend: trend, language: LANGUAGE, ok: true }

  when "execution_quality_score"
    slippages = Array(data["slippage_bps"]).map(&:to_f)
    avg = slippages.empty? ? 0.0 : slippages.sum / slippages.size
    score = [0.0, 1.0 - avg / 50.0].max
    { score_0_1: score.round(6), avg_slippage_bps: avg.round(4), language: LANGUAGE, ok: true }

  when "regime_duration"
    bars   = data["bars_in_regime"]&.to_i || 0
    regime = data["regime"]&.to_s         || "unknown"
    stable = bars >= 5
    { bars_in_regime: bars, regime_stable: stable, regime: regime, language: LANGUAGE, ok: true }

  else
    raise "Unknown task_type: #{task_type}"
  end
end

# ── Main loop ──────────────────────────────────────────────────────────────────
loop do
  line = $stdin.gets
  break if line.nil?
  line = line.strip
  next if line.empty?

  t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  begin
    req  = JSON.parse(line)
    task = req["task_type"].to_s
    data = req["data"] || {}
    res  = dispatch(task, data, t0)
    took = (Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0) * 1000.0
    puts JSON.generate({ ok: true, result: res, took_ms: took.round(4) })
  rescue => e
    took = (Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0) * 1000.0
    puts JSON.generate({ ok: false, error: e.message, took_ms: took.round(4) })
  end
  $stdout.flush
end
