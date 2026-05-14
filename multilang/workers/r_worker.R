# R 4.5 — Argus multilang worker
# Run: Rscript r_worker.R
# Protocol: one JSON line in -> one JSON line out, loop until EOF

suppressPackageStartupMessages({
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    library(jsonlite)
    JSON_BACKEND <- "jsonlite"
  } else if (requireNamespace("RJSONIO", quietly = TRUE)) {
    library(RJSONIO)
    JSON_BACKEND <- "RJSONIO"
  } else {
    stop("No JSON package available. Install jsonlite: install.packages('jsonlite')")
  }
})

LANGUAGE    <- "r"
RISK_MAX    <- 0.44
CYCLE_SCALE <- 1.05
VOL_W       <- 1.2
SIG_W       <- 1.05
SPREAD_MULT <- 1.02

# ── JSON helpers ──────────────────────────────────────────────────────────────
json_parse <- function(txt) {
  if (JSON_BACKEND == "jsonlite") fromJSON(txt, simplifyVector = TRUE)
  else RJSONIO::fromJSON(txt)
}

json_encode <- function(x) {
  if (JSON_BACKEND == "jsonlite") toJSON(x, auto_unbox = TRUE, digits = 8)
  else RJSONIO::toJSON(x)
}

# ── Simple hash (SHA256 fallback via digest, or manual) ───────────────────────
sha256_int <- function(s) {
  if (requireNamespace("digest", quietly = TRUE)) {
    hex <- digest::digest(s, algo = "sha256", serialize = FALSE)
  } else {
    # Fallback: djb2-style hash from raw bytes
    bytes <- utf8ToInt(as.character(s))
    h <- 5381L
    for (b in bytes) h <- bitwAnd(bitwXor(bitwShiftL(h, 5L) + h, b), 0xFFFFFFFFL)
    return(abs(h))
  }
  # Convert first 16 hex chars to integer (use double to avoid overflow)
  sub16 <- substr(hex, 1, 13)  # 13 hex digits ~ 52 bits, safe in double
  strtoi_safe <- function(h) {
    # manual base-16 -> double
    chars <- strsplit(h, "")[[1]]
    val <- 0
    for (ch in chars) {
      d <- ifelse(ch >= "0" && ch <= "9", as.integer(ch),
                  as.integer(chartr("abcdef", "ABCDEF", ch)) - as.integer("A") + 10L)
      val <- val * 16 + d
    }
    val
  }
  strtoi_safe(sub16)
}

# ── Task implementations ───────────────────────────────────────────────────────

task_cycle_plan <- function(data) {
  keys   <- sort(names(data))
  sorted <- paste(keys, sapply(keys, function(k) data[[k]]), sep = ":", collapse = ",")
  h      <- sha256_int(sorted)
  base   <- ((h %% 200) - 100) / 10000.0
  cash    <- as.numeric(data[["cash"]])
  signals <- as.numeric(data[["signals"]])
  if (is.na(cash))    cash    <- 0.0
  if (is.na(signals)) signals <- 0.0
  tilt  <- (cash - 0.5) * 0.002 + signals * 0.001
  boost <- (base + tilt) * CYCLE_SCALE
  boost <- max(-0.015, min(0.015, boost))
  list(language = LANGUAGE, cycle_boost = round(boost, 6), ok = TRUE)
}

task_order_book_processing <- function(data) {
  parse_levels <- function(lvls) {
    if (is.null(lvls) || length(lvls) == 0) return(matrix(0, nrow = 0, ncol = 2))
    if (is.list(lvls)) {
      m <- do.call(rbind, lapply(lvls, function(l) as.numeric(l)))
    } else {
      m <- matrix(as.numeric(lvls), ncol = 2, byrow = TRUE)
    }
    m
  }
  bids <- parse_levels(data[["bids"]])
  asks <- parse_levels(data[["asks"]])
  if (nrow(bids) > 0) bids <- bids[order(-bids[, 1]), , drop = FALSE]
  if (nrow(asks) > 0) asks <- asks[order( asks[, 1]), , drop = FALSE]
  best_bid <- if (nrow(bids) > 0) bids[1, 1] else 0.0
  best_ask <- if (nrow(asks) > 0) asks[1, 1] else 0.0
  mid <- (best_bid + best_ask) / 2.0
  spread_bps <- if (mid > 0) round((best_ask - best_bid) / mid * 10000 * SPREAD_MULT, 4) else 0.0
  top5b <- if (nrow(bids) > 0) bids[seq_len(min(5, nrow(bids))), 2] else numeric(0)
  top5a <- if (nrow(asks) > 0) asks[seq_len(min(5, nrow(asks))), 2] else numeric(0)
  bid_vol <- sum(top5b)
  ask_vol <- sum(top5a)
  total   <- bid_vol + ask_vol
  imbalance <- if (total > 0) round((bid_vol - ask_vol) / total, 6) else 0.0
  list(spread_bps = spread_bps, imbalance = imbalance, mid = round(mid, 6), language = LANGUAGE)
}

task_risk_calculation <- function(data) {
  pos_val <- as.numeric(data[["position_value"]]); if (is.na(pos_val)) pos_val <- 0.0
  capital <- as.numeric(data[["capital"]]);         if (is.na(capital) || capital == 0) capital <- 1.0
  exposure <- pos_val / capital
  passed   <- exposure <= RISK_MAX
  list(passed = passed, exposure_ratio = round(exposure, 6), max_ratio = RISK_MAX, language = LANGUAGE)
}

task_volatility_estimate <- function(data) {
  prices <- as.numeric(unlist(data[["prices"]]))
  prices <- prices[!is.na(prices) & prices > 0]
  if (length(prices) < 2) return(list(volatility_annual_bps = 0.0, volatility_weight = VOL_W, language = LANGUAGE, ok = TRUE))
  returns <- diff(log(prices))
  n <- length(returns)
  if (n < 2) return(list(volatility_annual_bps = 0.0, volatility_weight = VOL_W, language = LANGUAGE, ok = TRUE))
  # Welford online variance
  mean_r <- 0.0; m2 <- 0.0
  for (i in seq_along(returns)) {
    delta  <- returns[i] - mean_r
    mean_r <- mean_r + delta / i
    m2     <- m2 + delta * (returns[i] - mean_r)
  }
  var_r <- m2 / (n - 1)
  vol   <- sqrt(var_r * 252) * 10000 * VOL_W
  list(volatility_annual_bps = round(vol, 4), volatility_weight = VOL_W, language = LANGUAGE, ok = TRUE)
}

task_signal_score <- function(data) {
  keys    <- sort(names(data))
  payload <- paste(keys, sapply(keys, function(k) data[[k]]), sep = ":", collapse = ",")
  h       <- sha256_int(payload)
  delta   <- ((h %% 100) - 50) / 5000.0 * SIG_W
  list(score_delta = round(delta, 8), signal_score_weight = SIG_W, language = LANGUAGE, ok = TRUE)
}

task_regime_estimate <- function(data) {
  prices <- as.numeric(unlist(data[["prices"]]))
  prices <- prices[!is.na(prices) & prices > 0]
  if (length(prices) < 2) return(list(regime = "unknown", confidence = 0.0, regime_weight = 1.0, language = LANGUAGE, ok = TRUE))
  rets <- diff(log(prices))
  if (length(rets) < 2) return(list(regime = "unknown", confidence = 0.0, regime_weight = 1.0, language = LANGUAGE, ok = TRUE))
  vol_annual <- sd(rets) * sqrt(252)
  regime <- if (vol_annual > 0.3) "high_vol" else if (vol_annual > 0.15) "normal" else "low_vol"
  confidence <- max(0.1, min(1.0, 1.0 - vol_annual * 2.0))
  list(regime = regime, confidence = round(confidence, 4), regime_weight = 1.0, language = LANGUAGE, ok = TRUE)
}

task_slippage_estimate <- function(data) {
  sz <- as.numeric(data[["size"]]);   if (is.na(sz)) sz <- 1.0
  sp <- as.numeric(data[["spread"]]); if (is.na(sp)) sp <- 0.01
  slippage <- sz * sp * SPREAD_MULT
  list(slippage_bps = round(slippage, 4), language = LANGUAGE, ok = TRUE)
}

task_position_sizing <- function(data) {
  capital   <- as.numeric(data[["capital"]]);      if (is.na(capital))   capital   <- 100000.0
  risk_pct  <- as.numeric(data[["risk_pct"]]);     if (is.na(risk_pct))  risk_pct  <- 0.01
  stop_dist <- as.numeric(data[["stop_dist"]]);    if (is.na(stop_dist) || stop_dist == 0) stop_dist <- 1.0
  size_abs <- capital * risk_pct / stop_dist
  size_pct <- (size_abs / capital) * 100.0
  list(size_pct = round(size_pct, 6), size_abs = round(size_abs, 2), language = LANGUAGE, ok = TRUE)
}

task_drawdown_check <- function(data) {
  peak    <- as.numeric(data[["peak"]]);    if (is.na(peak)    || peak == 0) peak    <- 1.0
  current <- as.numeric(data[["current"]]); if (is.na(current)) current <- 1.0
  dd <- (peak - current) / peak
  drawdown_pct <- dd * 100
  passed <- drawdown_pct <= 20.0
  list(passed = passed, current_drawdown_pct = round(drawdown_pct, 4), language = LANGUAGE, ok = TRUE)
}

task_correlation_estimate <- function(data) {
  xs <- as.numeric(unlist(data[["series_a"]]))
  ys <- as.numeric(unlist(data[["series_b"]]))
  n  <- min(length(xs), length(ys))
  if (n < 2) return(list(correlation = 0.0, language = LANGUAGE, ok = TRUE))
  xs <- xs[1:n]; ys <- ys[1:n]
  corr <- tryCatch(cor(xs, ys), error = function(e) 0.0)
  if (is.na(corr)) corr <- 0.0
  corr <- max(-1.0, min(1.0, corr))
  list(correlation = round(corr, 6), language = LANGUAGE, ok = TRUE)
}

task_liquidity_score <- function(data) {
  parse_levels <- function(lvls) {
    if (is.null(lvls) || length(lvls) == 0) return(numeric(0))
    if (is.list(lvls)) sapply(lvls, function(l) as.numeric(l)[2])
    else {
      m <- matrix(as.numeric(lvls), ncol = 2, byrow = TRUE)
      m[, 2]
    }
  }
  bids_v <- parse_levels(data[["bids"]])
  asks_v <- parse_levels(data[["asks"]])
  top5_vol <- sum(head(bids_v, 5)) + sum(head(asks_v, 5))
  score <- min(1.0, top5_vol / 100.0)

  parse_prices <- function(lvls) {
    if (is.null(lvls) || length(lvls) == 0) return(numeric(0))
    if (is.list(lvls)) sapply(lvls, function(l) as.numeric(l)[1])
    else { m <- matrix(as.numeric(lvls), ncol = 2, byrow = TRUE); m[, 1] }
  }
  bid_prices <- parse_prices(data[["bids"]])
  ask_prices <- parse_prices(data[["asks"]])
  best_bid <- if (length(bid_prices) > 0) max(bid_prices) else 0.0
  best_ask <- if (length(ask_prices) > 0) min(ask_prices) else 0.0
  mid      <- (best_bid + best_ask) / 2.0
  depth_bps <- if (mid > 0) round((best_ask - best_bid) / mid * 10000, 4) else 0.0
  list(liquidity_score = round(score, 6), depth_bps = depth_bps, language = LANGUAGE, ok = TRUE)
}

task_market_impact <- function(data) {
  qty <- as.numeric(data[["quantity"]]);   if (is.na(qty) || qty <= 0) qty <- 0.0
  adv <- as.numeric(data[["adv"]]);        if (is.na(adv) || adv <= 0) adv <- 1.0
  vol <- as.numeric(data[["volatility"]]); if (is.na(vol)) vol <- 0.01
  impact <- 10.0 * sqrt(qty / adv) * vol * 10000
  list(impact_bps = round(impact, 4), language = LANGUAGE, ok = TRUE)
}

task_signal_filter <- function(data) {
  signal    <- as.numeric(data[["signal"]]);    if (is.na(signal))    signal    <- 0.0
  threshold <- as.numeric(data[["threshold"]]); if (is.na(threshold)) threshold <- 0.5
  accept <- abs(signal) >= threshold
  list(accept = accept, filter_reason = "", language = LANGUAGE, ok = TRUE)
}

task_confidence_calibration <- function(data) {
  confs    <- as.numeric(unlist(data[["confidences"]])); confs <- confs[!is.na(confs)]
  win_rate <- as.numeric(data[["win_rate"]]);             if (is.na(win_rate)) win_rate <- 0.5
  avg_conf <- if (length(confs) > 0) mean(confs) else 0.5
  calibrated <- 0.5 * avg_conf + 0.5 * win_rate
  list(calibrated_confidence = round(calibrated, 6), language = LANGUAGE, ok = TRUE)
}

task_heartbeat <- function(data) {
  cycle_id <- data[["cycle_id"]]
  if (is.null(cycle_id)) cycle_id <- 0L
  list(ok = TRUE, latency_ms = 0.0, language = LANGUAGE, cycle_id = cycle_id)
}

task_var_estimate <- function(data) {
  returns    <- sort(as.numeric(unlist(data[["returns"]]))); returns <- returns[!is.na(returns)]
  conf_level <- as.numeric(data[["confidence_level"]]); if (is.na(conf_level)) conf_level <- 0.95
  n <- length(returns)
  if (n == 0) return(list(var_pct = 0.0, cvar_pct = 0.0, language = LANGUAGE, ok = TRUE))
  idx     <- max(1L, floor(n * (1.0 - conf_level)))
  var_pct <- -returns[idx]
  cvar_arr <- returns[1:idx]
  cvar_pct <- if (length(cvar_arr) > 0) -mean(cvar_arr) else 0.0
  list(var_pct = round(var_pct, 6), cvar_pct = round(cvar_pct, 6), language = LANGUAGE, ok = TRUE)
}

task_skew_estimate <- function(data) {
  vals <- as.numeric(unlist(data[["returns"]])); vals <- vals[!is.na(vals)]
  n    <- length(vals)
  if (n < 3) return(list(skew = 0.0, language = LANGUAGE, ok = TRUE))
  mn  <- mean(vals)
  m2  <- mean((vals - mn)^2)
  m3  <- mean((vals - mn)^3)
  std <- sqrt(m2)
  skew <- if (std > 0) m3 / (std^3) else 0.0
  list(skew = round(skew, 6), language = LANGUAGE, ok = TRUE)
}

task_order_book_imbalance_series <- function(data) {
  parse_levels <- function(lvls, col) {
    if (is.null(lvls) || length(lvls) == 0) return(numeric(0))
    if (is.list(lvls)) sapply(lvls, function(l) as.numeric(l)[col])
    else { m <- matrix(as.numeric(lvls), ncol = 2, byrow = TRUE); m[, col] }
  }
  bid_vols <- parse_levels(data[["bids"]], 2)
  ask_vols <- parse_levels(data[["asks"]], 2)
  bid_vol  <- sum(head(bid_vols, 5))
  ask_vol  <- sum(head(ask_vols, 5))
  total    <- bid_vol + ask_vol
  imbalance <- if (total > 0) (bid_vol - ask_vol) / total else 0.0
  series <- round(imbalance, 6)
  trend  <- if (imbalance > 0.1) "bid_heavy" else if (imbalance < -0.1) "ask_heavy" else "balanced"
  list(imbalance_series = list(series), trend = trend, language = LANGUAGE, ok = TRUE)
}

task_execution_quality_score <- function(data) {
  slippages <- as.numeric(unlist(data[["slippage_bps"]])); slippages <- slippages[!is.na(slippages)]
  avg   <- if (length(slippages) > 0) mean(slippages) else 0.0
  score <- max(0.0, 1.0 - avg / 50.0)
  list(score_0_1 = round(score, 6), avg_slippage_bps = round(avg, 4), language = LANGUAGE, ok = TRUE)
}

task_regime_duration <- function(data) {
  bars   <- as.integer(data[["bars_in_regime"]]); if (is.na(bars))   bars   <- 0L
  regime <- as.character(data[["regime"]]);        if (is.na(regime)) regime <- "unknown"
  stable <- bars >= 5L
  list(bars_in_regime = bars, regime_stable = stable, regime = regime, language = LANGUAGE, ok = TRUE)
}

# ── Dispatcher ────────────────────────────────────────────────────────────────
dispatch <- function(task_type, data) {
  switch(task_type,
    "cycle_plan"                   = task_cycle_plan(data),
    "order_book_processing"        = task_order_book_processing(data),
    "risk_calculation"             = task_risk_calculation(data),
    "volatility_estimate"          = task_volatility_estimate(data),
    "signal_score"                 = task_signal_score(data),
    "regime_estimate"              = task_regime_estimate(data),
    "slippage_estimate"            = task_slippage_estimate(data),
    "position_sizing"              = task_position_sizing(data),
    "drawdown_check"               = task_drawdown_check(data),
    "correlation_estimate"         = task_correlation_estimate(data),
    "liquidity_score"              = task_liquidity_score(data),
    "market_impact"                = task_market_impact(data),
    "signal_filter"                = task_signal_filter(data),
    "confidence_calibration"       = task_confidence_calibration(data),
    "heartbeat"                    = task_heartbeat(data),
    "var_estimate"                 = task_var_estimate(data),
    "skew_estimate"                = task_skew_estimate(data),
    "order_book_imbalance_series"  = task_order_book_imbalance_series(data),
    "execution_quality_score"      = task_execution_quality_score(data),
    "regime_duration"              = task_regime_duration(data),
    stop(paste("Unknown task_type:", task_type))
  )
}

# ── Main loop ─────────────────────────────────────────────────────────────────
con <- file("stdin", open = "r", blocking = TRUE)
repeat {
  line <- tryCatch(
    readLines(con, n = 1L, warn = FALSE),
    error = function(e) NULL
  )
  if (is.null(line) || length(line) == 0) break
  line <- trimws(line)
  if (nchar(line) == 0) next

  t0 <- proc.time()[["elapsed"]]
  result_json <- tryCatch({
    req      <- json_parse(line)
    task     <- as.character(req[["task_type"]])
    data     <- req[["data"]]
    if (is.null(data)) data <- list()
    res      <- dispatch(task, data)
    took_ms  <- (proc.time()[["elapsed"]] - t0) * 1000.0
    json_encode(list(ok = TRUE, result = res, took_ms = round(took_ms, 4)))
  }, error = function(e) {
    took_ms <- (proc.time()[["elapsed"]] - t0) * 1000.0
    json_encode(list(ok = FALSE, error = conditionMessage(e), took_ms = round(took_ms, 4)))
  })

  cat(result_json, "\n", sep = "")
  flush(stdout())
}
close(con)
