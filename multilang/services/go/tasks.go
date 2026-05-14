package main

import (
	"hash/fnv"
	"math"
	"sort"
	"sync"
)

// ---------------------------------------------------------------------------
// Go language profile constants
// ---------------------------------------------------------------------------

const (
	profileRiskMaxRatio         = 0.47
	profileCycleBoostScale      = 1.0
	profileVolatilityWeight     = 0.95
	profileSignalScoreWeight    = 1.0
	profileSpreadMult           = 1.0
	profileRole                 = "speed"
	profileRegimeWeight         = 1.0
	profileDrawdownMaxRatio     = 1.0
	profileSlippageToleranceBPS = 80
	profileMinConfToAccept      = 0.5
)

// Profile returns the full profile map for /capabilities.
func Profile() M {
	return M{
		"risk_max_ratio":          profileRiskMaxRatio,
		"cycle_boost_scale":       profileCycleBoostScale,
		"volatility_weight":       profileVolatilityWeight,
		"signal_score_weight":     profileSignalScoreWeight,
		"spread_mult":             profileSpreadMult,
		"role":                    profileRole,
		"regime_weight":           profileRegimeWeight,
		"drawdown_max_ratio":      profileDrawdownMaxRatio,
		"slippage_tolerance_bps":  profileSlippageToleranceBPS,
		"min_confidence_to_accept": profileMinConfToAccept,
	}
}

// M is a shorthand for the generic JSON map type.
type M = map[string]interface{}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func fnv1aHash(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}

func toFloat(v interface{}) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case nil:
		return 0
	default:
		return 0
	}
}

func toFloatOr(v interface{}, fallback float64) float64 {
	if v == nil {
		return fallback
	}
	f := toFloat(v)
	if f == 0 {
		return fallback
	}
	return f
}

func toFloatSlice(v interface{}) []float64 {
	if v == nil {
		return nil
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	out := make([]float64, 0, len(arr))
	for _, elem := range arr {
		out = append(out, toFloat(elem))
	}
	return out
}

// toPairSlice converts [[price, size], ...] from JSON into [][2]float64.
func toPairSlice(v interface{}) [][2]float64 {
	if v == nil {
		return nil
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	out := make([][2]float64, 0, len(arr))
	for _, elem := range arr {
		switch row := elem.(type) {
		case []interface{}:
			if len(row) >= 2 {
				out = append(out, [2]float64{toFloat(row[0]), toFloat(row[1])})
			}
		}
	}
	return out
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func sortedKeyString(data M) string {
	// Deterministic serialisation for hashing: produce a stable key string.
	keys := make([]string, 0, len(data))
	for k := range data {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	buf := make([]byte, 0, 256)
	for _, k := range keys {
		buf = append(buf, k...)
		buf = append(buf, ':')
		switch v := data[k].(type) {
		case string:
			buf = append(buf, v...)
		case float64:
			buf = append(buf, []byte(floatStr(v))...)
		case bool:
			if v {
				buf = append(buf, '1')
			} else {
				buf = append(buf, '0')
			}
		default:
			buf = append(buf, '?')
		}
		buf = append(buf, ',')
	}
	return string(buf)
}

func floatStr(f float64) string {
	// Quick float-to-string for hashing (not for JSON output).
	if f == 0 {
		return "0"
	}
	sign := ""
	if f < 0 {
		sign = "-"
		f = -f
	}
	intPart := int64(f)
	fracPart := int64((f - float64(intPart)) * 1e6)
	if fracPart < 0 {
		fracPart = -fracPart
	}
	return sign + intToStr(intPart) + "." + intToStr(fracPart)
}

func intToStr(n int64) string {
	if n == 0 {
		return "0"
	}
	buf := [20]byte{}
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// ---------------------------------------------------------------------------
// 1. cycle_plan
// ---------------------------------------------------------------------------

func handleCyclePlan(data M) M {
	h := fnv1aHash(sortedKeyString(data))
	base := (float64(h%200) - 100.0) / 10000.0

	signals := int(toFloat(data["signals"]))
	cash := toFloat(data["cash_balance_aud"])
	pv := toFloatOr(data["portfolio_value_aud"], 1.0)
	cashRatio := 0.0
	if pv != 0 {
		cashRatio = cash / pv
	}
	tilt := (cashRatio-0.5)*0.002 + float64(signals%3-1)*0.001
	boost := clamp((base+tilt)*profileCycleBoostScale, -0.015, 0.015)

	return M{
		"language":         Lang,
		"cycle_boost":      boost,
		"cycle_boost_scale": profileCycleBoostScale,
		"ok":               true,
	}
}

// ---------------------------------------------------------------------------
// 2. order_book_processing
// ---------------------------------------------------------------------------

func handleOrderBookProcessing(data M) M {
	bids := toPairSlice(data["bids"])
	asks := toPairSlice(data["asks"])

	if len(bids) == 0 && len(asks) == 0 {
		return M{"spread_bps": 0.0, "imbalance": 0.0, "mid": 0.0, "language": Lang, "spread_mult": profileSpreadMult}
	}

	// Concurrent bid/ask volume calculation
	var bestBid, bestAsk float64
	var bidVol, askVol float64
	var wg sync.WaitGroup

	wg.Add(2)
	go func() {
		defer wg.Done()
		if len(bids) > 0 {
			bestBid = bids[0][0]
		}
		top := minInt(5, len(bids))
		for i := 0; i < top; i++ {
			bidVol += bids[i][1]
		}
	}()
	go func() {
		defer wg.Done()
		if len(asks) > 0 {
			bestAsk = asks[0][0]
		}
		top := minInt(5, len(asks))
		for i := 0; i < top; i++ {
			askVol += asks[i][1]
		}
	}()
	wg.Wait()

	mid := 0.0
	if bestBid > 0 && bestAsk > 0 {
		mid = (bestBid + bestAsk) / 2.0
	}
	rawSpread := 0.0
	if mid > 0 {
		rawSpread = (bestAsk - bestBid) / mid * 1e4
	}
	spreadBPS := rawSpread * profileSpreadMult

	total := bidVol + askVol
	imbalance := 0.0
	if total > 0 {
		imbalance = (bidVol - askVol) / total
	}

	return M{
		"spread_bps":  spreadBPS,
		"imbalance":   imbalance,
		"mid":         mid,
		"language":    Lang,
		"spread_mult": profileSpreadMult,
	}
}

// ---------------------------------------------------------------------------
// 3. risk_calculation
// ---------------------------------------------------------------------------

func handleRiskCalculation(data M) M {
	pv := toFloat(data["position_value"])
	capital := toFloatOr(data["capital"], 1.0)
	ratio := 0.0
	if capital != 0 {
		ratio = pv / capital
	}
	passed := ratio <= profileRiskMaxRatio
	reason := ""
	if !passed {
		reason = "exposure_exceeds_max"
	}
	return M{
		"passed":         passed,
		"exposure_ratio": ratio,
		"max_ratio":      profileRiskMaxRatio,
		"reason":         reason,
		"language":       Lang,
		"ok":             true,
	}
}

// ---------------------------------------------------------------------------
// 4. signal_score
// ---------------------------------------------------------------------------

func handleSignalScore(data M) M {
	confidence := toFloat(data["confidence"])
	baseScore := toFloat(data["score"])
	if baseScore == 0 {
		baseScore = confidence
	}

	h := fnv1aHash(Lang + sortedKeyString(data))
	delta := (float64(h%100) - 50.0) / 5000.0
	scoreDelta := delta * profileSignalScoreWeight

	return M{
		"score_delta":        scoreDelta,
		"signal_score_weight": profileSignalScoreWeight,
		"base_score":         baseScore,
		"language":           Lang,
		"ok":                 true,
	}
}

// ---------------------------------------------------------------------------
// 5. volatility_estimate  (Welford's online algorithm + math.FMA)
// ---------------------------------------------------------------------------

func handleVolatilityEstimate(data M) M {
	returns := toFloatSlice(data["returns"])
	prices := toFloatSlice(data["prices"])
	if prices == nil {
		prices = toFloatSlice(data["ohlcv_close"])
	}

	var vol float64

	if len(returns) > 0 {
		vol = welfordVolatility(returns)
	} else if len(prices) >= 2 {
		rets := make([]float64, 0, len(prices)-1)
		for i := 1; i < len(prices); i++ {
			if prices[i-1] != 0 {
				rets = append(rets, (prices[i]-prices[i-1])/prices[i-1])
			}
		}
		if len(rets) > 0 {
			vol = welfordVolatility(rets)
		} else {
			vol = 10.0
		}
	} else {
		vol = 10.0
	}

	// Language-specific seed adjustment (mirrors Python reference)
	seed := 0
	for _, c := range Lang {
		seed += int(c)
	}
	seed = seed % 7
	volAdj := vol * (1.0 + float64(seed-3)*0.01) * profileVolatilityWeight

	return M{
		"volatility_annual_bps": volAdj,
		"volatility_weight":     profileVolatilityWeight,
		"language":              Lang,
		"ok":                    true,
	}
}

// welfordVolatility computes annualised volatility in bps using Welford's
// numerically stable one-pass algorithm and math.FMA for precision.
func welfordVolatility(returns []float64) float64 {
	n := len(returns)
	if n == 0 {
		return 10.0
	}
	var mean, m2 float64
	for i, r := range returns {
		delta := r - mean
		mean += delta / float64(i+1)
		delta2 := r - mean
		// FMA: m2 = m2 + delta*delta2
		m2 = math.FMA(delta, delta2, m2)
	}
	variance := m2 / float64(n)
	if variance <= 0 {
		return 10.0
	}
	// annualised bps: sqrt(var * 252) * 10000
	return math.Sqrt(math.FMA(variance, 252.0, 0)) * 1e4
}

// ---------------------------------------------------------------------------
// 6. regime_estimate
// ---------------------------------------------------------------------------

func handleRegimeEstimate(data M) M {
	prices := toFloatSlice(data["prices"])
	if prices == nil {
		prices = toFloatSlice(data["returns"])
	}

	regime := "mean_revert"
	confidence := 0.5

	if len(prices) >= 3 {
		// Build returns
		rets := make([]float64, 0, len(prices)-1)
		for i := 1; i < len(prices); i++ {
			if prices[i-1] != 0 {
				rets = append(rets, (prices[i]-prices[i-1])/prices[i-1])
			}
		}
		// Rolling vol
		sumSq := 0.0
		for _, r := range rets {
			sumSq = math.FMA(r, r, sumSq)
		}
		vol := 10.0
		if len(rets) > 0 {
			vol = math.Sqrt(sumSq / float64(len(rets)) * 252.0 * 1e4)
		}
		// Trend slope
		trend := 0.0
		if prices[0] != 0 {
			trend = (prices[len(prices)-1] - prices[0]) / prices[0]
		}

		if vol > 20.0 {
			regime = "high_vol"
		} else if math.Abs(trend) > 0.02 {
			regime = "trend"
		} else {
			regime = "mean_revert"
		}
		confidence = math.Min(0.95, 0.5+math.Abs(trend)*5+vol/100)
	}

	return M{
		"regime":        regime,
		"confidence":    confidence,
		"language":      Lang,
		"regime_weight": profileRegimeWeight,
		"ok":            true,
	}
}

// ---------------------------------------------------------------------------
// 7. slippage_estimate  (walk-the-book)
// ---------------------------------------------------------------------------

func handleSlippageEstimate(data M) M {
	// Get order book from nested or flat structure
	ob, ok := data["order_book"].(map[string]interface{})
	var bids, asks [][2]float64
	if ok {
		bids = toPairSlice(ob["bids"])
		asks = toPairSlice(ob["asks"])
	}
	if bids == nil {
		bids = toPairSlice(data["bids"])
	}
	if asks == nil {
		asks = toPairSlice(data["asks"])
	}

	quantity := toFloat(data["quantity"])
	participation := toFloatOr(data["participation_rate"], 0.01)

	if len(bids) == 0 && len(asks) == 0 {
		return M{"slippage_bps": 0.0, "language": Lang, "ok": true}
	}

	bestBid := 0.0
	if len(bids) > 0 {
		bestBid = bids[0][0]
	}
	bestAsk := 0.0
	if len(asks) > 0 {
		bestAsk = asks[0][0]
	}
	mid := 0.0
	if bestBid > 0 && bestAsk > 0 {
		mid = (bestBid + bestAsk) / 2.0
	}

	// Walk the book on the ask side
	if quantity > 0 && len(asks) > 0 && mid > 0 {
		remaining := quantity
		totalCost := 0.0
		filled := 0.0
		for _, level := range asks {
			price := level[0]
			size := level[1]
			take := math.Min(remaining, size)
			totalCost = math.FMA(take, price, totalCost)
			filled += take
			remaining -= take
			if remaining <= 0 {
				break
			}
		}
		if filled > 0 && remaining <= 0 {
			avgPrice := totalCost / filled
			slippageBPS := (avgPrice - mid) / mid * 1e4
			return M{"slippage_bps": slippageBPS, "language": Lang, "ok": true}
		}
	}

	// Fallback: half-spread model
	halfSpreadBPS := 5.0
	if mid > 0 {
		halfSpreadBPS = (bestAsk - bestBid) / mid * 1e4 / 2.0
	}
	slippageBPS := halfSpreadBPS * profileSpreadMult * (1.0 + participation*10.0)

	return M{"slippage_bps": slippageBPS, "language": Lang, "ok": true}
}

// ---------------------------------------------------------------------------
// 8. position_sizing  (Kelly criterion)
// ---------------------------------------------------------------------------

func handlePositionSizing(data M) M {
	capital := toFloatOr(data["capital"], 1.0)
	volBPS := toFloat(data["volatility_bps"])
	if volBPS == 0 {
		volBPS = toFloat(data["volatility_annual_bps"])
	}
	if volBPS == 0 {
		volBPS = 10.0
	}
	confidence := toFloatOr(data["confidence"], 0.5)
	maxRiskPct := toFloatOr(data["max_risk_pct"], 0.02)

	// Kelly: edge = confidence - (1-confidence) = 2*confidence - 1
	// f* = edge/odds, simplified: use maxRiskPct scaled by vol & confidence
	sizePct := math.Min(profileRiskMaxRatio, maxRiskPct*(volBPS/10.0)*(0.5+confidence))

	return M{
		"size_pct": sizePct,
		"size_abs": sizePct * capital,
		"language": Lang,
		"ok":       true,
	}
}

// ---------------------------------------------------------------------------
// 9. drawdown_check
// ---------------------------------------------------------------------------

func handleDrawdownCheck(data M) M {
	maxDrawdownPct := toFloatOr(data["max_drawdown_pct"], 0.12)
	current := toFloat(data["current_equity"])
	peak := toFloatOr(data["peak_equity"], current)
	if peak == 0 {
		peak = 1.0
	}

	currentDrawdownPct := (peak - current) / peak
	passed := currentDrawdownPct <= maxDrawdownPct*profileDrawdownMaxRatio

	return M{
		"passed":               passed,
		"current_drawdown_pct": currentDrawdownPct,
		"language":             Lang,
		"ok":                   true,
	}
}

// ---------------------------------------------------------------------------
// 10. correlation_estimate  (Pearson, two-pass for stability)
// ---------------------------------------------------------------------------

func handleCorrelationEstimate(data M) M {
	a := toFloatSlice(data["series_a"])
	if a == nil {
		a = toFloatSlice(data["returns_a"])
	}
	b := toFloatSlice(data["series_b"])
	if b == nil {
		b = toFloatSlice(data["returns_b"])
	}

	if len(a) != len(b) || len(a) < 2 {
		return M{"correlation": 0.0, "language": Lang, "ok": true}
	}

	n := len(a)
	// Pass 1: means
	sumA, sumB := 0.0, 0.0
	for i := 0; i < n; i++ {
		sumA += a[i]
		sumB += b[i]
	}
	meanA := sumA / float64(n)
	meanB := sumB / float64(n)

	// Pass 2: variances and covariance
	var va, vb, cov float64
	for i := 0; i < n; i++ {
		da := a[i] - meanA
		db := b[i] - meanB
		va = math.FMA(da, da, va)
		vb = math.FMA(db, db, vb)
		cov = math.FMA(da, db, cov)
	}

	den := math.Sqrt(va * vb)
	corr := 0.0
	if den > 0 {
		corr = cov / den
	}
	corr = clamp(corr, -1.0, 1.0)

	return M{"correlation": corr, "language": Lang, "ok": true}
}

// ---------------------------------------------------------------------------
// 11. liquidity_score  (depth-weighted exponential decay)
// ---------------------------------------------------------------------------

func handleLiquidityScore(data M) M {
	bids := toPairSlice(data["bids"])
	asks := toPairSlice(data["asks"])
	depthLevels := int(toFloatOr(data["depth_levels"], 5))

	if len(bids) == 0 && len(asks) == 0 {
		return M{"liquidity_score": 0.0, "depth_bps": 0.0, "language": Lang, "ok": true}
	}

	bestBid := 0.0
	if len(bids) > 0 {
		bestBid = bids[0][0]
	}
	bestAsk := 0.0
	if len(asks) > 0 {
		bestAsk = asks[0][0]
	}
	mid := 0.0
	if bestBid > 0 && bestAsk > 0 {
		mid = (bestBid + bestAsk) / 2.0
	}
	depthBPS := 100.0
	if mid > 0 {
		depthBPS = (bestAsk - bestBid) / mid * 1e4
	}

	// Exponential decay weighting: level i gets weight exp(-0.3*i)
	weightedTotal := 0.0
	topBids := minInt(depthLevels, len(bids))
	topAsks := minInt(depthLevels, len(asks))
	for i := 0; i < topBids; i++ {
		w := math.Exp(-0.3 * float64(i))
		weightedTotal = math.FMA(w, bids[i][1], weightedTotal)
	}
	for i := 0; i < topAsks; i++ {
		w := math.Exp(-0.3 * float64(i))
		weightedTotal = math.FMA(w, asks[i][1], weightedTotal)
	}

	score := math.Min(1.0, weightedTotal/100.0)

	return M{"liquidity_score": score, "depth_bps": depthBPS, "language": Lang, "ok": true}
}

// ---------------------------------------------------------------------------
// 12. market_impact  (Almgren-Chriss)
// ---------------------------------------------------------------------------

func handleMarketImpact(data M) M {
	quantity := toFloat(data["quantity"])
	adv := toFloatOr(data["adv"], 1.0)
	volatility := toFloatOr(data["volatility"], 0.01)
	participation := 0.0
	if adv > 0 {
		participation = quantity / adv
	}

	// Almgren-Chriss: sigma * sqrt(Q/V) * (0.5 + 0.1 * participation)
	impactBPS := volatility * math.Sqrt(participation) * (0.5 + 0.1*participation) * 1e4

	return M{"impact_bps": impactBPS, "language": Lang, "ok": true}
}

// ---------------------------------------------------------------------------
// 13. signal_filter
// ---------------------------------------------------------------------------

func handleSignalFilter(data M) M {
	// Extract confidence from nested signal or top-level
	confidence := toFloat(data["confidence"])
	if sig, ok := data["signal"].(map[string]interface{}); ok {
		if c := toFloat(sig["confidence"]); c != 0 {
			confidence = c
		}
	}
	regime := "mean_revert"
	if r, ok := data["regime"].(string); ok {
		regime = r
	}
	volatility := toFloat(data["volatility"])

	accept := confidence >= profileMinConfToAccept && !(regime == "high_vol" && volatility >= 0.02)
	reason := ""
	if !accept {
		reason = "low_confidence_or_regime"
	}

	return M{
		"accept":        accept,
		"filter_reason": reason,
		"language":      Lang,
		"ok":            true,
	}
}

// ---------------------------------------------------------------------------
// 14. confidence_calibration  (Bayesian blend)
// ---------------------------------------------------------------------------

func handleConfidenceCalibration(data M) M {
	confs := toFloatSlice(data["historical_confidences"])
	pnls := toFloatSlice(data["historical_pnl"])

	if len(confs) != len(pnls) || len(confs) < 2 {
		return M{"calibrated_confidence": 0.5, "language": Lang, "ok": true}
	}

	wins := 0
	sumConf := 0.0
	for i, p := range pnls {
		if p > 0 {
			wins++
		}
		sumConf += confs[i]
	}
	avgConf := sumConf / float64(len(confs))
	winRate := float64(wins) / float64(len(pnls))
	calibrated := clamp(0.5*avgConf+0.5*winRate, 0.0, 1.0)

	return M{"calibrated_confidence": calibrated, "language": Lang, "ok": true}
}

// ---------------------------------------------------------------------------
// 15. heartbeat
// ---------------------------------------------------------------------------

func handleHeartbeat(data M) M {
	cycleID := data["cycle_id"]
	if cycleID == nil {
		cycleID = 0
	}
	return M{
		"ok":         true,
		"latency_ms": 0.0,
		"language":   Lang,
		"cycle_id":   cycleID,
	}
}

// ---------------------------------------------------------------------------
// 16. var_estimate  (historical simulation)
// ---------------------------------------------------------------------------

func handleVarEstimate(data M) M {
	returns := toFloatSlice(data["returns"])
	confidenceLevel := toFloatOr(data["confidence_level"], 0.95)

	if len(returns) < 5 {
		return M{"var_pct": 0.0, "cvar_pct": 0.0, "language": Lang, "ok": true}
	}

	sorted := make([]float64, len(returns))
	copy(sorted, returns)
	sort.Float64s(sorted)

	idx := int(float64(len(sorted)) * (1.0 - confidenceLevel))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	varPct := -sorted[idx] * 100.0

	// CVaR: mean of sorted[:idx+1]
	sum := 0.0
	cnt := idx + 1
	for i := 0; i < cnt; i++ {
		sum += sorted[i]
	}
	cvarPct := -sum / float64(cnt) * 100.0

	return M{"var_pct": varPct, "cvar_pct": cvarPct, "language": Lang, "ok": true}
}

// ---------------------------------------------------------------------------
// 17. skew_estimate  (Welford's extension for 3rd central moment)
// ---------------------------------------------------------------------------

func handleSkewEstimate(data M) M {
	returns := toFloatSlice(data["returns"])
	if len(returns) < 3 {
		return M{"skew": 0.0, "language": Lang, "ok": true}
	}

	n := float64(len(returns))

	// Single-pass Welford for mean and variance
	mean := 0.0
	m2 := 0.0
	for i, r := range returns {
		delta := r - mean
		mean += delta / float64(i+1)
		delta2 := r - mean
		m2 = math.FMA(delta, delta2, m2)
	}
	variance := m2 / n
	std := math.Sqrt(variance)

	if std == 0 {
		return M{"skew": 0.0, "language": Lang, "ok": true}
	}

	// Second pass for third central moment
	m3 := 0.0
	for _, r := range returns {
		d := (r - mean) / std
		m3 = math.FMA(d*d, d, m3)
	}
	skew := m3 / n

	return M{"skew": skew, "language": Lang, "ok": true}
}

// ---------------------------------------------------------------------------
// 18. order_book_imbalance_series  (1, 3, 5 depth levels)
// ---------------------------------------------------------------------------

func handleOrderBookImbalanceSeries(data M) M {
	bids := toPairSlice(data["bids"])
	asks := toPairSlice(data["asks"])

	if len(bids) == 0 && len(asks) == 0 {
		return M{"imbalance_series": []interface{}{}, "trend": 0.0, "language": Lang, "ok": true}
	}

	depths := []int{1, 3, 5}
	series := make([]interface{}, 0, len(depths))

	for _, d := range depths {
		bidVol := 0.0
		askVol := 0.0
		topB := minInt(d, len(bids))
		topA := minInt(d, len(asks))
		for i := 0; i < topB; i++ {
			bidVol += bids[i][1]
		}
		for i := 0; i < topA; i++ {
			askVol += asks[i][1]
		}
		total := bidVol + askVol
		imb := 0.0
		if total > 0 {
			imb = (bidVol - askVol) / total
		}
		series = append(series, imb)
	}

	// Trend: difference between deepest and shallowest imbalance
	first := series[0].(float64)
	last := series[len(series)-1].(float64)
	trend := last - first

	return M{"imbalance_series": series, "trend": trend, "language": Lang, "ok": true}
}

// ---------------------------------------------------------------------------
// 19. execution_quality_score
// ---------------------------------------------------------------------------

func handleExecutionQualityScore(data M) M {
	fillsRaw, _ := data["fills"].([]interface{})
	decisionsRaw, _ := data["decision_prices"].([]interface{})

	if len(fillsRaw) == 0 || len(fillsRaw) != len(decisionsRaw) {
		return M{"score_0_1": 1.0, "avg_slippage_bps": 0.0, "language": Lang, "ok": true}
	}

	limit := minInt(10, len(fillsRaw))
	totalSlippage := 0.0
	count := 0

	for i := 0; i < limit; i++ {
		// Fill price: could be a dict {"price": X} or a raw number
		var fp float64
		switch f := fillsRaw[i].(type) {
		case map[string]interface{}:
			fp = toFloat(f["price"])
		default:
			fp = toFloat(f)
		}
		dp := toFloat(decisionsRaw[i])

		if dp != 0 && fp != 0 {
			totalSlippage += math.Abs(fp-dp) / dp * 1e4
			count++
		}
	}

	avgBPS := 0.0
	if count > 0 {
		avgBPS = totalSlippage / float64(count)
	}
	score := clamp(1.0-avgBPS/50.0, 0.0, 1.0)

	return M{"score_0_1": score, "avg_slippage_bps": avgBPS, "language": Lang, "ok": true}
}

// ---------------------------------------------------------------------------
// 20. regime_duration
// ---------------------------------------------------------------------------

func handleRegimeDuration(data M) M {
	prices := toFloatSlice(data["prices"])
	regimeHistory, _ := data["regime_history"].([]interface{})

	if len(prices) < 2 {
		return M{"bars_in_regime": 0, "regime_stable": false, "regime": "unknown", "language": Lang, "ok": true}
	}

	// Compute current regime from prices
	rets := make([]float64, 0, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i-1] != 0 {
			rets = append(rets, (prices[i]-prices[i-1])/prices[i-1])
		}
	}
	sumSq := 0.0
	for _, r := range rets {
		sumSq = math.FMA(r, r, sumSq)
	}
	vol := 10.0
	if len(rets) > 0 {
		vol = math.Sqrt(sumSq / float64(len(rets)) * 252.0 * 1e4)
	}

	regime := "mean_revert"
	if vol > 20.0 {
		regime = "high_vol"
	}

	bars := len(regimeHistory)
	if bars == 0 {
		bars = minInt(10, len(prices))
	}

	return M{
		"bars_in_regime": bars,
		"regime_stable":  bars >= 5,
		"regime":         regime,
		"language":       Lang,
		"ok":             true,
	}
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

// TaskTypes lists all 20 supported task types.
var TaskTypes = []string{
	"cycle_plan", "order_book_processing", "risk_calculation",
	"signal_score", "volatility_estimate", "regime_estimate",
	"slippage_estimate", "position_sizing", "drawdown_check",
	"correlation_estimate", "liquidity_score", "market_impact",
	"signal_filter", "confidence_calibration", "heartbeat",
	"var_estimate", "skew_estimate", "order_book_imbalance_series",
	"execution_quality_score", "regime_duration",
}

var dispatch = map[string]func(M) M{
	"cycle_plan":                   handleCyclePlan,
	"order_book_processing":        handleOrderBookProcessing,
	"risk_calculation":             handleRiskCalculation,
	"signal_score":                 handleSignalScore,
	"volatility_estimate":          handleVolatilityEstimate,
	"regime_estimate":              handleRegimeEstimate,
	"slippage_estimate":            handleSlippageEstimate,
	"position_sizing":              handlePositionSizing,
	"drawdown_check":               handleDrawdownCheck,
	"correlation_estimate":         handleCorrelationEstimate,
	"liquidity_score":              handleLiquidityScore,
	"market_impact":                handleMarketImpact,
	"signal_filter":                handleSignalFilter,
	"confidence_calibration":       handleConfidenceCalibration,
	"heartbeat":                    handleHeartbeat,
	"var_estimate":                 handleVarEstimate,
	"skew_estimate":                handleSkewEstimate,
	"order_book_imbalance_series":  handleOrderBookImbalanceSeries,
	"execution_quality_score":      handleExecutionQualityScore,
	"regime_duration":              handleRegimeDuration,
}

// Execute dispatches a task to the appropriate handler.
func Execute(taskType string, data M) (M, error) {
	fn, ok := dispatch[taskType]
	if !ok {
		return M{"language": Lang, "ok": true}, nil
	}
	return fn(data), nil
}
