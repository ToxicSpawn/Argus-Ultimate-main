// go_worker.go — Argus Go stdin/stdout worker
// Reads one JSON line from stdin, processes the task, writes one JSON result
// line to stdout, then loops. Never exits until stdin is closed.
//
// Build:  go build -o go_worker go_worker.go
// Run:    echo '{"task_type":"heartbeat","data":{}}' | ./go_worker

package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"
)

// ---------------------------------------------------------------------------
// Language profile constants — Go
// ---------------------------------------------------------------------------

const (
	lang               = "go"
	riskMax            = 0.47
	cycleScale         = 1.0
	volWeight          = 0.95
	sigWeight          = 1.0
	spreadMult         = 1.0
	role               = "speed"
	minConfToAccept    = 0.5
)

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

type request struct {
	TaskType string                 `json:"task_type"`
	Data     map[string]interface{} `json:"data"`
}

type response struct {
	Ok     bool                   `json:"ok"`
	Result map[string]interface{} `json:"result"`
	TookMs float64                `json:"took_ms"`
	Error  string                 `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
	case json.Number:
		f, _ := n.Float64()
		return f
	}
	return 0
}

func toFloatOr(v interface{}, fallback float64) float64 {
	if v == nil {
		return fallback
	}
	if f := toFloat(v); f != 0 {
		return f
	}
	return fallback
}

func toFloatSlice(v interface{}) []float64 {
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	out := make([]float64, 0, len(arr))
	for _, e := range arr {
		out = append(out, toFloat(e))
	}
	return out
}

func toPairSlice(v interface{}) [][2]float64 {
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	out := make([][2]float64, 0, len(arr))
	for _, e := range arr {
		row, ok := e.([]interface{})
		if ok && len(row) >= 2 {
			out = append(out, [2]float64{toFloat(row[0]), toFloat(row[1])})
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

// sortedKeyString builds a deterministic string from a map for hashing.
func sortedKeyString(data map[string]interface{}) string {
	keys := make([]string, 0, len(data))
	for k := range data {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	buf := make([]byte, 0, 256)
	for _, k := range keys {
		buf = append(buf, k...)
		buf = append(buf, ':')
		switch val := data[k].(type) {
		case string:
			buf = append(buf, val...)
		case float64:
			buf = append(buf, fmt.Sprintf("%g", val)...)
		case json.Number:
			buf = append(buf, val.String()...)
		case bool:
			if val {
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

func fnv1aHash(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}

func sha256Int(s string) uint64 {
	sum := sha256.Sum256([]byte(s))
	var v uint64
	for i := 0; i < 8; i++ {
		v = (v << 8) | uint64(sum[i])
	}
	return v
}

func welfordVol(returns []float64) float64 {
	n := len(returns)
	if n == 0 {
		return 10.0
	}
	var mean, m2 float64
	for i, r := range returns {
		delta := r - mean
		mean += delta / float64(i+1)
		m2 += delta * (r - mean)
	}
	variance := m2 / float64(n)
	if variance <= 0 {
		return 10.0
	}
	return math.Sqrt(variance*252.0) * 1e4
}

func pearson(a, b []float64) float64 {
	n := len(a)
	if n < 2 || len(b) != n {
		return 0
	}
	var sa, sb float64
	for i := 0; i < n; i++ {
		sa += a[i]
		sb += b[i]
	}
	ma, mb := sa/float64(n), sb/float64(n)
	var va, vb, cov float64
	for i := 0; i < n; i++ {
		da, db := a[i]-ma, b[i]-mb
		va += da * da
		vb += db * db
		cov += da * db
	}
	den := math.Sqrt(va * vb)
	if den == 0 {
		return 0
	}
	return clamp(cov/den, -1, 1)
}

// ---------------------------------------------------------------------------
// Task handlers
// ---------------------------------------------------------------------------

// 1. cycle_plan
func handleCyclePlan(data map[string]interface{}) map[string]interface{} {
	// Sort data keys and marshal for SHA256
	keys := make([]string, 0, len(data))
	for k := range data {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	sorted := map[string]interface{}{}
	for _, k := range keys {
		sorted[k] = data[k]
	}
	jsonBytes, _ := json.Marshal(sorted)
	hashInt := sha256Int(string(jsonBytes))
	base := (float64(hashInt%200) - 100.0) / 10000.0

	pv := toFloatOr(data["portfolio_value_aud"], 1.0)
	cash := toFloat(data["cash_balance_aud"])
	signals := int(toFloat(data["signals"]))
	cashRatio := 0.0
	if pv != 0 {
		cashRatio = cash / pv
	}
	tilt := (cashRatio-0.5)*0.002 + float64(signals%3-1)*0.001
	boost := clamp((base+tilt)*cycleScale, -0.015, 0.015)

	return map[string]interface{}{
		"language":    lang,
		"cycle_boost": boost,
		"ok":          true,
	}
}

// 2. order_book_processing
func handleOrderBookProcessing(data map[string]interface{}) map[string]interface{} {
	bids := toPairSlice(data["bids"])
	asks := toPairSlice(data["asks"])

	bestBid, bestAsk := 0.0, 0.0
	if len(bids) > 0 {
		bestBid = bids[0][0]
	}
	if len(asks) > 0 {
		bestAsk = asks[0][0]
	}

	mid := 0.0
	if bestBid > 0 && bestAsk > 0 {
		mid = (bestBid + bestAsk) / 2.0
	}
	spreadBps := 0.0
	if mid > 0 {
		spreadBps = (bestAsk-bestBid)/mid*1e4*spreadMult
	}

	top := 5
	bidVol, askVol := 0.0, 0.0
	for i := 0; i < minInt(top, len(bids)); i++ {
		bidVol += bids[i][1]
	}
	for i := 0; i < minInt(top, len(asks)); i++ {
		askVol += asks[i][1]
	}
	imbalance := 0.0
	if total := bidVol + askVol; total > 0 {
		imbalance = (bidVol - askVol) / total
	}

	return map[string]interface{}{
		"spread_bps": spreadBps,
		"imbalance":  imbalance,
		"mid":        mid,
		"language":   lang,
	}
}

// 3. risk_calculation
func handleRiskCalculation(data map[string]interface{}) map[string]interface{} {
	pv := toFloat(data["position_value"])
	capital := toFloatOr(data["capital"], 1.0)
	ratio := 0.0
	if capital != 0 {
		ratio = pv / capital
	}
	passed := ratio <= riskMax
	return map[string]interface{}{
		"passed":         passed,
		"exposure_ratio": ratio,
		"max_ratio":      riskMax,
		"language":       lang,
	}
}

// 4. volatility_estimate
func handleVolatilityEstimate(data map[string]interface{}) map[string]interface{} {
	returns := toFloatSlice(data["returns"])
	prices := toFloatSlice(data["prices"])
	if prices == nil {
		prices = toFloatSlice(data["ohlcv_close"])
	}

	var vol float64
	if len(returns) > 0 {
		vol = welfordVol(returns)
	} else if len(prices) >= 2 {
		rets := make([]float64, 0, len(prices)-1)
		for i := 1; i < len(prices); i++ {
			if prices[i-1] != 0 {
				rets = append(rets, (prices[i]-prices[i-1])/prices[i-1])
			}
		}
		vol = welfordVol(rets)
	} else {
		vol = 10.0
	}

	return map[string]interface{}{
		"volatility_annual_bps": vol * volWeight,
		"volatility_weight":     volWeight,
		"language":              lang,
		"ok":                    true,
	}
}

// 5. signal_score
func handleSignalScore(data map[string]interface{}) map[string]interface{} {
	h := fnv1aHash(lang + sortedKeyString(data))
	delta := (float64(h%100) - 50.0) / 5000.0 * sigWeight
	return map[string]interface{}{
		"score_delta":         delta,
		"signal_score_weight": sigWeight,
		"language":            lang,
		"ok":                  true,
	}
}

// 6. regime_estimate
func handleRegimeEstimate(data map[string]interface{}) map[string]interface{} {
	prices := toFloatSlice(data["prices"])
	if prices == nil {
		prices = toFloatSlice(data["returns"])
	}

	regime := "mean_revert"
	confidence := 0.5

	if len(prices) >= 3 {
		rets := make([]float64, 0, len(prices)-1)
		for i := 1; i < len(prices); i++ {
			if prices[i-1] != 0 {
				rets = append(rets, (prices[i]-prices[i-1])/prices[i-1])
			}
		}
		vol := welfordVol(rets)
		trend := 0.0
		if prices[0] != 0 {
			trend = (prices[len(prices)-1] - prices[0]) / prices[0]
		}
		if vol > 20.0 {
			regime = "high_vol"
		} else if math.Abs(trend) > 0.02 {
			regime = "trend"
		}
		confidence = math.Min(0.95, 0.5+math.Abs(trend)*5+vol/100)
	}

	return map[string]interface{}{
		"regime":        regime,
		"confidence":    confidence,
		"regime_weight": 1.0,
		"language":      lang,
		"ok":            true,
	}
}

// 7. slippage_estimate
func handleSlippageEstimate(data map[string]interface{}) map[string]interface{} {
	bids := toPairSlice(data["bids"])
	asks := toPairSlice(data["asks"])
	if ob, ok := data["order_book"].(map[string]interface{}); ok {
		if bids == nil {
			bids = toPairSlice(ob["bids"])
		}
		if asks == nil {
			asks = toPairSlice(ob["asks"])
		}
	}
	participation := toFloatOr(data["participation_rate"], 0.01)

	bestBid, bestAsk := 0.0, 0.0
	if len(bids) > 0 {
		bestBid = bids[0][0]
	}
	if len(asks) > 0 {
		bestAsk = asks[0][0]
	}
	mid := 0.0
	if bestBid > 0 && bestAsk > 0 {
		mid = (bestBid + bestAsk) / 2.0
	}

	halfSpreadBps := 5.0
	if mid > 0 {
		halfSpreadBps = (bestAsk-bestBid)/mid*1e4/2.0
	}
	slippageBps := halfSpreadBps * spreadMult * (1.0 + participation*10.0)

	return map[string]interface{}{
		"slippage_bps": slippageBps,
		"language":     lang,
		"ok":           true,
	}
}

// 8. position_sizing
func handlePositionSizing(data map[string]interface{}) map[string]interface{} {
	capital := toFloatOr(data["capital"], 1.0)
	volBps := toFloat(data["volatility_bps"])
	if volBps == 0 {
		volBps = toFloatOr(data["volatility_annual_bps"], 10.0)
	}
	confidence := toFloatOr(data["confidence"], 0.5)
	maxRiskPct := toFloatOr(data["max_risk_pct"], 0.02)

	sizePct := math.Min(riskMax, maxRiskPct*(volBps/10.0)*(0.5+confidence))
	return map[string]interface{}{
		"size_pct": sizePct,
		"size_abs": sizePct * capital,
		"language": lang,
		"ok":       true,
	}
}

// 9. drawdown_check
func handleDrawdownCheck(data map[string]interface{}) map[string]interface{} {
	current := toFloat(data["current_equity"])
	peak := toFloatOr(data["peak_equity"], math.Max(current, 1.0))
	maxDD := toFloatOr(data["max_drawdown_pct"], 0.12)

	dd := 0.0
	if peak > 0 {
		dd = (peak - current) / peak
	}
	passed := dd <= maxDD*riskMax

	return map[string]interface{}{
		"passed":               passed,
		"current_drawdown_pct": dd,
		"language":             lang,
		"ok":                   true,
	}
}

// 10. correlation_estimate
func handleCorrelationEstimate(data map[string]interface{}) map[string]interface{} {
	a := toFloatSlice(data["series_a"])
	if a == nil {
		a = toFloatSlice(data["returns_a"])
	}
	b := toFloatSlice(data["series_b"])
	if b == nil {
		b = toFloatSlice(data["returns_b"])
	}
	return map[string]interface{}{
		"correlation": pearson(a, b),
		"language":    lang,
		"ok":          true,
	}
}

// 11. liquidity_score
func handleLiquidityScore(data map[string]interface{}) map[string]interface{} {
	bids := toPairSlice(data["bids"])
	asks := toPairSlice(data["asks"])

	bestBid, bestAsk := 0.0, 0.0
	if len(bids) > 0 {
		bestBid = bids[0][0]
	}
	if len(asks) > 0 {
		bestAsk = asks[0][0]
	}
	mid := 0.0
	if bestBid > 0 && bestAsk > 0 {
		mid = (bestBid + bestAsk) / 2.0
	}
	depthBps := 100.0
	if mid > 0 {
		depthBps = (bestAsk - bestBid) / mid * 1e4
	}

	totalVol := 0.0
	for i := 0; i < minInt(5, len(bids)); i++ {
		totalVol += bids[i][1]
	}
	for i := 0; i < minInt(5, len(asks)); i++ {
		totalVol += asks[i][1]
	}
	score := math.Min(1.0, totalVol/100.0)

	return map[string]interface{}{
		"liquidity_score": score,
		"depth_bps":       depthBps,
		"language":        lang,
		"ok":              true,
	}
}

// 12. market_impact
func handleMarketImpact(data map[string]interface{}) map[string]interface{} {
	quantity := toFloat(data["quantity"])
	adv := toFloatOr(data["adv"], 1.0)
	volatility := toFloatOr(data["volatility"], 0.01)
	participation := 0.0
	if adv > 0 {
		participation = quantity / adv
	}
	impactBps := 10.0 * math.Sqrt(participation) * volatility * 1e4

	return map[string]interface{}{
		"impact_bps": impactBps,
		"language":   lang,
		"ok":         true,
	}
}

// 13. signal_filter
func handleSignalFilter(data map[string]interface{}) map[string]interface{} {
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

	accept := confidence >= minConfToAccept && !(regime == "high_vol" && volatility >= 0.02)
	reason := ""
	if !accept {
		reason = "low_confidence_or_regime"
	}

	return map[string]interface{}{
		"accept":        accept,
		"filter_reason": reason,
		"language":      lang,
		"ok":            true,
	}
}

// 14. confidence_calibration
func handleConfidenceCalibration(data map[string]interface{}) map[string]interface{} {
	confs := toFloatSlice(data["historical_confidences"])
	pnls := toFloatSlice(data["historical_pnl"])

	if len(confs) == 0 || len(confs) != len(pnls) {
		return map[string]interface{}{"calibrated_confidence": 0.5, "language": lang, "ok": true}
	}
	wins, sumConf := 0, 0.0
	for i, p := range pnls {
		if p > 0 {
			wins++
		}
		sumConf += confs[i]
	}
	avgConf := sumConf / float64(len(confs))
	winRate := float64(wins) / float64(len(pnls))
	calibrated := clamp(avgConf*0.5+winRate*0.5, 0, 1)

	return map[string]interface{}{
		"calibrated_confidence": calibrated,
		"language":              lang,
		"ok":                    true,
	}
}

// 15. heartbeat
func handleHeartbeat(data map[string]interface{}) map[string]interface{} {
	cycleID := data["cycle_id"]
	if cycleID == nil {
		cycleID = 0
	}
	return map[string]interface{}{
		"ok":         true,
		"latency_ms": 0.0,
		"language":   lang,
		"cycle_id":   cycleID,
	}
}

// 16. var_estimate
func handleVarEstimate(data map[string]interface{}) map[string]interface{} {
	returns := toFloatSlice(data["returns"])
	confidenceLevel := toFloatOr(data["confidence_level"], 0.95)

	if len(returns) < 5 {
		return map[string]interface{}{"var_pct": 0.0, "cvar_pct": 0.0, "language": lang, "ok": true}
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

	sum := 0.0
	cnt := idx + 1
	for i := 0; i < cnt; i++ {
		sum += sorted[i]
	}
	cvarPct := -sum / float64(cnt) * 100.0

	return map[string]interface{}{
		"var_pct":  varPct,
		"cvar_pct": cvarPct,
		"language": lang,
		"ok":       true,
	}
}

// 17. skew_estimate
func handleSkewEstimate(data map[string]interface{}) map[string]interface{} {
	returns := toFloatSlice(data["returns"])
	if len(returns) < 3 {
		return map[string]interface{}{"skew": 0.0, "language": lang, "ok": true}
	}
	n := float64(len(returns))
	var mean, m2 float64
	for i, r := range returns {
		delta := r - mean
		mean += delta / float64(i+1)
		m2 += delta * (r - mean)
	}
	variance := m2 / n
	std := math.Sqrt(variance)
	if std == 0 {
		return map[string]interface{}{"skew": 0.0, "language": lang, "ok": true}
	}
	m3 := 0.0
	for _, r := range returns {
		d := (r - mean) / std
		m3 += d * d * d
	}
	skew := m3 / n

	return map[string]interface{}{
		"skew":     skew,
		"language": lang,
		"ok":       true,
	}
}

// 18. order_book_imbalance_series
func handleOrderBookImbalanceSeries(data map[string]interface{}) map[string]interface{} {
	bids := toPairSlice(data["bids"])
	asks := toPairSlice(data["asks"])

	bidVol, askVol := 0.0, 0.0
	for i := 0; i < minInt(5, len(bids)); i++ {
		bidVol += bids[i][1]
	}
	for i := 0; i < minInt(5, len(asks)); i++ {
		askVol += asks[i][1]
	}
	imb := 0.0
	if total := bidVol + askVol; total > 0 {
		imb = (bidVol - askVol) / total
	}

	return map[string]interface{}{
		"imbalance_series": []interface{}{imb},
		"trend":            imb,
		"language":         lang,
		"ok":               true,
	}
}

// 19. execution_quality_score
func handleExecutionQualityScore(data map[string]interface{}) map[string]interface{} {
	fillsRaw, _ := data["fills"].([]interface{})
	decisionsRaw, _ := data["decision_prices"].([]interface{})

	if len(fillsRaw) == 0 || len(fillsRaw) != len(decisionsRaw) {
		return map[string]interface{}{"score_0_1": 1.0, "avg_slippage_bps": 0.0, "language": lang, "ok": true}
	}

	limit := minInt(len(fillsRaw), len(decisionsRaw))
	totalSlippage, count := 0.0, 0
	for i := 0; i < limit; i++ {
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
	avgBps := 0.0
	if count > 0 {
		avgBps = totalSlippage / float64(count)
	}
	score := clamp(1.0-avgBps/50.0, 0, 1)

	return map[string]interface{}{
		"score_0_1":        score,
		"avg_slippage_bps": avgBps,
		"language":         lang,
		"ok":               true,
	}
}

// 20. regime_duration
func handleRegimeDuration(data map[string]interface{}) map[string]interface{} {
	prices := toFloatSlice(data["prices"])
	regimeHistory, _ := data["regime_history"].([]interface{})

	regime := "mean_revert"
	if len(prices) >= 2 {
		rets := make([]float64, 0, len(prices)-1)
		for i := 1; i < len(prices); i++ {
			if prices[i-1] != 0 {
				rets = append(rets, (prices[i]-prices[i-1])/prices[i-1])
			}
		}
		vol := welfordVol(rets)
		if vol > 20.0 {
			regime = "high_vol"
		}
	}

	bars := len(regimeHistory)
	if bars == 0 {
		bars = minInt(10, len(prices))
	}

	return map[string]interface{}{
		"bars_in_regime": bars,
		"regime_stable":  bars >= 5,
		"regime":         regime,
		"language":       lang,
		"ok":             true,
	}
}

// 21. enhanced_orderbook — deep order book analysis
func handleEnhancedOrderbook(data map[string]interface{}) map[string]interface{} {
	bids := toPairSlice(data["bids"])
	asks := toPairSlice(data["asks"])

	// Cumulative depth curves (running sum of volume at each price level)
	depthCurveBids := make([]interface{}, len(bids))
	cumBid := 0.0
	for i, level := range bids {
		cumBid += level[1]
		depthCurveBids[i] = cumBid
	}

	depthCurveAsks := make([]interface{}, len(asks))
	cumAsk := 0.0
	for i, level := range asks {
		cumAsk += level[1]
		depthCurveAsks[i] = cumAsk
	}

	// Volume-weighted mid price
	bidPriceVol, bidVolSum := 0.0, 0.0
	for _, level := range bids {
		bidPriceVol += level[0] * level[1]
		bidVolSum += level[1]
	}
	askPriceVol, askVolSum := 0.0, 0.0
	for _, level := range asks {
		askPriceVol += level[0] * level[1]
		askVolSum += level[1]
	}
	vwapMid := 0.0
	if bidVolSum > 0 && askVolSum > 0 {
		vwapBid := bidPriceVol / bidVolSum
		vwapAsk := askPriceVol / askVolSum
		// Weight by opposite-side volume for a true VWAP mid
		totalVol := bidVolSum + askVolSum
		vwapMid = (vwapBid*askVolSum + vwapAsk*bidVolSum) / totalVol
	}

	// Resilience score: depth at level 5 vs level 1 (how quickly depth rebuilds)
	// Higher ratio means more depth beyond the top of book => more resilient
	resilienceScore := 0.0
	if len(bids) >= 5 && len(asks) >= 5 {
		depthLevel1 := bids[0][1] + asks[0][1]
		depthLevel5 := bids[4][1] + asks[4][1]
		if depthLevel1 > 0 {
			resilienceScore = depthLevel5 / depthLevel1
		}
	} else if len(bids) > 0 && len(asks) > 0 {
		resilienceScore = 1.0 // not enough levels to judge
	}

	// Absorption ratio: ratio of top-3 volume to total volume
	top3Vol := 0.0
	for i := 0; i < minInt(3, len(bids)); i++ {
		top3Vol += bids[i][1]
	}
	for i := 0; i < minInt(3, len(asks)); i++ {
		top3Vol += asks[i][1]
	}
	totalVol := bidVolSum + askVolSum
	absorptionRatio := 0.0
	if totalVol > 0 {
		absorptionRatio = top3Vol / totalVol
	}

	return map[string]interface{}{
		"vwap_mid":         vwapMid,
		"depth_curve_bids": depthCurveBids,
		"depth_curve_asks": depthCurveAsks,
		"resilience_score": resilienceScore,
		"absorption_ratio": absorptionRatio,
		"language":         lang,
		"ok":               true,
	}
}

// 22. trade_flow_analysis — analyze recent trade flow for patterns
func handleTradeFlowAnalysis(data map[string]interface{}) map[string]interface{} {
	tradesRaw, _ := data["trades"].([]interface{})

	if len(tradesRaw) == 0 {
		return map[string]interface{}{
			"net_buy_pressure":   0.0,
			"arrival_rate_per_sec": 0.0,
			"trade_vwap":        0.0,
			"aggressiveness":    0.0,
			"volume_clustering": 0.0,
			"language":          lang,
			"ok":                true,
		}
	}

	type trade struct {
		price     float64
		volume    float64
		side      string
		timestamp float64
	}

	trades := make([]trade, 0, len(tradesRaw))
	for _, raw := range tradesRaw {
		t, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		side := ""
		if s, ok := t["side"].(string); ok {
			side = s
		}
		trades = append(trades, trade{
			price:     toFloat(t["price"]),
			volume:    toFloat(t["volume"]),
			side:      side,
			timestamp: toFloat(t["timestamp"]),
		})
	}

	if len(trades) == 0 {
		return map[string]interface{}{
			"net_buy_pressure":   0.0,
			"arrival_rate_per_sec": 0.0,
			"trade_vwap":        0.0,
			"aggressiveness":    0.0,
			"volume_clustering": 0.0,
			"language":          lang,
			"ok":                true,
		}
	}

	// Net buy pressure: (buy_volume - sell_volume) / total_volume
	buyVol, sellVol, totalVol := 0.0, 0.0, 0.0
	for _, t := range trades {
		totalVol += t.volume
		if t.side == "buy" {
			buyVol += t.volume
		} else {
			sellVol += t.volume
		}
	}
	netBuyPressure := 0.0
	if totalVol > 0 {
		netBuyPressure = (buyVol - sellVol) / totalVol
	}

	// VWAP of recent trades
	priceVolSum := 0.0
	for _, t := range trades {
		priceVolSum += t.price * t.volume
	}
	tradeVWAP := 0.0
	if totalVol > 0 {
		tradeVWAP = priceVolSum / totalVol
	}

	// Trade arrival rate: trades per second over the time window
	arrivalRate := 0.0
	if len(trades) >= 2 {
		minTS, maxTS := trades[0].timestamp, trades[0].timestamp
		for _, t := range trades[1:] {
			if t.timestamp < minTS {
				minTS = t.timestamp
			}
			if t.timestamp > maxTS {
				maxTS = t.timestamp
			}
		}
		duration := maxTS - minTS
		if duration > 0 {
			arrivalRate = float64(len(trades)) / duration
		}
	}

	// Aggressiveness: ratio of trades hitting the ask (buy side) vs total
	buyCount := 0
	for _, t := range trades {
		if t.side == "buy" {
			buyCount++
		}
	}
	aggressiveness := float64(buyCount) / float64(len(trades))

	// Volume clustering: std deviation of inter-trade times
	// High std means volume is bunched; low std means evenly spaced
	volumeClustering := 0.0
	if len(trades) >= 3 {
		interTimes := make([]float64, 0, len(trades)-1)
		for i := 1; i < len(trades); i++ {
			dt := trades[i].timestamp - trades[i-1].timestamp
			if dt < 0 {
				dt = -dt
			}
			interTimes = append(interTimes, dt)
		}
		if len(interTimes) > 0 {
			// Compute mean of inter-trade times
			sum := 0.0
			for _, dt := range interTimes {
				sum += dt
			}
			meanDT := sum / float64(len(interTimes))
			// Compute std of inter-trade times
			var m2 float64
			for _, dt := range interTimes {
				d := dt - meanDT
				m2 += d * d
			}
			if meanDT > 0 {
				// Coefficient of variation (normalized clustering)
				volumeClustering = math.Sqrt(m2/float64(len(interTimes))) / meanDT
			}
		}
	}

	return map[string]interface{}{
		"net_buy_pressure":   netBuyPressure,
		"arrival_rate_per_sec": arrivalRate,
		"trade_vwap":        tradeVWAP,
		"aggressiveness":    aggressiveness,
		"volume_clustering": volumeClustering,
		"language":          lang,
		"ok":                true,
	}
}

// 23. latency_benchmark — benchmark and report processing latency
func handleLatencyBenchmark(data map[string]interface{}) map[string]interface{} {
	iterations := int(toFloatOr(data["iterations"], 1000))
	if iterations <= 0 {
		iterations = 1000
	}
	if iterations > 1000000 {
		iterations = 1000000 // cap to prevent excessive runtime
	}

	payloadSize := int(toFloatOr(data["payload_size"], 1000))
	if payloadSize <= 0 {
		payloadSize = 1000
	}
	if payloadSize > 100000 {
		payloadSize = 100000
	}

	// CPU-bound benchmark: sort random arrays of payloadSize elements
	rng := rand.New(rand.NewSource(42))
	arr := make([]float64, payloadSize)

	start := time.Now()
	for iter := 0; iter < iterations; iter++ {
		// Fill with pseudo-random data
		for i := range arr {
			arr[i] = rng.Float64()
		}
		sort.Float64s(arr)
	}
	elapsed := time.Since(start)

	latencyNsPerOp := float64(elapsed.Nanoseconds()) / float64(iterations)
	opsPerMs := 0.0
	elapsedMs := float64(elapsed.Nanoseconds()) / 1e6
	if elapsedMs > 0 {
		opsPerMs = float64(iterations) / elapsedMs
	}

	return map[string]interface{}{
		"ops_per_ms":       opsPerMs,
		"latency_ns_per_op": latencyNsPerOp,
		"iterations":       iterations,
		"payload_size":     payloadSize,
		"elapsed_ms":       elapsedMs,
		"language":         lang,
		"ok":               true,
	}
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

var dispatch = map[string]func(map[string]interface{}) map[string]interface{}{
	"cycle_plan":                  handleCyclePlan,
	"order_book_processing":       handleOrderBookProcessing,
	"risk_calculation":            handleRiskCalculation,
	"volatility_estimate":         handleVolatilityEstimate,
	"signal_score":                handleSignalScore,
	"regime_estimate":             handleRegimeEstimate,
	"slippage_estimate":           handleSlippageEstimate,
	"position_sizing":             handlePositionSizing,
	"drawdown_check":              handleDrawdownCheck,
	"correlation_estimate":        handleCorrelationEstimate,
	"liquidity_score":             handleLiquidityScore,
	"market_impact":               handleMarketImpact,
	"signal_filter":               handleSignalFilter,
	"confidence_calibration":      handleConfidenceCalibration,
	"heartbeat":                   handleHeartbeat,
	"var_estimate":                handleVarEstimate,
	"skew_estimate":               handleSkewEstimate,
	"order_book_imbalance_series": handleOrderBookImbalanceSeries,
	"execution_quality_score":     handleExecutionQualityScore,
	"regime_duration":             handleRegimeDuration,
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	writer := bufio.NewWriter(os.Stdout)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		start := time.Now()
		var resp response

		var req request
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			resp = response{Ok: false, Result: map[string]interface{}{}, TookMs: 0, Error: "json parse error: " + err.Error()}
		} else {
			handler, ok := dispatch[req.TaskType]
			if !ok {
				resp = response{Ok: false, Result: map[string]interface{}{}, TookMs: 0, Error: "unknown task_type: " + req.TaskType}
			} else {
				func() {
					defer func() {
						if r := recover(); r != nil {
							resp = response{Ok: false, Result: map[string]interface{}{}, TookMs: float64(time.Since(start).Microseconds()) / 1000.0, Error: fmt.Sprintf("panic: %v", r)}
						}
					}()
					data := req.Data
					if data == nil {
						data = map[string]interface{}{}
					}
					result := handler(data)
					tookMs := float64(time.Since(start).Microseconds()) / 1000.0
					resp = response{Ok: true, Result: result, TookMs: tookMs}
				}()
			}
		}

		if resp.TookMs == 0 {
			resp.TookMs = float64(time.Since(start).Microseconds()) / 1000.0
		}

		out, _ := json.Marshal(resp)
		writer.Write(out)
		writer.WriteByte('\n')
		writer.Flush()
	}
}
