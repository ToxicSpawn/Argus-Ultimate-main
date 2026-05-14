// Argus Ultimate -- Kotlin multilang worker
// Profile: language=kotlin, risk_max=0.45, cycle_scale=0.99,
//          vol_w=1.0, sig_w=1.0, spread=1.01, role=correctness
//
// Compile:  kotlinc ArgusWorker.kt -include-runtime -d argus_worker.jar
// Run:      java -jar argus_worker.jar
// Protocol:
//   stdin  -> {"task_type": "...", "data": {...}}
//   stdout <- {"ok": true, "result": {...}, "took_ms": 0.12}
//
// Uses Kotlin stdlib only (no external deps).

import java.security.MessageDigest
import kotlin.math.*

// =================================================================
// Profile
// =================================================================

const val RISK_MAX    = 0.45
const val CYCLE_SCALE = 0.99
const val VOL_W       = 1.0
const val SIG_W       = 1.0
const val SPREAD      = 1.01

// =================================================================
// JSON extraction helpers (regex-based, stdlib only)
// =================================================================

fun extractString(json: String, key: String): String? {
    val regex = Regex(""""$key"\s*:\s*"([^"]*)"""")
    return regex.find(json)?.groupValues?.get(1)
}

fun extractDouble(json: String, key: String): Double? {
    val regex = Regex(""""$key"\s*:\s*(-?[\d.eE+\-]+)""")
    return regex.find(json)?.groupValues?.get(1)?.toDoubleOrNull()
}

fun extractList(json: String, key: String): List<Double> {
    val regex = Regex(""""$key"\s*:\s*\[([^\]]*)\]""")
    val content = regex.find(json)?.groupValues?.get(1) ?: return emptyList()
    return content.split(",").mapNotNull { it.trim().toDoubleOrNull() }
}

fun gd(json: String, key: String, default: Double): Double =
    extractDouble(json, key) ?: default

// =================================================================
// Math helpers
// =================================================================

fun logReturns(prices: List<Double>): List<Double> {
    if (prices.size < 2) return emptyList()
    return prices.zipWithNext().map { (p1, p2) ->
        if (p1 == 0.0) 0.0 else ln(p2 / p1)
    }
}

fun mean(xs: List<Double>): Double =
    if (xs.isEmpty()) 0.0 else xs.sum() / xs.size

fun variance(xs: List<Double>): Double {
    if (xs.size < 2) return 0.0
    val m = mean(xs)
    return xs.sumOf { (it - m).pow(2) } / xs.size
}

fun stdDev(xs: List<Double>): Double = sqrt(variance(xs))

fun computeVol(prices: List<Double>): Double {
    if (prices.size < 2) return 0.0
    return stdDev(logReturns(prices))
}

fun computeSkew(xs: List<Double>): Double {
    if (xs.size < 2) return 0.0
    val m  = mean(xs)
    val sd = stdDev(xs)
    if (sd == 0.0) return 0.0
    return xs.sumOf { ((it - m) / sd).pow(3) } / xs.size
}

fun cycleHash(json: String): Long {
    val regex = Regex(""""([^"]+)"\s*:\s*([^,}]+)""")
    val pairs = regex.findAll(json)
        .map { it.groupValues[1] + ":" + it.groupValues[2].trim() }
        .toList().sorted().joinToString(",")
    val bytes = MessageDigest.getInstance("MD5").digest(pairs.toByteArray(Charsets.UTF_8))
    return (bytes[0].toLong() and 0xFF shl 24) or
           (bytes[1].toLong() and 0xFF shl 16) or
           (bytes[2].toLong() and 0xFF shl  8) or
           (bytes[3].toLong() and 0xFF)
}

fun pickAction(h: Long): String = when (h % 3) {
    0L -> "buy"
    1L -> "sell"
    else -> "hold"
}

// Escape a string value for inline JSON
fun jstr(s: String): String = s
    .replace("\\\\", "\\\\\\\\")
    .replace("\"", "\\\\\"")
    .replace("\n", "\\n")
    .replace("\r", "\\r")
    .replace("\t", "\\t")

// =================================================================
// Dispatch -- returns inner JSON content as a String
// =================================================================

fun dispatch(json: String): String {
    val taskType = extractString(json, "task_type")
        ?: return "language\":\"kotlin\",\"error\":\"missing task_type\""
    val dataMatch = Regex(""""data"\s*:\s*\{([^}]*)\}""").find(json)
    val data = dataMatch?.value ?: json

    return when (taskType) {

        "heartbeat" -> {
            val ts = System.currentTimeMillis()
            """language":"kotlin","ts":$ts"""
        }

        "cycle_plan" -> {
            val h  = cycleHash(data)
            val ac = pickAction(h)
            val cf = 0.5 + (h % 50) / 100.0
            val sz = CYCLE_SCALE * (0.1 + (h % 10) / 100.0)
            """language":"kotlin","cycle_boost":$h,"action":"$ac","confidence":$cf,"size":$sz"""
        }

        "volatility_estimate" -> {
            val prices = extractList(data, "prices")
            val vol    = computeVol(prices) * VOL_W
            """language":"kotlin","volatility_annual_bps":$vol,"volatility_weight":$VOL_W"""
        }

        "signal_score" -> {
            val hash  = cycleHash(data)
            val delta = ((hash % 100) - 50) / 5000.0 * SIG_W
            """language":"kotlin","score_delta":$delta,"signal_score_weight":$SIG_W"""
        }

        "risk_calculation" -> {
            val pv  = gd(data, "position_value", 0.0)
            val cap = gd(data, "capital",        1.0)
            val c2  = if (cap == 0.0) 1.0 else cap
            val rr  = pv / c2
            val wl  = rr <= RISK_MAX
            """language":"kotlin","passed":$wl,"exposure_ratio":$rr,"max_ratio":$RISK_MAX"""
        }

        "position_sizing" -> {
            val cap = gd(data, "capital",   10000.0)
            val rp  = gd(data, "risk_pct",  0.01)
            val sd  = gd(data, "stop_dist", 1.0)
            val sd2 = if (sd == 0.0) 1.0 else sd
            val sz  = cap * rp / sd2
            val c2  = if (cap == 0.0) 1.0 else cap
            val pct = sz / c2
            """language":"kotlin","size_pct":$pct,"size_abs":$sz"""
        }

        "drawdown_check" -> {
            val pk  = gd(data, "peak",    1.0)
            val cu  = gd(data, "current", 1.0)
            val p2  = if (pk == 0.0) 1.0 else pk
            val dd  = (p2 - cu) / p2
            val passed = dd <= 0.20
            """language":"kotlin","passed":$passed,"current_drawdown_pct":$dd"""
        }

        "var_estimate" -> {
            val prices = extractList(data, "prices")
            val vol    = computeVol(prices)
            val v95    = vol * 1.645
            val cvar   = v95 * 1.2
            """language":"kotlin","var_pct":$v95,"cvar_pct":$cvar"""
        }

        "skew_estimate" -> {
            val prices = extractList(data, "prices")
            val skew   = computeSkew(prices)
            """language":"kotlin","skew":$skew"""
        }

        "order_book_imbalance_series" -> {
            val b   = gd(data, "bid_volume", 100.0)
            val a   = gd(data, "ask_volume", 100.0)
            val d   = b + a
            val imb = if (d == 0.0) 0.0 else (b - a) / d
            """language":"kotlin","imbalance_series":[$imb],"trend":$imb"""
        }

        "order_book_processing" -> {
            val b   = gd(data, "bid", 0.0)
            val a   = gd(data, "ask", 0.0)
            val sp  = if (a > 0.0) (a - b) / a * SPREAD else 0.0
            val d   = b + a
            val imb = if (d == 0.0) 0.0 else (b - a) / d
            val mid = (b + a) / 2.0
            """language":"kotlin","spread_bps":$sp,"imbalance":$imb,"mid":$mid"""
        }

        "regime_estimate" -> {
            val prices = extractList(data, "prices")
            val vol    = computeVol(prices)
            val regime = if (vol > 0.02) "high_vol" else "low_vol"
            val conf   = 0.5 + vol * 10.0
            """language":"kotlin","regime":"$regime","confidence":$conf,"regime_weight":1.0"""
        }

        "slippage_estimate" -> {
            val sz   = gd(data, "size",   1.0)
            val sp   = gd(data, "spread", 0.01)
            val slip = sz * sp * SPREAD
            """language":"kotlin","slippage_bps":$slip"""
        }

        "correlation_estimate" ->
            """language":"kotlin","correlation":0.0,"value":0.5"""

        "liquidity_score" -> {
            val vol   = gd(data, "volume", 1000.0)
            val score = minOf(1.0, vol / 10000.0)
            """language":"kotlin","liquidity_score":$score,"depth_bps":100"""
        }

        "market_impact" -> {
            val sz     = gd(data, "size",      1.0)
            val liq    = gd(data, "liquidity", 1000.0)
            val l2     = if (liq == 0.0) 1.0 else liq
            val impact = sz / l2
            """language":"kotlin","impact_bps":$impact"""
        }

        "signal_filter" -> {
            val sig    = gd(data, "signal",    0.0)
            val thr    = gd(data, "threshold", 0.1)
            val accept = abs(sig) >= thr
            """language":"kotlin","accept":$accept,"filter_reason":"","signal":$sig"""
        }

        "confidence_calibration" -> {
            val raw = gd(data, "confidence", 0.5)
            val cal = maxOf(0.0, minOf(1.0, raw))
            """language":"kotlin","calibrated_confidence":$cal"""
        }

        "execution_quality_score" -> {
            val slip  = gd(data, "slippage", 0.0)
            val score = maxOf(0.0, 1.0 - abs(slip) * 10.0)
            """language":"kotlin","score_0_1":$score,"avg_slippage_bps":0"""
        }

        "regime_duration" -> {
            val start   = gd(data, "start_ts", 0.0)
            val now     = System.currentTimeMillis().toDouble()
            val durMs   = now - start
            val bars    = (durMs / 60000.0).toInt()
            val stable  = bars > 5
            """language":"kotlin","bars_in_regime":$bars,"regime_stable":$stable,"regime":"unknown\""""
        }

        else -> {
            val escaped = jstr(taskType)
            """language":"kotlin","task":"$escaped","value":0.5"""
        }
    }
}

// =================================================================
// Main loop
// =================================================================

fun main() {
    val reader = System.`in`.bufferedReader()
    val writer = System.out.bufferedWriter()
    var line = reader.readLine()
    while (line != null) {
        line = line.trim()
        if (line.isNotEmpty()) {
            val t0     = System.nanoTime()
            val inner  = dispatch(line)
            val tookMs = (System.nanoTime() - t0) / 1_000_000.0
            writer.write("{\"ok\":true,\"result\":{\"$inner},\"took_ms\":$tookMs}")
            writer.newLine()
            writer.flush()
        }
        line = reader.readLine()
    }
}
