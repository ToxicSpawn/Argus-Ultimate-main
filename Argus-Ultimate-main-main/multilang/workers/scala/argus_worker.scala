// argus_worker.scala — Argus Scala stdin/stdout worker
// Run: scala argus_worker.scala  OR  scalac argus_worker.scala && scala ArgusWorker
// Protocol: one JSON line in -> one JSON line out, loop until EOF

import java.security.MessageDigest
import scala.io.StdIn
import scala.util.Try
import java.util.Locale

object ArgusWorker {
  val LANGUAGE = "scala"
  val RISK_MAX = 0.44
  val CYCLE_SCALE = 0.98
  val VOL_W = 1.0
  val SIG_W = 1.02
  val SPREAD = 1.01

  def main(args: Array[String]): Unit = {
    var line = StdIn.readLine()
    while (line != null) {
      val trimmed = line.trim
      if (trimmed.nonEmpty) {
        val t0 = System.nanoTime()
        try {
          val parsed = parseJson(trimmed)
          val taskType = getStr(parsed, "task_type", "heartbeat")
          val data = parsed.getOrElse("data", Map.empty[String, Any]) match {
            case m: Map[_, _] => m.asInstanceOf[Map[String, Any]]
            case _ => Map.empty[String, Any]
          }
          val result = dispatch(taskType, data)
          val tookMs = (System.nanoTime() - t0) / 1e6
          println(s"""{"ok":true,"result":${toJson(result)},"took_ms":${fmtD(tookMs)}}""")
        } catch {
          case e: Exception =>
            val tookMs = (System.nanoTime() - t0) / 1e6
            println(s"""{"ok":false,"error":"${esc(e.getMessage)}","took_ms":${fmtD(tookMs)}}""")
        }
        Console.flush()
      }
      line = StdIn.readLine()
    }
  }

  def dispatch(tt: String, d: Map[String, Any]): Map[String, Any] = tt match {
    case "cycle_plan"        => cyclePlan(d)
    case "heartbeat"         => heartbeat(d)
    case "volatility_estimate" => volatilityEstimate(d)
    case "signal_score"      => signalScore(d)
    case "risk_calculation"  => riskCalc(d)
    case "position_sizing"   => positionSizing(d)
    case "drawdown_check"    => drawdownCheck(d)
    case "regime_estimate"   => regimeEstimate(d)
    case "slippage_estimate" => slippageEstimate(d)
    case "correlation_estimate" => correlationEstimate(d)
    case "liquidity_score"   => liquidityScore(d)
    case "market_impact"     => marketImpact(d)
    case "order_book_processing" => orderBookProcessing(d)
    case "signal_filter"     => signalFilter(d)
    case "confidence_calibration" => confidenceCalibration(d)
    case "var_estimate"      => varEstimate(d)
    case "skew_estimate"     => skewEstimate(d)
    case "order_book_imbalance_series" => obImbalanceSeries(d)
    case "execution_quality_score" => execQuality(d)
    case "regime_duration"   => regimeDuration(d)
    case other => Map("language" -> LANGUAGE, "task_type" -> other, "value" -> 0.5)
  }

  // ── Tasks ──────────────────────────────────────────────────────────────

  def cyclePlan(d: Map[String, Any]): Map[String, Any] = {
    val sorted = d.toSeq.sortBy(_._1).map { case (k, v) => s"$k:$v" }.mkString(",")
    val h = sha256Long(sorted)
    val base = ((h % 200) - 100) / 10000.0
    val capital = getDbl(d, "capital", getDbl(d, "portfolio_value_aud", 1000.0))
    val actions = Array("buy", "sell", "hold")
    val action = actions((h % 3).toInt)
    val confidence = 0.5 + base
    val cycleBoost = base * CYCLE_SCALE
    val sizeAud = capital * 0.02 * CYCLE_SCALE * math.min(1.0, math.abs(base) * 50)
    Map("action" -> action, "cycle_boost" -> cycleBoost, "confidence" -> confidence, "size_aud" -> math.max(0, sizeAud),
      "spread_multiplier" -> SPREAD, "language" -> LANGUAGE)
  }

  def heartbeat(d: Map[String, Any]): Map[String, Any] =
    Map("ok" -> true, "language" -> LANGUAGE, "ts" -> System.currentTimeMillis())

  def volatilityEstimate(d: Map[String, Any]): Map[String, Any] = {
    val prices = getDblList(d, "prices")
    val vol = if (prices.length < 2) 1000.0 else {
      val lr = prices.sliding(2).map { case Seq(a, b) => math.log(b / a) }.toArray
      val mean = lr.sum / lr.length
      val v = lr.map(x => (x - mean) * (x - mean)).sum / math.max(1, lr.length - 1)
      math.sqrt(v) * math.sqrt(365) * 10000 * VOL_W
    }
    Map("volatility_annual_bps" -> vol, "volatility_weight" -> VOL_W, "n" -> prices.length, "language" -> LANGUAGE)
  }

  def signalScore(d: Map[String, Any]): Map[String, Any] = {
    val sig = getDbl(d, "signal", 0.5); val conf = getDbl(d, "confidence", 0.5)
    val regime = getStr(d, "regime", "unknown")
    val sorted = d.toSeq.sortBy(_._1).map { case (k, v) => s"$k:$v" }.mkString(",")
    val h = sha256Long(sorted)
    val scoreDelta = ((h % 100) - 50) / 5000.0 * SIG_W
    Map("score_delta" -> scoreDelta, "confidence" -> conf, "regime" -> regime,
      "signal_score_weight" -> SIG_W, "language" -> LANGUAGE)
  }

  def riskCalc(d: Map[String, Any]): Map[String, Any] = {
    val pv = getDbl(d, "position_value", 0); val cap = getDbl(d, "capital", 1000)
    val ratio = if (cap > 0) pv / cap else 0.0
    Map("exposure_ratio" -> ratio, "passed" -> (ratio <= RISK_MAX), "max_ratio" -> RISK_MAX, "language" -> LANGUAGE)
  }

  def positionSizing(d: Map[String, Any]): Map[String, Any] = {
    val cap = getDbl(d, "capital", 1000); val vol = getDbl(d, "volatility_bps", 1000)
    val conf = getDbl(d, "confidence", 0.5)
    val size = math.max(10, math.min(cap * RISK_MAX, cap * 0.02 * (1 - vol / 20000.0) * conf))
    val sizePct = if (cap > 0) size / cap * 100.0 else 0.0
    Map("size_pct" -> sizePct, "size_abs" -> size, "language" -> LANGUAGE)
  }

  def drawdownCheck(d: Map[String, Any]): Map[String, Any] = {
    val cur = getDbl(d, "current_equity", 1000); val peak = getDbl(d, "peak_equity", 1000)
    val maxDd = getDbl(d, "max_drawdown_pct", 20)
    val dd = if (peak > 0) (peak - cur) / peak else 0.0
    Map("current_drawdown_pct" -> dd * 100, "passed" -> (dd * 100 <= maxDd), "language" -> LANGUAGE)
  }

  def regimeEstimate(d: Map[String, Any]): Map[String, Any] = {
    val prices = getDblList(d, "prices")
    if (prices.length < 5) Map("regime" -> "unknown", "confidence" -> 0.5, "regime_weight" -> 1.0, "language" -> LANGUAGE)
    else {
      val change = (prices.last - prices.head) / prices.head
      val (regime, conf) =
        if (change > 0.02) ("trending_up", 0.6 + math.min(0.3, change))
        else if (change < -0.02) ("trending_down", 0.6 + math.min(0.3, math.abs(change)))
        else ("mean_reverting", 0.55)
      Map("regime" -> regime, "confidence" -> conf, "regime_weight" -> 1.0, "language" -> LANGUAGE)
    }
  }

  def slippageEstimate(d: Map[String, Any]): Map[String, Any] = {
    val spread = getDbl(d, "spread_bps", 10); val vol = getDbl(d, "volatility_bps", 1000)
    val size = getDbl(d, "order_size_aud", 100)
    Map("slippage_bps" -> (spread * 0.5 + vol * 0.001 + size * 0.0001) * SPREAD, "language" -> LANGUAGE)
  }

  def correlationEstimate(d: Map[String, Any]): Map[String, Any] = {
    val xs = getDblList(d, "series_a"); val ys = getDblList(d, "series_b")
    val n = math.min(xs.length, ys.length)
    val corr = if (n < 3) 0.0 else {
      val mx = xs.take(n).sum / n; val my = ys.take(n).sum / n
      var cov = 0.0; var sx = 0.0; var sy = 0.0
      for (i <- 0 until n) {
        val dx = xs(i) - mx; val dy = ys(i) - my; cov += dx * dy; sx += dx * dx; sy += dy * dy
      }
      val denom = math.sqrt(sx * sy); if (denom > 0) cov / denom else 0.0
    }
    Map("correlation" -> corr, "n" -> n, "language" -> LANGUAGE)
  }

  def liquidityScore(d: Map[String, Any]): Map[String, Any] = {
    val vol24h = getDbl(d, "volume_24h", 1e6); val spread = getDbl(d, "spread_bps", 10)
    val depth = getDbl(d, "order_book_depth", 100)
    val score = math.min(1.0, (vol24h / 1e7) * 0.4 + (1.0 / (1 + spread / 100)) * 0.3 + (depth / 1000) * 0.3)
    val depthBps = depth * spread / 100.0
    Map("liquidity_score" -> score, "depth_bps" -> depthBps, "language" -> LANGUAGE)
  }

  def marketImpact(d: Map[String, Any]): Map[String, Any] = {
    val size = getDbl(d, "order_size_aud", 100); val vol = getDbl(d, "volume_24h", 1e6)
    val impact = if (vol > 0) math.sqrt(size / vol) * 100 else 10.0
    Map("impact_bps" -> impact, "language" -> LANGUAGE)
  }

  def orderBookProcessing(d: Map[String, Any]): Map[String, Any] = {
    val bids = getDblList(d, "bids"); val asks = getDblList(d, "asks")
    val bb = bids.headOption.getOrElse(0.0); val ba = asks.headOption.getOrElse(0.0)
    val mid = (bb + ba) / 2; val spread = if (mid > 0) (ba - bb) / mid * 10000 else 0.0
    val imb = if (bids.nonEmpty && asks.nonEmpty) {
      val bs = bids.sum; val as_ = asks.sum
      if (bs + as_ > 0) (bs - as_) / (bs + as_) else 0.0
    } else 0.0
    Map("mid" -> mid, "spread_bps" -> spread, "imbalance" -> imb, "language" -> LANGUAGE)
  }

  def signalFilter(d: Map[String, Any]): Map[String, Any] = {
    val sig = getDbl(d, "signal", 0); val conf = getDbl(d, "confidence", 0.5)
    val accept = conf >= 0.6
    Map("accept" -> accept, "filtered_signal" -> (if (accept) sig else 0.0),
      "filter_reason" -> (if (accept) "confidence_ok" else "low_confidence"), "language" -> LANGUAGE)
  }

  def confidenceCalibration(d: Map[String, Any]): Map[String, Any] = {
    val raw = getDbl(d, "raw_confidence", 0.5)
    val cal = 1.0 / (1.0 + math.exp(-(raw - 0.5) * 4))
    Map("calibrated_confidence" -> cal, "raw_confidence" -> raw, "language" -> LANGUAGE)
  }

  def varEstimate(d: Map[String, Any]): Map[String, Any] = {
    val returns = getDblList(d, "returns"); val pctile = getDbl(d, "percentile", 5)
    val v = if (returns.isEmpty) 0.0 else {
      val s = returns.sorted; val idx = math.max(0, math.min((pctile / 100 * s.length).toInt, s.length - 1))
      s(idx)
    }
    val cvar = if (returns.isEmpty) 0.0 else {
      val s = returns.sorted; val cutoff = math.max(1, (pctile / 100 * s.length).toInt)
      s.take(cutoff).sum / cutoff
    }
    Map("var_pct" -> v, "cvar_pct" -> cvar, "percentile" -> pctile, "n" -> returns.length, "language" -> LANGUAGE)
  }

  def skewEstimate(d: Map[String, Any]): Map[String, Any] = {
    val returns = getDblList(d, "returns")
    val skew = if (returns.length < 3) 0.0 else {
      val mean = returns.sum / returns.length
      val m2 = returns.map(x => math.pow(x - mean, 2)).sum / returns.length
      val m3 = returns.map(x => math.pow(x - mean, 3)).sum / returns.length
      val sd = math.sqrt(m2); if (sd > 0) m3 / (sd * sd * sd) else 0.0
    }
    Map("skew" -> skew, "n" -> returns.length, "language" -> LANGUAGE)
  }

  def obImbalanceSeries(d: Map[String, Any]): Map[String, Any] = {
    val bv = getDblList(d, "bid_volumes"); val av = getDblList(d, "ask_volumes")
    val n = math.min(bv.length, av.length)
    val imbs = (0 until n).map { i =>
      val t = bv(i) + av(i); if (t > 0) (bv(i) - av(i)) / t else 0.0
    }
    val avg = if (imbs.nonEmpty) imbs.sum / imbs.length else 0.0
    val trend = if (imbs.length >= 2) {
      val last = imbs(imbs.length - 1); val first = imbs(0)
      if (last > first) "increasing" else if (last < first) "decreasing" else "stable"
    } else "stable"
    Map("imbalance_series" -> imbs, "trend" -> trend, "mean_imbalance" -> avg, "n" -> n, "language" -> LANGUAGE)
  }

  def execQuality(d: Map[String, Any]): Map[String, Any] = {
    val exp = getDbl(d, "expected_price", 100); val exec = getDbl(d, "executed_price", 100)
    val slip = if (exp > 0) math.abs(exec - exp) / exp * 10000 else 0.0
    Map("score_0_1" -> math.max(0, 1.0 - slip / 100), "avg_slippage_bps" -> slip, "language" -> LANGUAGE)
  }

  def regimeDuration(d: Map[String, Any]): Map[String, Any] = {
    val prices = getDblList(d, "prices")
    if (prices.length < 2) Map("regime" -> "unknown", "bars_in_regime" -> 0, "regime_stable" -> false, "language" -> LANGUAGE)
    else {
      val up = prices.last > prices(prices.length - 2)
      var dur = 1
      var i = prices.length - 2
      while (i >= 1 && (prices(i) > prices(i - 1)) == up) { dur += 1; i -= 1 }
      Map("regime" -> (if (up) "trending_up" else "trending_down"), "bars_in_regime" -> dur, "regime_stable" -> (dur > 10), "language" -> LANGUAGE)
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────

  def sha256Long(s: String): Long = {
    val md = MessageDigest.getInstance("SHA-256")
    val h = md.digest(s.getBytes("UTF-8"))
    var v = 0L; for (i <- 0 until 8) v = (v << 8) | (h(i) & 0xFF)
    math.abs(v)
  }

  def fmtD(d: Double): String = if (d == d.toLong.toDouble) d.toLong.toString else f"$d%.6f"

  def esc(s: String): String =
    if (s == null) "" else s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")

  def getDbl(m: Map[String, Any], k: String, default: Double): Double = m.get(k) match {
    case Some(n: Number) => n.doubleValue()
    case Some(s: String) => Try(s.toDouble).getOrElse(default)
    case _ => default
  }

  def getStr(m: Map[String, Any], k: String, default: String): String = m.get(k) match {
    case Some(s) => s.toString
    case None => default
  }

  def getDblList(m: Map[String, Any], k: String): IndexedSeq[Double] = m.get(k) match {
    case Some(l: Seq[_]) => l.map {
      case n: Number => n.doubleValue()
      case _ => 0.0
    }.toIndexedSeq
    case _ => IndexedSeq.empty
  }

  // ── Minimal JSON parser ────────────────────────────────────────────────

  var pos = 0
  def parseJson(s: String): Map[String, Any] = { pos = 0; parseObj(s) }

  def skipWs(s: String): Unit = while (pos < s.length && " \t\r\n".contains(s.charAt(pos))) pos += 1

  def parseObj(s: String): Map[String, Any] = {
    skipWs(s)
    if (pos >= s.length || s.charAt(pos) != '{') return Map.empty
    pos += 1
    var m = Map.empty[String, Any]
    skipWs(s)
    if (pos < s.length && s.charAt(pos) == '}') { pos += 1; return m }
    var cont = true
    while (cont && pos < s.length) {
      skipWs(s); val key = parseStr(s); skipWs(s)
      if (pos < s.length && s.charAt(pos) == ':') pos += 1
      skipWs(s); val v = parseVal(s); m = m + (key -> v)
      skipWs(s)
      if (pos < s.length && s.charAt(pos) == ',') pos += 1
      else if (pos < s.length && s.charAt(pos) == '}') { pos += 1; cont = false }
      else cont = false
    }
    m
  }

  def parseVal(s: String): Any = {
    skipWs(s)
    if (pos >= s.length) null
    else s.charAt(pos) match {
      case '"' => parseStr(s)
      case '{' => parseObj(s)
      case '[' => parseArr(s)
      case 't' => pos += 4; true
      case 'f' => pos += 5; false
      case 'n' => pos += 4; null
      case _ => parseNum(s)
    }
  }

  def parseStr(s: String): String = {
    if (pos >= s.length || s.charAt(pos) != '"') return ""
    pos += 1; val sb = new StringBuilder
    while (pos < s.length) {
      val c = s.charAt(pos)
      if (c == '\\' && pos + 1 < s.length) { pos += 1; sb += s.charAt(pos) }
      else if (c == '"') { pos += 1; return sb.toString() }
      else sb += c
      pos += 1
    }
    sb.toString()
  }

  def parseArr(s: String): Seq[Any] = {
    pos += 1; var list = Seq.empty[Any]; skipWs(s)
    if (pos < s.length && s.charAt(pos) == ']') { pos += 1; return list }
    var cont = true
    while (cont && pos < s.length) {
      list = list :+ parseVal(s); skipWs(s)
      if (pos < s.length && s.charAt(pos) == ',') pos += 1
      else if (pos < s.length && s.charAt(pos) == ']') { pos += 1; cont = false }
      else cont = false
    }
    list
  }

  def parseNum(s: String): Any = {
    val start = pos
    while (pos < s.length && "0123456789.eE+-".contains(s.charAt(pos))) pos += 1
    val num = s.substring(start, pos)
    if (num.contains(".") || num.contains("e") || num.contains("E")) num.toDouble
    else Try(num.toLong).getOrElse(num.toDouble)
  }

  // ── JSON serializer ───────────────────────────────────────────────────

  def toJson(m: Map[String, Any]): String = {
    val entries = m.map { case (k, v) => s""""${esc(k)}":${jsonVal(v)}""" }
    entries.mkString("{", ",", "}")
  }

  def jsonVal(v: Any): String = v match {
    case null => "null"
    case b: Boolean => b.toString
    case i: Int => i.toString
    case l: Long => l.toString
    case d: Double => if (d == d.toLong.toDouble) d.toLong.toString else String.format(Locale.US, "%.8g", d: java.lang.Double)
    case s: String => s""""${esc(s)}""""
    case seq: Seq[_] => seq.map(jsonVal).mkString("[", ",", "]")
    case m: Map[_, _] => toJson(m.asInstanceOf[Map[String, Any]])
    case other => s""""${esc(other.toString)}""""
  }
}
