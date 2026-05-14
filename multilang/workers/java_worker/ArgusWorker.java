// ArgusWorker.java — Argus Java stdin/stdout worker
// Compile: javac ArgusWorker.java
// Run:     java ArgusWorker
// Protocol: one JSON line in -> one JSON line out, loop until EOF

import java.io.*;
import java.security.MessageDigest;
import java.util.*;

public class ArgusWorker {

    static final String LANGUAGE = "java";
    static final double RISK_MAX = 0.45;
    static final double CYCLE_SCALE = 0.98;
    static final double VOL_W = 1.0;
    static final double SIG_W = 1.0;
    static final double SPREAD = 1.01;

    public static void main(String[] args) throws Exception {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out, "UTF-8")));
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            long t0 = System.nanoTime();
            try {
                Map<String, Object> req = parseJson(line);
                String taskType = getString(req, "task_type", "heartbeat");
                @SuppressWarnings("unchecked")
                Map<String, Object> data = (Map<String, Object>) req.getOrDefault("data", new HashMap<>());
                if (data == null) data = new HashMap<>();
                Map<String, Object> result = dispatch(taskType, data);
                double tookMs = (System.nanoTime() - t0) / 1_000_000.0;
                out.println("{\"ok\":true,\"result\":" + toJson(result) + ",\"took_ms\":" + fmt(tookMs) + "}");
            } catch (Exception e) {
                double tookMs = (System.nanoTime() - t0) / 1_000_000.0;
                out.println("{\"ok\":false,\"error\":\"" + escapeStr(e.getMessage()) + "\",\"took_ms\":" + fmt(tookMs) + "}");
            }
            out.flush();
        }
    }

    static Map<String, Object> dispatch(String taskType, Map<String, Object> data) throws Exception {
        switch (taskType) {
            case "cycle_plan":      return cyclePlan(data);
            case "heartbeat":       return heartbeat(data);
            case "volatility_estimate": return volatilityEstimate(data);
            case "signal_score":    return signalScore(data);
            case "risk_calculation": return riskCalculation(data);
            case "position_sizing": return positionSizing(data);
            case "drawdown_check":  return drawdownCheck(data);
            case "regime_estimate": return regimeEstimate(data);
            case "slippage_estimate": return slippageEstimate(data);
            case "correlation_estimate": return correlationEstimate(data);
            case "liquidity_score": return liquidityScore(data);
            case "market_impact":   return marketImpact(data);
            case "order_book_processing": return orderBookProcessing(data);
            case "signal_filter":   return signalFilter(data);
            case "confidence_calibration": return confidenceCalibration(data);
            case "var_estimate":    return varEstimate(data);
            case "skew_estimate":   return skewEstimate(data);
            case "order_book_imbalance_series": return orderBookImbalanceSeries(data);
            case "execution_quality_score": return executionQualityScore(data);
            case "regime_duration": return regimeDuration(data);
            default:
                Map<String, Object> m = new HashMap<>();
                m.put("language", LANGUAGE);
                m.put("task_type", taskType);
                m.put("value", 0.5);
                return m;
        }
    }

    // ── Task implementations ────────────────────────────────────────────────

    static Map<String, Object> cyclePlan(Map<String, Object> data) throws Exception {
        TreeMap<String, Object> sorted = new TreeMap<>(data);
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Object> e : sorted.entrySet()) {
            if (sb.length() > 0) sb.append(",");
            sb.append(e.getKey()).append(":").append(e.getValue());
        }
        long h = sha256Long(sb.toString());
        double base = ((Math.floorMod(h, 200) - 100)) / 10000.0;
        double cash = getDouble(data, "cash", getDouble(data, "cash_balance_aud", 0.0));
        double capital = getDouble(data, "capital", getDouble(data, "portfolio_value_aud", 1000.0));
        String[] actions = {"buy", "sell", "hold"};
        String action = actions[(int) Math.floorMod(h, 3)];
        double confidence = 0.5 + base;
        double cycleBoost = base * CYCLE_SCALE;
        double sizeAud = capital * 0.02 * CYCLE_SCALE * Math.min(1.0, Math.abs(base) * 50);
        Map<String, Object> r = new HashMap<>();
        r.put("action", action);
        r.put("cycle_boost", cycleBoost);
        r.put("confidence", confidence);
        r.put("size_aud", Math.max(0, sizeAud));
        r.put("spread_multiplier", SPREAD);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> heartbeat(Map<String, Object> data) {
        Map<String, Object> r = new HashMap<>();
        r.put("ok", true);
        r.put("language", LANGUAGE);
        r.put("ts", System.currentTimeMillis());
        return r;
    }

    static Map<String, Object> volatilityEstimate(Map<String, Object> data) {
        List<Double> prices = getDoubleList(data, "prices");
        double volBps;
        if (prices.size() < 2) {
            volBps = 1000.0;
        } else {
            double[] lr = new double[prices.size() - 1];
            for (int i = 0; i < lr.length; i++) {
                lr[i] = Math.log(prices.get(i + 1) / prices.get(i));
            }
            double mean = 0;
            for (double v : lr) mean += v;
            mean /= lr.length;
            double var = 0;
            for (double v : lr) var += (v - mean) * (v - mean);
            var /= (lr.length > 1 ? lr.length - 1 : 1);
            volBps = Math.sqrt(var) * Math.sqrt(365) * 10000 * VOL_W;
        }
        Map<String, Object> r = new HashMap<>();
        r.put("volatility_annual_bps", volBps);
        r.put("volatility_weight", VOL_W);
        r.put("n", prices.size());
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> signalScore(Map<String, Object> data) throws Exception {
        double signal = getDouble(data, "signal", 0.5);
        double confidence = getDouble(data, "confidence", 0.5);
        String regime = getString(data, "regime", "unknown");
        TreeMap<String, Object> sorted = new TreeMap<>(data);
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Object> e : sorted.entrySet()) {
            if (sb.length() > 0) sb.append(",");
            sb.append(e.getKey()).append(":").append(e.getValue());
        }
        long h = sha256Long(sb.toString());
        double scoreDelta = ((Math.floorMod(h, 100)) - 50) / 5000.0 * SIG_W;
        Map<String, Object> r = new HashMap<>();
        r.put("score_delta", scoreDelta);
        r.put("confidence", confidence);
        r.put("regime", regime);
        r.put("signal_score_weight", SIG_W);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> riskCalculation(Map<String, Object> data) {
        double pv = getDouble(data, "position_value", 0);
        double cap = getDouble(data, "capital", 1000);
        double maxDd = getDouble(data, "max_drawdown_pct", 20);
        double ratio = cap > 0 ? pv / cap : 0;
        Map<String, Object> r = new HashMap<>();
        r.put("exposure_ratio", ratio);
        r.put("passed", ratio <= RISK_MAX);
        r.put("max_ratio", RISK_MAX);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> positionSizing(Map<String, Object> data) {
        double cap = getDouble(data, "capital", 1000);
        double volBps = getDouble(data, "volatility_bps", 1000);
        double conf = getDouble(data, "confidence", 0.5);
        double size = cap * 0.02 * (1 - volBps / 20000.0) * conf;
        size = Math.max(10, Math.min(cap * RISK_MAX, size));
        double sizePct = cap > 0 ? (size / cap) * 100.0 : 0;
        Map<String, Object> r = new HashMap<>();
        r.put("size_pct", sizePct);
        r.put("size_abs", size);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> drawdownCheck(Map<String, Object> data) {
        double cur = getDouble(data, "current_equity", 1000);
        double peak = getDouble(data, "peak_equity", 1000);
        double maxDd = getDouble(data, "max_drawdown_pct", 20);
        double dd = peak > 0 ? (peak - cur) / peak : 0;
        boolean inDrawdown = dd * 100 > maxDd;
        Map<String, Object> r = new HashMap<>();
        r.put("passed", !inDrawdown);
        r.put("current_drawdown_pct", dd * 100);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> regimeEstimate(Map<String, Object> data) {
        List<Double> prices = getDoubleList(data, "prices");
        String regime = "unknown";
        double confidence = 0.5;
        if (prices.size() >= 5) {
            double first = prices.get(0);
            double last = prices.get(prices.size() - 1);
            double change = (last - first) / first;
            if (change > 0.02) { regime = "trending_up"; confidence = 0.6 + Math.min(0.3, change); }
            else if (change < -0.02) { regime = "trending_down"; confidence = 0.6 + Math.min(0.3, Math.abs(change)); }
            else { regime = "mean_reverting"; confidence = 0.55; }
        }
        Map<String, Object> r = new HashMap<>();
        r.put("regime", regime);
        r.put("confidence", confidence);
        r.put("regime_weight", 1.0);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> slippageEstimate(Map<String, Object> data) {
        double spread = getDouble(data, "spread_bps", 10);
        double vol = getDouble(data, "volatility_bps", 1000);
        double size = getDouble(data, "order_size_aud", 100);
        double slip = spread * 0.5 + vol * 0.001 + size * 0.0001;
        Map<String, Object> r = new HashMap<>();
        r.put("slippage_bps", slip * SPREAD);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> correlationEstimate(Map<String, Object> data) {
        List<Double> xs = getDoubleList(data, "series_a");
        List<Double> ys = getDoubleList(data, "series_b");
        double corr = 0;
        int n = Math.min(xs.size(), ys.size());
        if (n >= 3) {
            double mx = 0, my = 0;
            for (int i = 0; i < n; i++) { mx += xs.get(i); my += ys.get(i); }
            mx /= n; my /= n;
            double cov = 0, sx = 0, sy = 0;
            for (int i = 0; i < n; i++) {
                double dx = xs.get(i) - mx, dy = ys.get(i) - my;
                cov += dx * dy; sx += dx * dx; sy += dy * dy;
            }
            double denom = Math.sqrt(sx * sy);
            corr = denom > 0 ? cov / denom : 0;
        }
        Map<String, Object> r = new HashMap<>();
        r.put("correlation", corr);
        r.put("n", n);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> liquidityScore(Map<String, Object> data) {
        double vol24h = getDouble(data, "volume_24h", 1e6);
        double spread = getDouble(data, "spread_bps", 10);
        double depth = getDouble(data, "order_book_depth", 100);
        double score = Math.min(1.0, (vol24h / 1e7) * 0.4 + (1.0 / (1 + spread / 100)) * 0.3 + (depth / 1000) * 0.3);
        Map<String, Object> r = new HashMap<>();
        r.put("liquidity_score", score);
        r.put("depth_bps", spread);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> marketImpact(Map<String, Object> data) {
        double size = getDouble(data, "order_size_aud", 100);
        double vol = getDouble(data, "volume_24h", 1e6);
        double impact = vol > 0 ? Math.sqrt(size / vol) * 100 : 10;
        Map<String, Object> r = new HashMap<>();
        r.put("impact_bps", impact);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> orderBookProcessing(Map<String, Object> data) {
        List<Double> bids = getDoubleList(data, "bids");
        List<Double> asks = getDoubleList(data, "asks");
        double bestBid = bids.isEmpty() ? 0 : bids.get(0);
        double bestAsk = asks.isEmpty() ? 0 : asks.get(0);
        double mid = (bestBid + bestAsk) / 2;
        double spread = mid > 0 ? (bestAsk - bestBid) / mid * 10000 : 0;
        double imbalance = 0;
        if (!bids.isEmpty() && !asks.isEmpty()) {
            double bSum = 0, aSum = 0;
            for (double b : bids) bSum += b;
            for (double a : asks) aSum += a;
            imbalance = (bSum + aSum) > 0 ? (bSum - aSum) / (bSum + aSum) : 0;
        }
        Map<String, Object> r = new HashMap<>();
        r.put("mid", mid);
        r.put("spread_bps", spread);
        r.put("imbalance", imbalance);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> signalFilter(Map<String, Object> data) {
        double signal = getDouble(data, "signal", 0);
        double confidence = getDouble(data, "confidence", 0.5);
        String regime = getString(data, "regime", "unknown");
        boolean accept = confidence >= 0.6;
        Map<String, Object> r = new HashMap<>();
        r.put("accept", accept);
        r.put("filtered_signal", accept ? signal : 0);
        r.put("filter_reason", accept ? "confidence_ok" : "low_confidence");
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> confidenceCalibration(Map<String, Object> data) {
        double raw = getDouble(data, "raw_confidence", 0.5);
        double calibrated = 1.0 / (1.0 + Math.exp(-(raw - 0.5) * 4));
        Map<String, Object> r = new HashMap<>();
        r.put("calibrated_confidence", calibrated);
        r.put("raw_confidence", raw);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> varEstimate(Map<String, Object> data) {
        List<Double> returns = getDoubleList(data, "returns");
        double pctile = getDouble(data, "percentile", 5);
        double var = 0;
        double cvar = 0;
        if (!returns.isEmpty()) {
            List<Double> sorted = new ArrayList<>(returns);
            Collections.sort(sorted);
            int idx = (int) Math.floor(pctile / 100.0 * sorted.size());
            idx = Math.max(0, Math.min(idx, sorted.size() - 1));
            var = sorted.get(idx);
            // CVaR = mean of returns at or below the VaR threshold
            double sum = 0;
            int count = 0;
            for (int i = 0; i <= idx; i++) {
                sum += sorted.get(i);
                count++;
            }
            cvar = count > 0 ? sum / count : var;
        }
        Map<String, Object> r = new HashMap<>();
        r.put("var_pct", var);
        r.put("cvar_pct", cvar);
        r.put("percentile", pctile);
        r.put("n", returns.size());
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> skewEstimate(Map<String, Object> data) {
        List<Double> returns = getDoubleList(data, "returns");
        double skew = 0;
        if (returns.size() >= 3) {
            double mean = 0;
            for (double v : returns) mean += v;
            mean /= returns.size();
            double m2 = 0, m3 = 0;
            for (double v : returns) {
                double d = v - mean;
                m2 += d * d;
                m3 += d * d * d;
            }
            m2 /= returns.size();
            m3 /= returns.size();
            double sd = Math.sqrt(m2);
            skew = sd > 0 ? m3 / (sd * sd * sd) : 0;
        }
        Map<String, Object> r = new HashMap<>();
        r.put("skew", skew);
        r.put("n", returns.size());
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> orderBookImbalanceSeries(Map<String, Object> data) {
        List<Double> bidVols = getDoubleList(data, "bid_volumes");
        List<Double> askVols = getDoubleList(data, "ask_volumes");
        List<Double> imbalances = new ArrayList<>();
        int n = Math.min(bidVols.size(), askVols.size());
        for (int i = 0; i < n; i++) {
            double total = bidVols.get(i) + askVols.get(i);
            imbalances.add(total > 0 ? (bidVols.get(i) - askVols.get(i)) / total : 0);
        }
        double avg = 0;
        if (!imbalances.isEmpty()) {
            for (double v : imbalances) avg += v;
            avg /= imbalances.size();
        }
        String trend = avg > 0.05 ? "bid_heavy" : (avg < -0.05 ? "ask_heavy" : "balanced");
        Map<String, Object> r = new HashMap<>();
        r.put("imbalance_series", imbalances);
        r.put("trend", trend);
        r.put("n", n);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> executionQualityScore(Map<String, Object> data) {
        double expectedPrice = getDouble(data, "expected_price", 100);
        double executedPrice = getDouble(data, "executed_price", 100);
        double slippageBps = expectedPrice > 0 ? Math.abs(executedPrice - expectedPrice) / expectedPrice * 10000 : 0;
        double score = Math.max(0, 1.0 - slippageBps / 100);
        Map<String, Object> r = new HashMap<>();
        r.put("quality_score", score);
        r.put("slippage_bps", slippageBps);
        r.put("language", LANGUAGE);
        return r;
    }

    static Map<String, Object> regimeDuration(Map<String, Object> data) {
        List<Double> prices = getDoubleList(data, "prices");
        int duration = 0;
        String currentRegime = "unknown";
        if (prices.size() >= 2) {
            double last = prices.get(prices.size() - 1);
            double prev = prices.get(prices.size() - 2);
            boolean up = last > prev;
            currentRegime = up ? "trending_up" : "trending_down";
            duration = 1;
            for (int i = prices.size() - 2; i >= 1; i--) {
                boolean iUp = prices.get(i) > prices.get(i - 1);
                if (iUp == up) duration++;
                else break;
            }
        }
        Map<String, Object> r = new HashMap<>();
        r.put("regime", currentRegime);
        r.put("duration_bars", duration);
        r.put("language", LANGUAGE);
        return r;
    }

    // ── JSON helpers (manual, no deps) ──────────────────────────────────────

    static long sha256Long(String s) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] hash = md.digest(s.getBytes("UTF-8"));
        long v = 0;
        for (int i = 0; i < 8; i++) v = (v << 8) | (hash[i] & 0xFF);
        return Math.abs(v);
    }

    static String fmt(double v) {
        if (v == (long) v) return Long.toString((long) v);
        return String.format(Locale.US, "%.6f", v);
    }

    static String escapeStr(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }

    // Minimal JSON object serializer (flat maps only, values: String, Number, Boolean, List)
    static String toJson(Map<String, Object> m) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Object> e : m.entrySet()) {
            if (!first) sb.append(",");
            first = false;
            sb.append("\"").append(escapeStr(e.getKey())).append("\":");
            sb.append(jsonValue(e.getValue()));
        }
        sb.append("}");
        return sb.toString();
    }

    static String jsonValue(Object v) {
        if (v == null) return "null";
        if (v instanceof Boolean) return v.toString();
        if (v instanceof Integer || v instanceof Long) return v.toString();
        if (v instanceof Double || v instanceof Float) {
            double d = ((Number) v).doubleValue();
            if (d == (long) d) return Long.toString((long) d);
            return String.format(Locale.US, "%.8g", d);
        }
        if (v instanceof List) {
            StringBuilder sb = new StringBuilder("[");
            boolean f = true;
            for (Object item : (List<?>) v) {
                if (!f) sb.append(",");
                f = false;
                sb.append(jsonValue(item));
            }
            sb.append("]");
            return sb.toString();
        }
        return "\"" + escapeStr(v.toString()) + "\"";
    }

    // Minimal JSON parser (supports nested objects, arrays, strings, numbers, booleans)
    static int pos;

    static Map<String, Object> parseJson(String s) {
        pos = 0;
        return parseObject(s);
    }

    static void skipWs(String s) {
        while (pos < s.length() && " \t\r\n".indexOf(s.charAt(pos)) >= 0) pos++;
    }

    static Map<String, Object> parseObject(String s) {
        skipWs(s);
        if (pos >= s.length() || s.charAt(pos) != '{') return new HashMap<>();
        pos++;
        Map<String, Object> m = new LinkedHashMap<>();
        skipWs(s);
        if (pos < s.length() && s.charAt(pos) == '}') { pos++; return m; }
        while (pos < s.length()) {
            skipWs(s);
            String key = parseString(s);
            skipWs(s);
            if (pos < s.length() && s.charAt(pos) == ':') pos++;
            skipWs(s);
            Object val = parseValue(s);
            m.put(key, val);
            skipWs(s);
            if (pos < s.length() && s.charAt(pos) == ',') { pos++; continue; }
            if (pos < s.length() && s.charAt(pos) == '}') { pos++; break; }
            break;
        }
        return m;
    }

    static Object parseValue(String s) {
        skipWs(s);
        if (pos >= s.length()) return null;
        char c = s.charAt(pos);
        if (c == '"') return parseString(s);
        if (c == '{') return parseObject(s);
        if (c == '[') return parseArray(s);
        if (c == 't') { pos += 4; return true; }
        if (c == 'f') { pos += 5; return false; }
        if (c == 'n') { pos += 4; return null; }
        return parseNumber(s);
    }

    static String parseString(String s) {
        if (pos >= s.length() || s.charAt(pos) != '"') return "";
        pos++;
        StringBuilder sb = new StringBuilder();
        while (pos < s.length()) {
            char c = s.charAt(pos);
            if (c == '\\' && pos + 1 < s.length()) {
                pos++;
                char n = s.charAt(pos);
                if (n == '"') sb.append('"');
                else if (n == '\\') sb.append('\\');
                else if (n == 'n') sb.append('\n');
                else sb.append(n);
            } else if (c == '"') { pos++; return sb.toString(); }
            else sb.append(c);
            pos++;
        }
        return sb.toString();
    }

    static List<Object> parseArray(String s) {
        pos++;
        List<Object> list = new ArrayList<>();
        skipWs(s);
        if (pos < s.length() && s.charAt(pos) == ']') { pos++; return list; }
        while (pos < s.length()) {
            list.add(parseValue(s));
            skipWs(s);
            if (pos < s.length() && s.charAt(pos) == ',') { pos++; continue; }
            if (pos < s.length() && s.charAt(pos) == ']') { pos++; break; }
            break;
        }
        return list;
    }

    static Number parseNumber(String s) {
        int start = pos;
        while (pos < s.length() && "0123456789.eE+-".indexOf(s.charAt(pos)) >= 0) pos++;
        String num = s.substring(start, pos);
        if (num.contains(".") || num.contains("e") || num.contains("E"))
            return Double.parseDouble(num);
        try { return Long.parseLong(num); }
        catch (Exception e) { return Double.parseDouble(num); }
    }

    // Helpers to extract typed values from parsed maps
    static double getDouble(Map<String, Object> m, String key, double def) {
        Object v = m.get(key);
        if (v instanceof Number) return ((Number) v).doubleValue();
        if (v instanceof String) try { return Double.parseDouble((String) v); } catch (Exception e) {}
        return def;
    }

    static String getString(Map<String, Object> m, String key, String def) {
        Object v = m.get(key);
        return v != null ? v.toString() : def;
    }

    @SuppressWarnings("unchecked")
    static List<Double> getDoubleList(Map<String, Object> m, String key) {
        Object v = m.get(key);
        List<Double> out = new ArrayList<>();
        if (v instanceof List) {
            for (Object item : (List<Object>) v) {
                if (item instanceof Number) out.add(((Number) item).doubleValue());
            }
        }
        return out;
    }
}
