// Argus C# Worker — .NET 10 top-level statements
// Compile: dotnet run  OR  dotnet publish -c Release
// Reads JSON lines from stdin, writes JSON lines to stdout.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

// ─── Top-level statements (must precede type declarations) ─────────────────

static Dictionary<string, object?> Dispatch(string taskType, JsonElement data)
{
    return taskType switch
    {
        "cycle_plan"                   => Tasks.CyclePlan(data),
        "order_book_processing"        => Tasks.OrderBookProcessing(data),
        "risk_calculation"             => Tasks.RiskCalculation(data),
        "volatility_estimate"          => Tasks.VolatilityEstimate(data),
        "signal_score"                 => Tasks.SignalScore(data),
        "regime_estimate"              => Tasks.RegimeEstimate(data),
        "slippage_estimate"            => Tasks.SlippageEstimate(data),
        "position_sizing"              => Tasks.PositionSizing(data),
        "drawdown_check"               => Tasks.DrawdownCheck(data),
        "correlation_estimate"         => Tasks.CorrelationEstimate(data),
        "liquidity_score"              => Tasks.LiquidityScore(data),
        "market_impact"                => Tasks.MarketImpact(data),
        "signal_filter"                => Tasks.SignalFilter(data),
        "confidence_calibration"       => Tasks.ConfidenceCalibration(data),
        "heartbeat"                    => Tasks.Heartbeat(data),
        "var_estimate"                 => Tasks.VarEstimate(data),
        "skew_estimate"                => Tasks.SkewEstimate(data),
        "order_book_imbalance_series"  => Tasks.OrderBookImbalanceSeries(data),
        "execution_quality_score"      => Tasks.ExecutionQualityScore(data),
        "regime_duration"              => Tasks.RegimeDuration(data),
        _ => throw new ArgumentException($"unknown task: {taskType}")
    };
}

static void WriteDict(Utf8JsonWriter jw, Dictionary<string, object?> d)
{
    jw.WriteStartObject();
    foreach (var kv in d)
    {
        jw.WritePropertyName(kv.Key);
        WriteVal(jw, kv.Value);
    }
    jw.WriteEndObject();
}

static void WriteVal(Utf8JsonWriter jw, object? v)
{
    switch (v)
    {
        case null:
            jw.WriteNullValue(); break;
        case bool b:
            jw.WriteBooleanValue(b); break;
        case int i:
            jw.WriteNumberValue(i); break;
        case double d:
            if (double.IsNaN(d) || double.IsInfinity(d)) jw.WriteNumberValue(0.0);
            else jw.WriteNumberValue(d); break;
        case float f:
            jw.WriteNumberValue((double)f); break;
        case string s:
            jw.WriteStringValue(s); break;
        case List<double> ld:
            jw.WriteStartArray();
            foreach (var x in ld) jw.WriteNumberValue(x);
            jw.WriteEndArray(); break;
        case IEnumerable<object?> seq:
            jw.WriteStartArray();
            foreach (var x in seq) WriteVal(jw, x);
            jw.WriteEndArray(); break;
        case Dictionary<string, object?> sub:
            WriteDict(jw, sub); break;
        default:
            jw.WriteStringValue(v.ToString()); break;
    }
}

// ─── Main loop ──────────────────────────────────────────────────────────────
Console.InputEncoding  = Encoding.UTF8;
Console.OutputEncoding = Encoding.UTF8;

var opts = new JsonWriterOptions { Indented = false };

string? line;
while ((line = Console.ReadLine()) != null)
{
    if (string.IsNullOrWhiteSpace(line)) continue;
    var sw = Stopwatch.StartNew();
    try
    {
        using var doc  = JsonDocument.Parse(line);
        var root       = doc.RootElement;
        string taskType = root.TryGetProperty("task_type", out var tt) ? tt.GetString() ?? "" : "";
        JsonElement data = root.TryGetProperty("data", out var dd) ? dd : default;

        var result = Dispatch(taskType, data);
        sw.Stop();
        double tookMs = sw.Elapsed.TotalMilliseconds;

        using var ms  = new System.IO.MemoryStream();
        using var jw  = new Utf8JsonWriter(ms, opts);
        jw.WriteStartObject();
        jw.WriteBoolean("ok", true);
        jw.WritePropertyName("result");
        WriteDict(jw, result);
        jw.WriteNumber("took_ms", tookMs);
        jw.WriteEndObject();
        jw.Flush();
        Console.WriteLine(Encoding.UTF8.GetString(ms.ToArray()));
    }
    catch (Exception ex)
    {
        sw.Stop();
        var err = new Dictionary<string, object?>
        {
            ["ok"]      = false,
            ["error"]   = ex.Message,
            ["took_ms"] = sw.Elapsed.TotalMilliseconds
        };
        Console.WriteLine(JsonSerializer.Serialize(err));
    }
    Console.Out.Flush();
}

// ─── Language profile ───────────────────────────────────────────────────────
static class Profile
{
    public const string   Language   = "csharp";
    public const double   RiskMax    = 0.45;
    public const double   CycleScale = 0.99;
    public const double   VolWeight  = 1.0;
    public const double   SigWeight  = 1.0;
    public const double   SpreadMult = 1.01;
    public const string   Role       = "ecosystem";

    public static readonly int LangIdx = Language.Aggregate(0, (s, c) => s + c) % 100;
}

// ─── Helpers ────────────────────────────────────────────────────────────────
static class Helpers
{
    public static double Clamp(double v, double lo, double hi) => Math.Max(lo, Math.Min(hi, v));

    public static int PolyHash(string s)
    {
        unchecked
        {
            int h = 0;
            foreach (char c in s) h = h * 31 + c;
            return Math.Abs(h);
        }
    }

    public static double WelfordVar(IList<double> xs)
    {
        if (xs.Count == 0) return 0.0;
        double mean = 0, m2 = 0;
        for (int i = 0; i < xs.Count; i++)
        {
            double delta = xs[i] - mean;
            mean += delta / (i + 1);
            m2   += delta * (xs[i] - mean);
        }
        return m2 / xs.Count;
    }

    public static double Pearson(IList<double> xs, IList<double> ys)
    {
        int n = Math.Min(xs.Count, ys.Count);
        if (n < 2) return 0.0;
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        for (int i = 0; i < n; i++)
        {
            sx += xs[i]; sy += ys[i];
            sxx += xs[i] * xs[i]; syy += ys[i] * ys[i];
            sxy += xs[i] * ys[i];
        }
        double num = n * sxy - sx * sy;
        double den = Math.Sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
        return den == 0 ? 0.0 : num / den;
    }

    public static double GetNum(JsonElement e, string key, double def = 0.0)
    {
        if (e.TryGetProperty(key, out var v))
        {
            if (v.ValueKind == JsonValueKind.Number) return v.GetDouble();
            if (v.ValueKind == JsonValueKind.True)   return 1.0;
            if (v.ValueKind == JsonValueKind.False)  return 0.0;
            if (v.ValueKind == JsonValueKind.String && double.TryParse(v.GetString(), out var d)) return d;
        }
        return def;
    }

    public static string GetStr(JsonElement e, string key, string def = "")
    {
        if (e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String)
            return v.GetString() ?? def;
        return def;
    }

    public static bool GetBool(JsonElement e, string key, bool def = false)
    {
        if (e.TryGetProperty(key, out var v))
        {
            if (v.ValueKind == JsonValueKind.True)  return true;
            if (v.ValueKind == JsonValueKind.False) return false;
            if (v.ValueKind == JsonValueKind.Number) return v.GetDouble() != 0;
        }
        return def;
    }

    public static List<double> GetNumArr(JsonElement e, string key)
    {
        var list = new List<double>();
        if (e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Array)
            foreach (var item in v.EnumerateArray())
                if (item.ValueKind == JsonValueKind.Number)
                    list.Add(item.GetDouble());
        return list;
    }

    public static List<JsonElement> GetArr(JsonElement e, string key)
    {
        var list = new List<JsonElement>();
        if (e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Array)
            foreach (var item in v.EnumerateArray())
                list.Add(item);
        return list;
    }

    public static string HashSortedData(JsonElement data)
    {
        var sb = new StringBuilder();
        if (data.ValueKind == JsonValueKind.Object)
        {
            var pairs = data.EnumerateObject()
                .OrderBy(p => p.Name)
                .Select(p => $"{p.Name}:{p.Value}");
            foreach (var p in pairs) sb.Append(p);
        }
        return sb.ToString();
    }

    public static List<double> LogReturns(List<double> prices)
    {
        var rets = new List<double>(prices.Count - 1);
        for (int i = 1; i < prices.Count; i++)
        {
            double prev = prices[i - 1];
            rets.Add(prev == 0 ? 0.0 : (prices[i] - prev) / prev);
        }
        return rets;
    }
}

// ─── Task handlers ──────────────────────────────────────────────────────────
static class Tasks
{
    public static Dictionary<string, object?> CyclePlan(JsonElement d)
    {
        string hashStr = Helpers.HashSortedData(d);
        int h = Helpers.PolyHash(hashStr);
        double baseBoost = ((h % 200) - 100) / 10000.0 + (Profile.LangIdx - 50) / 10000.0;

        double cash = Helpers.GetNum(d, "cash_balance_aud", 0.0);
        double pv   = Helpers.GetNum(d, "portfolio_value_aud", 1.0);
        if (pv == 0) pv = 1.0;
        int signals  = (int)Helpers.GetNum(d, "signals", 0.0);
        double cashR = cash / pv;
        double tilt  = (cashR - 0.5) * 0.002 + ((signals % 3) - 1) * 0.001;
        double boost = Helpers.Clamp((baseBoost + tilt) * Profile.CycleScale, -0.015, 0.015);

        return new Dictionary<string, object?>
        {
            ["language"]          = Profile.Language,
            ["cycle_boost"]       = boost,
            ["cycle_boost_scale"] = Profile.CycleScale,
            ["ok"]                = true
        };
    }

    public static Dictionary<string, object?> OrderBookProcessing(JsonElement d)
    {
        var bids  = Helpers.GetNumArr(d, "bids");
        var asks  = Helpers.GetNumArr(d, "asks");
        double bid0 = bids.Count > 0 ? bids[0] : 0.0;
        double ask0 = asks.Count > 0 ? asks[0] : 0.0;
        double mid  = (bid0 + ask0) / 2.0;
        double spreadBps = mid == 0 ? 0.0 : (ask0 - bid0) / mid * 10000.0 * Profile.SpreadMult;
        double sumB = bids.Take(5).Sum();
        double sumA = asks.Take(5).Sum();
        double imbal = (sumB + sumA) == 0 ? 0.0 : (sumB - sumA) / (sumB + sumA);

        return new Dictionary<string, object?>
        {
            ["spread_bps"] = spreadBps,
            ["imbalance"]  = imbal,
            ["mid"]        = mid,
            ["language"]   = Profile.Language
        };
    }

    public static Dictionary<string, object?> RiskCalculation(JsonElement d)
    {
        double posVal  = Helpers.GetNum(d, "position_value", 0.0);
        double capital = Helpers.GetNum(d, "capital", 1.0);
        if (capital == 0) capital = 1.0;
        double ratio  = posVal / capital;
        bool passed   = ratio <= Profile.RiskMax;

        return new Dictionary<string, object?>
        {
            ["passed"]         = passed,
            ["exposure_ratio"] = ratio,
            ["max_ratio"]      = Profile.RiskMax,
            ["language"]       = Profile.Language
        };
    }

    public static Dictionary<string, object?> VolatilityEstimate(JsonElement d)
    {
        var prices  = Helpers.GetNumArr(d, "prices");
        var returns = Helpers.GetNumArr(d, "returns");
        if (returns.Count == 0 && prices.Count >= 2)
            returns = Helpers.LogReturns(prices);
        double var_ = Helpers.WelfordVar(returns);
        double vol  = var_ > 0 ? Math.Sqrt(var_ * 252 * 10000.0) : 10.0;
        double volAdj = vol * Profile.VolWeight;

        return new Dictionary<string, object?>
        {
            ["volatility_annual_bps"] = volAdj,
            ["volatility_weight"]     = Profile.VolWeight,
            ["language"]              = Profile.Language,
            ["ok"]                    = true
        };
    }

    public static Dictionary<string, object?> SignalScore(JsonElement d)
    {
        string seed = Profile.Language + d.ToString();
        int h = Helpers.PolyHash(seed);
        double delta = ((h % 100) - 50) / 5000.0 * Profile.SigWeight;

        return new Dictionary<string, object?>
        {
            ["score_delta"]         = delta,
            ["signal_score_weight"] = Profile.SigWeight,
            ["language"]            = Profile.Language,
            ["ok"]                  = true
        };
    }

    public static Dictionary<string, object?> RegimeEstimate(JsonElement d)
    {
        var prices  = Helpers.GetNumArr(d, "prices");
        var returns = prices.Count >= 2 ? Helpers.LogReturns(prices) : Helpers.GetNumArr(d, "returns");
        double var_ = Helpers.WelfordVar(returns);
        double vol  = var_ > 0 ? Math.Sqrt(var_ * 252) : 0.0;
        string regime; double conf;
        if      (vol > 0.25) { regime = "high_vol";    conf = 0.70; }
        else if (vol > 0.05) { regime = "trend";       conf = 0.60; }
        else                  { regime = "mean_revert"; conf = 0.55; }

        return new Dictionary<string, object?>
        {
            ["regime"]        = regime,
            ["confidence"]    = conf,
            ["regime_weight"] = 1.0,
            ["language"]      = Profile.Language,
            ["ok"]            = true
        };
    }

    public static Dictionary<string, object?> SlippageEstimate(JsonElement d)
    {
        double halfSpread    = Helpers.GetNum(d, "half_spread_bps", 1.0);
        double participation = Helpers.GetNum(d, "participation_rate", 0.01);
        double slippage      = halfSpread * Profile.SpreadMult * (1.0 + participation * 10.0);

        return new Dictionary<string, object?>
        {
            ["slippage_bps"] = slippage,
            ["language"]     = Profile.Language,
            ["ok"]           = true
        };
    }

    public static Dictionary<string, object?> PositionSizing(JsonElement d)
    {
        double volBps   = Helpers.GetNum(d, "volatility_bps", 50.0);
        double conf     = Helpers.GetNum(d, "confidence", 0.5);
        double maxRisk  = Helpers.GetNum(d, "max_risk_pct", 0.02);
        double capital  = Helpers.GetNum(d, "capital", 100000.0);
        double sizePct  = Math.Min(Profile.RiskMax, maxRisk * (volBps / 10.0) * (0.5 + conf));
        double sizeAbs  = sizePct * capital;

        return new Dictionary<string, object?>
        {
            ["size_pct"] = sizePct,
            ["size_abs"] = sizeAbs,
            ["language"] = Profile.Language,
            ["ok"]       = true
        };
    }

    public static Dictionary<string, object?> DrawdownCheck(JsonElement d)
    {
        double peak    = Helpers.GetNum(d, "peak_value", 1.0);
        double current = Helpers.GetNum(d, "current_value", 1.0);
        double maxDd   = Helpers.GetNum(d, "max_drawdown_pct", 0.20);
        double dd      = peak == 0 ? 0.0 : (peak - current) / peak;
        bool passed    = dd <= maxDd * Profile.RiskMax;

        return new Dictionary<string, object?>
        {
            ["passed"]               = passed,
            ["current_drawdown_pct"] = dd * 100.0,
            ["language"]             = Profile.Language,
            ["ok"]                   = true
        };
    }

    public static Dictionary<string, object?> CorrelationEstimate(JsonElement d)
    {
        var xs = Helpers.GetNumArr(d, "series_a");
        var ys = Helpers.GetNumArr(d, "series_b");
        double corr = Helpers.Pearson(xs, ys);

        return new Dictionary<string, object?>
        {
            ["correlation"] = corr,
            ["language"]    = Profile.Language,
            ["ok"]          = true
        };
    }

    public static Dictionary<string, object?> LiquidityScore(JsonElement d)
    {
        var bids = Helpers.GetNumArr(d, "bids");
        var asks = Helpers.GetNumArr(d, "asks");
        double total = bids.Take(5).Sum() + asks.Take(5).Sum();
        double score = Math.Min(1.0, total / 100.0);
        double bid0  = bids.Count > 0 ? bids[0] : 0.0;
        double ask0  = asks.Count > 0 ? asks[0] : 0.0;
        double mid   = (bid0 + ask0) / 2.0;
        double depth = mid == 0 ? 0.0 : (ask0 - bid0) / mid * 10000.0;

        return new Dictionary<string, object?>
        {
            ["liquidity_score"] = score,
            ["depth_bps"]       = depth,
            ["language"]        = Profile.Language,
            ["ok"]              = true
        };
    }

    public static Dictionary<string, object?> MarketImpact(JsonElement d)
    {
        double qty    = Helpers.GetNum(d, "order_qty", 0.0);
        double adv    = Helpers.GetNum(d, "adv", 1.0); if (adv == 0) adv = 1.0;
        double vol    = Helpers.GetNum(d, "volatility", 0.01);
        double impact = 10.0 * Math.Sqrt(qty / adv) * vol * 10000.0;

        return new Dictionary<string, object?>
        {
            ["impact_bps"] = impact,
            ["language"]   = Profile.Language,
            ["ok"]         = true
        };
    }

    public static Dictionary<string, object?> SignalFilter(JsonElement d)
    {
        double conf   = Helpers.GetNum(d, "confidence", 0.0);
        string regime = Helpers.GetStr(d, "regime", "unknown");
        double vol    = Helpers.GetNum(d, "volatility", 0.0);
        bool accept   = conf >= 0.5 && (regime != "high_vol" || vol < 0.02);
        string reason = conf < 0.5 ? "low_confidence"
                      : (regime == "high_vol" && vol >= 0.02) ? "high_vol_regime"
                      : "accepted";

        return new Dictionary<string, object?>
        {
            ["accept"]        = accept,
            ["filter_reason"] = reason,
            ["language"]      = Profile.Language,
            ["ok"]            = true
        };
    }

    public static Dictionary<string, object?> ConfidenceCalibration(JsonElement d)
    {
        var confs   = Helpers.GetNumArr(d, "confidence_history");
        double winRate = Helpers.GetNum(d, "win_rate", 0.5);
        double avgConf = confs.Count > 0 ? confs.Average() : 0.5;
        double calib   = 0.5 * avgConf + 0.5 * winRate;

        return new Dictionary<string, object?>
        {
            ["calibrated_confidence"] = calib,
            ["language"]              = Profile.Language,
            ["ok"]                    = true
        };
    }

    public static Dictionary<string, object?> Heartbeat(JsonElement d)
    {
        double cycleId = Helpers.GetNum(d, "cycle_id", 0.0);
        return new Dictionary<string, object?>
        {
            ["ok"]         = true,
            ["latency_ms"] = 0.0,
            ["language"]   = Profile.Language,
            ["cycle_id"]   = cycleId
        };
    }

    public static Dictionary<string, object?> VarEstimate(JsonElement d)
    {
        var returns = Helpers.GetNumArr(d, "returns");
        double conf = Helpers.GetNum(d, "confidence_level", 0.95);
        if (returns.Count < 5)
            return new Dictionary<string, object?> { ["var_pct"]=0.0,["cvar_pct"]=0.0,["language"]=Profile.Language,["ok"]=true };

        var arr = returns.OrderBy(x => x).ToList();
        int idx = Math.Max(0, Math.Min((int)((1.0 - conf) * arr.Count), arr.Count - 1));
        double varPct  = -arr[idx] * 100.0;
        double cvarSum = arr.Take(idx + 1).Sum();
        double cvarPct = -(cvarSum / (idx + 1)) * 100.0;

        return new Dictionary<string, object?>
        {
            ["var_pct"]  = varPct,
            ["cvar_pct"] = cvarPct,
            ["language"] = Profile.Language,
            ["ok"]       = true
        };
    }

    public static Dictionary<string, object?> SkewEstimate(JsonElement d)
    {
        var returns = Helpers.GetNumArr(d, "returns");
        if (returns.Count < 3)
            return new Dictionary<string, object?> { ["skew"]=0.0,["language"]=Profile.Language,["ok"]=true };

        double fn   = returns.Count;
        double mean = returns.Average();
        double var_ = returns.Sum(x => (x-mean)*(x-mean)) / fn;
        double std  = Math.Sqrt(var_);
        double skew = std == 0 ? 0.0 : returns.Sum(x => Math.Pow(x - mean, 3)) / (fn * std * std * std);

        return new Dictionary<string, object?>
        {
            ["skew"]     = skew,
            ["language"] = Profile.Language,
            ["ok"]       = true
        };
    }

    public static Dictionary<string, object?> OrderBookImbalanceSeries(JsonElement d)
    {
        var snapshots = Helpers.GetArr(d, "snapshots");
        var series    = snapshots.Select(snap =>
        {
            var bids = Helpers.GetNumArr(snap, "bids");
            var asks = Helpers.GetNumArr(snap, "asks");
            double sb = bids.Take(5).Sum(), sa = asks.Take(5).Sum();
            return (sb + sa) == 0 ? 0.0 : (sb - sa) / (sb + sa);
        }).ToList();

        string trend = series.Count < 2 ? "flat"
                     : series.Last() > series.First() + 0.05 ? "up"
                     : series.Last() < series.First() - 0.05 ? "down"
                     : "flat";

        return new Dictionary<string, object?>
        {
            ["imbalance_series"] = series,
            ["trend"]            = trend,
            ["language"]         = Profile.Language,
            ["ok"]               = true
        };
    }

    public static Dictionary<string, object?> ExecutionQualityScore(JsonElement d)
    {
        var trades = Helpers.GetArr(d, "trades");
        var slips  = trades.Select(t =>
        {
            double fill     = Helpers.GetNum(t, "fill_price", 0.0);
            double decision = Helpers.GetNum(t, "decision_price", 1.0);
            return decision == 0 ? 0.0 : Math.Abs(fill - decision) / decision * 10000.0;
        }).ToList();

        double avgSlip = slips.Count > 0 ? slips.Average() : 0.0;
        double score   = Math.Max(0.0, 1.0 - avgSlip / 50.0);

        return new Dictionary<string, object?>
        {
            ["score_0_1"]        = score,
            ["avg_slippage_bps"] = avgSlip,
            ["language"]         = Profile.Language,
            ["ok"]               = true
        };
    }

    public static Dictionary<string, object?> RegimeDuration(JsonElement d)
    {
        var history = Helpers.GetArr(d, "regime_history");
        var prices  = Helpers.GetNumArr(d, "prices");
        int bars    = history.Count > 0 ? history.Count : Math.Min(10, prices.Count);
        string regime = history.Count > 0 && history[0].ValueKind == JsonValueKind.String
                      ? history[0].GetString() ?? "unknown"
                      : Helpers.GetStr(d, "regime", "unknown");
        bool stable = bars >= 5;

        return new Dictionary<string, object?>
        {
            ["bars_in_regime"] = (double)bars,
            ["regime_stable"]  = stable,
            ["regime"]         = regime,
            ["language"]       = Profile.Language,
            ["ok"]             = true
        };
    }
}
