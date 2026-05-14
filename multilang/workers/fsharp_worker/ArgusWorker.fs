// ArgusWorker.fs — Argus F# stdin/stdout worker
// Build: dotnet publish -c Release -o bin/fsharp
// Protocol: one JSON line in -> one JSON line out, loop until EOF

open System
open System.IO
open System.Security.Cryptography
open System.Text
open System.Text.Json
open System.Globalization

let LANGUAGE = "fsharp"
let RISK_MAX = 0.42
let CYCLE_SCALE = 0.97
let VOL_W = 1.05
let SIG_W = 1.02
let SPREAD = 1.02

// ── Helpers ─────────────────────────────────────────────────────────────

let sha256Long (s: string) =
    use sha = SHA256.Create()
    let h = sha.ComputeHash(Encoding.UTF8.GetBytes(s))
    let mutable v = 0L
    for i in 0..7 do
        v <- (v <<< 8) ||| (int64 (h.[i] &&& 0xFFuy))
    abs v

let fmtD (d: float) =
    if d = float (int64 d) then (int64 d).ToString()
    else d.ToString("G8", CultureInfo.InvariantCulture)

let esc (s: string) =
    if isNull s then ""
    else s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n")

let getFloat (elem: JsonElement) (key: string) (def': float) =
    match elem.TryGetProperty(key) with
    | true, v ->
        match v.ValueKind with
        | JsonValueKind.Number -> v.GetDouble()
        | JsonValueKind.String ->
            match Double.TryParse(v.GetString(), NumberStyles.Any, CultureInfo.InvariantCulture) with
            | true, d -> d
            | _ -> def'
        | _ -> def'
    | _ -> def'

let getStr (elem: JsonElement) (key: string) (def': string) =
    match elem.TryGetProperty(key) with
    | true, v -> v.GetString() |> (fun s -> if isNull s then def' else s)
    | _ -> def'

let getFloatList (elem: JsonElement) (key: string) =
    match elem.TryGetProperty(key) with
    | true, v when v.ValueKind = JsonValueKind.Array ->
        [| for item in v.EnumerateArray() do
            match item.ValueKind with
            | JsonValueKind.Number -> yield item.GetDouble()
            | _ -> () |]
    | _ -> [||]

// ── JSON output ─────────────────────────────────────────────────────────

let jsonVal (v: obj) =
    match v with
    | null -> "null"
    | :? bool as b -> if b then "true" else "false"
    | :? int as i -> i.ToString()
    | :? int64 as l -> l.ToString()
    | :? float as d -> fmtD d
    | :? string as s -> sprintf "\"%s\"" (esc s)
    | :? (float array) as arr ->
        arr |> Array.map fmtD |> String.concat "," |> sprintf "[%s]"
    | _ -> sprintf "\"%s\"" (esc (v.ToString()))

let toJson (pairs: (string * obj) list) =
    pairs
    |> List.map (fun (k, v) -> sprintf "\"%s\":%s" (esc k) (jsonVal v))
    |> String.concat ","
    |> sprintf "{%s}"

// ── Tasks ───────────────────────────────────────────────────────────────

let cyclePlan (data: JsonElement) =
    let sorted =
        [ for prop in data.EnumerateObject() do
            yield sprintf "%s:%s" prop.Name (prop.Value.GetRawText()) ]
        |> List.sort |> String.concat ","
    let h = sha256Long sorted
    let base' = (float (h % 200L) - 100.0) / 10000.0
    let capital = getFloat data "capital" (getFloat data "portfolio_value_aud" 1000.0)
    let actions = [| "buy"; "sell"; "hold" |]
    let action = actions.[int (h % 3L)]
    let sizeAud = capital * 0.02 * CYCLE_SCALE * (min 1.0 (abs base' * 50.0))
    [ "action", box action; "confidence", box (0.5 + base')
      "size_aud", box (max 0.0 sizeAud); "spread_multiplier", box SPREAD
      "language", box LANGUAGE ]

let heartbeat (_: JsonElement) =
    [ "ok", box true; "language", box LANGUAGE
      "ts", box (DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()) ]

let volatilityEstimate (data: JsonElement) =
    let prices = getFloatList data "prices"
    let vol =
        if prices.Length < 2 then 1000.0
        else
            let lr = Array.init (prices.Length - 1) (fun i -> log (prices.[i+1] / prices.[i]))
            let mean = Array.average lr
            let var' = lr |> Array.sumBy (fun x -> (x - mean) ** 2.0) |> fun s -> s / float (max 1 (lr.Length - 1))
            sqrt var' * sqrt 365.0 * 10000.0 * VOL_W
    [ "volatility_annual_bps", box vol; "volatility_weight", box VOL_W
      "n", box prices.Length; "language", box LANGUAGE ]

let signalScore (data: JsonElement) =
    let signal = getFloat data "signal" 0.5
    let conf = getFloat data "confidence" 0.5
    let regime = getStr data "regime" "unknown"
    let sorted = sprintf "%f:%f:%s" signal conf regime
    let h = abs (int64 (sorted.GetHashCode()))
    let scoreDelta = (float (h % 100L) - 50.0) / 5000.0 * SIG_W
    [ "score_delta", box scoreDelta; "confidence", box conf
      "regime", box regime; "signal_score_weight", box SIG_W; "language", box LANGUAGE ]

let riskCalculation (data: JsonElement) =
    let pv = getFloat data "position_value" 0.0
    let cap = getFloat data "capital" 1000.0
    let ratio = if cap > 0.0 then pv / cap else 0.0
    [ "exposure_ratio", box ratio; "passed", box (ratio <= RISK_MAX)
      "max_ratio", box RISK_MAX; "language", box LANGUAGE ]

let positionSizing (data: JsonElement) =
    let cap = getFloat data "capital" 1000.0
    let vol = getFloat data "volatility_bps" 1000.0
    let conf = getFloat data "confidence" 0.5
    let size = max 10.0 (min (cap * RISK_MAX) (cap * 0.02 * (1.0 - vol / 20000.0) * conf))
    let sizePct = if cap > 0.0 then size / cap * 100.0 else 0.0
    [ "size_pct", box sizePct; "size_abs", box size; "language", box LANGUAGE ]

let drawdownCheck (data: JsonElement) =
    let cur = getFloat data "current_equity" 1000.0
    let peak = getFloat data "peak_equity" 1000.0
    let maxDd = getFloat data "max_drawdown_pct" 20.0
    let dd = if peak > 0.0 then (peak - cur) / peak else 0.0
    [ "current_drawdown_pct", box (dd * 100.0); "passed", box (dd * 100.0 <= maxDd); "language", box LANGUAGE ]

let regimeEstimate (data: JsonElement) =
    let prices = getFloatList data "prices"
    if prices.Length < 5 then
        [ "regime", box "unknown"; "confidence", box 0.5; "regime_weight", box 1.0; "language", box LANGUAGE ]
    else
        let change = (prices.[prices.Length - 1] - prices.[0]) / prices.[0]
        let regime, conf =
            if change > 0.02 then "trending_up", 0.6 + min 0.3 change
            elif change < -0.02 then "trending_down", 0.6 + min 0.3 (abs change)
            else "mean_reverting", 0.55
        [ "regime", box regime; "confidence", box conf; "regime_weight", box 1.0; "language", box LANGUAGE ]

let slippageEstimate (data: JsonElement) =
    let spread = getFloat data "spread_bps" 10.0
    let vol = getFloat data "volatility_bps" 1000.0
    let size = getFloat data "order_size_aud" 100.0
    let slip = (spread * 0.5 + vol * 0.001 + size * 0.0001) * SPREAD
    [ "slippage_bps", box slip; "language", box LANGUAGE ]

let correlationEstimate (data: JsonElement) =
    let xs = getFloatList data "series_a"
    let ys = getFloatList data "series_b"
    let n = min xs.Length ys.Length
    let corr =
        if n < 3 then 0.0
        else
            let mx = Array.take n xs |> Array.average
            let my = Array.take n ys |> Array.average
            let mutable cov = 0.0
            let mutable sx = 0.0
            let mutable sy = 0.0
            for i in 0 .. n - 1 do
                let dx = xs.[i] - mx
                let dy = ys.[i] - my
                cov <- cov + dx * dy; sx <- sx + dx * dx; sy <- sy + dy * dy
            let denom = sqrt (sx * sy)
            if denom > 0.0 then cov / denom else 0.0
    [ "correlation", box corr; "n", box n; "language", box LANGUAGE ]

let liquidityScore (data: JsonElement) =
    let vol24h = getFloat data "volume_24h" 1e6
    let spread = getFloat data "spread_bps" 10.0
    let depth = getFloat data "order_book_depth" 100.0
    let score = min 1.0 ((vol24h / 1e7) * 0.4 + (1.0 / (1.0 + spread / 100.0)) * 0.3 + (depth / 1000.0) * 0.3)
    let depthBps = depth * spread / 100.0
    [ "liquidity_score", box score; "depth_bps", box depthBps; "language", box LANGUAGE ]

let marketImpact (data: JsonElement) =
    let size = getFloat data "order_size_aud" 100.0
    let vol = getFloat data "volume_24h" 1e6
    let impact = if vol > 0.0 then sqrt (size / vol) * 100.0 else 10.0
    [ "impact_bps", box impact; "language", box LANGUAGE ]

let orderBookProcessing (data: JsonElement) =
    let bids = getFloatList data "bids"
    let asks = getFloatList data "asks"
    let bb = if bids.Length > 0 then bids.[0] else 0.0
    let ba = if asks.Length > 0 then asks.[0] else 0.0
    let mid = (bb + ba) / 2.0
    let spread = if mid > 0.0 then (ba - bb) / mid * 10000.0 else 0.0
    let imb =
        if bids.Length > 0 && asks.Length > 0 then
            let bs = Array.sum bids
            let as' = Array.sum asks
            if bs + as' > 0.0 then (bs - as') / (bs + as') else 0.0
        else 0.0
    [ "mid", box mid; "spread_bps", box spread; "imbalance", box imb; "language", box LANGUAGE ]

let signalFilter (data: JsonElement) =
    let signal = getFloat data "signal" 0.0
    let conf = getFloat data "confidence" 0.5
    let accept = conf >= 0.6
    [ "accept", box accept; "filtered_signal", box (if accept then signal else 0.0)
      "filter_reason", box (if accept then "confidence_ok" else "low_confidence"); "language", box LANGUAGE ]

let confidenceCalibration (data: JsonElement) =
    let raw = getFloat data "raw_confidence" 0.5
    let cal = 1.0 / (1.0 + exp (-(raw - 0.5) * 4.0))
    [ "calibrated_confidence", box cal; "raw_confidence", box raw; "language", box LANGUAGE ]

let varEstimate (data: JsonElement) =
    let returns = getFloatList data "returns"
    let pctile = getFloat data "percentile" 5.0
    let v =
        if returns.Length = 0 then 0.0
        else
            let s = returns |> Array.sort
            let idx = max 0 (min (int (pctile / 100.0 * float s.Length)) (s.Length - 1))
            s.[idx]
    let cvar = if returns.Length = 0 then 0.0
               else
                   let s = returns |> Array.sort
                   let cutoff = max 1 (int (pctile / 100.0 * float s.Length))
                   Array.take cutoff s |> Array.average
    [ "var_pct", box v; "cvar_pct", box cvar; "percentile", box pctile; "n", box returns.Length; "language", box LANGUAGE ]

let skewEstimate (data: JsonElement) =
    let returns = getFloatList data "returns"
    let skew =
        if returns.Length < 3 then 0.0
        else
            let mean = Array.average returns
            let m2 = returns |> Array.sumBy (fun x -> (x - mean) ** 2.0) |> fun s -> s / float returns.Length
            let m3 = returns |> Array.sumBy (fun x -> (x - mean) ** 3.0) |> fun s -> s / float returns.Length
            let sd = sqrt m2
            if sd > 0.0 then m3 / (sd * sd * sd) else 0.0
    [ "skew", box skew; "n", box returns.Length; "language", box LANGUAGE ]

let orderBookImbalanceSeries (data: JsonElement) =
    let bv = getFloatList data "bid_volumes"
    let av = getFloatList data "ask_volumes"
    let n = min bv.Length av.Length
    let imbs = [| for i in 0 .. n - 1 do
                    let t = bv.[i] + av.[i]
                    yield if t > 0.0 then (bv.[i] - av.[i]) / t else 0.0 |]
    let avg = if imbs.Length > 0 then Array.average imbs else 0.0
    [ "imbalance_series", box imbs; "trend", box avg; "n", box n; "language", box LANGUAGE ]

let executionQualityScore (data: JsonElement) =
    let exp' = getFloat data "expected_price" 100.0
    let exec = getFloat data "executed_price" 100.0
    let slip = if exp' > 0.0 then abs (exec - exp') / exp' * 10000.0 else 0.0
    [ "score_0_1", box (max 0.0 (1.0 - slip / 100.0)); "avg_slippage_bps", box slip; "language", box LANGUAGE ]

let regimeDuration (data: JsonElement) =
    let prices = getFloatList data "prices"
    if prices.Length < 2 then
        [ "regime", box "unknown"; "bars_in_regime", box 0; "regime_stable", box false; "language", box LANGUAGE ]
    else
        let up = prices.[prices.Length - 1] > prices.[prices.Length - 2]
        let mutable dur = 1
        let mutable i = prices.Length - 2
        while i >= 1 && (prices.[i] > prices.[i - 1]) = up do
            dur <- dur + 1
            i <- i - 1
        let regime = if up then "trending_up" else "trending_down"
        [ "regime", box regime; "bars_in_regime", box dur; "regime_stable", box (dur > 10); "language", box LANGUAGE ]

// ── Dispatch ────────────────────────────────────────────────────────────

let dispatch (taskType: string) (data: JsonElement) =
    match taskType with
    | "cycle_plan" -> cyclePlan data
    | "heartbeat" -> heartbeat data
    | "volatility_estimate" -> volatilityEstimate data
    | "signal_score" -> signalScore data
    | "risk_calculation" -> riskCalculation data
    | "position_sizing" -> positionSizing data
    | "drawdown_check" -> drawdownCheck data
    | "regime_estimate" -> regimeEstimate data
    | "slippage_estimate" -> slippageEstimate data
    | "correlation_estimate" -> correlationEstimate data
    | "liquidity_score" -> liquidityScore data
    | "market_impact" -> marketImpact data
    | "order_book_processing" -> orderBookProcessing data
    | "signal_filter" -> signalFilter data
    | "confidence_calibration" -> confidenceCalibration data
    | "var_estimate" -> varEstimate data
    | "skew_estimate" -> skewEstimate data
    | "order_book_imbalance_series" -> orderBookImbalanceSeries data
    | "execution_quality_score" -> executionQualityScore data
    | "regime_duration" -> regimeDuration data
    | other -> [ "language", box LANGUAGE; "task_type", box other; "value", box 0.5 ]

// ── Main loop ───────────────────────────────────────────────────────────

[<EntryPoint>]
let main _argv =
    let reader = new StreamReader(Console.OpenStandardInput(), Encoding.UTF8)
    let writer = new StreamWriter(Console.OpenStandardOutput(), Encoding.UTF8)
    writer.AutoFlush <- true
    let mutable running = true
    while running do
        let line = reader.ReadLine()
        if isNull line then
            running <- false
        else
            let trimmed = line.Trim()
            if trimmed.Length > 0 then
                let sw = System.Diagnostics.Stopwatch.StartNew()
                try
                    use doc = JsonDocument.Parse(trimmed)
                    let root = doc.RootElement
                    let taskType = getStr root "task_type" "heartbeat"
                    let data =
                        match root.TryGetProperty("data") with
                        | true, v -> v
                        | _ -> root
                    let result = dispatch taskType data
                    sw.Stop()
                    let tookMs = float sw.ElapsedTicks / float System.Diagnostics.Stopwatch.Frequency * 1000.0
                    writer.WriteLine(sprintf "{\"ok\":true,\"result\":%s,\"took_ms\":%s}" (toJson result) (fmtD tookMs))
                with
                | ex ->
                    sw.Stop()
                    let tookMs = float sw.ElapsedTicks / float System.Diagnostics.Stopwatch.Frequency * 1000.0
                    writer.WriteLine(sprintf "{\"ok\":false,\"error\":\"%s\",\"took_ms\":%s}" (esc ex.Message) (fmtD tookMs))
            else
                ()
    0
