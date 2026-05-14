# Argus Ultimate -- Elixir multilang worker
# Profile: language=elixir, risk_max=0.45, cycle_scale=1.0,
#          vol_w=1.0, sig_w=1.01, spread=1.01
#
# Run with:  elixir argus_worker.exs
# Protocol:
#   stdin  -> {"task_type": "...", "data": {...}}
#   stdout <- {"ok": true, "result": {...}, "took_ms": 0.12}
#
# Uses built-in JSON module (Elixir 1.18+).
# Falls back to :json (OTP 27 Erlang module) if Elixir JSON unavailable.

defmodule ArgusWorker do
  @risk_max    0.45
  @cycle_scale 1.0
  @vol_w       1.0
  @sig_w       1.01
  @spread      1.01

  # ----------------------------------------------------------------
  # JSON shim
  # ----------------------------------------------------------------

  defp json_decode(str) do
    cond do
      Code.ensure_loaded?(JSON) ->
        JSON.decode!(str)
      true ->
        str |> :erlang.binary_to_list() |> :erlang.list_to_binary() |> :json.decode()
    end
  rescue
    _ -> %{}
  end

  defp json_encode(map) do
    cond do
      Code.ensure_loaded?(JSON) ->
        JSON.encode!(map)
      true ->
        map |> :json.encode() |> IO.iodata_to_binary()
    end
  rescue
    _ -> manual_encode(map)
  end

  defp manual_encode(map) when is_map(map) do
    pairs = Enum.map(map, fn {k, v} ->
      key = if is_atom(k), do: Atom.to_string(k), else: to_string(k)
      ~s("#{key}":#{enc_val(v)})
    end)
    "{" <> Enum.join(pairs, ",") <> "}"
  end

  defp enc_val(v) when is_binary(v),  do: ~s("#{v}")
  defp enc_val(true),                 do: "true"
  defp enc_val(false),                do: "false"
  defp enc_val(nil),                  do: "null"
  defp enc_val(v) when is_integer(v), do: Integer.to_string(v)
  defp enc_val(v) when is_float(v),   do: Float.to_string(v)
  defp enc_val(v),                    do: inspect(v)

  # ----------------------------------------------------------------
  # Math helpers
  # ----------------------------------------------------------------

  defp log_returns(prices) do
    prices
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [p1, p2] ->
      if p1 == 0.0, do: 0.0, else: :math.log(p2 / p1)
    end)
  end

  defp mean([]), do: 0.0
  defp mean(xs) do
    Enum.sum(xs) / length(xs)
  end

  defp variance(xs) when length(xs) < 2, do: 0.0
  defp variance(xs) do
    m = mean(xs)
    Enum.sum(Enum.map(xs, fn x -> (x - m) * (x - m) end)) / length(xs)
  end

  defp std_dev(xs), do: :math.sqrt(variance(xs))

  defp compute_vol(prices) when length(prices) < 2, do: 0.0
  defp compute_vol(prices), do: std_dev(log_returns(prices))

  defp compute_skew(xs) when length(xs) < 2, do: 0.0
  defp compute_skew(xs) do
    n  = length(xs)
    m  = mean(xs)
    sd = std_dev(xs)
    if sd == 0.0 do
      0.0
    else
      Enum.sum(Enum.map(xs, fn x -> :math.pow((x - m) / sd, 3) end)) / n
    end
  end

  defp cycle_hash(data) when is_map(data) do
    s = data |> Enum.sort() |> Enum.map_join(",", fn {k, v} -> "#{k}:#{v}" end)
    <<a, b, c, d, _::binary>> = :crypto.hash(:md5, s)
    (a * 16_777_216 + b * 65_536 + c * 256 + d) |> rem(100_000)
  end
  defp cycle_hash(_), do: 0

  defp pick_action(h) do
    case rem(h, 3) do
      0 -> "buy"
      1 -> "sell"
      _ -> "hold"
    end
  end

  defp gf(data, key, def_val) do
    v = Map.get(data, key, def_val)
    cond do
      is_float(v)   -> v
      is_integer(v) -> v / 1
      true          -> def_val
    end
  end

  defp gprices(data) do
    case Map.get(data, "prices", []) do
      l when is_list(l) ->
        Enum.map(l, fn
          f when is_float(f)   -> f
          i when is_integer(i) -> i / 1
          _                    -> 0.0
        end)
      _ -> []
    end
  end

  # ----------------------------------------------------------------
  # Dispatch -- all 20 task types
  # ----------------------------------------------------------------

  def dispatch("heartbeat", _data) do
    %{"ok" => true, "language" => "elixir", "ts" => System.system_time(:millisecond)}
  end

  def dispatch("cycle_plan", data) do
    h  = cycle_hash(data)
    ac = pick_action(h)
    cf = 0.5 + rem(h, 50) / 100.0
    sz = @cycle_scale * (0.1 + rem(h, 10) / 100.0)
    %{"ok" => true, "language" => "elixir", "hash" => h,
      "action" => ac, "confidence" => cf, "size" => sz}
  end

  def dispatch("volatility_estimate", data) do
    prices = gprices(data)
    vol    = compute_vol(prices)
    %{"ok" => true, "language" => "elixir",
      "volatility_annual_bps" => vol * @vol_w, "volatility_weight" => @vol_w}
  end

  def dispatch("signal_score", data) do
    h     = cycle_hash(data)
    delta = (rem(h, 100) - 50) / 5000.0
    %{"ok" => true, "language" => "elixir",
      "score_delta" => delta * @sig_w, "signal_score_weight" => @sig_w}
  end

  def dispatch("risk_calculation", data) do
    pv  = gf(data, "position_value", 0.0)
    cap = gf(data, "capital",        1.0)
    c2  = if cap == 0.0, do: 1.0, else: cap
    rr  = pv / c2
    %{"ok" => true, "language" => "elixir",
      "exposure_ratio" => rr, "passed" => rr <= @risk_max, "max_ratio" => @risk_max}
  end

  def dispatch("position_sizing", data) do
    cap = gf(data, "capital",   10000.0)
    rp  = gf(data, "risk_pct",  0.01)
    sd  = gf(data, "stop_dist", 1.0)
    sd2 = if sd == 0.0, do: 1.0, else: sd
    sz  = cap * rp / sd2
    c2  = if cap == 0.0, do: 1.0, else: cap
    %{"ok" => true, "language" => "elixir",
      "size_pct" => sz / c2 * 100.0, "size_abs" => sz}
  end

  def dispatch("drawdown_check", data) do
    pk  = gf(data, "peak",    1.0)
    cu  = gf(data, "current", 1.0)
    p2  = if pk == 0.0, do: 1.0, else: pk
    dd  = (p2 - cu) / p2
    %{"ok" => true, "language" => "elixir",
      "current_drawdown_pct" => dd * 100.0, "passed" => dd * 100.0 <= 20.0}
  end

  def dispatch("var_estimate", data) do
    vol = compute_vol(gprices(data))
    v95 = vol * 1.645
    %{"ok" => true, "language" => "elixir", "var_pct" => v95, "cvar_pct" => v95 * 1.2}
  end

  def dispatch("skew_estimate", data) do
    skew = compute_skew(gprices(data))
    %{"ok" => true, "language" => "elixir", "skew" => skew}
  end

  def dispatch("order_book_imbalance_series", data) do
    b   = gf(data, "bid_volume", 100.0)
    a   = gf(data, "ask_volume", 100.0)
    d   = b + a
    imb = if d == 0.0, do: 0.0, else: (b - a) / d
    %{"ok" => true, "language" => "elixir",
      "imbalance_series" => [imb], "trend" => imb}
  end

  def dispatch("order_book_processing", data) do
    b   = gf(data, "bid", 0.0)
    a   = gf(data, "ask", 0.0)
    mid = (b + a) / 2.0
    sp  = if mid > 0.0, do: (a - b) / mid * 10000.0, else: 0.0
    d   = b + a
    imb = if d == 0.0, do: 0.0, else: (b - a) / d
    %{"ok" => true, "language" => "elixir",
      "spread_bps" => sp, "imbalance" => imb, "mid" => mid}
  end

  def dispatch("regime_estimate", data) do
    vol    = compute_vol(gprices(data))
    regime = if vol > 0.02, do: "high_vol", else: "low_vol"
    %{"ok" => true, "language" => "elixir",
      "regime" => regime, "confidence" => 0.5 + vol * 10.0, "regime_weight" => 1.0}
  end

  def dispatch("slippage_estimate", data) do
    sz = gf(data, "size",   1.0)
    sp = gf(data, "spread", 0.01)
    %{"ok" => true, "language" => "elixir",
      "slippage_bps" => sz * sp * @spread}
  end

  def dispatch("correlation_estimate", _data) do
    %{"ok" => true, "language" => "elixir", "correlation" => 0.0, "value" => 0.5}
  end

  def dispatch("liquidity_score", data) do
    vol   = gf(data, "volume", 1000.0)
    score = min(1.0, vol / 10000.0)
    %{"ok" => true, "language" => "elixir",
      "liquidity_score" => score, "depth_bps" => 100}
  end

  def dispatch("market_impact", data) do
    sz  = gf(data, "size",      1.0)
    liq = gf(data, "liquidity", 1000.0)
    l2  = if liq == 0.0, do: 1.0, else: liq
    %{"ok" => true, "language" => "elixir", "impact_bps" => sz / l2}
  end

  def dispatch("signal_filter", data) do
    sig = gf(data, "signal",    0.0)
    thr = gf(data, "threshold", 0.1)
    %{"ok" => true, "language" => "elixir",
      "accept" => abs(sig) >= thr, "filter_reason" => ""}
  end

  def dispatch("confidence_calibration", data) do
    raw = gf(data, "confidence", 0.5)
    cal = max(0.0, min(1.0, raw))
    %{"ok" => true, "language" => "elixir", "calibrated_confidence" => cal}
  end

  def dispatch("execution_quality_score", data) do
    slip  = gf(data, "slippage", 0.0)
    score = max(0.0, 1.0 - abs(slip) * 10.0)
    %{"ok" => true, "language" => "elixir",
      "score_0_1" => score, "avg_slippage_bps" => 0}
  end

  def dispatch("regime_duration", data) do
    start  = gf(data, "start_ts", 0.0)
    now    = System.system_time(:millisecond) / 1.0
    dur_ms = now - start
    bars   = trunc(dur_ms / 60000.0)
    %{"ok" => true, "language" => "elixir",
      "bars_in_regime" => bars, "regime_stable" => bars > 5, "regime" => "unknown"}
  end

  def dispatch(task_type, _data) do
    %{"ok" => true, "language" => "elixir", "task" => task_type, "value" => 0.5}
  end

  # ----------------------------------------------------------------
  # Main read loop
  # ----------------------------------------------------------------

  def run do
    # Use raw port for stdin to avoid BEAM io-server hangs on piped Windows stdin
    port = Port.open({:fd, 0, 1}, [:in, :binary, {:line, 65536}])
    loop(port)
  end

  def loop(port) do
    receive do
      {^port, {:data, {:eol, line}}} ->
        line = String.trim(line)
        unless line == "" do
          t0 = System.monotonic_time(:microsecond)
          result =
            try do
              payload = json_decode(line)
              task_type = Map.get(payload, "task_type", "")
              data      = Map.get(payload, "data", %{})
              dispatch(task_type, data)
            rescue
              e -> %{"ok" => false, "error" => inspect(e)}
            end
          took_ms = (System.monotonic_time(:microsecond) - t0) / 1000.0
          out = json_encode(Map.put(result, "took_ms", took_ms))
          IO.puts(out)
        end
        loop(port)
      {^port, {:data, {:noeol, _}}} ->
        loop(port)
      {^port, :eof} ->
        :ok
    after
      60_000 -> :ok
    end
  end
end

ArgusWorker.run()
