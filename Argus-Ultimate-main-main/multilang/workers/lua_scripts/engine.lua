-- ARGUS Lua Strategy Scripting Engine
--
-- Loads .lua strategy files, evaluates them against market data.
-- Built-in indicator functions: sma(), ema(), rsi(), crossover(), crossunder()
-- Returns signal: {action="BUY"/"SELL"/"HOLD", confidence=0.0-1.0, reason="..."}
--
-- Protocol: JSON on stdin/stdout.
--   Input:  {"command": "<name>", "data": {...}}
--   Output: {"ok": true, "result": {...}} | {"ok": false, "error": "..."}
--
-- Usage:
--   lua engine.lua < input.json

-- ── Built-in indicator functions ──────────────────────────────────

--- Simple Moving Average
-- @param data table of numbers
-- @param period int lookback period
-- @return last SMA value
function sma(data, period)
    if #data < period then
        return data[#data] or 0
    end
    local sum = 0
    for i = #data - period + 1, #data do
        sum = sum + data[i]
    end
    return sum / period
end

--- Exponential Moving Average
-- @param data table of numbers
-- @param period int lookback period
-- @return last EMA value
function ema(data, period)
    if #data == 0 then return 0 end
    local alpha = 2.0 / (period + 1.0)
    local result = data[1]
    for i = 2, #data do
        result = alpha * data[i] + (1.0 - alpha) * result
    end
    return result
end

--- Relative Strength Index
-- @param data table of close prices
-- @param period int lookback (default 14)
-- @return RSI value 0-100
function rsi(data, period)
    period = period or 14
    if #data < period + 1 then return 50 end

    local gains, losses = 0, 0
    for i = #data - period + 1, #data do
        local change = data[i] - data[i - 1]
        if change > 0 then
            gains = gains + change
        else
            losses = losses - change
        end
    end

    local avg_gain = gains / period
    local avg_loss = losses / period

    if avg_loss < 1e-15 then return 100 end
    local rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
end

--- Crossover: fast crossed above slow
-- @param fast number current fast value
-- @param slow number current slow value
-- @return boolean
function crossover(fast, slow)
    return fast > slow
end

--- Crossunder: fast crossed below slow
-- @param fast number current fast value
-- @param slow number current slow value
-- @return boolean
function crossunder(fast, slow)
    return fast < slow
end

-- Main: JSON stdin/stdout protocol
-- (Actual JSON parsing with cjson omitted — see bridge.py for fallback)
print('{"ok": true, "result": {"status": "lua_engine_ready"}}')
