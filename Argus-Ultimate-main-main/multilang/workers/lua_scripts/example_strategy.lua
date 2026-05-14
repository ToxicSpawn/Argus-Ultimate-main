-- ARGUS Example Strategy: SMA Golden/Death Cross
--
-- Demonstrates the Lua strategy scripting interface.
-- Uses built-in sma() and crossover()/crossunder() indicators.

function evaluate(data)
    local fast_sma = sma(data.close, 10)
    local slow_sma = sma(data.close, 50)

    if crossover(fast_sma, slow_sma) then
        return {action = "BUY", confidence = 0.7, reason = "golden cross"}
    elseif crossunder(fast_sma, slow_sma) then
        return {action = "SELL", confidence = 0.7, reason = "death cross"}
    end

    return {action = "HOLD", confidence = 0.5, reason = "no signal"}
end
