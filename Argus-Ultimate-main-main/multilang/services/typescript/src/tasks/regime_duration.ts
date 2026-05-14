const LANGUAGE = "typescript";

export default function regimeDuration(data: Record<string, unknown>): Record<string, unknown> {
  const prices = (data["prices"] as number[] | undefined) ?? [];
  const regimeHistory = (data["regime_history"] as unknown[] | undefined) ?? [];

  if (prices.length < 2) return { bars_in_regime: 0, regime_stable: false, language: LANGUAGE, ok: true };

  const n = prices.length - 1;
  let sumSq = 0.0;
  for (let i = 0; i < n; i++) {
    const prev = prices[i] as number;
    if (prev !== 0) { const r = ((prices[i + 1] as number) - prev) / prev; sumSq += r * r; }
  }
  const vol = n > 0 ? Math.sqrt(sumSq / n * 252 * 1e4) : 10.0;
  const regime = vol > 20.0 ? "high_vol" : "mean_revert";
  const bars = regimeHistory.length > 0 ? regimeHistory.length : Math.min(10, prices.length);

  return { bars_in_regime: bars, regime_stable: bars >= 5, regime, language: LANGUAGE, ok: true };
}
