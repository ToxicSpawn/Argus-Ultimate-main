import PROFILE from "../profile.js";

const LANGUAGE = "typescript";

export default function regimeEstimate(data: Record<string, unknown>): Record<string, unknown> {
  const prices = ((data["prices"] ?? data["returns"]) as number[] | undefined) ?? [];
  let regime = "mean_revert";
  let confidence = 0.5;

  if (prices.length >= 3) {
    let sumSq = 0.0;
    for (let i = 1; i < prices.length; i++) {
      const prev = prices[i - 1] as number;
      if (prev !== 0) { const r = ((prices[i] as number) - prev) / prev; sumSq += r * r; }
    }
    const n = prices.length - 1;
    const vol = n > 0 ? Math.sqrt(sumSq / n * 252 * 1e4) : 10.0;
    const trend = (prices[0] as number) !== 0 ? ((prices[prices.length - 1] as number) - (prices[0] as number)) / (prices[0] as number) : 0.0;
    regime = vol > 20.0 ? "high_vol" : Math.abs(trend) > 0.02 ? "trend" : "mean_revert";
    confidence = Math.min(0.95, 0.5 + Math.abs(trend) * 5 + vol / 100);
  }

  return { regime, confidence, language: LANGUAGE, regime_weight: PROFILE.regime_weight, ok: true };
}
