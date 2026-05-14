import PROFILE from "../profile.js";

const LANGUAGE = "typescript";
const WEIGHT = PROFILE.volatility_weight;
const SEED: number = (() => { let s = 0; for (let i = 0; i < LANGUAGE.length; i++) s += LANGUAGE.charCodeAt(i); return s % 7; })();

function welfordVariance(arr: number[]): number {
  const n = arr.length;
  if (n === 0) return 0.0;
  let mean = 0.0, m2 = 0.0;
  for (let i = 0; i < n; i++) {
    const x = arr[i] as number;
    const delta = x - mean;
    mean += delta / (i + 1);
    m2 += delta * (x - mean);
  }
  return m2 / n;
}

export default function volatilityEstimate(data: Record<string, unknown>): Record<string, unknown> {
  const prices = (data["prices"] ?? data["ohlcv_close"]) as number[] | undefined;
  const returns = data["returns"] as number[] | undefined;
  let vol: number;

  if (returns && returns.length > 0) {
    const variance = welfordVariance(returns);
    vol = variance > 0 ? Math.sqrt(variance * 252 * 1e4) : 10.0;
  } else if (prices && prices.length >= 2) {
    const rets: number[] = [];
    for (let i = 1; i < prices.length; i++) {
      const prev = prices[i - 1] as number;
      if (prev !== 0) rets.push(((prices[i] as number) - prev) / prev);
    }
    const variance = welfordVariance(rets);
    vol = variance > 0 ? Math.sqrt(variance * 252 * 1e4) : 10.0;
  } else {
    vol = 10.0;
  }

  const volAdj = vol * (1.0 + (SEED - 3) * 0.01) * WEIGHT;
  return { volatility_annual_bps: volAdj, volatility_weight: WEIGHT, language: LANGUAGE, ok: true };
}
