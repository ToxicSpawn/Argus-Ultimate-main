const LANGUAGE = "typescript";
const DEPTH = 5;

export default function orderBookImbalanceSeries(data: Record<string, unknown>): Record<string, unknown> {
  const bids = (data["bids"] as [number, number][] | undefined) ?? [];
  const asks = (data["asks"] as [number, number][] | undefined) ?? [];

  if (bids.length === 0 && asks.length === 0) return { imbalance_series: [], trend: 0.0, language: LANGUAGE, ok: true };

  let bidVol = 0.0, askVol = 0.0;
  for (let i = 0; i < Math.min(bids.length, DEPTH); i++) bidVol += Number(bids[i]![1]);
  for (let i = 0; i < Math.min(asks.length, DEPTH); i++) askVol += Number(asks[i]![1]);
  const total = bidVol + askVol;
  const imb = total > 0 ? (bidVol - askVol) / total : 0.0;

  return { imbalance_series: [imb], trend: imb, language: LANGUAGE, ok: true };
}
