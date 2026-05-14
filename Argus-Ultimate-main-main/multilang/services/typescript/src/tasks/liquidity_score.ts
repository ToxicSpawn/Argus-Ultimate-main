const LANGUAGE = "typescript";

export default function liquidityScore(data: Record<string, unknown>): Record<string, unknown> {
  const bids = (data["bids"] as [number, number][] | undefined) ?? [];
  const asks = (data["asks"] as [number, number][] | undefined) ?? [];
  const depthLevels = (Number(data["depth_levels"]) | 0) || 5;

  if (bids.length === 0 && asks.length === 0) return { liquidity_score: 0.0, depth_bps: 0.0, language: LANGUAGE, ok: true };

  const bestBid = bids.length > 0 ? Number(bids[0]![0]) : 0.0;
  const bestAsk = asks.length > 0 ? Number(asks[0]![0]) : 0.0;
  const mid = bestBid > 0 && bestAsk > 0 ? (bestBid + bestAsk) / 2.0 : 0.0;
  const depthBps = mid > 0 ? ((bestAsk - bestBid) / mid) * 1e4 : 100.0;

  let totalVol = 0.0;
  for (let i = 0; i < Math.min(bids.length, depthLevels); i++) totalVol += Number(bids[i]![1]);
  for (let i = 0; i < Math.min(asks.length, depthLevels); i++) totalVol += Number(asks[i]![1]);
  const score = totalVol > 0 ? Math.min(1.0, totalVol / 100.0) : 0.0;

  return { liquidity_score: score, depth_bps: depthBps, language: LANGUAGE, ok: true };
}
