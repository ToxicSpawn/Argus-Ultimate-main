import PROFILE from "../profile.js";

const LANGUAGE = "typescript";

export default function orderBookProcessing(data: Record<string, unknown>): Record<string, unknown> {
  const spreadMult = PROFILE.spread_mult;
  const bids = (data["bids"] as [number, number][] | undefined) ?? [];
  const asks = (data["asks"] as [number, number][] | undefined) ?? [];

  if (bids.length === 0 && asks.length === 0) {
    return { spread_bps: 0.0, imbalance: 0.0, mid: 0.0, language: LANGUAGE, spread_mult: spreadMult };
  }

  const bestBid = bids.length > 0 ? Number(bids[0]![0]) : 0.0;
  const bestAsk = asks.length > 0 ? Number(asks[0]![0]) : 0.0;
  const mid = bestBid > 0 && bestAsk > 0 ? (bestBid + bestAsk) / 2.0 : 0.0;
  const rawSpreadBps = mid > 0 ? ((bestAsk - bestBid) / mid) * 1e4 : 0.0;
  const spreadBps = rawSpreadBps * spreadMult;

  const depth = 5;
  let bidVol = 0.0, askVol = 0.0;
  for (let i = 0; i < Math.min(bids.length, depth); i++) bidVol += Number(bids[i]![1]);
  for (let i = 0; i < Math.min(asks.length, depth); i++) askVol += Number(asks[i]![1]);
  const total = bidVol + askVol;
  const imbalance = total > 0 ? (bidVol - askVol) / total : 0.0;

  return { spread_bps: spreadBps, imbalance, mid, language: LANGUAGE, spread_mult: spreadMult };
}
