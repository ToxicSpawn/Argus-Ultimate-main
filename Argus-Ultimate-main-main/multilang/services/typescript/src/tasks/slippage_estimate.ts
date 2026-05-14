import PROFILE from "../profile.js";

const LANGUAGE = "typescript";

export default function slippageEstimate(data: Record<string, unknown>): Record<string, unknown> {
  const spreadMult = PROFILE.spread_mult;
  const ob = (data["order_book"] as Record<string, unknown> | undefined) ?? {};
  const bids = ((ob["bids"] ?? data["bids"]) as [number, number][] | undefined) ?? [];
  const asks = ((ob["asks"] ?? data["asks"]) as [number, number][] | undefined) ?? [];
  const participation = Number(data["participation_rate"] ?? 0.01);

  if (bids.length === 0 && asks.length === 0) return { slippage_bps: 0.0, language: LANGUAGE, ok: true };

  const bestBid = bids.length > 0 ? Number(bids[0]![0]) : 0.0;
  const bestAsk = asks.length > 0 ? Number(asks[0]![0]) : 0.0;
  const mid = bestBid > 0 && bestAsk > 0 ? (bestBid + bestAsk) / 2.0 : 0.0;
  const halfSpreadBps = mid > 0 ? ((bestAsk - bestBid) / mid) * 1e4 / 2 : 5.0;
  const slippageBps = halfSpreadBps * spreadMult * (1.0 + participation * 10);

  return { slippage_bps: slippageBps, language: LANGUAGE, ok: true };
}
