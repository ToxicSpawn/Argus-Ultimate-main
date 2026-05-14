/**
 * liquidity_score – Depth-weighted order book liquidity.
 *
 * V8 optimisation: typed loops over bid/ask arrays, no temporary allocations.
 */

const LANGUAGE = "javascript";

/**
 * @param {object} data
 * @returns {object}
 */
export default function liquidityScore(data) {
  const bids = data.bids || [];
  const asks = data.asks || [];
  const depthLevels = (data.depth_levels | 0) || 5;

  if (bids.length === 0 && asks.length === 0) {
    return { liquidity_score: 0.0, depth_bps: 0.0, language: LANGUAGE, ok: true };
  }

  const bestBid = bids.length > 0 ? +bids[0][0] : 0.0;
  const bestAsk = asks.length > 0 ? +asks[0][0] : 0.0;
  const mid = bestBid > 0 && bestAsk > 0 ? (bestBid + bestAsk) / 2.0 : 0.0;
  const depthBps = mid > 0 ? (bestAsk - bestBid) / mid * 1e4 : 100.0;

  let totalVol = 0.0;
  const bidLevels = Math.min(bids.length, depthLevels);
  const askLevels = Math.min(asks.length, depthLevels);
  for (let i = 0; i < bidLevels; i++) totalVol += +bids[i][1];
  for (let i = 0; i < askLevels; i++) totalVol += +asks[i][1];

  const score = totalVol > 0 ? Math.min(1.0, totalVol / 100.0) : 0.0;

  return {
    liquidity_score: score,
    depth_bps: depthBps,
    language: LANGUAGE,
    ok: true,
  };
}
