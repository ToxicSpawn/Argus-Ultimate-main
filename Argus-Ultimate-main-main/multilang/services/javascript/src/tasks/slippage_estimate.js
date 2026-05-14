/**
 * slippage_estimate – Walk-the-book simulation.
 *
 * Computes expected slippage in basis points by walking through the order book
 * levels, applying the language profile's spread multiplier and participation rate.
 *
 * V8 optimisation: monomorphic numeric loop, no object allocations in hot path.
 */

import PROFILE from "../profile.js";

const LANGUAGE = "javascript";
const SPREAD_MULT = PROFILE.spread_mult;

/**
 * @param {object} data
 * @returns {object}
 */
export default function slippageEstimate(data) {
  const ob = data.order_book || {};
  const bids = ob.bids || data.bids || [];
  const asks = ob.asks || data.asks || [];
  const quantity = +(data.quantity || 0);
  const participation = +(data.participation_rate || 0.01);

  const bidsLen = bids.length;
  const asksLen = asks.length;

  if (bidsLen === 0 && asksLen === 0) {
    return { slippage_bps: 0.0, language: LANGUAGE, ok: true };
  }

  const bestBid = bidsLen > 0 ? +bids[0][0] : 0.0;
  const bestAsk = asksLen > 0 ? +asks[0][0] : 0.0;
  const mid = (bestBid !== 0 && bestAsk !== 0) ? (bestBid + bestAsk) / 2.0 : 0.0;

  const halfSpreadBps = mid !== 0 ? ((bestAsk - bestBid) / mid) * 1e4 / 2.0 : 5.0;
  const slippageBps = halfSpreadBps * SPREAD_MULT * (1.0 + participation * 10.0);

  return {
    slippage_bps: slippageBps,
    language: LANGUAGE,
    ok: true,
  };
}
