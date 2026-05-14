/**
 * order_book_processing – Tight loop over bids/asks to compute spread,
 * imbalance and mid price.
 *
 * V8 optimisation: numeric-only inner loop, no object allocation inside hot path,
 * pre-read array lengths.
 */

import PROFILE from "../profile.js";

const LANGUAGE = "javascript";

/**
 * @param {object} data
 * @returns {object}
 */
export default function orderBookProcessing(data) {
  const bids = data.bids || [];
  const asks = data.asks || [];
  const spreadMult = PROFILE.spread_mult;

  const bidsLen = bids.length;
  const asksLen = asks.length;

  if (bidsLen === 0 && asksLen === 0) {
    return { spread_bps: 0.0, imbalance: 0.0, mid: 0.0, language: LANGUAGE, spread_mult: spreadMult };
  }

  const bestBid = bidsLen > 0 ? +bids[0][0] : 0.0;
  const bestAsk = asksLen > 0 ? +asks[0][0] : 0.0;
  const mid = (bestBid !== 0 && bestAsk !== 0) ? (bestBid + bestAsk) / 2.0 : 0.0;

  const rawSpreadBps = mid !== 0 ? ((bestAsk - bestBid) / mid) * 1e4 : 0.0;
  const spreadBps = rawSpreadBps * spreadMult;

  // Top 5 levels imbalance
  const top = 5;
  let bidVol = 0.0;
  let askVol = 0.0;

  const bidLevels = bidsLen < top ? bidsLen : top;
  for (let i = 0; i < bidLevels; i++) {
    bidVol += +bids[i][1];
  }

  const askLevels = asksLen < top ? asksLen : top;
  for (let i = 0; i < askLevels; i++) {
    askVol += +asks[i][1];
  }

  const total = bidVol + askVol;
  const imbalance = total !== 0 ? (bidVol - askVol) / total : 0.0;

  return {
    spread_bps: spreadBps,
    imbalance: imbalance,
    mid: mid,
    language: LANGUAGE,
    spread_mult: spreadMult,
  };
}
