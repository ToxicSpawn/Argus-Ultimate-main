/**
 * order_book_imbalance_series – Bid/ask volume imbalance from top-5 levels.
 *
 * Returns a single-element series (current snapshot) and trend (same value).
 *
 * V8 optimisation: fixed-depth loop, monomorphic result shape.
 */

const LANGUAGE = "javascript";
const DEPTH = 5;

/**
 * @param {object} data
 * @returns {object}
 */
export default function orderBookImbalanceSeries(data) {
  const bids = data.bids || [];
  const asks = data.asks || [];

  if (bids.length === 0 && asks.length === 0) {
    return { imbalance_series: [], trend: 0.0, language: LANGUAGE, ok: true };
  }

  let bidVol = 0.0;
  let askVol = 0.0;
  const bidLevels = Math.min(bids.length, DEPTH);
  const askLevels = Math.min(asks.length, DEPTH);
  for (let i = 0; i < bidLevels; i++) bidVol += +bids[i][1];
  for (let i = 0; i < askLevels; i++) askVol += +asks[i][1];

  const total = bidVol + askVol;
  const imb = total > 0 ? (bidVol - askVol) / total : 0.0;

  return {
    imbalance_series: [imb],
    trend: imb,
    language: LANGUAGE,
    ok: true,
  };
}
