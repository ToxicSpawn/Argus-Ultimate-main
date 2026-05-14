/**
 * market_impact – Almgren-Chriss sqrt-participation market impact model.
 *
 * V8 optimisation: pure numeric, monomorphic result shape.
 */

const LANGUAGE = "javascript";

/**
 * @param {object} data
 * @returns {object}
 */
export default function marketImpact(data) {
  const quantity = +(data.quantity || 0.0);
  const adv = +(data.adv || 1.0);
  const volatility = +(data.volatility || 0.01);

  const participation = adv !== 0 ? quantity / adv : 0.0;
  const impactBps = 10.0 * Math.sqrt(participation) * volatility * 1e4;

  return {
    impact_bps: impactBps,
    language: LANGUAGE,
    ok: true,
  };
}
