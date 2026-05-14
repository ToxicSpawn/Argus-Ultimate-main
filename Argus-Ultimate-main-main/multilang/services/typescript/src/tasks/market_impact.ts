const LANGUAGE = "typescript";

export default function marketImpact(data: Record<string, unknown>): Record<string, unknown> {
  const quantity = Number(data["quantity"] ?? 0);
  const adv = Number(data["adv"] ?? 1) || 1;
  const volatility = Number(data["volatility"] ?? 0.01);
  const participation = quantity / adv;
  const impactBps = 10.0 * Math.sqrt(participation) * volatility * 1e4;
  return { impact_bps: impactBps, language: LANGUAGE, ok: true };
}
