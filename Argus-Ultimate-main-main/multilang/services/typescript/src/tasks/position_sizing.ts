import PROFILE from "../profile.js";

const LANGUAGE = "typescript";

export default function positionSizing(data: Record<string, unknown>): Record<string, unknown> {
  const capital = Number(data["capital"] ?? 1) || 1;
  const volBps = Number(data["volatility_bps"] ?? data["volatility_annual_bps"] ?? 10);
  const confidence = Number(data["confidence"] ?? 0.5);
  const maxRiskPct = Number(data["max_risk_pct"] ?? 0.02);
  const riskMax = PROFILE.risk_max_ratio;
  const sizePct = Math.min(riskMax, maxRiskPct * (volBps / 10.0) * (0.5 + confidence));

  return { size_pct: sizePct, size_abs: sizePct * capital, language: LANGUAGE, ok: true };
}
