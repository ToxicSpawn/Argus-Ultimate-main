import PROFILE from "../profile.js";

const LANGUAGE = "typescript";

export default function riskCalculation(data: Record<string, unknown>): Record<string, unknown> {
  const maxRatio = PROFILE.risk_max_ratio;
  const pv = Number(data["position_value"] ?? 0);
  const capital = Number(data["capital"] ?? 1) || 1;
  const ratio = pv / capital;
  const passed = ratio <= maxRatio;

  return { passed, exposure_ratio: ratio, max_ratio: maxRatio, language: LANGUAGE };
}
