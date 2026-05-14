import PROFILE from "../profile.js";

const LANGUAGE = "typescript";

export default function drawdownCheck(data: Record<string, unknown>): Record<string, unknown> {
  const maxDrawdownPct = Number(data["max_drawdown_pct"] ?? 0.12);
  const current = Number(data["current_equity"] ?? 0);
  const peak = Number(data["peak_equity"] ?? current) || 1;
  const currentDrawdownPct = peak !== 0 ? (peak - current) / peak : 0.0;
  const passed = currentDrawdownPct <= maxDrawdownPct * PROFILE.risk_max_ratio;

  return { passed, current_drawdown_pct: currentDrawdownPct, language: LANGUAGE, ok: true };
}
