const LANGUAGE = "typescript";

export default function confidenceCalibration(data: Record<string, unknown>): Record<string, unknown> {
  const confs = (data["historical_confidences"] as number[] | undefined) ?? [];
  const pnls = (data["historical_pnl"] as number[] | undefined) ?? [];

  if (confs.length !== pnls.length || confs.length < 2) return { calibrated_confidence: 0.5, language: LANGUAGE, ok: true };

  const n = confs.length;
  let sumConf = 0.0, wins = 0;
  for (let i = 0; i < n; i++) { sumConf += confs[i] as number; if ((pnls[i] as number) > 0) wins++; }
  const calibrated = Math.min(1.0, Math.max(0.0, 0.5 * (sumConf / n) + 0.5 * (wins / n)));

  return { calibrated_confidence: calibrated, language: LANGUAGE, ok: true };
}
