const LANGUAGE = "typescript";

export default function varEstimate(data: Record<string, unknown>): Record<string, unknown> {
  const returns = (data["returns"] as number[] | undefined) ?? [];
  const confidenceLevel = Number(data["confidence_level"] ?? 0.95);

  if (returns.length < 5) return { var_pct: 0.0, cvar_pct: 0.0, language: LANGUAGE, ok: true };

  const arr = [...returns].map(Number).sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(Math.floor((1 - confidenceLevel) * arr.length), arr.length - 1));
  const varPct = -(arr[idx] as number) * 100.0;
  let cvarSum = 0.0;
  for (let i = 0; i <= idx; i++) cvarSum += arr[i] as number;
  const cvarPct = -(cvarSum / (idx + 1)) * 100.0;

  return { var_pct: varPct, cvar_pct: cvarPct, language: LANGUAGE, ok: true };
}
