const LANGUAGE = "typescript";

export default function skewEstimate(data: Record<string, unknown>): Record<string, unknown> {
  const returns = (data["returns"] as number[] | undefined) ?? [];
  const n = returns.length;
  if (n < 3) return { skew: 0.0, language: LANGUAGE, ok: true };

  let mean = 0.0;
  for (let i = 0; i < n; i++) mean += returns[i] as number;
  mean /= n;

  let sumVar = 0.0, sumCube = 0.0;
  for (let i = 0; i < n; i++) {
    const d = (returns[i] as number) - mean;
    sumVar += d * d; sumCube += d * d * d;
  }
  const std = sumVar > 0 ? Math.sqrt(sumVar / n) : 0.0;
  const skew = std > 0 ? (sumCube / n) / (std * std * std) : 0.0;

  return { skew, language: LANGUAGE, ok: true };
}
