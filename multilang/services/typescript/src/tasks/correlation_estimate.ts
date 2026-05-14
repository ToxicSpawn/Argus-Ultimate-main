const LANGUAGE = "typescript";

export default function correlationEstimate(data: Record<string, unknown>): Record<string, unknown> {
  const a = ((data["series_a"] ?? data["returns_a"]) as number[] | undefined) ?? [];
  const b = ((data["series_b"] ?? data["returns_b"]) as number[] | undefined) ?? [];
  const n = a.length;

  if (n !== b.length || n < 2) return { correlation: 0.0, language: LANGUAGE, ok: true };

  let meanA = 0.0, meanB = 0.0, c = 0.0, varA = 0.0, varB = 0.0;
  for (let i = 0; i < n; i++) {
    const ai = a[i] as number, bi = b[i] as number, k = i + 1;
    const dA = ai - meanA, dB = bi - meanB;
    meanA += dA / k; meanB += dB / k;
    varA += dA * (ai - meanA); varB += dB * (bi - meanB); c += dA * (bi - meanB);
  }

  const denom = Math.sqrt(varA * varB);
  let correlation = denom !== 0 ? c / denom : 0.0;
  if (correlation > 1.0) correlation = 1.0;
  if (correlation < -1.0) correlation = -1.0;

  return { correlation, language: LANGUAGE, ok: true };
}
