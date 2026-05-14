const LANGUAGE = "typescript";

export default function executionQualityScore(data: Record<string, unknown>): Record<string, unknown> {
  const fills = (data["fills"] as unknown[] | undefined) ?? [];
  const decisionPrices = (data["decision_prices"] as number[] | undefined) ?? [];

  if (fills.length === 0 || fills.length !== decisionPrices.length) {
    return { score_0_1: 1.0, avg_slippage_bps: 0.0, language: LANGUAGE, ok: true };
  }

  const count = Math.min(fills.length, 10);
  let totalBps = 0.0, used = 0;
  for (let i = 0; i < count; i++) {
    const f = fills[i];
    const fp = typeof f === "object" && f !== null ? Number((f as Record<string, unknown>)["price"] ?? f) : Number(f);
    const dp = Number(decisionPrices[i]);
    if (dp > 0 && fp > 0) { totalBps += Math.abs(fp - dp) / dp * 1e4; used++; }
  }
  const avgBps = used > 0 ? totalBps / used : 0.0;
  return { score_0_1: Math.max(0.0, Math.min(1.0, 1.0 - avgBps / 50.0)), avg_slippage_bps: avgBps, language: LANGUAGE, ok: true };
}
