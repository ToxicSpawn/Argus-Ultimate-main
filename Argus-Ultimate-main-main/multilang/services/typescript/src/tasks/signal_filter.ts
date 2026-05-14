import PROFILE from "../profile.js";

const LANGUAGE = "typescript";
const SEED: number = (() => { let s = 0; for (let i = 0; i < LANGUAGE.length; i++) s += LANGUAGE.charCodeAt(i); return s % 5; })();

export default function signalFilter(data: Record<string, unknown>): Record<string, unknown> {
  const sig = (data["signal"] && typeof data["signal"] === "object") ? data["signal"] as Record<string, unknown> : data;
  const confidence = Number((sig as Record<string, unknown>)["confidence"] ?? data["confidence"] ?? 0);
  const regime = String(data["regime"] ?? "mean_revert");
  const volatility = Number(data["volatility"] ?? 0.01);

  let accept = confidence >= PROFILE.min_confidence_to_accept && (regime !== "high_vol" || volatility < 0.02);
  if (SEED === 0 && confidence < 0.8) accept = false;

  return { accept, filter_reason: accept ? "" : "low_confidence_or_regime", language: LANGUAGE, ok: true };
}
