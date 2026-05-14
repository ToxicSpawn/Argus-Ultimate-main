import PROFILE from "../profile.js";

const LANGUAGE = "typescript";

function fnv1a32(buf: Buffer): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < buf.length; i++) {
    hash ^= buf[i] as number;
    hash = (hash + (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24)) >>> 0;
  }
  return hash;
}

export default function signalScore(data: Record<string, unknown>): Record<string, unknown> {
  const weight = PROFILE.signal_score_weight;
  const confidence = Number(data["confidence"] ?? 0);
  const baseScore = Number(data["score"] ?? confidence);

  const keys = Object.keys(data).sort();
  const pairs = keys.map((k) => [k, data[k]]);
  const h = fnv1a32(Buffer.from(LANGUAGE + JSON.stringify(pairs)));
  const delta = ((h % 100) - 50) / 5000.0;

  return { score_delta: delta * weight, signal_score_weight: weight, base_score: baseScore, language: LANGUAGE, ok: true };
}
