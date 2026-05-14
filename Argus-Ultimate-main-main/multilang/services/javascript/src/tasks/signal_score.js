/**
 * signal_score – Hash-based deterministic delta applied to signal confidence.
 *
 * V8 optimisation: reuse FNV-1a, monomorphic encoding path.
 */

import PROFILE from "../profile.js";

const LANGUAGE = "javascript";

/**
 * FNV-1a 32-bit over a Uint8Array.
 * @param {Uint8Array} bytes
 * @returns {number}
 */
function fnv1a32(bytes) {
  let hash = 0x811c9dc5 | 0;
  for (let i = 0, len = bytes.length; i < len; i++) {
    hash ^= bytes[i];
    hash = (hash + (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24)) | 0;
  }
  return hash >>> 0;
}

const encoder = new TextEncoder();

/**
 * @param {object} data
 * @returns {object}
 */
export default function signalScore(data) {
  const weight = PROFILE.signal_score_weight;
  const confidence = +(data.confidence || 0);
  const baseScore = +(data.score != null ? data.score : confidence);

  // Deterministic hash from language + sorted data keys
  const keys = Object.keys(data).sort();
  const pairs = new Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    pairs[i] = [keys[i], data[keys[i]]];
  }
  const encoded = encoder.encode(LANGUAGE + JSON.stringify(pairs));
  const h = fnv1a32(encoded);

  const delta = ((h % 100) - 50) / 5000.0;
  const scoreDelta = delta * weight;

  return {
    score_delta: scoreDelta,
    signal_score_weight: weight,
    base_score: baseScore,
    language: LANGUAGE,
    ok: true,
  };
}
