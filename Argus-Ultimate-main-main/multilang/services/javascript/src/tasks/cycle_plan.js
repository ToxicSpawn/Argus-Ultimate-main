/**
 * cycle_plan – FNV-1a hash of JSON-stringified context for deterministic boost.
 *
 * V8 optimisation: monomorphic fnv1a operating on a Uint8Array buffer,
 * no polymorphic property accesses inside hot path.
 */

import PROFILE from "../profile.js";

const LANGUAGE = "javascript";

/**
 * FNV-1a 32-bit hash over a byte buffer.
 * Kept monomorphic: always receives Uint8Array.
 * @param {Uint8Array} bytes
 * @returns {number} unsigned 32-bit hash
 */
function fnv1a32(bytes) {
  let hash = 0x811c9dc5 | 0;
  for (let i = 0, len = bytes.length; i < len; i++) {
    hash ^= bytes[i];
    hash = (hash + (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24)) | 0;
  }
  return hash >>> 0;
}

// Pre-computed language index: sum of char codes mod 100
const LANG_IDX = (() => {
  let s = 0;
  for (let i = 0; i < LANGUAGE.length; i++) s += LANGUAGE.charCodeAt(i);
  return s % 100;
})();

const encoder = new TextEncoder();

/**
 * @param {object} data
 * @returns {object}
 */
export default function cyclePlan(data) {
  const scale = PROFILE.cycle_boost_scale;

  // Deterministic key: sorted JSON of input context
  const keys = Object.keys(data).sort();
  const pairs = new Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    pairs[i] = [keys[i], data[keys[i]]];
  }
  const encoded = encoder.encode(JSON.stringify(pairs));
  const h = fnv1a32(encoded);

  const base = ((h % 200) - 100) / 10000.0 + (LANG_IDX - 50) / 10000.0;

  const signals = (data.signals | 0) || 0;
  const cash = +(data.cash_balance_aud || 0);
  const pv = +(data.portfolio_value_aud || 1);
  const cashRatio = pv !== 0 ? cash / pv : 0;
  const tilt = (cashRatio - 0.5) * 0.002 + ((signals % 3) - 1) * 0.001;

  let boost = (base + tilt) * scale;
  if (boost < -0.015) boost = -0.015;
  if (boost > 0.015) boost = 0.015;

  return {
    language: LANGUAGE,
    cycle_boost: boost,
    cycle_boost_scale: scale,
    ok: true,
  };
}
