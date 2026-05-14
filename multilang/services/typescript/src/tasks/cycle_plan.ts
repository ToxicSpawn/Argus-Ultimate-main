import PROFILE from "../profile.js";

const LANGUAGE = "typescript";

// Pre-computed language index: sum of char codes mod 100
const LANG_IDX: number = (() => {
  let s = 0;
  for (let i = 0; i < LANGUAGE.length; i++) s += LANGUAGE.charCodeAt(i);
  return s % 100;
})();

/** FNV-1a 32-bit over a Buffer. */
function fnv1a32(buf: Buffer): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < buf.length; i++) {
    hash ^= buf[i] as number;
    hash = (hash + (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24)) >>> 0;
  }
  return hash;
}

export default function cyclePlan(data: Record<string, unknown>): Record<string, unknown> {
  const scale = PROFILE.cycle_boost_scale;
  const keys = Object.keys(data).sort();
  const pairs = keys.map((k) => [k, data[k]]);
  const h = fnv1a32(Buffer.from(JSON.stringify(pairs)));
  const base = ((h % 200) - 100) / 10000.0 + (LANG_IDX - 50) / 10000.0;

  const signals = ((data["signals"] as number) | 0) || 0;
  const cash = Number(data["cash_balance_aud"] ?? 0);
  const pv = Number(data["portfolio_value_aud"] ?? 1) || 1;
  const cashRatio = cash / pv;
  const tilt = (cashRatio - 0.5) * 0.002 + ((signals % 3) - 1) * 0.001;

  let boost = (base + tilt) * scale;
  if (boost < -0.015) boost = -0.015;
  if (boost > 0.015) boost = 0.015;

  return { language: LANGUAGE, cycle_boost: boost, cycle_boost_scale: scale, ok: true };
}
