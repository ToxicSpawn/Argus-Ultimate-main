/**
 * heartbeat – Liveness ping with cycle echo.
 *
 * V8 optimisation: trivial, always same result shape.
 */

const LANGUAGE = "javascript";

/**
 * @param {object} data
 * @returns {object}
 */
export default function heartbeat(data) {
  return {
    ok: true,
    latency_ms: 0.0,
    language: LANGUAGE,
    cycle_id: data.cycle_id != null ? data.cycle_id : 0,
  };
}
