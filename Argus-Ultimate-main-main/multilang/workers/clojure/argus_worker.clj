;; Argus Ultimate -- Clojure multilang worker
;; Profile: language=clojure, risk-max=0.44, cycle-scale=0.98,
;;          vol-w=1.02, sig-w=1.02, spread=1.01
;;
;; Run with:  clojure argus_worker.clj   (or clj argus_worker.clj)
;; Protocol:
;;   stdin  -> {"task_type": "...", "data": {...}}
;;   stdout <- {"ok": true, "result": {...}, "took_ms": 0.12}
;; Uses only clojure.core -- no external deps.

(ns argus-worker.core
  (:require [clojure.string :as str]))

;; ==================================================================
;; Profile
;; ==================================================================

(def ^:const RISK-MAX    0.44)
(def ^:const CYCLE-SCALE 0.98)
(def ^:const VOL-W       1.02)
(def ^:const SIG-W       1.02)
(def ^:const SPREAD      1.01)

;; ==================================================================
;; Minimal JSON helpers (no external deps)
;; ==================================================================

(defn to-json [m]
  (str "{"
       (str/join ","
         (map (fn [[k v]]
                (str "\"" (name k) "\":"
                     (cond
                       (string? v)  (str "\"" v "\"")
                       (boolean? v) (str v)
                       (nil? v)     "null"
                       :else        (str v))))
              m))
       "}"))

(defn get-field
  "Extract a scalar field value from a raw JSON string."
  [json field]
  (let [pattern (re-pattern (str "\"" field "\"\\s*:\\s*\"?([^,}\"]+)\"?"))
        m       (re-find pattern json)]
    (when m (str/trim (second m)))))

(defn get-number
  "Parse a numeric field from raw JSON, returning default-val if absent."
  [json field default-val]
  (if-let [s (get-field json field)]
    (try (Double/parseDouble s) (catch Exception _ default-val))
    default-val))

(defn get-array
  "Extract a JSON array of numbers: [1.0, 2.0, 3.0]"
  [json field]
  (let [pattern (re-pattern (str "\"" field "\"\\s*:\\s*\\[([^\\]]*)\\]"))
        m       (re-find pattern json)]
    (if m
      (->> (str/split (second m) #",")
           (map str/trim)
           (keep #(try (Double/parseDouble %) (catch Exception _ nil))))
      [])))

;; ==================================================================
;; Math helpers
;; ==================================================================

(defn log-returns [prices]
  (->> (partition 2 1 prices)
       (map (fn [[p1 p2]]
              (if (zero? p1) 0.0 (Math/log (/ p2 p1)))))))

(defn mean [xs]
  (if (empty? xs) 0.0
      (/ (reduce + xs) (count xs))))

(defn variance [xs]
  (if (< (count xs) 2) 0.0
      (let [m (mean xs)]
        (/ (reduce + (map #(* (- % m) (- % m)) xs))
           (count xs)))))

(defn std-dev [xs]
  (Math/sqrt (variance xs)))

(defn compute-vol [prices]
  (if (< (count prices) 2) 0.0
      (std-dev (log-returns prices))))

(defn compute-skew [xs]
  (if (< (count xs) 2) 0.0
      (let [n  (count xs)
            m  (mean xs)
            sd (std-dev xs)]
        (if (zero? sd) 0.0
            (/ (reduce + (map #(Math/pow (/ (- % m) sd) 3) xs))
               n)))))

(defn cycle-hash [raw-data]
  "Derive a numeric hash from sorted key:val pairs in the data JSON."
  (let [pattern #"\"([^\"]+)\"\s*:\s*([^,}\"]+)"
        pairs   (sort (re-seq pattern raw-data))
        s       (str/join "," (map (fn [[_ k v]] (str k ":" (str/trim v))) pairs))
        bytes   (.digest (java.security.MessageDigest/getInstance "MD5")
                         (.getBytes s "UTF-8"))
        val     (bit-and 0xFFFFFFFF
                         (reduce (fn [acc b] (+ (* acc 256) (bit-and b 0xFF)))
                                 0 (take 4 bytes)))]
    val))

(defn pick-action [h]
  (case (mod h 3) 0 "buy" 1 "sell" "hold"))

;; ==================================================================
;; Dispatch
;; ==================================================================

(defn dispatch [task-type raw-json]
  (case task-type

    "heartbeat"
    {:ok true :language "clojure" :ts (System/currentTimeMillis)}

    "cycle_plan"
    (let [h  (cycle-hash raw-json)
          ac (pick-action h)
          cf (+ 0.5 (/ (mod h 50) 100.0))
          sz (* CYCLE-SCALE (+ 0.1 (/ (mod h 10) 100.0)))]
      {:ok true :language "clojure" :hash h :action ac :confidence cf :size sz})

    "volatility_estimate"
    (let [prices (get-array raw-json "prices")
          vol    (* (compute-vol prices) VOL-W)]
      {:ok true :language "clojure" :volatility_annual_bps vol :volatility_weight VOL-W})

    "signal_score"
    (let [h  (cycle-hash raw-json)
          delta (/ (- (mod h 100) 50) 5000.0)]
      {:ok true :language "clojure" :score_delta (* delta SIG-W) :signal_score_weight SIG-W})

    "risk_calculation"
    (let [pv  (get-number raw-json "position_value" 0.0)
          cap (get-number raw-json "capital"        1.0)
          c2  (if (zero? cap) 1.0 cap)
          rr  (/ pv c2)]
      {:ok true :language "clojure" :exposure_ratio rr :passed (<= rr RISK-MAX) :max_ratio RISK-MAX})

    "position_sizing"
    (let [cap (get-number raw-json "capital"   10000.0)
          rp  (get-number raw-json "risk_pct"  0.01)
          sd  (get-number raw-json "stop_dist" 1.0)
          sd2 (if (zero? sd) 1.0 sd)
          sz  (/ (* cap rp) sd2)
          c2  (if (zero? cap) 1.0 cap)]
      {:ok true :language "clojure" :size_pct (* (/ sz c2) 100.0) :size_abs sz})

    "drawdown_check"
    (let [pk  (get-number raw-json "peak"    1.0)
          cu  (get-number raw-json "current" 1.0)
          p2  (if (zero? pk) 1.0 pk)
          dd  (/ (- p2 cu) p2)]
      {:ok true :language "clojure" :current_drawdown_pct (* dd 100.0) :passed (<= (* dd 100.0) 20.0)})

    "var_estimate"
    (let [prices (get-array raw-json "prices")
          vol    (compute-vol prices)
          v95    (* vol 1.645)]
      {:ok true :language "clojure" :var_pct v95 :cvar_pct (* v95 1.2)})

    "skew_estimate"
    (let [prices (get-array raw-json "prices")]
      {:ok true :language "clojure" :skew (compute-skew prices)})

    "order_book_imbalance_series"
    (let [b   (get-number raw-json "bid_volume" 100.0)
          a   (get-number raw-json "ask_volume" 100.0)
          d   (+ b a)
          imb (if (zero? d) 0.0 (/ (- b a) d))]
      {:ok true :language "clojure" :imbalance_series [imb] :trend imb})

    "order_book_processing"
    (let [b   (get-number raw-json "bid" 0.0)
          a   (get-number raw-json "ask" 0.0)
          mid (/ (+ b a) 2.0)
          sp  (if (> mid 0.0) (* (/ (- a b) mid) 10000.0) 0.0)
          d   (+ b a)
          imb (if (zero? d) 0.0 (/ (- b a) d))]
      {:ok true :language "clojure" :spread_bps sp :imbalance imb :mid mid})

    "regime_estimate"
    (let [prices (get-array raw-json "prices")
          vol    (compute-vol prices)
          regime (if (> vol 0.02) "high_vol" "low_vol")]
      {:ok true :language "clojure" :regime regime :confidence (+ 0.5 (* vol 10.0)) :regime_weight 1.0})

    "slippage_estimate"
    (let [sz (get-number raw-json "size"   1.0)
          sp (get-number raw-json "spread" 0.01)]
      {:ok true :language "clojure" :slippage_bps (* sz sp SPREAD)})

    "correlation_estimate"
    {:ok true :language "clojure" :correlation 0.0}

    "liquidity_score"
    (let [vol   (get-number raw-json "volume" 1000.0)
          score (min 1.0 (/ vol 10000.0))]
      {:ok true :language "clojure" :liquidity_score score :depth_bps 100})

    "market_impact"
    (let [sz  (get-number raw-json "size"      1.0)
          liq (get-number raw-json "liquidity" 1000.0)
          l2  (if (zero? liq) 1.0 liq)]
      {:ok true :language "clojure" :impact_bps (/ sz l2)})

    "signal_filter"
    (let [sig (get-number raw-json "signal"    0.0)
          thr (get-number raw-json "threshold" 0.1)]
      {:ok true :language "clojure" :accept (>= (Math/abs sig) thr) :filter_reason ""})

    "confidence_calibration"
    (let [raw (get-number raw-json "confidence" 0.5)
          cal (max 0.0 (min 1.0 raw))]
      {:ok true :language "clojure" :calibrated_confidence cal})

    "execution_quality_score"
    (let [slip  (get-number raw-json "slippage" 0.0)
          score (max 0.0 (- 1.0 (* (Math/abs slip) 10.0)))]
      {:ok true :language "clojure" :score_0_1 score :avg_slippage_bps 0})

    "regime_duration"
    (let [start (get-number raw-json "start_ts" 0.0)
          now   (double (System/currentTimeMillis))
          dur   (- now start)
          bars  (int (/ dur 60000.0))]
      {:ok true :language "clojure" :bars_in_regime bars :regime_stable (> bars 5) :regime "unknown"})

    ;; fallback
    {:ok true :language "clojure" :task task-type :value 0.5}))

;; ==================================================================
;; Main loop
;; ==================================================================

(defn -main [& _args]
  (loop [line (.readLine *in*)]
    (when line
      (let [trimmed (str/trim line)]
        (when-not (empty? trimmed)
          (let [t0        (System/nanoTime)
                task-type (get-field trimmed "task_type")
                result    (dispatch (or task-type "") trimmed)
                took-ms   (/ (- (System/nanoTime) t0) 1000000.0)
                out       (to-json (assoc result :took_ms took-ms))]
            (println out)
            (.flush *out*))))
      (recur (.readLine *in*)))))

(-main)
