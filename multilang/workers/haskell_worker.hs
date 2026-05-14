-- Argus Haskell Worker
-- Compile: ghc -O2 haskell_worker.hs -o haskell_worker.exe
-- Uses only GHC base libraries: Data.Map.Strict, Data.List, Data.Maybe, Data.Char, System.IO, Data.IORef, Control.Monad

module Main where

import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.List (sortBy, foldl')
import Data.Maybe (fromMaybe)
import Data.Char (isDigit, isAlpha, isSpace, ord)
import System.IO (hSetBuffering, hPutStrLn, hFlush, stdin, stdout, stderr, BufferMode(..), isEOF)
import Data.IORef
import Control.Monad (when, forM_)
import Control.Exception (SomeException, try, evaluate)

-- ─────────────────────────────────────────────
-- Language profile constants (Haskell)
-- ─────────────────────────────────────────────
langName :: String
langName = "haskell"

riskMax :: Double
riskMax = 0.40

cycleScale :: Double
cycleScale = 0.95

volWeight :: Double
volWeight = 1.05

sigWeight :: Double
sigWeight = 1.02

spreadMult :: Double
spreadMult = 1.03

langIdx :: Int
langIdx = foldl' (\acc c -> acc + ord c) 0 langName `mod` 100

-- ─────────────────────────────────────────────
-- Minimal JSON value type
-- ─────────────────────────────────────────────
data JVal
  = JStr String
  | JNum Double
  | JBool Bool
  | JNull
  | JArr [JVal]
  | JObj (Map String JVal)
  deriving (Show)

-- ─────────────────────────────────────────────
-- JSON Encoder
-- ─────────────────────────────────────────────
encodeJSON :: JVal -> String
encodeJSON (JStr s)  = "\"" ++ escapeStr s ++ "\""
encodeJSON (JNum d)
  | isNaN d || isInfinite d = "0.0"
  | d == fromIntegral (round d :: Int) && abs d < 1e15 = show (round d :: Int) ++ ".0"
  | otherwise = show d
encodeJSON (JBool True)  = "true"
encodeJSON (JBool False) = "false"
encodeJSON JNull         = "null"
encodeJSON (JArr xs)     = "[" ++ joinWith "," (map encodeJSON xs) ++ "]"
encodeJSON (JObj m)      =
  "{" ++ joinWith "," (map encPair (Map.toAscList m)) ++ "}"
  where encPair (k,v) = "\"" ++ escapeStr k ++ "\":" ++ encodeJSON v

escapeStr :: String -> String
escapeStr []       = []
escapeStr ('"':cs) = '\\':'"' : escapeStr cs
escapeStr ('\\':cs)= '\\':'\\': escapeStr cs
escapeStr ('\n':cs)= '\\':'n' : escapeStr cs
escapeStr ('\r':cs)= '\\':'r' : escapeStr cs
escapeStr ('\t':cs)= '\\':'t' : escapeStr cs
escapeStr (c:cs)   = c : escapeStr cs

joinWith :: String -> [String] -> String
joinWith _ []     = ""
joinWith _ [x]    = x
joinWith sep (x:xs) = x ++ sep ++ joinWith sep xs

-- ─────────────────────────────────────────────
-- Minimal JSON Parser
-- ─────────────────────────────────────────────
type Parser a = String -> Maybe (a, String)

skipWS :: String -> String
skipWS = dropWhile isSpace

parseJSON :: String -> Maybe JVal
parseJSON s = case parseValue (skipWS s) of
  Just (v, rest) -> if all isSpace rest then Just v else Just v
  Nothing        -> Nothing

parseValue :: Parser JVal
parseValue [] = Nothing
parseValue (c:cs)
  | c == '"'  = fmap (\(s,r) -> (JStr s, r)) (parseString cs)
  | c == '{'  = fmap (\(m,r) -> (JObj m, r)) (parseObject cs)
  | c == '['  = fmap (\(a,r) -> (JArr a, r)) (parseArray cs)
  | c == 't'  = case cs of
      'r':'u':'e':rest -> Just (JBool True, rest)
      _ -> Nothing
  | c == 'f'  = case cs of
      'a':'l':'s':'e':rest -> Just (JBool False, rest)
      _ -> Nothing
  | c == 'n'  = case cs of
      'u':'l':'l':rest -> Just (JNull, rest)
      _ -> Nothing
  | c == '-' || isDigit c = parseNumber (c:cs)
  | otherwise = Nothing

parseString :: Parser String
parseString [] = Nothing
parseString s  = go [] s
  where
    go acc []           = Nothing
    go acc ('"':rest)   = Just (reverse acc, rest)
    go acc ('\\':c:rest)= go (unescape c : acc) rest
    go acc (c:rest)     = go (c:acc) rest
    unescape 'n' = '\n'
    unescape 'r' = '\r'
    unescape 't' = '\t'
    unescape c   = c

parseNumber :: Parser JVal
parseNumber s =
  let (numStr, rest) = span (\c -> isDigit c || c `elem` ".-+eE") s
  in case reads numStr :: [(Double, String)] of
       [(d, "")] -> Just (JNum d, rest)
       _         -> Nothing

parseObject :: Parser (Map String JVal)
parseObject s =
  let s' = skipWS s
  in case s' of
    '}':rest -> Just (Map.empty, rest)
    _ -> parsePairs Map.empty s'
  where
    parsePairs acc input =
      let input' = skipWS input
      in case input' of
        '}':rest -> Just (acc, rest)
        '"':cs -> case parseString cs of
          Just (key, afterKey) ->
            let afterKey' = skipWS afterKey
            in case afterKey' of
              ':':afterColon ->
                case parseValue (skipWS afterColon) of
                  Just (val, afterVal) ->
                    let afterVal' = skipWS afterVal
                        acc' = Map.insert key val acc
                    in case afterVal' of
                      ',':rest -> parsePairs acc' (skipWS rest)
                      '}':rest -> Just (acc', rest)
                      _        -> Just (acc', afterVal')
                  Nothing -> Nothing
              _ -> Nothing
          Nothing -> Nothing
        _ -> Nothing

parseArray :: Parser [JVal]
parseArray s =
  let s' = skipWS s
  in case s' of
    ']':rest -> Just ([], rest)
    _ -> parseElems [] s'
  where
    parseElems acc input =
      case parseValue (skipWS input) of
        Just (v, rest) ->
          let rest' = skipWS rest
              acc'  = acc ++ [v]
          in case rest' of
            ',':r -> parseElems acc' (skipWS r)
            ']':r -> Just (acc', r)
            _     -> Just (acc', rest')
        Nothing -> Nothing

-- ─────────────────────────────────────────────
-- Helpers to extract values from JObj
-- ─────────────────────────────────────────────
getNum :: Map String JVal -> String -> Double -> Double
getNum m k def = case Map.lookup k m of
  Just (JNum d)  -> d
  Just (JBool b) -> if b then 1.0 else 0.0
  Just JNull     -> def
  Just (JStr s)  -> case reads s :: [(Double,String)] of
                      [(d,"")] -> d
                      _        -> def
  _              -> def

getStr :: Map String JVal -> String -> String -> String
getStr m k def = case Map.lookup k m of
  Just (JStr s) -> s
  _             -> def

getBool :: Map String JVal -> String -> Bool -> Bool
getBool m k def = case Map.lookup k m of
  Just (JBool b) -> b
  Just (JNum d)  -> d /= 0.0
  _              -> def

getArr :: Map String JVal -> String -> [JVal]
getArr m k = case Map.lookup k m of
  Just (JArr xs) -> xs
  _              -> []

getNumArr :: Map String JVal -> String -> [Double]
getNumArr m k = map toD (getArr m k)
  where
    toD (JNum d)  = d
    toD _         = 0.0

-- ─────────────────────────────────────────────
-- Polynomial hash (deterministic)
-- ─────────────────────────────────────────────
polyHash :: String -> Int
polyHash s = foldl' step 0 s `mod` maxBound
  where step acc c = (acc * 31 + ord c) `mod` 1000000007

-- Sort-key pairs and hash
hashSortedData :: Map String JVal -> Int
hashSortedData m =
  let pairs = Map.toAscList m
      str   = concatMap (\(k,v) -> k ++ encodeJSON v) pairs
  in abs (polyHash str)

-- ─────────────────────────────────────────────
-- Welford variance
-- ─────────────────────────────────────────────
welfordVar :: [Double] -> Double
welfordVar [] = 0.0
welfordVar xs = m2 / fromIntegral n
  where
    n = length xs
    (_, _, m2) = foldl' step (0.0, 0.0, 0.0) (zip [1..] xs)
    step (mean, _, m2acc) (i, x) =
      let delta  = x - mean
          mean'  = mean + delta / fromIntegral i
          delta2 = x - mean'
          m2'    = m2acc + delta * delta2
      in (mean', 0.0, m2')

-- Pearson correlation single pass
pearson :: [Double] -> [Double] -> Double
pearson xs ys
  | n < 2 = 0.0
  | otherwise =
      let (sx, sy, sxx, syy, sxy) =
            foldl' step (0,0,0,0,0) (zip xs ys)
          step (ax,ay,axx,ayy,axy) (x,y) =
            (ax+x, ay+y, axx+x*x, ayy+y*y, axy+x*y)
          fn = fromIntegral n
          num  = fn*sxy - sx*sy
          den  = sqrt ((fn*sxx - sx*sx) * (fn*syy - sy*sy))
      in if den == 0 then 0.0 else num/den
  where n = min (length xs) (length ys)

-- ─────────────────────────────────────────────
-- Task implementations
-- ─────────────────────────────────────────────
clamp :: Double -> Double -> Double -> Double
clamp lo hi x = max lo (min hi x)

handleCyclePlan :: Map String JVal -> Map String JVal
handleCyclePlan d =
  let h       = hashSortedData d
      base    = fromIntegral ((h `mod` 200) - 100) / 10000.0
                + fromIntegral (langIdx - 50) / 10000.0
      cash    = getNum d "cash_balance_aud" 0.0
      pv      = let v = getNum d "portfolio_value_aud" 1.0 in if v == 0 then 1.0 else v
      signals = round (getNum d "signals" 0.0) :: Int
      cashR   = cash / pv
      tilt    = (cashR - 0.5) * 0.002 + fromIntegral ((signals `mod` 3) - 1) * 0.001
      boost   = clamp (-0.015) 0.015 ((base + tilt) * cycleScale)
  in Map.fromList
      [ ("language",          JStr langName)
      , ("cycle_boost",       JNum boost)
      , ("cycle_boost_scale", JNum cycleScale)
      , ("ok",                JBool True)
      ]

handleOrderBook :: Map String JVal -> Map String JVal
handleOrderBook d =
  let bids  = getNumArr d "bids"
      asks  = getNumArr d "asks"
      bid0  = if null bids then 0.0 else head bids
      ask0  = if null asks then 0.0 else head asks
      mid   = (bid0 + ask0) / 2.0
      spreadBps = if mid == 0 then 0.0
                  else (ask0 - bid0) / mid * 10000.0 * spreadMult
      top5b = take 5 bids
      top5a = take 5 asks
      sumB  = sum top5b
      sumA  = sum top5a
      imbal = if (sumB + sumA) == 0 then 0.0 else (sumB - sumA) / (sumB + sumA)
  in Map.fromList
      [ ("spread_bps", JNum spreadBps)
      , ("imbalance",  JNum imbal)
      , ("mid",        JNum mid)
      , ("language",   JStr langName)
      ]

handleRisk :: Map String JVal -> Map String JVal
handleRisk d =
  let posVal  = getNum d "position_value" 0.0
      capital = let c = getNum d "capital" 1.0 in if c == 0 then 1.0 else c
      ratio   = posVal / capital
      passed  = ratio <= riskMax
  in Map.fromList
      [ ("passed",         JBool passed)
      , ("exposure_ratio", JNum ratio)
      , ("max_ratio",      JNum riskMax)
      , ("language",       JStr langName)
      ]

handleVolatility :: Map String JVal -> Map String JVal
handleVolatility d =
  let prices  = getNumArr d "prices"
      rets    = getNumArr d "returns"
      returns = if not (null rets) then rets
                else if length prices >= 2
                     then zipWith (\p n -> if p == 0 then 0.0 else (n - p) / p)
                                  (init prices) (tail prices)
                     else []
      var   = welfordVar returns
      vol   = if var > 0 then sqrt (var * 252 * 10000.0) else 10.0
      volAdj = vol * volWeight
  in Map.fromList
      [ ("volatility_annual_bps", JNum volAdj)
      , ("volatility_weight",     JNum volWeight)
      , ("language",              JStr langName)
      , ("ok",                    JBool True)
      ]

handleSignalScore :: Map String JVal -> Map String JVal
handleSignalScore d =
  let seed    = langName ++ encodeJSON (JObj d)
      h       = abs (polyHash seed)
      delta   = fromIntegral ((h `mod` 100) - 50) / 5000.0 * sigWeight
  in Map.fromList
      [ ("score_delta",         JNum delta)
      , ("signal_score_weight", JNum sigWeight)
      , ("language",            JStr langName)
      , ("ok",                  JBool True)
      ]

handleRegime :: Map String JVal -> Map String JVal
handleRegime d =
  let prices = getNumArr d "prices"
      rets   = if length prices >= 2
               then zipWith (\p n -> if p == 0 then 0.0 else (n - p) / p)
                            (init prices) (tail prices)
               else getNumArr d "returns"
      var    = welfordVar rets
      vol    = if var > 0 then sqrt (var * 252) else 0.0
      (regime, conf)
        | vol > 0.25 = ("high_vol",    0.70)
        | vol > 0.05 = ("trend",       0.60)
        | otherwise  = ("mean_revert", 0.55)
  in Map.fromList
      [ ("regime",        JStr regime)
      , ("confidence",    JNum conf)
      , ("regime_weight", JNum 1.0)
      , ("language",      JStr langName)
      , ("ok",            JBool True)
      ]

handleSlippage :: Map String JVal -> Map String JVal
handleSlippage d =
  let halfSpread    = getNum d "half_spread_bps" 1.0
      participation = getNum d "participation_rate" 0.01
      slippage      = halfSpread * spreadMult * (1.0 + participation * 10.0)
  in Map.fromList
      [ ("slippage_bps", JNum slippage)
      , ("language",     JStr langName)
      , ("ok",           JBool True)
      ]

handlePositionSizing :: Map String JVal -> Map String JVal
handlePositionSizing d =
  let volBps   = getNum d "volatility_bps" 50.0
      conf     = getNum d "confidence" 0.5
      maxRisk  = getNum d "max_risk_pct" 0.02
      capital  = getNum d "capital" 100000.0
      sizePct  = min riskMax (maxRisk * (volBps / 10.0) * (0.5 + conf))
      sizeAbs  = sizePct * capital
  in Map.fromList
      [ ("size_pct",  JNum sizePct)
      , ("size_abs",  JNum sizeAbs)
      , ("language",  JStr langName)
      , ("ok",        JBool True)
      ]

handleDrawdown :: Map String JVal -> Map String JVal
handleDrawdown d =
  let peak    = getNum d "peak_value" 1.0
      current = getNum d "current_value" 1.0
      maxDd   = getNum d "max_drawdown_pct" 0.20
      dd      = if peak == 0 then 0.0 else (peak - current) / peak
      passed  = dd <= maxDd * riskMax
  in Map.fromList
      [ ("passed",               JBool passed)
      , ("current_drawdown_pct", JNum (dd * 100.0))
      , ("language",             JStr langName)
      , ("ok",                   JBool True)
      ]

handleCorrelation :: Map String JVal -> Map String JVal
handleCorrelation d =
  let xs   = getNumArr d "series_a"
      ys   = getNumArr d "series_b"
      corr = pearson xs ys
  in Map.fromList
      [ ("correlation", JNum corr)
      , ("language",    JStr langName)
      , ("ok",          JBool True)
      ]

handleLiquidity :: Map String JVal -> Map String JVal
handleLiquidity d =
  let bids   = getNumArr d "bids"
      asks   = getNumArr d "asks"
      top5b  = take 5 bids
      top5a  = take 5 asks
      total  = sum top5b + sum top5a
      score  = min 1.0 (total / 100.0)
      bid0   = if null bids then 0.0 else head bids
      ask0   = if null asks then 0.0 else head asks
      mid    = (bid0 + ask0) / 2.0
      depth  = if mid == 0 then 0.0 else (ask0 - bid0) / mid * 10000.0
  in Map.fromList
      [ ("liquidity_score", JNum score)
      , ("depth_bps",       JNum depth)
      , ("language",        JStr langName)
      , ("ok",              JBool True)
      ]

handleMarketImpact :: Map String JVal -> Map String JVal
handleMarketImpact d =
  let qty  = getNum d "order_qty" 0.0
      adv  = let a = getNum d "adv" 1.0 in if a == 0 then 1.0 else a
      vol  = getNum d "volatility" 0.01
      impact = 10.0 * sqrt (qty / adv) * vol * 10000.0
  in Map.fromList
      [ ("impact_bps", JNum impact)
      , ("language",   JStr langName)
      , ("ok",         JBool True)
      ]

handleSignalFilter :: Map String JVal -> Map String JVal
handleSignalFilter d =
  let conf   = getNum d "confidence" 0.0
      regime = getStr d "regime" "unknown"
      vol    = getNum d "volatility" 0.0
      accept = conf >= 0.5 && (regime /= "high_vol" || vol < 0.02)
      reason = if conf < 0.5 then "low_confidence"
               else if regime == "high_vol" && vol >= 0.02 then "high_vol_regime"
               else "accepted"
  in Map.fromList
      [ ("accept",        JBool accept)
      , ("filter_reason", JStr reason)
      , ("language",      JStr langName)
      , ("ok",            JBool True)
      ]

handleConfidenceCalib :: Map String JVal -> Map String JVal
handleConfidenceCalib d =
  let confs   = getNumArr d "confidence_history"
      winRate = getNum d "win_rate" 0.5
      avgConf = if null confs then 0.5
                else sum confs / fromIntegral (length confs)
      calib   = 0.5 * avgConf + 0.5 * winRate
  in Map.fromList
      [ ("calibrated_confidence", JNum calib)
      , ("language",              JStr langName)
      , ("ok",                    JBool True)
      ]

handleHeartbeat :: Map String JVal -> Map String JVal
handleHeartbeat d =
  let cycleId = getNum d "cycle_id" 0.0
  in Map.fromList
      [ ("ok",         JBool True)
      , ("latency_ms", JNum 0.0)
      , ("language",   JStr langName)
      , ("cycle_id",   JNum cycleId)
      ]

handleVaR :: Map String JVal -> Map String JVal
handleVaR d =
  let returns = getNumArr d "returns"
      conf    = getNum d "confidence_level" 0.95
  in if length returns < 5
     then Map.fromList [("var_pct",JNum 0.0),("cvar_pct",JNum 0.0),("language",JStr langName),("ok",JBool True)]
     else
       let arr    = sortBy compare returns
           idx    = max 0 $ min (floor ((1.0 - conf) * fromIntegral (length arr))) (length arr - 1)
           varPct = negate (arr !! idx) * 100.0
           cvarSub = take (idx + 1) arr
           cvarPct = if null cvarSub then varPct
                     else negate (sum cvarSub / fromIntegral (length cvarSub)) * 100.0
       in Map.fromList
           [ ("var_pct",  JNum varPct)
           , ("cvar_pct", JNum cvarPct)
           , ("language", JStr langName)
           , ("ok",       JBool True)
           ]

handleSkew :: Map String JVal -> Map String JVal
handleSkew d =
  let returns = getNumArr d "returns"
      n       = length returns
  in if n < 3
     then Map.fromList [("skew",JNum 0.0),("language",JStr langName),("ok",JBool True)]
     else
       let fn   = fromIntegral n
           mean = sum returns / fn
           diffs = map (\x -> x - mean) returns
           var  = sum (map (\x -> x*x) diffs) / fn
           std  = sqrt var
           skew = if std == 0 then 0.0
                  else sum (map (\x -> x*x*x) diffs) / (fn * std * std * std)
       in Map.fromList
           [ ("skew",     JNum skew)
           , ("language", JStr langName)
           , ("ok",       JBool True)
           ]

handleOBImbalanceSeries :: Map String JVal -> Map String JVal
handleOBImbalanceSeries d =
  let snapshots = getArr d "snapshots"
      series = map computeImbal snapshots
      trend  = if length series < 2 then "flat"
               else let first = head series
                        last' = last series
                    in if last' > first + 0.05 then "up"
                       else if last' < first - 0.05 then "down"
                       else "flat"
  in Map.fromList
      [ ("imbalance_series", JArr (map JNum series))
      , ("trend",            JStr trend)
      , ("language",         JStr langName)
      , ("ok",               JBool True)
      ]
  where
    computeImbal (JObj m) =
      let bids = getNumArr m "bids"
          asks = getNumArr m "asks"
          sb   = sum (take 5 bids)
          sa   = sum (take 5 asks)
      in if (sb + sa) == 0 then 0.0 else (sb - sa) / (sb + sa)
    computeImbal _ = 0.0

handleExecQuality :: Map String JVal -> Map String JVal
handleExecQuality d =
  let trades = getArr d "trades"
      slippages = map computeSlip trades
      avgSlip   = if null slippages then 0.0
                  else sum slippages / fromIntegral (length slippages)
      score     = max 0.0 (1.0 - avgSlip / 50.0)
  in Map.fromList
      [ ("score_0_1",        JNum score)
      , ("avg_slippage_bps", JNum avgSlip)
      , ("language",         JStr langName)
      , ("ok",               JBool True)
      ]
  where
    computeSlip (JObj m) =
      let fill     = getNum m "fill_price" 0.0
          decision = getNum m "decision_price" 1.0
      in if decision == 0 then 0.0 else abs (fill - decision) / decision * 10000.0
    computeSlip _ = 0.0

handleRegimeDuration :: Map String JVal -> Map String JVal
handleRegimeDuration d =
  let history  = getArr d "regime_history"
      prices   = getNumArr d "prices"
      bars     = if not (null history)
                 then length history
                 else min 10 (length prices)
      regime   = case history of
                   (JStr s:_) -> s
                   _          -> getStr d "regime" "unknown"
      stable   = bars >= 5
  in Map.fromList
      [ ("bars_in_regime", JNum (fromIntegral bars))
      , ("regime_stable",  JBool stable)
      , ("regime",         JStr regime)
      , ("language",       JStr langName)
      , ("ok",             JBool True)
      ]

-- ─────────────────────────────────────────────
-- Formal risk verification tasks
-- ─────────────────────────────────────────────
handleKellyBounds :: Map String JVal -> Map String JVal
handleKellyBounds d =
  let winRate  = getNum d "win_rate" 0.0
      avgWin   = getNum d "avg_win" 0.0
      avgLoss  = getNum d "avg_loss" 0.0
      proposed = getNum d "proposed_fraction" 0.0
      p        = winRate
      q        = 1.0 - p
      b        = if avgLoss == 0 then 0.0 else avgWin / avgLoss
      kellyFull    = if b == 0 then 0.0 else (p * b - q) / b
      kellyHalf    = kellyFull / 2.0
      kellyQuarter = kellyFull / 4.0
      maxAllowed   = min kellyHalf 0.25
      verdict
        | proposed <= maxAllowed = "SAFE"
        | proposed <= kellyHalf  = "WARNING"
        | otherwise              = "REJECT"
  in Map.fromList
      [ ("kelly_full",    JNum kellyFull)
      , ("kelly_half",    JNum kellyHalf)
      , ("kelly_quarter", JNum kellyQuarter)
      , ("proposed",      JNum proposed)
      , ("verdict",       JStr verdict)
      , ("max_allowed",   JNum maxAllowed)
      , ("language",      JStr langName)
      ]

handleDrawdownCascadeCheck :: Map String JVal -> Map String JVal
handleDrawdownCascadeCheck d =
  let currentDd   = getNum d "current_drawdown_pct" 0.0
      posSize     = getNum d "position_size_pct" 0.0
      winRate     = getNum d "win_rate" 0.5
      maxConsLoss = round (getNum d "max_consecutive_losses" 10.0) :: Int
      lossProb    = 1.0 - winRate
      probRuin    = lossProb ^ maxConsLoss
      worstCaseDd = currentDd + posSize * fromIntegral maxConsLoss
      verdict
        | worstCaseDd > 50.0 || probRuin > 0.01 = "REJECT"
        | otherwise                              = "SAFE"
  in Map.fromList
      [ ("prob_ruin",        JNum probRuin)
      , ("worst_case_dd_pct", JNum worstCaseDd)
      , ("current_dd_pct",   JNum currentDd)
      , ("verdict",          JStr verdict)
      , ("language",         JStr langName)
      ]

handleRiskInvariants :: Map String JVal -> Map String JVal
handleRiskInvariants d =
  let capital       = getNum d "capital" 0.0
      positionValue = getNum d "position_value" 0.0
      numPositions  = round (getNum d "num_positions" 0.0) :: Int
      dailyPnl      = getNum d "daily_pnl" 0.0
      maxDailyLoss  = getNum d "max_daily_loss_pct" 5.0
      maxPosPct     = getNum d "max_position_pct" 15.0
      maxPos        = round (getNum d "max_positions" 6.0) :: Int
      -- Invariant 1: position concentration
      posRatio      = if capital == 0 then 1.0e9 else positionValue / capital
      inv1Passed    = posRatio <= maxPosPct / 100.0
      inv1          = Map.fromList
                        [ ("name",   JStr "position_concentration")
                        , ("passed", JBool inv1Passed)
                        , ("value",  JNum posRatio)
                        , ("limit",  JNum (maxPosPct / 100.0))
                        ]
      -- Invariant 2: max positions count
      inv2Passed    = numPositions <= maxPos
      inv2          = Map.fromList
                        [ ("name",   JStr "max_positions")
                        , ("passed", JBool inv2Passed)
                        , ("value",  JNum (fromIntegral numPositions))
                        , ("limit",  JNum (fromIntegral maxPos))
                        ]
      -- Invariant 3: daily loss limit (only when dailyPnl < 0)
      dailyLossRatio = if capital == 0 then 1.0e9 else abs dailyPnl / capital * 100.0
      inv3Passed     = if dailyPnl < 0 then dailyLossRatio <= maxDailyLoss else True
      inv3           = Map.fromList
                        [ ("name",   JStr "daily_loss_limit")
                        , ("passed", JBool inv3Passed)
                        , ("value",  JNum dailyLossRatio)
                        , ("limit",  JNum maxDailyLoss)
                        ]
      -- Invariant 4: capital positive
      inv4Passed    = capital > 0
      inv4          = Map.fromList
                        [ ("name",   JStr "capital_positive")
                        , ("passed", JBool inv4Passed)
                        , ("value",  JNum capital)
                        , ("limit",  JNum 0.0)
                        ]
      allPassed     = inv1Passed && inv2Passed && inv3Passed && inv4Passed
      verdict       = if allPassed then "PASS" else "BREACH"
  in Map.fromList
      [ ("all_passed", JBool allPassed)
      , ("invariants", JArr [ JObj inv1, JObj inv2, JObj inv3, JObj inv4 ])
      , ("verdict",    JStr verdict)
      , ("language",   JStr langName)
      ]

-- ─────────────────────────────────────────────
-- Dispatch
-- ─────────────────────────────────────────────
dispatch :: String -> Map String JVal -> Either String (Map String JVal)
dispatch taskType dat = case taskType of
  "cycle_plan"                  -> Right (handleCyclePlan dat)
  "order_book_processing"       -> Right (handleOrderBook dat)
  "risk_calculation"            -> Right (handleRisk dat)
  "volatility_estimate"         -> Right (handleVolatility dat)
  "signal_score"                -> Right (handleSignalScore dat)
  "regime_estimate"             -> Right (handleRegime dat)
  "slippage_estimate"           -> Right (handleSlippage dat)
  "position_sizing"             -> Right (handlePositionSizing dat)
  "drawdown_check"              -> Right (handleDrawdown dat)
  "correlation_estimate"        -> Right (handleCorrelation dat)
  "liquidity_score"             -> Right (handleLiquidity dat)
  "market_impact"               -> Right (handleMarketImpact dat)
  "signal_filter"               -> Right (handleSignalFilter dat)
  "confidence_calibration"      -> Right (handleConfidenceCalib dat)
  "heartbeat"                   -> Right (handleHeartbeat dat)
  "var_estimate"                -> Right (handleVaR dat)
  "skew_estimate"               -> Right (handleSkew dat)
  "order_book_imbalance_series" -> Right (handleOBImbalanceSeries dat)
  "execution_quality_score"     -> Right (handleExecQuality dat)
  "regime_duration"             -> Right (handleRegimeDuration dat)
  "kelly_bounds"                -> Right (handleKellyBounds dat)
  "drawdown_cascade_check"      -> Right (handleDrawdownCascadeCheck dat)
  "risk_invariants"             -> Right (handleRiskInvariants dat)
  _                             -> Left ("unknown task: " ++ taskType)

-- ─────────────────────────────────────────────
-- Process one input line
-- ─────────────────────────────────────────────
processLine :: String -> Double -> String
processLine line tookMs =
  case parseJSON line of
    Nothing -> encodeJSON $ JObj $ Map.fromList
      [ ("ok",    JBool False)
      , ("error", JStr "json_parse_error")
      , ("took_ms", JNum tookMs)
      ]
    Just (JObj top) ->
      let taskType = getStr top "task_type" ""
          dat      = case Map.lookup "data" top of
                       Just (JObj m) -> m
                       _             -> Map.empty
      in case dispatch taskType dat of
           Left err ->
             encodeJSON $ JObj $ Map.fromList
               [ ("ok",      JBool False)
               , ("error",   JStr err)
               , ("took_ms", JNum tookMs)
               ]
           Right result ->
             encodeJSON $ JObj $ Map.fromList
               [ ("ok",      JBool True)
               , ("result",  JObj result)
               , ("took_ms", JNum tookMs)
               ]
    Just _ -> encodeJSON $ JObj $ Map.fromList
      [ ("ok",    JBool False)
      , ("error", JStr "expected_json_object")
      , ("took_ms", JNum tookMs)
      ]

-- ─────────────────────────────────────────────
-- Main loop
-- ─────────────────────────────────────────────
main :: IO ()
main = do
  hSetBuffering stdin  LineBuffering
  hSetBuffering stdout LineBuffering
  hSetBuffering stderr LineBuffering
  loop
  where
    loop = do
      eof <- isEOF_safe
      when (not eof) $ do
        line <- getLine_safe
        case line of
          Nothing -> return ()
          Just ln -> do
            result <- try (evaluate (processLine ln 0.0)) :: IO (Either SomeException String)
            case result of
              Left ex  -> do
                let errOut = encodeJSON $ JObj $ Map.fromList
                              [ ("ok",      JBool False)
                              , ("error",   JStr (show ex))
                              , ("took_ms", JNum 0.0)
                              ]
                hPutStrLn stdout errOut
                hFlush stdout
              Right out -> do
                hPutStrLn stdout out
                hFlush stdout
            loop

isEOF_safe :: IO Bool
isEOF_safe = do
  r <- try isEOF :: IO (Either SomeException Bool)
  case r of
    Left _  -> return True
    Right b -> return b

getLine_safe :: IO (Maybe String)
getLine_safe = do
  r <- try getLine :: IO (Either SomeException String)
  case r of
    Left _  -> return Nothing
    Right s -> return (Just s)
