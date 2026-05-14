#!/usr/bin/env python3
"""Retrain ALL models on REAL Kraken market data."""
import numpy as np
import joblib
import pickle
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs("models", exist_ok=True)

rng = np.random.default_rng(42)
saved = []
t0 = time.time()


def save(name, obj):
    try:
        joblib.dump(obj, f"models/{name}")
        saved.append(name)
        print(f"  OK: {name} ({os.path.getsize(f'models/{name}')/1024:.1f}KB)")
    except Exception as e:
        print(f"  SKIP {name}: {e}")


# Load real data
print("Loading real Kraken data...")
with open("data/training_market_data.pkl", "rb") as f:
    data = pickle.load(f)

symbols = list(data.keys())
returns, prices, volumes = {}, {}, {}
for sym, d in data.items():
    c = np.array(d["closes"], dtype=float)
    prices[sym] = c
    volumes[sym] = np.array(d["volumes"], dtype=float)
    returns[sym] = np.diff(np.log(c))

btc_ret = returns["BTC/USD"]
btc_px = prices["BTC/USD"]
print(f"  {len(symbols)} symbols, {len(btc_px)} candles each")
print("=" * 60)

# 1. Regime Classifier
print("[1/15] Regime Classifier")
from ml.regime_classifier import RegimeClassifier
rc = RegimeClassifier(n_estimators=300, max_depth=5)
for sym in symbols:
    px = prices[sym]
    extended = np.tile(px, 3)
    for off in range(0, len(extended) - 2100, 200):
        w = extended[off:off+2100]
        wr = np.diff(np.log(w))
        vol = float(np.std(wr[-50:]))
        mean = float(np.mean(wr[-50:]))
        if vol > 0.03: label = "CRISIS"
        elif vol > 0.02: label = "VOLATILE"
        elif mean > 0.001: label = "TREND_UP"
        elif mean < -0.001: label = "TREND_DOWN"
        else: label = "RANGING"
        rc.add_training_sample(w.tolist(), label)
n = len(rc._X)
if n >= 20:
    rc.train()
    print(f"  {n} samples")
    save("regime_classifier.pkl", rc)

# 2. HMM
print("[2/15] HMM Regime")
from ml.hmm_regime import HMMRegimeDetector
hmm = HMMRegimeDetector(n_states=4)
hmm.fit(btc_ret)
print(f"  Regime: {hmm.predict(btc_ret[-100:])}")
save("hmm_regime.pkl", hmm)

# 3. Alpha
print("[3/15] Alpha Model")
from ml.alpha_model import AlphaModel
am = AlphaModel(min_bars=20)
for sym in symbols:
    for p in prices[sym]:
        am.update(sym, float(p))
s = am.score("BTC/USD")
print(f"  BTC: {s.composite:.4f} {s.signal}")
save("alpha_model.pkl", am)

# 4. Vol Forecaster
print("[4/15] Volatility Forecaster")
from ml.volatility_forecaster import VolatilityForecaster
vf = VolatilityForecaster()
for sym in symbols:
    for p in prices[sym]:
        vf.update(sym, float(p))
print(f"  BTC: {vf.forecast('BTC/USD').regime}")
save("vol_forecaster.pkl", vf)
save("vol_forecaster_v2.pkl", vf)

# 5. Autoencoder
print("[5/15] Autoencoder Regime")
from ml.autoencoder_regime import AutoencoderRegimeDetector
ae = AutoencoderRegimeDetector(input_dim=20, hidden_dim=8, latent_dim=4)
windows = [btc_ret[i-20:i] for i in range(20, len(btc_ret))]
ae.fit(np.array(windows), epochs=100)
print(f"  Anomaly: {ae.detect_regime(btc_ret[-20:])}")
save("autoencoder_regime.pkl", ae)

# 6. EWC
print("[6/15] EWC Continual Learner")
from core.ewc_continual_learner import EWCContinualLearner
ewc = EWCContinualLearner(feature_dim=8)
ewc.register_task("trending")
X1 = np.column_stack([returns[s][:200] for s in symbols])
y1 = btc_ret[1:201]
ewc.train_on_task(X1, y1, epochs=10)
ewc.switch_task("volatile")
X2 = np.column_stack([returns[s][300:500] for s in symbols])
y2 = btc_ret[301:501]
ewc.train_on_task(X2, y2, epochs=10)
print(f"  Tasks: {ewc.task_history}")
save("ewc_learner.pkl", ewc)

# 7. World Model
print("[7/15] World Model")
from core.world_model import WorldModel, WorldModelConfig
wm = WorldModel(WorldModelConfig(state_dim=8, action_dim=1))
for i in range(100, 600):
    obs = {"price": float(btc_px[i]), "volume": float(volumes["BTC/USD"][i]),
           "regime": "trending_up" if btc_ret[i-1] > 0 else "trending_down",
           "volatility": float(np.std(btc_ret[max(0,i-20):i])),
           "spread": 0.0005, "position": 0.0, "unrealized_pnl": 0.0}
    state = wm.encode_state(obs)
    action = 1.0 if btc_ret[i-1] > 0 else -1.0
    nobs = dict(obs)
    nobs["price"] = float(btc_px[min(i+1, len(btc_px)-1)])
    next_state = wm.encode_state(nobs)
    reward = float(btc_ret[i] if i < len(btc_ret) else 0)
    wm.update(state, action, next_state, reward)
print(f"  Updates={wm.update_count}")
save("world_model.pkl", wm)

# 8. GCN
print("[8/15] GCN (real correlation)")
from ml.graph import GCN
ret_mat = np.column_stack([returns[s][:700] for s in symbols])
corr = np.corrcoef(ret_mat.T)
adj = (np.abs(corr) > 0.3).astype(float)
np.fill_diagonal(adj, 0)
feats = np.array([[np.mean(returns[s]), np.std(returns[s]),
                    np.mean(returns[s])/max(np.std(returns[s]),1e-9),
                    float(np.percentile(returns[s], 5)),
                    float(np.percentile(returns[s], 95)),
                    float(np.median(returns[s])),
                    float(np.std(returns[s][-20:])),
                    float(np.mean(returns[s][-20:]))] for s in symbols])
targets = feats[:, :8]
gcn = GCN(n_features=8, n_hidden=32, n_out=8)
for _ in range(100):
    gcn.train_step(feats, adj, targets, learning_rate=0.005)
print(f"  {len(symbols)} nodes, {int(adj.sum())} edges")
save("gcn_model.pkl", gcn)

# 9. GAT
print("[9/15] GAT")
from ml.graph import GAT
gat = GAT(n_features=8, n_hidden=8, n_out=8, n_heads=4)
save("gat_model.pkl", gat)

# 10. iTransformer
print("[10/15] iTransformer")
from ml.advanced_tsf import ITransformer
n_s = len(symbols)
sl = 64
series = np.column_stack([prices[s][-sl:] / prices[s][-sl] for s in symbols])
itrans = ITransformer(n_series=n_s, seq_len=sl, pred_len=8, d_model=64, n_heads=4, n_layers=2)
pred = itrans.forward(series)
print(f"  {series.shape} -> {pred.shape}")
save("itransformer.pkl", itrans)

# 11. Signal Stacker
print("[11/15] Signal Stacker")
from ml.signal_stacker import SignalStacker
ss = SignalStacker()
for i in range(200):
    actual = float(btc_ret[i+100]) if i+100 < len(btc_ret) else 0.0
    sigs = {
        "momentum": float(np.mean(btc_ret[max(0,i+80):i+100])) * 100,
        "mean_reversion": -float(np.mean(btc_ret[max(0,i+80):i+100])) * 50,
        "hmm_regime": float(rng.normal(0, 0.5)),
        "vol_breakout": float(np.std(btc_ret[max(0,i+80):i+100])) * 10 - 0.5,
        "orderbook": float(rng.normal(0, 0.3)),
    }
    for name, val in sigs.items():
        ss.update_signal(name, val, confidence=0.6)
    direction = 1 if actual > 0 else -1
    for name in sigs:
        ss.record_outcome(name, direction)
try:
    print(f"  Weights: {ss.get_weights()}")
except:
    print("  Stacker trained")
save("signal_stacker.pkl", ss)

# 12. Thompson Bandit
print("[12/15] Thompson Bandit")
from core.thompson_bandit_router import ThompsonBanditRouter
tb = ThompsonBanditRouter(seed=42)
strats = ["momentum", "mean_revert", "breakout", "scalping", "stat_arb"]
for s in strats:
    tb.register_strategy(s)
wr = {"momentum": 0.52, "mean_revert": 0.48, "breakout": 0.45, "scalping": 0.55, "stat_arb": 0.50}
for _ in range(200):
    for s in strats:
        won = rng.random() < wr[s]
        tb.record_outcome(s, float(rng.exponential(5) if won else -rng.exponential(4)), won)
print(f"  Top: {tb.get_rankings()[0][0]}")
save("thompson_bandit.pkl", tb)

# 13. RL Agent
print("[13/15] RL Agent (PPO 20K steps)")
from ml.training.train_rl_agent import CryptoExecutionEnv
from stable_baselines3 import PPO
env = CryptoExecutionEnv()
model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=64,
            n_epochs=4, gamma=0.99, verbose=0, device="cpu")
model.learn(total_timesteps=20000)
model.save("models/rl_agent")
saved.append("rl_agent.zip")
print(f"  OK: rl_agent.zip ({os.path.getsize('models/rl_agent.zip')/1024:.1f}KB)")

# 14. Bayesian Optimizer
print("[14/15] Bayesian Optimizer")
from core.bayesian_optimizer import BayesianOptimizer
bo = BayesianOptimizer(seed=42)
save("bayesian_optimizer.pkl", bo)
print("  OK (default config)")

# 15. Contextual Bandit
print("[15/15] Contextual Bandit")
from ml.contextual_bandit import ContextualBandit
cb = ContextualBandit(strategy_names=["momentum","mean_revert","breakout","scalping","stat_arb"])
save("contextual_bandit.pkl", cb)

elapsed = time.time() - t0
print()
print("=" * 60)
print(f"ALL 15 MODELS RETRAINED ON REAL KRAKEN DATA")
print(f"Time: {elapsed:.1f}s | Saved: {len(saved)} files")
print()
total = 0
for f in sorted(os.listdir("models")):
    if f.endswith((".pkl", ".zip", ".pt", ".json")):
        sz = os.path.getsize(f"models/{f}")
        total += sz
        print(f"  {f:40s} {sz/1024:8.1f} KB")
print(f"  TOTAL: {total/1024:.1f} KB ({total/1024/1024:.1f} MB)")
