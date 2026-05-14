#!/usr/bin/env python3
import pickle, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

print('Loading data...')
with open('data/historical/historical_data.pkl', 'rb') as f:
    data = pickle.load(f)

def process(sd):
    ohlcv = sd['1h']
    base = pd.DataFrame(ohlcv)
    base['datetime'] = pd.to_datetime(base['timestamp'], unit='ms')
    base = base.set_index('datetime').sort_index()
    
    f = pd.DataFrame(index=base.index)
    f['r1'] = base['close'].pct_change(1)
    f['r4'] = base['close'].pct_change(4)
    f['r24'] = base['close'].pct_change(24)
    f['v12'] = f['r1'].rolling(12).std()
    f['v24'] = f['r1'].rolling(24).std()
    
    delta = base['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    f['rsi'] = 100 - (100 / (1 + gain / loss.clip(lower=1e-8)))
    
    f['pp'] = (base['close'] - base['low'].rolling(24).min()) / (base['high'].rolling(24).max() - base['low'].rolling(24).min()).clip(lower=1e-8)
    f['vr'] = base['volume'] / base['volume'].rolling(24).mean().clip(lower=1e-8)
    
    fwd = base['close'].pct_change(4).shift(-4)
    fwd24 = base['close'].pct_change(24).shift(-24)
    
    l = pd.DataFrame(index=base.index)
    l['sig'] = pd.cut(fwd, bins=[-np.inf,-0.01,0.01,np.inf], labels=[0,1,2])
    l['reg'] = pd.cut(fwd24, bins=[-np.inf,-0.03,0.03,np.inf], labels=[0,1,2])
    l['pos'] = np.clip(np.abs(fwd)*30, 0, 1)
    l['vol'] = fwd24.rolling(24).std().shift(-24)
    l['trend'] = np.abs((fwd>0).rolling(24).mean()-0.5)*2
    
    return f, l

all_f, all_l = [], []
for sym, sd in data.items():
    f, l = process(sd)
    c = pd.concat([f,l], axis=1).dropna()
    if len(c) > 500:
        all_f.append(c[f.columns])
        all_l.append(c)
        print(f'{sym}: {len(c)} samples')

X = pd.concat(all_f, ignore_index=True).replace([np.inf,-np.inf], np.nan).fillna(0)
y = pd.concat(all_l, ignore_index=True)
print(f'Total: {len(X)}, Features: {len(X.columns)}')

sc = StandardScaler()
Xs = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
sp = int(len(X)*0.8)
Xtr, Xte = Xs.iloc[:sp], Xs.iloc[sp:]

out = Path('data/models_mtf')
out.mkdir(exist_ok=True)
metrics = json.load(open(out/'metrics.json')) if (out/'metrics.json').exists() else {}

# Regime
print('\nRegime Classifier...')
ytr, yte = y['reg'].iloc[:sp], y['reg'].iloc[sp:]
m = GradientBoostingClassifier(n_estimators=80, max_depth=5).fit(Xtr, ytr)
tr, te = m.score(Xtr, ytr), m.score(Xte, yte)
print(f'  Train: {tr:.4f}, Test: {te:.4f}')
metrics['regime_classifier'] = {'train': round(tr,4), 'test': round(te,4)}
pickle.dump(m, open(out/'regime_classifier.pkl','wb'))

# Position
print('\nPosition Sizer...')
ytr, yte = y['pos'].iloc[:sp], y['pos'].iloc[sp:]
m = GradientBoostingRegressor(n_estimators=80, max_depth=5).fit(Xtr, ytr)
tr, te = m.score(Xtr, ytr), m.score(Xte, yte)
print(f'  Train R2: {tr:.4f}, Test R2: {te:.4f}')
metrics['position_sizer'] = {'train_r2': round(tr,4), 'test_r2': round(te,4)}
pickle.dump(m, open(out/'position_sizer.pkl','wb'))

# Volatility
print('\nVolatility Model...')
ytr, yte = y['vol'].iloc[:sp], y['vol'].iloc[sp:]
m = GradientBoostingRegressor(n_estimators=80, max_depth=5).fit(Xtr, ytr)
tr, te = m.score(Xtr, ytr), m.score(Xte, yte)
print(f'  Train R2: {tr:.4f}, Test R2: {te:.4f}')
metrics['volatility_model'] = {'train_r2': round(tr,4), 'test_r2': round(te,4)}
pickle.dump(m, open(out/'volatility_model.pkl','wb'))

# Trend
print('\nTrend Strength...')
ytr, yte = y['trend'].iloc[:sp], y['trend'].iloc[sp:]
m = GradientBoostingRegressor(n_estimators=80, max_depth=5).fit(Xtr, ytr)
tr, te = m.score(Xtr, ytr), m.score(Xte, yte)
print(f'  Train R2: {tr:.4f}, Test R2: {te:.4f}')
metrics['trend_strength'] = {'train_r2': round(tr,4), 'test_r2': round(te,4)}
pickle.dump(m, open(out/'trend_strength.pkl','wb'))

# Regime-specific
print('\nRegime-Specific Models...')
reg_model = pickle.load(open(out/'regime_classifier.pkl','rb'))
preds = reg_model.predict(Xs)
for rid, rname in [(0,'bear'),(1,'sideways'),(2,'bull')]:
    mask = preds == rid
    Xr, yr = Xs[mask], y['sig'].iloc[mask]
    if len(Xr) < 500:
        print(f'  {rname}: skipped ({len(Xr)})')
        continue
    sr = int(len(Xr)*0.8)
    m = GradientBoostingClassifier(n_estimators=60, max_depth=5).fit(Xr.iloc[:sr], yr.iloc[:sr])
    acc = m.score(Xr.iloc[sr:], yr.iloc[sr:])
    print(f'  {rname}: {acc:.4f} ({len(Xr)})')
    metrics[f'signal_{rname}'] = {'test': round(acc,4), 'samples': len(Xr)}
    pickle.dump(m, open(out/f'signal_{rname}.pkl','wb'))

pickle.dump(sc, open(out/'scaler.pkl','wb'))
pickle.dump(list(X.columns), open(out/'feature_names.pkl','wb'))
json.dump(metrics, open(out/'metrics.json','w'), indent=2)

print('\n' + '='*70)
print('TRAINING COMPLETE - SUMMARY')
print('='*70)
for n, m in metrics.items():
    if 'test' in m:
        print(f'  {n}: accuracy={m["test"]:.4f}')
    elif 'test_r2' in m:
        print(f'  {n}: R2={m["test_r2"]:.4f}')
print(f'\nModels saved to: {out}')
