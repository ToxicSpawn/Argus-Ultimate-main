#!/usr/bin/env python3
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

print('='*70)
print('COMPLETE ML TRAINING - 3 YEARS DATA')
print('='*70)

with open('data/historical/historical_data.pkl', 'rb') as f:
    data = pickle.load(f)

def process(symbol_data):
    base = pd.DataFrame(symbol_data['1h'])
    base['datetime'] = pd.to_datetime(base['timestamp'], unit='ms')
    base.set_index('datetime', inplace=True)
    base = base.sort_index()
    
    feats = pd.DataFrame(index=base.index)
    feats['ret_1'] = base['close'].pct_change(1)
    feats['ret_4'] = base['close'].pct_change(4)
    feats['ret_12'] = base['close'].pct_change(12)
    feats['ret_24'] = base['close'].pct_change(24)
    feats['ret_48'] = base['close'].pct_change(48)
    feats['vol_12'] = feats['ret_1'].rolling(12).std()
    feats['vol_24'] = feats['ret_1'].rolling(24).std()
    feats['vol_48'] = feats['ret_1'].rolling(48).std()
    
    delta = base['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    feats['rsi'] = 100 - (100 / (1 + gain / loss.clip(lower=1e-8)))
    
    feats['price_pos'] = (base['close'] - base['low'].rolling(24).min()) / (base['high'].rolling(24).max() - base['low'].rolling(24).min()).clip(lower=1e-8)
    feats['vol_ratio'] = base['volume'] / base['volume'].rolling(24).mean().clip(lower=1e-8)
    feats['atr'] = ((base['high'] - base['low']).rolling(14).mean()) / base['close']
    
    bb_mid = base['close'].rolling(20).mean()
    bb_std = base['close'].rolling(20).std()
    feats['bb_pos'] = (base['close'] - (bb_mid - 2*bb_std)) / (4*bb_std).clip(lower=1e-8)
    
    fwd_4h = base['close'].pct_change(4).shift(-4)
    fwd_24h = base['close'].pct_change(24).shift(-24)
    
    labels = pd.DataFrame(index=base.index)
    labels['signal'] = pd.cut(fwd_4h, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])
    labels['regime'] = pd.cut(fwd_24h, bins=[-np.inf, -0.03, 0.03, np.inf], labels=[0, 1, 2])
    labels['position_size'] = np.clip(np.abs(fwd_4h) * 30, 0, 1)
    labels['volatility'] = fwd_24h.rolling(24).std().shift(-24)
    labels['trend_strength'] = np.abs((fwd_4h > 0).rolling(24).mean() - 0.5) * 2
    
    return feats, labels

all_feats, all_labels = [], []
for sym, sd in data.items():
    f, l = process(sd)
    c = pd.concat([f, l], axis=1).dropna()
    if len(c) > 1000:
        all_feats.append(c[f.columns])
        all_labels.append(c)
        print(f'{sym}: {len(c)} samples')

X = pd.concat(all_feats, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)
y = pd.concat(all_labels, ignore_index=True)

print(f'\nTotal: {len(X):,} samples, {len(X.columns)} features')

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
split = int(len(X) * 0.8)
X_train, X_test = X_scaled.iloc[:split], X_scaled.iloc[split:]

output = Path('data/models_mtf')
output.mkdir(exist_ok=True)
metrics = {}

# 1. Signal Classifier
print('\n[1/5] SIGNAL CLASSIFIER')
y_train, y_test = y['signal'].iloc[:split], y['signal'].iloc[split:]
model = GradientBoostingClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f'  Train: {train_acc:.4f}, Test: {test_acc:.4f}')
metrics['signal_classifier'] = {'train': round(train_acc,4), 'test': round(test_acc,4)}
pickle.dump(model, open(output/'signal_classifier.pkl','wb'))

# 2. Regime Classifier
print('\n[2/5] REGIME CLASSIFIER')
y_train, y_test = y['regime'].iloc[:split], y['regime'].iloc[split:]
model = GradientBoostingClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f'  Train: {train_acc:.4f}, Test: {test_acc:.4f}')
metrics['regime_classifier'] = {'train': round(train_acc,4), 'test': round(test_acc,4)}
pickle.dump(model, open(output/'regime_classifier.pkl','wb'))

# 3. Position Sizer
print('\n[3/5] POSITION SIZER')
y_train, y_test = y['position_size'].iloc[:split], y['position_size'].iloc[split:]
model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
print(f'  Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}')
metrics['position_sizer'] = {'train_r2': round(train_r2,4), 'test_r2': round(test_r2,4)}
pickle.dump(model, open(output/'position_sizer.pkl','wb'))

# 4. Volatility Model
print('\n[4/5] VOLATILITY MODEL')
y_train, y_test = y['volatility'].iloc[:split], y['volatility'].iloc[split:]
model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
print(f'  Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}')
metrics['volatility_model'] = {'train_r2': round(train_r2,4), 'test_r2': round(test_r2,4)}
pickle.dump(model, open(output/'volatility_model.pkl','wb'))

# 5. Trend Strength
print('\n[5/5] TREND STRENGTH')
y_train, y_test = y['trend_strength'].iloc[:split], y['trend_strength'].iloc[split:]
model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
print(f'  Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}')
metrics['trend_strength'] = {'train_r2': round(train_r2,4), 'test_r2': round(test_r2,4)}
pickle.dump(model, open(output/'trend_strength.pkl','wb'))

# 6. Regime-specific models
print('\n[BONUS] REGIME-SPECIFIC SIGNAL MODELS')
regime_names = {0: 'bear', 1: 'sideways', 2: 'bull'}
regime_model = pickle.load(open(output/'regime_classifier.pkl','rb'))
regime_preds = regime_model.predict(X_scaled)

for regime_id, regime_name in regime_names.items():
    mask = regime_preds == regime_id
    X_r = X_scaled[mask]
    y_r = y['signal'].iloc[mask]
    
    if len(X_r) < 1000:
        print(f'  {regime_name}: skipped ({len(X_r)} samples)')
        continue
    
    split_r = int(len(X_r) * 0.8)
    model = GradientBoostingClassifier(n_estimators=80, max_depth=5)
    model.fit(X_r.iloc[:split_r], y_r.iloc[:split_r])
    acc = model.score(X_r.iloc[split_r:], y_r.iloc[split_r:])
    print(f'  {regime_name}: test accuracy={acc:.4f} ({len(X_r):,} samples)')
    metrics[f'signal_{regime_name}'] = {'test': round(acc,4), 'samples': len(X_r)}
    pickle.dump(model, open(output/f'signal_{regime_name}.pkl','wb'))

pickle.dump(scaler, open(output/'scaler.pkl','wb'))
pickle.dump(list(X.columns), open(output/'feature_names.pkl','wb'))
json.dump(metrics, open(output/'metrics.json','w'), indent=2)

print('\n' + '='*70)
print('TRAINING COMPLETE - SUMMARY')
print('='*70)
for name, m in metrics.items():
    test_val = m.get('test', m.get('test_r2', 0))
    label = 'accuracy' if 'test' in m else 'R2'
    print(f'  {name}: {label}={test_val:.4f}')
print(f'\nModels saved to: {output}')
