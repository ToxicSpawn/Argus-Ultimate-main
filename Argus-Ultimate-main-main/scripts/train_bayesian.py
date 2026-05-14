import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

print("Loading data...")
with open('data/historical/historical_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Quick feature extraction
def make_features(sd):
    import pandas as pd
    base = pd.DataFrame(sd['1h'])
    f = pd.DataFrame(index=base.index)
    f['r1'] = base['close'].pct_change(1)
    f['r4'] = base['close'].pct_change(4)
    f['r24'] = base['close'].pct_change(24)
    f['v12'] = f['r1'].rolling(12).std()
    delta = base['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    f['rsi'] = 100 - (100 / (1 + gain / loss.clip(lower=1e-8)))
    f['pp'] = (base['close'] - base['low'].rolling(24).min()) / (base['high'].rolling(24).max() - base['low'].rolling(24).min()).clip(lower=1e-8)
    f['vol'] = base['volume'] / base['volume'].rolling(24).mean()
    fwd = base['close'].pct_change(4).shift(-4)
    y = pd.cut(fwd, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])
    data_out = pd.concat([f, y], axis=1).dropna()
    return data_out.values[:, :-1].astype(np.float32), data_out.values[:, -1].astype(np.int64)

all_X, all_y = [], []
for sym in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
    X, y = make_features(data[sym])
    all_X.append(X)
    all_y.append(y)
    print(f"  {sym}: {len(X)}")

X = np.vstack(all_X).astype(np.float32)
y = np.concatenate(all_y).astype(np.int64)
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

sc = StandardScaler()
Xs = sc.fit_transform(X).astype(np.float32)

split = int(len(Xs) * 0.8)
Xtr_t = torch.FloatTensor(Xs[:split])
Xte_t = torch.FloatTensor(Xs[split:])
ytr_t = torch.LongTensor(y[:split])
yte_t = torch.LongTensor(y[split:])

model_path = Path('data/models_deep')
n_features = X.shape[1]

print(f"\nTraining Bayesian NN ({n_features} features)...")

class BayesianNN(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

model = BayesianNN(n_features)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()

for epoch in range(25):
    model.train()
    opt.zero_grad()
    out = model(Xtr_t)
    loss = crit(out, ytr_t)
    loss.backward()
    opt.step()
    
    if (epoch+1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
        print(f'  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.4f}')

# Save
torch.save(model.state_dict(), model_path / 'bayesian_nn.pth')
print("\nBayesian NN saved successfully!")
