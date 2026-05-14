# API Setup Guide - Get Argus Trading Live
## Where to Put Your API Keys

---

## 🚀 QUICK START (5 Minutes)

### **Step 1: Copy the Template File**

Open terminal/command prompt in the Argus folder:

```bash
# Windows:
copy .env.example .env

# Mac/Linux:
cp .env.example .env
```

---

### **Step 2: Edit .env File**

Open `.env` in any text editor and fill in your API keys:

```bash
# REQUIRED - Kraken Exchange (Primary)
KRAKEN_API_KEY=K-1234567890abcdef1234567890abcdef
KRAKEN_API_SECRET=abcdefghijklmnopqrstuvwxyz1234567890abcdef=
KRAKEN_SANDBOX=true        # Keep true for testing

# Trading Mode (IMPORTANT!)
TRADING_MODE=paper         # paper = fake money, live = real money

# Initial capital for paper trading
PAPER_INITIAL_BALANCE=1000.0
```

---

### **Step 3: Get Free API Keys**

#### **Kraken (Required - 2 minutes)**
1. Go to: https://www.kraken.com/
2. Create account (or login)
3. Go to: Settings → API → Generate New Key
4. Enable permissions: `Query Funds`, `Query Open Orders & Trades`, `Query Closed Orders & Trades`, `Create & Modify Orders`
5. Copy API Key and Private Key to .env

#### **Coinbase (Optional - 2 minutes)**
1. Go to: https://www.coinbase.com/advanced-trade
2. Settings → API → New API Key
3. Copy to .env

#### **External Data (Optional but recommended)**

**CoinGlass (Funding rates):**
- https://coinglass.com/pricing → Free tier
- Copy API key to: `COINGLASS_API_KEY=...`

**Whale Alert:**
- https://api.whale-alert.io/ → Sign up free
- Copy API key to: `WHALE_ALERT_API_KEY=...`

**NewsAPI:**
- https://newsapi.org/ → Free tier (100 requests/day)
- Copy API key to: `NEWS_API_KEY=...`

---

## 📁 WHERE FILES GO

```
Argus-Ultimate-main-1/
├── .env                          ← YOUR API KEYS GO HERE (created from .env.example)
├── .env.example                  ← Template (don't edit this)
├── config/
│   └── api_config.py             ← Reads your .env file
├── wiring/
│   └── argus_realtime_data_flow.py  ← Uses API keys to connect
└── API_SETUP_GUIDE.md            ← This file
```

---

## 🔐 SECURITY RULES

### **✅ DO:**
- ✅ Keep `.env` file secret
- ✅ Add `.env` to `.gitignore` (already done)
- ✅ Use `KRAKEN_SANDBOX=true` for testing
- ✅ Start with small amounts ($100)

### **❌ DON'T:**
- ❌ Never commit `.env` to GitHub
- ❌ Never share API keys
- ❌ Don't set `TRADING_MODE=live` until ready
- ❌ Don't enable withdrawals via API

---

## 🧪 TEST YOUR SETUP

### **Test 1: Check Config Loads**

```python
# Run this Python test:
from config.api_config import get_api_config

config = get_api_config()
print(config.get_summary())
```

**Expected output:**
```
{'trading_mode': 'paper', 'kraken_configured': True, 'is_valid': True, ...}
```

### **Test 2: Test Kraken Connection**

```python
# Run this to test API connection:
import asyncio
from config.api_config import get_api_config

async def test_kraken():
    config = get_api_config()
    
    if config.kraken_api_key:
        print("✅ Kraken API key configured")
        print(f"   Sandbox mode: {config.kraken_sandbox}")
        print(f"   Trading mode: {config.trading_mode}")
    else:
        print("❌ Kraken API key not found")

asyncio.run(test_kraken())
```

---

## 🎯 START TRADING

### **Option A: Paper Trading (Recommended First)**

```bash
# Edit .env:
TRADING_MODE=paper
KRAKEN_SANDBOX=true
PAPER_INITIAL_BALANCE=1000.0

# Then run:
python -m wiring.argus_omega_supreme
```

**What happens:**
- Uses real market data from Kraken
- Trades are simulated (no real money)
- Tests all 62 systems
- Safe to run 24/7

### **Option B: Live Trading (When Ready)**

```bash
# Edit .env:
TRADING_MODE=live
KRAKEN_SANDBOX=false

# Then run with small capital:
INITIAL_CAPITAL=100  # Start small!
python -m wiring.argus_omega_supreme
```

**⚠️ WARNING:** Only do this after paper trading works for 1 week!

---

## 🔧 TROUBLESHOOTING

### **"KRAKEN_API_KEY not set" Error**

```bash
# Check your .env file exists:
dir .env  # Windows
ls .env   # Mac/Linux

# If missing, create it:
copy .env.example .env

# Edit and add your keys
```

### **Invalid API Key Error**

```bash
# 1. Check key is copied correctly (no extra spaces)
# 2. Verify permissions on Kraken:
#    - Query Funds: ✅
#    - Query Orders: ✅
#    - Create Orders: ✅
#    - Withdraw: ❌ (keep disabled for safety)
```

### **Config Not Loading**

```python
# Make sure you're in the right directory:
import os
print(os.getcwd())  # Should show Argus-Ultimate-main-1

# Install python-dotenv:
pip install python-dotenv
```

---

## 📊 EXAMPLE .env FILE

```bash
# =============================================================================
# MINIMUM REQUIRED CONFIG
# =============================================================================

# Kraken (Required)
KRAKEN_API_KEY=K-1234567890abcdef1234567890abcdef
KRAKEN_API_SECRET=abcdefghijklmnopqrstuvwxyz1234567890abcdef=
KRAKEN_SANDBOX=true

# Trading Mode (paper = safe testing)
TRADING_MODE=paper
PAPER_INITIAL_BALANCE=1000.0

# Risk Limits (conservative)
MAX_POSITION_SIZE=0.05      # 5% max per trade
MAX_DRAWDOWN=0.10           # 10% max loss
DAILY_LOSS_LIMIT=100.0      # $100/day max
STOP_LOSS_PCT=0.02          # 2% stop loss

# Optional Data Sources (get better predictions)
COINGLASS_API_KEY=your_key_here
WHALE_ALERT_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

---

## 🚀 NEXT STEPS

1. ✅ **Copy .env.example to .env**
2. ✅ **Get Kraken API key** (2 min)
3. ✅ **Add key to .env file**
4. ✅ **Set TRADING_MODE=paper**
5. ✅ **Run test:** `python -c "from config.api_config import get_api_config; print(get_api_config().get_summary())"`
6. ✅ **Start Argus:** `python -m wiring.argus_omega_supreme`

**Once paper trading works for 1 week → Switch to live with $100 → Scale to $1,000**

---

## 💡 PRO TIPS

- **Start with paper trading** - Never risk real money on day 1
- **Use small positions** - Even when live, start with $100
- **Monitor closely** - Watch the logs for first few days
- **Keep backups** - Save your .env file somewhere safe
- **Don't share keys** - API keys = money access

**Questions? The .env.example file has detailed comments for every setting.**
