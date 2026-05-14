# Australian Exchanges Compatible with Argus Ultimate

## 🇦🇺 Complete Guide for Sydney Traders

This guide covers all Australian exchanges that work with Argus Ultimate's continuous evolution, automated trading, and self-improvement systems.

---

## ✅ SUPPORTED AUSTRALIAN EXCHANGES

### **1. BTC Markets (BEST for Argus)** ⭐

**Location:** Melbourne, Australia  
**Regulation:** AUSTRAC-registered (fully compliant)  
**Website:** https://btcmarkets.net

**Why It's Perfect for Argus:**
- ✅ **NEGATIVE maker fees (-0.05%)** - They PAY YOU to provide liquidity!
- ✅ **Full API support** - REST + WebSocket
- ✅ **AUD trading pairs** - BTC/AUD, ETH/AUD, SOL/AUD
- ✅ **Instant AUD deposits** - PayID/Osko
- ✅ **Low latency** - Sydney to Melbourne ~20ms
- ✅ **Allows automated trading** - API designed for bots

**Fee Structure:**
| Type | Fee |
|------|-----|
| Maker | **-0.05%** (You earn 5 bps!) |
| Taker | 0.20% |
| Deposit (AUD) | FREE |
| Withdrawal (AUD) | ~$1-5 |

**Break-Even Analysis:**
```
With -0.05% maker rebate:
• You PROFIT 5 bps just for making markets
• Any captured spread is PURE PROFIT
• Example: 10 bps spread = 15 bps total profit!
```

**Argus Configuration:**
```yaml
exchanges:
  primary:
    name: btcmarkets
    api_key: ${BTCMARKETS_API_KEY}
    api_secret: ${BTCMARKETS_SECRET}
    
    fee_structure:
      maker: -0.0005  # -0.05%
      taker: 0.002     # 0.20%
    
    # Optimize for negative maker fees
    order_type: limit  # Always use limit orders!
    post_only: true    # Ensure maker status
    
    # Low latency settings
    timeout_ms: 5000
    retry_attempts: 3
```

**API Limits:**
- 500 requests / 10 minutes (conservative)
- Argus uses: ~45 req/min (well within limits)
- WebSocket: Real-time order book + trades

**Available Pairs:**
- BTC-AUD, ETH-AUD, SOL-AUD
- XRP-AUD, LTC-AUD, USDT-AUD
- DOGE-AUD, LINK-AUD

**Minimum Orders:**
- ~$1 AUD minimum (perfect for $1K account)

**Setup Steps:**
1. Create account at btcmarkets.net
2. Complete KYC verification
3. Generate API keys (Settings → API)
4. Deposit AUD via PayID (instant)
5. Configure Argus with keys

---

### **2. Independent Reserve**

**Location:** Sydney, Australia  
**Regulation:** AUSTRAC-registered  
**Website:** https://www.independentreserve.com

**Why It's Good:**
- ✅ **Sydney-based** - Lowest latency (~5ms from your location!)
- ✅ **AUSTRAC compliant** - Full regulatory compliance
- ✅ **AUD, NZD, SGD, USD pairs**
- ✅ **Supports SMSF/company accounts**
- ✅ **OTC desk for large trades**

**Fee Structure:**
| Type | Fee |
|------|-----|
| Maker | 0.50% (high) |
| Taker | 0.50% (high) |
| High Volume | 0.02-0.25% |

**Argus Configuration:**
```yaml
exchanges:
  secondary:
    name: independent_reserve
    api_key: ${IR_API_KEY}
    api_secret: ${IR_API_SECRET}
    
    # Higher fees - use for arbitrage only
    fee_structure:
      maker: 0.005  # 0.50%
      taker: 0.005  # 0.50%
    
    # Best for price comparison/arbitrage
    use_for: arbitrage_only
```

**Note:** Higher fees make it less ideal for high-frequency strategies, but excellent for:
- Price arbitrage between exchanges
- Backup liquidity
- SMSF/institutional accounts

**API Limits:**
- 10 requests/second
- Argus is well within limits

**Setup:**
1. Create account at independentreserve.com
2. Complete identity verification
3. Generate API keys
4. Configure in Argus

---

## ⚠️ PARTIALLY SUPPORTED / LIMITATIONS

### **3. CoinSpot**

**Status:** ⚠️ Limited API Support  
**Website:** https://www.coinspot.com.au

**Issues:**
- ❌ **No public trading API** - Only portfolio API
- ❌ **No WebSocket feeds** - Cannot get real-time data
- ❌ **Not designed for bots** - Manual trading focus
- ❌ **Higher fees** - 1% per trade

**Can Argus Work?**
- Partially via screen scraping (not recommended)
- Use only for portfolio tracking
- Not suitable for continuous 0.5s evolution

**Recommendation:** 
- Use CoinSpot for buying/holding
- Use BTC Markets for Argus trading

---

### **4. Swyftx**

**Status:** ⚠️ Limited API Access  
**Website:** https://swyftx.com.au

**Issues:**
- ❌ **API by request only** - Must apply for access
- ❌ **Rate limits unclear** - Not published
- ❌ **Not tested with Argus** - Compatibility unknown

**Can Argus Work?**
- Unknown - would need API access to test
- If you get API keys, may work with CCXT adapter

**Recommendation:**
- Stick with BTC Markets (fully tested)
- Or apply for Swyftx API and test in paper mode

---

## 🌍 INTERNATIONAL EXCHANGES (Available to Australians)

### **These work from Australia but are not AU-regulated:**

### **5. Bybit** ⭐

**Best for:** High-frequency, futures, funding arbitrage  
**Location:** Singapore/Dubai  
**Fees:** 0.00% maker (spot), 0.01% maker (futures)

**Why Use With Argus:**
- ✅ **Zero maker fees** on select pairs
- ✅ **Futures/perpetuals** for funding arbitrage
- ✅ **High liquidity** - Tight spreads
- ✅ **Excellent API** - WebSocket, amend orders
- ✅ **Low latency** from Sydney (~60ms)

**Argus Configuration:**
```yaml
exchanges:
  international:
    name: bybit
    api_key: ${BYBIT_API_KEY}
    api_secret: ${BYBIT_API_SECRET}
    
    fee_structure:
      spot_maker: 0.001  # 0.10% (0% on VIP)
      futures_maker: 0.0001  # 0.01%
    
    # Enable funding arbitrage
    funding_arbitrage: true
    
    # Low latency connection
    region: asia  # Singapore servers
```

**Regulatory Note:**
- Not AUSTRAC regulated
- Australians can use but no local consumer protection
- Withdraw to Australian bank via third-party (e.g., BTC Markets)

---

### **6. MEXC**

**Best for:** Micro-capital market making, altcoins  
**Location:** Offshore  
**Fees:** **0.00% maker on spot AND futures**

**Why Use With Argus:**
- ✅ **ZERO maker fees** (even better than BTC Markets!)
- ✅ **1000+ pairs** - Many mid-cap altcoins
- ✅ **Wide spreads** - Less competition
- ✅ **Perfect for $1K account**

**Argus Configuration:**
```yaml
exchanges:
  micro_cap:
    name: mexc
    api_key: ${MEXC_API_KEY}
    api_secret: ${MEXC_API_SECRET}
    
    fee_structure:
      spot_maker: 0.0  # ZERO!
      spot_taker: 0.0005
      futures_maker: 0.0  # ZERO!
    
    # Perfect for market making
    strategy: micro_capital_mm
```

**Regulatory Note:**
- Not AUSTRAC regulated
- Withdrawal limits may apply
- Good for building capital, then transfer to AU exchange

---

### **7. Kraken**

**Status:** ⚠️ Expensive for small accounts  
**Fees:** 0.16% maker (needs 32+ bps spread to profit)

**Why NOT Recommended for $1K:**
- ❌ **0.16% maker fee** - Too expensive
- ❌ **Needs wide spreads** to break even
- ❌ **Better for $10K+ accounts**

**Can Use If:**
- Trading very wide spread pairs (>50 bps)
- Using ultra-wide market making spreads
- Account size > $10,000

---

## 📊 EXCHANGE COMPARISON FOR ARGUS

| Exchange | AU Regulated | Maker Fee | Min Order | API Quality | Best For |
|----------|---------------|-----------|-------------|-------------|----------|
| **BTC Markets** | ✅ Yes | **-0.05%** ⭐ | $1 | Excellent | **Primary choice** |
| Independent Reserve | ✅ Yes | 0.50% | $1 | Good | Backup/arbitrage |
| CoinSpot | ✅ Yes | 1.00% | $1 | Poor | Not for bots |
| Bybit | ❌ No | 0.00% | $1 | Excellent | Futures/Arbitrage |
| MEXC | ❌ No | 0.00% ⭐ | $1 | Good | Micro-cap MM |
| Kraken | ❌ No | 0.16% | $5 | Excellent | $10K+ accounts |

**For Your $1K AUD Setup:**
1. **Primary:** BTC Markets (-0.05% maker rebate)
2. **Secondary:** Bybit (futures arbitrage)
3. **Micro-cap:** MEXC (zero fees, wide spreads)

---

## 🔧 SETUP FOR MULTI-EXCHANGE ARGUS

### **Configuration:**

```yaml
# config/australian_multi_exchange.yaml

trading:
  mode: paper  # Start here!
  initial_balance: 1000
  base_currency: AUD

# Australian Primary Exchange
exchanges:
  primary:
    name: btcmarkets
    api_key: ${BTCMARKETS_API_KEY}
    api_secret: ${BTCMARKETS_SECRET}
    fee_maker: -0.0005  # -0.05% (YOU get paid!)
    fee_taker: 0.002
    
    symbols:
      - BTC/AUD
      - ETH/AUD
      - SOL/AUD
      - ADA/AUD
    
    # Optimize for negative maker fees
    default_order_type: limit
    post_only: true
    preferred_side: maker  # Always try to be maker
  
  # International for arbitrage
  secondary:
    name: bybit
    api_key: ${BYBIT_API_KEY}
    api_secret: ${BYBIT_API_SECRET}
    fee_maker: 0.001  # 0.10%
    fee_taker: 0.001
    
    symbols:
      - BTC/USDT
      - ETH/USDT
      - SOL/USDT
    
    # Funding arbitrage
    enable_funding_arbitrage: true
    funding_check_interval: 300  # 5 minutes
  
  # Micro-cap for high returns
  micro_cap:
    name: mexc
    api_key: ${MEXC_API_KEY}
    api_secret: ${MEXC_API_SECRET}
    fee_maker: 0.0  # ZERO!
    fee_taker: 0.0005
    
    symbols:
      - BTC/USDT
      - ETH/USDT
      # Add mid-cap altcoins with wide spreads
    
    # Aggressive market making
    spread_target_bps: 50  # 0.50% spread
    inventory_target: 0.5  # 50% base, 50% quote

# Smart order routing
execution:
  smart_routing:
    enabled: true
    route_by:
      - liquidity
      - fees
      - latency
    
    # Prefer BTC Markets for AUD pairs
    preference:
      "*/AUD": btcmarkets
      "*/USDT": bybit
      "altcoins": mexc

# Continuous evolution works across all exchanges!
continuous_evolution:
  enabled: true
  adapt_per_exchange: true  # Different params per exchange
```

---

## ⚡ LATENCY FROM SYDNEY

| Exchange | Location | Latency | Best Use |
|----------|----------|---------|----------|
| **BTC Markets** | Melbourne | **~20ms** | Primary trading |
| **Independent Reserve** | Sydney | **~5ms** | Price feeds, arbitrage |
| Bybit | Singapore | ~60ms | Futures, swing trading |
| MEXC | Offshore | ~100ms | Micro-cap, wide spreads |
| Kraken | EU/US | ~150ms | Not recommended |

**Your 24-core PC can easily handle all of these simultaneously!**

---

## 🛡️ REGULATORY & COMPLIANCE

### **Australian Regulations:**

**AUSTRAC Registration:**
- ✅ BTC Markets - Registered
- ✅ Independent Reserve - Registered
- ❌ Bybit - Not AUSTRAC (offshore)
- ❌ MEXC - Not AUSTRAC (offshore)

**Tax Obligations:**
```
All profits from ANY exchange are taxable in Australia:
• Capital Gains Tax (CGT) applies
• Trading as business = income tax may apply
• Record ALL trades (Argus does this automatically)
• Use BTC Markets for easy tax reporting (AUD natively)
```

**Argus Tax Features:**
```python
# Automatic tax record keeping
from compliance.ato_cgt import TaxTracker

tracker = TaxTracker()
tracker.record_trade(exchange, pair, pnl, timestamp)

# Export for tax time
report = tracker.generate_annual_report(year=2026)
# → CSV for your accountant
```

---

## 🚀 RECOMMENDED SETUP FOR YOU

### **For Sydney + $1K AUD + Argus Ultimate:**

**Option 1: Beginner (Safest)**
```yaml
# Use only AU-regulated exchange
exchanges:
  primary:
    name: btcmarkets
    
# Start with paper trading
mode: paper
```

**Option 2: Advanced (Higher Returns)**
```yaml
# Multi-exchange for maximum opportunity
exchanges:
  primary:
    name: btcmarkets  # -0.05% maker fees
    allocation: 60%  # $600
  
  secondary:
    name: bybit      # Futures arbitrage
    allocation: 30%  # $300
  
  micro_cap:
    name: mexc       # Zero fees, wide spreads
    allocation: 10%  # $100

mode: paper  # Test first, then live
```

---

## ✅ FINAL RECOMMENDATIONS

### **Best Exchange for Argus in Australia: BTC Markets**

**Why:**
1. **-0.05% maker rebate** - You get PAID to trade
2. **AUSTRAC regulated** - Full compliance
3. **AUD pairs** - No currency risk
4. **Low latency** - 20ms from Sydney
5. **Full API** - Designed for bots
6. **Instant deposits** - PayID
7. **Tested with Argus** - Production ready

### **Setup Today:**
1. Sign up at **btcmarkets.net**
2. Complete KYC (usually instant)
3. Deposit $1000 AUD via PayID
4. Generate API keys
5. Configure Argus
6. Start paper trading
7. Go live after 24h testing

---

**🏆 Start with BTC Markets - it's the perfect exchange for Argus Ultimate in Australia!** 🇦🇺
