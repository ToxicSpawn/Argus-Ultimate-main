#!/bin/bash
# =============================================================================
# Argus Flash Loan Setup Script
# =============================================================================
# Run this after completing the manual steps below
#
# MANUAL STEPS (do these first):
# 1. Create MetaMask wallet at https://metamask.io
# 2. Buy $10 MATIC on Binance/Coinbase
# 3. Get free RPC at https://alchemy.com (create app, select Polygon)
# 4. Deploy contract using Remix (instructions below)
#
# THEN RUN: bash setup_flash_loans.sh
# =============================================================================

set -e

echo "=========================================="
echo "ARGUS FLASH LOAN SETUP"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    touch .env
fi

# Prompt for configuration
echo -e "${GREEN}Step 1: Wallet Configuration${NC}"
echo "Enter your wallet address (0x...):"
read -r WALLET_ADDRESS

echo "Enter your private key (hidden):"
read -s PRIVATE_KEY
echo ""

echo ""
echo -e "${GREEN}Step 2: RPC Configuration${NC}"
echo "Enter your Polygon RPC URL:"
echo "(Format: https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY)"
read -r POLYGON_RPC

echo "Enter your Ethereum RPC URL (optional, press enter to skip):"
read -r ETH_RPC

echo ""
echo -e "${GREEN}Step 3: Contract Configuration${NC}"
echo "Enter your deployed contract address (0x...):"
echo "(Deploy using Remix first - see instructions below)"
read -r CONTRACT_ADDRESS

# Write to .env
echo ""
echo -e "${GREEN}Writing configuration...${NC}"

cat >> .env << EOF

# Flash Loan Configuration (auto-generated)
FLASH_LOAN_WALLET_ADDRESS=$WALLET_ADDRESS
FLASH_LOAN_PRIVATE_KEY=$PRIVATE_KEY
FLASH_LOAN_CONTRACT_ADDRESS=$CONTRACT_ADDRESS
POLYGON_RPC_URL=$POLYGON_RPC
ETH_RPC_URL=$ETH_RPC
EOF

echo -e "${GREEN}✓ .env updated${NC}"

# Create unified_config.yaml flash loan section
echo ""
echo -e "${GREEN}Updating unified_config.yaml...${NC}"

if [ -f unified_config.yaml ]; then
    # Backup existing config
    cp unified_config.yaml unified_config.yaml.backup
    echo -e "${GREEN}✓ Backed up unified_config.yaml${NC}"
fi

# Create flash loan config snippet
cat > flash_loan_config_active.yaml << EOF
# Flash Loan Configuration - Active
flash_loans:
  enabled: true
  
  # Profitability
  min_profit_usd: 20
  min_profit_percent: 0.001
  
  # Risk Limits
  max_loan_usd: 100000
  max_daily_loans: 20
  max_gas_price_gwei: 100
  
  # Scan Settings
  scan_interval_seconds: 1
  verify_before_execute: true
  simulation_required: true
  
  # Chains
  chains:
    polygon:
      enabled: true
      rpc_url: "\${POLYGON_RPC_URL}"
      gas_cost_usd: 0.02
      min_profit_usd: 20
    arbitrum:
      enabled: false
      rpc_url: "\${ARBITRUM_RPC_URL}"
      gas_cost_usd: 0.50
      min_profit_usd: 50
  
  # Tokens
  tokens:
    - symbol: "WMATIC"
      min_loan_usd: 1000
    - symbol: "USDC"
      min_loan_usd: 10000
    - symbol: "USDT"
      min_loan_usd: 10000
    - symbol: "WETH"
      min_loan_usd: 5000
  
  # DEXes
  dexes:
    quickswap:
      enabled: true
    sushiswap:
      enabled: true
    curve:
      enabled: true
  
  # MEV Protection
  mev_protection:
    enabled: true
    use_private_mempool: false  # Polygon doesn't need Flashbots
  
  # Contract
  contract:
    executor_address: "\${FLASH_LOAN_CONTRACT_ADDRESS}"
  
  # Wallet
  wallet:
    address: "\${FLASH_LOAN_WALLET_ADDRESS}"
  
  # Monitoring
  monitoring:
    enabled: true
    log_executions: true
EOF

echo -e "${GREEN}✓ Created flash_loan_config_active.yaml${NC}"
echo -e "${YELLOW}  Add this to your unified_config.yaml manually${NC}"

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo "=========================================="
echo ""
echo "Configuration saved to:"
echo "  - .env (wallet, RPC, contract)"
echo "  - flash_loan_config_active.yaml (config snippet)"
echo ""
echo "Next steps:"
echo "  1. Add flash_loan_config_active.yaml to unified_config.yaml"
echo "  2. Run: py main.py paper --flash-loans-scan-only"
echo "  3. Monitor for opportunities"
echo "  4. When ready: py main.py live"
echo ""
echo "=========================================="
