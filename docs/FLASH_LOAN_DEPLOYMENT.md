# Flash Loan Contract Deployment Guide

## Quick Deploy Using Remix IDE (5 minutes)

### Step 1: Open Remix
1. Go to **https://remix.ethereum.org**
2. Click **"Create New File"**
3. Name it `ArgusFlashLoanExecutor.sol`

### Step 2: Copy Contract Code
Copy the contents of `contracts/ArgusFlashLoanExecutor.sol` into Remix.

### Step 3: Compile
1. Click the **Solidity Compiler** icon (left sidebar)
2. Select compiler version **0.8.10** or higher
3. Click **"Compile ArgusFlashLoanExecutor.sol"**
4. Check for any errors (should be none)

### Step 4: Connect Wallet
1. Click **"Deploy & Run Transactions"** icon
2. Environment: **"Injected Provider - MetaMask"**
3. MetaMask will pop up - connect your wallet
4. Make sure you're on **Polygon Mainnet** (cheaper gas)

### Step 5: Deploy
1. In the "Deploy" section, enter constructor argument:
   ```
   0x794a61358D6845594F94dc1DB02A252b5b4814aD
   ```
   (This is Aave V3 Pool on Polygon)

2. Click **"Transact"**
3. Confirm in MetaMask (gas will be ~$0.10-0.50 on Polygon)

4. Wait for confirmation

### Step 6: Get Contract Address
1. After deployment, look in "Deployed Contracts" section
2. Copy the contract address (starts with 0x...)
3. Save this address - you'll need it for Argus config

### Step 7: Whitelist Tokens
1. In "Deployed Contracts", expand your contract
2. Find `whitelistToken` function
3. Enter token addresses one by one:

```
WMATIC: 0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270
USDC:   0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359
USDT:   0xc2132D05D31c914a87C6611C10748AEb04B58e8F
WETH:   0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619
```

4. Click "transact" for each

### Step 8: Fund Your Wallet
Send some MATIC to your wallet for gas:
- Minimum: 10 MATIC (~$10)
- Recommended: 50 MATIC (~$50)

---

## Verify Contract (Optional but Recommended)

1. Go to **https://polygonscan.com**
2. Search for your contract address
3. Click **"Contract"** tab
4. Click **"Verify and Publish"**
5. Enter:
   - Compiler: 0.8.10
   - License: MIT
   - Paste contract code
6. Click verify

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Insufficient funds" | Add more MATIC to wallet |
| "Gas estimation failed" | Check Aave pool address is correct |
| "Only owner" | Make sure you're using the deploying wallet |
| "Token not whitelisted" | Call whitelistToken first |

---

## Contract Addresses (Polygon)

| Contract | Address |
|----------|---------|
| Aave V3 Pool | `0x794a61358D6845594F94dc1DB02A252b5b4814aD` |
| WMATIC | `0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270` |
| USDC | `0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359` |
| USDT | `0xc2132D05D31c914a87C6611C10748AEb04B58e8F` |
| WETH | `0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619` |
