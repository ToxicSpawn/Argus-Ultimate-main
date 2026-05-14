@echo off
REM =============================================================================
REM Argus Flash Loan Setup Script (Windows)
REM =============================================================================
REM Run this after completing the manual steps below
REM
REM MANUAL STEPS (do these first):
REM 1. Create MetaMask wallet at https://metamask.io
REM 2. Buy $10 MATIC on Binance/Coinbase
REM 3. Get free RPC at https://alchemy.com (create app, select Polygon)
REM 4. Deploy contract using Remix (instructions below)
REM
REM THEN RUN: setup_flash_loans.bat
REM =============================================================================

echo ==========================================
echo ARGUS FLASH LOAN SETUP (Windows)
echo ==========================================
echo.

REM Check if .env exists
if not exist .env (
    echo Creating .env file...
    type nul > .env
)

echo.
echo ==========================================
echo Step 1: Wallet Configuration
echo ==========================================
echo.
echo Enter your wallet address (0x...):
set /p WALLET_ADDRESS="> "

echo.
echo Enter your private key:
set /p PRIVATE_KEY="> "

echo.
echo ==========================================
echo Step 2: RPC Configuration
echo ==========================================
echo.
echo Enter your Polygon RPC URL:
echo ^(Format: https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY^)
set /p POLYGON_RPC="> "

echo.
echo Enter your Contract Address (deployed via Remix):
set /p CONTRACT_ADDRESS="> "

REM Write to .env
echo. >> .env
echo # Flash Loan Configuration (auto-generated) >> .env
echo FLASH_LOAN_WALLET_ADDRESS=%WALLET_ADDRESS% >> .env
echo FLASH_LOAN_PRIVATE_KEY=%PRIVATE_KEY% >> .env
echo FLASH_LOAN_CONTRACT_ADDRESS=%CONTRACT_ADDRESS% >> .env
echo POLYGON_RPC_URL=%POLYGON_RPC% >> .env

echo.
echo ==========================================
echo SETUP COMPLETE!
echo ==========================================
echo.
echo Configuration saved to .env
echo.
echo Next steps:
echo   1. Run: py main.py paper --flash-loans-scan-only
echo   2. Monitor for opportunities
echo   3. When ready: py main.py live
echo.
pause
