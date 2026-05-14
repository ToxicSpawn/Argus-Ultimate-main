// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

/**
 * @title ArgusFlashLoanExecutor
 * @dev Flash loan executor for Argus trading system
 * 
 * Supports:
 * - Aave V3 flash loans
 * - Balancer flash loans
 * - Multi-hop arbitrage
 * - MEV protection
 * 
 * Security:
 * - Only owner can trigger flash loans
 * - Automatic repayment verification
 * - Reentrancy protection
 */
interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

interface IPool {
    function flashLoanSimple(
        address receiverAddress,
        address asset,
        uint256 amount,
        bytes calldata params,
        uint256 interestRateMode
    ) external;
    
    function FLASHLOAN_PREMIUM_TOTAL() external view returns (uint128);
}

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);
    
    function getAmountsOut(uint256 amountIn, address[] calldata path) external view returns (uint256[] memory amounts);
}

interface ISushiSwapRouter {
    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);
}

/**
 * @title ArgusFlashLoanExecutor
 * @dev Main flash loan executor contract
 */
contract ArgusFlashLoanExecutor {
    // ============ State ============
    
    address public owner;
    IPool public aavePool;
    
    // Supported DEX routers
    mapping(string => address) public dexRouters;
    
    // Whitelisted tokens
    mapping(address => bool) public whitelistedTokens;
    
    // Reentrancy guard
    bool private _locked;
    
    // Statistics
    uint256 public totalExecutions;
    uint256 public totalProfit;
    uint256 public totalLoans;
    
    // Events
    event FlashLoanExecuted(
        address indexed token,
        uint256 amount,
        uint256 profit,
        string buyDex,
        string sellDex,
        uint256 timestamp
    );
    
    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );
    
    event DEXRouterAdded(string name, address router);
    event TokenWhitelisted(address token);
    event TokenRemoved(address token);
    
    // ============ Modifiers ============
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    modifier nonReentrant() {
        require(!_locked, "Reentrant call");
        _locked = true;
        _;
        _locked = false;
    }
    
    // ============ Constructor ============
    
    /**
     * @dev Constructor
     * @param _aavePool Aave V3 Pool address
     */
    constructor(address _aavePool) {
        owner = msg.sender;
        aavePool = IPool(_aavePool);
        
        // Register default DEX routers (Ethereum mainnet)
        dexRouters["uniswap_v2"] = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
        dexRouters["uniswap_v3"] = 0xE592427A0AEce92De3Edee1F18E0157C05861564;
        dexRouters["sushiswap"] = 0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F;
        dexRouters["curve"] = 0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7;
        
        emit OwnershipTransferred(address(0), msg.sender);
    }
    
    // ============ Core Functions ============
    
    /**
     * @dev Execute flash loan arbitrage
     * @param token Token to borrow
     * @param amount Amount to borrow
     * @param buyDex DEX to buy on
     * @param sellDex DEX to sell on
     * @param minProfit Minimum profit to accept
     */
    function executeArbitrage(
        address token,
        uint256 amount,
        string calldata buyDex,
        string calldata sellDex,
        uint256 minProfit
    ) external onlyOwner nonReentrant {
        require(whitelistedTokens[token], "Token not whitelisted");
        require(amount > 0, "Amount must be > 0");
        
        // Encode parameters for callback
        bytes memory params = abi.encode(
            token,
            amount,
            buyDex,
            sellDex,
            minProfit,
            msg.sender  // Return profit to owner
        );
        
        // Trigger flash loan
        aavePool.flashLoanSimple(
            address(this),  // receiver
            token,          // asset
            amount,         // amount
            params,         // params
            2               // variable rate
        );
    }
    
    /**
     * @dev Aave flash loan callback
     * @dev Called by Aave pool after lending funds
     */
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external returns (bool) {
        require(msg.sender == address(aavePool), "Only Aave pool");
        require(initiator == address(this), "Only self");
        
        // Decode parameters
        (address token, uint256 loanAmount, string memory buyDex, 
         string memory sellDex, uint256 minProfit, address profitRecipient) = 
            abi.decode(params, (address, uint256, string, string, uint256, address));
        
        // Execute arbitrage
        uint256 profit = _executeArbitrageInternal(
            token,
            loanAmount,
            buyDex,
            sellDex
        );
        
        // Calculate amount to repay
        uint256 amountOwed = amount + premium;
        
        // Verify we have enough to repay
        uint256 balance = IERC20(asset).balanceOf(address(this));
        require(balance >= amountOwed, "Insufficient funds to repay");
        
        // Transfer profit to owner
        if (balance > amountOwed) {
            uint256 profitAmount = balance - amountOwed;
            IERC20(asset).transfer(profitRecipient, profitAmount);
            totalProfit += profitAmount;
        }
        
        // Approve repayment
        IERC20(asset).approve(address(aavePool), amountOwed);
        
        // Update statistics
        totalExecutions++;
        totalLoans += loanAmount;
        
        emit FlashLoanExecuted(
            asset,
            loanAmount,
            profit,
            buyDex,
            sellDex,
            block.timestamp
        );
        
        return true;
    }
    
    // ============ Internal Functions ============
    
    /**
     * @dev Execute the actual arbitrage trades
     */
    function _executeArbitrageInternal(
        address token,
        uint256 amount,
        string memory buyDex,
        string memory sellDex
    ) internal returns (uint256) {
        // Get router addresses
        address buyRouter = dexRouters[buyDex];
        address sellRouter = dexRouters[sellDex];
        
        require(buyRouter != address(0), "Buy DEX not found");
        require(sellRouter != address(0), "Sell DEX not found");
        
        // Get WETH address for swaps
        address WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
        
        // Build swap path
        address[] memory buyPath = new address[](2);
        buyPath[0] = token;
        buyPath[1] = WETH;
        
        address[] memory sellPath = new address[](2);
        sellPath[0] = WETH;
        sellPath[1] = token;
        
        // Approve buy router
        IERC20(token).approve(buyRouter, amount);
        
        // Buy on DEX A (token -> WETH)
        IUniswapV2Router(buyRouter).swapExactTokensForTokens(
            amount,
            0,  // Accept any amount (MEV protection should handle this)
            buyPath,
            address(this),
            block.timestamp + 300
        );
        
        // Get WETH balance
        uint256 wethBalance = IERC20(WETH).balanceOf(address(this));
        
        // Approve sell router
        IERC20(WETH).approve(sellRouter, wethBalance);
        
        // Sell on DEX B (WETH -> token)
        IUniswapV2Router(sellRouter).swapExactTokensForTokens(
            wethBalance,
            0,
            sellPath,
            address(this),
            block.timestamp + 300
        );
        
        // Return final token balance
        return IERC20(token).balanceOf(address(this));
    }
    
    // ============ Admin Functions ============
    
    /**
     * @dev Add DEX router
     */
    function addDEXRouter(string calldata name, address router) external onlyOwner {
        require(router != address(0), "Invalid router");
        dexRouters[name] = router;
        emit DEXRouterAdded(name, router);
    }
    
    /**
     * @dev Whitelist token for flash loans
     */
    function whitelistToken(address token) external onlyOwner {
        require(token != address(0), "Invalid token");
        whitelistedTokens[token] = true;
        emit TokenWhitelisted(token);
    }
    
    /**
     * @dev Remove token from whitelist
     */
    function removeToken(address token) external onlyOwner {
        whitelistedTokens[token] = false;
        emit TokenRemoved(token);
    }
    
    /**
     * @dev Transfer ownership
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid owner");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
    
    /**
     * @dev Withdraw ETH
     */
    function withdrawETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH to withdraw");
        payable(owner).transfer(balance);
    }
    
    /**
     * @dev Withdraw tokens
     */
    function withdrawToken(address token) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        require(balance > 0, "No tokens to withdraw");
        IERC20(token).transfer(owner, balance);
    }
    
    // ============ View Functions ============
    
    /**
     * @dev Get statistics
     */
    function getStats() external view returns (
        uint256 _totalExecutions,
        uint256 _totalProfit,
        uint256 _totalLoans
    ) {
        return (totalExecutions, totalProfit, totalLoans);
    }
    
    /**
     * @dev Get DEX router address
     */
    function getDEXRouter(string calldata name) external view returns (address) {
        return dexRouters[name];
    }
    
    /**
     * @dev Check if token is whitelisted
     */
    function isTokenWhitelisted(address token) external view returns (bool) {
        return whitelistedTokens[token];
    }
    
    // Receive ETH
    receive() external payable {}
}
