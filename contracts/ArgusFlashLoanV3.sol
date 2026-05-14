// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

interface IERC20 {
    function balanceOf(address) external view returns (uint256);
    function transfer(address, uint256) external returns (bool);
    function approve(address, uint256) external returns (bool);
}

interface IPool {
    function flashLoanSimple(address, address, uint256, bytes calldata, uint256) external;
}

interface IRouter {
    function swapExactTokensForTokens(uint256, uint256, address[] calldata, address, uint256) external returns (uint256[] memory);
}

contract ArgusFlashLoan {
    address public owner;
    IPool pool;
    mapping(string => address) routers;
    mapping(address => bool) public whitelisted;
    
    // Storage variables to avoid stack depth
    address storedToken;
    address storedProfitTo;
    uint256 storedAmount;
    uint256 storedPremium;
    
    event Executed(address token, uint256 profit);

    modifier onlyOwner() { require(msg.sender == owner); _; }

    constructor(address _pool) {
        owner = msg.sender;
        pool = IPool(_pool);
        routers["uni"] = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
        routers["sushi"] = 0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F;
    }

    function execute(address tk, uint256 amt, string calldata buy, string calldata sell) external onlyOwner {
        require(whitelisted[tk]);
        pool.flashLoanSimple(address(this), tk, amt, abi.encode(tk, amt, buy, sell, msg.sender), 2);
    }

    function executeOperation(address asset, uint256 amount, uint256 premium, address, bytes calldata params) external returns (bool) {
        require(msg.sender == address(pool));
        
        (storedToken, storedAmount, , , storedProfitTo) = abi.decode(params, (address, uint256, string, string, address));
        storedPremium = premium;
        
        _doSwap1();
        _doSwap2();
        _repay(asset);
        
        return true;
    }
    
    function _doSwap1() internal {
        address W = 0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270;
        address[] memory path = new address[](2);
        path[0] = storedToken;
        path[1] = W;
        IERC20(storedToken).approve(routers["uni"], storedAmount);
        IRouter(routers["uni"]).swapExactTokensForTokens(storedAmount, 0, path, address(this), block.timestamp + 300);
    }
    
    function _doSwap2() internal {
        address W = 0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270;
        uint256 wBalance = IERC20(W).balanceOf(address(this));
        address[] memory path = new address[](2);
        path[0] = W;
        path[1] = storedToken;
        IERC20(W).approve(routers["sushi"], wBalance);
        IRouter(routers["sushi"]).swapExactTokensForTokens(wBalance, 0, path, address(this), block.timestamp + 300);
    }
    
    function _repay(address asset) internal {
        uint256 balance = IERC20(asset).balanceOf(address(this));
        uint256 owed = storedAmount + storedPremium;
        
        if (balance > owed) {
            IERC20(asset).transfer(storedProfitTo, balance - owed);
            emit Executed(asset, balance - owed);
        }
        
        IERC20(asset).approve(address(pool), owed);
    }

    function addRouter(string calldata n, address r) external onlyOwner { routers[n] = r; }
    function whitelist(address t) external onlyOwner { whitelisted[t] = true; }
    receive() external payable {}
}
