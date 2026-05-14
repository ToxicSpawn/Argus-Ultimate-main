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
        
        (address tk, uint256 amt, string memory b, string memory s, address to) = abi.decode(params, (address, uint256, string, string, address));
        
        address W = 0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270;
        address[] memory p1 = new address[](2);
        p1[0] = tk; p1[1] = W;
        
        IERC20(tk).approve(routers[b], amt);
        IRouter(routers[b]).swapExactTokensForTokens(amt, 0, p1, address(this), block.timestamp + 300);
        
        uint256 w = IERC20(W).balanceOf(address(this));
        address[] memory p2 = new address[](2);
        p2[0] = W; p2[1] = tk;
        
        IERC20(W).approve(routers[s], w);
        IRouter(routers[s]).swapExactTokensForTokens(w, 0, p2, address(this), block.timestamp + 300);
        
        uint256 bal = IERC20(asset).balanceOf(address(this));
        uint256 owed = amount + premium;
        
        if (bal > owed) IERC20(asset).transfer(to, bal - owed);
        
        IERC20(asset).approve(address(pool), owed);
        emit Executed(asset, bal - owed);
        return true;
    }

    function addRouter(string calldata n, address r) external onlyOwner { routers[n] = r; }
    function whitelist(address t) external onlyOwner { whitelisted[t] = true; }
    receive() external payable {}
}
