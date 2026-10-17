# Smart Contracts Layer (`contracts/`)

This directory contains the Solidity smart contract, deployment scripts, and Hardhat test suite that provide the immutable **on-chain settlement ledger** for peer-to-peer energy trades in the smart energy microgrid.

---

## 1. Architecture & Design Principles

The microgrid operates an hourly local energy market among 25 households. To avoid high public network gas fees and network latency, settlements are executed on a local Ethereum chain (Hardhat node) via an authorized contract owner (the backend oracle / simulation engine).

```
 +-------------------------------------------------------------+
 |                    Hardhat Node (:8545)                     |
 |                                                             |
 |   +-----------------------------------------------------+   |
 |   |                 EnergyTrading.sol                   |   |
 |   |                                                     |   |
 |   |  - Registry: mapping(householdId => walletAddress)  |   |
 |   |  - Balances:                                        |   |
 |   |      mapping(householdId => int256 energyBalance)   |   |
 |   |      mapping(householdId => int256 tokenBalance)    |   |
 |   |  - Trades Array: Trade[] (append-only ledger)       |   |
 |   +-----------------------------------------------------+   |
 +-------------------------------------------------------------+
         ^                                             |
         | executeTrade() transactions                 | events & view calls
         |                                             v
+------------------+                       +-----------------------+
|  Backend Service |                       | Frontend Dashboard    |
| (ethers.js v6)   |                       | (tx hashes, balances) |
+------------------+                       +-----------------------+
```

### Key Contract: `EnergyTrading.sol`
- **Inheritance:** OpenZeppelin `Ownable` (only the designated deployer/backend signer can execute trades and register households).
- **Fixed-Point Milli Units:** Floating-point numbers are not supported on the EVM. All physical energy values and monetary balances are scaled by $1000$:
  - Energy: **milli-kWh** ($1 \text{ kWh} = 1{,}000 \text{ milli-kWh}$)
  - Tokens: **milli-credits** ($1 \text{ INR} = 1{,}000 \text{ milli-credits}$)
  - Calculation: $\text{costMilli} = \frac{\text{kWhMilli} \times \text{pricePerKwhMilli}}{1000}$ (pure integer arithmetic).
- **Signed Double-Entry Ledger:** Households maintain signed balances (`int256`). When a trade settles:
  - Seller: `energyBalance -= kWhMilli`, `tokenBalance += costMilli`
  - Buyer: `energyBalance += kWhMilli`, `tokenBalance -= costMilli`
- **Invariants:** 
  $$\sum \text{energyBalance} = 0 \quad \text{and} \quad \sum \text{tokenBalance} = 0$$
  Every milli-kWh sold by one home is credited to another. No tokens or energy are created or destroyed.

---

## 2. Data Structures & Functions

### `struct Trade`
```solidity
struct Trade {
    uint256 sellerId;
    uint256 buyerId;
    uint256 kWhMilli;
    uint256 pricePerKwhMilli;
    uint256 costMilli;
    uint256 slotTimestamp; // Simulated slot time (unix epoch seconds)
    uint256 timestamp;     // Block timestamp
}
```

### Core Functions
- `registerHousehold(uint256 householdId, address wallet)`: Registers a household with a unique wallet address (25 households numbered `0` to `24`).
- `executeTrade(uint256 sellerId, uint256 buyerId, uint256 kWhMilli, uint256 pricePerKwhMilli, uint256 slotTimestamp)`: Atomically updates signed energy and token balances, pushes the trade to the immutable `trades` array, and emits `TradeExecuted`.
- `getEnergyBalance(uint256 householdId) returns (int256)`: Returns the net energy balance in milli-kWh.
- `getTokenBalance(uint256 householdId) returns (int256)`: Returns the net token/cash balance in milli-credits.
- `getTradeCount() returns (uint256)`: Total number of trades recorded on-chain.
- `getTrade(uint256 tradeId) returns (Trade memory)`: Retrieves full trade receipt by index.

---

## 3. Directory Structure

```
contracts/
├── contracts/
│   └── EnergyTrading.sol     # Main Solidity smart contract
├── scripts/
│   └── deploy.js             # Deployment script (deploys & registers 25 households)
├── test/
│   └── EnergyTrading.test.js # Hardhat test suite (ledger math, invariants, permissions)
├── hardhat.config.js         # Hardhat configuration (Solidity 0.8.24)
└── package.json              # Scripts and devDependencies (@nomicfoundation/hardhat-toolbox)
```

---

## 4. Getting Started

### Installation
From the repository root:
```bash
npm --prefix contracts install
```
Or directly inside `contracts/`:
```bash
npm install
```

### Compile Contract
```bash
npm run compile
```

### Run Tests
The automated test suite verifies owner permissions, registration validation, trade execution, balance debits/credits, zero-division safety, and conservation of balances:
```bash
npm test
```

### Running the Local Blockchain Node
```bash
npm run chain
# Starts JSON-RPC server at http://127.0.0.1:8545 with 20 prefunded test accounts
```

### Deploying the Contract
In a separate terminal while `npm run chain` is active:
```bash
npm run deploy
```
`scripts/deploy.js` performs two critical steps:
1. Deploys `EnergyTrading` using Hardhat Account #0.
2. Registers households `0` through `24` with test wallet addresses.
3. Exports the contract address and ABI directly to `backend/src/contracts/EnergyTrading.json` so the Express backend can immediately interact with it without manual copy-pasting.

---

## 5. Things to Know
- **In-Memory Chain State:** The Hardhat node runs in memory. If you stop or restart `npm run chain`, the contract must be redeployed (`npm run deploy`), and the MongoDB simulation state should be reset (`POST /simulate/reset` or `npm run start:all -- --reset`) so MongoDB does not hold trades that the fresh chain does not know about.
- **Gas Costs:** On a local Hardhat node, transactions confirm instantly (block time ~0s). Gas is tracked but does not consume real ETH.
