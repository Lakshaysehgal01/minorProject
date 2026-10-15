// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/Ownable.sol";

/// Settlement ledger for hourly P2P energy trades.
/// Energy is in milli-kWh (kWh x 1000). Token balances are in milli-credits.
/// cost (milli-credits) = kWhMilli * pricePerKwhMilli / 1000 (integer division).
/// Balances are signed ledger entries: a buyer may go negative (net debtor).
contract EnergyTrading is Ownable {
    struct Trade {
        uint256 sellerId;
        uint256 buyerId;
        uint256 kWhMilli;
        uint256 pricePerKwhMilli;
        uint256 costMilli;
        uint256 slotTimestamp; // simulated slot time (unix seconds), not block time
        uint256 timestamp; // block time
    }

    mapping(uint256 => address) public householdWallet;
    mapping(uint256 => bool) public isRegistered;
    mapping(uint256 => int256) private energyBalance;
    mapping(uint256 => int256) private tokenBalance;
    Trade[] private trades;

    event HouseholdRegistered(uint256 indexed householdId, address wallet);
    event TradeExecuted(
        uint256 indexed tradeId,
        uint256 indexed sellerId,
        uint256 indexed buyerId,
        uint256 kWhMilli,
        uint256 pricePerKwhMilli,
        uint256 slotTimestamp,
        uint256 timestamp
    );

    constructor() Ownable(msg.sender) {}

    function registerHousehold(uint256 householdId, address wallet) external onlyOwner {
        require(wallet != address(0), "zero wallet");
        require(!isRegistered[householdId], "already registered");
        isRegistered[householdId] = true;
        householdWallet[householdId] = wallet;
        emit HouseholdRegistered(householdId, wallet);
    }

    function executeTrade(
        uint256 sellerId,
        uint256 buyerId,
        uint256 kWhMilli,
        uint256 pricePerKwhMilli,
        uint256 slotTimestamp
    ) external onlyOwner returns (uint256 tradeId) {
        require(isRegistered[sellerId], "seller not registered");
        require(isRegistered[buyerId], "buyer not registered");
        require(sellerId != buyerId, "seller == buyer");
        require(kWhMilli > 0, "zero energy");
        require(pricePerKwhMilli > 0, "zero price");

        uint256 cost = (kWhMilli * pricePerKwhMilli) / 1000;

        energyBalance[sellerId] -= int256(kWhMilli);
        energyBalance[buyerId] += int256(kWhMilli);
        tokenBalance[buyerId] -= int256(cost);
        tokenBalance[sellerId] += int256(cost);

        tradeId = trades.length;
        trades.push(
            Trade(sellerId, buyerId, kWhMilli, pricePerKwhMilli, cost, slotTimestamp, block.timestamp)
        );
        emit TradeExecuted(tradeId, sellerId, buyerId, kWhMilli, pricePerKwhMilli, slotTimestamp, block.timestamp);
    }

    function getEnergyBalance(uint256 householdId) external view returns (int256) {
        return energyBalance[householdId];
    }

    function getTokenBalance(uint256 householdId) external view returns (int256) {
        return tokenBalance[householdId];
    }

    function getTradeCount() external view returns (uint256) {
        return trades.length;
    }

    function getTrade(uint256 tradeId) external view returns (Trade memory) {
        require(tradeId < trades.length, "no such trade");
        return trades[tradeId];
    }
}


