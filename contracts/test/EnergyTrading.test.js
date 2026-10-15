const { expect } = require("chai");
const { ethers } = require("hardhat");


describe("EnergyTrading", () => {
  let c, owner, other, w1, w2;
  const SLOT = 1792454400; // 2026-10-20T00:00:00Z


  beforeEach(async () => {
    [owner, other, w1, w2] = await ethers.getSigners();
    c = await (await ethers.getContractFactory("EnergyTrading")).deploy();
    await c.registerHousehold(1, w1.address);
    await c.registerHousehold(2, w2.address);
  });


  describe("registration", () => {
    it("registers households", async () => {
      expect(await c.isRegistered(1)).to.equal(true);
      expect(await c.householdWallet(2)).to.equal(w2.address);
      expect(await c.isRegistered(3)).to.equal(false);
    });
    it("rejects duplicates and zero address", async () => {
      await expect(c.registerHousehold(1, w2.address)).to.be.revertedWith("already registered");
      await expect(c.registerHousehold(9, ethers.ZeroAddress)).to.be.revertedWith("zero wallet");
    });
    it("is owner-only", async () => {
      await expect(c.connect(other).registerHousehold(5, w1.address)).to.be.revertedWithCustomError(
        c,
        "OwnableUnauthorizedAccount"
      );
    });
  });


  describe("executeTrade", () => {
    it("updates energy and token balances on both sides and stores the trade", async () => {
      // 2.5 kWh at 6.000 credits/kWh -> cost 15.000 credits = 15000 milli
      await expect(c.executeTrade(1, 2, 2500, 6000, SLOT))
        .to.emit(c, "TradeExecuted")
        .withArgs(0, 1, 2, 2500, 6000, SLOT, (v) => v > 0);


      expect(await c.getEnergyBalance(1)).to.equal(-2500);
      expect(await c.getEnergyBalance(2)).to.equal(2500);
      expect(await c.getTokenBalance(1)).to.equal(15000);
      expect(await c.getTokenBalance(2)).to.equal(-15000);
      expect(await c.getTradeCount()).to.equal(1);


      const t = await c.getTrade(0);
      expect(t.sellerId).to.equal(1);
      expect(t.buyerId).to.equal(2);
      expect(t.kWhMilli).to.equal(2500);
      expect(t.costMilli).to.equal(15000);
      expect(t.slotTimestamp).to.equal(SLOT);
    });


    it("accumulates over several trades and conserves totals", async () => {
      await c.executeTrade(1, 2, 1000, 6000, SLOT);
      await c.executeTrade(2, 1, 400, 6000, SLOT);
      expect(await c.getEnergyBalance(1)).to.equal(-600);
      expect(await c.getEnergyBalance(2)).to.equal(600);
      expect((await c.getTokenBalance(1)) + (await c.getTokenBalance(2))).to.equal(0);
    });


    it("rounds cost down with integer division", async () => {
      await c.executeTrade(1, 2, 1, 6000, SLOT); // 1*6000/1000 = 6
      expect(await c.getTokenBalance(1)).to.equal(6);
      await c.executeTrade(1, 2, 1, 1, SLOT); // 1*1/1000 = 0
      expect(await c.getTokenBalance(1)).to.equal(6);
    });


    it("reverts for non-owner", async () => {
      await expect(c.connect(other).executeTrade(1, 2, 1000, 6000, SLOT)).to.be.revertedWithCustomError(
        c,
        "OwnableUnauthorizedAccount"
      );
    });


    it("reverts for unregistered households", async () => {
      await expect(c.executeTrade(99, 2, 1000, 6000, SLOT)).to.be.revertedWith("seller not registered");
      await expect(c.executeTrade(1, 99, 1000, 6000, SLOT)).to.be.revertedWith("buyer not registered");
    });


    it("reverts for self-trade, zero energy and zero price", async () => {
      await expect(c.executeTrade(1, 1, 1000, 6000, SLOT)).to.be.revertedWith("seller == buyer");
      await expect(c.executeTrade(1, 2, 0, 6000, SLOT)).to.be.revertedWith("zero energy");
      await expect(c.executeTrade(1, 2, 1000, 0, SLOT)).to.be.revertedWith("zero price");
    });


    it("getTrade reverts out of range", async () => {
      await expect(c.getTrade(0)).to.be.revertedWith("no such trade");
    });
  });
});



