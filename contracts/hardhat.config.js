require("@nomicfoundation/hardhat-toolbox");


/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.24",
    settings: { optimizer: { enabled: true, runs: 200 } },
  },
  networks: {
    // Hardhat defaults to 20 accounts; we register 25 households.
    hardhat: { chainId: 31337, accounts: { count: 26 } },
    localhost: { url: "http://127.0.0.1:8545", chainId: 31337 },
  },
};



