const fs = require("fs");
const path = require("path");
const { ethers } = require("hardhat");


const N_HOUSEHOLDS = 25;
const OUT_PATH = path.resolve(__dirname, "../../backend/src/contracts/EnergyTrading.json");


async function main() {
  const signers = await ethers.getSigners();
  if (signers.length < N_HOUSEHOLDS) {
    throw new Error(`Need ${N_HOUSEHOLDS} accounts, node exposes ${signers.length}`);
  }


  const factory = await ethers.getContractFactory("EnergyTrading");
  const contract = await factory.deploy();
  await contract.waitForDeployment();
  const address = await contract.getAddress();
  const deployBlock = (await ethers.provider.getBlockNumber());


  for (let id = 0; id < N_HOUSEHOLDS; id++) {
    await (await contract.registerHousehold(id, signers[id].address)).wait();
  }


  const { chainId } = await ethers.provider.getNetwork();
  const artifact = await hre.artifacts.readArtifact("EnergyTrading");
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(
    OUT_PATH,
    JSON.stringify({ address, chainId: Number(chainId), deployBlock, abi: artifact.abi }, null, 2)
  );


  console.log(`EnergyTrading deployed at ${address} (chainId ${chainId})`);
  console.log(`Registered ${N_HOUSEHOLDS} households`);
  console.log(`Wrote ${OUT_PATH}`);
}


main().catch((e) => {
  console.error(e);
  process.exit(1);
});



