import fs from "fs";
import path from "path";
import { Contract, EventLog, JsonRpcProvider, Network, Wallet } from "ethers";
import { config } from "../config/env";
import { httpError } from "../utils/errors";
import { kwhToMilli, priceToMilli } from "../utils/units";


interface Artifact {
  address: string;
  chainId: number;
  deployBlock: number;
  abi: unknown[];
}


interface ChainContext {
  provider: JsonRpcProvider;
  wallet: Wallet;
  contract: Contract;
  address: string;
  chainId: number;
  deployBlock: number;
}


const ARTIFACT_PATH = path.resolve(process.cwd(), "src/contracts/EnergyTrading.json");


let ctx: ChainContext | null = null;


function loadArtifact(): Artifact {
  if (!fs.existsSync(ARTIFACT_PATH)) {
    throw httpError(503, `Contract artifact not found at ${ARTIFACT_PATH}. Start the Hardhat node and run \`npm run deploy\`.`);
  }
  return JSON.parse(fs.readFileSync(ARTIFACT_PATH, "utf8")) as Artifact;
}


export function getChain(): ChainContext {
  if (ctx) return ctx;
  const artifact = loadArtifact();
  const address = config.CONTRACT_ADDRESS ?? artifact.address;
  // staticNetwork avoids ethers' noisy network-detection retry loop when the node is down.
  const provider = new JsonRpcProvider(config.RPC_URL, artifact.chainId, {
    staticNetwork: Network.from(artifact.chainId),
  });
  const wallet = new Wallet(config.OWNER_PRIVATE_KEY, provider);
  const contract = new Contract(address, artifact.abi as never, wallet);
  ctx = { provider, wallet, contract, address, chainId: artifact.chainId, deployBlock: artifact.deployBlock };
  return ctx;
}


function chainDown(e: unknown) {
  return httpError(503, `Blockchain unreachable at ${config.RPC_URL}. Is \`npm run chain\` running? (${(e as Error).message})`);
}


export async function verifyContractDeployed(): Promise<void> {
  const { provider, address } = getChain();
  let code: string;
  try {
    code = await provider.getCode(address);
  } catch (e) {
    throw chainDown(e);
  }
  if (code === "0x") {
    throw httpError(409, `No contract at ${address} on ${config.RPC_URL}. The node was likely restarted: run \`npm run deploy\` again.`);
  }
}


export interface ExecuteTradeParams {
  sellerId: number;
  buyerId: number;
  kWh: number;
  pricePerKwh: number;
  slotTimestampMs: number;
}


// Sequential send + wait keeps nonces ordered; callers must not run this concurrently.
export async function executeTradeOnChain(p: ExecuteTradeParams): Promise<{ txHash: string; tradeId: number }> {
  const { contract } = getChain();
  const tx = await contract.executeTrade(
    BigInt(p.sellerId),
    BigInt(p.buyerId),
    BigInt(kwhToMilli(p.kWh)),
    BigInt(priceToMilli(p.pricePerKwh)),
    BigInt(Math.floor(p.slotTimestampMs / 1000))
  );
  const receipt = await tx.wait();
  if (!receipt || receipt.status !== 1) throw new Error(`Transaction ${tx.hash} did not succeed`);
  for (const log of receipt.logs) {
    const parsed = contract.interface.parseLog(log);
    if (parsed?.name === "TradeExecuted") {
      return { txHash: receipt.hash as string, tradeId: Number(parsed.args.tradeId) };
    }
  }
  throw new Error(`TradeExecuted event not found in receipt ${receipt.hash}`);
}


// Used to recover a trade whose DB row is still `pending` after a crash.
export async function findTradeOnChain(
  sellerId: number,
  buyerId: number,
  slotTimestampMs: number
): Promise<{ txHash: string; tradeId: number } | null> {
  const { contract, deployBlock } = getChain();
  const slotSec = BigInt(Math.floor(slotTimestampMs / 1000));
  const events = await contract.queryFilter(
    contract.filters.TradeExecuted(null, BigInt(sellerId), BigInt(buyerId)),
    deployBlock
  );
  for (const e of events) {
    const log = e as EventLog;
    if (log.args.slotTimestamp === slotSec) return { txHash: log.transactionHash, tradeId: Number(log.args.tradeId) };
  }
  return null;
}


export interface OnChainBalances {
  energyBalanceMilli: number;
  tokenBalanceMilli: number;
  energyBalanceKwh: number;
  tokenBalance: number;
}


export async function getOnChainBalances(householdId: number): Promise<OnChainBalances> {
  const { contract } = getChain();
  try {
    const [energy, token] = await Promise.all([
      contract.getEnergyBalance(BigInt(householdId)),
      contract.getTokenBalance(BigInt(householdId)),
    ]);
    return {
      energyBalanceMilli: Number(energy),
      tokenBalanceMilli: Number(token),
      energyBalanceKwh: Number(energy) / 1000,
      tokenBalance: Number(token) / 1000,
    };
  } catch (e) {
    throw chainDown(e);
  }
}


export async function getOnChainTradeCount(): Promise<number> {
  const { contract } = getChain();
  try {
    return Number(await contract.getTradeCount());
  } catch (e) {
    throw chainDown(e);
  }
}


export async function checkChain(): Promise<{
  up: boolean;
  chainId?: number;
  blockNumber?: number;
  contractAddress?: string;
  tradeCount?: number;
  error?: string;
}> {
  try {
    const { provider, address, chainId } = getChain();
    const blockNumber = await provider.getBlockNumber();
    await verifyContractDeployed();
    const tradeCount = await getOnChainTradeCount();
    return { up: true, chainId, blockNumber, contractAddress: address, tradeCount };
  } catch (e) {
    return { up: false, error: (e as Error).message };
  }
}



