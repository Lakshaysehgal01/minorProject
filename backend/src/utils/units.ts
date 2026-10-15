// On-chain units: energy in milli-kWh, prices in milli-credits per kWh.
export const kwhToMilli = (kwh: number): number => Math.round(kwh * 1000);
export const milliToKwh = (milli: number): number => milli / 1000;
export const priceToMilli = (price: number): number => Math.round(price * 1000);


// Must mirror EnergyTrading.sol: cost = kWhMilli * pricePerKwhMilli / 1000 (integer division).
export const costMilli = (kWhMilli: number, pricePerKwhMilli: number): number =>
  Math.floor((kWhMilli * pricePerKwhMilli) / 1000);



