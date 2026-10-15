import { Schema, model } from "mongoose";


export type TradeStatus = "pending" | "confirmed" | "failed";


export interface TradeDoc {
  tradeId?: number; // on-chain id, set once confirmed
  sellerId: number;
  buyerId: number;
  kWh: number;
  kWhMilli: number;
  pricePerKwh: number;
  costMilli: number;
  txHash?: string;
  status: TradeStatus;
  error?: string;
  slotTimestamp: Date;
  createdAt: Date;
}


const tradeSchema = new Schema<TradeDoc>(
  {
    tradeId: { type: Number },
    sellerId: { type: Number, required: true },
    buyerId: { type: Number, required: true },
    kWh: { type: Number, required: true },
    kWhMilli: { type: Number, required: true },
    pricePerKwh: { type: Number, required: true },
    costMilli: { type: Number, required: true },
    txHash: { type: String },
    status: { type: String, enum: ["pending", "confirmed", "failed"], required: true, default: "pending" },
    error: { type: String },
    slotTimestamp: { type: Date, required: true },
  },
  { versionKey: false, timestamps: { createdAt: true, updatedAt: false } }
);


// A greedy match never pairs the same seller/buyer twice within a slot.
tradeSchema.index({ slotTimestamp: 1, sellerId: 1, buyerId: 1 }, { unique: true });
tradeSchema.index({ sellerId: 1 });
tradeSchema.index({ buyerId: 1 });
tradeSchema.index({ status: 1 });


export const Trade = model<TradeDoc>("Trade", tradeSchema);



