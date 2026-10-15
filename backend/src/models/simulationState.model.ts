import { Schema, model } from "mongoose";


export interface SimulationStateDoc {
  key: "singleton";
  cursor: Date; // next slot to simulate
  contractAddress?: string;
  chainId?: number;
  updatedAt: Date;
}


const simulationStateSchema = new Schema<SimulationStateDoc>(
  {
    key: { type: String, required: true, unique: true, default: "singleton" },
    cursor: { type: Date, required: true },
    contractAddress: { type: String },
    chainId: { type: Number },
  },
  { versionKey: false, timestamps: { createdAt: false, updatedAt: true } }
);


export const SimulationState = model<SimulationStateDoc>("SimulationState", simulationStateSchema);



