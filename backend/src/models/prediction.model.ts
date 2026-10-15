import { Schema, model } from "mongoose";


export interface PredictionDoc {
  householdId: number;
  slotTimestamp: Date;
  predictedSolarKwh: number;
  predictedConsumptionKwh: number;
  actualSolarKwh: number;
  actualConsumptionKwh: number;
  surplusKwh: number; // predictedSolar - predictedConsumption
  modelUsed: string;
  isSolarAnomaly: boolean;
  isConsumptionAnomaly: boolean;
}


const predictionSchema = new Schema<PredictionDoc>(
  {
    householdId: { type: Number, required: true },
    slotTimestamp: { type: Date, required: true },
    predictedSolarKwh: { type: Number, required: true },
    predictedConsumptionKwh: { type: Number, required: true },
    actualSolarKwh: { type: Number, required: true },
    actualConsumptionKwh: { type: Number, required: true },
    surplusKwh: { type: Number, required: true },
    modelUsed: { type: String, required: true },
    isSolarAnomaly: { type: Boolean, required: true, default: false },
    isConsumptionAnomaly: { type: Boolean, required: true, default: false },
  },
  { versionKey: false }
);


predictionSchema.index({ householdId: 1, slotTimestamp: 1 }, { unique: true });
predictionSchema.index({ slotTimestamp: 1 });


export const Prediction = model<PredictionDoc>("Prediction", predictionSchema);



