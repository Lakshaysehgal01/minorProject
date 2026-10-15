import mongoose from "mongoose";
import { config } from "./env";


export async function connectDb(): Promise<void> {
  mongoose.set("strictQuery", true);
  await mongoose.connect(config.MONGO_URI, { serverSelectionTimeoutMS: 5000 });
}


export async function disconnectDb(): Promise<void> {
  await mongoose.disconnect();
}



