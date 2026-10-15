import { Router } from "express";
import * as sim from "../controllers/simulation.controller";
import * as data from "../controllers/data.controller";


export const router = Router();


router.post("/simulate/tick", sim.tick);
router.post("/simulate/day", sim.day);
router.post("/simulate/run", sim.run);
router.post("/simulate/reset", sim.reset);


router.get("/daily-summary", data.dailySummary);
router.get("/households", data.households);
router.get("/trades", data.trades);
router.get("/trades/household/:id", data.tradesByHousehold);
router.get("/predictions/:householdId", data.predictions);
router.get("/forecast/:householdId", data.forecast);
router.get("/stats", data.stats);
router.get("/results/models", data.modelResults);
router.get("/models/active", data.activeModels);
router.get("/health", data.health);



