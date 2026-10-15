import { NextFunction, Request, Response } from "express";
import { z } from "zod";
import { Prediction } from "../models/prediction.model";
import { Trade } from "../models/trade.model";
import { getHouseholdInfos } from "../services/dataSource.service";
import { getOnChainBalances } from "../services/blockchain.service";
import { getDailySummary } from "../services/daily-summary.service";
import { getForecast } from "../services/forecast.service";
import { getModelComparison } from "../services/results.service";
import { getStats } from "../services/stats.service";
import { getHealth } from "../services/health.service";
import { getActiveModels } from "../services/modelSelection.service";
import { DAY_MS, parseDateKey } from "../utils/time";
import { httpError } from "../utils/errors";
import { dateSchema, householdIdSchema, paginationSchema } from "./validation";


function dayRange(date: string) {
  const start = parseDateKey(date)!;
  return { $gte: new Date(start), $lt: new Date(start + DAY_MS) };
}


export async function dailySummary(req: Request, res: Response, next: NextFunction) {
  try {
    const { date } = z.object({ date: dateSchema }).parse(req.query);
    res.json(await getDailySummary(date));
  } catch (e) {
    next(e);
  }
}


export async function households(_req: Request, res: Response, next: NextFunction) {
  try {
    const infos = getHouseholdInfos();
    const balances = await Promise.all(infos.map((h) => getOnChainBalances(h.householdId)));
    res.json(infos.map((h, i) => ({ ...h, ...balances[i] })));
  } catch (e) {
    next(e);
  }
}


const tradeSelect = "-__v";


export async function trades(req: Request, res: Response, next: NextFunction) {
  try {
    const q = paginationSchema
      .extend({
        date: dateSchema.optional(),
        status: z.enum(["pending", "confirmed", "failed"]).optional(),
      })
      .parse(req.query);
    const filter: Record<string, unknown> = {};
    if (q.date) filter.slotTimestamp = dayRange(q.date);
    if (q.status) filter.status = q.status;
    const [items, total] = await Promise.all([
      Trade.find(filter)
        .select(tradeSelect)
        .sort({ slotTimestamp: -1, _id: -1 })
        .skip((q.page - 1) * q.limit)
        .limit(q.limit)
        .lean(),
      Trade.countDocuments(filter),
    ]);
    res.json({ page: q.page, limit: q.limit, total, trades: items });
  } catch (e) {
    next(e);
  }
}


export async function tradesByHousehold(req: Request, res: Response, next: NextFunction) {
  try {
    const id = householdIdSchema.parse(req.params.id);
    if (!getHouseholdInfos().some((h) => h.householdId === id)) throw httpError(404, `Unknown household ${id}`);
    const q = paginationSchema.parse(req.query);
    const filter = { $or: [{ sellerId: id }, { buyerId: id }] };
    const [items, total] = await Promise.all([
      Trade.find(filter)
        .select(tradeSelect)
        .sort({ slotTimestamp: -1, _id: -1 })
        .skip((q.page - 1) * q.limit)
        .limit(q.limit)
        .lean(),
      Trade.countDocuments(filter),
    ]);
    res.json({ householdId: id, page: q.page, limit: q.limit, total, trades: items });
  } catch (e) {
    next(e);
  }
}


export async function predictions(req: Request, res: Response, next: NextFunction) {
  try {
    const id = householdIdSchema.parse(req.params.householdId);
    if (!getHouseholdInfos().some((h) => h.householdId === id)) throw httpError(404, `Unknown household ${id}`);
    const { date } = z.object({ date: dateSchema.optional() }).parse(req.query);


    let day = date;
    if (!day) {
      const latest = await Prediction.findOne({ householdId: id }).sort({ slotTimestamp: -1 }).lean();
      if (!latest) throw httpError(404, `No predictions for household ${id} yet. Run the simulation first.`);
      day = latest.slotTimestamp.toISOString().slice(0, 10);
    }
    const items = await Prediction.find({ householdId: id, slotTimestamp: dayRange(day) })
      .select("-__v -_id")
      .sort({ slotTimestamp: 1 })
      .lean();
    if (!items.length) throw httpError(404, `No predictions for household ${id} on ${day}`);
    res.json({ householdId: id, date: day, hours: items });
  } catch (e) {
    next(e);
  }
}


export async function forecast(req: Request, res: Response, next: NextFunction) {
  try {
    const id = householdIdSchema.parse(req.params.householdId);
    const { date } = z.object({ date: dateSchema }).parse(req.query);
    res.json(await getForecast(id, date));
  } catch (e) {
    next(e);
  }
}


export async function stats(_req: Request, res: Response, next: NextFunction) {
  try {
    res.json(await getStats());
  } catch (e) {
    next(e);
  }
}


export async function modelResults(_req: Request, res: Response, next: NextFunction) {
  try {
    res.json(getModelComparison());
  } catch (e) {
    next(e);
  }
}


export async function activeModels(_req: Request, res: Response, next: NextFunction) {
  try {
    res.json(getActiveModels());
  } catch (e) {
    next(e);
  }
}


export async function health(_req: Request, res: Response, next: NextFunction) {
  try {
    const result = await getHealth();
    res.status(result.status === "ok" ? 200 : 503).json(result);
  } catch (e) {
    next(e);
  }
}





