import { NextFunction, Request, Response } from "express";
import { z } from "zod";
import { MAX_HOURS_PER_RUN, resetSimulation, runDay, runHours, runSlot } from "../services/simulation.service";


export async function tick(_req: Request, res: Response, next: NextFunction) {
  try {
    res.json(await runSlot());
  } catch (e) {
    next(e);
  }
}


export async function day(_req: Request, res: Response, next: NextFunction) {
  try {
    res.json(await runDay());
  } catch (e) {
    next(e);
  }
}


export async function run(req: Request, res: Response, next: NextFunction) {
  try {
    const { hours } = z
      .object({ hours: z.coerce.number().int().min(1).max(MAX_HOURS_PER_RUN) })
      .parse(req.query);
    res.json(await runHours(hours));
  } catch (e) {
    next(e);
  }
}


export async function reset(_req: Request, res: Response, next: NextFunction) {
  try {
    res.json(await resetSimulation());
  } catch (e) {
    next(e);
  }
}



