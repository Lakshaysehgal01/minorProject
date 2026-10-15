import cors from "cors";
import express, { NextFunction, Request, Response } from "express";
import { ZodError } from "zod";
import { config } from "./config/env";
import { router } from "./routes";
import { isHttpError } from "./utils/errors";


export function createApp() {
  const app = express();
  app.use(cors({ origin: config.CORS_ORIGINS }));
  app.use(express.json());
  app.use(router);


  app.use((_req, res) => {
    res.status(404).json({ error: "Not found" });
  });


  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
    if (err instanceof ZodError) {
      res.status(400).json({
        error: "Validation failed",
        issues: err.issues.map((i) => ({ path: i.path.join("."), message: i.message })),
      });
      return;
    }
    if (isHttpError(err)) {
      res.status(err.status).json({ error: err.message, ...(err.details ? { details: err.details } : {}) });
      return;
    }
    if ((err as Error)?.name === "CastError") {
      res.status(400).json({ error: (err as Error).message });
      return;
    }
    console.error(err);
    res.status(500).json({ error: "Internal server error" });
  });


  return app;
}



