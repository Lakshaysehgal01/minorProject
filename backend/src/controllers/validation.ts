import { z } from "zod";
import { parseDateKey } from "../utils/time";


export const dateSchema = z
  .string()
  .refine((s) => parseDateKey(s) !== null, { message: "must be a valid date in YYYY-MM-DD format" });


export const paginationSchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(500).default(50),
});


export const householdIdSchema = z.coerce.number().int().min(0);



