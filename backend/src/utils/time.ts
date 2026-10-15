// All timestamps in the dataset are naive local-time strings. We treat them as UTC
// everywhere so the simulation never depends on the host machine's timezone.
export const HOUR_MS = 3_600_000;
export const DAY_MS = 24 * HOUR_MS;


export const SIM_START_MS = Date.UTC(2026, 9, 20); // 2026-10-20T00:00Z (test period start)
export const SIM_END_MS = Date.UTC(2027, 0, 1); // exclusive
export const TOTAL_SLOTS = (SIM_END_MS - SIM_START_MS) / HOUR_MS; // 1752


export function parseNaiveUtc(s: string): number {
  return Date.parse(s.trim().replace(" ", "T") + "Z");
}


export function dateKey(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}


export function startOfDay(ms: number): number {
  return Math.floor(ms / DAY_MS) * DAY_MS;
}


export function parseDateKey(s: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return null;
  const ms = Date.parse(`${s}T00:00:00Z`);
  return Number.isNaN(ms) || dateKey(ms) !== s ? null : ms;
}



