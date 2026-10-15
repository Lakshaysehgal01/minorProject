// Backend timestamps are naive local time stored as UTC ("2026-10-20T08:00:00.000Z" means 08:00).
// Everything here works on the string/UTC fields so the browser's time zone can never shift an hour.


export const REPLAY_MIN = "2026-10-20";
export const REPLAY_MAX = "2026-12-31";
const HOUR_MS = 3_600_000;


export function addDays(date: string, n: number): string {
  const [y, m, d] = date.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d + n)).toISOString().slice(0, 10);
}


export function isInReplayRange(date: string): boolean {
  return date >= REPLAY_MIN && date <= REPLAY_MAX;
}


export function dateOf(iso: string): string {
  return iso.slice(0, 10);
}


export function hourOf(iso: string): number {
  return Number(iso.slice(11, 13));
}


export function hourLabel(hour: number): string {
  return `${String(hour).padStart(2, "0")}:00`;
}


// "2026-10-20 08:00"
export function slotLabel(iso: string): string {
  return `${iso.slice(0, 10)} ${iso.slice(11, 16)}`;
}


export function minusOneHour(iso: string): string {
  return new Date(Date.parse(iso) - HOUR_MS).toISOString();
}


// The day most recently simulated, derived from the backend's cursor (the NEXT slot to run).
export function lastSimulatedDate(cursorIso: string, slotsCompleted: number): string {
  return slotsCompleted <= 0 ? REPLAY_MIN : dateOf(minusOneHour(cursorIso));
}



