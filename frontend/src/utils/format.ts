const num2 = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const int0 = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });


export const NA = "n/a";


export function fmtKwh(n: number | null | undefined): string {
  return n === null || n === undefined || !Number.isFinite(n) ? NA : num2.format(n);
}


export function fmtNum(n: number | null | undefined, digits = 2): string {
  return n === null || n === undefined || !Number.isFinite(n) ? NA : n.toFixed(digits);
}


export function fmtInt(n: number): string {
  return int0.format(n);
}


// "Rs. 1,833.56" (negative: "-Rs. 12.00")
export function fmtRs(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return NA;
  return `${n < 0 ? "-" : ""}Rs. ${num2.format(Math.abs(n))}`;
}


export function fmtPct(n: number | null | undefined, digits = 1): string {
  return n === null || n === undefined || !Number.isFinite(n) ? NA : `${n.toFixed(digits)}%`;
}


// Explicit sign so positive/negative is never conveyed by colour alone.
export function fmtSigned(n: number, digits = 2): string {
  const s = Math.abs(n).toFixed(digits);
  return n > 0 ? `+${s}` : n < 0 ? `−${s}` : s;
}


export function shortHash(h: string): string {
  return h.length > 14 ? `${h.slice(0, 6)}...${h.slice(-4)}` : h;
}



