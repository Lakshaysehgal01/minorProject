import type { ReactNode } from "react";


export type Tone = "green" | "red" | "amber" | "blue" | "slate";


const TONES: Record<Tone, string> = {
  green: "bg-emerald-100 text-emerald-900 border-emerald-600",
  red: "bg-red-100 text-red-900 border-red-600",
  amber: "bg-amber-100 text-amber-900 border-amber-600",
  blue: "bg-blue-100 text-blue-900 border-blue-600",
  slate: "bg-slate-100 text-slate-800 border-slate-500",
};


export function Badge({ tone = "slate", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-sm font-semibold whitespace-nowrap ${TONES[tone]}`}
    >
      {children}
    </span>
  );
}



