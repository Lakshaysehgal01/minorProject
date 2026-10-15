import { useState } from "react";


export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);


  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }


  return (
    <button
      type="button"
      onClick={copy}
      aria-label={`${label} transaction hash`}
      className="rounded border border-slate-400 bg-white px-2 py-0.5 text-sm font-medium text-slate-800 hover:bg-slate-100"
    >
      {copied ? "Copied" : label}
    </button>
  );
}



