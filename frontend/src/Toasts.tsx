// G3 shared toasts + clipboard helper (STANDARD.md 3.8). Copy verbatim to frontend/src/Toasts.tsx.
// Render <ToastHost /> once at the root of App. Call toast.ok("Saved."), toast.error("..."), toast.info("...").
import { useEffect, useState } from "react";

type Kind = "ok" | "info" | "warn" | "danger";
interface Toast { id: number; kind: Kind; text: string }

let items: Toast[] = [];
let nextId = 1;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

function push(kind: Kind, text: string) {
  const id = nextId++;
  items = [...items.filter((t) => t.text !== text), { id, kind, text }].slice(-4);
  emit();
  if (kind === "ok" || kind === "info") window.setTimeout(() => dismiss(id), 4000); // errors and warnings stay
}
function dismiss(id: number) { items = items.filter((t) => t.id !== id); emit(); }

export const toast = {
  ok: (text: string) => push("ok", text),
  info: (text: string) => push("info", text),
  warn: (text: string) => push("warn", text),
  error: (text: string) => push("danger", text),
  clear: () => { items = []; emit(); },
};

export function ToastHost() {
  const [, setTick] = useState(0);
  useEffect(() => {
    const l = () => setTick((n) => n + 1);
    listeners.add(l);
    return () => { listeners.delete(l); };
  }, []);
  return (
    <div className="toast-stack">
      {items.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} role={t.kind === "danger" ? "alert" : "status"}
          aria-live={t.kind === "danger" ? "assertive" : "polite"}>
          <span>{t.text}</span>
          <button type="button" className="toast-close" aria-label="Dismiss" onClick={() => dismiss(t.id)}>×</button>
        </div>
      ))}
    </div>
  );
}

/**
 * Copy text. Works on plain http LAN origins (navigator.clipboard is undefined there) via a hidden
 * textarea + execCommand. Never throws; returns false when both ways fail.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the legacy path
  }
  try {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(area);
    return ok;
  } catch {
    return false;
  }
}

/** Button that copies `text` and shows "Copied" for 2 s (or an error toast). */
export function CopyButton({ text, label = "Copy", className = "secondary-button small-button" }: { text: string; label?: string; className?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button type="button" className={className} onClick={async () => {
      if (await copyText(text)) {
        setDone(true);
        window.setTimeout(() => setDone(false), 2000);
      } else {
        toast.error("Copy failed. Select the text and copy it manually.");
      }
    }}>
      {done ? "Copied" : label}
    </button>
  );
}
