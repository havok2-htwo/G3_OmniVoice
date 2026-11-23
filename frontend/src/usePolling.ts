// G3 shared polling hooks. Copied verbatim from G3_RERANK frontend/src/hooks.ts (reference).
// Copy to frontend/src/usePolling.ts. Do not edit per repo.
import { useEffect, useRef, useState } from "react";

/** True while the browser tab is visible (document.visibilityState === "visible"). */
export function usePageVisible() {
  const [visible, setVisible] = useState(() => document.visibilityState === "visible");
  useEffect(() => {
    const handler = () => setVisible(document.visibilityState === "visible");
    document.addEventListener("visibilitychange", handler);
    return () => document.removeEventListener("visibilitychange", handler);
  }, []);
  return visible;
}

/**
 * Calls `callback` immediately and then every `intervalMs`, but only while `enabled` and the
 * browser tab is visible. Calls never overlap: the next tick is scheduled after the previous
 * call settled, so a slow backend cannot pile up requests. Returning to the tab polls at once.
 * The callback receives an AbortSignal that fires on unmount/disable, so late responses
 * (for example after logout) can be dropped.
 */
export function useVisiblePolling(
  callback: (signal: AbortSignal) => Promise<unknown> | void,
  intervalMs: number,
  enabled: boolean,
) {
  const visible = usePageVisible();
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    if (!enabled || !visible) {
      return;
    }
    const controller = new AbortController();
    let timer: number | undefined;
    const tick = async () => {
      try {
        await callbackRef.current(controller.signal);
      } catch {
        // the callback reports its own errors; keep polling
      }
      if (!controller.signal.aborted) {
        timer = window.setTimeout(() => void tick(), intervalMs);
      }
    };
    void tick();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [enabled, visible, intervalMs]);
}
