import { useEffect, useState } from "react";

/** Mock results only when explicitly asked for (VITE_USE_MOCK=true), e.g. for UI work without a backend. */
export const USE_MOCK = import.meta.env.VITE_USE_MOCK === "true";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((e: { msg: string }) => e.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return `The server returned an error (${response.status}).`;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(
      "Couldn’t reach the detector. If it hasn’t been used for a while it may be starting up, so try again in a minute.",
      0,
    );
  }
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  return (await response.json()) as T;
}

export type Health = {
  status: "ok" | "loading" | "no-model";
  model: {
    loaded: boolean;
    loading: boolean;
    version: string | null;
    task: string | null;
    classes: { id: string; label: string }[];
    device: string | null;
    macro_f1: number | null;
    error: string | null;
  };
  database: boolean;
  storage: boolean;
  max_upload_mb: number;
};

export type HealthState =
  | { kind: "checking" }
  | { kind: "mock" }
  | { kind: "offline"; message: string }
  | { kind: "ready" | "loading" | "no-model"; health: Health };

let cached: HealthState | null = null;
const listeners = new Set<(s: HealthState) => void>();
let timer: number | undefined;

async function poll(): Promise<void> {
  window.clearTimeout(timer);
  let next: HealthState;
  if (USE_MOCK) {
    next = { kind: "mock" };
  } else {
    try {
      const health = await request<Health>("/health");
      next = {
        kind: health.model.loaded ? "ready" : health.model.loading ? "loading" : "no-model",
        health,
      };
    } catch (err) {
      next = { kind: "offline", message: (err as Error).message };
    }
  }
  cached = next;
  listeners.forEach((l) => l(next));
  // Keep checking until the model is up (free hosts sleep and take a while to wake).
  if (next.kind === "offline" || next.kind === "loading") timer = window.setTimeout(poll, 8000);
}

/** Shared detector status, checked once and re-checked while the server is waking up. */
export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>(cached ?? { kind: "checking" });
  useEffect(() => {
    listeners.add(setState);
    if (!cached) void poll();
    return () => {
      listeners.delete(setState);
    };
  }, []);
  return state;
}

export function refreshHealth(): void {
  void poll();
}

export function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const t = window.setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(t);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}
