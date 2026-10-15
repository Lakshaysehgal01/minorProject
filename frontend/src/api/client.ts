import axios, { AxiosError } from "axios";
import { mockAdapter } from "./mock/adapter";


export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:4000";
export const USE_MOCK: boolean = import.meta.env.VITE_USE_MOCK === "true";


// A simulated day is one synchronous request (~30 s); everything else is quick.
export const DEFAULT_TIMEOUT_MS = 15_000;
export const SIMULATION_TIMEOUT_MS = 5 * 60_000;


export interface ApiError extends Error {
  status: number | null;
  isNetwork: boolean;
}


function makeApiError(message: string, status: number | null, isNetwork: boolean): ApiError {
  const err = new Error(message) as ApiError;
  err.name = "ApiError";
  err.status = status;
  err.isNetwork = isNetwork;
  return err;
}


export function isApiError(e: unknown): e is ApiError {
  return e instanceof Error && e.name === "ApiError";
}


export function isCancelled(e: unknown): boolean {
  return axios.isCancel(e);
}


export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}


export const api = axios.create({
  baseURL: API_URL,
  timeout: DEFAULT_TIMEOUT_MS,
  adapter: USE_MOCK ? mockAdapter : undefined,
});


api.interceptors.response.use(
  (res) => res,
  (error: unknown) => {
    if (axios.isCancel(error)) return Promise.reject(error);
    const ax = error as AxiosError<{ error?: string; issues?: { path: string; message: string }[] }>;
    if (!ax.response) {
      const timedOut = ax.code === "ECONNABORTED" || ax.code === "ETIMEDOUT";
      return Promise.reject(
        makeApiError(timedOut ? "The request timed out" : `Backend unreachable at ${API_URL}`, null, true)
      );
    }
    const body = ax.response.data;
    let message = typeof body?.error === "string" ? body.error : `Request failed (${ax.response.status})`;
    if (body?.issues?.length) message += `: ${body.issues.map((i) => `${i.path} ${i.message}`).join("; ")}`;
    return Promise.reject(makeApiError(message, ax.response.status, false));
  }
);



