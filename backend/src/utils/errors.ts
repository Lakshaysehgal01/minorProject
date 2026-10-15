export interface HttpError extends Error {
  status: number;
  details?: unknown;
}


export function httpError(status: number, message: string, details?: unknown): HttpError {
  const err = new Error(message) as HttpError;
  err.status = status;
  err.details = details;
  return err;
}


export function isHttpError(e: unknown): e is HttpError {
  return e instanceof Error && typeof (e as HttpError).status === "number";
}


export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}



