import { useCallback, useEffect, useRef, useState } from "react";
import { isCancelled, errorMessage, isApiError, type ApiError } from "../api/client";
import { cacheDelete, cacheGet, cacheSet } from "../utils/fetchCache";
import { useEpoch } from "./EpochContext";


export interface Resource<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  refetch: () => void;
}


function toApiError(e: unknown): ApiError {
  if (isApiError(e)) return e;
  const err = new Error(errorMessage(e)) as ApiError;
  err.name = "ApiError";
  err.status = null;
  err.isNetwork = false;
  return err;
}


// key === null means "nothing to fetch yet". Responses are cached by key; a simulation run clears
// the cache and bumps the epoch so everything refetches. A newer request aborts the previous one.
export function useResource<T>(key: string | null, fetcher: (signal: AbortSignal) => Promise<T>): Resource<T> {
  const { epoch } = useEpoch();
  const [nonce, setNonce] = useState(0);
  const [state, setState] = useState<{ data: T | null; error: ApiError | null; loading: boolean }>({
    data: null,
    error: null,
    loading: key !== null,
  });
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;


  useEffect(() => {
    if (key === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    const cached = cacheGet<T>(key);
    if (cached !== undefined) {
      setState({ data: cached, error: null, loading: false });
      return;
    }
    const controller = new AbortController();
    setState({ data: null, error: null, loading: true });
    fetcherRef
      .current(controller.signal)
      .then((data) => {
        cacheSet(key, data);
        setState({ data, error: null, loading: false });
      })
      .catch((e: unknown) => {
        if (controller.signal.aborted || isCancelled(e)) return;
        setState({ data: null, error: toApiError(e), loading: false });
      });
    return () => controller.abort();
  }, [key, epoch, nonce]);


  const refetch = useCallback(() => {
    if (key !== null) cacheDelete(key);
    setNonce((n) => n + 1);
  }, [key]);


  return { ...state, refetch };
}



