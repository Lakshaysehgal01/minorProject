import type { ReactNode } from "react";
import type { Resource } from "../hooks/useResource";
import { Spinner } from "./Spinner";


interface Props<T> {
  resource: Resource<T>;
  // Shown instead of an error when the backend answers 404 (e.g. a day that has not been simulated yet).
  notFoundMessage?: string;
  isEmpty?: (data: T) => boolean;
  emptyMessage?: string;
  children: (data: T) => ReactNode;
}


// Every section renders its own loading / error / empty state so one failing endpoint
// never takes the rest of the page down.
export function AsyncState<T>({ resource, notFoundMessage, isEmpty, emptyMessage, children }: Props<T>) {
  const { data, error, loading, refetch } = resource;


  if (loading) {
    return (
      <div className="flex items-center gap-3 py-8 text-base text-slate-700">
        <Spinner /> Loading…
      </div>
    );
  }


  if (error) {
    if (error.status === 404 && notFoundMessage) {
      return <p className="rounded-lg border border-slate-300 bg-slate-50 p-4 text-base text-slate-800">{notFoundMessage}</p>;
    }
    return (
      <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-lg border-2 border-red-600 bg-red-50 p-4">
        <p className="text-base font-medium text-red-900">{error.message}</p>
        <button
          type="button"
          onClick={refetch}
          className="rounded-md border border-red-700 bg-white px-3 py-1 text-sm font-semibold text-red-900 hover:bg-red-100"
        >
          Retry
        </button>
      </div>
    );
  }


  if (data === null) return null;
  if (isEmpty?.(data)) {
    return <p className="rounded-lg border border-slate-300 bg-slate-50 p-4 text-base text-slate-800">{emptyMessage ?? "No data."}</p>;
  }
  return <>{children(data)}</>;
}



