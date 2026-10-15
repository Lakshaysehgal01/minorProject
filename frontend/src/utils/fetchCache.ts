// Tiny in-memory response cache. Cleared whenever a simulation run changes server data.
const cache = new Map<string, unknown>();


export function cacheGet<T>(key: string): T | undefined {
  return cache.get(key) as T | undefined;
}


export function cacheSet(key: string, value: unknown): void {
  cache.set(key, value);
}


export function cacheDelete(key: string): void {
  cache.delete(key);
}


export function cacheClear(): void {
  cache.clear();
}



