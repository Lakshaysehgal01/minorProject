import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { cacheClear } from "../utils/fetchCache";


interface EpochValue {
  epoch: number;
  // Call after a simulation run: clears the cache and makes every hook refetch.
  invalidateAll: () => void;
}


const EpochContext = createContext<EpochValue>({ epoch: 0, invalidateAll: () => undefined });


export function EpochProvider({ children }: { children: ReactNode }) {
  const [epoch, setEpoch] = useState(0);
  const invalidateAll = useCallback(() => {
    cacheClear();
    setEpoch((e) => e + 1);
  }, []);
  const value = useMemo(() => ({ epoch, invalidateAll }), [epoch, invalidateAll]);
  return <EpochContext.Provider value={value}>{children}</EpochContext.Provider>;
}


// eslint-disable-next-line react-refresh/only-export-components
export function useEpoch(): EpochValue {
  return useContext(EpochContext);
}



