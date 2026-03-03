import { createContext, useContext, type ReactNode } from "react";
import {
  useUIChannelManager,
  type UIChannelManager,
} from "@/hooks/useUIChannel";

const UIChannelContext = createContext<UIChannelManager | null>(null);

export function UIChannelProvider({ children }: { children: ReactNode }) {
  const manager = useUIChannelManager();
  return (
    <UIChannelContext.Provider value={manager}>
      {children}
    </UIChannelContext.Provider>
  );
}

export function useUIChannel(): UIChannelManager {
  const ctx = useContext(UIChannelContext);
  if (!ctx)
    throw new Error("useUIChannel must be used within UIChannelProvider");
  return ctx;
}
