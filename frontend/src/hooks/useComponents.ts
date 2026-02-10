import { useEffect, useState } from "react";
import { fetchComponents } from "@/lib/api";
import type { ComponentInfo } from "@/lib/types";

export function useComponents() {
  const [components, setComponents] = useState<ComponentInfo[]>([]);

  useEffect(() => {
    fetchComponents()
      .then(setComponents)
      .catch((err) => console.error("[components] Fetch failed:", err));
  }, []);

  return components;
}
