import { useEffect, useState } from "react";

export function useSSE<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const es = new EventSource(url);

    es.onopen = () => setConnected(true);
    es.onmessage = (e) => {
      try {
        setData(JSON.parse(e.data));
      } catch { /* ignore */ }
    };
    es.onerror = () => setConnected(false);

    return () => es.close();
  }, [url]);

  return { data, connected };
}
