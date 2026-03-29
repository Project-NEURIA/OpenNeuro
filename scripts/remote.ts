/**
 * `bun run remote` — start OpenNeuro for remote access via Tailscale.
 *
 * Detects the local machine's Tailscale IP and sets VRCHAT_IP so all
 * VRChat/OSC components default to routing traffic back to the user's PC.
 *
 * Usage:
 *   bun run remote                  # auto-detect peer IP from `tailscale status`
 *   bun run remote 100.64.1.12      # explicit local IP override
 */

async function getTailscalePeerIP(): Promise<string | null> {
  try {
    const proc = Bun.spawn(["tailscale", "status", "--json"], {
      stdout: "pipe",
      stderr: "pipe",
    });
    const text = await new Response(proc.stdout).text();
    await proc.exited;

    const status = JSON.parse(text);
    const selfID: string = status.Self?.ID ?? "";

    // Find the first online peer that isn't this machine
    for (const [id, peer] of Object.entries<any>(status.Peer ?? {})) {
      if (id === selfID) continue;
      if (peer.Online && peer.TailscaleIPs?.length) {
        return peer.TailscaleIPs[0];
      }
    }
  } catch {
    // tailscale CLI not available
  }
  return null;
}

async function waitForBackend(
  url: string,
  timeoutMs: number = 120_000,
): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      await fetch(url);
      return;
    } catch {
      await Bun.sleep(500);
    }
  }
  console.warn("[remote] Backend did not become ready in time, starting frontend anyway.");
}

const explicitIP = Bun.argv[2];
const vrchatIP =
  explicitIP ??
  process.env.VRCHAT_IP ??
  (await getTailscalePeerIP());

if (!vrchatIP) {
  console.error(
    "[remote] Could not detect a Tailscale peer IP.\n" +
      "  Pass it explicitly:  bun run remote <tailscale-ip>\n" +
      "  Or set VRCHAT_IP in your environment.",
  );
  process.exit(1);
}

console.log(`[remote] VRCHAT_IP=${vrchatIP}`);
console.log(
  "[remote] VRChat/OSC components will target this IP by default.\n",
);

const env = { ...process.env, VRCHAT_IP: vrchatIP };

const backend = Bun.spawn(["uv", "run", "python", "-m", "src.main"], {
  cwd: "./backend",
  stdout: "inherit",
  stderr: "inherit",
  env,
});

console.log("[remote] Waiting for backend to be ready...");
await waitForBackend("http://localhost:8000/component");
console.log("[remote] Backend is ready, starting frontend.\n");

// Run Vite on Node to avoid Bun's broken node:http proxy (oven-sh/bun#28396).
const frontend = Bun.spawn(
  ["node", "-e", "require('vite/bin/vite.js')"],
  {
    cwd: "./frontend",
    stdout: "inherit",
    stderr: "inherit",
    env,
  },
);

process.on("SIGINT", () => {
  frontend.kill();
  backend.kill();
  process.exit();
});

await Promise.all([frontend.exited, backend.exited]);
