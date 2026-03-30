/**
 * `bun run remote <runpod-tailscale-ip>` — run the frontend locally,
 * proxying to a backend on a remote server (e.g. RunPod) via Tailscale.
 *
 * Usage:
 *   bun run remote 100.78.23.45       # RunPod's Tailscale IP
 *   BACKEND_URL=http://100.78.23.45:8000 bun run remote   # alternative
 *
 * On RunPod, start the backend separately:
 *   VRCHAT_IP=<your-local-tailscale-ip> uv run python -m src.main
 */

const ip = Bun.argv[2];
const backendUrl =
  ip != null
    ? `http://${ip}:8000`
    : process.env.BACKEND_URL;

if (!backendUrl) {
  console.error(
    "[remote] Usage: bun run remote <runpod-tailscale-ip>\n" +
      "  Or set BACKEND_URL in your environment.",
  );
  process.exit(1);
}

console.log(`[remote] BACKEND_URL=${backendUrl}`);
console.log("[remote] Proxying frontend to remote backend.\n");

const frontend = Bun.spawn(["bunx", "--bun", "vite"], {
  cwd: "./frontend",
  stdout: "inherit",
  stderr: "inherit",
  env: { ...process.env, BACKEND_URL: backendUrl },
});

process.on("SIGINT", () => {
  frontend.kill();
  process.exit();
});

await frontend.exited;
