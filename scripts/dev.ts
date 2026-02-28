const backend = Bun.spawn(["uv", "run", "python", "-m", "src.main"], {
  cwd: "./backend",
  stdout: "inherit",
  stderr: "inherit",
});

async function waitForBackend(url: string, timeoutMs = 30000): Promise<void> {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 1000);
      const res = await fetch(url, { signal: controller.signal });
      clearTimeout(t);
      if (res.ok) return;
    } catch {
      // backend not ready yet
    }
    await Bun.sleep(500);
  }
  console.warn(`[dev] Backend not ready after ${timeoutMs}ms, starting frontend anyway.`);
}

await waitForBackend("http://localhost:8000/project/current");

const frontend = Bun.spawn(["bun", "run", "dev"], {
  cwd: "./frontend",
  stdout: "inherit",
  stderr: "inherit",
});

process.on("SIGINT", () => {
  frontend.kill();
  backend.kill();
  process.exit();
});

await Promise.all([frontend.exited, backend.exited]);
