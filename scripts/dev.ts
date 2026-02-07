const frontend = Bun.spawn(["bun", "run", "dev"], {
  cwd: "./frontend",
  stdout: "inherit",
  stderr: "inherit",
});

const backend = Bun.spawn(["uv", "run", "python", "-m", "src.main"], {
  cwd: "./backend",
  stdout: "inherit",
  stderr: "inherit",
});

process.on("SIGINT", () => {
  frontend.kill();
  backend.kill();
  process.exit();
});

await Promise.all([frontend.exited, backend.exited]);
