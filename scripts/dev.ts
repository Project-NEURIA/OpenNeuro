const rocm = process.argv.includes("--rocm");
const uvArgs = rocm
  ? ["run", "--no-group", "cuda12", "--group", "rocm", "python", "-m", "src.main"]
  : ["run", "python", "-m", "src.main"];

const backend = Bun.spawn(["uv", ...uvArgs], {
  cwd: "./backend",
  stdout: "inherit",
  stderr: "inherit",
});

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
