const rocm = process.argv.includes("--rocm") || process.argv.includes("--amd");
const uvArgs = rocm
  ? ["run", "--group", "rocm", "--no-default-groups", "python", "-m", "src.main"]
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
