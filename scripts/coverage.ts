type CoverageMode = "all" | "backend" | "frontend" | "badges";

const rocm = process.argv.includes("--rocm");
const mode = (process.argv.filter((a) => !a.startsWith("--"))[2] ?? "all") as CoverageMode;

async function run(command: string[], cwd = ".") {
  const proc = Bun.spawn(command, {
    cwd,
    stdout: "inherit",
    stderr: "inherit",
  });

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    process.exit(exitCode);
  }
}

async function updateBadges(target: "all" | "backend" | "frontend") {
  const command = ["python", "scripts/generate_coverage_badges.py"];

  if (target === "all" || target === "backend") {
    command.push(
      "--backend-xml",
      "backend/tests_runtime/coverage/coverage.xml",
      "--backend-svg",
      ".github/badges/backend-coverage.svg",
    );
  }

  if (target === "all" || target === "frontend") {
    command.push(
      "--frontend-lcov",
      "frontend/coverage/lcov.info",
      "--frontend-svg",
      ".github/badges/frontend-coverage.svg",
    );
  }

  await run(command);
}

async function main() {
  if (!["all", "backend", "frontend", "badges"].includes(mode)) {
    console.error("Usage: bun run scripts/coverage.ts [all|backend|frontend|badges]");
    process.exit(1);
  }

  if (mode === "all" || mode === "backend") {
    const uvRun = rocm
      ? ["uv", "run", "--no-group", "cuda12", "--group", "rocm", "pytest"]
      : ["uv", "run", "pytest"];
    await run(uvRun, "./backend");
  }

  if (mode === "all" || mode === "frontend") {
    await run(["bun", "run", "coverage"], "./frontend");
  }

  if (mode === "badges") {
    await updateBadges("all");
    return;
  }

  await updateBadges(mode === "all" ? "all" : mode);
}

void main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
