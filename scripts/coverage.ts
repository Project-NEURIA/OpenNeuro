type CoverageMode = "all" | "backend" | "frontend" | "badges";

const mode = (process.argv[2] ?? "all") as CoverageMode;

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
      "docs/backend-coverage.svg",
    );
  }

  if (target === "all" || target === "frontend") {
    command.push(
      "--frontend-lcov",
      "frontend/coverage/lcov.info",
      "--frontend-svg",
      "docs/frontend-coverage.svg",
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
    await run(["uv", "run", "pytest"], "./backend");
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
