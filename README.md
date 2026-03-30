# OpenNeuro
[![Frontend Test Coverage](docs/frontend-coverage.svg)](./docs/frontend-coverage.svg)
[![Backend Test Coverage](docs/backend-coverage.svg)](./docs/backend-coverage.svg)

## Prerequisites

- [Bun](https://bun.sh/)
- [uv](https://docs.astral.sh/uv/)
- [Rust](https://rustup.rs/) (for desktop app)

## Development

```sh
bun install
```

### Browser (backend + frontend)

**NVIDIA (default):**
```sh
bun dev
```

**macOS (Apple Silicon):**
```sh
cd backend && uv run --no-group cuda12 python -m src.main &
cd frontend && bun run dev
```

**AMD (ROCm):**
```sh
cd backend && uv sync --no-group cuda12 --group rocm && uv run python -m src.main &
cd frontend && bun run dev
```

### Desktop app (Tauri)

```sh
bun tauri dev
```

## Build

```sh
bun tauri build
```

## Test Coverage

- A Windows operating system with NVIDIA is highly recommended for coverage reports.
- Run both suites and refresh both badges locally: `bun run coverage:update`
- Run only backend coverage and refresh the backend badge: `bun run coverage:backend`
- Run only frontend coverage and refresh the frontend badge: `bun run coverage:frontend`
- Refresh both badges from existing coverage reports: `bun run coverage:badges`
- CI workflow: `.github/workflows/coverage.yml`
- CI backend coverage skips `tests/conduit/dart_control` to keep GitHub Actions runtime down; the local coverage commands above still run the full backend suite.
- Pushes to `main` refresh both README badges automatically.
- PRs get an automatic sticky coverage comment with frontend and backend coverage.
