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

```sh
bun dev
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

- Run both suites and refresh both badges locally: `bun run coverage:update`
- Run only backend coverage and refresh the backend badge: `bun run coverage:backend`
- Run only frontend coverage and refresh the frontend badge: `bun run coverage:frontend`
- Refresh both badges from existing coverage reports: `bun run coverage:badges`
- CI workflow: `.github/workflows/coverage.yml`
- Pushes to `main` refresh both README badges automatically.
- PRs get an automatic sticky coverage comment with frontend and backend coverage.
