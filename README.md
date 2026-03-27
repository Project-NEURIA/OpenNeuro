# OpenNeuro
[![Backend Coverage](docs/backend-coverage.svg)](./docs/backend-coverage.svg)

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

## Backend Test Coverage

- Run locally: `cd backend && uv run pytest`
- CI workflow: `.github/workflows/backend_coverage.yml`
- PRs get an automatic sticky coverage comment.
