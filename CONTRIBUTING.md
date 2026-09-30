# Contributing

Contributions are welcome. Keep changes focused on the single-node index, storage backends, API, or evaluation tooling, and include tests for observable behavior.

## Development setup

The project requires Rust 1.85 or newer. Clone the repository, then run:

```bash
cargo test
```

Before opening a pull request, run the same checks as CI:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
npx --yes @redocly/cli@2.56.1 lint openapi.yaml
```

## Pull requests

- Explain the behavior being changed and why.
- Add or update tests for correctness-sensitive changes.
- Update `README.md` and `openapi.yaml` when public commands or HTTP behavior change.
- Avoid unrelated formatting or dependency changes.
- Call out compatibility changes to persisted JSON or point-ID behavior.

## Design constraints

- Deleted HNSW nodes remain traversable until compaction.
- Compaction may reassign point IDs and must return a complete old-to-new mapping.
- The HTTP service defaults to loopback and intentionally has no built-in authentication.
- Existing storage implementations use an infallible `push` trait method; do not hide new failure modes behind panics.
