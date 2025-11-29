# vector-db

Single-threaded HNSW-based vector index in Rust with configurable distance metric (L2 or cosine) and an in-memory storage backend.

## Features
- HNSW insert/search with level sampling, beam search, and neighbor pruning.
- Distance metrics: L2 or cosine.
- In-memory storage backend; ready for extension to mmap/persistence later.
- Simple CLI demo.

## Quickstart
- Build/tests: `cargo test`
- Demo: `cargo run -- demo`
- Random dataset: `cargo run -- random --n 50 --dim 8 --k 5 --metric l2 --seed 42`
  - Metrics: `l2` or `cosine`

## CLI
- `demo`: Inserts a small 2D set and prints neighbors for a fixed query.
- `random`: Inserts `n` random points of dimension `dim`, runs a random query, and prints the top `k` neighbors. Parameters: `--n`, `--dim`, `--k`, `--seed`, `--metric l2|cosine`.

## API
The main types are exported from the crate root:
- `HnswIndex` — index with configurable `m`, `m_max0`, `ef_construction`, `ef_search`, and `metric`.
- `Metric` — choose `L2` or `Cosine`.
- `InMemoryStorage`, `VectorStorage` — storage abstractions.

## Notes
- Currently single-threaded and in-memory. Concurrency, persistence, and quantization can be layered on later.
