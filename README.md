# vector-db

HNSW-based vector index in Rust with configurable distance metric (L2 or cosine), an in-memory storage backend, optional scalar-quantized storage, coarse-grained concurrency wrapper, and basic JSON persistence.

## Features
- HNSW insert/search with level sampling, beam search, and neighbor pruning.
- Distance metrics: L2 or cosine.
- In-memory storage backend plus scalar quantized storage (i8 codes with per-dim min/max).
- Concurrency wrapper: `ConcurrentIndex` uses `Arc<RwLock<...>>` for safe shared inserts/searches.
- Persistence: save/load the full index as JSON (graph + storage).
- Simple CLI demo.

## Quickstart
- Build/tests: `cargo test`
- Demo: `cargo run -- demo`
- Random dataset: `cargo run -- random --n 50 --dim 8 --k 5 --metric l2 --seed 42`
  - Metrics: `l2` or `cosine`

## Concurrency, Persistence, Quantization
- Concurrency: wrap any `HnswIndex` in `ConcurrentIndex::new(index)` to share across threads (coarse-grained `RwLock` guard).
- Persistence: call `save_to_json(path)` / `load_from_json(path)` on the index to round-trip the graph + storage.
- Quantization: build a `ScalarQuantizerConfig` from a sample dataset and initialize `QuantizedStorage`, then construct `HnswIndex` with it for i8-coded vectors.

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
