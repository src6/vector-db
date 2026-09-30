# vector-db

HNSW-based vector index in Rust with configurable distance metrics, pluggable storage, deterministic construction, coarse-grained concurrency, and JSON persistence.

## Features
- HNSW insert/search with level sampling, beam search, and neighbor pruning.
- Distance metrics: L2 or cosine.
- In-memory storage backend, scalar quantized storage (i8 codes with per-dim min/max), and mmap-backed fixed-capacity storage.
- Concurrency wrapper: `ConcurrentIndex` uses `Arc<RwLock<...>>` for safe shared inserts/searches.
- Persistence: save/load the graph and Serde-compatible storage representation as JSON.
- Parallel search helpers with Rayon for batch queries.
- Segment model: `SegmentedIndex` combines immutable + mutable segments and supports flush.
- Fallible construction, insert, and search APIs validate configuration, dimensions, and finite values.
- CLI demo and seeded random-data mode with reproducible graph construction.
- CI checks formatting, Clippy with warnings denied, and the test suite on pushes and pull requests.

## Quickstart
- Build/tests: `cargo test`
- Demo: `cargo run -- demo`
- Random dataset: `cargo run -- random --n 50 --dim 8 --k 5 --metric l2 --seed 42`
  - Metrics: `l2` or `cosine`
  - Graph tuning: `--m`, `--m-max0`, `--ef-construction`, and `--ef-search`
  - The seed controls both generated data and HNSW level sampling, so a fixed command builds the same graph.
- Benchmark: `cargo run --release -- benchmark --n 10000 --dim 128 --queries 100 --k 10 --seed 42`
  - Reports index build time, Rayon batch-query throughput, and recall@k against an exact brute-force search.
- Production checks: `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`

## Concurrency, Persistence, Quantization
- Concurrency: wrap any `HnswIndex` in `ConcurrentIndex::new(index)` to share across threads (coarse-grained `RwLock` guard).
- Persistence: call `save_to_json(path)` / `load_from_json(path)` on an index with a Serde-compatible backend to round-trip its graph and storage representation.
- Quantization: build a `ScalarQuantizerConfig` from a sample dataset and initialize `QuantizedStorage`, then construct `HnswIndex` with it for i8-coded vectors.
- Out-of-core: use `MmapStorage::create(path, dim, capacity)` to store vectors in a memory-mapped file with fixed capacity.
- Segments: use `SegmentedIndex` to pair an immutable segment with a mutable one and flush updates when desired.
- Parallel queries: use `search_batch_parallel` (on `HnswIndex` or `ConcurrentIndex`) to fan out queries via Rayon.

## CLI
- `demo`: Inserts a small 2D set and prints neighbors for a fixed query.
- `random`: Inserts `n` random points of dimension `dim`, runs a random query, and prints the top `k` neighbors and effective graph configuration. Parameters: `--n`, `--dim`, `--k`, `--seed`, `--metric l2|cosine`, `--m`, `--m-max0`, `--ef-construction`, and `--ef-search`.
- `benchmark`: Uses the same data and graph parameters plus `--queries`; run it with `--release` for meaningful throughput measurements.

## API
The main types are exported from the crate root:
- `HnswIndex` — index with configurable `m`, `m_max0`, `ef_construction`, `ef_search`, and `metric`.
- `Metric` — choose `L2` or `Cosine`.
- `InMemoryStorage`, `VectorStorage` — storage abstractions.
- `try_new`, `try_insert`, and `try_search` — validated APIs for caller-controlled error handling. The original convenience methods remain available and panic on invalid input.

## Scope and production limitations

This is a compact learning implementation, not a drop-in replacement for a distributed production vector database. In particular:

- `ConcurrentIndex` deliberately uses one coarse `RwLock`; searches can run together, while each insert holds the write lock and therefore serializes writers.
- JSON is intended for transparent, portable snapshots rather than compact or crash-atomic persistence. For `MmapStorage`, JSON stores the mapped file path and metadata, not the vector bytes; the original backing file must remain available.
- `MmapStorage` has a fixed capacity and currently panics when that capacity is exceeded through the infallible `VectorStorage::push` interface.
- Scalar quantization dequantizes vectors before distance evaluation. It reduces stored vector size but is not a SIMD-optimized quantized distance kernel.
- There is no delete/update path, write-ahead log, online compaction, network service, authentication, or stable on-disk format guarantee.

These boundaries are intentional and make the implemented claims independently verifiable without presenting the crate as production-ready.

## Resume claim map

| Claim | Repository evidence |
| --- | --- |
| HNSW level sampling, beam search, neighbor pruning, L2/cosine | `src/hnsw/mod.rs`, `src/distance.rs` |
| Optional i8 scalar quantization | `src/storage/quantized.rs` |
| Segmented storage | `src/segment.rs` |
| JSON persistence | `HnswIndex::save_to_json` / `load_from_json` and the persistence round-trip test |
| Rayon batch queries and coarse `RwLock` concurrency | `src/concurrent.rs` and `HnswIndex::search_batch_parallel` |
| Exposed graph/search tuning and reproducible benchmark runs | `benchmark` CLI flags plus `HnswIndex::with_level_seed` and its reproducibility test |
| CLI demo and seeded random dataset | `src/main.rs` |
