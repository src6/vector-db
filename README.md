# vector-db

[![CI](https://github.com/src6/vector-db/actions/workflows/ci.yml/badge.svg)](https://github.com/src6/vector-db/actions/workflows/ci.yml)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A focused, single-node HNSW vector search engine written in Rust. The project combines a reusable library, reproducible benchmark CLI, and versioned Axum API with pluggable storage, logical deletion, compaction, and JSON snapshots.

[Quickstart](#quickstart) · [Library usage](#library-usage) · [HTTP API](#http-api) · [OpenAPI specification](openapi.yaml) · [Architecture](#architecture) · [Current scope](#current-scope)

## Features

- HNSW insert/search with level sampling, beam search, and neighbor pruning.
- Distance metrics: squared Euclidean (L2) or cosine distance.
- In-memory storage backend, scalar quantized storage (i8 codes with per-dim min/max), and mmap-backed fixed-capacity storage.
- Concurrency wrapper: `ConcurrentIndex` uses `Arc<RwLock<...>>` for safe shared inserts/searches.
- Persistence: save/load the graph and Serde-compatible storage representation as JSON.
- Parallel search helpers with Rayon for batch queries.
- Segment model: `SegmentedIndex` combines immutable + mutable segments and supports flush.
- Logical deletion preserves graph connectivity; explicit compaction rebuilds active vectors and reports ID changes.
- Fallible construction, insert, and search APIs validate configuration, dimensions, and finite values.
- Versioned Axum HTTP API with single/batch insert and search, deletion, statistics, and compaction.
- CLI demo and seeded random-data mode with reproducible graph construction.
- CI checks formatting, Clippy with warnings denied, and the test suite on pushes and pull requests.

## Quickstart

The crate requires Rust 1.85 or newer.

```bash
# Run the small 2D demo.
cargo run -- demo

# Build and query a reproducible random dataset.
cargo run -- random --n 50 --dim 8 --k 5 --metric l2 --seed 42

# Measure build time, Rayon batch-query throughput, and exact recall@k.
cargo run --release -- benchmark --n 10000 --dim 128 --queries 100 --k 10 --seed 42

# Start a local API accepting 128-dimensional cosine vectors.
cargo run --release -- serve --dim 128 --metric cosine --port 3000
```

Both data generation and HNSW level sampling use `--seed`, so a fixed command builds the same graph. The `random` and `benchmark` commands also accept `--m`, `--m-max0`, `--ef-construction`, and `--ef-search`.

## Library usage

```rust
use vector_db::{HnswIndex, InMemoryStorage, Metric};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut index = HnswIndex::try_new(
        16,
        32,
        64,
        32,
        Metric::Cosine,
        InMemoryStorage::new(),
    )?
    .with_level_seed(42);

    index.try_insert(vec![1.0, 0.0])?;
    index.try_insert(vec![0.0, 1.0])?;

    let neighbors = index.try_search(&[0.9, 0.1], 1)?;
    println!("{neighbors:?}");
    Ok(())
}
```

The fallible API rejects invalid configuration, empty vectors, dimension mismatches, and non-finite values. `new`, `insert`, and `search` remain available as convenience methods that panic on invalid input.

## Architecture

```text
Library callers / CLI
├── Axum HTTP API                versioned JSON endpoints
├── HnswIndex                    graph construction and ANN search
│   └── VectorStorage
│       ├── InMemoryStorage      full-precision vectors in memory
│       ├── QuantizedStorage     per-dimension i8 scalar quantization
│       └── MmapStorage          fixed-capacity, mmap-backed vectors
├── ConcurrentIndex              Arc<RwLock<HnswIndex<_>>>
└── SegmentedIndex
    ├── immutable HnswIndex
    └── mutable HnswIndex
```

`HnswIndex` owns graph topology and delegates vector access to the `VectorStorage` trait. The concurrency and segmentation types compose indexes without changing the core search implementation.

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
- `serve`: Starts the HTTP service. The dimension, metric, seed, bind address, and HNSW parameters are fixed for the lifetime of the process.

## HTTP API

The server binds to `127.0.0.1:3000` by default, enforces an 8 MiB request limit, and moves index work off Tokio's async worker threads. The complete machine-readable contract is available in [`openapi.yaml`](openapi.yaml).

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/health` | Liveness check |
| `GET` | `/v1/stats` | Vector counts and effective index configuration |
| `POST` | `/v1/vectors` | Insert one vector |
| `POST` | `/v1/vectors/batch` | Atomically validate, then insert a batch |
| `DELETE` | `/v1/vectors/{id}` | Logically delete a point |
| `POST` | `/v1/search` | Search one query vector |
| `POST` | `/v1/search/batch` | Search a query batch with Rayon |
| `POST` | `/v1/maintenance/compact` | Rebuild without tombstones and return ID mappings |

```bash
# In a separate terminal, start a three-dimensional index for this example.
cargo run --release -- serve --dim 3 --metric cosine

curl -X POST http://127.0.0.1:3000/v1/vectors \
  -H 'content-type: application/json' \
  -d '{"vector":[0.1,0.2,0.3]}'

curl -X POST http://127.0.0.1:3000/v1/search \
  -H 'content-type: application/json' \
  -d '{"vector":[0.1,0.2,0.3],"k":10}'
```

Deletion uses tombstones: deleted nodes remain traversable so removing a bridge does not disconnect search, but they are excluded from results. Compaction physically removes them under the write lock. Because compaction assigns dense IDs, clients must consume the returned `old_id` → `new_id` mapping.

## Current scope

This is a focused single-node implementation, not a drop-in replacement for a distributed production vector database. In particular:

- `ConcurrentIndex` deliberately uses one coarse `RwLock`; searches can run together, while each insert holds the write lock and therefore serializes writers.
- JSON is intended for transparent, portable snapshots rather than compact or crash-atomic persistence. For `MmapStorage`, JSON stores the mapped file path and metadata, not the vector bytes; the original backing file must remain available.
- `MmapStorage` has a fixed capacity and currently panics when that capacity is exceeded through the infallible `VectorStorage::push` interface.
- Scalar quantization dequantizes vectors before distance evaluation. It reduces stored vector size but is not a SIMD-optimized quantized distance kernel.
- The HTTP service is in-memory and has no authentication; keep the default loopback binding unless it is placed behind an appropriate trusted gateway.
- There is no in-place vector update, write-ahead log, online/background compaction, or stable on-disk format guarantee.

These boundaries keep the implementation focused on the indexing, storage, concurrency, and evaluation mechanics rather than presenting it as a distributed database.

## Implementation map

| Area | Source |
| --- | --- |
| HNSW level sampling, beam search, neighbor pruning, and persistence | [`src/hnsw/mod.rs`](src/hnsw/mod.rs) |
| HTTP routing, validation, errors, and lifecycle tests | [`src/api.rs`](src/api.rs) |
| L2 and cosine distance | [`src/distance.rs`](src/distance.rs) |
| i8 scalar quantization | [`src/storage/quantized.rs`](src/storage/quantized.rs) |
| In-memory and mmap storage | [`src/storage`](src/storage) |
| Immutable/mutable segmentation | [`src/segment.rs`](src/segment.rs) |
| Rayon batch queries and coarse `RwLock` concurrency | [`src/concurrent.rs`](src/concurrent.rs) |
| Reproducible CLI datasets and benchmarks | [`src/main.rs`](src/main.rs) |
| OpenAPI 3.1 service contract | [`openapi.yaml`](openapi.yaml) |

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
npx --yes @redocly/cli@2.56.1 lint openapi.yaml
```

The same checks run in [GitHub Actions](.github/workflows/ci.yml). See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow. The project is available under the [MIT License](LICENSE).
