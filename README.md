# vector-db

[![CI](https://github.com/src6/vector-db/actions/workflows/ci.yml/badge.svg)](https://github.com/src6/vector-db/actions/workflows/ci.yml)

A compact HNSW vector-index library and CLI written in Rust. It supports configurable distance metrics, pluggable storage, deterministic construction, coarse-grained concurrency, and JSON persistence.

## Features

- HNSW insert/search with level sampling, beam search, and neighbor pruning.
- Distance metrics: squared Euclidean (L2) or cosine distance.
- In-memory storage backend, scalar quantized storage (i8 codes with per-dim min/max), and mmap-backed fixed-capacity storage.
- Concurrency wrapper: `ConcurrentIndex` uses `Arc<RwLock<...>>` for safe shared inserts/searches.
- Persistence: save/load the graph and Serde-compatible storage representation as JSON.
- Parallel search helpers with Rayon for batch queries.
- Segment model: `SegmentedIndex` combines immutable + mutable segments and supports flush.
- Fallible construction, insert, and search APIs validate configuration, dimensions, and finite values.
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

## Current scope

This is a compact learning implementation, not a drop-in replacement for a distributed production vector database. In particular:

- `ConcurrentIndex` deliberately uses one coarse `RwLock`; searches can run together, while each insert holds the write lock and therefore serializes writers.
- JSON is intended for transparent, portable snapshots rather than compact or crash-atomic persistence. For `MmapStorage`, JSON stores the mapped file path and metadata, not the vector bytes; the original backing file must remain available.
- `MmapStorage` has a fixed capacity and currently panics when that capacity is exceeded through the infallible `VectorStorage::push` interface.
- Scalar quantization dequantizes vectors before distance evaluation. It reduces stored vector size but is not a SIMD-optimized quantized distance kernel.
- There is no delete/update path, write-ahead log, online compaction, network service, authentication, or stable on-disk format guarantee.

These boundaries keep the implementation focused on the indexing, storage, concurrency, and evaluation mechanics rather than presenting it as a distributed database.

## Implementation map

| Area | Source |
| --- | --- |
| HNSW level sampling, beam search, neighbor pruning, and persistence | [`src/hnsw/mod.rs`](src/hnsw/mod.rs) |
| L2 and cosine distance | [`src/distance.rs`](src/distance.rs) |
| i8 scalar quantization | [`src/storage/quantized.rs`](src/storage/quantized.rs) |
| In-memory and mmap storage | [`src/storage`](src/storage) |
| Immutable/mutable segmentation | [`src/segment.rs`](src/segment.rs) |
| Rayon batch queries and coarse `RwLock` concurrency | [`src/concurrent.rs`](src/concurrent.rs) |
| Reproducible CLI datasets and benchmarks | [`src/main.rs`](src/main.rs) |

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

The same checks run in [GitHub Actions](.github/workflows/ci.yml). The project is available under the [MIT License](LICENSE).
