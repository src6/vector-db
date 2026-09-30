use std::collections::HashSet;
use std::time::Instant;

use clap::{Args, Parser, Subcommand, ValueEnum};
use rand::{Rng, SeedableRng, rngs::StdRng};

use vector_db::{HnswIndex, InMemoryStorage, Metric, cosine_distance, l2};

#[derive(Parser)]
#[command(name = "vector-db", about = "Demo CLI for the HNSW index")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the simple 2D demo
    Demo,
    /// Build a random dataset and query it
    Random(RandomArgs),
    /// Measure build time, parallel search throughput, and recall
    Benchmark(BenchmarkArgs),
}

#[derive(Args)]
struct RandomArgs {
    /// Number of points to insert
    #[arg(long, default_value_t = 50, value_parser = parse_positive_usize)]
    n: usize,
    /// Dimension of each point
    #[arg(long, default_value_t = 8, value_parser = parse_positive_usize)]
    dim: usize,
    /// Number of neighbors to return
    #[arg(long, default_value_t = 5, value_parser = parse_positive_usize)]
    k: usize,
    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Metric: l2 or cosine
    #[arg(long, default_value = "l2", value_enum)]
    metric: MetricArg,
    /// Maximum neighbors per upper graph layer
    #[arg(long, default_value_t = 16, value_parser = parse_positive_usize)]
    m: usize,
    /// Maximum neighbors at the base graph layer
    #[arg(long, default_value_t = 32, value_parser = parse_positive_usize)]
    m_max0: usize,
    /// Candidate-list size used while constructing the graph
    #[arg(long, default_value_t = 64, value_parser = parse_positive_usize)]
    ef_construction: usize,
    /// Candidate-list size used while searching the graph
    #[arg(long, default_value_t = 64, value_parser = parse_positive_usize)]
    ef_search: usize,
}

#[derive(Args)]
struct BenchmarkArgs {
    #[command(flatten)]
    index: RandomArgs,
    /// Number of queries in the measured parallel batch
    #[arg(long, default_value_t = 100, value_parser = parse_positive_usize)]
    queries: usize,
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("expected a positive integer, got {value:?}"))?;
    if parsed == 0 {
        return Err("value must be greater than zero".to_owned());
    }
    Ok(parsed)
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MetricArg {
    L2,
    Cosine,
}

impl From<MetricArg> for Metric {
    fn from(m: MetricArg) -> Self {
        match m {
            MetricArg::L2 => Metric::L2,
            MetricArg::Cosine => Metric::Cosine,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Demo => run_demo(),
        Commands::Random(args) => run_random(args),
        Commands::Benchmark(args) => run_benchmark(args),
    }
}

fn run_demo() {
    let mut index = HnswIndex::new(8, 16, 32, 32, Metric::L2, InMemoryStorage::new());

    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
    ];
    for p in points {
        index.insert(p);
    }

    let query = [0.9, 0.1];
    let neighbors = index.search(&query, 3);
    println!("Query: {:?}\nNeighbors: {:?}", query, neighbors);
}

fn run_random(args: RandomArgs) {
    let mut rng = StdRng::seed_from_u64(args.seed);
    let metric = args.metric;
    let mut index = HnswIndex::try_new(
        args.m,
        args.m_max0,
        args.ef_construction,
        args.ef_search,
        metric.into(),
        InMemoryStorage::new(),
    )
    .unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(2);
    })
    .with_level_seed(args.seed);

    for _ in 0..args.n {
        let vec: Vec<f32> = (0..args.dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        index.insert(vec);
    }

    let query: Vec<f32> = (0..args.dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let neighbors = index.search(&query, args.k);
    println!(
        "Inserted: {} points, dim: {}, metric: {:?}, seed: {}\nHNSW: m={}, m_max0={}, ef_construction={}, ef_search={}\nQuery: {:?}\nNeighbors: {:?}",
        args.n,
        args.dim,
        metric,
        args.seed,
        args.m,
        args.m_max0,
        args.ef_construction,
        args.ef_search,
        query,
        neighbors
    );
}

fn run_benchmark(args: BenchmarkArgs) {
    let config = args.index;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let metric: Metric = config.metric.into();
    let data: Vec<Vec<f32>> = (0..config.n)
        .map(|_| (0..config.dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..args.queries)
        .map(|_| (0..config.dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    let mut index = HnswIndex::try_new(
        config.m,
        config.m_max0,
        config.ef_construction,
        config.ef_search,
        metric,
        InMemoryStorage::new(),
    )
    .expect("CLI validates HNSW parameters")
    .with_level_seed(config.seed);

    let build_started = Instant::now();
    for vector in data.iter().cloned() {
        index.insert(vector);
    }
    let build_elapsed = build_started.elapsed();

    // Warm the search path before the measured batch.
    let _ = index.search(&queries[0], config.k);
    let search_started = Instant::now();
    let approximate = index.search_batch_parallel(queries.clone(), config.k);
    let search_elapsed = search_started.elapsed();

    let expected_per_query = config.k.min(config.n);
    let matched = queries
        .iter()
        .zip(&approximate)
        .map(|(query, actual)| {
            let mut exact: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(id, vector)| {
                    let distance = match metric {
                        Metric::L2 => l2(query, vector),
                        Metric::Cosine => cosine_distance(query, vector),
                    };
                    (id, distance)
                })
                .collect();
            exact.sort_by(|a, b| a.1.total_cmp(&b.1));
            let expected: HashSet<_> = exact
                .into_iter()
                .take(expected_per_query)
                .map(|(id, _)| id)
                .collect();
            actual
                .iter()
                .filter(|neighbor| expected.contains(&neighbor.id))
                .count()
        })
        .sum::<usize>();

    let total_expected = args.queries * expected_per_query;
    let recall = matched as f64 / total_expected as f64;
    let queries_per_second = args.queries as f64 / search_elapsed.as_secs_f64();
    println!(
        "Dataset: n={}, dim={}, queries={}, k={}, metric={:?}, seed={}\nHNSW: m={}, m_max0={}, ef_construction={}, ef_search={}\nBuild: {:.3}s\nParallel search: {:.3}s ({:.0} queries/s)\nRecall@{}: {:.2}%",
        config.n,
        config.dim,
        args.queries,
        config.k,
        metric,
        config.seed,
        config.m,
        config.m_max0,
        config.ef_construction,
        config.ef_search,
        build_elapsed.as_secs_f64(),
        search_elapsed.as_secs_f64(),
        queries_per_second,
        expected_per_query,
        recall * 100.0,
    );
}
