use clap::{Parser, Subcommand, ValueEnum};
use rand::{Rng, SeedableRng, rngs::StdRng};

use vector_db::{HnswIndex, InMemoryStorage, Metric};

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
}

#[derive(Parser)]
struct RandomArgs {
    /// Number of points to insert
    #[arg(long, default_value_t = 50)]
    n: usize,
    /// Dimension of each point
    #[arg(long, default_value_t = 8)]
    dim: usize,
    /// Number of neighbors to return
    #[arg(long, default_value_t = 5)]
    k: usize,
    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Metric: l2 or cosine
    #[arg(long, default_value = "l2", value_enum)]
    metric: MetricArg,
}

#[derive(Clone, Debug, ValueEnum)]
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
    let metric = args.metric.clone();
    let mut index = HnswIndex::new(
        16,
        32,
        64,
        64,
        metric.clone().into(),
        InMemoryStorage::new(),
    );

    for _ in 0..args.n {
        let vec: Vec<f32> = (0..args.dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        index.insert(vec);
    }

    let query: Vec<f32> = (0..args.dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let neighbors = index.search(&query, args.k);
    println!(
        "Inserted: {} points, dim: {}, metric: {:?}\nQuery: {:?}\nNeighbors: {:?}",
        args.n, args.dim, metric, query, neighbors
    );
}
