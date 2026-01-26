//! Trading example: Fair Federated Learning with Bybit data.
//!
//! Fetches BTCUSDT klines from Bybit, distributes data among clients with
//! different quality levels, runs FL with multiple aggregation strategies,
//! and compares fairness metrics.

use fair_fl_trading::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn run_experiment(
    strategy_name: &str,
    strategy: AggregationStrategy,
    base_features: &ndarray::Array2<f64>,
    base_labels: &ndarray::Array1<f64>,
    rng_seed: u64,
) -> FairnessMetrics {
    let mut rng = StdRng::seed_from_u64(rng_seed);

    // Create four clients with different data quality, simulating trading desks
    let desk_configs = [
        ("Desk_A (BTC spot)", DataQuality::High),
        ("Desk_B (Altcoins)", DataQuality::Medium),
        ("Desk_C (Emerging)", DataQuality::Low),
        ("Desk_D (Cross-exch)", DataQuality::Medium),
    ];

    let mut clients: Vec<FLClient> = desk_configs
        .iter()
        .map(|(name, quality)| {
            let (noisy_feat, noisy_lab) =
                degrade_data(base_features, base_labels, *quality, &mut rng);
            FLClient::new(name, *quality, noisy_feat, noisy_lab)
        })
        .collect();

    let num_features = base_features.ncols();
    let mut server = FairFLServer::new(num_features, strategy);

    // Train for 20 rounds
    let rounds = 20;
    let local_epochs = 5;
    let lr = 0.005;

    let all_metrics = server.train(&mut clients, rounds, local_epochs, lr);

    println!("\n=== {} ===", strategy_name);
    println!("Round-by-round average loss:");
    for (i, m) in all_metrics.iter().enumerate() {
        if i % 5 == 0 || i == rounds - 1 {
            println!("  Round {:2}: avg={:.6}  worst={:.6}  gini={:.4}",
                i + 1, m.average_loss, m.worst_loss, m.gini_coefficient);
        }
    }

    let final_metrics = all_metrics.last().unwrap().clone();
    println!("\nFinal fairness metrics:");
    print!("{}", final_metrics);

    // Show aggregation weights at the end
    let agg_weights = server.compute_aggregation_weights(&clients);
    println!("  Aggregation weights:");
    for (client, w) in clients.iter().zip(agg_weights.iter()) {
        println!("    {}: {:.4} (quality: {})", client.name, w, client.quality.label());
    }

    final_metrics
}

fn main() -> anyhow::Result<()> {
    println!("Fair Federated Learning for Trading");
    println!("====================================");
    println!();

    // ---- Fetch Bybit data ----
    println!("Fetching BTCUSDT 1h klines from Bybit...");
    let bars = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(b) => {
            println!("Fetched {} bars from Bybit.", b.len());
            b
        }
        Err(e) => {
            println!("Warning: Could not fetch Bybit data ({}). Using synthetic data.", e);
            Vec::new()
        }
    };

    // ---- Compute features ----
    let (features, labels) = if bars.len() >= 10 {
        let (f, l) = compute_features(&bars);
        println!("Computed {} samples with {} features from market data.\n",
            f.nrows(), f.ncols());
        (f, l)
    } else {
        println!("Using synthetic data (100 samples, 5 features).\n");
        let mut rng = StdRng::seed_from_u64(0);
        generate_synthetic_data(100, 5, &mut rng)
    };

    if features.nrows() < 5 {
        println!("Not enough data to run experiment.");
        return Ok(());
    }

    let seed = 42u64;

    // ---- Run experiments with different strategies ----
    let m_fedavg = run_experiment(
        "Standard FedAvg",
        AggregationStrategy::StandardFedAvg,
        &features,
        &labels,
        seed,
    );

    let m_q1 = run_experiment(
        "q-FedAvg (q=1)",
        AggregationStrategy::QFedAvg { q: 1.0 },
        &features,
        &labels,
        seed,
    );

    let m_q3 = run_experiment(
        "q-FedAvg (q=3)",
        AggregationStrategy::QFedAvg { q: 3.0 },
        &features,
        &labels,
        seed,
    );

    let m_contrib = run_experiment(
        "Contribution-Based",
        AggregationStrategy::ContributionBased,
        &features,
        &labels,
        seed,
    );

    // ---- Summary comparison ----
    println!("\n\n======================================");
    println!("            SUMMARY COMPARISON");
    println!("======================================");
    println!(
        "{:<22} {:>10} {:>10} {:>10} {:>10}",
        "Metric", "FedAvg", "q=1", "q=3", "Contrib"
    );
    println!("{}", "-".repeat(62));
    println!(
        "{:<22} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
        "Avg Loss", m_fedavg.average_loss, m_q1.average_loss, m_q3.average_loss, m_contrib.average_loss
    );
    println!(
        "{:<22} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
        "Worst-case Loss", m_fedavg.worst_loss, m_q1.worst_loss, m_q3.worst_loss, m_contrib.worst_loss
    );
    println!(
        "{:<22} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
        "Gini Coefficient", m_fedavg.gini_coefficient, m_q1.gini_coefficient, m_q3.gini_coefficient, m_contrib.gini_coefficient
    );
    println!(
        "{:<22} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
        "Min/Max Ratio", m_fedavg.min_max_ratio, m_q1.min_max_ratio, m_q3.min_max_ratio, m_contrib.min_max_ratio
    );
    println!(
        "{:<22} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
        "Loss Variance", m_fedavg.loss_variance, m_q1.loss_variance, m_q3.loss_variance, m_contrib.loss_variance
    );

    println!("\nDone.");
    Ok(())
}
