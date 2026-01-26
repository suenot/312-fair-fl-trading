//! Fair Federated Learning for Trading
//!
//! Implements q-FedAvg, contribution-based scoring, and fair aggregation
//! for collaborative trading model training with heterogeneous data quality.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Bybit data fetching
// ---------------------------------------------------------------------------

/// Raw kline entry from Bybit V5 API.
/// Fields: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
#[derive(Debug, Deserialize)]
pub struct BybitKlineEntry(
    pub String, // startTime
    pub String, // open
    pub String, // high
    pub String, // low
    pub String, // close
    pub String, // volume
    pub String, // turnover
);

#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub list: Vec<BybitKlineEntry>,
}

#[derive(Debug, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

/// Parsed OHLCV bar.
#[derive(Debug, Clone)]
pub struct OhlcvBar {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline data from Bybit V5 public API (blocking).
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<OhlcvBar>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let resp: BybitKlineResponse = reqwest::blocking::get(&url)
        .context("Failed to call Bybit API")?
        .json()
        .context("Failed to parse Bybit response")?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut bars: Vec<OhlcvBar> = resp
        .result
        .list
        .iter()
        .map(|e| {
            Ok(OhlcvBar {
                timestamp: e.0.parse().unwrap_or(0),
                open: e.1.parse().unwrap_or(0.0),
                high: e.2.parse().unwrap_or(0.0),
                low: e.3.parse().unwrap_or(0.0),
                close: e.4.parse().unwrap_or(0.0),
                volume: e.5.parse().unwrap_or(0.0),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Bybit returns newest first; reverse to chronological order.
    bars.reverse();
    Ok(bars)
}

// ---------------------------------------------------------------------------
// Feature engineering
// ---------------------------------------------------------------------------

/// Compute features and labels from OHLCV bars.
///
/// Features per bar (using look-back of 5):
///   0: return
///   1: rolling volatility (5-period std of returns)
///   2: volume ratio (current / 5-period mean)
///   3: price range (high-low)/close
///   4: momentum (5-period return)
///
/// Label: next-period return.
pub fn compute_features(bars: &[OhlcvBar]) -> (Array2<f64>, Array1<f64>) {
    let lookback = 5usize;
    let n = bars.len();
    if n < lookback + 2 {
        return (Array2::zeros((0, 5)), Array1::zeros(0));
    }

    let returns: Vec<f64> = (1..n)
        .map(|i| (bars[i].close - bars[i - 1].close) / bars[i - 1].close)
        .collect();

    let usable = n - lookback - 1; // need lookback history + 1 for label
    let num_features = 5;
    let mut features = Array2::<f64>::zeros((usable, num_features));
    let mut labels = Array1::<f64>::zeros(usable);

    for i in 0..usable {
        let idx = i + lookback; // index into returns (0-based, returns[0] corresponds to bar 1)
        let bar_idx = idx + 1; // index into bars

        // Feature 0: current return
        features[[i, 0]] = returns[idx - 1];

        // Feature 1: rolling volatility
        let window: Vec<f64> = returns[idx - lookback..idx].to_vec();
        let mean_r: f64 = window.iter().sum::<f64>() / lookback as f64;
        let var: f64 = window.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / lookback as f64;
        features[[i, 1]] = var.sqrt();

        // Feature 2: volume ratio
        let vol_mean: f64 = (0..lookback)
            .map(|j| bars[bar_idx - lookback + j].volume)
            .sum::<f64>()
            / lookback as f64;
        features[[i, 2]] = if vol_mean > 0.0 {
            bars[bar_idx].volume / vol_mean
        } else {
            1.0
        };

        // Feature 3: price range
        features[[i, 3]] = if bars[bar_idx].close > 0.0 {
            (bars[bar_idx].high - bars[bar_idx].low) / bars[bar_idx].close
        } else {
            0.0
        };

        // Feature 4: momentum
        features[[i, 4]] = if bars[bar_idx - lookback].close > 0.0 {
            (bars[bar_idx].close - bars[bar_idx - lookback].close)
                / bars[bar_idx - lookback].close
        } else {
            0.0
        };

        // Label: next-period return
        if idx < returns.len() {
            labels[i] = returns[idx];
        }
    }

    (features, labels)
}

// ---------------------------------------------------------------------------
// Data quality simulation
// ---------------------------------------------------------------------------

/// Data quality levels for FL clients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataQuality {
    High,
    Medium,
    Low,
}

impl DataQuality {
    /// Noise standard deviation associated with this quality level.
    pub fn noise_sigma(&self) -> f64 {
        match self {
            DataQuality::High => 0.01,
            DataQuality::Medium => 0.05,
            DataQuality::Low => 0.15,
        }
    }

    /// Fraction of values to randomly zero-out (simulating missing data).
    pub fn missing_rate(&self) -> f64 {
        match self {
            DataQuality::High => 0.0,
            DataQuality::Medium => 0.05,
            DataQuality::Low => 0.20,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            DataQuality::High => "High",
            DataQuality::Medium => "Medium",
            DataQuality::Low => "Low",
        }
    }
}

/// Add noise and simulate missing data according to quality level.
pub fn degrade_data(
    features: &Array2<f64>,
    labels: &Array1<f64>,
    quality: DataQuality,
    rng: &mut impl Rng,
) -> (Array2<f64>, Array1<f64>) {
    let sigma = quality.noise_sigma();
    let missing = quality.missing_rate();
    let (nrows, ncols) = features.dim();

    let mut noisy_features = features.clone();
    let mut noisy_labels = labels.clone();

    for i in 0..nrows {
        for j in 0..ncols {
            // Add Gaussian noise
            let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0; // uniform approx
            noisy_features[[i, j]] += noise * sigma;
            // Simulate missing data
            if rng.gen::<f64>() < missing {
                noisy_features[[i, j]] = 0.0;
            }
        }
        let label_noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
        noisy_labels[i] += label_noise * sigma;
    }

    (noisy_features, noisy_labels)
}

// ---------------------------------------------------------------------------
// Linear model utilities
// ---------------------------------------------------------------------------

/// Predict y = X * w  (no bias for simplicity).
pub fn predict(features: &Array2<f64>, weights: &Array1<f64>) -> Array1<f64> {
    features.dot(weights)
}

/// Mean squared error.
pub fn mse_loss(predictions: &Array1<f64>, labels: &Array1<f64>) -> f64 {
    let n = predictions.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let diff = predictions - labels;
    diff.mapv(|v| v * v).sum() / n
}

/// Gradient of MSE w.r.t. weights: (2/n) X^T (X w - y).
pub fn mse_gradient(
    features: &Array2<f64>,
    labels: &Array1<f64>,
    weights: &Array1<f64>,
) -> Array1<f64> {
    let n = features.nrows() as f64;
    if n == 0.0 {
        return Array1::zeros(weights.len());
    }
    let preds = features.dot(weights);
    let diff = &preds - labels;
    let grad = features.t().dot(&diff) * (2.0 / n);
    grad
}

// ---------------------------------------------------------------------------
// FL Client
// ---------------------------------------------------------------------------

/// A federated learning client (trading desk).
#[derive(Debug, Clone)]
pub struct FLClient {
    pub name: String,
    pub quality: DataQuality,
    pub features: Array2<f64>,
    pub labels: Array1<f64>,
    pub weights: Array1<f64>,
    pub local_loss: f64,
    pub data_size: usize,
}

impl FLClient {
    pub fn new(
        name: &str,
        quality: DataQuality,
        features: Array2<f64>,
        labels: Array1<f64>,
    ) -> Self {
        let ncols = features.ncols();
        let data_size = features.nrows();
        Self {
            name: name.to_string(),
            quality,
            features,
            labels,
            weights: Array1::zeros(ncols),
            local_loss: f64::MAX,
            data_size,
        }
    }

    /// Receive global model weights.
    pub fn receive_global_model(&mut self, global_weights: &Array1<f64>) {
        self.weights = global_weights.clone();
    }

    /// Perform local SGD for the given number of epochs with learning rate.
    pub fn local_train(&mut self, epochs: usize, lr: f64) {
        for _ in 0..epochs {
            let grad = mse_gradient(&self.features, &self.labels, &self.weights);
            self.weights = &self.weights - &(grad * lr);
        }
        let preds = predict(&self.features, &self.weights);
        self.local_loss = mse_loss(&preds, &self.labels);
    }
}

// ---------------------------------------------------------------------------
// Aggregation strategies
// ---------------------------------------------------------------------------

/// Aggregation strategy for the FL server.
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    /// Standard data-size-weighted FedAvg.
    StandardFedAvg,
    /// q-FedAvg with fairness parameter q.
    QFedAvg { q: f64 },
    /// Contribution-based weighting (leave-one-out approximation).
    ContributionBased,
}

// ---------------------------------------------------------------------------
// Fairness metrics
// ---------------------------------------------------------------------------

/// Fairness evaluation metrics across FL clients.
#[derive(Debug, Clone)]
pub struct FairnessMetrics {
    pub per_client_loss: Vec<(String, f64)>,
    pub average_loss: f64,
    pub worst_loss: f64,
    pub best_loss: f64,
    pub loss_variance: f64,
    pub min_max_ratio: f64,
    pub gini_coefficient: f64,
}

impl std::fmt::Display for FairnessMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Average loss:      {:.6}", self.average_loss)?;
        writeln!(f, "  Worst-case loss:   {:.6}", self.worst_loss)?;
        writeln!(f, "  Best-case loss:    {:.6}", self.best_loss)?;
        writeln!(f, "  Loss variance:     {:.6}", self.loss_variance)?;
        writeln!(f, "  Min/Max ratio:     {:.4}", self.min_max_ratio)?;
        writeln!(f, "  Gini coefficient:  {:.4}", self.gini_coefficient)?;
        for (name, loss) in &self.per_client_loss {
            writeln!(f, "    {}: {:.6}", name, loss)?;
        }
        Ok(())
    }
}

/// Compute fairness metrics from per-client losses.
pub fn compute_fairness_metrics(client_losses: &[(String, f64)]) -> FairnessMetrics {
    let losses: Vec<f64> = client_losses.iter().map(|(_, l)| *l).collect();
    let n = losses.len() as f64;
    let avg = losses.iter().sum::<f64>() / n;
    let worst = losses.iter().cloned().fold(f64::MIN, f64::max);
    let best = losses.iter().cloned().fold(f64::MAX, f64::min);
    let variance = losses.iter().map(|l| (l - avg).powi(2)).sum::<f64>() / n;
    let min_max = if worst > 0.0 { best / worst } else { 0.0 };

    // Gini coefficient
    let mut sorted = losses.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total: f64 = sorted.iter().sum();
    let gini = if total > 0.0 {
        let k = sorted.len();
        let mut numerator = 0.0;
        for (i, &val) in sorted.iter().enumerate() {
            numerator += (2.0 * (i as f64 + 1.0) - k as f64 - 1.0) * val;
        }
        numerator / (k as f64 * total)
    } else {
        0.0
    };

    FairnessMetrics {
        per_client_loss: client_losses.to_vec(),
        average_loss: avg,
        worst_loss: worst,
        best_loss: best,
        loss_variance: variance,
        min_max_ratio: min_max,
        gini_coefficient: gini,
    }
}

// ---------------------------------------------------------------------------
// FL Server
// ---------------------------------------------------------------------------

/// Fair Federated Learning server.
pub struct FairFLServer {
    pub global_weights: Array1<f64>,
    pub strategy: AggregationStrategy,
    pub num_features: usize,
}

impl FairFLServer {
    pub fn new(num_features: usize, strategy: AggregationStrategy) -> Self {
        Self {
            global_weights: Array1::zeros(num_features),
            strategy,
            num_features,
        }
    }

    /// Compute aggregation weights for clients based on the current strategy.
    pub fn compute_aggregation_weights(&self, clients: &[FLClient]) -> Vec<f64> {
        let k = clients.len();
        match &self.strategy {
            AggregationStrategy::StandardFedAvg => {
                let total: f64 = clients.iter().map(|c| c.data_size as f64).sum();
                clients
                    .iter()
                    .map(|c| c.data_size as f64 / total)
                    .collect()
            }
            AggregationStrategy::QFedAvg { q } => {
                // Weight each client by p_k * F_k^q
                let total_data: f64 = clients.iter().map(|c| c.data_size as f64).sum();
                let raw: Vec<f64> = clients
                    .iter()
                    .map(|c| {
                        let p_k = c.data_size as f64 / total_data;
                        let loss_q = c.local_loss.max(1e-10).powf(*q);
                        p_k * loss_q
                    })
                    .collect();
                let total: f64 = raw.iter().sum();
                if total > 0.0 {
                    raw.iter().map(|w| w / total).collect()
                } else {
                    vec![1.0 / k as f64; k]
                }
            }
            AggregationStrategy::ContributionBased => {
                // Leave-one-out contribution scoring.
                // Score_k = avg_loss_without_k - avg_loss_with_all
                // Higher score => client k is more valuable.
                let all_loss_avg: f64 =
                    clients.iter().map(|c| c.local_loss).sum::<f64>() / k as f64;

                let scores: Vec<f64> = (0..k)
                    .map(|leave_out| {
                        let without_avg: f64 = clients
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != leave_out)
                            .map(|(_, c)| c.local_loss)
                            .sum::<f64>()
                            / (k - 1) as f64;
                        // If removing client increases avg loss, client is valuable
                        (without_avg - all_loss_avg).max(0.0)
                    })
                    .collect();

                let total: f64 = scores.iter().sum();
                if total > 0.0 {
                    scores.iter().map(|s| s / total).collect()
                } else {
                    vec![1.0 / k as f64; k]
                }
            }
        }
    }

    /// Aggregate client model updates using the chosen strategy.
    pub fn aggregate(&mut self, clients: &[FLClient]) -> Vec<f64> {
        let agg_weights = self.compute_aggregation_weights(clients);

        let mut new_global = Array1::<f64>::zeros(self.num_features);
        for (client, &w) in clients.iter().zip(agg_weights.iter()) {
            new_global = new_global + &(&client.weights * w);
        }
        self.global_weights = new_global;

        agg_weights
    }

    /// Run a full FL training round.
    pub fn train_round(
        &mut self,
        clients: &mut [FLClient],
        local_epochs: usize,
        lr: f64,
    ) -> (Vec<f64>, FairnessMetrics) {
        // Broadcast global model
        for client in clients.iter_mut() {
            client.receive_global_model(&self.global_weights);
        }

        // Local training
        for client in clients.iter_mut() {
            client.local_train(local_epochs, lr);
        }

        // Aggregate
        let agg_weights = self.aggregate(clients);

        // Evaluate global model on each client's data
        let client_losses: Vec<(String, f64)> = clients
            .iter()
            .map(|c| {
                let preds = predict(&c.features, &self.global_weights);
                let loss = mse_loss(&preds, &c.labels);
                (c.name.clone(), loss)
            })
            .collect();

        let metrics = compute_fairness_metrics(&client_losses);
        (agg_weights, metrics)
    }

    /// Run multiple rounds of FL training and return per-round metrics.
    pub fn train(
        &mut self,
        clients: &mut [FLClient],
        rounds: usize,
        local_epochs: usize,
        lr: f64,
    ) -> Vec<FairnessMetrics> {
        let mut all_metrics = Vec::new();
        for _ in 0..rounds {
            let (_, metrics) = self.train_round(clients, local_epochs, lr);
            all_metrics.push(metrics);
        }
        all_metrics
    }
}

// ---------------------------------------------------------------------------
// Convenience: generate synthetic data (for testing without API)
// ---------------------------------------------------------------------------

/// Generate synthetic trading-like data for testing.
pub fn generate_synthetic_data(
    n_samples: usize,
    n_features: usize,
    rng: &mut impl Rng,
) -> (Array2<f64>, Array1<f64>) {
    let true_weights: Vec<f64> = (0..n_features).map(|_| rng.gen::<f64>() - 0.5).collect();
    let true_w = Array1::from(true_weights);

    let mut features = Array2::<f64>::zeros((n_samples, n_features));
    let mut labels = Array1::<f64>::zeros(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            features[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0;
        }
        let row = features.row(i);
        labels[i] = row.dot(&true_w) + (rng.gen::<f64>() - 0.5) * 0.01;
    }

    (features, labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_mse_loss() {
        let preds = Array1::from(vec![1.0, 2.0, 3.0]);
        let labels = Array1::from(vec![1.0, 2.0, 3.0]);
        assert!((mse_loss(&preds, &labels)).abs() < 1e-10);
    }

    #[test]
    fn test_standard_fedavg_weights() {
        let mut rng = StdRng::seed_from_u64(42);
        let (f1, l1) = generate_synthetic_data(100, 5, &mut rng);
        let (f2, l2) = generate_synthetic_data(200, 5, &mut rng);

        let c1 = FLClient::new("A", DataQuality::High, f1, l1);
        let c2 = FLClient::new("B", DataQuality::Medium, f2, l2);

        let server = FairFLServer::new(5, AggregationStrategy::StandardFedAvg);
        let weights = server.compute_aggregation_weights(&[c1, c2]);

        assert!((weights[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((weights[1] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_fairness_metrics() {
        let losses = vec![
            ("A".to_string(), 0.1),
            ("B".to_string(), 0.2),
            ("C".to_string(), 0.5),
        ];
        let metrics = compute_fairness_metrics(&losses);
        assert!((metrics.worst_loss - 0.5).abs() < 1e-10);
        assert!((metrics.best_loss - 0.1).abs() < 1e-10);
        assert!(metrics.gini_coefficient > 0.0);
    }

    #[test]
    fn test_training_reduces_loss() {
        let mut rng = StdRng::seed_from_u64(42);
        let (f, l) = generate_synthetic_data(100, 5, &mut rng);
        let mut clients = vec![FLClient::new("A", DataQuality::High, f, l)];

        let mut server = FairFLServer::new(5, AggregationStrategy::StandardFedAvg);
        let metrics = server.train(&mut clients, 10, 5, 0.01);
        let first_loss = metrics[0].average_loss;
        let last_loss = metrics[9].average_loss;
        assert!(last_loss < first_loss, "Training should reduce loss");
    }
}
