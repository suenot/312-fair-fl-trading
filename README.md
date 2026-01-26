# Chapter 182: Fair Federated Learning for Trading

## 1. Introduction: Fairness in Collaborative Trading Model Training

Federated Learning (FL) has emerged as a powerful paradigm for training machine learning models across distributed data sources without sharing raw data. In finance, this enables multiple trading desks, hedge funds, or institutional investors to collaboratively train predictive models while preserving data privacy and complying with regulatory constraints.

However, standard FL algorithms like FedAvg treat all participants uniformly, aggregating model updates by simple data-size weighting. This creates a critical fairness problem in trading contexts:

- **Data quality disparity**: Some desks operate in liquid markets with clean, high-frequency data, while others trade in illiquid markets with noisy, sparse observations.
- **Contribution asymmetry**: A desk contributing genuinely informative signals should receive more benefit than one contributing noise.
- **Performance uniformity vs. equity**: Should the global model perform equally well for all participants, or should it reward those who contribute more?

Fair Federated Learning addresses these questions by introducing mechanisms that ensure equitable outcomes across heterogeneous participants. In this chapter, we explore three key approaches -- q-FedAvg, Agnostic Federated Learning (AFL), and contribution-based weighting -- and demonstrate their application to collaborative trading model training using Bybit cryptocurrency market data.

The stakes are high: unfair FL can cause smaller or lower-quality participants to receive a model that actively harms their trading performance, while a few dominant participants capture all the benefit. Fair FL aims to prevent this by balancing the tension between aggregate performance and individual participant welfare.

## 2. Mathematical Foundations

### 2.1 Standard FedAvg Recap

In standard Federated Averaging, given K clients each with local dataset D_k of size n_k, the global objective is:

$$\min_w F(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)$$

where $n = \sum_k n_k$ and $F_k(w) = \frac{1}{n_k} \sum_{i \in D_k} \ell(w; x_i, y_i)$ is the local loss function.

The aggregation rule is:

$$w^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^{t+1}$$

This gives larger clients more influence, which can be unfair when data size does not correlate with data quality.

### 2.2 q-FedAvg: Fairness via Reweighting

q-FedAvg (Li et al., 2020) introduces a hyperparameter q >= 0 that controls the fairness-accuracy trade-off. The objective becomes:

$$\min_w f_q(w) = \sum_{k=1}^{K} \frac{p_k}{q+1} (F_k(w))^{q+1}$$

where $p_k = n_k / n$ are the mixing weights.

When q = 0, this reduces to standard FedAvg. As q increases, the optimization places more weight on clients with higher loss, encouraging the model to improve performance for the worst-off participants.

The key insight is that the gradient of $f_q$ with respect to w is:

$$\nabla f_q(w) = \sum_{k=1}^{K} p_k (F_k(w))^q \nabla F_k(w)$$

This means clients with higher loss (worse performance) receive amplified gradient contributions, steering the model toward reducing their loss more aggressively.

In the aggregation step, client updates are reweighted:

$$\Delta_k = (F_k(w^t))^q \cdot (w_k^{t+1} - w^t)$$

$$w^{t+1} = w^t + \frac{\sum_k p_k \Delta_k}{\sum_k p_k (F_k(w^t))^q}$$

### 2.3 Agnostic Federated Learning (AFL)

AFL (Mohri et al., 2019) takes a minimax approach, optimizing the model for the worst-case mixture of client distributions:

$$\min_w \max_{\lambda \in \Delta_K} \sum_{k=1}^{K} \lambda_k F_k(w)$$

where $\Delta_K$ is the K-dimensional simplex. This is a saddle-point problem: we minimize over model parameters while maximizing over the mixture weights lambda.

The solution alternates between:

1. **Inner maximization**: Given current w, the worst-case lambda places all weight on the client with the highest loss:
$$\lambda_k^* = \begin{cases} 1 & \text{if } k = \arg\max_j F_j(w) \\ 0 & \text{otherwise} \end{cases}$$

2. **Outer minimization**: Update w using the weighted gradient:
$$w^{t+1} = w^t - \eta \sum_k \lambda_k^t \nabla F_k(w^t)$$

In practice, a softmax relaxation is used to avoid degenerate solutions:

$$\lambda_k \propto \exp(\gamma \cdot F_k(w))$$

where gamma controls the sharpness of the distribution.

### 2.4 Contribution-Based Weighting

Contribution-based approaches measure each client's marginal contribution to the global model's performance, inspired by Shapley values from cooperative game theory.

The Shapley value for client k is:

$$\phi_k = \sum_{S \subseteq K \setminus \{k\}} \frac{|S|!(K-|S|-1)!}{K!} [V(S \cup \{k\}) - V(S)]$$

where V(S) is the value function (e.g., negative validation loss) of the model trained on the coalition S.

Computing exact Shapley values is exponential in K, so we use Monte Carlo approximation:

1. Sample random permutations pi of clients
2. For each permutation, compute the marginal contribution of client k when added to the set of clients preceding it in pi
3. Average over permutations

The resulting Shapley-based weights are used for aggregation:

$$w^{t+1} = \sum_{k=1}^{K} \frac{\max(\phi_k, 0)}{\sum_j \max(\phi_j, 0)} w_k^{t+1}$$

Clients with negative contributions (those that hurt the model) receive zero weight, effectively excluding them from aggregation.

## 3. Trading Application: Fair Model Sharing Between Trading Desks

Consider a scenario with four trading desks collaborating to predict short-term cryptocurrency price movements:

| Desk | Market Focus | Data Quality | Characteristics |
|------|-------------|-------------|-----------------|
| Desk A | BTC spot (Bybit) | High | Large volume, clean tick data, low latency |
| Desk B | Altcoin pairs | Medium | Moderate volume, some missing data |
| Desk C | Emerging tokens | Low | Sparse data, high noise, frequent gaps |
| Desk D | Cross-exchange arb | Mixed | Aggregated data with latency artifacts |

**Without fair FL**: Standard FedAvg would weight Desk A's updates most heavily (largest dataset), potentially producing a model that works well for liquid BTC trading but fails for altcoins and emerging tokens. Desk C would receive a model poorly suited to its trading environment.

**With q-FedAvg (q > 0)**: The algorithm amplifies updates from desks with higher loss, forcing the model to improve performance for Desk C and Desk D. The resulting model may sacrifice a small amount of Desk A's performance but provides meaningfully better predictions for all desks.

**With AFL**: The minimax formulation ensures that the worst-performing desk's loss is minimized. This provides a strong fairness guarantee but may be overly conservative.

**With contribution scoring**: Each desk's actual contribution to model quality is measured. If Desk C's noisy data hurts the model, its weight is reduced. If Desk D's cross-exchange data provides unique signals, its weight increases despite having mixed quality.

The choice between these approaches depends on the federation's governance structure. Cooperative federations (e.g., desks within the same firm) may prefer AFL's egalitarian guarantee, while competitive federations (e.g., independent funds) may prefer contribution-based weighting to incentivize high-quality data sharing.

## 4. Implementation Walkthrough

Our Rust implementation in `rust/src/lib.rs` provides the following components:

### 4.1 Core Data Structures

- **`FLClient`**: Represents a trading desk with local model weights, local data, data quality level, and local loss tracking.
- **`FairFLServer`**: The central aggregation server implementing multiple fairness strategies.
- **`FairnessMetrics`**: Tracks per-client losses, Gini coefficient, and min-max loss ratios.

### 4.2 Fair Aggregation Strategies

The server supports three aggregation modes via the `AggregationStrategy` enum:

1. **`StandardFedAvg`**: Baseline data-size-weighted averaging.
2. **`QFedAvg { q: f64 }`**: Loss-reweighted averaging with configurable q parameter.
3. **`ContributionBased`**: Shapley-inspired contribution scoring with leave-one-out approximation.

### 4.3 Training Loop

Each round proceeds as follows:

1. Server broadcasts the current global model to all clients.
2. Each client performs local SGD on its data for a configurable number of local epochs.
3. Each client reports its updated weights and local loss to the server.
4. The server computes aggregation weights based on the chosen strategy.
5. The server aggregates updates and computes fairness metrics.

### 4.4 Simulated Trading Model

We use a simple linear prediction model: given a feature vector x (derived from price returns, volume, and volatility), predict the next-period return. The loss function is mean squared error (MSE). While simple, this model isolates the effects of fair aggregation from model architecture concerns.

### 4.5 Data Quality Simulation

Clients receive data with different quality levels:
- **High quality**: Raw features with minimal noise (sigma = 0.01)
- **Medium quality**: Features with moderate noise (sigma = 0.05)
- **Low quality**: Features with high noise and 20% missing values (sigma = 0.15)

## 5. Bybit Data Integration

We fetch real market data from the Bybit public API to ground our experiments in real trading scenarios.

### 5.1 API Endpoint

We use the Bybit V5 kline (candlestick) endpoint:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

This returns hourly OHLCV data for the BTC/USDT perpetual contract.

### 5.2 Feature Engineering

From raw OHLCV data, we compute:

- **Returns**: `r_t = (close_t - close_{t-1}) / close_{t-1}`
- **Volatility**: Rolling 5-period standard deviation of returns
- **Volume ratio**: `vol_t / mean(vol_{t-5:t})`
- **Price range**: `(high_t - low_t) / close_t`
- **Momentum**: `(close_t - close_{t-5}) / close_{t-5}`

### 5.3 Label Construction

The prediction target is the next-period return sign and magnitude:

$$y_t = r_{t+1} = \frac{close_{t+1} - close_t}{close_t}$$

This is a regression target; for trading, the sign indicates direction and the magnitude indicates conviction.

## 6. Fairness Metrics and Results

### 6.1 Metrics

We evaluate fairness using several complementary metrics:

- **Loss variance**: $\text{Var}(\{F_k(w^*)\}_{k=1}^K)$ -- lower is fairer.
- **Min-max ratio**: $\min_k F_k / \max_k F_k$ -- closer to 1 is fairer.
- **Gini coefficient of losses**: Measures inequality in loss distribution across clients. 0 = perfect equality, 1 = maximum inequality.
- **Worst-case loss**: $\max_k F_k(w^*)$ -- the loss of the worst-off client.

### 6.2 Expected Results

Based on the theoretical properties and typical empirical outcomes:

| Metric | FedAvg | q-FedAvg (q=2) | AFL | Contribution |
|--------|--------|----------------|-----|--------------|
| Avg Loss | **Best** | Slightly higher | Higher | Moderate |
| Worst Loss | High | **Lower** | **Lowest** | Moderate |
| Gini | High | Low | **Lowest** | Moderate |
| Min-Max Ratio | Low | Higher | **Highest** | Moderate |

**Key observations**:

1. **Standard FedAvg** achieves the lowest average loss but at the cost of high inequality. The best-data client (Desk A) benefits most while the worst-data client (Desk C) may see degraded performance compared to training alone.

2. **q-FedAvg** provides a smooth trade-off. Increasing q from 0 to 5 progressively reduces inequality at a diminishing cost to average performance. q = 1-3 is typically a good operating range for trading applications.

3. **AFL** provides the strongest fairness guarantee but can be overly conservative, sacrificing too much average performance to improve the worst case. Best suited for risk-averse federations.

4. **Contribution-based** weighting is most appropriate when participant incentives matter. It naturally handles free-riders (clients contributing noise) and rewards genuine signal providers.

### 6.3 Practical Considerations

- **Non-stationarity**: Financial markets are non-stationary, so fairness must be re-evaluated periodically. A client with high-quality data today may have degraded quality tomorrow.
- **Strategic behavior**: In competitive settings, clients may strategically degrade their reported loss to gain influence under q-FedAvg. Contribution-based methods are more robust to such manipulation.
- **Convergence speed**: Fair methods typically converge slower than standard FedAvg due to the reweighting overhead. Budget 1.5-2x more communication rounds.
- **Privacy**: Fair aggregation methods are compatible with secure aggregation and differential privacy, though the interaction between fairness and privacy is an active research area.

## 7. Key Takeaways

1. **Standard FedAvg is not fair by default**: Weighting by data size benefits large-data participants and can harm smaller ones. In trading, this can lead to models that only work well for the most liquid markets.

2. **q-FedAvg provides tunable fairness**: The q parameter offers a smooth trade-off between average performance and equality. Start with q = 1 and increase based on the federation's fairness requirements.

3. **AFL guarantees worst-case performance**: The minimax formulation ensures no single participant is left behind. Use this when regulatory or contractual requirements demand performance parity.

4. **Contribution scoring aligns incentives**: Shapley-based weighting rewards genuine contributions and penalizes free-riders. Essential for competitive federations where participants may act strategically.

5. **Fairness is not free but is often worthwhile**: Fair FL methods typically sacrifice 5-15% of average performance to achieve significantly more equitable outcomes. In trading, this can mean the difference between a federation that retains members and one that fractures.

6. **Monitor fairness dynamically**: Financial markets evolve, and so do data quality and contribution patterns. Implement continuous fairness monitoring and adjust strategies accordingly.

7. **Combine approaches judiciously**: In practice, a hybrid approach -- contribution-based weighting with a q-FedAvg fairness floor -- often performs best, providing both incentive alignment and minimum fairness guarantees.

## References

- Li, T., Sanjabi, M., Beirami, A., & Smith, V. (2020). Fair Resource Allocation in Federated Learning. ICLR 2020.
- Mohri, M., Sivek, G., & Suresh, A. T. (2019). Agnostic Federated Learning. ICML 2019.
- Wang, T., Rauber, J., & Bethge, M. (2020). Data Shapley: Equitable Valuation of Data for Machine Learning. JMLR.
- Kairouz, P., et al. (2021). Advances and Open Problems in Federated Learning. Foundations and Trends in Machine Learning.
