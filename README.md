# ChronosRep

ChronosRep is an agent-based simulation framework for studying trust and reputation dynamics in decentralised identity ecosystems.
The system models how verifiable credentials propagate, decay, and get contested across a population of heterogeneous agents,
and provides a formal mechanism for detecting adversarial behaviour through a combination of evidence fusion, stochastic differential dynamics,
and interaction topology analysis.

The design closely follows the theoretical development in the accompanying paper:

> Entropy-Regularized Evidence Fusion and Stochastic Differential Trust Dynamics for Decentralized Identity Intelligence

---

## Motivation

Decentralised identity systems allow participants to present cryptographic proofs of attributes without relying on a central authority.
This flexibility introduces a structural vulnerability: an agent that accumulates legitimate credentials early in its lifecycle
may later exploit them to participate in collusion rings, mount sybil attacks, or gradually erode the trust of honest parties
through coordinated false endorsements.

Classical reputation models respond to this threat by computing scalar scores from interaction histories.
Such scores fail to distinguish between an agent that is genuinely trustworthy and one that is strategically dormant,
and they offer no mechanism to express epistemic uncertainty about conflicting evidence.

ChronosRep addresses these limitations through four tightly coupled subsystems.
Evidence from heterogeneous credential sources is fused under Dempster-Shafer theory with an entropy-regularised combination rule.
The resulting belief vector is then consumed by a stochastic differential equation that models the continuous evolution of reputation
as an Ornstein-Uhlenbeck process with jump discontinuities.
A behavioural stream monitor detects statistical anomalies in interaction outcomes in real time.
An interaction topology engine inspects the community structure of the agent graph to penalise structurally implausible endorsement patterns.

---

## Architecture

The repository is organised as a Python package named `chronosrep`.
The top-level entry points are `run.py` for a single baseline simulation and `experiments/run_experiments.py`
for the full set of evaluation scenarios described in the paper.

```
chronosrep/
    model.py              Main Mesa model — ChronosRepModel, ChronosAgent
    scenarios.py          Three adversarial injection scenarios
    modules/
        vcgen.py          Verifiable Credential generator
        irv_pe.py         Identity Reputation Vector Processing Engine  (DST fusion)
        vadm.py           Volatility-Adaptive Decay Module              (OU-Jump SDE)
        bsm.py            Behavioural Stream Monitor                    (CUSUM detector)
        ite.py            Interaction Topology Engine                   (Louvain + structural penalty)
    core/
        types.py          Shared dataclasses — VCRecord, BehaviorEvent, IRV, FrameOfDiscernment
        interfaces.py     Abstract base classes — BaseModule, BaseScenario
    trust/
        graph.py          Directed weighted trust graph with edge-level success tracking
        propagation.py    Weighted PageRank trust propagation engine
        decay.py          Configurable decay scheduler — exponential, power-law, hyperbolic
    identity/
        registry.py       Credential registry — register, revoke, query by subject or type
        resolver.py       DID resolver — five DID methods, key rotation, resolution metadata
        revocation.py     Epoch-bucketed revocation index — velocity and accumulation metrics
    network/
        topology.py       Network topology builder — Erdos-Renyi, Barabasi-Albert, Watts-Strogatz
        edge_policy.py    Edge weight policy — recency decay, outcome momentum, volatility penalty
        partition.py      Louvain partition cache — modularity history, stability scoring
    stochastic/
        wiener.py         Per-agent Wiener process with realised variance and quadratic variation
        jump.py           Gamma-threshold gated jump process with Poisson arrival
        solver.py         Unified SDE solver — Euler-Maruyama and Milstein schemes
        volatility.py     Volatility estimator — realised vol, EWMA, Parkinson, Yang-Zhang
    analysis/
        metrics.py        MetricsCollector, time-to-detection tracker, reputation distribution
    evaluation/
        benchmark.py      BenchmarkRunner — single runs and parameter sweeps
        scenario_matrix.py  N-dimensional scenario grid with pivot table and coverage reporting
        results_writer.py   CSV, JSON, and summary output
    utils/
        math_utils.py     Softmax, sigmoid, entropy, Gini coefficient, KL divergence, Lorenz curve
        seed_manager.py   Reproducibility utilities — global seed, fork_rng, temp_seed context
        serialization.py  JSON and pickle I/O with NumPy encoding, checkpoint saving
        logging_utils.py  Structured logger and per-step metric logger

experiments/
    run_experiments.py                Full four-scenario evaluation run with plots
    sweep.py                          Parameter sweep over agent count, time horizon, and isolation threshold
    ou_mean_reversion_sensitivity.py  OU reversion trajectories across three values of theta
    ewma_vs_ou_jump_detector.py       Baseline EWMA vs. OU-Jump detector on a flash-loan exploit sequence
    reputation_inequality_gini.py     Gini coefficient of reputation inequality over simulation time
    trust_propagation_damping.py      PageRank trust propagation sensitivity to the damping factor
    revocation_velocity_analysis.py   Per-epoch revocation velocity and cohort accumulation curves
    volatility_regime_transitions.py  OU-Jump trajectory and volatility estimates across three regimes
    network_partition_stability.py    Louvain modularity over time for three network topology classes
```

---

## Core Subsystems

### Verifiable Credential Generator

The `VCGen` module produces synthetic credential portfolios for each agent.
Five credential types are supported: KYC, DID_DOC, BEHAVIORAL, DELEGATED, and GOVERNANCE.
Each type carries a distinct evidential weight that reflects its semantic authority in identity proofs.
Issuers are drawn from a three-tier hierarchy — ROOT_CA, INTERMEDIATE_CA, and LEAF_ISSUER —
with a chain-depth penalty applied to credentials that pass through more intermediate authorities.
Honest agents receive basic belief assignments strongly concentrated on the `trusted` hypothesis,
while attacker agents receive assignments that are more evenly spread or biased toward `untrusted`.
A per-type revocation probability is applied at generation time, with honest agents experiencing
approximately two percent revocation and attacker agents experiencing thirty-five percent.

### Identity Reputation Vector Processing Engine

The IRV_PE module implements the entropy-regularised Dempster combination rule from Equation 9 of the paper.

For a set of mass functions $m_1, \ldots, m_n$ defined over the frame of discernment
$\Theta = \{trusted, untrusted, unknown\}$, the combined mass is computed as

$$m^*(A) = \frac{\sum_{B \cap C = A} m_1(B) \cdot m_2(C)}{1 - K}$$

where $K = \sum_{B \cap C = \emptyset} m_1(B) \cdot m_2(C)$ is the conflict coefficient.
When $K$ is close to one, indicating irreconcilable evidence, the combination step is skipped
and the credence is redistributed according to an entropy filter with threshold $\eta = 2.5$.

The output of the engine is a five-dimensional Identity Reputation Vector:

$$\text{IRV} = \left[\text{Bel}(trusted),\ \text{Bel}(untrusted),\ \text{Bel}(unknown),\ \text{Pl}(trusted),\ \text{BetP}(trusted)\right]$$

All five components lie in $[0, 1]$.
The pignistic probability $\text{BetP}(trusted)$ is computed via the pignistic transform
and serves as the scalar reputation signal consumed by the downstream SDE.

### Volatility-Adaptive Decay Module

The VADM module governs the continuous evolution of reputation using an Ornstein-Uhlenbeck process
augmented with Gamma-threshold gated jumps:

$$dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t + J \cdot \mathbf{1}\left[\Gamma > 3\right]$$

where $\Gamma = |\varepsilon_t| / (\sigma / \sqrt{2\theta})$ is the dimensionless anomaly ratio
and $\varepsilon_t = X_t - \mu$ is the deviation from the long-run mean.
The mean reversion speed $\theta$ is updated adaptively at each step:

$$\theta \leftarrow \theta + \alpha(\varepsilon_t - \theta \cdot \Delta t)$$

This allows the SDE to tighten reversion pressure when an agent deviates persistently from its expected value,
and to relax reversion when behaviour stabilises.
The jump term $J$ is directed toward $\mu$ when $\Gamma$ exceeds the threshold,
producing an additional corrective impulse on top of the diffusion term.

The stochastic layer is factored into three composable primitives in `chronosrep/stochastic/`:
a `WienerProcess` that maintains per-agent increments and realised variance,
a `JumpProcess` that gates Poisson arrivals through the Gamma criterion,
and an `SDESolver` that combines both under either Euler-Maruyama or Milstein discretisation.

### Behavioural Stream Monitor

The BSM module maintains a per-agent CUSUM detector over the binary stream of interaction outcomes.
The statistic is updated as

$$S_t^+ = \max\left(0,\ S_{t-1}^+ + z_t - k\right)$$

where $z_t$ is the Z-score-normalised outcome and $k = 0.5$ is the allowance parameter.
An alarm is raised when $S_t^+$ exceeds the decision threshold $h = 4.0$.
The module also tracks Shannon entropy over a sliding outcome distribution window of thirty steps.
A low entropy score combined with an active alarm indicates stereotyped, non-random behaviour,
which is characteristic of automated collusion bots.

The anomaly score produced by BSM is fed back into the model as a multiplicative penalty on the static reputation:

$$r_{static} \leftarrow r_{static} \cdot (1 - 0.5 \cdot \text{anomaly})$$

### Interaction Topology Engine

The ITE module maintains a directed NetworkX graph of interactions and recomputes Louvain community partitions
every ten steps.
For each community, it computes two statistics: the internal density (DIN) and the external connectivity ratio.
A community is flagged as suspect when its internal density meets or exceeds 0.60 and its external connectivity
ratio falls at or below 0.20, meaning it is internally dense but isolated from the rest of the graph.

The structural penalty applied to evidence from a suspect community is

$$w_{penalty} = 1 - (\beta \cdot \text{Jaccard}(u, v) + \gamma \cdot \text{DIN})$$

with $\beta = \gamma = 0.5$.
Jaccard similarity between agent neighbourhoods detects agents that share an unusually high proportion
of common interaction partners, a topological signature of coordinated endorsement rings.

---

## Adversarial Scenarios

Three attack scenarios from the paper are implemented in `chronosrep/scenarios.py`.

**Sleeper Agent Attack.**
Twenty attacker agents behave honestly for the first two hundred steps, accumulating sufficient reputation
to exceed the isolation threshold $\tau = 0.4$.
At step 201 they activate, begin producing false endorsements, and attempt to exploit their accumulated credit.
The evaluation metric is isolation rate at the end of the simulation.

**Transgression and Recovery.**
Twenty percent of agents misbehave between steps 101 and 200, then resume honest behaviour.
This scenario tests whether the system can both detect the transgression period with low latency
and allow genuine recovery once behaviour normalises.
The primary metric is time-to-detection (TTD), defined as the number of steps from activation to isolation.

**Collusion Farming.**
Twenty percent of agents form a closed endorsement ring using the `force_interaction` mechanism of ITE.
Members exchange high-value mutual endorsements to inflate each other's reputation above $\tau$.
The structural penalty from ITE is the principal defence, and the evaluation metric is isolation rate
and TTD at the end of the simulation.

---

## Evaluation

The `chronosrep/evaluation/` package provides three utilities for systematic evaluation.

`BenchmarkRunner` accepts a `SweepConfig` that specifies lists of agent counts, time horizons, isolation thresholds,
random seeds, and scenario names.
It runs each combination sequentially, recording wall time alongside the standard reputation and isolation metrics.

`ScenarioMatrix` provides an N-dimensional grid abstraction over the sweep dimensions.
After a sweep completes, results can be sliced along any subset of dimensions
or aggregated into a pivot table for LaTeX export.

`ResultsWriter` serialises the full result list to CSV and JSON, writes a summary file
aggregated by scenario, and optionally writes the full matrix snapshot.
All output is written to `results/` by default.

---

## Experiments

The figure scripts in `experiments/` reproduce the plots from the paper.

| Script | Description |
|--------|-------------|
| `ou_mean_reversion_sensitivity.py` | OU reversion trajectories across three values of the reversion speed theta |
| `ewma_vs_ou_jump_detector.py` | Baseline EWMA vs. ChronosRep OU-Jump detector on a flash-loan exploit sequence |
| `reputation_inequality_gini.py` | Gini coefficient of reputation inequality over simulation time, three conditions |
| `trust_propagation_damping.py` | PageRank trust propagation sensitivity to the damping factor alpha |
| `revocation_velocity_analysis.py` | Per-epoch revocation velocity and cumulative accumulation curves by cohort |
| `volatility_regime_transitions.py` | OU-Jump trajectory and volatility estimates under calm, stress, and attack regimes |
| `network_partition_stability.py` | Louvain modularity over time for Erdos-Renyi, Barabasi-Albert, and Watts-Strogatz topologies |

To run a single script:

```
python experiments/ou_mean_reversion_sensitivity.py
```

To run the full evaluation suite:

```
python experiments/run_experiments.py
```

To run the parameter sweep:

```
python experiments/sweep.py
```

---

## Setup

Python 3.10 or later is required.
All dependencies are installed into a virtual environment.

```
python -m venv venv
source venv/bin/activate
pip install mesa numpy networkx python-louvain matplotlib pytest
```

To run the smoke test with N = 30 agents and T = 15 steps:

```
python test_small.py
```

To run the full unit test suite:

```
pytest tests/
```
## Citation

If you use this codebase in your research, please cite:

```
@article{chronosrep2026,
  title   = {Entropy-Regularized Evidence Fusion and Stochastic Differential Trust Dynamics
             for Decentralized Identity Intelligence},
  year    = {2026},
}
```
