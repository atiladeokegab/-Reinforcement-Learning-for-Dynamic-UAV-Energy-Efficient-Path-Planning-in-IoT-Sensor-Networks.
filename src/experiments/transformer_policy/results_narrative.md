# GTrXL for High-Variance LoRaWAN Duty-Cycle Environments: Technical Justification

## 1. The Core Challenge: Stochastic Duty-Cycle Bursts

LoRaWAN sensors transmit with an effective probability of **10%** per timestep.
This means the UAV observes a sensor's buffer state as a **Bernoulli burst process**: at any
given step, nine out of ten sensors report zero new data regardless of how urgently their
buffers are filling.  A naïve MLP policy conflates "sensor is silent right now" with
"sensor does not need service", producing systematic under-collection from sensors that are
accumulating data in the background.

Standard vanilla Transformers amplify this problem via **attention score saturation**: a high
link-quality sensor that happens to be transmitting at timestep *t* dominates the attention
distribution, overshadowing sensors whose urgency has been silently increasing for dozens of
steps.  Once the softmax saturates around a single active sensor, gradient signal to the
other query positions collapses, yielding a policy that greedily orbits high-SNR sensors
rather than planning global coverage.

---

## 2. How GTrXL's Gated Residual Connections Resolve This

### 2.1 Gate-Controlled Forgetting vs. Remembering

The Gated Transformer-XL replaces the standard residual addition

```
h' = LayerNorm(h + F(h))
```

with a learned GRU-style gate:

```
g  = σ(W_g · [h, F(h)] + b_g)          # reset gate
h' = (1 − g) ⊙ h  +  g ⊙ tanh(W_h · F(h))
```

When a LoRa sensor is silent (F(h) carries near-zero attention update), the gate
*g* closes toward 0 and the hidden state **persists unchanged** from the previous
segment.  Concretely, the network remembers "sensor 7 had 82% buffer fill 12 steps
ago and has been silent since" rather than interpreting silence as low urgency.

### 2.2 EMA-ADR Latency Coverage

The EMA-ADR spreading-factor adaptation uses λ = 0.1, so the effective update
lag is τ = 1/λ = **10 timesteps**.  During this window, a sensor that just lost
packets may still report SF12 (lowest data rate, longest range) even though the
RF environment has improved.  An agent acting on the instantaneous SF value would
systematically under-estimate the available throughput.

By setting `attention_memory_training = attention_memory_inference = 50` steps,
the GTrXL memory horizon covers **5 × τ**, which is long enough for the attention
heads to observe a complete ADR adaptation cycle — the initial loss event, the
slow SF reduction, and the stabilisation at the new spreading factor — and
condition its collection decision on the *trend* rather than the snapshot.

### 2.3 Parallel Urgency Tracking via Multi-Head Attention

With **8 attention heads** operating in parallel over the 50-step memory, each
head can specialise on a different temporal pattern without interfering with the
others:

| Head role (emergent) | Query feature | Key feature |
|---|---|---|
| Urgency scout | UAV position | Buffer fill + loss rate |
| ADR tracker | SF history | Recent RSSI delta |
| Proximity planner | UAV position | Relative (dx, dy) |
| Starvation alarm | Time since last visit | Terminal penalty proxy |

This decomposition is **not hard-coded** — it emerges during training — but it is
enabled by the combination of eight independent projection matrices and the
inclusion of relative sensor positions (`dx`, `dy`) in the 5-feature-per-sensor
observation.  A single-headed architecture or an LSTM would have to serialise all
of this reasoning through a single bottleneck vector.

### 2.4 Gradient Flow Through Long Sequences

The Gaussian shadowing term N(0, 4 dB) introduces variance of ±4 dB in the
received signal strength, which translates into stochastic link quality changes
of up to 2 SF grades in a single step.  When back-propagating through a 50-step
memory window, standard residual connections accumulate this variance multiplicatively
across layers, leading to gradient explosion in the early curriculum stages.

The GRU gate's **bounded output** (tanh non-linearity on the update path, sigmoid
on the gate) bounds each layer's Jacobian eigenvalues close to 1, giving stable
gradient norms even in the highest-variance episodes of Stage 4 (500×500 grid,
50 sensors, full shadowing).  The `init_gru_gate_bias = 2.0` setting initialises
σ(b_g) ≈ 0.88, meaning gates start mostly open so gradients propagate freely
before the model has learned which information to retain.

---

## 3. Comparison with Baseline Architectures

| Property | MLP + FrameStack | LSTM | Vanilla Transformer | GTrXL (ours) |
|---|---|---|---|---|
| Handles burst silence | ✗ aliased | ✓ (leaky) | ✗ saturates | ✓ gated persistence |
| Covers ADR lag (10 ts) | ✗ k=4 too short | ✓ | ✓ | ✓ |
| Parallel sensor tracking | ✗ | ✗ | ✓ | ✓ (8 heads) |
| Stable gradient over 50 ts | — | ✓ | ✗ softmax collapse | ✓ |
| Relational (dx, dy) attention | ✗ | ✗ | ✓ | ✓ |

The key advantage over a plain LSTM is the **relational** capacity: the LSTM's
hidden state is a single fixed-width vector that must summarise all N ≤ 50 sensors
implicitly, so its capacity saturates as N grows.  GTrXL's attention mechanism
scales to 50 sensors with constant model size because the attention matrix is
computed dynamically at each step over the full sensor set.

---

## 4. Expected Behavioural Signatures in Training Curves

1. **Faster Jain's fairness convergence**: The multi-head attention should detect
   starvation early in each episode and prompt the UAV to re-route, so JFI should
   exceed the 0.85 gate sooner than the DQN MLP baseline.

2. **Lower return variance in Stage 4**: The gated memory's resistance to
   shadowing-induced gradient noise should manifest as tighter confidence intervals
   in the Stage 4 (500×500) reward curves.

3. **Earlier Stage 4 graduation**: By implicitly modelling the EMA-ADR lag, the
   policy should achieve near-optimal throughput without the DQN baseline's
   explicit hovering behaviour while waiting for ADR convergence.
