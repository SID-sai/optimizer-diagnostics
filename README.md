# Optimizer Behavior Analysis: Beyond Loss Curves

A from-scratch NumPy neural network (no PyTorch/TensorFlow autodiff) trained on Fashion-MNIST,
used to compare SGD, Adagrad, RMSprop, and Adam — not just on which one "wins," but on *why*
they behave differently, using three diagnostic lenses computed directly from the optimizers'
own internal state and the loss landscape itself.

## Motivation

It's common knowledge that "Adam is usually faster, SGD sometimes generalizes better." This
project doesn't just repeat that — it builds the mechanism underneath it, from scratch, and
checks whether a well-known research finding (that adaptive optimizers tend to land in sharper,
less robust minima than SGD) actually shows up on a small, controlled setup.

## Setup

- **Dataset:** Fashion-MNIST, loaded from raw image files, normalized to `[-1, 1]`, flattened
  to 784-dim vectors, shuffled once before training.
- **Architecture:** `784 → 128 → 128 → 128 → 10`, ReLU hidden activations, Softmax + Categorical
  Cross-Entropy output.
- **Batch size:** 32 (mini-batch training)
- **Epochs:** 50 per optimizer
- **Optimizers compared:** SGD (no momentum), Adagrad, RMSprop, Adam — each run independently,
  same architecture, same dataset.

## The three lenses

**1. Effective Learning Rate** — the *real* per-weight step size, not just the configured
`learning_rate`. Adagrad/RMSprop/Adam divide the learning rate by a per-weight running measure
of past gradient magnitude, so the actual step taken can be orders of magnitude different from
the number you set. Tracked once per epoch, per optimizer.

**2. Gradient Noise Scale** — how much individual training samples *disagree* about which
direction to move the weights, hidden underneath the batch-averaged gradient. Computed via
per-sample forward/backward passes every 5 epochs (expensive, hence the sparser sampling).

**3. Minimum Sharpness** — the top eigenvalue of the Hessian at the end of training, estimated
via finite-difference Hessian-vector products and power iteration (no autodiff available, so
this is done with two nearby gradient evaluations per iteration). A higher value means a more
fragile minimum — small weight perturbations cause large loss increases.

## Results

| Optimizer | Train Loss | Train Acc | Test Acc | Train–Test Gap | Sharpness |
|---|---|---|---|---|---|
| SGD      | 0.317 | 0.885 | 0.862 | 2.3% | 144.8 |
| Adagrad  | 0.109 | 0.966 | 0.885 | 8.1% | 123.9 |
| RMSprop  | 0.405 | 0.901 | 0.851 | 5.0% | **12,770.6** |
| Adam     | 0.094 | 0.965 | 0.878 | 8.7% | **29.5** |

### Effective Learning Rate
SGD stays flat at the configured `0.001`, as expected. Adagrad's effective LR decreases
monotonically over training — consistent with theory, since its gradient-squared cache only
accumulates and never resets. RMSprop's effective LR rises fast and stays elevated
(roughly 5,000–7,000), the most aggressive of the four. Adam dips briefly early on, then rises
smoothly to around 500 by epoch 50.

### Gradient Noise Scale
All four optimizers show rising noise scale over the first ~10 epochs, then roughly plateau.
SGD stays lowest throughout (~24–25), with no mechanism amplifying its raw gradient signal.
Adagrad ends highest (~38).

### Sharpness — the key finding
**RMSprop's minimum is dramatically sharper than the other three** — its sharpness (12,770) is
roughly 100–400x higher than SGD, Adagrad, or Adam. This lines up with RMSprop's own training
curve, which was visibly less stable than the others (loss occasionally increased between
epochs rather than monotonically decreasing).

**Sharpness did not predict the generalization gap in this setup.** Ranking by sharpness:
Adam (29.5) < Adagrad (123.9) < SGD (144.8) << RMSprop (12,770.6). Ranking by train-test gap:
SGD (2.3%) < RMSprop (5.0%) < Adagrad (8.1%) < Adam (8.7%). Adam had the *flattest* minimum of
the four but one of the *largest* generalization gaps — the opposite of what the common
"sharp minima generalize worse" narrative would predict. RMSprop, despite being wildly sharper
than everything else, didn't have the worst gap either.

This is a genuine, honestly-reported mismatch with the textbook story — not a bug. Plausible
explanations: the model/dataset here are small-scale compared to the settings where the
sharpness-generalization link was originally studied, the Hessian estimate uses only 64 samples
and a single finite-difference step size (`r=1e-5`), which introduces its own noise, and more
broadly, sharpness alone is an incomplete predictor of generalization even in the literature —
this remains a genuinely open question in ML research, not a settled one.

## Known limitations

- Initial weights are **not identical** across the four optimizer runs — each run re-initializes
  layers with a fresh random seed rather than sharing one saved starting point. A stricter
  comparison would control for this.
- Gradient noise scale is tracked for layer 1 only, as a representative sample, not averaged
  across all four layers.
- The effective-LR formula for Adam is a simplification — it uses the raw `weight_cache`, not
  Adam's internal bias-corrected version, so it reflects the right trend but not the exact
  internal value Adam uses to update weights.
- 50 epochs is enough to see clear separation between optimizers but may not represent full
  convergence for all four.

## Repo structure

```
main.ipynb         — full pipeline: framework, training, diagnostics, plots
README.md          — this file
requirements.txt    — Python dependencies
.gitignore          — excludes dataset files from version control
```

## Setup to run

```bash
pip install -r requirements.txt
```

Download Fashion-MNIST (not included in this repo — see `.gitignore`) using the download cell
in `main.ipynb`, then run all cells top to bottom.
