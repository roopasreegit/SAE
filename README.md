# Sparse Autoencoders on GPT-2

I came across the Cunningham et al. 2023 paper while going down a rabbit hole on mechanistic interpretability and wanted to actually run it myself rather than just read about it. This is my from scratch implementation.

The core question that got me interested: GPT-2 has 768 neurons per layer. That's not a lot, but it somehow represents an enormous number of concepts. How? The answer seems to be that neurons are *polysemantic*, i.e a single neuron responds to completely unrelated things. Token position 42 in layer 8 might fire on French text, and also on code comments, and also on dates. That's weird and makes interpretability really hard.

SAEs are one attempt to fix this. The idea is to learn a much larger set of "features" (6144 in this case) such that each feature is *monosemantic*, i.e it fires for one thing. We train a bottleneck autoencoder with an L1 penalty that forces most features to be zero for any given token, which pushes the model to find clean, sparse, human-readable directions in activation space.

---

## What I actually found

After training on 200k tokens from WikiText-2, some features are clearly interpretable:

- one feature fires almost exclusively on month names
- another fires on numbers written as words (one, two, three etc)
- several fire on punctuation in specific syntactic positions
- a few that I genuinely can't explain, which is also interesting

The dashboard lets you scroll through features and see the top 20 tokens that activate each one. Browsing it is surprisingly fun.

---

## Running it

Tested on Google Colab T4. Takes about 35 minutes end to end.

```bash
git clone https://github.com/YOUR_USERNAME/sparse-autoencoders.git
cd sparse-autoencoders
pip install -r requirements.txt

python experiments/run_pipeline.py   # collect activations → train → analyse
python dashboard/app.py --share      # launch the feature browser
```

On Colab specifically:

```python
!git clone https://github.com/YOUR_USERNAME/sparse-autoencoders.git
%cd sparse-autoencoders
!pip install -r requirements.txt -q
%run experiments/run_pipeline.py
%run dashboard/app.py --share
```

The pipeline caches each step — if it crashes halfway through, re-running picks up where it left off.

---

## How it works

Hook into GPT-2 small's residual stream at layer 8. Run WikiText-2 through it. Save the 200k × 768 activation matrix. Then train a sparse autoencoder on those vectors:

```
h    = ReLU( W_enc @ (x - b_dec) + b_enc )     # 768 → 6144, sparse
x̂   = W_dec @ h + b_dec                         # 6144 → 768

Loss = ||x - x̂||²  +  λ · ||h||₁
```

The L1 term is what makes features sparse. λ = 2e-4 here — tuning this is the main thing that affects results. Too high and most features die (never activate). Too low and the sparsity is fake — features fire on everything.

One constraint that matters: decoder columns have to be unit norm. Without it the model cheats — it makes encoder values huge and decoder columns tiny, satisfying the L1 penalty without actually being sparse. Enforcing `||W_dec[:, i]|| = 1` closes that loophole.

Layer 8 specifically because it's deep enough to have semantic content but not so deep that the residual stream is dominated by the unembedding direction. Layer 4 would also work.

---

## Repo structure

```
src/
  activations.py   — GPT-2 hook, streams WikiText-2, saves activation vectors
  sae.py           — the autoencoder (encoder, decoder, loss, unit-norm constraint)
  train.py         — training loop with dead feature tracking
  analyze.py       — vectorised top-K search across all features
experiments/
  run_pipeline.py  — runs all four steps in order
dashboard/
  app.py           — Gradio app, shows top activating tokens per feature
```

---

## Numbers

| thing | value |
|---|---|
| base model | GPT-2 small (124M params) |
| hook layer | 8 of 12 |
| d_model | 768 |
| d_hidden | 6144 (8× expansion) |
| λ (L1 coeff) | 2e-4 |
| training steps | 30,000 |
| tokens | 200k from WikiText-2 |
| variance explained | ~70–75% |
| avg L0 | ~40–60 features per token |

---

## Things I want to try next

- Training on a larger / more diverse corpus (WikiText-2 is pretty clean English, which probably biases what features emerge)
- Comparing layer 4 vs layer 8 vs layer 12 — do earlier layers have more syntactic features and later layers more semantic ones?
- The Anthropic monosemanticity paper uses a much larger expansion factor (up to 131k features on a one-layer model). Curious how quality scales with d_hidden

---

## References

- [Cunningham et al. 2023](https://arxiv.org/abs/2309.08600) — the paper this implements
- [Bricken et al. 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html) — Anthropic's version, much larger scale, worth reading alongside
- [Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html) — why superposition happens in the first place, good background


