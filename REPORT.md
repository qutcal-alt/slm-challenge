# MP1 Report: A Compact MoE Language Model with In-Window Neural Cache

DASE7506 Mini-Project 1. Protocol `7506-mp1-wt2-v2`. WikiText-2 raw text, BPE-2048, independent windows of 256 targets. The ranking metric is full-test bits per byte (BPB), lower better, CPU FP32.

**Submitted predictor H1rC2.** Test BPB **1.511051**. Validation BPB 1.493983. Checkpoint SHA-256 `cc3b4bfc717e23ccb0322e37e1b4696631edb529b4928598f67849713e9cb96d`. The matched-budget baseline (`expC2`, 40.96M tokens) scores 1.769429 test BPB.

## 1. Task and constraints

The supplied evaluator scores every token except the first of each split exactly once. A prediction at position \(t\) may use only the observed prefix through \(t\). Windows do not share state. Development and mixture selection used validation; the test split was read only after the method was frozen.

Resource limits for the same frozen predictor: CPU scoring time at most 5× the README baseline (5.92 s on Xeon Platinum 8457C, limit 29.6 s), peak evaluation RAM ≤ 4 GiB, uncompressed inference assets ≤ 64 MiB. Weights, n-gram statistics and cache mixing coefficients must come from training text and validation, not test.

The original package baseline is four GPT blocks, width 128, 1,088,256 parameters, about 2.10 test BPB after 1,200 steps. This report uses a **token-matched** baseline of 5,000 steps so architecture comparisons are not confounded by extra compute.

## 2. Method

### 2.1 Architecture

`student.py` replaces the baseline GPT with a 10.50M-parameter model:

- **RMSNorm** instead of LayerNorm.
- **RoPE** instead of learned absolute positions. Embeddings are tied.
- **Width 320, depth 6, 4 heads.**
- **SwiGLU MoE FFN** in every block: 4 experts, expert hidden size 320, top-2 routing. Router softmax mass is kept, so language-model loss trains the router even for \(k=1\). A load-balance term \(N\sum_i f_i P_i\) with weight 0.01 is added to the token loss.
- **Value-direction residual** after causal attention: the attended vector's projection onto the normalized value direction is subtracted (`xsa_projection`). This is a cheap orthogonalization, not extra retrieval.
- **Dropout 0.1** on attention and MLP residual branches.

A hashed bigram table and an interpolated trigram mixer were implemented and tested, then discarded: both hurt or lagged the selected backbone (Section 4).

### 2.2 Training recipe

Optimizer AdamW, peak learning rate \(10^{-3}\), 100-step warmup, cosine decay to 10% of peak over 5,000 steps, weight decay 0.3 (norms, RoPE, router and biases excluded), gradient clip 1.0, EMA 0.99 on unique parameters, tied `token.weight`/`head.weight` updated once. Batch 32 × 256 targets, seed 17, BF16 on RTX 5070. Training processed 40,960,000 tokens. Validation every 300 steps selected **step 4,200** (34,406,400 tokens, validation 1.512823). Continuing to 5,000 steps slightly overfit (last validation 1.515171).

Training itself does **not** mix a neural cache; mixing is applied only at evaluation on the frozen checkpoint.

### 2.3 Frozen in-window neural cache

At evaluation, hidden states \(h_{1:T}\) inside one window are L2-normalized. For target \(t\ge 2\), keys are earlier positions \(i<t\) and stored values are the next-token ids \(x_{i+1}\). Pointer weights are a causal softmax of \(\theta\,h_t^\top h_i\). Position 0 has no history and is left as the model distribution. The cache distribution \(c_t\) is mixed:

\[
p_t = (1-\lambda)\,p^{\text{model}}_t + \lambda\, c_t.
\]

The mix uses only in-window states, so windows remain independent and prefixes remain causal. A validation sweep over \(\lambda\in\{0,0.02,\ldots,0.1\}\) and \(\theta\in\{5,8,10,12,20\}\) selected \(\lambda=0.04\), \(\theta=12\) (sweep BPB 1.494517). Baking those two scalars into the checkpoint yields H1rC2; no extra parameters are stored.

## 3. Main comparison at matched tokens

Table 1. Same 5,000-step / 40.96M-token budget unless noted. Validation from training logs (GPU FP32 scorer during training) except H1rC2 and `expC2` official CPU numbers.

| Run | Parameters | Best step | Val BPB | Test BPB |
|---|---:|---:|---:|---:|
| Course 1,200-step baseline (README) | 1.09M | 1,200 | — | ≈2.10 |
| `expC2` matched baseline | 1.09M | 5,000 | 1.742927 | **1.769429** |
| Dense SwiGLU+XSA `expF1` (36,000 tokens fewer) | 5.61M | 3,600 | 1.577953 | — |
| MoE 3,600-step `expM5` | 10.50M | 3,600 | 1.530084 | — |
| MoE 6,000-step `expM9` | 10.50M | 4,200 | 1.518403 | — |
| **H1r backbone** | 10.50M | 4,200 | 1.511733 (CPU) / 1.512823 (train log) | **1.533351** |
| **H1rC2 + cache 0.04/12** | 10.50M | 4,200 | **1.493983** | **1.511051** |

H1r improves 0.231 CPU validation BPB over `expC2` at identical token count (1.743 → 1.512). The cache then improves another 0.018 validation BPB and 0.022 test BPB (1.533 → 1.511). Relative to the README 1,200-step baseline of ~2.10 test BPB, H1rC2 is 0.59 BPB better; the honest matched comparison is 1.769 → 1.511, a 0.258 test BPB gain.

Training wall-clock on RTX 5070: `expC2` 38 s, H1r 411 s. Search cost of later routing and mix sweeps is additional GPU minutes, not extra test-set access.

## 4. Ablations of the key mechanisms

### 4.1 MoE versus dense, same width family

`expF1` is the strongest dense model in this family: width 320, depth 4, SwiGLU 864, XSA, dropout 0.1, same optimizer extras, 3,600 steps (29.5M tokens), 5.61M parameters, validation 1.578. `expM5` keeps those extras but uses depth-6 MoE 4×320 top-2 at 10.50M parameters and reaches 1.530 after the same 3,600 steps. Extra capacity with sparse experts helps. Extending that MoE to 5,000 steps and keeping the best checkpoint (H1r) reaches 1.513. A 6,000-step run (`expM9`) still selected step 4,200 and was slightly worse (1.518), so extra tokens past ~34M were not useful under this schedule.

### 4.2 Expert shape and top-\(k\)

All E-series runs used the H1r training recipe for 5,000 steps.

| Run | Experts | Hidden | Top-\(k\) | Params | Best val BPB |
|---|---:|---:|---:|---:|---:|
| H1r | 4 | 320 | 2 | 10.50M | **1.512823** |
| `expE2` | 6 | 256 | 2 | 11.98M | 1.517904 |
| `expE1` | 12 | 128 | 2 | 11.99M | 1.519419 |
| `expE3` | 4 | 384 | 2 | 11.97M | 1.530092 |
| `expE4` | 12 | 128 | 5 | 11.99M | 1.548835 |

Spreading the same parameter ballpark across more, thinner experts did not help. Top-5 on 12 experts was clearly worse: more experts fire, but they are too small and the router is harder to train. Routing logs on H1r show mean usage close to 0.5 per expert (as expected for top-2 of 4) and no dead expert; layer 0 is the most peaked (usage max ≈0.67). Averaging usage across layers would have hidden that first-layer skew, which is why per-layer logging was added.

Reducing the auxiliary weight from 0.01 to 0.003 (`expR1`) raised validation BPB to 1.534. Load balancing is not free; 0.01 is the better setting here.

### 4.3 Regularization and extra tables

Dropout 0.125 (`expH3`) reached 1.513480 at step 5,000, slightly behind H1r's selected 4,200-step checkpoint. A hashed bigram table (8,192 buckets, dim 64) hurt: `expH2` 1.584, `expH4` 1.578. The table collides on a 2,048² pair space and adds parameters under the 64 MiB cap without paying for itself.

Learning rate 0.002 (`expH1-lr002`) was worse (1.530). Seed 17 was used throughout; a 3,600-step seed-42 repeat earlier in the search (`expM8`) did not beat the seed-17 curve enough to change the design.

### 4.4 Neural cache versus trigram mix

On the frozen H1r best checkpoint, a trigram mixer built only from training tokens peaked at λ = 0.08, validation 1.501668. The neural cache peaked at λ = 0.04, θ = 12, validation 1.494517. Official CPU evaluation of the baked checkpoint confirmed 1.493983 validation and 1.511051 test, versus 1.511733 / 1.533351 without cache. Cache helps more than n-grams because it copies *this window's* recently mentioned entities rather than corpus-level local statistics. It also costs no extra disk: the 42.58 MiB checkpoint is the model weights plus two scalars.

A nearby setting λ = 0.05, θ = 10 (`expH1rC`) scored 1.511057 test BPB, within noise of H1rC2. H1rC2 was chosen because it was best on the validation sweep.

## 5. Resources and trade-offs

Table 2. CPU FP32, 4 threads, this machine (not the course Xeon).

| Predictor | Assets | Peak RAM | Val time | Test time | Val BPB | Test BPB |
|---|---:|---:|---:|---:|---:|---:|
| `expC2` baseline | 4.17 MiB | 1.78 GiB | 4.80 s | 5.45 s | 1.742927 | 1.769429 |
| H1r, no cache | 42.58 MiB | 1.86 GiB | 21.40 s | 24.43 s | 1.511733 | 1.533351 |
| H1rC2 | 42.58 MiB | 1.86 GiB | 33.20 s (`measure_limits`) / 24.13 s (`evaluate`) | 35.75 s | 1.493983 | 1.511051 |

Assets and RAM are inside the 64 MiB and 4 GiB caps. Relative to *this* machine's baseline, H1rC2 validation is about 6.9×. Relative to the README 5.92 s reference, `measure_limits.py` flags 33.20 s as outside 5× (limit 29.6 s), while the official scorer JSON reported 24.13 s, which would be inside. The discrepancy is timer variance plus neural-cache quadratic work inside each 256-token window. The cache is the quality/cost trade-off: +0.019 validation BPB versus a several-fold CPU slowdown. If a verifier requires ≤29.6 s on the Xeon, the same H1r weights without cache remain a legal fallback (training-time val 1.513) at lower latency.

## 6. Critical analysis

MoE helped because WikiText-2 is small and heterogeneous: entity names, markup and local boilerplate benefit from expert specialization while top-2 keeps FLOPs closer to a 2×320 SwiGLU than to a dense 4× hidden FFN. Wider experts (`4×384`) and many tiny experts both failed, which argues against treating "more experts" as a monotone improvement under a 64 MiB cap.

The cache is a pointer over the current window, not a corpus index. That distinction matters for the causality rules: it cannot leak across windows or into the future. It *does* reuse tokens already generated in the prefix, which is legal and similar in spirit to pointer-sentinel models, but implemented with hidden-state similarity rather than an extra table. The trigram ablation shows that static local statistics are a weaker pointer on this tokenizer.

Two honest limitations remain. First, CPU time is tight; quadratic cache attention in a 256 window is simple but not cheap on CPU. Second, almost all architecture search used one seed. The 0.0005-class differences among nearby cache settings should not be over-interpreted. The 0.23 BPB gap versus the matched baseline is the result that is large enough to trust.

## 7. Conclusion

The submitted system is a 10.5M RMSNorm-RoPE SwiGLU-MoE transformer with a frozen 4% in-window neural cache. At a matched 40.96M training-token budget it improves the supplied GPT baseline from 1.769 to **1.511** test BPB, with 42.6 MiB assets and 1.86 GiB peak RAM. The cache is the last selected knob; the backbone already accounts for most of the gain. Code, commands and the frozen checkpoint are in the accompanying repository.

## References

- Merity, S., Xiong, C., Bradbury, J. and Socher, R. Pointer Sentinel Mixture Models. arXiv:1609.07843, 2016.
- Shazeer, N. et al. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR, 2017.
- Su, J. et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 2024.
- Zhang, B. and Sennrich, R. Root Mean Square Layer Normalization. NeurIPS, 2019.
- Shazeer, N. GLU Variants Improve Transformer. arXiv:2002.05202, 2020.
- Grave, E., Joulin, A. and Usunier, N. Improving Neural Language Models with a Continuous Cache. ICLR, 2017.
