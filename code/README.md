# DASE7506 MP1 submission

Student model for the WikiText-2 raw BPB challenge (`7506-mp1-wt2-v2`).

Reported complete-test score: **1.511051 BPB** (CPU FP32).
Frozen predictor: `H1rC2`, checkpoint SHA-256 `cc3b4bfc717e23ccb0322e37e1b4696631edb529b4928598f67849713e9cb96d`.

This repository is a reproduction package. Search-stage configs, extra experiment checkpoints, virtual environments and caches are omitted.

## 1. Install

Use **Python 3.12**. From the extracted package:

```bash
cd code
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install PyTorch for **one** device:

```bash
# CPU
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU with a compatible driver, instead of the CPU wheel:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

Remaining dependencies and contract tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

No API key, pretrained weights or extra dataset download is required.

## 2. Final model download

The frozen H1rC2 checkpoint is about 43 MiB. Place it at:

```
code/runs/expH1rC2-w320-d6-h4-moe4x320-drop01-4200-cache004-theta12/checkpoint.pt
```

If the Git repository omits large `.pt` files, download the matching GitHub Release asset (`h1rc2-checkpoint.pt`) and keep SHA-256 `cc3b4bfc717e23ccb0322e37e1b4696631edb529b4928598f67849713e9cb96d`. After creating the Release, replace this sentence with the URL.

Expected files in that directory:

- `checkpoint.pt`
- `source.json`
- `validation_cpu_fp32.json`
- `validation_cpu_limits.json`
- `test_cpu_fp32.json`

The supplied `data/` splits and tokenizer must remain unchanged.

## 3. Train

All commands below run from `code/`. Training used an NVIDIA GeForce RTX 5070, BF16, seed 17, AdamW, EMA 0.99 and 5,000 updates × 32 × 256 tokens = 40,960,000 processed training targets. Cosine horizon equals 5,000 steps. Validation every 300 steps selected step 4,200 (34,406,400 tokens).

**Matched-budget baseline** (`expC2`, same 40.96M tokens):

```bash
python train.py --implementation model --config configs/baseline.json --device cuda --precision bf16 --seed 17 --steps 5000 --lr-horizon 5000 --eval-every 300 --run-dir runs/expC2-baseline-5000 --threads 4
```

**H1r backbone** (`expH1r`, no neural cache during training):

```bash
python train.py --implementation student --config configs/student_moe_w320_d6_h4_swiglu_xsa_dropout01.json --device cuda --precision bf16 --seed 17 --steps 5000 --lr-horizon 5000 --eval-every 300 --lr 0.001 --weight-decay 0.3 --ema-decay 0.99 --moe-aux-weight 0.01 --run-dir runs/expH1r-w320-d6-h4-moe4x320-drop01-5000 --threads 4
```

Validation-selected weights are written to `best_checkpoint.pt` in a fresh training run. This slim package keeps the H1r `metrics.json` and mix sweeps, and ships the baked H1rC2 checkpoint as the frozen predictor. Use a new `--run-dir` for each run.

**Optional cache-mix sweep** on the frozen H1r checkpoint, validation only:

```bash
python sweep_mix.py --checkpoint runs/expH1r-w320-d6-h4-moe4x320-drop01-5000/best_checkpoint.pt --mode cache --device cuda --precision fp32 --lambdas 0,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.1 --thetas 5,8,10,12,20 --output runs/expH1r-w320-d6-h4-moe4x320-drop01-5000/sweep_cache.json
```

The selected mix is λ = 0.04, θ = 12.

**Bake the frozen H1rC2 predictor** (no extra training):

```bash
python bake_cache.py --checkpoint runs/expH1r-w320-d6-h4-moe4x320-drop01-5000/best_checkpoint.pt --output runs/expH1rC2-w320-d6-h4-moe4x320-drop01-4200-cache004-theta12/checkpoint.pt --cache-mix 0.04 --cache-theta 12
```

Equivalent evaluation-time config: `configs/student_moe_w320_d6_h4_swiglu_xsa_dropout01_cache004_theta12.json`.

## 4. Validate, test and measure resources

Final H1rC2 evaluation:

```bash
python evaluate.py --checkpoint runs/expH1rC2-w320-d6-h4-moe4x320-drop01-4200-cache004-theta12/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4
python evaluate.py --checkpoint runs/expH1rC2-w320-d6-h4-moe4x320-drop01-4200-cache004-theta12/checkpoint.pt --device cpu --precision fp32 --split test --threads 4
python measure_limits.py --checkpoint runs/expH1rC2-w320-d6-h4-moe4x320-drop01-4200-cache004-theta12/checkpoint.pt --split validation --threads 4
```

Recorded H1rC2 results:

| Split / measurement | Value |
|---|---|
| Validation CPU FP32 BPB | 1.493983 |
| Test CPU FP32 BPB | **1.511051** |
| Validation CPU time | 33.20 s (this machine) / 24.13 s official scorer JSON |
| Peak working set | 1.86 GiB |
| Uncompressed checkpoint | 42.58 MiB |
| Course CPU 5× reference | 29.6 s (5.92 s × 5, README Xeon Platinum 8457C) |

`measure_limits.py` compares CPU time with the course README reference of 5.92 s, not this laptop's baseline. This machine scored the matched baseline in **5.45 s** test / **4.80 s** validation. H1rC2's 33 s validation therefore exceeds 5 × 5.92 s on the reference clock, but remains about 6.9× this machine's own baseline. RAM and asset-size limits are met.

Matched-budget baseline evaluation:

```bash
python evaluate.py --checkpoint runs/expC2-baseline-5000/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4
python evaluate.py --checkpoint runs/expC2-baseline-5000/checkpoint.pt --device cpu --precision fp32 --split test --threads 4
python measure_limits.py --checkpoint runs/expC2-baseline-5000/checkpoint.pt --split validation --threads 4
```

Recorded `expC2` results: validation 1.742927 BPB, test **1.769429 BPB**, 4.17 MiB assets, 1.78 GiB peak RAM, 4.80 s validation CPU.

H1r without cache (uses the validation-selected `best_checkpoint.pt`):

```bash
python evaluate.py --checkpoint runs/expH1r-w320-d6-h4-moe4x320-drop01-5000/best_checkpoint.pt --device cpu --precision fp32 --split validation --threads 4
python evaluate.py --checkpoint runs/expH1r-w320-d6-h4-moe4x320-drop01-5000/best_checkpoint.pt --device cpu --precision fp32 --split test --threads 4
python measure_limits.py --checkpoint runs/expH1r-w320-d6-h4-moe4x320-drop01-5000/best_checkpoint.pt --split validation --threads 4
```

Recorded H1r-without-cache CPU FP32 results (`best_checkpoint.pt`): validation 1.511733 BPB (21.40–22.87 s), test **1.533351 BPB** (24.43 s), 42.58 MiB assets, 1.86 GiB peak RAM, within the 5×/4 GiB/64 MiB caps. The cache mix is therefore worth about 0.022 test BPB (1.533 → 1.511) at extra CPU cost.

Submit the **bpb** field from `test_cpu_fp32.json`, not token perplexity.

## 5. Method in brief

- RMSNorm, RoPE, width 320, depth 6, 4 heads.
- Per-block SwiGLU MoE: 4 experts × hidden 320, top-2, load-balance auxiliary 0.01.
- Residual value-direction subtraction after attention (`xsa_projection`).
- Dropout 0.1, weight decay 0.3, EMA 0.99.
- Frozen in-window neural cache mix λ = 0.04, temperature θ = 12. Cache uses only earlier tokens in the same 256-token window.

## 6. Reproduction of a peer checkpoint

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

## 7. AI assistance

AI assistance is allowed by the course guide. Cursor (Grok) was used for:

- scaffolding training flags, logging, EMA/tied-embedding handling and MoE routing diagnostics
- drafting sweep/bake/resource-measurement scripts
- summarizing run logs and writing this README and the report
- assembling the slim submission layout

I chose the architecture, decided which experiments to run, inspected routing/cache results, selected H1rC2 on validation, and take responsibility for the implementation and reported score.

## 8. Data attribution

WikiText-2: Merity, Xiong, Bradbury and Socher, [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). Text by Wikipedia contributors. Upstream [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and [GNU FDL](https://www.gnu.org/licenses/fdl-1.3.html). Tokenizer fitted only to the supplied training split.
