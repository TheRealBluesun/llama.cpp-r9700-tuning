# llama.cpp tuning on a Radeon AI PRO R9700 (RDNA4, gfx1201)

Single-user llama.cpp serving on one **AMD Radeon AI PRO R9700** (32 GB, ~640 GB/s), Fedora 44,
Mesa 26.2 RADV, ROCm 7.1 (Fedora packages). Workload: chat + agent use with long contexts (up to 262K).

- **Phase 1:** Qwen3.8-35B-A3B Q4_K_M (hybrid Gated-DeltaNet + attention MoE, ~3B active), MTP speculative decoding.
  **+23–27% decode.**
- See PR-MAP.md for how the patches map to upstream PRs.
- **Phase 2:** Qwen3.8-27B (dense hybrid). Unsloth UD-Q4_K_XL + MTP (n=3) + draft head + RDNA4 mat-vec tuning:
  **2.4–2.7× decode on code/JSON, 1.6–2.0× on prose/explanations, 1.9× at 100K context** vs stock llama.cpp without speculation.

**No approximations beyond floating-point rounding.** Measured on greedy output (4 prompts, ~2K tokens each):

| Change | Greedy output vs without it |
|---|---|
| Reduced-vocab draft head (0001/0003), GDN fusion (0002), `GGML_VK_ALLOW_GRAPHICS_QUEUE` | byte-identical |
| `GGML_VK_DISABLE_GRAPH_OPTIMIZE` (op order/fusion), RDNA4 mat-vec tuning (0004: reduction order; q6_K verify via the standard q8_1-activation MMVQ path) | rounding-level: can flip a near-tie late in a long greedy generation |
| For reference: stock llama.cpp, speculative vs non-speculative decode; or llama.cpp 09-22 vs 09-26 | same kind of rounding-level divergence (2 of 4 prompts) |

The one real quality choice (which quantization to run) is measured separately with KL divergence against Q8_0.

## Results: Qwen3.8-27B (128K context, f16 KV)

Median decode tok/s (greedy / sampled at the server defaults; `tools/bench.py`) and the depth sweep (`tools/depth.py`):

| | code | prose | explain | json | decode at 1K / 30K / 57K / 103K |
|---|---|---|---|---|---|
| Stock llama.cpp Vulkan, UD-IQ4_XS, no speculation | 36 / 36 | 36 / 36 | 36 / 36 | 36 / 36 | 36 / 32 / 30 / 26 |
| **UD-Q4_K_XL, MTP n=3, this repo** | **88 / 79** | **57 / 56** | **71 / 71** | **96 / 96** | **76 / 68 / 61 / 49** |
| UD-IQ4_XS, MTP n=3, this repo (faster, lower quality) | 91 / 87 | 58 / 56 | 80 / 71 | 101 / 101 | 82 / 69 / 59 / 50 |

Prefill is ~1,000 tok/s up to 30K and ~700 tok/s at 100K (cold 100K prompt ≈ 145 s); speculation costs 5–10% of it.

### Why UD-Q4_K_XL (quality vs speed)

`llama-perplexity --kl-divergence` against Q8_0 (6 × 4,096 tokens of long prose), plus MTP-3 step time:

| Quant | GB | PPL / PPL(Q8_0) | mean KLD | 99% KLD | same top token | ms/step (MTP 3) |
|---|---|---|---|---|---|---|
| UD-IQ4_XS | 14.3 | 1.0113 | 0.0244 | 0.356 | 95.1% | 37.8 |
| **UD-Q4_K_XL** | 17.6 | **1.0034** | **0.0082** | **0.117** | **97.4%** | **40.2** |
| UD-Q5_K_M | 19.8 | 1.0013 | 0.0045 | 0.061 | 98.2% | 44.0 |
| UD-Q6_K | 22.0 | 1.0012 | 0.0026 | 0.039 | 98.6% | — |

UD-Q4_K_XL diverges 3× less than IQ4_XS for 6% more step time.

### What the 27B changes are

1. **Speculative decoding with the model's own MTP head** (`--spec-type draft-mtp --spec-draft-n-max 3`) plus the
   98K draft head (`patches/0003` ports patch 0001 to the dense `qwen35` graph; same tokenizer).
2. **RDNA4 mat-vec tuning for multi-token verify** (`patches/0004`). Verifying k drafted tokens runs every weight
   matrix against k+1 columns. On stock llama.cpp this stays at the memory-bandwidth limit up to 4 columns, then
   degrades: at 8 columns Q4_K/Q5_K/Q6_K ran at 368/425/294 GB/s (of ~640). RDNA3 already had a "4 rows per
   workgroup above 4 columns" rule; RDNA4 did not, and q6_K never used the integer-dot (MMVQ) path outside Intel.
   With both, 8 columns run at 570/590/595 GB/s (cache-busting 20480×17408 matrices, `tools/mmbench.sh`).
3. The Phase 1 settings apply unchanged (`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`, GDN fusion).

### Drafting options measured (UD-Q4_K_XL, greedy code / prose / explain / json, ms per step)

| Drafter | tok/s | ms/step |
|---|---|---|
| MTP n=2 | 75 / 57 / 65 / 79 | 37.4 |
| **MTP n=3** | **88 / 57 / 71 / 96** | **40.2** |
| MTP n=4 | 93 / 52 / 72 / 104 | 46.2 |
| DFlash2 (z-lab Q8_0 GGUF) n=3 | 86 / 58 / 68 / 93 | 41.5 |
| DFlash2 n=7 (z-lab's recommendation) | 104 / 47 / 71 / 139 | 53.5 |
| MTP with `--spec-draft-p-min` 0.5–0.8 (n=3–6) | never better on average than plain MTP 3 | |

DFlash2 guesses better (up to 7.4 tokens/step on JSON) but costs ~5–6 ms/step for its drafter, and its
acceptance collapses deep in long contexts (0.45 → 0.25 by 30K), so MTP n=3 is the default. DFlash2 n=7 is
the better choice for code/JSON-heavy short-context work.

### Measured and rejected (27B)

| Idea | Result |
|---|---|
| HIP/ROCm build (gfx1201) | decode −11% (no spec) / −26% (MTP 3); prefill ≈ |
| int8 coopmat MMQ for Q4_K/Q5_K on RDNA4 (excluded upstream) | slower (63 → 50 TF): exclusion confirmed |
| Q5_K_M / Q6_K | +10% / more step time for smaller quality gains |

### Where the time goes (27B, MTP 3, UD-Q4_K_XL)

- **Short context (40 ms/step):** weight GEMVs ~30 ms (~86% of bandwidth over 16.5 GB), MTP drafting ~5 ms
  (each draft pass reads ~0.76 GB), small ops ~3–4 ms. Realistic floor ≈ 30 ms.
- **100K context (~52 ms/step):** verify attention reads 6.5 GB of KV per step (64 KB/token) at ~86% of bandwidth.
  Only a smaller KV cache would help (quantized KV is slower on this backend).
- **Prefill:** weight GEMMs at 44–72 TFLOPS depending on the quant format (30–40% of matrix peak); the next big
  lever, and a kernel project.

## Results: Qwen3.8-35B-A3B Q4_K_M, MTP n_max=2 (Phase 1)

Short prompts, median decode tok/s (greedy; `tools/bench.py`):

| | code | prose | explain | json | ms/step |
|---|---|---|---|---|---|
| Baseline (stock llama.cpp Vulkan) | 196 | 164 | 179 | 211 | 13.63 |
| + `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | 219 | 179 | 205 | 240 | 12.10 |
| + 98K-token MTP draft head (patch 0001) | 226 | 191 | 219 | 253 | 11.51 |
| + `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` | 237 | 203 | 219 | 256 | 11.2–11.3 |
| + GDN+CPY fusion (patch 0002) | **241** | **209** | **225** | **266** | **10.9** |

Total: **+23–27% decode** on short prompts. At depth (`tools/depth.py`), decode after a 115K / 179K prompt
goes from 122 / 107 to 143 / 123 tok/s. Prefill is unchanged (179K cold prefill ≈ 106 s).

### What the changes are

1. **`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`** (env only): about −12% step time on RADV, with no prefill cost.
2. **Reduced-vocab MTP draft head** (`patches/0001`, `tools/add_draft_head.py`, `data/draft_vocab_chat96k.npy`).
   The MTP draft pass spent ~0.73 ms per draft token on the full 248K-vocab Q6_K LM head. An optional
   `blk.N.nextn.draft_head` (the head rows for the 98,320 most frequent chat tokens, copied byte-for-byte)
   plus `blk.N.nextn.draft_ids` are added to the GGUF. The draft computes logits only over that subset and
   scatters them into a full-vocab row filled with −1e30, so the sampler and the verify step are unchanged
   (the target still verifies with the full head). The vocab list comes from Qwen3.8-Flash-Next work; the
   tokenizers are identical. `LLAMA_MTP_DRAFT_HEAD_DISABLE=1` turns it off.
   ```
   python tools/add_draft_head.py model.gguf model-dv98k.gguf data/draft_vocab_chat96k.npy
   ```
3. **`GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`** (env only): the node reordering hurts slightly on this model.
4. **GDN + snapshot-CPY fusion** (`patches/0002`, Vulkan). With speculative decoding, each Gated-DeltaNet
   layer wrote up to K state snapshots into its own output, and a separate CPY moved them into the
   recurrent cache (30 layers × several MB per step). The GDN shader now writes them straight into the cache.
   `GGML_VK_DISABLE_GDN_CPY_FUSION=1` turns it off.

### Measured and rejected

| Idea | Result |
|---|---|
| HIP/ROCm + rocWMMA FA (built for gfx1201) | decode −20–28% vs Vulkan (prefill ≈) |
| Newer llama.cpp (4 days) | no change |
| `-b/-ub` other than 4096/2048 | −3–7% prefill |
| COMPUTE power profile | no change |
| MTP n_max 3 / 4 | code/json +5–10%, prose −10–20% |
| 64K draft vocab | slightly faster step, acceptance loss on code/short answers |
| MTP draft attention window (mask-only, 1K/4K) | acceptance 0.68 → 0.55, decode −7–20% |
| q8_0 KV cache | decode −9–11% at 115K+, prefill −4–8% (quantized-KV FA path is slower) |
| FA coopmat1 knobs (subgroups 2/4/8, shmem staging) | no gain |

### Where the time goes now

- **Decode (short context):** MoE expert GEMVs ~4 ms/step (~85% of bandwidth), full LM head for verify
  ~0.8 ms, and about 1,100 small dispatches (norms/adds/activations, ~0.1–0.25 ms per category).
- **Decode (deep context):** flash attention reads ~2.4 GB of KV per step at 116K, at ~79% of bandwidth.
- **Prefill (long):** coopmat1 flash attention is ~2/3 of a 179K prefill, running at ~37 TFLOPS
  (the plain f16 Vulkan GEMM reaches 71 TFLOPS; the HIP FA kernel gets 34). Multi-turn prefix reuse works
  (follow-up turns prefill only the new ~100–300 tokens), so this only affects cold long prompts.

## Using it

The patches are maintained as branch **`r9700`** of the fork https://github.com/TheRealBluesun/llama.cpp
(base: upstream `95887577a`, 2026-09-26). `patches/` here holds the same commits as files.
```
git clone -b r9700 https://github.com/TheRealBluesun/llama.cpp
cmake -S llama.cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON
cmake --build build -j
# 35B-A3B
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 build/bin/llama-server \
  -m Qwen3.8-35B-A3B-Q4_K_M-dv98k.gguf -ngl 99 -fa on --spec-type draft-mtp --spec-draft-n-max 2 ...
# 27B
python tools/add_draft_head.py Qwen3.8-27B-UD-Q4_K_XL.gguf Qwen3.8-27B-UD-Q4_K_XL-dv98k.gguf data/draft_vocab_chat96k.npy
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 build/bin/llama-server \
  -m Qwen3.8-27B-UD-Q4_K_XL-dv98k.gguf -ngl 99 -fa on -c 131072 --spec-type draft-mtp --spec-draft-n-max 3 ...
```

## Tools

- `tools/bench.py`: short-prompt suite (code/prose/explain/json/short, greedy + sampled), 16K prefill, optional 110K deep test.
  Reads MTP acceptance from llama-server `timings`.
- `tools/depth.py`: cache-busting context-depth sweep (prefill + decode at 1K…179K).
- `tools/multiturn.py`: checks prefix reuse across chat turns.
- `tools/gen.py`: greedy outputs for losslessness checks.
- `tools/serve.sh`, `runcfg.sh`, `quick.sh`, `pf.sh`, `run27.sh`, `q27quick.sh`: start/stop a test server by PID and run the suites.
- `tools/summ.py`: summarize results; `tools/mmbench.sh`, `mmbench_up.sh`: cache-busting small-N mat-vec bandwidth
  (needs patches 0004/0005 for the test shapes); `tools/ggufmix.py`: bytes per quant type in a GGUF.
- `results/`: raw JSON from every run.

The harness expects two long plain-text files next to it (not included): `longtext.txt` (~16K tokens) and
`longtext110k.txt` (~110K tokens). Any long English prose works; `depth.py` doubles the latter for the 179K test.
