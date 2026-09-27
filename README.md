# llama.cpp tuning on a Radeon AI PRO R9700 (RDNA4, gfx1201)

Single-user llama.cpp serving on one **AMD Radeon AI PRO R9700** (32 GB, ~640 GB/s), Fedora 44,
Mesa 26.2 RADV, ROCm 7.1 (Fedora packages). Workload: chat + agent use with long contexts (up to 262K).

- **Phase 1 (done):** Qwen3.8-35B-A3B Q4_K_M (hybrid Gated-DeltaNet + attention MoE, ~3B active), MTP speculative decoding.
- **Phase 2 (next):** Qwen3.8-27B.

All changes are **lossless**: greedy output is byte-identical with and without each change.

## Results: Qwen3.8-35B-A3B Q4_K_M, MTP n_max=2

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

Base: llama.cpp `95887577a` (2026-09-26).
```
git -C llama.cpp checkout 95887577a && git -C llama.cpp am ../patches/*.patch
cmake -S llama.cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON
cmake --build build -j
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 build/bin/llama-server \
  -m model-dv98k.gguf -ngl 99 -fa on --spec-type draft-mtp --spec-draft-n-max 2 ...
```

## Tools

- `tools/bench.py`: short-prompt suite (code/prose/explain/json/short, greedy + sampled), 16K prefill, optional 110K deep test.
  Reads MTP acceptance from llama-server `timings`.
- `tools/depth.py`: cache-busting context-depth sweep (prefill + decode at 1K…179K).
- `tools/multiturn.py`: checks prefix reuse across chat turns.
- `tools/gen.py`: greedy outputs for losslessness checks.
- `tools/serve.sh`, `runcfg.sh`, `quick.sh`, `pf.sh`: start/stop a test server by PID and run the suites.
- `results/`: raw JSON from every run.

The harness expects two long plain-text files next to it (not included): `longtext.txt` (~16K tokens) and
`longtext110k.txt` (~110K tokens). Any long English prose works; `depth.py` doubles the latter for the 179K test.
