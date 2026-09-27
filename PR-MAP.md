# PR map

Branch `r9700` of https://github.com/TheRealBluesun/llama.cpp is a linear series on upstream `95887577a`. Each commit
carries a `PR-group:` trailer (and `Depends-on:` where needed), so a PR can be rebuilt with
`git log --grep 'PR-group: <tag>'` and cherry-picked onto upstream master. `patches/` holds the same commits as files.

| # | PR group | Commit | What | Evidence |
|---|---|---|---|---|
| 1 | `mtp-draft-vocab` | llama: optional reduced-vocab MTP draft head | Optional `blk.N.nextn.draft_head`/`draft_ids` tensors; the MTP draft pass computes logits over a frequent-token subset (qwen35, qwen35moe). Needs a GGUF produced by `tools/add_draft_head.py` (a gguf-py writer would be needed upstream). | README: 35B +5-7% decode; byte-identical greedy output |
| 2 | `vk-gdn-snapshot-fusion` | vulkan: fuse GATED_DELTA_NET with snapshot CPY | GDN writes speculative-decoding state snapshots straight into the recurrent cache. | README: -0.3 ms/step, +2-6%; byte-identical |
| 3 | `vk-rdna4-mmv` | vulkan: tune RDNA4 int-dot mat-vec and q6_K MMVQ | RDNA3's 4-rows-above-4-columns rule extended to RDNA4; MMVQ for q6_K on RDNA4 when n > 1. | README: 8-column verify 370 -> 590 GB/s; MUL_MAT suite 1130/1130 |
| 4 | `vk-rdna4-cm1-symbias` | vulkan: magic bias on the RDNA4 symmetric cm1 epilogue | Existing magic-bias epilogue enabled on RDNA4 for the symmetric int8 coopmat MMQ types. | +3.5% prefill GEMM for q4_0/q8_0/iq4_xs/iq4_nl |
| 5 | `spec-draft-ctx-size` | speculative: size block-draft context to one verify block | DFlash/DSpark draft context n_batch/n_ubatch = one verify block (its compute buffer 2.76 GiB -> 9 MiB). | -2.7 GiB VRAM, no speed change |
| 6 | `spec-multi-drafter` (depends on 5) | speculative: second MTP context, per-type n-max, and DFlash gating | A draft-model drafter (DFlash) and MTP in one server; `--spec-mtp-n-max`; DFlash declines past `--spec-dflash-ctx-max` or when its acceptance EMA < `--spec-dflash-min-acc` (probe every `--spec-dflash-probe` steps). | 27B: JSON +44%, code +12%, geomean +6.4% vs MTP-only |
| - | `local` | 4 commits | Dev knobs (`GGML_VK_MMV_ROWS_N`, `GGML_VK_RDNA4_CM1_ALL`), speculative timing stats, extra test-backend-ops cases. Not for upstream. | |

Upstream llama.cpp requires the human submitter to write PR descriptions and replies themselves and to disclose AI
assistance; see their CONTRIBUTING.md / AGENTS.md.
