import re, pathlib
R = pathlib.Path.home() / "src/llama.cpp"
def edit(rel, old, new, count=1):
    p = R / rel; s = p.read_text()
    assert s.count(old) == count, (rel, old[:60], s.count(old))
    p.write_text(s.replace(old, new)); print("edited", rel)

edit("src/llama-arch.h", "    LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,\n",
     "    LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,\n    LLM_TENSOR_NEXTN_DRAFT_HEAD,\n    LLM_TENSOR_NEXTN_DRAFT_IDS,\n")
edit("src/llama-arch.cpp", '    { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,                 "blk.%d.nextn.shared_head_head" },\n',
     '    { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,                 "blk.%d.nextn.shared_head_head" },\n'
     '    { LLM_TENSOR_NEXTN_DRAFT_HEAD,                       "blk.%d.nextn.draft_head" },\n'
     '    { LLM_TENSOR_NEXTN_DRAFT_IDS,                        "blk.%d.nextn.draft_ids" },\n')
edit("src/llama-arch.cpp", "    {LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},\n",
     "    {LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},\n"
     "    {LLM_TENSOR_NEXTN_DRAFT_HEAD,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},\n"
     "    {LLM_TENSOR_NEXTN_DRAFT_IDS,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_NONE}},\n")
edit("src/llama-model.h", "    struct ggml_tensor * shared_head_norm      = nullptr;\n};",
     "    struct ggml_tensor * shared_head_norm      = nullptr;\n"
     "    // optional draft-only LM head over a token subset (rows of the output head) + their token ids\n"
     "    struct ggml_tensor * draft_head            = nullptr;\n"
     "    struct ggml_tensor * draft_ids             = nullptr;\n};")
edit("src/models/qwen35moe.cpp",
     '        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);\n    };',
     '        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);\n'
     '\n'
     '        // optional reduced-vocab draft head (see graph_mtp); its size comes from the file\n'
     '        if (const auto * meta = ml.get_tensor_meta(tn(LLM_TENSOR_NEXTN_DRAFT_IDS, "weight", il).str().c_str())) {\n'
     '            const int64_t n_draft_vocab = meta->ne[0];\n'
     '            layer.nextn.draft_head = create_tensor(tn(LLM_TENSOR_NEXTN_DRAFT_HEAD, "weight", il), { n_embd, n_draft_vocab }, mtp_flags|TENSOR_NOT_REQUIRED);\n'
     '            layer.nextn.draft_ids  = create_tensor(tn(LLM_TENSOR_NEXTN_DRAFT_IDS,  "weight", il), { n_draft_vocab },         mtp_flags|TENSOR_NOT_REQUIRED);\n'
     '        }\n    };')
edit("src/models/qwen35moe.cpp",
     "    cur = build_lora_mm(head_w, cur, head_s);\n    cb(cur, \"result_output\", -1);\n\n    res->t_logits = cur;\n    ggml_build_forward_expand(gf, cur);\n}",
     "    static const bool draft_head_disabled = getenv(\"LLAMA_MTP_DRAFT_HEAD_DISABLE\") != nullptr;\n"
     "    if (layer.nextn.draft_head && layer.nextn.draft_ids && !head_s && !draft_head_disabled) {\n"
     "        // Draft-only LM head over a frequent-token subset: the drafter only needs the top\n"
     "        // candidates, and the full head is ~40% of a draft step. Logits for the subset are\n"
     "        // scattered into a full-vocab row filled with a large negative value, so the draft\n"
     "        // sampler and everything downstream are unchanged. The target verifies with the full head.\n"
     "        const int64_t n_vocab_full = head_w->ne[1];\n"
     "        const int64_t n_sub        = layer.nextn.draft_head->ne[1];\n"
     "        const int64_t n_out        = cur->ne[1];\n"
     "        ggml_tensor * sub  = ggml_mul_mat(ctx0, layer.nextn.draft_head, cur);            // [n_sub, n_out]\n"
     "        ggml_tensor * full = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab_full, n_out);\n"
     "        full = ggml_fill(ctx0, full, -1e30f);\n"
     "        full = ggml_set_rows(ctx0, full, ggml_reshape_3d(ctx0, sub, 1, n_sub, n_out), layer.nextn.draft_ids);\n"
     "        cur  = ggml_reshape_2d(ctx0, full, n_vocab_full, n_out);\n"
     "    } else {\n"
     "        cur = build_lora_mm(head_w, cur, head_s);\n"
     "    }\n"
     "    cb(cur, \"result_output\", -1);\n\n    res->t_logits = cur;\n    ggml_build_forward_expand(gf, cur);\n}")
