import pathlib
p = pathlib.Path.home() / "src/llama.cpp/src/models/qwen35.cpp"; s = p.read_text()
def rep(old, new):
    global s; assert s.count(old) == 1, old[:60]; s = s.replace(old, new)
rep('''        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };''', '''        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);

        // optional reduced-vocab draft head (see graph_mtp); its size comes from the file
        if (const auto * meta = ml.get_tensor_meta(tn(LLM_TENSOR_NEXTN_DRAFT_IDS, "weight", il).str().c_str())) {
            const int64_t n_draft_vocab = meta->ne[0];
            layer.nextn.draft_head = create_tensor(tn(LLM_TENSOR_NEXTN_DRAFT_HEAD, "weight", il), { n_embd, n_draft_vocab }, mtp_flags|TENSOR_NOT_REQUIRED);
            layer.nextn.draft_ids  = create_tensor(tn(LLM_TENSOR_NEXTN_DRAFT_IDS,  "weight", il), { n_draft_vocab },         mtp_flags|TENSOR_NOT_REQUIRED);
        }
    };''')
rep('''    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);''', '''    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    static const bool draft_head_disabled = getenv("LLAMA_MTP_DRAFT_HEAD_DISABLE") != nullptr;
    if (layer.nextn.draft_head && layer.nextn.draft_ids && !head_s && !draft_head_disabled) {
        // reduced-vocab draft head, scattered into a full-vocab row (see qwen35moe.cpp)
        const int64_t n_vocab_full = head_w->ne[1];
        const int64_t n_sub        = layer.nextn.draft_head->ne[1];
        const int64_t n_out        = cur->ne[1];
        ggml_tensor * sub  = ggml_mul_mat(ctx0, layer.nextn.draft_head, cur);
        ggml_tensor * full = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab_full, n_out);
        full = ggml_fill(ctx0, full, -1e30f);
        full = ggml_set_rows(ctx0, full, ggml_reshape_3d(ctx0, sub, 1, n_sub, n_out), layer.nextn.draft_ids);
        cur  = ggml_reshape_2d(ctx0, full, n_vocab_full, n_out);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);''')
p.write_text(s); print("ok")
