"""Vulkan: fuse GATED_DELTA_NET (+view nodes) + CPY of its snapshot slots into the recurrent cache.
The GDN shader writes snapshots directly to the cache view instead of its own dst + a separate copy."""
import pathlib
R = pathlib.Path.home() / "src/llama.cpp/ggml/src/ggml-vulkan"
def edit(rel, old, new, count=1):
    p = R / rel; s = p.read_text()
    assert s.count(old) == count, (rel, old[:70], s.count(old))
    p.write_text(s.replace(old, new)); print("edited", rel)

# --- push constants
edit("ggml-vulkan-push-constants.h", """    float scale;
    uint32_t K;
};

struct vk_op_ssm_scan_push_constants {""", """    float scale;
    uint32_t K;
    // fused snapshot store (GDN + CPY): 0 = snapshots go to dst at s_off
    uint32_t so_fused;
    uint32_t so_off;          // element offset into the state-out buffer
    uint32_t so_seq_stride;   // elements between sequences
    uint32_t so_slot_stride;  // elements between snapshot slots
};

struct vk_op_ssm_scan_push_constants {""")

# --- shader
edit("vulkan-shaders/gated_delta_net.comp", """    float scale;
    uint K;
};""", """    float scale;
    uint K;
    uint so_fused;
    uint so_off;
    uint so_seq_stride;
    uint so_slot_stride;
};""")
edit("vulkan-shaders/gated_delta_net.comp", """layout(binding = 6)           buffer DstBuf   { FLOAT_TYPE data_dst[];   };""",
"""layout(binding = 6)           buffer DstBuf   { FLOAT_TYPE data_dst[];   };
layout(binding = 7)           buffer SOutBuf  { FLOAT_TYPE data_sout[];  };""")
edit("vulkan-shaders/gated_delta_net.comp", """            if (target_slot >= 0 && target_slot < int(K)) {
                const uint slot_base = s_off + uint(target_slot) * state_size_per_snap + state_out_base;
                [[unroll]] for (uint r = 0; r < ROWS_PER_LANE; r++) {
                    data_dst[slot_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
                }
            }""", """            if (target_slot >= 0 && target_slot < int(K)) {
                if (so_fused != 0u) {
                    // write straight into the recurrent cache (fused CPY)
                    const uint so_base = so_off + uint(target_slot) * so_slot_stride + seq_id * so_seq_stride + head_id * state_size;
                    [[unroll]] for (uint r = 0; r < ROWS_PER_LANE; r++) {
                        data_sout[so_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
                    }
                } else {
                    const uint slot_base = s_off + uint(target_slot) * state_size_per_snap + state_out_base;
                    [[unroll]] for (uint r = 0; r < ROWS_PER_LANE; r++) {
                        data_dst[slot_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
                    }
                }
            }""")

# --- pipeline: 8 bindings
edit("ggml-vulkan.cpp", """gdn_names[si][kda], gdn_len, gdn_data, "main", 7, sizeof(vk_op_gated_delta_net_push_constants),""",
     """gdn_names[si][kda], gdn_len, gdn_data, "main", 8, sizeof(vk_op_gated_delta_net_push_constants),""")

# --- dispatch
edit("ggml-vulkan.cpp", """void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {""",
     """void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, const ggml_tensor * fused_cpy = nullptr) {""")
edit("ggml-vulkan.cpp", """        scale,
        K
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
        pc, { H, n_seqs, S_v });""", """        scale,
        K,
        0u, 0u, 0u, 0u
    };

    vk_subbuffer sout_buf = dst_buf;
    if (fused_cpy != nullptr) {
        // fused_cpy is the CPY node: a strided view of the recurrent cache [D, n_seqs, n_written]
        sout_buf = ggml_vk_tensor_subbuffer(ctx, fused_cpy, true);
        const size_t full_off = vk_tensor_offset(fused_cpy) + fused_cpy->view_offs;
        const size_t misalign = full_off - sout_buf.offset;
        pc.so_fused       = 1u;
        pc.so_off         = (uint32_t)(misalign / sizeof(float));
        pc.so_seq_stride  = (uint32_t)(fused_cpy->nb[1] / sizeof(float));
        pc.so_slot_stride = (uint32_t)(fused_cpy->nb[2] / sizeof(float));
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf, sout_buf},
        pc, { H, n_seqs, S_v });""")
# pc was const
edit("ggml-vulkan.cpp", """    const float scale = 1.0f / sqrtf((float)S_v);
    const vk_op_gated_delta_net_push_constants pc = {""", """    const float scale = 1.0f / sqrtf((float)S_v);
    vk_op_gated_delta_net_push_constants pc = {""")
edit("ggml-vulkan.cpp", """    case GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, node);
""", """    case GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, node,
            ctx->num_additional_fused_ops ? cgraph->nodes[node_idx + ctx->num_additional_fused_ops] : nullptr);
""")

# --- fusion detection
edit("ggml-vulkan.cpp", """bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                           int node_idx) {""", """// GATED_DELTA_NET with K>1 snapshot slots, followed (possibly after view-only nodes) by the CPY of
// its snapshot region into the recurrent cache. Returns the number of additional nodes to fuse (0 = no).
static int ggml_vk_can_fuse_gdn_cpy(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {
    static const bool disabled = getenv("GGML_VK_DISABLE_GDN_CPY_FUSION") != nullptr;
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    if (disabled || gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32) {
        return 0;
    }
    const int64_t K = ggml_get_op_params_i32(gdn, 0);
    if (K <= 1) {
        return 0;
    }
    const ggml_tensor * v = gdn->src[2];
    const int64_t S_v = v->ne[0], H = v->ne[1], n_tokens = v->ne[2], n_seqs = v->ne[3];
    const int64_t D = S_v * S_v * H;
    const size_t  s_off_bytes = (size_t) (S_v * H * n_tokens * n_seqs) * sizeof(float);
    const size_t  snap_bytes  = (size_t) (S_v * S_v * H * n_seqs) * sizeof(float);
    for (int k = 1; k <= 4 && node_idx + k < cgraph->n_nodes; ++k) {
        const ggml_tensor * n = cgraph->nodes[node_idx + k];
        if (n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE ||
            n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE) {
            continue;
        }
        if (n->op != GGML_OP_CPY) {
            return 0;
        }
        const ggml_tensor * src = n->src[0];
        if (src->view_src != gdn || src->type != GGML_TYPE_F32 || n->type != GGML_TYPE_F32) {
            return 0;
        }
        // src: [D, n_seqs, n_written] view of the snapshot region, exactly as the shader lays it out
        if (src->view_offs != s_off_bytes || src->ne[0] != D || src->ne[1] != n_seqs ||
            src->ne[2] != std::min<int64_t>(n_tokens, K) || src->ne[3] != 1 ||
            src->nb[0] != sizeof(float) || src->nb[1] != (size_t) D * sizeof(float) || src->nb[2] != snap_bytes) {
            return 0;
        }
        // dst: rows of D contiguous floats
        if (n->ne[0] != D || n->ne[1] != n_seqs || n->ne[2] != src->ne[2] || n->nb[0] != sizeof(float) ||
            n->nb[1] % sizeof(float) || n->nb[2] % sizeof(float) || n->nb[2] / sizeof(float) > UINT32_MAX) {
            return 0;
        }
        GGML_UNUSED(ctx);
        return k;
    }
    return 0;
}

bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                           int node_idx) {""")
edit("ggml-vulkan.cpp", """            } else if (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2)) {""", """            } else if (int n_gdn = ggml_vk_can_fuse_gdn_cpy(ctx, cgraph, i)) {
                ctx->num_additional_fused_ops = n_gdn;
                ctx->fused_ops_write_mask |= 1;  // the GDN attention output is still written
                fusion_string = "GDN_CPY";
                std::fill_n(op_srcs_fused_elementwise, n_gdn + 1, false);
            } else if (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2)) {""")
