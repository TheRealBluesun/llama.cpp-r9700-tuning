import pathlib
p = pathlib.Path.home() / "src/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp"; s = p.read_text()
def rep(old, new):
    global s; assert s.count(old) == 1, old[:70]; s = s.replace(old, new)
rep("""    // RDNA3: above four columns, static 4 rows for all types bench faster than the default
    const bool is_rdna3 = device->vendor_id == VK_VENDOR_ID_AMD && device->architecture == AMD_RDNA3;""",
"""    // RDNA3: above four columns, static 4 rows for all types bench faster than the default
    const bool is_rdna3 = device->vendor_id == VK_VENDOR_ID_AMD && device->architecture == AMD_RDNA3;
    // RDNA4 (gfx1201): same rule for the int-dot path; at 8 columns K-quants go from ~370-425 to ~570-595 GB/s
    const bool is_rdna4 = device->vendor_id == VK_VENDOR_ID_AMD && device->architecture == AMD_RDNA4;""")
rep("""((is_rdna3 && i >= 4) ? 4u : rows); };""", """(((is_rdna3 || is_rdna4) && i >= 4) ? 4u : rows); };""")
rep("""    bool mmvq_q6 = device->vendor_id == VK_VENDOR_ID_INTEL;""",
"""    bool mmvq_q6 = device->vendor_id == VK_VENDOR_ID_INTEL;
    // RDNA4: for batches (speculative verify) MMVQ q6_K stays at ~95% of bandwidth up to 8 columns
    // (the float path drops to ~46% at 8 columns); at n == 1 they tie.
    if (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture == vk_device_architecture::AMD_RDNA4 && n > 1) {
        mmvq_q6 = true;
    }""")
p.write_text(s); print("ok")
