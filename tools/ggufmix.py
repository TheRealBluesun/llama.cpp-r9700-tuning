import gguf, sys, collections
for fn in sys.argv[1:]:
    r = gguf.GGUFReader(fn); by = collections.Counter(); tot = 0
    for t in r.tensors:
        if t.name.startswith("blk.") and ".nextn." not in t.name and "blk.64." not in t.name:
            by[t.tensor_type.name] += t.n_bytes; tot += t.n_bytes
    print(fn.split("/")[-1], f"{tot/1e9:.2f} GB layers:", ", ".join(f"{k} {v/tot*100:.0f}%" for k, v in by.most_common()))
