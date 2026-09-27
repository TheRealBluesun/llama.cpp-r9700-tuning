"""Copy a GGUF and add an MTP draft-vocab head:
  blk.<L>.nextn.draft_head.weight  = rows `ids` of output.weight (raw copy, same quant type)
  blk.<L>.nextn.draft_ids.weight   = I32 token ids (sorted)
Usage: add_draft_head.py IN.gguf OUT.gguf ids.npy"""
import sys, numpy as np, gguf
src, dst, idf = sys.argv[1:4]
ids = np.unique(np.load(idf).astype(np.int64)).astype(np.int32)
r = gguf.GGUFReader(src)
arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
n_layer = int(r.fields[f"{arch}.block_count"].parts[-1][0])
nextn = int(r.fields[f"{arch}.nextn_predict_layers"].parts[-1][0]) if f"{arch}.nextn_predict_layers" in r.fields else 0
L = n_layer - nextn  # index of the MTP block
assert any(t.name.startswith(f"blk.{L}.nextn.") for t in r.tensors), f"no nextn tensors at blk.{L}"
w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
w.data_alignment = r.alignment
for f in r.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture":
        continue
    vt = f.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        sub = f.types[-1]
        if sub == gguf.GGUFValueType.STRING:
            val = [bytes(f.parts[i]) for i in f.data]
        else:
            val = [f.parts[i][0].item() for i in f.data]
        w.add_key_value(f.name, val, vt, sub_type=sub)
    elif vt == gguf.GGUFValueType.STRING:
        w.add_key_value(f.name, bytes(f.parts[-1]), vt)
    else:
        w.add_key_value(f.name, f.parts[-1][0].item(), vt)
head = None
for t in r.tensors:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    if t.name == "output.weight":
        head = t
assert head is not None
sub = np.ascontiguousarray(head.data[ids])  # quantized rows are self-contained byte runs
w.add_tensor_info(f"blk.{L}.nextn.draft_head.weight", sub.shape, sub.dtype, sub.nbytes, head.tensor_type)
w.add_tensor_info(f"blk.{L}.nextn.draft_ids.weight", ids.shape, ids.dtype, ids.nbytes, gguf.GGMLQuantizationType.I32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in r.tensors:
    w.write_tensor_data(t.data)
w.write_tensor_data(sub); w.write_tensor_data(ids)
w.close()
print(f"wrote {dst}: draft head {len(ids)} rows ({sub.nbytes/2**20:.1f} MiB, {head.tensor_type.name}) at blk.{L}")
