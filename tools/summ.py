"""summ.py LABEL... -> median decode per prompt type, prefill16k, and the depth sweep for each label."""
import json, glob, sys, statistics as st
for lab in sys.argv[1:]:
    fs = sorted(f for f in glob.glob(f"results/*-{lab}.json") if "-depth-" not in f)
    if fs:
        d = json.load(open(fs[-1]))
        for s in (False, True):
            row = []
            for n in ["code", "prose", "explain", "json", "short"]:
                v = [r["decode_tps"] for r in d["runs"] if r["name"] == n and r["sampled"] == s and r["decode_tps"]]
                a = [r["tok_per_step"] for r in d["runs"] if r["name"] == n and r["sampled"] == s and r.get("tok_per_step")]
                row.append((f"{n} {st.median(v):.1f}" + (f" ({st.median(a):.1f})" if a else "")) if v else f"{n} -")
            print(f"{lab:22s} {'sampled' if s else 'greedy '}  " + " | ".join(row))
        ms = [r["ms_per_step"] for r in d["runs"] if r.get("ms_per_step") and r["name"] not in ("short", "prefill16k", "deep110k")]
        if ms:
            print(f"{lab:22s} ms/step median {st.median(ms):.2f}")
    fs = sorted(glob.glob(f"results/*-depth-{lab}.json"))
    if fs:
        d = json.load(open(fs[-1]))
        print(f"{lab:22s} depth: " + "  ".join(f"{r['prompt_tokens']//1000}K pp {r['prefill_tps']:.0f}/tg {r['decode_tps']:.1f}" for r in d["runs"]))
