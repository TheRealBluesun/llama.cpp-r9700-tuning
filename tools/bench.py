"""Engine-neutral benchmark for an OpenAI-compatible endpoint (llama-server or vLLM).
API key from $API_KEY. Uses llama-server `timings` (draft_n/draft_n_accepted) when present.
Streams each response and measures TTFT and decode tok/s from the stream itself
(decode = (completion_tokens - 1) / (t_last_chunk - t_first_chunk)).
Usage: bench27.py URL LABEL [reps]   -> results/<time>-<label>.json"""
import json, os, statistics as st, sys, time, urllib.request
URL, LABEL = sys.argv[1].rstrip("/"), sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
HERE = os.path.dirname(os.path.abspath(__file__))
HDR = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    HDR["Authorization"] = "Bearer " + os.environ["API_KEY"]
PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into total seconds, with unit tests.",
    "prose": "Write a 400-word essay on why the Roman Republic fell.",
    "explain": "Explain how a B-tree differs from a binary search tree and why databases use B-trees. Be thorough.",
    "json": "Return a JSON array describing five fictional users with fields id, name, email, signup_date and a nested 'preferences' object. Output only JSON.",
    "short": "What is the capital of Australia? Answer in one sentence.",
}
def drafts():
    """Speculative verify steps so far (vLLM /metrics); None if the server doesn't expose it."""
    try:
        for line in urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_drafts_total"):
                return float(line.rsplit(" ", 1)[1])
    except Exception:
        return None
def model_id():
    return json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/models", headers=HDR), timeout=30))["data"][0]["id"]
def stream(messages, max_tokens, sampled):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if not sampled:
        body["temperature"] = 0
    body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), HDR)
    d0 = drafts(); t0 = time.time(); t_first = t_last = None; usage = None; n_chunks = 0; tim = None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("timings"):
                tim = d["timings"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                now = time.time(); n_chunks += 1
                t_first = t_first or now; t_last = now
    ct = (usage or {}).get("completion_tokens", n_chunks)
    pt = (usage or {}).get("prompt_tokens")
    dec = (ct - 1) / (t_last - t_first) if t_first and t_last > t_first and ct > 1 else None
    d1 = drafts(); steps = (d1 - d0) if d0 is not None and d1 is not None else None
    acc = None
    if tim and tim.get("draft_n"):  # llama-server: target forward passes = accepted-draft-free tokens
        steps = ct - tim.get("draft_n_accepted", 0); acc = tim["draft_n_accepted"] / tim["draft_n"]
    ms_step = (t_last - t_first) * 1e3 / (steps - 1) if steps and steps > 1 and t_first and t_last > t_first else None
    return {"ttft": (t_first or time.time()) - t0, "decode_tps": dec, "completion_tokens": ct, "prompt_tokens": pt,
            "steps": steps, "ms_per_step": ms_step, "tok_per_step": ct / steps if steps else None,
            "accept": acc, "server_pp_tps": (tim or {}).get("prompt_per_second"), "server_tg_tps": (tim or {}).get("predicted_per_second")}
MODEL = model_id()
res = {"label": LABEL, "url": URL, "model": os.path.basename(MODEL), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
stream([{"role": "user", "content": "Say hi."}], 8, False)  # warm-up
for sampled in (() if os.environ.get('ONLY_LONG') == '1' else ((False,) if os.environ.get('ONLY_GREEDY') == '1' else (False, True))):
    for name, p in PROMPTS.items():
        for rep in range(REPS):
            x = stream([{"role": "user", "content": p}], 128 if name == "short" else 600, sampled)
            x.update(name=name, sampled=sampled, rep=rep); res["runs"].append(x)
            print(f"{'sampled' if sampled else 'greedy '} {name:8s} rep{rep}: {x['completion_tokens']:4d} tok  decode {x['decode_tps'] or 0:6.1f} tok/s  ttft {x['ttft']:.2f}s  {x['ms_per_step'] or 0:5.2f} ms/step  {x['tok_per_step'] or 0:4.2f} tok/step  acc {x['accept'] or 0:.2f}", flush=True)
# prefill: unique 16K-token-ish prompt per rep (defeats prefix caching)
base = open(os.path.join(HERE, "longtext.txt")).read()
for rep in range(0 if os.environ.get('ONLY_LONG') == '1' or os.environ.get('ONLY_GREEDY') == '1' else 2):
    txt = f"[run {time.time()}]\n" + base
    x = stream([{"role": "user", "content": txt + "\n\nSummarize the above in three bullet points."}], 64, False)
    x.update(name="prefill16k", sampled=False, rep=rep); res["runs"].append(x)
    print(f"prefill  {x['prompt_tokens']} prompt tok: ttft {x['ttft']:.2f}s -> {x['prompt_tokens']/x['ttft']:.0f} tok/s", flush=True)
if os.environ.get("LONG") == "1":
    base = open(os.path.join(HERE, "longtext110k.txt")).read()
    txt = f"[run {time.time()}]\n" + base
    x = stream([{"role": "user", "content": txt + "\n\nWrite a detailed summary of the material above (about 300 words)."}], 400, False)
    x.update(name="deep110k", sampled=False, rep=0); res["runs"].append(x)
    print(f"deep     {x['prompt_tokens']} prompt tok: ttft {x['ttft']:.1f}s -> prefill {x['prompt_tokens']/x['ttft']:.0f} tok/s | decode after it {x['decode_tps'] or 0:.1f} tok/s ({x['completion_tokens']} tok)", flush=True)
print("\nmedian decode tok/s:")
for sampled in (False, True):
    row = []
    for name in PROMPTS:
        v = [r["decode_tps"] for r in res["runs"] if r["name"] == name and r["sampled"] == sampled and r["decode_tps"]]
        row.append(f"{name} {st.median(v):.1f}" if v else f"{name} -")
    print(("  sampled " if sampled else "  greedy  ") + " | ".join(row))
ms = [r["ms_per_step"] for r in res["runs"] if r.get("ms_per_step") and r["name"] not in ("short", "prefill16k", "deep110k")]
tps = [r["tok_per_step"] for r in res["runs"] if r.get("tok_per_step") and r["name"] not in ("short", "prefill16k", "deep110k")]
if ms:
    print(f"  step time: median {st.median(ms):.2f} ms/step over {len(ms)} runs (IQR {sorted(ms)[len(ms)//4]:.2f}-{sorted(ms)[3*len(ms)//4]:.2f}); tokens/step median {st.median(tps):.2f}")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
fn = os.path.join(HERE, "results", time.strftime("%Y%m%d-%H%M%S") + f"-{LABEL}.json")
json.dump(res, open(fn, "w"), indent=1); print("saved", fn)
