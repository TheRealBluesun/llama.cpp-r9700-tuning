"""Context-depth sweep: for each target depth, send a fresh (cache-busting) prompt of ~N tokens,
measure prefill (TTFT) and greedy decode of the answer after it.
Usage: depth.py URL LABEL [depths_k, default 1,8,32,64,128,200] -> results/<time>-depth-<label>.json"""
import json, os, sys, time, urllib.request
URL, LABEL = sys.argv[1].rstrip("/"), sys.argv[2]
DEPTHS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,8,32,64,128,200").split(",")]
HERE = os.path.dirname(os.path.abspath(__file__))
HDR = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    HDR["Authorization"] = "Bearer " + os.environ["API_KEY"]
TEXT = open(os.path.join(HERE, "longtext110k.txt")).read()
TEXT = TEXT + "\n\n" + TEXT  # ~220K tokens of material for the deepest runs
CPT = 3.85  # chars per token for this text (calibrated below from the first run)
MODEL = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/models", headers=HDR), timeout=30))["data"][0]["id"]
def run(ntok, max_tokens=300):
    body = {"model": MODEL, "max_tokens": max_tokens, "stream": True, "temperature": 0,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"[run {time.time()}]\n" + TEXT[:int(ntok * CPT)] +
                          "\n\nWrite a detailed summary of the material above (about 250 words)."}]}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), HDR)
    t0 = time.time(); tf = tl = None; usage = tim = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage; tim = d.get("timings") or tim
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                now = time.time(); tf = tf or now; tl = now
    ct, pt = usage["completion_tokens"], usage["prompt_tokens"]
    acc = tim["draft_n_accepted"] / tim["draft_n"] if tim and tim.get("draft_n") else None
    return {"target_k": ntok // 1000, "prompt_tokens": pt, "ttft": tf - t0, "prefill_tps": pt / (tf - t0),
            "completion_tokens": ct, "decode_tps": (ct - 1) / (tl - tf) if tl > tf else None, "accept": acc,
            "server_pp_tps": (tim or {}).get("prompt_per_second"), "server_tg_tps": (tim or {}).get("predicted_per_second")}
res = {"label": LABEL, "url": URL, "model": os.path.basename(MODEL), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
run(1000, 8)  # warm-up
for k in DEPTHS:
    x = run(k * 1000)
    if k == DEPTHS[0] and x["prompt_tokens"] > 500:  # recalibrate chars/token
        CPT = CPT * k * 1000 / x["prompt_tokens"]
    res["runs"].append(x)
    print(f"depth {x['prompt_tokens']:7d} tok: ttft {x['ttft']:7.2f}s  prefill {x['prefill_tps']:7.0f} tok/s | "
          f"decode {x['decode_tps'] or 0:6.1f} tok/s ({x['completion_tokens']} tok)  acc {x['accept'] or 0:.2f}", flush=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
fn = os.path.join(HERE, "results", time.strftime("%Y%m%d-%H%M%S") + f"-depth-{LABEL}.json")
json.dump(res, open(fn, "w"), indent=1); print("saved", fn)
