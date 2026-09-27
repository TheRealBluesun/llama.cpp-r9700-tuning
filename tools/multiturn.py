"""Multi-turn prefix-reuse test: long doc in turn 1, then follow-up questions.
Reports per turn: prompt tokens, tokens actually processed (prompt_n), reused (cache_n), TTFT.
Usage: multiturn.py URL DOC_KTOK [think 0/1]"""
import json, sys, time, urllib.request
URL = sys.argv[1]; K = int(sys.argv[2]); THINK = len(sys.argv) > 3 and sys.argv[3] == "1"
import os
doc = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "longtext110k.txt")).read()[:int(K * 1000 * 3.66)]
Q = ["Summarize the first part of the material in 5 bullets.", "What are the three most important claims made? Quote them briefly.",
     "Now write a short critique of the weakest argument.", "Give me a one-paragraph TL;DR.", "List any open questions it leaves."]
msgs = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"[session {time.time()}]\n" + doc + "\n\n" + Q[0]}]
for t in range(len(Q)):
    body = {"messages": msgs, "max_tokens": 1500 if THINK else 250, "temperature": 0.7, "stream": True,
            "chat_template_kwargs": {"enable_thinking": THINK}}
    t0 = time.time(); tf = None; tim = None; content = ""
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]": continue
            d = json.loads(line[5:]); tim = d.get("timings") or tim
            ch = d.get("choices") or []
            if ch:
                dl = ch[0].get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content"): tf = tf or time.time()
                content += dl.get("content") or ""
    print(f"turn {t}: prompt_n {tim['prompt_n']:7d}  cache_n {tim['cache_n']:7d}  ttft {tf - t0:6.2f}s", flush=True)
    msgs.append({"role": "assistant", "content": content})  # what a client sends back: no reasoning
    if t + 1 < len(Q): msgs.append({"role": "user", "content": Q[t + 1]})
