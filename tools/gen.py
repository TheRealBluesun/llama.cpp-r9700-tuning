import json, sys, urllib.request
P = ["Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into total seconds, with unit tests.",
     "Write a 400-word essay on why the Roman Republic fell.",
     "Explain how a B-tree differs from a binary search tree and why databases use B-trees. Be thorough.",
     "Return a JSON array describing five fictional users with fields id, name, email, signup_date and a nested 'preferences' object. Output only JSON."]
out = []
for p in P:
    b = {"messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})))
    out.append(r["choices"][0]["message"]["content"])
json.dump(out, open(sys.argv[1], "w"))
