#!/bin/bash
# quick.sh LABEL [extra args]  -> start new-vulkan server, greedy short suite, stop. Env passes through.
LABEL=$1; shift
cd ~/bench
bash serve.sh start $LABEL ${BIN:-$HOME/src/llama.cpp/build-vulkan/bin} "$@" > /dev/null || { echo "$LABEL FAILED TO START"; exit 1; }
ONLY_GREEDY=1 python3 bench.py http://127.0.0.1:8081 $LABEL 3 2>&1 | grep -E "greedy  code|step time" | sed "s/^/$LABEL: /"
bash serve.sh stop > /dev/null
