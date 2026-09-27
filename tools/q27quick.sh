#!/bin/bash
# q27quick.sh LABEL MODEL "spec args" -> greedy short suite on the 27B, summary line
LABEL=$1; MODEL=$2; SPEC=$3
cd ~/bench
export GGML_VK_ALLOW_GRAPHICS_QUEUE=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1
export BASE_ARGS="--ctx-size 131072 --n-gpu-layers 99 --fit off --flash-attn on --batch-size 4096 --ubatch-size 2048 --threads 8 --parallel 1 --jinja --reasoning on --no-reasoning-preserve --no-webui $SPEC"
MODEL=$MODEL bash serve.sh start $LABEL ${BIN:-$HOME/src/llama.cpp/build-vulkan-dv/bin} > /dev/null || { echo "$LABEL FAILED"; exit 1; }
ONLY_GREEDY=1 python3 bench.py http://127.0.0.1:8081 $LABEL 3 > /dev/null 2>&1
bash serve.sh stop > /dev/null
python3 summ.py $LABEL | grep -E "greedy|ms/step"
