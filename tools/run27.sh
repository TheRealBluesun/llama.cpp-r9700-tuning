#!/bin/bash
# run27.sh LABEL BIN "extra args" [depths]  -> short suite + depth sweep for Qwen3.8-27B configs
LABEL=$1; BIN=$2; EXTRA=$3; DEPTHS=${4:-1,8,32,64,115}
export BASE_ARGS="--ctx-size 131072 --n-gpu-layers 99 --fit off --flash-attn on --cache-type-k f16 --cache-type-v f16 --batch-size 4096 --ubatch-size 2048 --threads 8 --threads-batch 8 --parallel 1 --cache-ram 8192 --jinja --reasoning on --no-reasoning-preserve --no-webui $EXTRA"
bash serve.sh start $LABEL $BIN || exit 1
python3 bench.py http://127.0.0.1:8081 $LABEL 3 2>&1 | tail -5
python3 depth.py http://127.0.0.1:8081 $LABEL $DEPTHS 2>&1 | grep depth
bash serve.sh stop
