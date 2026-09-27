#!/bin/bash
# pf.sh LABEL DEPTHS  (BASE_ARGS/env pass through) -> depth sweep only
LABEL=$1; DEPTHS=$2
cd ~/bench
bash serve.sh start $LABEL ${BIN:-$HOME/src/llama.cpp/build-vulkan/bin} > /dev/null || { echo "$LABEL FAILED"; tail -5 $(ls -t logs/*$LABEL.log|head -1); exit 1; }
echo "vram $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used)/1048576 )) MiB"
python3 depth.py http://127.0.0.1:8081 $LABEL $DEPTHS 2>&1 | grep depth | sed "s/^/$LABEL: /"
bash serve.sh stop > /dev/null
