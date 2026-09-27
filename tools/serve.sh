#!/bin/bash
# serve.sh start LABEL BINDIR [extra llama-server args...]   -> server on :8081, pid in ~/bench/run/server.pid
# serve.sh stop
# Env: MODEL (default 35B-A3B Q4_K_M), DEV (default Vulkan1), BASE_ARGS override. Any other env (RADV_*, GGML_*) passes through.
set -u
RUN=~/bench/run; mkdir -p $RUN ~/bench/logs
stop() {
  if [ -f $RUN/server.pid ]; then
    P=$(cat $RUN/server.pid)
    if [ -d /proc/$P ] && grep -q llama-server /proc/$P/cmdline; then
      kill $P; for i in $(seq 60); do [ -d /proc/$P ] || break; sleep 0.5; done
      [ -d /proc/$P ] && kill -9 $P
    fi
    rm -f $RUN/server.pid
  fi
}
case "$1" in
stop) stop; echo stopped ;;
start)
  LABEL=$2; BIN=$3; shift 3
  stop
  MODEL=${MODEL:-$HOME/models/Qwen3.8-35B-A3B-Q4_K_M.gguf}
  DEV=${DEV:-Vulkan1}
  # his production flags (llama-qwen35b-a3b.service), minus api key / alias / host
  BASE=${BASE_ARGS:---ctx-size 262144 --n-gpu-layers 99 --fit off --load-mode mmap --flash-attn on --cache-type-k f16 --cache-type-v f16 --batch-size 4096 --ubatch-size 2048 --threads 8 --threads-batch 8 --parallel 1 --cache-ram 8192 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k f16 --spec-draft-type-v f16 --jinja --reasoning on --no-reasoning-preserve --no-webui}
  LOG=~/bench/logs/$(date +%H%M%S)-$LABEL.log
  echo "$BIN/llama-server --model $MODEL --device $DEV $BASE $* --host 127.0.0.1 --port 8081" > $LOG
  LD_LIBRARY_PATH=$BIN setsid nohup $BIN/llama-server --model $MODEL --device $DEV $BASE "$@" --host 127.0.0.1 --port 8081 >> $LOG 2>&1 < /dev/null &
  echo $! > $RUN/server.pid
  for i in $(seq 300); do
    curl -sf -m2 localhost:8081/health >/dev/null && { echo "UP after ${i}s  pid $(cat $RUN/server.pid)  vram $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used)/1048576 )) MiB  log $LOG"; exit 0; }
    [ -d /proc/$(cat $RUN/server.pid) ] || { echo "DIED"; tail -20 $LOG; exit 1; }
    sleep 1
  done; echo TIMEOUT; tail -20 $LOG; exit 1 ;;
esac
