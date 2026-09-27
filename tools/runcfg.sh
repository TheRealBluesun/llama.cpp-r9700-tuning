#!/bin/bash
# runcfg.sh LABEL BINDIR [extra server args]  (env DEV, MODEL, BASE_ARGS, RADV_*/GGML_* pass through)
# start server -> short-context suite -> depth sweep -> stop
LABEL=$1; BIN=$2; shift 2
cd ~/bench
bash serve.sh start $LABEL $BIN "$@" || exit 1
python3 bench.py http://127.0.0.1:8081 $LABEL 3 2>&1 | tail -6
python3 depth.py http://127.0.0.1:8081 $LABEL 2>&1
bash serve.sh stop
