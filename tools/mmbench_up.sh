#!/bin/bash
# ffn_up-shaped (short k=5120) small-n bench: 69632x5120 (4 x 17408 rows)
BIN=${BIN:-$HOME/src/llama.cpp/build-vulkan-dv/bin}
for tb in iq4_xs:4.25 q4_K:4.5 q5_K:5.5 q6_K:6.5625 q8_0:8.5; do t=${tb%%:*}; b=${tb##*:}
  printf "%-7s" $t
  GGML_VK_VISIBLE_DEVICES=1 $BIN/test-backend-ops perf -o MUL_MAT -b Vulkan0 -p "type_a=$t,type_b=f32,m=69632," 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep "k=5120,bs=\[1,1\]" | grep -oE "n=[0-9]+|[0-9.]+ us/run" | paste - - | \
    awk -v b=$b '{n=substr($1,3); us=$2; gbs=69632*5120*b/8/(us*1e3); printf "  n=%-2s %6.0fus %4.0fGB/s", n, us, gbs}'
  echo
done
