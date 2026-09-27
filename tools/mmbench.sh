#!/bin/bash
# mmbench.sh [BIN] -> us/run and effective GB/s for 27B-shaped small-n matmuls (4 x 5120x17408, cache-busting)
BIN=${BIN:-$HOME/src/llama.cpp/build-vulkan-dv/bin}
BPW="iq4_xs:4.25 q4_K:4.5 q5_K:5.5 q6_K:6.5625 q8_0:8.5"
for tb in $BPW; do t=${tb%%:*}; b=${tb##*:}
  printf "%-7s" $t
  GGML_VK_VISIBLE_DEVICES=1 $BIN/test-backend-ops perf -o MUL_MAT -b Vulkan0 -p "type_a=$t,type_b=f32,m=5120," 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep "k=17408,bs=\[4,1\]" | grep -oE "n=[0-9]+|[0-9.]+ us/run" | paste - - | \
    awk -v b=$b '{n=substr($1,3); us=$2; gbs=4*5120*17408*b/8/(us*1e3); printf "  n=%-2s %6.0fus %4.0fGB/s", n, us, gbs}'
  echo
done
