#!/bin/bash
# Measure the per-GPU slowdown factor k for whatever card D0 is running on, vs the H200 anchor.
#
# Why this works without a synthetic benchmark: D0 (b787_dense_g48_e006) uses --granularity 48, and
# the g48/n1800 H200 anchor from the 2026-07-21 handoff (936 s/epoch on 1 H200) has the SAME voxel
# count -- 110,592. --extent changes the box, not the number of voxels, and cost scales with voxels.
# So k = (D0 s/epoch) / 936, directly comparable, on the real workload rather than a proxy.
#
# Usage: scripts/measure_l40s_k.sh [job-name]      (default rift_b787_dense_g48)
set -u
ANCHOR_S_PER_EPOCH=936
NAME=${1:-rift_b787_dense_g48}
cd /storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT || exit 1

jid=$(squeue -u "$USER" -h -n "$NAME" -o '%i' | head -1)
state=RUNNING
if [ -z "$jid" ]; then
  jid=$(sacct -u "$USER" -n -X --name="$NAME" -o JobID -S 2026-07-26 | tail -1 | tr -d ' ')
  state=$(sacct -j "$jid" -n -X -o State%20 2>/dev/null | head -1 | tr -d ' ')
fi
[ -z "$jid" ] && { echo "no job named $NAME"; exit 1; }

rep="Report-${jid}.out"
[ -f "$rep" ] || { echo "no report $rep yet"; exit 1; }

card=$(sacct -j "$jid" -n -X -o AllocTRES%80 2>/dev/null | grep -oE 'gres/gpu:[a-z0-9_]+=[0-9]+' | head -1)
elapsed=$(squeue -j "$jid" -h -o '%M' 2>/dev/null)
[ -z "$elapsed" ] && elapsed=$(sacct -j "$jid" -n -X -o Elapsed 2>/dev/null | head -1 | tr -d ' ')
secs=$(awk -F: '{n=NF; s=0; m=1; for(i=n;i>0;i--){s+=$i*m; m*= (i==n?60:(i==n-1?60:24))} print s}' <<<"$elapsed")
epochs=$(grep -cE '^Epoch \[[0-9]+/[0-9]+\] Training Summary:' "$rep" 2>/dev/null)

echo "job          : $jid ($NAME, $state)"
echo "allocation   : ${card:-unknown}"
echo "elapsed      : $elapsed (${secs}s)"
echo "epochs done  : $epochs"
if [ "${epochs:-0}" -lt 1 ]; then echo "-> no completed epoch yet; re-run later"; exit 0; fi
# Startup (2.29 GB npz load + bp-init) is charged to epoch 1, so drop it when we can.
if [ "$epochs" -ge 3 ]; then
  e1=$(( epochs - 1 ))
  echo "-> using epochs 2..$epochs (excludes one-off startup charged to epoch 1)"
  awk -v s="$secs" -v e="$epochs" -v e1="$e1" -v a="$ANCHOR_S_PER_EPOCH" 'BEGIN{
    spe_all=s/e; spe=s/e;                       # conservative: whole-run average
    printf "s/epoch (avg) : %.0f\n", spe_all;
    printf "k vs H200     : %.2f   (anchor %d s/epoch)\n", spe_all/a, a;
    printf "D1 150 epochs : %.0f h at this rate\n", spe_all*150/3600;
  }'
else
  echo "-> only $epochs epoch(s); estimate INCLUDES startup, treat as an upper bound on k"
  awk -v s="$secs" -v e="$epochs" -v a="$ANCHOR_S_PER_EPOCH" 'BEGIN{
    printf "s/epoch (incl startup): %.0f\n", s/e;
    printf "k upper bound        : %.2f\n", (s/e)/a;
  }'
fi
echo
echo "Handoff action: if k < ~2, 6x L40S is oversized for D1 (predicted k was 3.0);"
echo "if k > ~4, 6 GPUs is undersized -- raise the count or fall back to h100."
echo "Report the s/epoch line back so CARDS['l40s'] can be set k_measured=True."
