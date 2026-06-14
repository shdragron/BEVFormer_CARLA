#!/usr/bin/env bash
# PD-BEV CTS (cross-platform transfer): the SEDAN384 model evaluated on the
# suv/bus TARGET eval sets under 4 input conditions, normalized by the target
# native oracle (paper Eq.6: CTS_c = NDS(cond) / P_TARGET).
#   NORMAL = sedan img + sedan ext   EXT = sedan img + target ext
#   IMG    = target img + sedan ext  CAL = target img + target ext (full deploy)
# P_TARGET oracles (already measured, native-384): suv 0.5464, bus 0.6064.
# 8 numerator runs (2 targets x 4 conds): build condition pkl -> dump (pdbev-b200,
# sedan384 model, arch 384/[1,100,1]) -> score against target GT (bevdet-b200).
set -o pipefail
GEN=/home/hanyan_arch/viewpoint/BEVFormer/Generalizable-BEV
BENCH=/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark
CB="$(conda info --base)"
CFG=configs/bevdet_our/bevdepth-r50-cbgs-pc-carla-sedan384.py
CK=work_dirs/pdbev-r50-cbgs-CARLA-dg-sedan384/epoch_24.pth
OUT=$BENCH/out/cts_pdbev_sedan384
PKLS=$OUT/pkls; DETS=$OUT/dets; mkdir -p "$PKLS" "$DETS"
TSV=$OUT/cts_nds.tsv; [ -f "$TSV" ] || echo -e "target\tcond\tnds\tmap" > "$TSV"
GPU=${GPU:-0}

declare -A ORACLE=( [suv]=0.5464 [bus]=0.6064 )

run_one () {   # $1 target  $2 cond
  local T=$1 C=$2
  local pk="$PKLS/${T}_${C}.pkl" dt="$DETS/${T}_${C}.pkl"
  grep -qaE "^$T\b.*\b$C\b" "$TSV" && { echo "[cts] $T $C already scored, skip"; return 0; }
  echo "[cts] ==== $T $C build pkl $(date +%H:%M:%S) ===="
  ( source "$CB/etc/profile.d/conda.sh"; conda activate pdbev-b200
    cd "$BENCH"
    python build_condition_pkls_bevdet.py --target "$T" --condition "$C" --out "$pk"
  ) || { echo "[cts] $T $C BUILD FAIL"; return 1; }
  echo "[cts] $T $C dump (sedan384 model) $(date +%H:%M:%S)"
  ( source "$CB/etc/profile.d/conda.sh"; conda activate pdbev-b200
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 NVIDIA_TF32_OVERRIDE=0 TMPDIR=/tmp CUDA_VISIBLE_DEVICES=$GPU
    cd "$GEN"
    python "$BENCH/pdbev_dump_val.py" "$CFG" "$CK" "$dt" --ann-file "$pk" --batch 16 --workers 6
  ) || { echo "[cts] $T $C DUMP FAIL"; return 1; }
  echo "[cts] $T $C score vs target GT $(date +%H:%M:%S)"
  local out
  out=$( ( source "$CB/etc/profile.d/conda.sh"; conda activate bevdet-b200
           export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
           python "$BENCH/pdbev_score_carla.py" "$dt" --vehicle "$T" ) 2>/dev/null )
  echo "$out" | grep -E 'PDBEV-CARLA|NDS_RESULT'
  local line; line=$(echo "$out" | grep '^NDS_RESULT')
  [ -n "$line" ] && echo -e "$T\t$C\t$(echo "$line"|awk '{print $2"\t"$3}')" >> "$TSV" \
    && rm -f "$dt"     # free the dets pkl after scoring
}

echo "[cts] ==== START $(date) ===="
for T in suv bus; do
  for C in NORMAL EXT IMG CAL; do
    run_one "$T" "$C"
  done
done
echo "[cts] ==== ALL DONE $(date) ===="

# ---- CTS table: CTS_c = NDS(c) / P_TARGET --------------------------------- #
python - "$TSV" <<'PY'
import sys, csv
oracle = {'suv': 0.5464, 'bus': 0.6064}
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
nds = {(r['target'], r['cond']): float(r['nds']) for r in rows}
print('\nCTS (PD-BEV sedan384 -> target)  CTS_c = NDS(c)/P_TARGET')
hdr = f'{"target":6} {"P_TARGET":>9} {"NORMAL":>8} {"EXT":>8} {"IMG":>8} {"CAL":>8}'
print(hdr)
for T in ('suv', 'bus'):
    P = oracle[T]
    cells = []
    for C in ('NORMAL', 'EXT', 'IMG', 'CAL'):
        v = nds.get((T, C))
        cells.append(f'{v/P:8.4f}' if v is not None else f'{"--":>8}')
    print(f'{T:6} {P:9.4f} ' + ' '.join(cells))
print('\nraw NDS:')
for T in ('suv', 'bus'):
    print(' ', T, {C: nds.get((T, C)) for C in ('NORMAL', 'EXT', 'IMG', 'CAL')})
PY
