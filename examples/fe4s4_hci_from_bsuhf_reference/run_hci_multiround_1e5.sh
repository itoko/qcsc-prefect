#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=small"
#PJM -g ra010014
#PJM -L "elapse=08:00:00"
#PJM --mpi "max-proc-per-node=16"
#PJM -j
#PJM -S

# Multi-node, checkpoint/restart-capable continuation of run_hci_multiround.sh at a tighter
# heat-bath cutoff (1e-5), seeded from the LAST detfile the 1e-4 run produced (converged basis
# at that cutoff, or the last round it reached before its own elapse budget cut it off).
#
# Scale-out: 2 nodes x 16 ranks/node = 32 total MPI ranks (vs the 1e-4 run's 1 node x 16). The
# real parallel dimension for a growing determinant basis is b_comm_size, since
# h_comm_size = mpi_size / (t_comm_size * b_comm_size) and b_comm shards the basis/Davidson
# vector across ranks -- t_comm_size stays 1 (unused dimension here), b_comm_size=8 matches
# Stage 1's already-validated value exactly (h_comm_size=32/(8*1)=4, vs Stage 1's h_comm_size=8
# -- h_comm is just a replica/history dimension derived as whatever's left over, not a
# physics-correctness-sensitive parameter).
#
# Picked 2 nodes, not 16: the 16-node/256-rank config was tried first and showed a real PJM
# queue estimate of ~1 week out (rscgrp=small is apparently heavily contended at that node
# count) -- cancelled before it ran. rscgrp=large REJECTED node=4 outright with "node=4 is less
# than the lower limit (385)" -- large has a hard 385-node floor on this system. 2 nodes is the
# smallest real multi-node step (validates MPI works across node boundaries at all) while
# staying inside whatever range of rscgrp=small actually schedules promptly.
#
# Checkpoint/restart: --savename after every round persists the actual Davidson wavefunction
# (not just the determinant list) via sbd::SaveWavefunction; a resubmitted job detects the last
# completed round's saved wavefunction + detfile and passes it back in via --loadname, so
# Davidson warm-starts from the previous solution instead of from scratch. This also means a
# job killed mid-round by the elapse limit loses at most that one round's work, not the whole
# run -- and the SAME resubmit command works whether the previous job finished cleanly, was
# killed by walltime, or died with a real error (heatbath OOM, MPI fault, etc.) as long as at
# least one prior round finished.
#
# Verified against the sbd source (caop/basic/restart.h) that --loadname is safe across a
# growing/reordered basis: LoadWavefunction does a per-determinant lower_bound lookup against
# the CURRENT basis, carries over the coefficient for every determinant present in both, leaves
# newly-added determinants at a zero initial guess, and re-normalizes -- exactly the
# round-to-round situation here. One real constraint: the save file is sharded by mpi_rank_b
# (statefilename(name, mpi_rank_b)), so --loadname is only valid if --b_comm_size matches
# between the save and load runs -- kept fixed at 8 throughout this stage for that reason.

cd "${GDB_HCI_DIR:?set GDB_HCI_DIR to the directory holding gdb_diag_uhf and the FCIDUMP}"

FCIDUMP=fe4s4_bsuhf.uhf.fcidump
HEATBATH_CUTOFF=1.0e-5
N_ROUNDS_TOTAL=10          # overall stop condition across ALL resubmissions, not per-job
LOGFILE=hci_multiround_timing_1e5.log
STATEFILE=hci_1e5_state.txt   # "round_completed,detfile,wavefile" of the last finished round
SELF_SCRIPT="$(readlink -f "$0")"
NODE_COUNT=2
MPI_RANKS_PER_NODE=16
TOTAL_RANKS=$((NODE_COUNT * MPI_RANKS_PER_NODE))
B_COMM_SIZE=8
T_COMM_SIZE=1

if [ -f "$STATEFILE" ]; then
  IFS=',' read -r LAST_ROUND DETFILE LOADWAVE < "$STATEFILE"
  START_ROUND=$((LAST_ROUND + 1))
  echo "=== RESUMING from state file: last completed round ${LAST_ROUND}, detfile=${DETFILE}, wavefile=${LOADWAVE} ==="
else
  # First invocation: seed from the LAST detfile the 1e-4 run produced.
  DETFILE=$(ls -v detfile_round*.txt 2>/dev/null | grep -v '_1e5_' | tail -1)
  if [ -z "$DETFILE" ]; then
    echo "ERROR: no detfile_round*.txt found -- did the 1e-4 run produce any rounds?"
    exit 1
  fi
  LAST_ROUND=$(echo "$DETFILE" | grep -oP 'detfile_round\K[0-9]+')
  START_ROUND=$((LAST_ROUND + 1))
  LOADWAVE=""
  echo "round,cutoff,n_det_in,n_det_out,energy,wall_seconds,davidson_seconds,heatbath_seconds,nodes,total_ranks,b_comm_size" > "$LOGFILE"
  echo "=== FRESH START seeding 1e-5 from ${DETFILE} (round ${LAST_ROUND} of the 1e-4 run), ${NODE_COUNT} nodes / ${TOTAL_RANKS} ranks ==="
fi

JOB_START=$(date +%s)
ELAPSE_BUDGET_SECONDS=28800   # 8h PJM elapse limit; leave room to exit + self-resubmit cleanly
                               # rather than being killed mid-round with no state saved.

for round in $(seq "$START_ROUND" $((N_ROUNDS_TOTAL - 1))); do
  ELAPSED_SO_FAR=$(( $(date +%s) - JOB_START ))
  if [ "$ELAPSED_SO_FAR" -ge "$((ELAPSE_BUDGET_SECONDS - 2400))" ]; then
    echo "=== stopping before round ${round}: ${ELAPSED_SO_FAR}s elapsed, within 40min of the ${ELAPSE_BUDGET_SECONDS}s budget -- self-resubmitting ==="
    # -x PJM_LLIO_GFSCACHE=/vol0002 is required on this filesystem (the vol0002-hosted data
    # area) -- a plain `pjsub "$SELF_SCRIPT"` is rejected with "The current directory is
    # contained in /vol0002, so set PJM_LLIO_GFSCACHE to /vol0002." confirmed by a real pjsub
    # call when first submitting this script.
    pjsub -x PJM_LLIO_GFSCACHE=/vol0002 "$SELF_SCRIPT"
    echo "=== resubmitted $SELF_SCRIPT, exiting this job ==="
    exit 0
  fi

  N_IN=$(wc -l < "$DETFILE")
  CARRYOVER_PREFIX="carryover_1e5_round${round}.txt"
  SAVEWAVE="wavefunction_1e5_round${round}.dat"
  echo "=== ROUND ${round}: cutoff=${HEATBATH_CUTOFF}, n_det_in=${N_IN}, nodes=${NODE_COUNT}, ranks=${TOTAL_RANKS} ==="

  LOAD_ARGS=()
  if [ -n "$LOADWAVE" ] && [ -f "$LOADWAVE" ]; then
    LOAD_ARGS=(--loadname "$LOADWAVE")
    echo "=== warm-starting Davidson from ${LOADWAVE} ==="
  fi

  T0=$(date +%s.%N)
  mpiexec -n "$TOTAL_RANKS" ./gdb_diag_uhf --fcidump "$FCIDUMP" \
      --detfiles "$DETFILE" \
      --do_redist_det 1 \
      --method 0 --block 10 --iteration 50 --tolerance 1.0e-5 \
      --b_comm_size "$B_COMM_SIZE" --t_comm_size "$T_COMM_SIZE" \
      --init 0 --shuffle 0 \
      "${LOAD_ARGS[@]}" \
      --savename "$SAVEWAVE" \
      --carryovername "$CARRYOVER_PREFIX" \
      --rdm 0 --carryover_type 2 --carryover_ratio 0.1 \
      --heatbath_cutoff "$HEATBATH_CUTOFF" --heatbath_truncation 0.0 --heatbath_batch_size 1000 \
      > "round${round}_1e5.stdout.log" 2>&1
  RC=$?
  T1=$(date +%s.%N)
  WALL=$(echo "$T1 - $T0" | bc)

  echo "=== ROUND ${round} exit code: ${RC}, wall=${WALL}s ==="
  if [ "$RC" -ne 0 ]; then
    echo "=== ROUND ${round} FAILED (exit ${RC}) -- NOT advancing state file, so a resubmit retries this same round. See round${round}_1e5.stdout.log ==="
    exit 1
  fi

  RANK_STDOUT=$(find output.* -name "stdout*" -newermt "@${T0%.*}" 2>/dev/null | head -1)
  if [ -z "$RANK_STDOUT" ]; then
    echo "WARNING: could not find per-rank stdout for round ${round}, skipping timing extraction"
    ENERGY="NaN"
    DAVIDSON_S="NaN"
    HEATBATH_S="NaN"
  else
    ENERGY=$(grep -oP "sbd: Energy = \K[-0-9.eE+]+" "$RANK_STDOUT" | tail -1)
    DAVIDSON_S=$(grep -oP "end davidson \[Elapsed time \K[0-9.eE+-]+" "$RANK_STDOUT" | tail -1)
    HEATBATH_S=$(grep -oP "end heatbath expansion \[Elapsed time \K[0-9.eE+-]+" "$RANK_STDOUT" | tail -1)
  fi

  NEXT_DETFILE="detfile_1e5_round$((round + 1)).txt"
  cat "${CARRYOVER_PREFIX}"*.txt 2>/dev/null | sort -u > "$NEXT_DETFILE"
  N_OUT=$(wc -l < "$NEXT_DETFILE")

  echo "${round},${HEATBATH_CUTOFF},${N_IN},${N_OUT},${ENERGY},${WALL},${DAVIDSON_S},${HEATBATH_S},${NODE_COUNT},${TOTAL_RANKS},${B_COMM_SIZE}" >> "$LOGFILE"
  echo "ROUND ${round} SUMMARY: n_det ${N_IN} -> ${N_OUT}, energy=${ENERGY}, wall=${WALL}s, davidson=${DAVIDSON_S}s, heatbath=${HEATBATH_S}s"

  # Advance the state file only after a fully successful round -- this is the resume point.
  echo "${round},${NEXT_DETFILE},${SAVEWAVE}" > "$STATEFILE"

  if [ "$N_OUT" -eq "$N_IN" ]; then
    echo "=== basis unchanged (${N_OUT} dets) -- HCI converged at 1e-5, stopping for good at round ${round} ==="
    echo "=== HCI 1e-5 MULTI-ROUND RUN COMPLETE (converged) ==="
    cat "$LOGFILE"
    exit 0
  fi

  DETFILE="$NEXT_DETFILE"
  LOADWAVE="$SAVEWAVE"
done

echo "=== HCI 1e-5 MULTI-ROUND RUN COMPLETE (round budget ${N_ROUNDS_TOTAL} exhausted) ==="
cat "$LOGFILE"
