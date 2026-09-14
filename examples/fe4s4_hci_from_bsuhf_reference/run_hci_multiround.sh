#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=small"
#PJM -g ra010014
#PJM -L "elapse=07:00:00"
#PJM --mpi "max-proc-per-node=16"
#PJM -j
#PJM -S

# Multi-round heat-bath CI at a FIXED cutoff, starting from the single BS-UHF Hartree-Fock
# determinant (per collaborator guidance: don't seed from SQD sample-pool determinants for the
# standard HCI reference). Each round's deduplicated --carryovername output becomes the next
# round's --detfiles; stops when the basis stops growing (converged at this cutoff) or the round
# budget/elapse limit is exhausted. See docs/reference/sbd_gdb_heatbath_and_selection_gap.md
# section 7 for real results from this script.
#
# Run from the directory holding gdb_diag_uhf, the merged FCIDUMP, and the HF seed detfile
# (see merge_bsuhf_to_uhf_fcidump.py / pool_to_gdb_detfile.py in this same directory for how
# those are produced). Adjust the cd target below for your deployment path.

cd "${GDB_HCI_DIR:?set GDB_HCI_DIR to the directory holding gdb_diag_uhf and the FCIDUMP}"

FCIDUMP=fe4s4_bsuhf.uhf.fcidump
HEATBATH_CUTOFF=1.0e-4
N_ROUNDS=6
LOGFILE=hci_multiround_timing.log

DETFILE=fe4s4_bsuhf_hf_seed.detfile.txt

echo "round,cutoff,n_det_in,n_det_out,energy,wall_seconds,davidson_seconds,heatbath_seconds" > "$LOGFILE"

JOB_START=$(date +%s)
ELAPSE_BUDGET_SECONDS=25200  # 7h PJM elapse limit; leave the script room to exit gracefully
                             # rather than being killed mid-round with no log line written.

for round in $(seq 0 $((N_ROUNDS - 1))); do
  ELAPSED_SO_FAR=$(( $(date +%s) - JOB_START ))
  if [ "$ELAPSED_SO_FAR" -ge "$((ELAPSE_BUDGET_SECONDS - 1800))" ]; then
    echo "=== stopping before round ${round}: ${ELAPSED_SO_FAR}s elapsed, within 30min of the ${ELAPSE_BUDGET_SECONDS}s budget ==="
    break
  fi
  N_IN=$(wc -l < "$DETFILE")
  CARRYOVER_PREFIX="carryover_round${round}.txt"
  echo "=== ROUND ${round}: cutoff=${HEATBATH_CUTOFF}, n_det_in=${N_IN} ==="

  T0=$(date +%s.%N)
  mpiexec -n 16 ./gdb_diag_uhf --fcidump "$FCIDUMP" \
      --detfiles "$DETFILE" \
      --do_redist_det 1 \
      --method 0 --block 10 --iteration 50 --tolerance 1.0e-5 \
      --b_comm_size 2 --t_comm_size 1 \
      --init 0 --shuffle 0 \
      --carryovername "$CARRYOVER_PREFIX" \
      --rdm 0 --carryover_type 2 --carryover_ratio 0.1 \
      --heatbath_cutoff "$HEATBATH_CUTOFF" --heatbath_truncation 0.0 --heatbath_batch_size 1000 \
      > "round${round}.stdout.log" 2>&1
  RC=$?
  T1=$(date +%s.%N)
  WALL=$(echo "$T1 - $T0" | bc)

  echo "=== ROUND ${round} exit code: ${RC}, wall=${WALL}s ==="

  # Real per-rank solver stdout lives under output.<jobid>/0/<b_rank>/stdout.<t>.<b> per the
  # PJM/plexec convention -- mpiexec's own redirected stdout above only captures pjsub's own
  # (usually empty) parent-launcher output, not the MPI ranks' prints. Find it by recency
  # (newest stdout* file created after T0) rather than relying on a possibly-wrong PJM env var
  # name for the job ID.
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

  # Deduplicate the sharded carryover output across all 16 ranks into one combined detfile for
  # the next round -- HeatbathExpansion's own sort_global_bitarray guarantees global uniqueness
  # WITHIN one run, but the per-rank shard files on disk are the b-distributed pieces of that
  # single global set, so a plain concatenation is already the correct (already-deduplicated)
  # union; sort -u here is just defensive/idempotent, not correcting a real duplication bug.
  NEXT_DETFILE="detfile_round$((round + 1)).txt"
  cat "${CARRYOVER_PREFIX}"*.txt 2>/dev/null | sort -u > "$NEXT_DETFILE"
  N_OUT=$(wc -l < "$NEXT_DETFILE")

  echo "${round},${HEATBATH_CUTOFF},${N_IN},${N_OUT},${ENERGY},${WALL},${DAVIDSON_S},${HEATBATH_S}" >> "$LOGFILE"
  echo "ROUND ${round} SUMMARY: n_det ${N_IN} -> ${N_OUT}, energy=${ENERGY}, wall=${WALL}s, davidson=${DAVIDSON_S}s, heatbath=${HEATBATH_S}s"

  if [ "$N_OUT" -eq "$N_IN" ]; then
    echo "=== basis unchanged (${N_OUT} dets) -- HCI converged, stopping early at round ${round} ==="
    break
  fi

  DETFILE="$NEXT_DETFILE"
done

echo "=== HCI MULTI-ROUND RUN COMPLETE ==="
cat "$LOGFILE"
