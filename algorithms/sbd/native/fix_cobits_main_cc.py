#!/usr/bin/env python3
"""Fix the undefined `cobits` in main.cc's SBD_PREFECT carryover block.

The SBD_PREFECT block references `cobits`, which is declared NOWHERE in the file (verified
across every branch in the tree: uses=3, decls=0 on all of them). The real carryover data
lives in two separate per-spin vectors, `co_adet` and `co_bdet` -- the same ones the
plain-text `--carryover_adetfile` / `--carryover_bdetfile` writers just above use.

This is why the binary could not be rebuilt natively: the working Aug-2 binaries were
compiled from an uncommitted local edit that the storage migration discarded.

Our Python side (solver_job.py::_read_carryover_bin + _run_sbd_inner) reads BOTH
`carryover.bin` (alpha) and `carryover_b.bin` (beta), so writing only one file would
silently degrade UHF carryover to alpha-only. The byte format is kept bit-identical to the
original loop, which already matches the reader's contract exactly:
    bytes_per_config = (norb + 7) // 8, np.unpackbits(bitorder="big")[:, :norb]
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()

# Match the whole stale block: the count printout through the ofs_co_bin.close().
start = src.index('    std::cout << "Number of carryover determinants: "')
end_marker = "    ofs_co_bin.close();\n"
end = src.index(end_marker, start) + len(end_marker)
old = src[start:end]

if "cobits" not in old:
    print("no-op: block does not reference cobits (already fixed?)")
    sys.exit(0)

new = '''    // Write per-spin carryover determinants as packed bitstrings.
    // NOTE: the previous version of this block referenced an undefined `cobits` and wrote a
    // single file; the real data is in co_adet / co_bdet (see the carryover_adetfile /
    // carryover_bdetfile writers above). Our Python reader
    // (solver_job.py::_read_carryover_bin) consumes carryover.bin AND carryover_b.bin, so
    // both spins must be emitted or UHF carryover silently degrades to alpha-only.
    // Byte layout is unchanged and matches the reader:
    //   bytes_per_config = (L + 7) / 8, big-endian bit order, first L bits significant.
    {
      const size_t bytes_per_config = (L + 7) / 8;
      std::vector<uint8_t> bytes(bytes_per_config);
      // co_adet/co_bdet are sbd::det_vector<size_t, sbd::det_kind::half>; take them by
      // reference via a pointer array so the element type never has to be spelled out.
      const char* co_names[2] = {"carryover.bin", "carryover_b.bin"};
      const sbd::det_vector<size_t, sbd::det_kind::half>* co_lists[2] = {&co_adet, &co_bdet};
      for (int s = 0; s < 2; ++s) {
        const auto& co = *co_lists[s];
        std::cout << "Number of carryover determinants (" << co_names[s] << "): "
                  << co.size() << std::endl;
        std::ofstream ofs_co_bin(co_names[s], std::ios::binary);
        for (size_t i = 0; i < co.size(); ++i) {
          std::fill(bytes.begin(), bytes.end(), 0);
          for (size_t j = 0; j < L; ++j) {
            size_t rev_idx = L - 1 - j;                 // sbd::makestring order
            size_t pw = rev_idx % sbd_data.bit_length;  // position in word
            size_t bw = rev_idx / sbd_data.bit_length;  // index of word
            bool bit = (co[i][bw] >> pw) & 1ULL;
            size_t pb = 7 - (j % 8);                    // big-endian bit order
            size_t bb = j / 8;                          // index of byte
            bytes[bb] |= static_cast<uint8_t>(bit << pb);
          }
          ofs_co_bin.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        ofs_co_bin.close();
      }
    }
'''

path.write_text(src[:start] + new + src[end:])
print("patched: cobits -> co_adet/co_bdet, writes carryover.bin + carryover_b.bin")
