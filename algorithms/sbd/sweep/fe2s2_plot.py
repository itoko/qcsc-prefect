"""Presentation-quality plots comparing RHF vs UHF across configuration-recovery iterations.

    python fe2s2_plot.py                 # both methods + refs.json, writes PNG/SVG/CSV

Reads runs/<mol>_<method>/recover/*.json (the recovery_trace written by fe2s2_recover.py) and
runs/refs.json (flat reference energies), and produces, in runs/<mol>_post/:

  fe2s2_energy_vs_iter.{png,svg}   -- E_SQD vs recovery iteration, RHF & UHF, flat ref lines
  fe2s2_panels.{png,svg}           -- 2x2: energy, ΔE vs DMRG, net subspace dim, spin/useful-frac
  energies.csv                     -- method, iter, E, dE_vs_dmrg_mHa, net_dim, sum_2Sz, useful_frac

Color follows the dataviz method: two categorical series in fixed order --
UHF = blue (#2a78d6), RHF = orange (#eb6834) -- the maximally separated slots. Reference lines
are gray/dashed with DIRECT labels (identity never by color alone). One y-axis per panel (no dual
axis): the mHa error lives in its own panel, not a second scale.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import fe2s2_common as C
import matplotlib

matplotlib.use("Agg")  # headless: set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

# --- Validated palette (dataviz references/palette.md, light surface) --------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = {"uhf": "#2a78d6", "rhf": "#eb6834"}  # slot 1 (blue), slot 8 (orange)
LABEL = {"uhf": "UHF", "rhf": "RHF"}
MARKER = {"uhf": "o", "rhf": "s"}
# Pretty molecule titles (matplotlib mathtext). Falls back to the raw id if unknown.
MOL_TITLE = {
    "fe2s2": "Fe$_2$S$_2$ (30e, 20o)",
    "fe4s4": "Fe$_4$S$_4$ (54e, 36o)",
}


def _mol_title() -> str:
    return MOL_TITLE.get(C.MOLECULE, C.MOLECULE)
# Reference flat lines: distinct dashes + direct labels (never color-alone).
REF_STYLE = {
    "DMRG": dict(color="#0b0b0b", ls="-", lw=1.6),      # near-exact = solid dark
    "CCSD(T)": dict(color="#52514e", ls="--", lw=1.4),
    "UCCSD": dict(color="#898781", ls="-.", lw=1.4),
    "HCI": dict(color="#4a3aa7", ls=":", lw=1.6),        # violet, distinct from series
    "UHF": dict(color="#c3c2b7", ls=(0, (1, 1)), lw=1.2),
    "RHF": dict(color="#c3c2b7", ls=(0, (1, 1)), lw=1.2),
}


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def load_traces() -> dict[str, list[dict]]:
    """method -> recovery_trace (deepest-recovery run wins if several present)."""
    out: dict[str, list[dict]] = {}
    for method in C.METHODS:
        rec_dir = C.run_dir(method) / "recover"
        best: list[dict] = []
        if rec_dir.is_dir():
            for jf in sorted(rec_dir.glob("*.json")):
                data = json.loads(jf.read_text())
                trace = data.get("recovery_trace") or []
                if len(trace) > len(best):
                    best = trace
        if best:
            out[method] = best
    return out


def load_refs() -> dict[str, float]:
    p = C.refs_path()
    return json.loads(p.read_text()) if p.is_file() else {}


def _iters(trace: list[dict]) -> list[int]:
    return [t["step"] + 1 for t in trace]  # 1-based for display


def plot_energy(traces, refs, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    all_iters: list[int] = []
    for method, trace in traces.items():
        xs, ys = _iters(trace), [t["energy"] for t in trace]
        all_iters += xs
        ax.plot(xs, ys, marker=MARKER[method], ms=6, lw=2, color=SERIES[method],
                label=LABEL[method], zorder=5, markeredgecolor=SURFACE, markeredgewidth=1)
        # direct label at the last point
        ax.annotate(LABEL[method], (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), color=SERIES[method], fontsize=10, fontweight="bold",
                    va="center")

    xmax = max(all_iters) if all_iters else 1
    # Place reference labels at the LEFT edge and nudge them apart vertically so nearby refs
    # (e.g. HCI/DMRG) don't overprint each other or collide with the top-right legend.
    ax.set_xlim(right=xmax + 0.6)
    ref_items = sorted(
        ((n, e) for n, e in refs.items() if n in REF_STYLE), key=lambda kv: kv[1]
    )
    ymin, ymax = ax.get_ylim()
    min_gap = (ymax - ymin) * 0.035
    last_y = None
    for name, e in ref_items:
        ax.axhline(e, zorder=2, **REF_STYLE[name])
        y = e if last_y is None else max(e, last_y + min_gap)
        ax.annotate(name, (0.012, y), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 2),
                    color=REF_STYLE[name]["color"], fontsize=8, va="bottom",
                    fontfamily="monospace", zorder=6)
        last_y = y

    ax.set_xlabel("Configuration-recovery iteration")
    ax.set_ylabel("Energy  (Ha)")
    ax.set_title(f"{_mol_title()}: SQD energy vs recovery iteration",
                 color=INK, fontsize=12, pad=12)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fe2s2_energy_vs_iter.{ext}", facecolor=SURFACE,
                    bbox_inches="tight")
    plt.close(fig)


def plot_panels(traces, refs, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    (ax_e, ax_de), (ax_dim, ax_spin) = axes
    for ax in axes.flat:
        _style_axes(ax)

    dmrg = refs.get("DMRG")

    for method, trace in traces.items():
        xs = _iters(trace)
        col, lab, mk = SERIES[method], LABEL[method], MARKER[method]
        common = dict(marker=mk, ms=5, lw=2, color=col, label=lab,
                      markeredgecolor=SURFACE, markeredgewidth=1, zorder=5)
        ax_e.plot(xs, [t["energy"] for t in trace], **common)
        if dmrg is not None:
            ax_de.plot(xs, [(t["energy"] - dmrg) * 1000 for t in trace], **common)
        ax_dim.plot(xs, [t["net_dim"] for t in trace], **common)
        ax_spin.plot(xs, [t["sum_2Sz"] for t in trace], **common)

    # energy panel + refs
    for name, e in refs.items():
        if name in REF_STYLE:
            ax_e.axhline(e, zorder=2, **REF_STYLE[name])
    ax_e.set_title("Energy (Ha)", color=INK, fontsize=11)
    ax_e.set_xlabel("recovery iteration")
    ax_e.legend(frameon=False, fontsize=8, loc="upper right")

    ax_de.axhline(0, color=INK, lw=1.0, zorder=2)
    ax_de.annotate("DMRG", (0.98, 0), xycoords=("axes fraction", "data"),
                   xytext=(0, 3), textcoords="offset points", fontsize=8,
                   color=INK, ha="right", fontfamily="monospace")
    ax_de.set_title("Error vs DMRG (mHa)", color=INK, fontsize=11)
    ax_de.set_xlabel("recovery iteration")

    ax_dim.set_title("Net subspace dimension", color=INK, fontsize=11)
    ax_dim.set_xlabel("recovery iteration")
    ax_dim.legend(frameon=False, fontsize=8, loc="lower right")

    ax_spin.axhline(0, color=GRID, lw=1.0, zorder=1)
    ax_spin.set_title(r"Spin density $\sum(n_\alpha-n_\beta)=2S_z$", color=INK, fontsize=11)
    ax_spin.set_xlabel("recovery iteration")
    ax_spin.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(f"{_mol_title()} RHF vs UHF — multi-iteration recovery diagnostics",
                 color=INK, fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fe2s2_panels.{ext}", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_csv(traces, refs, out_dir: Path) -> None:
    dmrg = refs.get("DMRG")
    with open(out_dir / "energies.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "iter", "energy_Ha", "dE_vs_dmrg_mHa", "net_dim",
                    "n_post", "sum_2Sz", "useful_frac"])
        for method, trace in traces.items():
            for t in trace:
                de = (t["energy"] - dmrg) * 1000 if dmrg is not None else ""
                w.writerow([method, t["step"] + 1, f"{t['energy']:.8f}",
                            f"{de:.3f}" if de != "" else "",
                            t["net_dim"], t.get("n_post", ""),
                            f"{t['sum_2Sz']:.4f}", f"{t.get('useful_frac', 0):.6f}"])


def main() -> None:
    traces = load_traces()
    if not traces:
        raise SystemExit(
            "No recovery traces found. Run fe2s2_recover.py --method {uhf,rhf} first."
        )
    refs = load_refs()
    out_dir = C.combined_post_dir()
    plot_energy(traces, refs, out_dir)
    plot_panels(traces, refs, out_dir)
    write_csv(traces, refs, out_dir)
    # Mirror into each method's own post/ for convenience.
    for method in traces:
        pdir = C.run_subdirs(method)["post"]
        for f in ("fe2s2_energy_vs_iter.png", "fe2s2_panels.png", "energies.csv"):
            src = out_dir / f
            if src.exists():
                (pdir / f).write_bytes(src.read_bytes())
    print(f"WROTE plots + energies.csv to {out_dir} (methods: {list(traces)}, refs: {list(refs)})")


if __name__ == "__main__":
    main()
