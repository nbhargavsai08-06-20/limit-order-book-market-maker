"""
main.py
=======
Simulation entry-point: runs both the Avellaneda-Stoikov and Naive Symmetric
market makers on the same price realisation, prints summary statistics, and
saves a publication-ready figure.

Usage
-----
    python main.py
"""

from __future__ import annotations

import math
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless rendering (no display required)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from market_maker import (
    AvellanedaStoikovMM,
    SimParams,
    MarketSimulator,
    StepRecord,
    SymmetricMarketMaker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pnl_series(history: List[StepRecord]) -> np.ndarray:
    """Cumulative PnL = wealth(t) - wealth(0)."""
    w = np.array([r.wealth for r in history])
    return w - w[0]


def _inv_series(history: List[StepRecord]) -> np.ndarray:
    return np.array([r.inventory for r in history])


def _time_axis(history: List[StepRecord]) -> np.ndarray:
    return np.array([r.t for r in history])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

STYLE = {
    "as":     {"color": "#1f77b4", "label": "Avellaneda-Stoikov"},
    "naive":  {"color": "#d62728", "label": "Naive Symmetric", "linestyle": "--"},
}


def make_figure(
    as_history:    List[StepRecord],
    naive_history: List[StepRecord],
    filename:      str = "performance_plot.png",
) -> None:
    """
    Generate and save a two-panel publication-ready figure.

    Panel 1: Cumulative PnL — AS vs Naive
    Panel 2: Inventory trajectory — AS vs Naive
    """
    t      = _time_axis(as_history)
    as_pnl = _pnl_series(as_history)
    na_pnl = _pnl_series(naive_history)
    as_inv = _inv_series(as_history)
    na_inv = _inv_series(naive_history)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        "Market Making Performance: Avellaneda-Stoikov vs. Naive Symmetric Quoter",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # ---- Subplot 1: PnL ----
    ax1 = axes[0]
    ax1.plot(t, as_pnl, lw=1.5, **STYLE["as"])
    ax1.plot(t, na_pnl, lw=1.5, **STYLE["naive"])
    ax1.axhline(0, color="black", lw=0.6, ls=":")
    ax1.set_ylabel("Cumulative PnL", fontsize=11)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Cumulative PnL", fontsize=11, loc="left")

    # ---- Subplot 2: Inventory ----
    ax2 = axes[1]
    ax2.plot(t, as_inv, lw=1.5, **STYLE["as"])
    ax2.plot(t, na_inv, lw=1.5, **STYLE["naive"])
    ax2.axhline(0, color="black", lw=0.6, ls=":")
    ax2.set_ylabel("Inventory  q(t)", fontsize=11)
    ax2.set_xlabel("Time  t in [0, T]", fontsize=11)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f"))
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Inventory Trajectory", fontsize=11, loc="left")

    # Annotation: AS inventory std
    as_std = np.std(as_inv)
    na_std = np.std(na_inv)
    ax2.annotate(
        f"AS std(q)={as_std:.2f}   Naive std(q)={na_std:.2f}",
        xy=(0.01, 0.07),
        xycoords="axes fraction",
        fontsize=9,
        color="dimgray",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Figure saved -> {filename}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(
    as_stats:    dict,
    naive_stats: dict,
) -> None:
    """Pretty-print a side-by-side statistics table."""
    cols = ["final_pnl", "max_inventory", "sharpe", "max_drawdown", "return_to_dd"]
    labels = {
        "final_pnl":     "Final PnL",
        "max_inventory": "Max |Inventory|",
        "sharpe":        "Sharpe Ratio",
        "max_drawdown":  "Max Drawdown",
        "return_to_dd":  "Return / Drawdown",
    }

    sep = "-" * 54
    header = f"\n{'Metric':<22} {'AS':>14} {'Naive':>14}"
    print("\n" + sep)
    print("  SIMULATION SUMMARY")
    print(sep)
    print(header)
    print("-" * 54)
    for col in cols:
        lbl = labels[col]
        av  = as_stats[col]
        nv  = naive_stats[col]
        print(f"  {lbl:<20} {av:>14.4f} {nv:>14.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    params = SimParams(
        S0        = 100.0,
        sigma     = 2.0,
        T         = 1.0,
        n_steps   = 3_000,
        seed      = 42,
        gamma     = 0.1,
        k         = 1.5,
        A         = 140.0,
        fixed_half_spread = 1.0,
    )

    print("=" * 54)
    print("  Event-Driven LOB Market Making Simulation")
    print(f"  Steps={params.n_steps}  sigma={params.sigma}  gamma={params.gamma}  "
          f"k={params.k}  A={params.A}")
    print("=" * 54)

    sim = MarketSimulator(params)

    # ---- Avellaneda-Stoikov ----
    as_agent = AvellanedaStoikovMM(sim)
    as_agent.run()
    as_stats = as_agent.summary()

    # ---- Naive Symmetric (same sim, same random draws) ----
    naive_agent = SymmetricMarketMaker(sim)
    naive_agent.run()
    naive_stats = naive_agent.summary()

    # ---- Output ----
    print_summary(as_stats, naive_stats)
    make_figure(as_agent.history, naive_agent.history)


if __name__ == "__main__":
    main()
