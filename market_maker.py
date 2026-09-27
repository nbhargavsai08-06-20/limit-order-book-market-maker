"""
market_maker.py
===============
Implementation of the Avellaneda-Stoikov (2008) market-making model and a
naive symmetric baseline.

Mathematical background
-----------------------
Let S_t be the mid-price following:
    dS_t = σ dW_t  (Brownian motion; Poisson jumps optional)

Reservation price (inventory-skewed midprice):
    r(s, q, t) = s - q · γ · σ² · (T - t)

Optimal half-spread:
    δ^a + δ^b = γ σ² (T - t) + (2/γ) · ln(1 + γ/k)

Market-order arrival intensity (from the Poisson thinning model):
    λ(δ) = A · exp(-k · δ)

Quotes:
    bid = r - δ^b
    ask = r + δ^a
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from order_book import Fill, Order, OrderBook, Side

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """All hyper-parameters for a single simulation run."""

    # ---- Market dynamics ----
    S0:    float = 100.0   # initial mid-price
    sigma: float = 2.0     # annualised (per-unit-T) vol
    T:     float = 1.0     # simulation horizon (normalised)
    n_steps: int = 2_000   # number of discrete time steps
    seed:  Optional[int] = 42

    # ---- Avellaneda-Stoikov ----
    gamma: float = 0.1     # risk-aversion coefficient
    k:     float = 1.5     # order-book depth / fill-rate shape
    A:     float = 140.0   # baseline Poisson arrival intensity

    # ---- Symmetric baseline ----
    fixed_half_spread: float = 1.0

    # ---- LOB seeding ----
    lob_levels: int   = 10   # levels to seed on each side
    lob_qty:    float = 5.0  # resting qty per level


# ---------------------------------------------------------------------------
# Simulation record (one step)
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    t:          float
    mid:        float
    inventory:  float
    cash:       float
    wealth:     float
    bid_quote:  float
    ask_quote:  float


# ---------------------------------------------------------------------------
# Market simulator (shared price-path + event generator)
# ---------------------------------------------------------------------------

class MarketSimulator:
    """
    Generates a shared mid-price path and Poisson market-order events.
    Both the AS and the Naive agent run on identical realisations.
    """

    def __init__(self, params: SimParams) -> None:
        self.p = params
        rng = np.random.default_rng(params.seed)

        dt = params.T / params.n_steps
        # Geometric Brownian increments (arithmetic for simplicity; no drift)
        increments = rng.normal(0.0, params.sigma * math.sqrt(dt), size=params.n_steps)
        price_path = np.empty(params.n_steps + 1)
        price_path[0] = params.S0
        for i in range(params.n_steps):
            price_path[i + 1] = max(price_path[i] + increments[i], 1e-3)

        self.price_path = price_path          # shape (n_steps+1,)
        self.dt         = dt
        self.times      = np.linspace(0.0, params.T, params.n_steps + 1)

        # Pre-generate uniform draws for Poisson event thinning
        # We'll use these in the agent loops via index
        self._buy_draws  = rng.random(size=(params.n_steps, 10))
        self._sell_draws = rng.random(size=(params.n_steps, 10))

    def arrival_prob(self, half_spread: float) -> float:
        """
        Probability of at least one market order on one side in dt,
        given the Poisson thinning model λ(δ) = A·exp(-k·δ).
        """
        lam = self.p.A * math.exp(-self.p.k * max(half_spread, 1e-6))
        return 1.0 - math.exp(-lam * self.dt)      # P(N ≥ 1) for Poisson(λ·dt)


# ---------------------------------------------------------------------------
# Base agent interface
# ---------------------------------------------------------------------------

class BaseMarketMaker:
    """Abstract base class for market-making agents."""

    def __init__(self, sim: MarketSimulator) -> None:
        self.sim       = sim
        self.p         = sim.p
        self.cash:      float = 0.0
        self.inventory: float = 0.0
        self.history:   List[StepRecord] = []

    # ------------------------------------------------------------------
    # To be implemented by subclasses
    # ------------------------------------------------------------------

    def compute_quotes(self, step: int) -> Tuple[float, float]:
        """Return (bid_price, ask_price) given the current step index."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared simulation loop
    # ------------------------------------------------------------------

    def run(self) -> List[StepRecord]:
        """
        Run the full simulation.

        At each step:
          1. Observe new mid-price S_t.
          2. Compute quotes (bid, ask).
          3. Sample Poisson arrivals on each side (independent thinning).
          4. Fill at quoted price, updating cash & inventory.
          5. Log state.
        """
        sim = self.sim
        p   = self.p

        for step in range(p.n_steps):
            t   = sim.times[step]
            S_t = sim.price_path[step]
            bid_q, ask_q = self.compute_quotes(step)

            # ---- half-spreads from mid ----
            delta_b = S_t - bid_q   # our distance below mid
            delta_a = ask_q - S_t   # our distance above mid

            # ---- Buy-side arrivals: market seller hits our bid ----
            p_buy = sim.arrival_prob(max(delta_b, 1e-6))
            buy_draw = sim.price_path[step]   # deterministic seed per step

            # Use pre-generated uniform draws
            if sim._buy_draws[step, 0] < p_buy:
                # We buy 1 unit at our bid
                self.cash      -= bid_q
                self.inventory += 1.0

            # ---- Sell-side arrivals: market buyer lifts our ask ----
            p_sell = sim.arrival_prob(max(delta_a, 1e-6))
            if sim._sell_draws[step, 0] < p_sell:
                # We sell 1 unit at our ask
                self.cash      += ask_q
                self.inventory -= 1.0

            # ---- Wealth = cash + inventory mark-to-market ----
            S_next = sim.price_path[step + 1]
            wealth = self.cash + self.inventory * S_next

            rec = StepRecord(
                t=t,
                mid=S_t,
                inventory=self.inventory,
                cash=self.cash,
                wealth=wealth,
                bid_quote=bid_q,
                ask_quote=ask_q,
            )
            self.history.append(rec)

        return self.history

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Compute key performance metrics from the history."""
        if not self.history:
            raise RuntimeError("Run the simulation first.")

        wealths   = np.array([r.wealth    for r in self.history])
        invs      = np.array([r.inventory for r in self.history])
        pnl       = wealths - wealths[0]            # cumulative PnL
        returns   = np.diff(pnl)
        final_pnl = pnl[-1]
        max_inv   = float(np.max(np.abs(invs)))

        # Sharpe (annualised) — using per-step returns
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * math.sqrt(len(returns))
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(wealths)
        drawdowns   = running_max - wealths
        max_dd      = float(np.max(drawdowns))
        ret_to_dd   = final_pnl / max_dd if max_dd > 0 else float("inf")

        return {
            "final_pnl":   round(final_pnl, 4),
            "max_inventory": round(max_inv, 1),
            "sharpe":       round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "return_to_dd": round(ret_to_dd, 4),
        }


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov agent
# ---------------------------------------------------------------------------

class AvellanedaStoikovMM(BaseMarketMaker):
    """
    Optimal market maker from Avellaneda & Stoikov (2008).

    Reservation price  :  r = S - q·γ·σ²·(T - t)
    Optimal half-spread:  δ = γσ²(T-t)/2 + (1/γ)·ln(1 + γ/k)
    Quotes             :  bid = r - δ,  ask = r + δ
    """

    def compute_quotes(self, step: int) -> Tuple[float, float]:
        sim = self.sim
        p   = self.p

        S = sim.price_path[step]
        t = sim.times[step]
        q = self.inventory
        tau = p.T - t                   # time remaining

        # ---- Reservation price ----
        r = S - q * p.gamma * (p.sigma ** 2) * tau

        # ---- Optimal half-spread ----
        # δ^a + δ^b = γσ²τ + (2/γ)·ln(1 + γ/k)
        spread_sum = (
            p.gamma * (p.sigma ** 2) * tau
            + (2.0 / p.gamma) * math.log(1.0 + p.gamma / p.k)
        )
        delta = spread_sum / 2.0        # symmetric split

        bid = r - delta
        ask = r + delta

        # Safety: ensure positive prices
        bid = max(bid, 1e-2)
        ask = max(ask, bid + 1e-4)

        return bid, ask


# ---------------------------------------------------------------------------
# Naive Symmetric baseline
# ---------------------------------------------------------------------------

class SymmetricMarketMaker(BaseMarketMaker):
    """
    Naive baseline: always quotes a fixed half-spread around the raw
    mid-price, ignoring inventory.

    bid = S - Δ
    ask = S + Δ
    """

    def compute_quotes(self, step: int) -> Tuple[float, float]:
        S     = self.sim.price_path[step]
        delta = self.p.fixed_half_spread

        bid = max(S - delta, 1e-2)
        ask = S + delta

        return bid, ask
