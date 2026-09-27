"""
test_lob.py
===========
Unit tests for the Limit Order Book engine and market-making model.

Run with:  pytest test_lob.py -v
"""

from __future__ import annotations

import math
import pytest

from order_book import Fill, Order, OrderBook, Side
from market_maker import (
    AvellanedaStoikovMM,
    MarketSimulator,
    SimParams,
    SymmetricMarketMaker,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def empty_book() -> OrderBook:
    return OrderBook()


@pytest.fixture
def seeded_book() -> OrderBook:
    """OrderBook with 3 bids and 3 asks pre-loaded."""
    book = OrderBook()
    # Bids: 99, 98, 97
    book.add_limit_order(Side.BUY,  99.0, 10.0, timestamp=1.0)
    book.add_limit_order(Side.BUY,  98.0,  5.0, timestamp=2.0)
    book.add_limit_order(Side.BUY,  97.0,  5.0, timestamp=3.0)
    # Asks: 101, 102, 103
    book.add_limit_order(Side.SELL, 101.0, 10.0, timestamp=4.0)
    book.add_limit_order(Side.SELL, 102.0,  5.0, timestamp=5.0)
    book.add_limit_order(Side.SELL, 103.0,  5.0, timestamp=6.0)
    return book


@pytest.fixture
def sim_params() -> SimParams:
    return SimParams(
        S0=100.0, sigma=2.0, T=1.0, n_steps=500,
        seed=0, gamma=0.1, k=1.5, A=140.0, fixed_half_spread=1.0,
    )


@pytest.fixture
def sim(sim_params: SimParams) -> MarketSimulator:
    return MarketSimulator(sim_params)


# ===========================================================================
# 1. Order Book – Basic Quotes
# ===========================================================================

class TestBestQuotes:
    def test_best_bid_ask(self, seeded_book: OrderBook) -> None:
        assert seeded_book.best_bid() == pytest.approx(99.0)
        assert seeded_book.best_ask() == pytest.approx(101.0)

    def test_mid_price(self, seeded_book: OrderBook) -> None:
        assert seeded_book.mid_price() == pytest.approx(100.0)

    def test_empty_book_quotes(self, empty_book: OrderBook) -> None:
        assert empty_book.best_bid() is None
        assert empty_book.best_ask() is None
        assert empty_book.mid_price() is None


# ===========================================================================
# 2. FIFO Matching & Price Improvement
# ===========================================================================

class TestFIFOMatching:
    def test_buy_market_hits_best_ask(self, seeded_book: OrderBook) -> None:
        """A BUY market order should fill at the lowest ask (101)."""
        fills, remaining = seeded_book.process_market_order(Side.BUY, 5.0)
        assert remaining == pytest.approx(0.0)
        assert len(fills) == 1
        assert fills[0].price == pytest.approx(101.0)
        assert fills[0].quantity == pytest.approx(5.0)

    def test_sell_market_hits_best_bid(self, seeded_book: OrderBook) -> None:
        """A SELL market order should fill at the highest bid (99)."""
        fills, remaining = seeded_book.process_market_order(Side.SELL, 5.0)
        assert remaining == pytest.approx(0.0)
        assert fills[0].price == pytest.approx(99.0)
        assert fills[0].quantity == pytest.approx(5.0)

    def test_fifo_priority_within_level(self) -> None:
        """Two orders at the same price: earliest timestamp fills first."""
        book = OrderBook()
        o1 = book.add_limit_order(Side.SELL, 100.0, 3.0, timestamp=1.0)
        o2 = book.add_limit_order(Side.SELL, 100.0, 7.0, timestamp=2.0)

        fills, remaining = book.process_market_order(Side.BUY, 4.0)

        assert remaining == pytest.approx(0.0)
        # First 3 units from o1, then 1 unit from o2
        assert fills[0].quantity == pytest.approx(3.0)
        assert fills[1].quantity == pytest.approx(1.0)

    def test_price_improvement_walks_book(self, seeded_book: OrderBook) -> None:
        """A large BUY order should walk up the ask ladder."""
        fills, remaining = seeded_book.process_market_order(Side.BUY, 15.0)
        prices = [f.price for f in fills]
        assert 101.0 in prices
        assert 102.0 in prices
        assert remaining == pytest.approx(0.0)

    def test_insufficient_liquidity(self, seeded_book: OrderBook) -> None:
        """Requesting more than book depth leaves residual remaining."""
        _, remaining = seeded_book.process_market_order(Side.BUY, 1000.0)
        assert remaining > 0.0


# ===========================================================================
# 3. Partial Fills
# ===========================================================================

class TestPartialFills:
    def test_partial_fill_reduces_resting_qty(self) -> None:
        book = OrderBook()
        book.add_limit_order(Side.SELL, 100.0, 10.0)
        fills, remaining = book.process_market_order(Side.BUY, 4.0)

        assert remaining == pytest.approx(0.0)
        assert fills[0].quantity == pytest.approx(4.0)

        # Resting order still has 6 units
        assert book.best_ask() == pytest.approx(100.0)
        depth = book.depth(1)
        assert depth["asks"][0][1] == pytest.approx(6.0)


# ===========================================================================
# 4. Order Cancellation
# ===========================================================================

class TestCancellation:
    def test_cancel_existing_order(self, seeded_book: OrderBook) -> None:
        """Cancel the best bid (first order added, id=1)."""
        removed = seeded_book.cancel_order(1)   # bid @ 99, id=1
        assert removed is True
        assert seeded_book.best_bid() == pytest.approx(98.0)

    def test_cancel_nonexistent_order(self, seeded_book: OrderBook) -> None:
        removed = seeded_book.cancel_order(9999)
        assert removed is False

    def test_cancel_cleans_price_level(self) -> None:
        book = OrderBook()
        order = book.add_limit_order(Side.BUY, 50.0, 1.0)
        book.cancel_order(order.id)
        # Level should be gone → best_bid should not be 50
        assert book.best_bid() is None


# ===========================================================================
# 5. Cash & Inventory Conservation
# ===========================================================================

class TestConservation:
    def test_cash_and_inventory_balance(self, sim: MarketSimulator) -> None:
        """
        After every buy fill the agent pays bid_price and gains 1 unit,
        after every sell fill the agent receives ask_price and loses 1 unit.
        Cash + inventory × last_mid should be tracked consistently.
        """
        agent = AvellanedaStoikovMM(sim)
        history = agent.run()

        # Reconstruct cash naively: check that last wealth is cash + inv * S
        last = history[-1]
        recomputed_wealth = last.cash + last.inventory * sim.price_path[sim.p.n_steps]
        assert recomputed_wealth == pytest.approx(last.wealth, rel=1e-9)

    def test_inventory_updates_per_fill(self, sim: MarketSimulator) -> None:
        """Inventory changes should be integer-valued at each step (±1 or 0)."""
        agent = SymmetricMarketMaker(sim)
        history = agent.run()

        inv_series = [r.inventory for r in history]
        diffs = [abs(inv_series[i+1] - inv_series[i]) for i in range(len(inv_series)-1)]

        # Each step can result in at most one buy (+1) and one sell (-1)
        # so the net change per step is in {-2, -1, 0, 1, 2}
        for d in diffs:
            assert d <= 2.0 + 1e-9


# ===========================================================================
# 6. Reservation Price Skew (Avellaneda-Stoikov)
# ===========================================================================

class TestReservationPrice:
    """
    Verify that the reservation price properly skews quotes.

    r = S - q·γ·σ²·(T - t)

    When q > 0  (long inventory):  r < S  →  bid and ask shift down.
    When q < 0  (short inventory): r > S  →  bid and ask shift up.
    """

    def _quotes_at(
        self,
        sim: MarketSimulator,
        inventory: float,
        step: int = 500,
    ) -> tuple[float, float]:
        agent = AvellanedaStoikovMM(sim)
        agent.inventory = inventory
        return agent.compute_quotes(step)

    def test_long_inventory_skews_quotes_down(self, sim: MarketSimulator) -> None:
        bid_flat, ask_flat = self._quotes_at(sim, inventory=0.0, step=100)
        bid_long, ask_long = self._quotes_at(sim, inventory=5.0, step=100)

        assert bid_long < bid_flat, "Long inventory should lower bid quote"
        assert ask_long < ask_flat, "Long inventory should lower ask quote"

    def test_short_inventory_skews_quotes_up(self, sim: MarketSimulator) -> None:
        bid_flat, ask_flat = self._quotes_at(sim, inventory= 0.0, step=100)
        bid_short, ask_short = self._quotes_at(sim, inventory=-5.0, step=100)

        assert bid_short > bid_flat, "Short inventory should raise bid quote"
        assert ask_short > ask_flat, "Short inventory should raise ask quote"

    def test_spread_positive(self, sim: MarketSimulator) -> None:
        """Ask must always be strictly above bid."""
        for inv in [-10, -5, 0, 5, 10]:
            bid, ask = self._quotes_at(sim, inventory=float(inv), step=100)
            assert ask > bid, f"ask={ask} must exceed bid={bid} for inv={inv}"

    def test_reservation_price_formula(self, sim: MarketSimulator) -> None:
        """
        Spot-check the reservation price formula:
          r = S - q·γ·σ²·(T - t)
        """
        p   = sim.p
        step = 200
        S   = sim.price_path[step]
        t   = sim.times[step]
        q   = 3.0
        tau = p.T - t

        expected_r = S - q * p.gamma * (p.sigma ** 2) * tau

        # The bid = r - delta, ask = r + delta → mid of quotes ≈ r
        agent = AvellanedaStoikovMM(sim)
        agent.inventory = q
        bid, ask = agent.compute_quotes(step)
        quote_mid = (bid + ask) / 2.0

        assert quote_mid == pytest.approx(expected_r, rel=1e-6)


# ===========================================================================
# 7. Order creation guards
# ===========================================================================

class TestOrderValidation:
    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValueError, match="Price must be positive"):
            Order(id=1, side=Side.BUY, price=-1.0, quantity=1.0)

    def test_zero_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="Quantity must be positive"):
            Order(id=2, side=Side.SELL, price=100.0, quantity=0.0)
