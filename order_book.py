"""
order_book.py
=============
Event-driven Limit Order Book (LOB) engine with price-time (FIFO) priority.

Supports:
  - Limit order submission and cancellation
  - Market order matching with partial fills
  - Best bid / ask / midprice queries
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Side enum
# ---------------------------------------------------------------------------

class Side(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


# ---------------------------------------------------------------------------
# Order dataclass
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """Represents a single limit order resting in the book."""

    id: int
    side: Side
    price: float
    quantity: float           # remaining unfilled quantity
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")


# ---------------------------------------------------------------------------
# Fill record
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    """Record of a matched trade."""

    aggressor_side: Side
    price: float
    quantity: float


# ---------------------------------------------------------------------------
# Price level: sorted queue of orders at the same price
# ---------------------------------------------------------------------------

class PriceLevel:
    """FIFO queue of orders at a single price level."""

    def __init__(self, price: float) -> None:
        self.price = price
        self._orders: List[Order] = []

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add(self, order: Order) -> None:
        """Append order at the back (preserves FIFO)."""
        self._orders.append(order)

    def remove(self, order_id: int) -> bool:
        """Cancel order by id. Returns True if found."""
        for i, o in enumerate(self._orders):
            if o.id == order_id:
                self._orders.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Match against an incoming market order quantity
    # ------------------------------------------------------------------

    def match(self, qty: float) -> Tuple[List[Fill], float, Side]:
        """
        Match *qty* against this level.

        Returns
        -------
        fills     : list of Fill records
        remaining : unfilled quantity
        resting_side : side of the resting orders
        """
        fills: List[Fill] = []
        remaining = qty

        while self._orders and remaining > 0:
            resting = self._orders[0]
            exec_qty = min(remaining, resting.quantity)

            fills.append(Fill(
                aggressor_side=Side.BUY if resting.side == Side.SELL else Side.SELL,
                price=resting.price,
                quantity=exec_qty,
            ))

            resting.quantity -= exec_qty
            remaining       -= exec_qty

            if resting.quantity == 0:
                self._orders.pop(0)

        return fills, remaining, self._orders[0].side if self._orders else resting.side

    @property
    def total_qty(self) -> float:
        return sum(o.quantity for o in self._orders)

    def __len__(self) -> int:
        return len(self._orders)

    def __iter__(self) -> Iterator[Order]:
        return iter(self._orders)

    def __bool__(self) -> bool:
        return bool(self._orders)


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Price-time priority limit order book.

    Bids  → sorted *descending* (highest price at front)
    Asks  → sorted *ascending*  (lowest  price at front)
    """

    def __init__(self) -> None:
        # price → PriceLevel
        self._bids: Dict[float, PriceLevel] = {}
        self._asks: Dict[float, PriceLevel] = {}

        # order_id → (side, price)  for O(1) cancel lookup
        self._order_map: Dict[int, Tuple[Side, float]] = {}

        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _side_book(self, side: Side) -> Dict[float, PriceLevel]:
        return self._bids if side == Side.BUY else self._asks

    def _sorted_prices(self, side: Side) -> List[float]:
        if side == Side.BUY:
            return sorted(self._bids.keys(), reverse=True)
        return sorted(self._asks.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_limit_order(
        self,
        side: Side,
        price: float,
        quantity: float,
        timestamp: Optional[float] = None,
    ) -> Order:
        """
        Submit a new limit order.

        Parameters
        ----------
        side      : BUY or SELL
        price     : limit price
        quantity  : order size
        timestamp : optional UNIX timestamp (defaults to now)

        Returns
        -------
        The newly created Order object.
        """
        oid = self._next_id
        self._next_id += 1

        ts = timestamp if timestamp is not None else time.time()
        order = Order(id=oid, side=side, price=price, quantity=quantity, timestamp=ts)

        book = self._side_book(side)
        if price not in book:
            book[price] = PriceLevel(price)
        book[price].add(order)

        self._order_map[oid] = (side, price)
        return order

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel a resting limit order by id.

        Returns True if the order was found and removed, False otherwise.
        """
        if order_id not in self._order_map:
            return False

        side, price = self._order_map.pop(order_id)
        book = self._side_book(side)

        removed = book[price].remove(order_id)
        if removed and not book[price]:
            del book[price]

        return removed

    def process_market_order(
        self,
        side: Side,
        quantity: float,
    ) -> Tuple[List[Fill], float]:
        """
        Execute a market order by walking the book.

        Parameters
        ----------
        side     : BUY (lifts asks) or SELL (hits bids)
        quantity : desired quantity

        Returns
        -------
        fills     : list of Fill records
        remaining : unfilled quantity (>0 means insufficient liquidity)
        """
        if side == Side.BUY:
            opposing_book = self._asks
            opposing_sorted = sorted(opposing_book.keys())          # ascending
        else:
            opposing_book = self._bids
            opposing_sorted = sorted(opposing_book.keys(), reverse=True)  # descending

        all_fills: List[Fill] = []
        remaining = quantity

        for price in opposing_sorted:
            if remaining <= 0:
                break
            level = opposing_book[price]
            fills, remaining, _ = level.match(remaining)
            all_fills.extend(fills)

            # clean up empty levels and order_map
            dead_ids = [o.id for o in level if o.quantity == 0]   # noqa: already popped
            for oid in dead_ids:
                self._order_map.pop(oid, None)

            if not level:
                del opposing_book[price]

        return all_fills, remaining

    # ------------------------------------------------------------------
    # Quote helpers
    # ------------------------------------------------------------------

    def best_bid(self) -> Optional[float]:
        """Return the highest resting bid price, or None if empty."""
        return max(self._bids.keys()) if self._bids else None

    def best_ask(self) -> Optional[float]:
        """Return the lowest resting ask price, or None if empty."""
        return min(self._asks.keys()) if self._asks else None

    def mid_price(self) -> Optional[float]:
        """Return the midpoint between best bid and best ask."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """Return up to *levels* price levels on each side."""
        bids = [
            (p, self._bids[p].total_qty)
            for p in sorted(self._bids.keys(), reverse=True)[:levels]
        ]
        asks = [
            (p, self._asks[p].total_qty)
            for p in sorted(self._asks.keys())[:levels]
        ]
        return {"bids": bids, "asks": asks}

    def __repr__(self) -> str:
        bb = self.best_bid()
        ba = self.best_ask()
        return (
            f"OrderBook(best_bid={bb}, best_ask={ba}, "
            f"mid={self.mid_price()}, "
            f"bid_levels={len(self._bids)}, ask_levels={len(self._asks)})"
        )
