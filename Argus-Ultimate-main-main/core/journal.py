#!/usr/bin/env python3
"""
Trading Journal - S+ Tier
Comprehensive trade logging and analysis system for Argus Ultimate.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import sqlite3
import os


@dataclass
class TradeRecord:
    """Complete trade record for journal"""

    intent_id: str
    plan_id: str
    client_order_id: str
    exchange_order_id: Optional[str]
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    order_type: str
    timestamp: datetime
    status: str
    filled: float = 0.0
    average: float = 0.0
    cost: float = 0.0
    last_error: Optional[str] = None
    plan_json: str = "{}"


class TradingJournal:
    """
    Trading Journal - S+ Tier
    Comprehensive trade logging and analysis system.
    """

    def __init__(self, db_path: str = "data/trading_journal.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the journal database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
              intent_id TEXT PRIMARY KEY,
    plan_id TEXT,
    client_order_id TEXT,
              exchange_order_id TEXT,
    symbol TEXT,
    side TEXT,
    quantity REAL,
    price REAL,
    order_type TEXT,
    timestamp TEXT,
    status TEXT,
    filled REAL DEFAULT 0,
    average REAL DEFAULT 0,
    cost REAL DEFAULT 0,
    last_error TEXT,
    plan_json TEXT
                )
            """
            )

            # Create indexes for efficient queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON trades(status)")

    def log_trade(self, trade: TradeRecord):
        """Log a trade to the journal"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trades
                (intent_id, plan_id, client_order_id, exchange_order_id, symbol, side,
                 quantity, price, order_type, timestamp, status, filled, average, cost,
                 last_error, plan_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.intent_id,
                    trade.plan_id,
                    trade.client_order_id,
                    trade.exchange_order_id,
                    trade.symbol,
                    trade.side,
                    trade.quantity,
                    trade.price,
                    trade.order_type,
                    trade.timestamp.isoformat(),
                    trade.status,
                    trade.filled,
                    trade.average,
                    trade.cost,
                    trade.last_error,
                    trade.plan_json,
                ),
            )

    def get_trade(self, intent_id: str) -> Optional[TradeRecord]:
        """Get a trade by intent ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM trades WHERE intent_id = ?", (intent_id,))
            row = cursor.fetchone()

            if row:
                return TradeRecord(
                    intent_id=row[0],
                    plan_id=row[1],
                    client_order_id=row[2],
                    exchange_order_id=row[3],
                    symbol=row[4],
                    side=row[5],
                    quantity=row[6],
                    price=row[7],
                    order_type=row[8],
                    timestamp=datetime.fromisoformat(row[9]),
                    status=row[10],
                    filled=row[11],
                    average=row[12],
                    cost=row[13],
                    last_error=row[14],
                    plan_json=row[15],
                )

        return None

    def get_trades_by_symbol(self, symbol: str, limit: int = 100) -> List[TradeRecord]:
        """Get recent trades for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM trades
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (symbol, limit),
            )

            trades = []
            for row in cursor:
                trades.append(
                    TradeRecord(
                        intent_id=row[0],
                        plan_id=row[1],
                        client_order_id=row[2],
                        exchange_order_id=row[3],
                        symbol=row[4],
                        side=row[5],
                        quantity=row[6],
                        price=row[7],
                        order_type=row[8],
                        timestamp=datetime.fromisoformat(row[9]),
                        status=row[10],
                        filled=row[11],
                        average=row[12],
                        cost=row[13],
                        last_error=row[14],
                        plan_json=row[15],
                    )
                )

            return trades

    def get_daily_summary(self, date: str) -> Dict[str, Any]:
        """Get trading summary for a specific date"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
    COUNT(*) as total_trades,
    SUM(CASE WHEN side = 'buy' THEN filled ELSE 0 END) as buy_volume,
    SUM(CASE WHEN side = 'sell' THEN filled ELSE 0 END) as sell_volume,
    AVG(CASE WHEN status = 'filled' THEN average END) as avg_price,
    SUM(cost) as total_cost
                FROM trades
                WHERE DATE(timestamp) = ?
            """,
                (date,),
            )

            row = cursor.fetchone()

            return {
                "date": date,
                "total_trades": row[0] or 0,
                "buy_volume": row[1] or 0,
                "sell_volume": row[2] or 0,
                "avg_price": row[3] or 0,
                "total_cost": row[4] or 0,
            }

    def update_trade_status(
        self,
        intent_id: str,
        status: str,
        filled: float = None,
        average: float = None,
        cost: float = None,
        last_error: str = None,
    ):
        """Update trade status and execution details"""
        with sqlite3.connect(self.db_path) as conn:
            update_fields = ["status = ?"]
            values = [status]

            if filled is not None:
                update_fields.append("filled = ?")
                values.append(filled)

            if average is not None:
                update_fields.append("average = ?")
                values.append(average)

            if cost is not None:
                update_fields.append("cost = ?")
                values.append(cost)

            if last_error is not None:
                update_fields.append("last_error = ?")
                values.append(last_error)

            query = f"UPDATE trades SET {', '.join(update_fields)} WHERE intent_id = ?"
            values.append(intent_id)

            conn.execute(query, values)
