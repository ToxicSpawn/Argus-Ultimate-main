"""
Persistence Module
==================

State persistence and recovery.
Refactored from unified_trading_system.py.
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages system state persistence.
    """
    
    def __init__(self, db_path: str = "data/system_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        
        logger.info("StateManager initialized")
    
    async def initialize(self):
        """Initialize state manager."""
        # Create tables
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                state_data TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        
        logger.info("State manager initialized")
    
    async def save_state(self, state_data: Dict[str, Any]):
        """Save system state."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO system_state (timestamp, state_data) VALUES (?, ?)",
            (datetime.utcnow().isoformat(), json.dumps(state_data))
        )
        conn.commit()
        conn.close()
        
        logger.info("System state saved")
    
    async def load_state(self) -> Optional[Dict[str, Any]]:
        """Load latest system state."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "SELECT state_data FROM system_state ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        
        return None
