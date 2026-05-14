'''
Tests for Unified Database

NOTE: Depends on core.database_unified which was moved to archive
during restructuring. Skipped until migrated to new architecture.
'''

import pytest

pytest.importorskip("core.database_unified", reason="core.database_unified moved to archive during restructuring")

from core.database_unified import UnifiedDatabase


@pytest.mark.asyncio
async def test_database_connection():
    '''Test database connection'''
    config = {"database": {"type": "sqlite", "path": ":memory:"}}
    db = UnifiedDatabase(config)
    connected = await db.connect()
    assert connected is True
    await db.disconnect()
