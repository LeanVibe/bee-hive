#!/usr/bin/env python3
"""Simple enum test."""

import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from sqlalchemy import select, func
from app.core.database import get_session, init_database
from app.models.task import Task, TaskStatus

async def simple_test():
    await init_database()
    
    async with get_session() as db:
        result = await db.execute(
            select(func.count(Task.id)).where(Task.status == TaskStatus.PENDING)
        )
        pending_count = result.scalar()
        print(f"✅ SUCCESS: Found {pending_count} pending tasks")
        return True

if __name__ == "__main__":
    try:
        success = asyncio.run(simple_test())  
        print("🎉 Enum fix is working!")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)