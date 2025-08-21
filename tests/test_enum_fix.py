#!/usr/bin/env python3
"""
Test script to validate enum column fixes.
"""

import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.abspath('.'))

from sqlalchemy import select, func
from app.core.database import get_session, init_database
from app.models.task import Task, TaskStatus


async def test_enum_queries():
    """Test that the enum queries work correctly."""
    
    # Initialize database connection
    await init_database()
    
    async with get_session() as db:
        try:
            # Test the problematic query from autonomous_development.py
            print("Testing TaskStatus enum queries...")
            
            task_stats = await db.execute(
                select(
                    func.count(Task.id).label("total_tasks"),
                    func.count().filter(Task.status == TaskStatus.PENDING).label("pending_tasks"),
                    func.count().filter(Task.status == TaskStatus.IN_PROGRESS).label("in_progress_tasks"),
                    func.count().filter(Task.status == TaskStatus.COMPLETED).label("completed_tasks"),
                    func.count().filter(Task.status == TaskStatus.FAILED).label("failed_tasks"),
                )
            )
            
            task_row = task_stats.first()
            print("✅ Query successful!")
            print(f"Total tasks: {task_row.total_tasks}")
            print(f"Pending tasks: {task_row.pending_tasks}")
            print(f"In progress tasks: {task_row.in_progress_tasks}")
            print(f"Completed tasks: {task_row.completed_tasks}")
            print(f"Failed tasks: {task_row.failed_tasks}")
            
            # Test individual enum comparisons
            print("\nTesting individual enum comparisons...")
            
            pending_count = await db.execute(
                select(func.count(Task.id)).where(Task.status == TaskStatus.PENDING)
            )
            print(f"✅ Pending count query: {pending_count.scalar()}")
            
            completed_count = await db.execute(
                select(func.count(Task.id)).where(Task.status == TaskStatus.COMPLETED)
            )
            print(f"✅ Completed count query: {completed_count.scalar()}")
            
            print("\n🎉 All enum queries are working correctly!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_enum_queries())
    sys.exit(0 if success else 1)