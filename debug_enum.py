#!/usr/bin/env python3
"""Debug enum values."""

from app.models.task import TaskStatus, TaskPriority, TaskType

print("TaskStatus enum:")
for status in TaskStatus:
    print(f"  {status.name} = {status.value}")

print("\nTaskPriority enum:")
for priority in TaskPriority:
    print(f"  {priority.name} = {priority.value}")

print("\nTaskType enum:")
for task_type in TaskType:
    print(f"  {task_type.name} = {task_type.value}")

# Test what SQLAlchemy should be getting
print(f"\nTaskStatus.PENDING should send: '{TaskStatus.PENDING.value}'")
print(f"But SQLAlchemy is sending: '{TaskStatus.PENDING}'")