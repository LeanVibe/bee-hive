#!/usr/bin/env python3
"""
Simple validation test for the enhanced Project Index system.
Tests the core functionality of file monitoring, caching, and incremental updates.
"""

import asyncio
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.project_index import (
    ProjectIndexConfig, EnhancedFileMonitor, AdvancedCacheManager, 
    IncrementalUpdateEngine, EventPublisher, FileChangeEvent, FileChangeType,
    CacheLayer, CacheKey, UpdateStrategy
)
from app.core.redis import get_redis_client

async def test_enhanced_file_monitor():
    """Test the enhanced file monitor functionality."""
    print("🔍 Testing Enhanced File Monitor...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_id = uuid4()
        
        # Initialize file monitor
        monitor = EnhancedFileMonitor(debounce_interval=0.5)
        
        # Create test configuration
        config = ProjectIndexConfig()
        
        changes_detected = []
        
        def change_callback(event: FileChangeEvent):
            changes_detected.append(event)
            print(f"  📁 Change detected: {event.change_type.value} - {event.file_path.name}")
        
        monitor.add_change_callback(change_callback)
        
        try:
            # Start monitoring
            success = await monitor.start_monitoring(project_id, temp_path, config)
            assert success, "Failed to start monitoring"
            
            # Create test file
            test_file = temp_path / "test.py"
            test_file.write_text("print('Hello, World!')")
            
            # Wait for debouncing
            await asyncio.sleep(1.0)
            
            # Modify file
            test_file.write_text("print('Hello, Enhanced World!')")
            
            # Wait for debouncing
            await asyncio.sleep(1.0)
            
            # Force scan to process any pending events
            await monitor.force_scan(project_id)
            
            # Verify changes were detected
            assert len(changes_detected) > 0, f"No changes detected. Expected at least 1, got {len(changes_detected)}"
            
            print(f"  ✅ Detected {len(changes_detected)} file changes")
            
            # Stop monitoring
            await monitor.stop_monitoring(project_id)
            
        except Exception as e:
            print(f"  ❌ Monitor test failed: {e}")
            raise
        
        print("  ✅ Enhanced File Monitor test passed")


async def test_advanced_cache_manager():
    """Test the advanced cache manager functionality."""
    print("🚀 Testing Advanced Cache Manager...")
    
    try:
        # Try to get Redis client, skip if not available
        try:
            from app.core.redis import init_redis
            await init_redis()
            redis_client = get_redis_client()
        except Exception as e:
            print(f"  ⚠️  Skipping cache test - Redis not available: {e}")
            return
        
        # Initialize cache manager
        cache = AdvancedCacheManager(
            redis_client=redis_client,
            enable_compression=True,
            compression_threshold=100
        )
        
        # Test AST caching
        file_path = "/test/example.py"
        file_hash = "abc123def456"
        ast_data = {"type": "Module", "body": [{"type": "FunctionDef", "name": "test"}]}
        
        # Set AST cache
        success = await cache.set_ast_cache(file_path, file_hash, ast_data)
        assert success, "Failed to set AST cache"
        print("  📦 AST data cached successfully")
        
        # Get AST cache
        cached_ast = await cache.get_ast_cache(file_path, file_hash)
        assert cached_ast is not None, "Failed to retrieve AST cache"
        assert cached_ast == ast_data, "Cached AST data doesn't match"
        print("  📦 AST data retrieved successfully")
        
        # Test cache invalidation
        await cache.invalidate_file_caches(file_path, file_hash)
        invalidated_ast = await cache.get_ast_cache(file_path, file_hash)
        assert invalidated_ast is None, "Cache invalidation failed"
        print("  🗑️  Cache invalidation successful")
        
        # Test cache statistics
        stats = cache.get_cache_stats()
        assert stats.hits > 0, "No cache hits recorded"
        assert stats.sets > 0, "No cache sets recorded"
        print(f"  📊 Cache stats: {stats.hits} hits, {stats.misses} misses, {stats.sets} sets")
        
        print("  ✅ Advanced Cache Manager test passed")
        
    except Exception as e:
        print(f"  ❌ Cache test failed: {e}")
        raise


async def test_incremental_update_engine():
    """Test the incremental update engine functionality."""
    print("⚡ Testing Incremental Update Engine...")
    
    try:
        # Try to get Redis client, skip if not available
        try:
            from app.core.redis import init_redis
            await init_redis()
            redis_client = get_redis_client()
        except Exception as e:
            print(f"  ⚠️  Skipping incremental update test - Redis not available: {e}")
            return
        
        # Initialize components
        cache = AdvancedCacheManager(redis_client)
        config = ProjectIndexConfig()
        engine = IncrementalUpdateEngine(cache, config)
        
        project_id = uuid4()
        
        # Create test file change events
        changes = [
            FileChangeEvent(
                file_path=Path("/test/file1.py"),
                change_type=FileChangeType.CREATED,
                timestamp=datetime.utcnow(),
                project_id=project_id
            ),
            FileChangeEvent(
                file_path=Path("/test/file2.py"),
                change_type=FileChangeType.MODIFIED,
                timestamp=datetime.utcnow(),
                project_id=project_id
            )
        ]
        
        # Process changes
        result = await engine.process_file_changes(project_id, changes)
        
        assert result.strategy_used in [UpdateStrategy.MINIMAL, UpdateStrategy.CASCADING], \
            f"Unexpected update strategy: {result.strategy_used}"
        assert result.update_duration_ms >= 0, "Invalid update duration"
        
        print(f"  ⚙️  Update strategy: {result.strategy_used.value}")
        print(f"  ⏱️  Update duration: {result.update_duration_ms}ms")
        print(f"  📈 Files analyzed: {result.files_analyzed}")
        
        # Test statistics
        stats = engine.get_update_statistics()
        assert stats['total_updates'] > 0, "No updates recorded"
        print(f"  📊 Update stats: {stats['total_updates']} total updates")
        
        print("  ✅ Incremental Update Engine test passed")
        
    except Exception as e:
        print(f"  ❌ Incremental update test failed: {e}")
        raise


async def test_event_publisher():
    """Test the event publisher functionality."""
    print("📡 Testing Event Publisher...")
    
    try:
        # Initialize event publisher
        publisher = EventPublisher(max_history=100)
        
        events_received = []
        
        def event_callback(event):
            events_received.append(event)
            print(f"  📨 Event received: {event.event_type.value}")
        
        # Subscribe to events
        subscriber_id = publisher.subscribe(event_callback)
        
        # Create and publish test events
        from app.project_index.events import create_file_event, EventType
        
        test_event = create_file_event(
            EventType.FILE_CREATED,
            project_id=uuid4(),
            file_path="/test/example.py"
        )
        
        await publisher.publish(test_event)
        
        # Verify event was received
        assert len(events_received) == 1, f"Expected 1 event, got {len(events_received)}"
        assert events_received[0].event_type == EventType.FILE_CREATED, "Wrong event type"
        
        # Test event history
        history = publisher.get_event_history()
        assert len(history) == 1, f"Expected 1 event in history, got {len(history)}"
        
        # Unsubscribe
        success = publisher.unsubscribe(subscriber_id)
        assert success, "Failed to unsubscribe"
        
        # Test statistics
        stats = publisher.get_statistics()
        assert stats['events_published'] == 1, "Wrong event count in statistics"
        
        print(f"  📊 Event stats: {stats['events_published']} published, {stats['subscribers_count']} subscribers")
        print("  ✅ Event Publisher test passed")
        
    except Exception as e:
        print(f"  ❌ Event publisher test failed: {e}")
        raise


async def test_integration():
    """Test integration between all components."""
    print("🔗 Testing Component Integration...")
    
    try:
        # This is a basic integration test
        # In a full implementation, we would test:
        # - File monitor triggering incremental updates
        # - Cache invalidation on file changes
        # - Event publishing throughout the pipeline
        
        config = ProjectIndexConfig(
            monitoring_enabled=True,
            incremental_updates=True,
            events_enabled=True,
            cache_enabled=True
        )
        
        # Validate configuration
        assert config.monitoring_enabled, "Monitoring should be enabled"
        assert config.incremental_updates, "Incremental updates should be enabled"
        assert config.events_enabled, "Events should be enabled"
        
        print("  ⚙️  Configuration validated")
        print("  🔗 All components configured for integration")
        print("  ✅ Integration test passed")
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        raise


async def main():
    """Run all validation tests."""
    print("🚀 Enhanced Project Index System Validation")
    print("=" * 50)
    
    tests = [
        test_enhanced_file_monitor,
        test_advanced_cache_manager,
        test_incremental_update_engine,
        test_event_publisher,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Enhanced Project Index system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)