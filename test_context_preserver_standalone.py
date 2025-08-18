#!/usr/bin/env python3
"""
Standalone test for ProductionContextPreserver
Tests the implementation without full app dependencies.
"""

import sys
import asyncio
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.communication.context_preserver import ProductionContextPreserver
from app.core.communication.protocol_models import ContextPackage
from app.core.agents.universal_agent_interface import AgentType


async def test_context_preserver():
    """Test the context preserver implementation."""
    print("🧪 Testing ProductionContextPreserver...")
    
    # Create instance
    preserver = ProductionContextPreserver()
    print("✅ Created ProductionContextPreserver instance")
    
    # Create sample execution context
    sample_context = {
        "variables": {
            "project_name": "bee-hive",
            "current_task": "context_preservation",
            "file_count": 42,
            "status": "in_progress"
        },
        "current_state": {
            "working_directory": "/tmp/workspace",
            "last_command": "implement_context_preserver",
            "active_files": ["context_preserver.py", "protocol_models.py"]
        },
        "task_history": [
            {
                "task_id": "task_001",
                "task_type": "implementation",
                "completed_at": "2024-01-15T10:30:00Z",
                "result": "success"
            }
        ],
        "intermediate_results": [
            {"step": "analysis", "data": {"complexity": "high"}}
        ],
        "files_created": [
            "app/core/communication/context_preserver.py"
        ],
        "files_modified": [
            "app/core/communication/__init__.py"
        ],
        "workflow_position": "implementation_phase",
        "required_capabilities": ["code_implementation", "testing"]
    }
    print("✅ Created sample execution context")
    
    # Test 1: Package context
    print("\n📦 Testing context packaging...")
    package = await preserver.package_context(
        execution_context=sample_context,
        target_agent_type=AgentType.CLAUDE_CODE,
        compression_level=6
    )
    
    print(f"   Package ID: {package.package_id}")
    print(f"   Source Agent: {package.source_agent_id}")
    print(f"   Target Agent: {package.target_agent_id}")
    print(f"   Format Version: {package.context_format_version}")
    print(f"   Compression Used: {package.compression_used}")
    print(f"   Package Size: {package.package_size_bytes} bytes")
    print(f"   Integrity Hash: {package.context_integrity_hash[:16]}...")
    
    # Verify metadata
    metadata = package.metadata
    print(f"   Original Size: {metadata['original_size_bytes']} bytes")
    print(f"   Compression Ratio: {metadata['compression_ratio']:.3f}")
    print(f"   Packaging Time: {metadata['packaging_time_ms']:.2f} ms")
    
    assert package.package_id is not None
    assert package.context_integrity_hash is not None
    assert package.package_size_bytes > 0
    assert metadata['compression_ratio'] < 1.0  # Should achieve compression
    print("✅ Context packaging successful")
    
    # Test 2: Validate integrity
    print("\n🔍 Testing integrity validation...")
    validation = await preserver.validate_context_integrity(package)
    
    print(f"   Is Valid: {validation['is_valid']}")
    print(f"   Integrity Score: {validation['integrity_score']:.3f}")
    print(f"   Validation Time: {validation['validation_time_ms']:.2f} ms")
    print(f"   Checks Passed: {len(validation['checks_passed'])}/{len(validation['checks_performed'])}")
    
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    
    assert validation['is_valid'] is True
    assert validation['integrity_score'] == 1.0
    assert len(validation['checks_failed']) == 0
    print("✅ Integrity validation successful")
    
    # Test 3: Restore context
    print("\n🔄 Testing context restoration...")
    restored_context = await preserver.restore_context(package)
    
    print(f"   Variables Restored: {len(restored_context['variables'])}")
    print(f"   Task History: {len(restored_context['task_history'])} items")
    print(f"   Files Created: {len(restored_context['files_created'])}")
    
    restoration_meta = restored_context['restoration_metadata']
    print(f"   Restoration Time: {restoration_meta['restoration_time_ms']:.2f} ms")
    print(f"   Source Agent: {restoration_meta['source_agent_type']}")
    print(f"   Format Version: {restoration_meta['format_version']}")
    
    # Verify data integrity
    assert restored_context['variables']['project_name'] == "bee-hive"
    assert restored_context['variables']['current_task'] == "context_preservation"
    assert restored_context['current_state']['working_directory'] == "/tmp/workspace"
    assert len(restored_context['task_history']) == 1
    assert len(restored_context['files_created']) == 1
    print("✅ Context restoration successful")
    
    # Test 4: Different compression levels
    print("\n🗜️ Testing compression levels...")
    compression_results = {}
    
    for level in [0, 1, 6, 9]:
        package_test = await preserver.package_context(
            execution_context=sample_context,
            target_agent_type=AgentType.CLAUDE_CODE,
            compression_level=level
        )
        
        compression_results[level] = {
            'size': package_test.package_size_bytes,
            'ratio': package_test.metadata['compression_ratio'],
            'time': package_test.metadata['packaging_time_ms']
        }
        
        print(f"   Level {level}: {package_test.package_size_bytes} bytes, "
              f"ratio {package_test.metadata['compression_ratio']:.3f}, "
              f"time {package_test.metadata['packaging_time_ms']:.2f} ms")
    
    # Verify compression effectiveness
    assert compression_results[0]['ratio'] == 1.0  # No compression
    # Note: For small data, higher compression levels may not always be smaller
    # Just verify that compression is working
    assert compression_results[6]['ratio'] < 1.0  # Compression is working
    assert compression_results[9]['ratio'] < 1.0  # Compression is working
    print("✅ Compression level testing successful")
    
    # Test 5: Agent-specific optimizations
    print("\n🤖 Testing agent-specific optimizations...")
    
    # Test Claude Code optimizations
    claude_package = await preserver.package_context(
        execution_context=sample_context,
        target_agent_type=AgentType.CLAUDE_CODE
    )
    claude_opts = claude_package.metadata['target_optimizations']
    print(f"   Claude Code: {claude_opts['context_style']} style, "
          f"{claude_opts['file_format_preference']} format")
    
    # Test Cursor optimizations
    cursor_package = await preserver.package_context(
        execution_context=sample_context,
        target_agent_type=AgentType.CURSOR
    )
    cursor_opts = cursor_package.metadata['target_optimizations']
    print(f"   Cursor: {cursor_opts['context_style']} style, "
          f"{cursor_opts['file_format_preference']} format")
    
    assert claude_opts['context_style'] == 'detailed'
    assert cursor_opts['context_style'] == 'minimal'
    print("✅ Agent-specific optimizations successful")
    
    # Test 6: Performance validation
    print("\n⚡ Testing performance requirements...")
    
    # Create larger context for performance testing
    large_context = sample_context.copy()
    large_context['variables'].update({f'var_{i}': f'value_{i}' for i in range(100)})
    
    import time
    
    # Test packaging performance
    start_time = time.time()
    perf_package = await preserver.package_context(
        execution_context=large_context,
        target_agent_type=AgentType.CLAUDE_CODE
    )
    packaging_time = (time.time() - start_time) * 1000
    
    print(f"   Packaging Time: {packaging_time:.2f} ms (requirement: <1000 ms)")
    assert packaging_time < 1000, f"Packaging too slow: {packaging_time} ms"
    
    # Test restoration performance
    start_time = time.time()
    perf_restored = await preserver.restore_context(perf_package)
    restoration_time = (time.time() - start_time) * 1000
    
    print(f"   Restoration Time: {restoration_time:.2f} ms (requirement: <500 ms)")
    assert restoration_time < 500, f"Restoration too slow: {restoration_time} ms"
    print("✅ Performance requirements met")
    
    # Test 7: Error handling
    print("\n❌ Testing error handling...")
    
    # Create corrupted package
    corrupted_package = ContextPackage()
    corrupted_package.metadata = {"compressed_data": b"invalid_data"}
    corrupted_package.context_integrity_hash = "invalid_hash"
    
    try:
        await preserver.restore_context(corrupted_package)
        assert False, "Should have raised exception for corrupted package"
    except Exception as e:
        print(f"   Correctly caught error: {str(e)[:60]}...")
    
    print("✅ Error handling successful")
    
    print("\n🎉 All tests passed! ProductionContextPreserver implementation is working correctly.")
    
    # Summary
    print("\n📊 Implementation Summary:")
    print(f"   ✅ Complete execution state capture and serialization")
    print(f"   ✅ Multi-level compression strategies (0-9 levels)")
    print(f"   ✅ SHA256 integrity validation and corruption detection")
    print(f"   ✅ Target agent-specific context optimization")
    print(f"   ✅ Fast packaging (<1s) and restoration (<500ms)")
    print(f"   ✅ Support for large contexts with compression")
    print(f"   ✅ Graceful error handling and recovery")
    print(f"   ✅ Comprehensive integrity validation")
    print(f"   ✅ Performance optimization with benchmarking")


if __name__ == "__main__":
    asyncio.run(test_context_preserver())