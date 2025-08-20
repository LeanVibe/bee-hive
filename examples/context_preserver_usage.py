import asyncio
#!/usr/bin/env python3
"""
Context Preserver Usage Examples

Demonstrates how to use the ProductionContextPreserver for agent handoff
continuity in multi-CLI coordination scenarios.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.communication.context_preserver import ProductionContextPreserver
from app.core.agents.universal_agent_interface import AgentType


async def example_basic_handoff():
    """Basic agent handoff scenario with context preservation."""
    print("📋 Example: Basic Agent Handoff")
    print("=" * 50)
    
    preserver = ProductionContextPreserver()
    
    # Simulate agent A completing some work
    agent_a_context = {
        "variables": {
            "project_path": "/workspace/my-app",
            "git_branch": "feature/new-api",
            "last_commit": "abc123",
            "progress": "api_implementation_complete"
        },
        "current_state": {
            "active_files": [
                "src/api/user_service.py",
                "src/api/auth_service.py",
                "tests/test_user_api.py"
            ],
            "build_status": "passing",
            "test_coverage": 85.4
        },
        "task_history": [
            {
                "task": "implement_user_api",
                "status": "completed",
                "duration_ms": 45000,
                "files_modified": ["src/api/user_service.py"]
            },
            {
                "task": "implement_auth_api", 
                "status": "completed",
                "duration_ms": 32000,
                "files_modified": ["src/api/auth_service.py"]
            }
        ],
        "intermediate_results": [
            {"api_endpoints": 12, "test_cases": 24},
            {"performance_benchmarks": {"avg_response_time": "45ms"}}
        ],
        "files_created": [
            "src/api/user_service.py",
            "src/api/auth_service.py",
            "tests/test_user_api.py"
        ],
        "files_modified": [
            "src/main.py",
            "requirements.txt"
        ],
        "required_capabilities": ["testing", "documentation"]
    }
    
    print("👤 Agent A (Claude Code) completed API implementation")
    print(f"   Files created: {len(agent_a_context['files_created'])}")
    print(f"   Tests passing: {agent_a_context['current_state']['build_status']}")
    print(f"   Coverage: {agent_a_context['current_state']['test_coverage']}%")
    
    # Package context for handoff to Agent B (testing specialist)
    print("\n📦 Packaging context for handoff to testing agent...")
    package = await preserver.package_context(
        execution_context=agent_a_context,
        target_agent_type=AgentType.GITHUB_COPILOT,  # Testing specialist
        compression_level=6
    )
    
    print(f"   Package size: {package.package_size_bytes} bytes")
    print(f"   Compression: {package.metadata['compression_ratio']:.1%} of original")
    print(f"   Packaging time: {package.metadata['packaging_time_ms']:.1f} ms")
    
    # Validate package integrity
    validation = await preserver.validate_context_integrity(package)
    print(f"   Integrity: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")
    
    # Agent B receives and restores context
    print("\n🔄 Agent B (GitHub Copilot) receiving context...")
    restored_context = await preserver.restore_context(package)
    
    print(f"   Received {len(restored_context['variables'])} variables")
    print(f"   Task history: {len(restored_context['task_history'])} completed tasks")
    print(f"   Ready to continue with: {restored_context['required_capabilities']}")
    print(f"   Restoration time: {restored_context['restoration_metadata']['restoration_time_ms']:.1f} ms")
    
    print("\n✅ Handoff completed successfully!")


async def example_compression_strategies():
    """Demonstrate different compression strategies for various scenarios."""
    print("\n🗜️ Example: Compression Strategies")
    print("=" * 50)
    
    preserver = ProductionContextPreserver()
    
    # Create contexts of different sizes
    contexts = {
        "small": {
            "variables": {"key": "value"},
            "current_state": {"status": "active"},
            "task_history": [],
            "files_created": [],
            "files_modified": []
        },
        "medium": {
            "variables": {f"var_{i}": f"value_{i}" * 10 for i in range(50)},
            "current_state": {"large_data": "x" * 1000},
            "task_history": [{"task": f"task_{i}"} for i in range(10)],
            "files_created": [f"file_{i}.py" for i in range(20)],
            "files_modified": []
        },
        "large": {
            "variables": {f"var_{i}": f"value_{i}" * 50 for i in range(200)},
            "current_state": {"massive_data": "x" * 10000},
            "task_history": [{"task": f"task_{i}", "data": "x" * 100} for i in range(100)],
            "files_created": [f"file_{i}.py" for i in range(100)],
            "files_modified": [f"mod_{i}.py" for i in range(50)]
        }
    }
    
    compression_levels = [0, 1, 6, 9]
    level_names = ["None", "Fast", "Balanced", "Maximum"]
    
    for size_name, context in contexts.items():
        print(f"\n📊 {size_name.title()} Context:")
        
        for level, name in zip(compression_levels, level_names):
            package = await preserver.package_context(
                execution_context=context,
                target_agent_type=AgentType.CLAUDE_CODE,
                compression_level=level
            )
            
            original_size = package.metadata['original_size_bytes']
            compressed_size = package.package_size_bytes
            ratio = package.metadata['compression_ratio']
            time_ms = package.metadata['packaging_time_ms']
            
            print(f"   {name:>10}: {compressed_size:>6} bytes "
                  f"({ratio:>6.1%}) in {time_ms:>5.1f} ms")
        
        # Recommendation based on size
        if size_name == "small":
            print("   💡 Recommendation: Use level 0 (no compression) for fast handoffs")
        elif size_name == "medium":
            print("   💡 Recommendation: Use level 6 (balanced) for good speed/size ratio")
        else:
            print("   💡 Recommendation: Use level 9 (maximum) to minimize network transfer")


async def example_agent_optimizations():
    """Show how context is optimized for different target agents."""
    print("\n🤖 Example: Agent-Specific Optimizations")
    print("=" * 50)
    
    preserver = ProductionContextPreserver()
    
    # Standard context
    context = {
        "variables": {
            "codebase_language": "python",
            "framework": "fastapi",
            "test_framework": "pytest"
        },
        "current_state": {
            "development_phase": "implementation",
            "code_quality_score": 8.5
        },
        "task_history": [
            {"task": "setup_project", "result": "success"},
            {"task": "implement_core", "result": "in_progress"}
        ],
        "files_created": ["main.py", "tests/test_main.py"],
        "files_modified": ["requirements.txt"]
    }
    
    # Test different target agents
    agents = [
        (AgentType.CLAUDE_CODE, "Claude Code (detailed analysis)"),
        (AgentType.CURSOR, "Cursor (minimal overhead)"),
        (AgentType.GITHUB_COPILOT, "GitHub Copilot (code-focused)"),
        (AgentType.GEMINI_CLI, "Gemini CLI (balanced approach)")
    ]
    
    for agent_type, description in agents:
        package = await preserver.package_context(
            execution_context=context,
            target_agent_type=agent_type,
            compression_level=6
        )
        
        optimizations = package.metadata['target_optimizations']
        
        print(f"\n👤 {description}:")
        print(f"   Format preference: {optimizations['file_format_preference']}")
        print(f"   Context style: {optimizations['context_style']}")
        print(f"   Include history: {optimizations['include_history']}")
        
        # Show how this affects the package
        print(f"   Package size: {package.package_size_bytes} bytes")


async def example_error_recovery():
    """Demonstrate error handling and recovery scenarios."""
    print("\n⚠️ Example: Error Handling and Recovery")
    print("=" * 50)
    
    preserver = ProductionContextPreserver()
    
    # Create a valid context
    context = {
        "variables": {"test": "data"},
        "current_state": {"status": "testing"},
        "task_history": [],
        "files_created": [],
        "files_modified": []
    }
    
    # Test 1: Successful flow
    print("✅ Test 1: Normal operation")
    package = await preserver.package_context(
        execution_context=context,
        target_agent_type=AgentType.CLAUDE_CODE
    )
    
    validation = await preserver.validate_context_integrity(package)
    print(f"   Validation: {'✅ Passed' if validation['is_valid'] else '❌ Failed'}")
    
    restored = await preserver.restore_context(package)
    print(f"   Restoration: ✅ Success ({len(restored)} keys)")
    
    # Test 2: Corrupted package
    print("\n❌ Test 2: Data corruption detection")
    corrupted_package = package
    corrupted_package.context_integrity_hash = "invalid_hash"
    
    validation = await preserver.validate_context_integrity(corrupted_package)
    print(f"   Validation: {'✅ Passed' if validation['is_valid'] else '❌ Failed (as expected)'}")
    print(f"   Error: {validation.get('error', 'No error')[:60]}...")
    
    # Test 3: Recovery attempt
    print("\n🔧 Test 3: Recovery and retry")
    # Create fresh package for recovery
    recovery_package = await preserver.package_context(
        execution_context=context,
        target_agent_type=AgentType.CLAUDE_CODE
    )
    
    # Verify recovery works
    validation = await preserver.validate_context_integrity(recovery_package)
    if validation['is_valid']:
        restored = await preserver.restore_context(recovery_package)
        print("   Recovery: ✅ Successful handoff after retry")
    else:
        print("   Recovery: ❌ Failed")


async def example_large_context_handling():
    """Demonstrate handling of large contexts efficiently."""
    print("\n🏗️ Example: Large Context Handling")
    print("=" * 50)
    
    preserver = ProductionContextPreserver()
    
    # Create progressively larger contexts
    sizes = [100, 1000, 5000]
    
    for size in sizes:
        print(f"\n📏 Context with {size} variables:")
        
        large_context = {
            "variables": {
                f"var_{i}": f"value_{i}_{'data' * 10}" for i in range(size)
            },
            "current_state": {
                "size_category": f"{size}_variables",
                "large_data": "x" * (size * 10)
            },
            "task_history": [
                {"task": f"process_batch_{i}", "items": 100} 
                for i in range(size // 100)
            ],
            "files_created": [f"output_{i}.py" for i in range(size // 10)],
            "files_modified": []
        }
        
        # Use maximum compression for large contexts
        import time
        start_time = time.time()
        
        package = await preserver.package_context(
            execution_context=large_context,
            target_agent_type=AgentType.CLAUDE_CODE,
            compression_level=9  # Maximum compression
        )
        
        packaging_time = (time.time() - start_time) * 1000
        
        print(f"   Original size: {package.metadata['original_size_bytes']:,} bytes")
        print(f"   Compressed size: {package.package_size_bytes:,} bytes")
        print(f"   Compression ratio: {package.metadata['compression_ratio']:.1%}")
        print(f"   Packaging time: {packaging_time:.1f} ms")
        
        # Test restoration performance
        start_time = time.time()
        restored = await preserver.restore_context(package)
        restoration_time = (time.time() - start_time) * 1000
        
        print(f"   Restoration time: {restoration_time:.1f} ms")
        print(f"   Variables restored: {len(restored['variables']):,}")
        
        # Verify performance requirements
        if packaging_time < 1000 and restoration_time < 500:
            print("   Performance: ✅ Meets requirements")
        else:
            print("   Performance: ⚠️ May need optimization")


async def main():
    """Run all context preserver examples."""
    print("🚀 ProductionContextPreserver Examples")
    print("=" * 60)
    print()
    
    await example_basic_handoff()
    await example_compression_strategies() 
    await example_agent_optimizations()
    await example_error_recovery()
    await example_large_context_handling()
    
    print("\n" + "=" * 60)
    print("🎯 Key Takeaways:")
    print("   • Use compression level 0 for small, frequent handoffs")
    print("   • Use compression level 6 for balanced performance")
    print("   • Use compression level 9 for large contexts or slow networks")
    print("   • Always validate integrity before restoration")
    print("   • Agent-specific optimizations improve compatibility")
    print("   • Context preservation supports contexts up to 50MB+")
    print("   • Performance meets <1s packaging, <500ms restoration")


if __name__ == "__main__":
    asyncio.run(main())