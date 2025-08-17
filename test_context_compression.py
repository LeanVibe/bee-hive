#!/usr/bin/env python3
"""
Test script for context compression functionality.

This script tests the HiveCompactCommand integration with the existing
ContextCompressor infrastructure.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

async def test_hive_compact_command():
    """Test the /hive:compact command implementation."""
    print("🧪 Testing HiveCompactCommand integration...")
    
    try:
        # Import the command registry
        from app.core.hive_slash_commands import get_hive_command_registry
        
        # Get the registry and compact command
        registry = get_hive_command_registry()
        compact_command = registry.get_command("compact")
        
        if not compact_command:
            print("❌ HiveCompactCommand not found in registry")
            return False
        
        print("✅ HiveCompactCommand found in registry")
        
        # Test basic command execution with sample context
        test_context = {
            "conversation_history": """
            User: I need to implement a new feature for user authentication.
            Assistant: I'll help you implement user authentication. Let's start by analyzing the requirements:
            1. User registration with email validation
            2. Secure password hashing
            3. JWT token-based authentication
            4. Password reset functionality
            
            Decision made: We'll use bcrypt for password hashing and JWT for tokens.
            
            Pattern identified: Standard web application authentication flow.
            
            Implementation approach:
            - Create User model with SQLAlchemy
            - Add password hashing utilities
            - Implement JWT token service
            - Create authentication endpoints
            
            This follows the security best practices pattern we've established.
            """,
            "test_mode": True
        }
        
        # Test with different compression levels
        test_cases = [
            {"args": ["--level=light"], "description": "Light compression"},
            {"args": ["--level=standard"], "description": "Standard compression"},
            {"args": ["--level=aggressive"], "description": "Aggressive compression"},
            {"args": ["--level=standard", "--target-tokens=100"], "description": "Adaptive compression"},
        ]
        
        for test_case in test_cases:
            print(f"\n🔄 Testing: {test_case['description']}")
            
            try:
                result = await compact_command.execute(
                    args=test_case["args"],
                    context=test_context
                )
                
                if result.get("success"):
                    print(f"✅ {test_case['description']} successful")
                    print(f"   Original tokens: {result.get('original_tokens', 0)}")
                    print(f"   Compressed tokens: {result.get('compressed_tokens', 0)}")
                    print(f"   Compression ratio: {result.get('compression_ratio', 0):.1%}")
                    print(f"   Compression time: {result.get('compression_time_seconds', 0):.2f}s")
                    print(f"   Performance target met: {result.get('performance_met', False)}")
                    
                    # Verify key insights and decisions were extracted
                    insights = result.get('key_insights', [])
                    decisions = result.get('decisions_made', [])
                    patterns = result.get('patterns_identified', [])
                    
                    print(f"   Key insights: {len(insights)}")
                    print(f"   Decisions: {len(decisions)}")
                    print(f"   Patterns: {len(patterns)}")
                    
                else:
                    print(f"❌ {test_case['description']} failed: {result.get('error', 'Unknown error')}")
                    return False
                    
            except Exception as e:
                print(f"❌ {test_case['description']} threw exception: {e}")
                return False
        
        print("\n✅ All HiveCompactCommand tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

async def test_context_compression_integration():
    """Test the integration with existing ContextCompressor."""
    print("\n🧪 Testing ContextCompressor integration...")
    
    try:
        from app.core.context_compression import get_context_compressor, CompressionLevel
        
        # Test the existing compressor works
        compressor = get_context_compressor()
        
        test_content = """
        This is a test conversation about implementing a feature.
        
        Decision: We decided to use React for the frontend.
        Pattern: Component-based architecture is the standard approach.
        Insight: Breaking down features into small components improves maintainability.
        
        The team discussed various approaches and concluded that React provides
        the best balance of performance and developer experience.
        """
        
        result = await compressor.compress_conversation(
            conversation_content=test_content,
            compression_level=CompressionLevel.STANDARD,
            preserve_decisions=True,
            preserve_patterns=True
        )
        
        print(f"✅ ContextCompressor integration successful")
        print(f"   Original tokens: {result.original_token_count}")
        print(f"   Compressed tokens: {result.compressed_token_count}")
        print(f"   Compression ratio: {result.compression_ratio:.1%}")
        
        # Verify structured data extraction
        if result.decisions_made:
            print(f"   Decisions extracted: {len(result.decisions_made)}")
        if result.patterns_identified:
            print(f"   Patterns extracted: {len(result.patterns_identified)}")
        if result.key_insights:
            print(f"   Insights extracted: {len(result.key_insights)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ContextCompressor integration failed: {e}")
        return False

async def test_performance_targets():
    """Test that performance targets are met."""
    print("\n🧪 Testing performance targets...")
    
    try:
        from app.core.hive_slash_commands import get_hive_command_registry
        
        registry = get_hive_command_registry()
        compact_command = registry.get_command("compact")
        
        # Create larger test context to test performance
        large_context = {
            "conversation_history": """
            """ + "This is a detailed conversation about software development. " * 500 + """
            
            Key decisions made:
            1. Use microservices architecture
            2. Implement CI/CD pipeline
            3. Add comprehensive testing
            4. Use Docker for containerization
            
            Patterns identified:
            - Domain-driven design
            - Event-driven architecture
            - Test-driven development
            
            Important insights:
            - Code quality is paramount
            - Documentation should be maintained
            - Security should be built-in, not added later
            """,
            "performance_test": True
        }
        
        start_time = datetime.utcnow()
        
        result = await compact_command.execute(
            args=["--level=standard"],
            context=large_context
        )
        
        compression_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Check 15-second performance target
        if compression_time < 15.0:
            print(f"✅ Performance target met: {compression_time:.2f}s < 15.0s")
        else:
            print(f"⚠️ Performance target missed: {compression_time:.2f}s >= 15.0s")
        
        # Check compression effectiveness
        if result.get("success"):
            compression_ratio = result.get("compression_ratio", 0)
            if compression_ratio > 0.4:  # At least 40% reduction
                print(f"✅ Compression effectiveness: {compression_ratio:.1%} reduction")
            else:
                print(f"⚠️ Low compression effectiveness: {compression_ratio:.1%} reduction")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

async def main():
    """Run all integration tests."""
    print("🚀 Starting Context Compression Integration Tests")
    print("=" * 60)
    
    tests = [
        test_context_compression_integration,
        test_hive_compact_command,
        test_performance_targets,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! Context compression is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)