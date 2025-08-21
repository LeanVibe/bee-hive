#!/usr/bin/env python3
"""
Environment Configuration Validation Script

Tests basic functionality of core infrastructure components:
- Database connection
- Redis connection  
- Project Index basic functionality
"""

import asyncio
import sys
import traceback
from typing import Optional

print("=== Environment Configuration Validation ===")
print()

async def test_database_connection():
    """Test database connection and basic functionality."""
    print("🗃️  Testing Database Connection:")
    
    try:
        from app.core.database import get_session
        from sqlalchemy import text
        
        # Test getting a session (properly use async context manager)
        async with get_session() as session:
            # Test basic database query
            result = await session.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            
            if row and row.test == 1:
                print("✅ Database connection successful")
                print("✅ Basic query execution working")
                return True
            else:
                print("❌ Database query returned unexpected result")
                return False
                
    except ImportError as e:
        print(f"❌ Database import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print(f"   Note: Make sure PostgreSQL is running and DATABASE_URL is configured")
        return False

async def test_redis_connection():
    """Test Redis connection and basic functionality."""
    print("\n📡 Testing Redis Connection:")
    
    try:
        from app.core.redis import init_redis, get_redis_client
        
        # Initialize Redis first
        await init_redis()
        print("✅ Redis initialization successful")
        
        # Get Redis client
        redis_client = get_redis_client()
        
        # Test basic Redis operations
        test_key = "test:validation:key"
        test_value = "validation_test_value"
        
        # Set a test value
        await redis_client.set(test_key, test_value, expire=10)  # Expires in 10 seconds
        
        # Get the test value
        retrieved_value = await redis_client.get(test_key)
        
        if retrieved_value and retrieved_value.decode() == test_value:
            print("✅ Redis connection successful")
            print("✅ Basic Redis operations working")
            
            # Clean up
            await redis_client.delete(test_key)
            return True
        else:
            print("❌ Redis operations failed - values don't match")
            return False
            
    except ImportError as e:
        print(f"❌ Redis import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print(f"   Note: Make sure Redis is running and REDIS_URL is configured")
        return False

async def test_project_index_basic():
    """Test Project Index basic functionality."""
    print("\n🏗️  Testing Project Index Basic Functionality:")
    
    try:
        # Test core imports
        from app.project_index.core import ProjectIndexer
        from app.project_index.models import ProjectIndexConfig
        from app.project_index.websocket_events import ProjectIndexEventPublisher
        
        print("✅ Project Index core imports successful")
        
        # Test basic instantiation
        config = ProjectIndexConfig(
            project_name="test_project",
            root_path="/tmp/test",
            enable_real_time_monitoring=False,  # Don't start file monitoring
            enable_ml_analysis=False,  # Don't require ML models
            max_file_size_mb=10
        )
        print("✅ Project Index configuration creation successful")
        
        # Test event publisher instantiation (don't pass redis_client for basic test)
        publisher = ProjectIndexEventPublisher(redis_client=None)
        print("✅ Project Index event publisher creation successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Project Index import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Project Index instantiation failed: {e}")
        print(f"   Note: This may require Redis initialization for full functionality")
        return False

async def test_configuration_loading():
    """Test configuration loading."""
    print("\n⚙️  Testing Configuration Loading:")
    
    try:
        from app.core.config import settings
        
        # Test basic settings access (use correct attribute names)
        db_url = getattr(settings, 'DATABASE_URL', None)
        redis_url = getattr(settings, 'REDIS_URL', None)
        
        print("✅ Settings loading successful")
        print(f"✅ Database URL configured: {bool(db_url)}")
        print(f"✅ Redis URL configured: {bool(redis_url)}")
        
        if not db_url:
            print("⚠️  DATABASE_URL not configured - database tests may fail")
        if not redis_url:
            print("⚠️  REDIS_URL not configured - Redis tests may fail")
        
        return True
        
    except ImportError as e:
        print(f"❌ Settings import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Settings access failed: {e}")
        print(f"   Note: Make sure .env file exists with required variables")
        return False

async def main():
    """Run all environment validation tests."""
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Database Connection", test_database_connection),
        ("Redis Connection", test_redis_connection),
        ("Project Index Basic", test_project_index_basic),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 ENVIRONMENT VALIDATION SUMMARY:")
    print("="*60)
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if passed:
            passed_tests += 1
    
    print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL ENVIRONMENT TESTS PASSED - System ready for Project Index!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} environment tests failed")
        print("\nRECOMMENDATIONS:")
        
        for test_name, passed in results.items():
            if not passed:
                if "Database" in test_name:
                    print("  - Check PostgreSQL service is running")
                    print("  - Verify database URL in .env file")
                    print("  - Run database migrations: alembic upgrade head")
                elif "Redis" in test_name:
                    print("  - Check Redis service is running")
                    print("  - Verify Redis URL in .env file")
                elif "Project Index" in test_name:
                    print("  - Ensure all Project Index dependencies are installed")
                    print("  - Check for any missing imports")
                elif "Configuration" in test_name:
                    print("  - Check .env file exists and is properly configured")
                    print("  - Verify all required environment variables are set")
        
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Validation script crashed: {e}")
        traceback.print_exc()
        sys.exit(1)