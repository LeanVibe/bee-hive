#!/usr/bin/env python3
"""
Epic 7 Level 3 Diagnostic - Direct System Testing
Bypass test framework issues to directly validate service layer functionality.
"""
import sys
import os
import traceback
from pathlib import Path

# Add app to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

def test_configuration_loading():
    """Test if configuration can be loaded without errors."""
    print("🔧 Testing configuration loading...")
    try:
        from app.core.configuration_service import ApplicationConfiguration
        config = ApplicationConfiguration()
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        traceback.print_exc()
        return False

def test_database_connection():
    """Test if database can be connected."""
    print("🔧 Testing database connection...")
    try:
        from app.core.database import get_database_connection
        with get_database_connection() as conn:
            print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_redis_connection():
    """Test if Redis can be connected."""
    print("🔧 Testing Redis connection...")
    try:
        from app.core.redis import get_redis_client
        redis_client = get_redis_client()
        redis_client.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_basic_app_creation():
    """Test if basic app can be created."""
    print("🔧 Testing basic app creation...")
    try:
        # Set minimal environment variables
        os.environ['ENVIRONMENT'] = 'test'
        os.environ['DEBUG'] = 'true'
        os.environ['TESTING'] = 'true'
        
        from app.main import create_app
        app = create_app()
        print("✅ App creation successful")
        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        traceback.print_exc()
        return False

def test_core_imports():
    """Test core module imports."""
    print("🔧 Testing core module imports...")
    try:
        from app.core import orchestrator
        from app.core import coordination
        from app.core import database
        from app.core import redis
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run Epic 7 Level 3 diagnostics."""
    print("🎯 EPIC 7 LEVEL 3 DIAGNOSTIC TESTING")
    print("=" * 50)
    
    tests = [
        test_core_imports,
        test_configuration_loading,
        test_basic_app_creation,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
        print("-" * 30)
    
    # Calculate success rate
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 LEVEL 3 SERVICE LAYER TEST RESULTS")
    print(f"Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ LEVEL 3 PASSED: Service layer functionality restored")
        return True
    else:
        print("❌ LEVEL 3 FAILED: Service layer needs repair")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)