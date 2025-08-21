#!/usr/bin/env python3
"""
Simple smoke test for LeanVibe Agent Hive 2.0
Tests basic functionality without requiring full server startup
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_basic_imports():
    """Test that core modules can be imported without database dependency."""
    print("🧪 Testing basic imports...")
    
    try:
        # Test configuration loading
        from app.core.config import settings
        print(f"✅ Configuration loaded: {settings.APP_NAME}")
        
        # Test database and Redis configuration
        print(f"✅ Database URL: {settings.DATABASE_URL}")
        print(f"✅ Redis URL: {settings.REDIS_URL}")
        
        # Test CLI imports
        from app.cli.main import hive_cli
        print("✅ CLI imports working")
        
        # Test API imports
        from app.main import app
        print(f"✅ FastAPI app created with {len(app.routes)} routes")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_database_connectivity():
    """Test database connectivity without table creation."""
    print("\n🧪 Testing database connectivity...")
    
    try:
        import asyncpg
        
        # Use correct asyncpg connection string
        conn = await asyncpg.connect(
            host='localhost',
            port=15432,
            user='leanvibe_user',
            password='leanvibe_secure_pass',
            database='leanvibe_agent_hive'
        )
        result = await conn.fetchval('SELECT 1')
        await conn.close()
        
        print(f"✅ Database connection successful: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

async def test_redis_connectivity():
    """Test Redis connectivity."""
    print("\n🧪 Testing Redis connectivity...")
    
    try:
        import redis
        
        # Connect to Redis
        r = redis.Redis(host='localhost', port=16379, db=0, decode_responses=True)
        result = r.ping()
        
        print(f"✅ Redis connection successful: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

async def test_cli_commands():
    """Test CLI command structure without API calls."""
    print("\n🧪 Testing CLI command structure...")
    
    try:
        from click.testing import CliRunner
        from app.cli.main import hive_cli
        
        runner = CliRunner()
        
        # Test help command
        result = runner.invoke(hive_cli, ['--help'])
        if result.exit_code == 0:
            print("✅ CLI help command working")
        else:
            print(f"❌ CLI help failed: {result.output}")
            return False
        
        # Test version command  
        result = runner.invoke(hive_cli, ['--version'])
        if result.exit_code == 0:
            print("✅ CLI version command working")
        else:
            print(f"❌ CLI version failed: {result.output}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

async def test_performance_baseline():
    """Test performance regression detection system."""
    print("\n🧪 Testing performance baseline system...")
    
    try:
        from tests.isolation.performance.performance_regression_detector import PerformanceRegressionDetector
        
        detector = PerformanceRegressionDetector()
        print("✅ Performance detector initialized")
        
        # Check if baselines exist
        if detector.baselines:
            print(f"✅ Found {len(detector.baselines)} performance baselines")
            for key in list(detector.baselines.keys())[:3]:  # Show first 3
                baseline = detector.baselines[key]
                print(f"   - {key}: {baseline.sample_count} samples")
        else:
            print("ℹ️  No performance baselines found (expected for first run)")
            
        return True
        
    except Exception as e:
        print(f"❌ Performance system test failed: {e}")
        return False

async def main():
    """Run complete smoke test suite."""
    print("🚀 LeanVibe Agent Hive 2.0 - Simple Smoke Test")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Database Connectivity", test_database_connectivity), 
        ("Redis Connectivity", test_redis_connectivity),
        ("CLI Commands", test_cli_commands),
        ("Performance System", test_performance_baseline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Smoke Test Results:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All systems operational! Basic functionality confirmed.")
        return 0
    else:
        print("⚠️  Some systems have issues. Check individual test results.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)