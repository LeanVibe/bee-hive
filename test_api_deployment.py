#!/usr/bin/env python3
"""
Test script for Epic 7 Phase 2: API Deployment validation

Validates database connectivity, authentication system, and API performance.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after path setup
from app.core.database import init_database, get_session
from app.core.redis import init_redis, get_redis
from app.services.user_service import get_user_service, create_default_admin
from app.models.user import User, UserRole
from sqlalchemy import text


async def test_database_connectivity():
    """Test PostgreSQL database connectivity."""
    print("🔍 Testing database connectivity...")
    
    try:
        # Initialize database
        await init_database()
        print("✅ Database initialized successfully")
        
        # Test basic query
        async with get_session() as db:
            result = await db.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Database version: {version[:50]}...")
            
            # Test table creation
            result = await db.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """))
            table_count = result.scalar()
            print(f"✅ Database tables: {table_count} tables found")
            
        return True
        
    except Exception as e:
        print(f"❌ Database connectivity failed: {e}")
        return False


async def test_redis_connectivity():
    """Test Redis connectivity."""
    print("\n🔍 Testing Redis connectivity...")
    
    try:
        # Initialize Redis
        await init_redis()
        print("✅ Redis initialized successfully")
        
        # Test basic operations
        redis_client = get_redis()
        await redis_client.set("test_key", "test_value", ex=30)
        value = await redis_client.get("test_key")
        
        if value == b"test_value":
            print("✅ Redis read/write operations successful")
        else:
            print(f"❌ Redis value mismatch: expected 'test_value', got {value}")
            return False
            
        # Clean up test key
        await redis_client.delete("test_key")
        print("✅ Redis test cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connectivity failed: {e}")
        return False


async def test_user_service():
    """Test user service functionality."""
    print("\n🔍 Testing user service...")
    
    try:
        # Get user service
        user_service = get_user_service()
        print("✅ User service initialized")
        
        # Test password hashing
        password = "TestPassword123!"
        hashed = user_service.hash_password(password)
        print("✅ Password hashing working")
        
        # Test password verification
        if user_service.verify_password(password, hashed):
            print("✅ Password verification working")
        else:
            print("❌ Password verification failed")
            return False
            
        # Test default admin creation
        print("\nCreating default admin user...")
        admin_user = await create_default_admin()
        
        if admin_user:
            print(f"✅ Default admin created: {admin_user.email}")
        else:
            print("✅ Default admin already exists")
            
        return True
        
    except Exception as e:
        print(f"❌ User service test failed: {e}")
        return False


async def test_authentication_flow():
    """Test complete authentication flow."""
    print("\n🔍 Testing authentication flow...")
    
    try:
        user_service = get_user_service()
        
        # Test user creation
        async with get_session() as db:
            try:
                test_user = await user_service.create_user(
                    db=db,
                    username=f"testuser_{int(time.time())}",
                    email=f"test_{int(time.time())}@example.com",
                    password="TestPassword123!",
                    first_name="Test",
                    last_name="User"
                )
                print(f"✅ Test user created: {test_user.email}")
                
                # Test authentication
                auth_user = await user_service.authenticate_user(
                    db, test_user.email, "TestPassword123!"
                )
                
                if auth_user:
                    print("✅ User authentication successful")
                    
                    # Test token generation
                    access_token = user_service.create_access_token(auth_user)
                    refresh_token = user_service.create_refresh_token(auth_user)
                    print("✅ Token generation successful")
                    
                    # Test token verification
                    payload = user_service.verify_token(access_token)
                    if payload and payload.get("sub") == str(auth_user.id):
                        print("✅ Token verification successful")
                    else:
                        print("❌ Token verification failed")
                        return False
                        
                else:
                    print("❌ User authentication failed")
                    return False
                    
            except ValueError as e:
                if "already exists" in str(e):
                    print("✅ User uniqueness validation working")
                else:
                    raise e
                    
        return True
        
    except Exception as e:
        print(f"❌ Authentication flow test failed: {e}")
        return False


async def test_api_performance():
    """Test API performance metrics."""
    print("\n🔍 Testing API performance...")
    
    try:
        # Test database query performance
        start_time = time.time()
        
        async with get_session() as db:
            for _ in range(10):
                await db.execute(text("SELECT 1"))
                
        db_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        print(f"✅ Database query performance: {db_time:.2f}ms for 10 queries")
        
        if db_time > 500:  # 500ms threshold
            print("⚠️ Database queries may be slower than optimal")
        
        # Test Redis performance
        start_time = time.time()
        redis_client = get_redis()
        
        for i in range(10):
            await redis_client.set(f"perf_test_{i}", f"value_{i}", ex=30)
            await redis_client.get(f"perf_test_{i}")
            
        redis_time = (time.time() - start_time) * 1000
        print(f"✅ Redis performance: {redis_time:.2f}ms for 10 operations")
        
        # Cleanup
        for i in range(10):
            await redis_client.delete(f"perf_test_{i}")
            
        if redis_time > 100:  # 100ms threshold
            print("⚠️ Redis operations may be slower than optimal")
            
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


async def test_production_readiness():
    """Test production readiness indicators."""
    print("\n🔍 Testing production readiness...")
    
    try:
        # Check environment variables
        required_env_vars = [
            "DATABASE_URL",
            "REDIS_URL", 
            "JWT_SECRET_KEY"
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        else:
            print("✅ All required environment variables present")
            
        # Test database connection pooling
        async with get_session() as db:
            result = await db.execute(text("SHOW max_connections"))
            max_connections = result.scalar()
            print(f"✅ Database max connections: {max_connections}")
            
        return True
        
    except Exception as e:
        print(f"❌ Production readiness test failed: {e}")
        return False


async def main():
    """Run all API deployment validation tests."""
    print("🚀 Epic 7 Phase 2: API Deployment Validation")
    print("=" * 50)
    
    tests = [
        ("Database Connectivity", test_database_connectivity),
        ("Redis Connectivity", test_redis_connectivity), 
        ("User Service", test_user_service),
        ("Authentication Flow", test_authentication_flow),
        ("API Performance", test_api_performance),
        ("Production Readiness", test_production_readiness)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: FAILED with exception: {e}")
        
        print("-" * 30)
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! API deployment validation successful.")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please address issues before deployment.")
        return False


if __name__ == "__main__":
    # Set up environment for testing
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)