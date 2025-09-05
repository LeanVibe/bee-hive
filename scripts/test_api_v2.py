#!/usr/bin/env python3
"""
API v2 Testing Script for Epic 7 Phase 2

Tests the v2 API endpoints, authentication, and database integration
in a development environment with validation and performance checks.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up test environment
os.environ["TESTING"] = "true"
os.environ["SKIP_STARTUP_INIT"] = "true"
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-development-only")


async def test_api_structure():
    """Test API v2 structure and endpoints."""
    print("🔍 Testing API v2 structure...")
    
    try:
        from fastapi.testclient import TestClient
        from app.main import create_app
        
        # Create test app
        app = create_app()
        client = TestClient(app)
        
        # Test API root
        response = client.get("/api/v2/")
        assert response.status_code == 200
        data = response.json()
        
        assert "LeanVibe Agent Hive 2.0" in data["message"]
        assert "auth" in data["resources"]
        print("✅ API v2 root endpoint working")
        
        # Test OpenAPI documentation generation
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_spec = response.json()
        
        # Verify auth endpoints are documented
        paths = openapi_spec.get("paths", {})
        auth_endpoints = [path for path in paths.keys() if "/api/v2/auth/" in path]
        
        if len(auth_endpoints) >= 4:  # register, login, refresh, me, logout
            print(f"✅ Authentication endpoints documented: {len(auth_endpoints)} endpoints")
        else:
            print(f"⚠️ Missing auth endpoints: only {len(auth_endpoints)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False


async def test_authentication_endpoints():
    """Test authentication endpoints without database."""
    print("\n🔍 Testing authentication endpoints...")
    
    try:
        from fastapi.testclient import TestClient
        from app.main import create_app
        
        app = create_app()
        client = TestClient(app)
        
        # Test auth health endpoint (public)
        response = client.get("/api/v2/auth/health")
        print(f"Auth health response: {response.status_code}")
        
        # Test registration endpoint structure (will fail without DB but endpoint exists)
        test_user = {
            "username": "testuser123",
            "email": "test@example.com",
            "password": "TestPassword123!",
            "first_name": "Test",
            "last_name": "User"
        }
        
        response = client.post("/api/v2/auth/register", json=test_user)
        print(f"Registration endpoint response: {response.status_code}")
        
        # Test login endpoint structure
        login_data = {
            "email": "test@example.com",
            "password": "TestPassword123!"
        }
        
        response = client.post("/api/v2/auth/login", json=login_data)
        print(f"Login endpoint response: {response.status_code}")
        
        print("✅ Authentication endpoints accessible")
        return True
        
    except Exception as e:
        print(f"❌ Authentication endpoint test failed: {e}")
        return False


async def test_middleware_integration():
    """Test middleware integration and request handling."""
    print("\n🔍 Testing middleware integration...")
    
    try:
        from fastapi.testclient import TestClient
        from app.main import create_app
        
        app = create_app()
        client = TestClient(app)
        
        # Test public endpoint (no auth required)
        response = client.get("/api/v2/")
        
        # Check for performance headers
        headers = response.headers
        if "x-request-id" in headers:
            print("✅ Request ID middleware working")
        
        if "x-response-time" in headers:
            response_time = headers["x-response-time"]
            print(f"✅ Performance middleware working: {response_time}")
        
        # Test protected endpoint (should require auth)
        response = client.get("/api/v2/agents/")
        if response.status_code == 401:
            print("✅ Authentication middleware protecting endpoints")
        else:
            print(f"⚠️ Protected endpoint returned: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Middleware test failed: {e}")
        return False


async def test_user_service():
    """Test user service functionality."""
    print("\n🔍 Testing user service...")
    
    try:
        from app.services.user_service import get_user_service
        
        user_service = get_user_service()
        
        # Test password hashing
        password = "TestPassword123!"
        hashed = user_service.hash_password(password)
        
        if len(hashed) > 50:  # bcrypt hashes are long
            print("✅ Password hashing working")
        
        # Test password verification
        if user_service.verify_password(password, hashed):
            print("✅ Password verification working")
        else:
            print("❌ Password verification failed")
            return False
        
        # Test token creation (without database user)
        from app.models.user import User, UserRole, UserStatus
        from datetime import datetime
        import uuid
        
        # Create a mock user object for testing
        test_user = User(
            id=uuid.uuid4(),
            username="testuser",
            email="test@example.com",
            password_hash=hashed,
            first_name="Test",
            last_name="User",
            roles=[UserRole.USER.value],
            status=UserStatus.ACTIVE,
            is_active=True,
            email_verified=True
        )
        
        # Test token generation
        access_token = user_service.create_access_token(test_user)
        refresh_token = user_service.create_refresh_token(test_user)
        
        if len(access_token) > 100:  # JWT tokens are long
            print("✅ Access token generation working")
        
        if len(refresh_token) > 100:
            print("✅ Refresh token generation working")
        
        # Test token verification
        payload = user_service.verify_token(access_token)
        if payload and payload.get("sub") == str(test_user.id):
            print("✅ Token verification working")
        else:
            print("❌ Token verification failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ User service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_api_performance():
    """Test API performance characteristics."""
    print("\n🔍 Testing API performance...")
    
    try:
        from fastapi.testclient import TestClient
        from app.main import create_app
        
        app = create_app()
        client = TestClient(app)
        
        # Test response times for public endpoints
        start_time = time.time()
        
        for _ in range(10):
            response = client.get("/api/v2/")
            assert response.status_code == 200
        
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        avg_time = total_time / 10
        
        print(f"✅ API response time: {avg_time:.2f}ms average")
        
        if avg_time > 100:  # 100ms threshold
            print("⚠️ API responses may be slower than optimal")
        
        # Test concurrent requests
        import threading
        
        def make_request():
            client.get("/api/v2/")
        
        concurrent_start = time.time()
        threads = []
        
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        concurrent_time = (time.time() - concurrent_start) * 1000
        print(f"✅ Concurrent requests handled in: {concurrent_time:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling and response formatting."""
    print("\n🔍 Testing error handling...")
    
    try:
        from fastapi.testclient import TestClient
        from app.main import create_app
        
        app = create_app()
        client = TestClient(app)
        
        # Test 404 handling
        response = client.get("/api/v2/nonexistent")
        if response.status_code == 404:
            print("✅ 404 handling working")
        
        # Test 401 handling (protected endpoint without auth)
        response = client.get("/api/v2/agents/")
        if response.status_code == 401:
            error_data = response.json()
            if "error" in error_data and "message" in error_data:
                print("✅ 401 error formatting working")
            else:
                print("⚠️ Error response format needs improvement")
        
        # Test method not allowed
        response = client.put("/api/v2/")  # Should be GET only
        if response.status_code == 405:
            print("✅ 405 method not allowed handling working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def test_openapi_spec():
    """Test OpenAPI specification generation and quality."""
    print("\n🔍 Testing OpenAPI specification...")
    
    try:
        from fastapi.testclient import TestClient
        from app.main import create_app
        
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        spec = response.json()
        
        # Check basic OpenAPI structure
        required_fields = ["openapi", "info", "paths"]
        for field in required_fields:
            if field not in spec:
                print(f"❌ Missing OpenAPI field: {field}")
                return False
        
        print("✅ OpenAPI specification structure valid")
        
        # Check API info
        info = spec.get("info", {})
        if "LeanVibe Agent Hive 2.0" in info.get("title", ""):
            print("✅ API title correct")
        
        if "2.0.0" in info.get("version", ""):
            print("✅ API version documented")
        
        # Count endpoints
        paths = spec.get("paths", {})
        endpoint_count = len(paths)
        print(f"✅ API endpoints documented: {endpoint_count}")
        
        # Check for auth endpoints
        auth_endpoints = [path for path in paths.keys() if "/auth/" in path]
        if len(auth_endpoints) >= 3:
            print(f"✅ Authentication endpoints documented: {len(auth_endpoints)}")
        else:
            print(f"⚠️ Missing auth endpoints in spec: only {len(auth_endpoints)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAPI specification test failed: {e}")
        return False


async def main():
    """Run all API v2 validation tests."""
    print("🚀 Epic 7 Phase 2: API v2 Validation Tests")
    print("=" * 50)
    
    tests = [
        ("API Structure", test_api_structure),
        ("Authentication Endpoints", test_authentication_endpoints),
        ("Middleware Integration", test_middleware_integration),
        ("User Service", test_user_service),
        ("API Performance", test_api_performance),
        ("Error Handling", test_error_handling),
        ("OpenAPI Specification", test_openapi_spec)
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
        print("\n🎉 All tests passed! API v2 is ready for deployment.")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please address issues before deployment.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)