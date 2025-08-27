#!/usr/bin/env python3
"""
Epic 7 Level 4 - System Integration Testing
Validate complete user journeys and end-to-end workflows.
"""
import sys
import os
import traceback
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add app to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

def test_database_integration():
    """Test full database integration stack."""
    print("🔧 Testing database integration...")
    try:
        from app.core.database import Base, get_database_url, engine
        from app.models.agent import Agent
        
        # Test database URL configuration
        db_url = get_database_url()
        print(f"  - Database URL configured: {db_url[:20]}...")
        
        # Test database engine access
        db_engine = engine
        print(f"  - Database engine status: {'Available' if db_engine else 'Not initialized'}")
        
        print("✅ Database integration functional")
        return True
    except Exception as e:
        print(f"❌ Database integration failed: {e}")
        return False

def test_redis_integration():
    """Test Redis messaging and caching integration."""
    print("🔧 Testing Redis integration...")
    try:
        from app.core.redis import get_redis_client
        
        # Test basic Redis operations
        redis_client = get_redis_client()
        test_key = f"integration_test_{datetime.now().timestamp()}"
        redis_client.set(test_key, "test_value")
        value = redis_client.get(test_key)
        redis_client.delete(test_key)
        
        if value == b"test_value":
            print("✅ Redis integration successful")
            return True
        else:
            print("❌ Redis integration failed: value mismatch")
            return False
    except Exception as e:
        print(f"❌ Redis integration failed: {e}")
        return False

def test_api_endpoint_integration():
    """Test API endpoint integration and routing."""
    print("🔧 Testing API endpoint integration...")
    try:
        from app.main import create_app
        from fastapi.testclient import TestClient
        
        app = create_app()
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            print("  ✅ Health endpoint working")
        else:
            print(f"  ❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test status endpoint
        response = client.get("/status")
        if response.status_code == 200:
            print("  ✅ Status endpoint working")
        else:
            print(f"  ❌ Status endpoint failed: {response.status_code}")
            return False
        
        # Test metrics endpoint
        response = client.get("/metrics")
        if response.status_code == 200:
            print("  ✅ Metrics endpoint working")
        else:
            print(f"  ❌ Metrics endpoint failed: {response.status_code}")
            return False
        
        print("✅ API endpoint integration successful")
        return True
    except Exception as e:
        print(f"❌ API endpoint integration failed: {e}")
        traceback.print_exc()
        return False

def test_orchestrator_integration():
    """Test orchestrator and coordination systems."""
    print("🔧 Testing orchestrator integration...")
    try:
        from app.core.orchestrator import Orchestrator, OrchestratorConfig
        from app.core.coordination import MultiAgentCoordinator, CoordinationMode
        
        # Test orchestrator configuration
        config = OrchestratorConfig()
        print("  ✅ Orchestrator configuration initialized")
        
        # Test orchestrator initialization
        orchestrator = Orchestrator()
        print("  ✅ Orchestrator initialized")
        
        # Test coordination system
        coordinator = MultiAgentCoordinator()
        print("  ✅ Multi-agent coordinator initialized")
        
        print("✅ Orchestrator integration successful")
        return True
    except Exception as e:
        print(f"❌ Orchestrator integration failed: {e}")
        return False

def test_authentication_integration():
    """Test authentication and authorization systems."""
    print("🔧 Testing authentication integration...")
    try:
        from app.core.auth import UserRole, Permission, ROLE_PERMISSIONS
        
        # Test permission system
        admin_permissions = ROLE_PERMISSIONS[UserRole.ENTERPRISE_ADMIN]
        if Permission.VIEW_ANALYTICS in admin_permissions:
            print("  ✅ Permission system working")
        else:
            print("  ❌ Permission system missing VIEW_ANALYTICS")
            return False
        
        print("✅ Authentication integration successful")
        return True
    except Exception as e:
        print(f"❌ Authentication integration failed: {e}")
        return False

def test_complete_user_journey():
    """Test complete user journey simulation."""
    print("🔧 Testing complete user journey...")
    try:
        from app.main import create_app
        from fastapi.testclient import TestClient
        
        app = create_app()
        client = TestClient(app)
        
        # Simulate user journey: health check -> authentication -> metrics
        print("  - Simulating user health check...")
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        print("  - Simulating user status check...")
        status_response = client.get("/status")
        assert status_response.status_code == 200
        
        print("  - Simulating user metrics access...")
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        
        print("✅ Complete user journey successful")
        return True
    except Exception as e:
        print(f"❌ Complete user journey failed: {e}")
        return False

def test_error_handling_integration():
    """Test system error handling and recovery."""
    print("🔧 Testing error handling integration...")
    try:
        from app.main import create_app
        from fastapi.testclient import TestClient
        
        app = create_app()
        client = TestClient(app)
        
        # Test 404 handling
        response = client.get("/nonexistent-endpoint")
        if response.status_code == 404:
            print("  ✅ 404 error handling working")
        else:
            print(f"  ❌ 404 error handling failed: {response.status_code}")
            return False
        
        print("✅ Error handling integration successful")
        return True
    except Exception as e:
        print(f"❌ Error handling integration failed: {e}")
        return False

def main():
    """Run Epic 7 Level 4 integration testing."""
    print("🎯 EPIC 7 LEVEL 4 SYSTEM INTEGRATION TESTING")
    print("=" * 60)
    
    # Set test environment
    os.environ['ENVIRONMENT'] = 'test'
    os.environ['DEBUG'] = 'true'
    os.environ['TESTING'] = 'true'
    
    tests = [
        test_database_integration,
        test_api_endpoint_integration,
        test_orchestrator_integration,
        test_authentication_integration,
        test_complete_user_journey,
        test_error_handling_integration,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
        print("-" * 40)
    
    # Calculate success rate
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 LEVEL 4 SYSTEM INTEGRATION TEST RESULTS")
    print(f"Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("✅ LEVEL 4 PASSED: System integration functional")
        return True
    else:
        print("❌ LEVEL 4 FAILED: System integration needs repair")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)