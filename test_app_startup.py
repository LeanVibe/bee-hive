#!/usr/bin/env python3
"""
Diagnostic script to test FastAPI app startup issues.
"""

import os
import sys
import traceback
import psutil
import gc

def test_app_creation():
    """Test if the FastAPI app can be created successfully."""
    print("🔍 Testing FastAPI app creation...")
    
    # Set testing environment to avoid heavy initialization
    os.environ["TESTING"] = "true"
    os.environ["SKIP_STARTUP_INIT"] = "true"
    
    try:
        # Check memory before import
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        print(f"Memory before import: {memory_before:.1f} MB")
        
        # Import and create app
        from app.main import create_app
        app = create_app()
        
        # Check memory after creation
        memory_after = process.memory_info().rss / 1024 / 1024
        print(f"Memory after creation: {memory_after:.1f} MB")
        print(f"Memory increase: {memory_after - memory_before:.1f} MB")
        
        # Check app properties
        print(f"✅ App created successfully")
        print(f"Routes registered: {len(app.routes)}")
        print(f"App title: {app.title}")
        print(f"App version: {app.version}")
        
        # Test basic route discovery
        route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
        health_routes = [path for path in route_paths if 'health' in path.lower()]
        api_routes = [path for path in route_paths if path.startswith('/api')]
        
        print(f"Health-related routes: {health_routes}")
        print(f"API routes count: {len(api_routes)}")
        
        return True, app
        
    except Exception as e:
        print(f"❌ Failed to create app: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return False, None

def test_minimal_uvicorn():
    """Test if uvicorn can start with minimal config."""
    print("\n🔍 Testing minimal uvicorn startup...")
    
    import socket
    
    # Check if default port is available
    def check_port(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    print(f"Port 8000 in use: {check_port(8000)}")
    print(f"Port 8001 in use: {check_port(8001)}")
    
    # Test app import in uvicorn context
    try:
        os.environ["TESTING"] = "true"
        os.environ["SKIP_STARTUP_INIT"] = "true"
        
        # This is what uvicorn does
        from app.main import app
        print(f"✅ App import successful for uvicorn")
        print(f"App instance: {type(app)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed uvicorn import: {e}")
        traceback.print_exc()
        return False

def test_testclient():
    """Test if TestClient works with the app."""
    print("\n🔍 Testing FastAPI TestClient...")
    
    try:
        from fastapi.testclient import TestClient
        from app.main import create_app
        
        os.environ["TESTING"] = "true"
        os.environ["SKIP_STARTUP_INIT"] = "true"
        
        app = create_app()
        client = TestClient(app)
        
        print("✅ TestClient created successfully")
        
        # Test basic endpoint
        response = client.get("/health")
        print(f"Health endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Health response: {data}")
        
        return True, client
        
    except Exception as e:
        print(f"❌ TestClient failed: {e}")
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    print("🚀 LeanVibe Agent Hive Startup Diagnostics")
    print("=" * 50)
    
    # Test 1: Basic app creation
    app_success, app = test_app_creation()
    
    # Test 2: Uvicorn import compatibility
    uvicorn_success = test_minimal_uvicorn()
    
    # Test 3: TestClient functionality
    testclient_success, client = test_testclient()
    
    print("\n📊 Diagnostic Summary:")
    print(f"App Creation: {'✅ SUCCESS' if app_success else '❌ FAILED'}")
    print(f"Uvicorn Import: {'✅ SUCCESS' if uvicorn_success else '❌ FAILED'}")
    print(f"TestClient: {'✅ SUCCESS' if testclient_success else '❌ FAILED'}")
    
    if all([app_success, uvicorn_success, testclient_success]):
        print("\n🎉 All diagnostics passed! The startup issue may be resource-related.")
        print("Try running uvicorn with:")
        print("  python -m uvicorn app.main:app --port 8001 --workers 1")
    else:
        print("\n⚠️ Some diagnostics failed. Check the errors above.")
        sys.exit(1)