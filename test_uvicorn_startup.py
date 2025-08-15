#!/usr/bin/env python3
"""
Test uvicorn startup with proper initialization.

This script attempts to start uvicorn with the actual FastAPI app,
but with test-friendly initialization that bypasses heavy startup.
"""

import os
import sys
import asyncio
import signal
import subprocess
import time
from pathlib import Path


def setup_test_environment():
    """Setup environment for testing."""
    os.environ.update({
        "TESTING": "true",
        "SKIP_STARTUP_INIT": "true",
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/15",  # Use test database
        "DEBUG": "true",
        "LOG_LEVEL": "INFO"
    })


def test_app_import():
    """Test if the app can be imported successfully."""
    print("🔍 Testing app import...")
    
    try:
        setup_test_environment()
        
        # Test if we can import the app module
        from app.main import app
        print(f"✅ App imported successfully")
        print(f"   Routes: {len(app.routes)}")
        print(f"   Title: {app.title}")
        return True
        
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False


def test_uvicorn_startup_minimal():
    """Test uvicorn startup with minimal configuration."""
    print("\n🔍 Testing uvicorn startup...")
    
    setup_test_environment()
    
    # Try to start uvicorn with minimal config
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "127.0.0.1",
        "--port", "8001",
        "--workers", "1",
        "--timeout-keep-alive", "5",
        "--log-level", "info",
        "--no-access-log"
    ]
    
    print(f"Starting uvicorn with: {' '.join(cmd)}")
    
    try:
        # Start uvicorn in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for startup
        print("Waiting for startup...")
        time.sleep(3)
        
        # Check if process is still running
        poll_result = process.poll()
        
        if poll_result is None:
            print("✅ uvicorn started successfully!")
            
            # Try to make a request
            try:
                import requests
                response = requests.get("http://127.0.0.1:8001/health", timeout=5)
                print(f"✅ Health check response: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   Status: {data.get('status', 'unknown')}")
                
            except requests.RequestException as e:
                print(f"⚠️ Request failed: {e}")
            except ImportError:
                print("⚠️ requests not available, skipping HTTP test")
            
            # Terminate the process
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ uvicorn shut down cleanly")
            except subprocess.TimeoutExpired:
                process.kill()
                print("⚠️ uvicorn had to be killed")
            
            return True
            
        else:
            # Process already exited
            stdout, stderr = process.communicate()
            print(f"❌ uvicorn exited with code {poll_result}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start uvicorn: {e}")
        return False


def test_hypercorn_startup():
    """Test alternative ASGI server (hypercorn)."""
    print("\n🔍 Testing hypercorn startup...")
    
    setup_test_environment()
    
    # Check if hypercorn is available
    try:
        import hypercorn
        print(f"✅ Hypercorn available: {hypercorn.__version__}")
    except ImportError:
        print("❌ Hypercorn not available, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "hypercorn"])
            import hypercorn
            print(f"✅ Hypercorn installed: {hypercorn.__version__}")
        except Exception as e:
            print(f"❌ Failed to install hypercorn: {e}")
            return False
    
    # Try to start hypercorn
    cmd = [
        sys.executable, "-m", "hypercorn",
        "app.main:app",
        "--bind", "127.0.0.1:8002",
        "--workers", "1",
        "--access-log", "-"
    ]
    
    print(f"Starting hypercorn with: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for startup
        print("Waiting for startup...")
        time.sleep(3)
        
        poll_result = process.poll()
        
        if poll_result is None:
            print("✅ hypercorn started successfully!")
            
            # Try to make a request
            try:
                import requests
                response = requests.get("http://127.0.0.1:8002/health", timeout=5)
                print(f"✅ Health check response: {response.status_code}")
                
            except requests.RequestException as e:
                print(f"⚠️ Request failed: {e}")
            except ImportError:
                print("⚠️ requests not available, skipping HTTP test")
            
            # Terminate
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ hypercorn shut down cleanly")
            except subprocess.TimeoutExpired:
                process.kill()
                print("⚠️ hypercorn had to be killed")
            
            return True
            
        else:
            stdout, stderr = process.communicate()
            print(f"❌ hypercorn exited with code {poll_result}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start hypercorn: {e}")
        return False


def test_direct_uvicorn_run():
    """Test running uvicorn directly from Python."""
    print("\n🔍 Testing direct uvicorn.run()...")
    
    setup_test_environment()
    
    try:
        import uvicorn
        from app.main import app
        
        print("Starting uvicorn.run() in background...")
        
        # Create a function to run uvicorn
        def run_server():
            try:
                uvicorn.run(
                    app,
                    host="127.0.0.1",
                    port=8003,
                    log_level="info",
                    access_log=False
                )
            except Exception as e:
                print(f"Server error: {e}")
        
        import threading
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Give it time to start
        time.sleep(3)
        
        # Check if server is responding
        try:
            import requests
            response = requests.get("http://127.0.0.1:8003/health", timeout=5)
            print(f"✅ Direct uvicorn.run() successful: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Status: {data.get('status', 'unknown')}")
                return True
            else:
                print(f"⚠️ Server responded with {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Request failed: {e}")
            return False
        except ImportError:
            print("⚠️ requests not available")
            return False
            
    except Exception as e:
        print(f"❌ Direct uvicorn.run() failed: {e}")
        return False


def main():
    """Run all startup tests."""
    print("🚀 LeanVibe Agent Hive - uvicorn Startup Testing")
    print("=" * 60)
    
    results = {}
    
    # Test 1: App import
    results['app_import'] = test_app_import()
    
    # Test 2: uvicorn subprocess
    results['uvicorn_subprocess'] = test_uvicorn_startup_minimal()
    
    # Test 3: hypercorn alternative
    results['hypercorn'] = test_hypercorn_startup()
    
    # Test 4: Direct uvicorn.run()
    results['uvicorn_direct'] = test_direct_uvicorn_run()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 STARTUP TEST RESULTS")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if results.get('uvicorn_subprocess') or results.get('hypercorn') or results.get('uvicorn_direct'):
        print("\n🎉 SUCCESS: At least one server startup method works!")
        print("\nRecommended startup command:")
        if results.get('uvicorn_subprocess'):
            print("  python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1")
        elif results.get('hypercorn'):
            print("  python -m hypercorn app.main:app --bind 127.0.0.1:8002 --workers 1")
        else:
            print("  Use uvicorn.run() directly in Python")
    else:
        print("\n❌ FAILED: No server startup method worked")
        print("The exit code 137 issue needs further investigation")
        
        if not results.get('app_import'):
            print("Issue: App cannot be imported - check dependency issues")
        else:
            print("Issue: App imports fine but servers fail to start")
            print("Possible causes:")
            print("- Memory limit exceeded")
            print("- Port conflicts")
            print("- Process killed by system")
            print("- Resource exhaustion during startup")


if __name__ == "__main__":
    main()