import asyncio
#!/usr/bin/env python3
"""
Simple system test - Check basic API functionality
"""

import requests
import time
import sys
from pathlib import Path

def test_simple_health():
    """Test the basic health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Health Check Successful")
            print(f"   Status: {data.get('status', 'unknown')}")
            
            components = data.get('components', {})
            for name, info in components.items():
                status = info.get('status', 'unknown')
                print(f"   {name}: {status}")
            
            return True
        else:
            print(f"❌ API responded with status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ API health check timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def test_debug_agents():
    """Test the debug agents endpoint"""
    try:
        response = requests.get("http://localhost:8000/debug-agents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Debug Agents Endpoint Working")
            print(f"   Agent Count: {data.get('agent_count', 0)}")
            print(f"   Status: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Debug agents responded with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Debug agents check failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🧪 Running Simple System Test...")
    
    # Test basic connectivity
    health_ok = test_simple_health()
    
    if health_ok:
        # Test additional endpoints
        debug_ok = test_debug_agents()
        
        if debug_ok:
            print("🎉 All tests passed! System is operational.")
            return 0
        else:
            print("⚠️ Basic health OK, but some endpoints have issues.")
            return 1
    else:
        print("❌ System is not operational.")
        return 1

if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class SimpleTest(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            sys.exit(main())
            
            return {"status": "completed"}
    
    script_main(SimpleTest)