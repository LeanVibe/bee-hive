"""
Test suite for Dashboard API Infrastructure

Validates all dashboard monitoring, task management, WebSocket, and Prometheus APIs
to ensure they work correctly with the coordination system.

CRITICAL: Tests the APIs that address the 20% coordination success rate issue.
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import websockets
import httpx

# Test configuration
BASE_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000"


class DashboardAPITester:
    """Comprehensive tester for dashboard APIs."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        if success:
            self.test_results["passed"] += 1
            print(f"✅ {test_name}")
            if details:
                print(f"   {details}")
        else:
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"{test_name}: {details}")
            print(f"❌ {test_name}")
            if details:
                print(f"   Error: {details}")
    
    async def test_basic_connectivity(self):
        """Test basic API connectivity."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Components: {data.get('summary', {}).get('total', 0)}"
        except Exception as e:
            success = False
            details = str(e)
        
        self.log_test("Basic API Connectivity", success, details)
        return success
    
    async def test_agent_status_apis(self):
        """Test agent status and health APIs."""
        endpoints = [
            "/api/dashboard/agents/status",
            "/api/dashboard/agents/heartbeat"
        ]
        
        for endpoint in endpoints:
            try:
                response = await self.client.get(f"{self.base_url}{endpoint}")
                success = response.status_code == 200
                details = f"Status: {response.status_code}"
                
                if success:
                    data = response.json()
                    if endpoint.endswith("/status"):
                        agent_count = len(data.get("agents", []))
                        details += f", Agents: {agent_count}"
                    elif endpoint.endswith("/heartbeat"):
                        total = data.get("summary", {}).get("total_agents", 0)
                        details += f", Total agents: {total}"
            except Exception as e:
                success = False
                details = str(e)
            
            self.log_test(f"Agent API: {endpoint}", success, details)
    
    async def test_coordination_monitoring_apis(self):
        """Test coordination monitoring APIs."""
        endpoints = [
            "/api/dashboard/coordination/success-rate",
            "/api/dashboard/coordination/failures",
            "/api/dashboard/coordination/diagnostics"
        ]
        
        for endpoint in endpoints:
            try:
                response = await self.client.get(f"{self.base_url}{endpoint}")
                success = response.status_code == 200
                details = f"Status: {response.status_code}"
                
                if success:
                    data = response.json()
                    if endpoint.endswith("/success-rate"):
                        metrics = data.get("current_metrics", {})
                        success_rate = metrics.get("success_rate", 0)
                        details += f", Success rate: {success_rate:.1f}%"
                    elif endpoint.endswith("/failures"):
                        failure_count = data.get("total_failures", 0)
                        details += f", Total failures: {failure_count}"
                    elif endpoint.endswith("/diagnostics"):
                        overall = data.get("diagnostics", {}).get("overall_health", {})
                        health_score = overall.get("score", 0)
                        details += f", Health score: {health_score}"
            except Exception as e:
                success = False
                details = str(e)
            
            self.log_test(f"Coordination API: {endpoint}", success, details)
    
    async def test_task_distribution_apis(self):
        """Test task distribution and management APIs."""
        endpoints = [
            "/api/dashboard/tasks/queue",
            "/api/dashboard/tasks/distribution"
        ]
        
        for endpoint in endpoints:
            try:
                response = await self.client.get(f"{self.base_url}{endpoint}")
                success = response.status_code == 200
                details = f"Status: {response.status_code}"
                
                if success:
                    data = response.json()
                    if endpoint.endswith("/queue"):
                        metrics = data.get("distribution_metrics", {})
                        active_tasks = metrics.get("total_active_tasks", 0)
                        details += f", Active tasks: {active_tasks}"
                    elif endpoint.endswith("/distribution"):
                        agents = len(data.get("agent_distribution", []))
                        details += f", Agents with tasks: {agents}"
            except Exception as e:
                success = False
                details = str(e)
            
            self.log_test(f"Task API: {endpoint}", success, details)
    
    async def test_system_health_apis(self):
        """Test system health and recovery APIs."""
        endpoints = [
            "/api/dashboard/system/health",
            "/api/dashboard/logs/coordination"
        ]
        
        for endpoint in endpoints:
            try:
                response = await self.client.get(f"{self.base_url}{endpoint}")
                success = response.status_code == 200
                details = f"Status: {response.status_code}"
                
                if success:
                    data = response.json()
                    if endpoint.endswith("/health"):
                        overall = data.get("overall_health", {})
                        health_status = overall.get("status", "unknown")
                        details += f", Health: {health_status}"
                    elif endpoint.endswith("/coordination"):
                        log_count = data.get("summary", {}).get("total_errors", 0)
                        details += f", Error logs: {log_count}"
            except Exception as e:
                success = False
                details = str(e)
            
            self.log_test(f"System API: {endpoint}", success, details)
    
    async def test_prometheus_metrics(self):
        """Test Prometheus metrics endpoints."""
        endpoints = [
            "/api/dashboard/metrics",
            "/api/dashboard/metrics/coordination",
            "/api/dashboard/metrics/agents",
            "/api/dashboard/metrics/system"
        ]
        
        for endpoint in endpoints:
            try:
                response = await self.client.get(f"{self.base_url}{endpoint}")
                success = response.status_code == 200
                details = f"Status: {response.status_code}"
                
                if success:
                    content = response.text
                    # Check for Prometheus format
                    has_help = "# HELP" in content
                    has_type = "# TYPE" in content
                    has_metrics = any(line and not line.startswith("#") for line in content.split("\n"))
                    
                    if has_help and has_type and has_metrics:
                        metric_lines = len([line for line in content.split("\n") if line and not line.startswith("#")])
                        details += f", Metrics: {metric_lines} lines"
                    else:
                        success = False
                        details += ", Invalid Prometheus format"
            except Exception as e:
                success = False
                details = str(e)
            
            self.log_test(f"Prometheus API: {endpoint}", success, details)
    
    async def test_websocket_connectivity(self):
        """Test WebSocket endpoint connectivity."""
        websocket_endpoints = [
            "/api/dashboard/ws/agents",
            "/api/dashboard/ws/coordination",
            "/api/dashboard/ws/tasks",
            "/api/dashboard/ws/system",
            "/api/dashboard/ws/dashboard"
        ]
        
        for endpoint in websocket_endpoints:
            connection_id = str(uuid.uuid4())
            websocket_url = f"{WEBSOCKET_URL}{endpoint}?connection_id={connection_id}"
            
            try:
                # Test WebSocket connection
                async with websockets.connect(websocket_url, timeout=5) as websocket:
                    # Send ping message
                    ping_msg = {"type": "ping", "timestamp": datetime.utcnow().isoformat()}
                    await websocket.send(json.dumps(ping_msg))
                    
                    # Wait for response
                    response = await asyncio.wait_for(websocket.recv(), timeout=3)
                    response_data = json.loads(response)
                    
                    # Check for pong or connection established
                    if response_data.get("type") in ["pong", "connection_established"]:
                        success = True
                        details = f"Connected, Response: {response_data.get('type')}"
                    else:
                        success = False
                        details = f"Unexpected response: {response_data.get('type')}"
            
            except asyncio.TimeoutError:
                success = False
                details = "Connection timeout"
            except Exception as e:
                success = False
                details = str(e)
            
            self.log_test(f"WebSocket: {endpoint}", success, details)
    
    async def test_control_apis(self):
        """Test control and management APIs (dry run)."""
        # Test coordination reset (dry run)
        try:
            response = await self.client.post(
                f"{self.base_url}/api/dashboard/coordination/reset",
                params={"reset_type": "soft", "confirm": False}
            )
            success = response.status_code == 200
            if success:
                data = response.json()
                has_confirmation = "confirmation" in data.get("error", "").lower()
                details = "Confirmation required (expected)" if has_confirmation else "Unexpected response"
            else:
                details = f"Status: {response.status_code}"
        except Exception as e:
            success = False
            details = str(e)
        
        self.log_test("Control API: Reset (dry run)", success, details)
        
        # Test auto-recovery (dry run)
        try:
            response = await self.client.post(
                f"{self.base_url}/api/dashboard/recovery/auto-heal",
                params={"recovery_type": "smart", "dry_run": True}
            )
            success = response.status_code == 200
            if success:
                data = response.json()
                is_dry_run = data.get("dry_run", False)
                action_count = len(data.get("actions", []))
                details = f"Dry run: {is_dry_run}, Actions: {action_count}"
            else:
                details = f"Status: {response.status_code}"
        except Exception as e:
            success = False
            details = str(e)
        
        self.log_test("Control API: Auto-heal (dry run)", success, details)
    
    async def test_api_performance(self):
        """Test API response times."""
        test_endpoints = [
            "/api/dashboard/agents/status",
            "/api/dashboard/coordination/success-rate",
            "/api/dashboard/tasks/queue",
            "/api/dashboard/system/health"
        ]
        
        for endpoint in test_endpoints:
            try:
                start_time = datetime.utcnow()
                response = await self.client.get(f"{self.base_url}{endpoint}")
                end_time = datetime.utcnow()
                
                response_time_ms = (end_time - start_time).total_seconds() * 1000
                success = response.status_code == 200 and response_time_ms < 1000  # Under 1 second
                
                details = f"Response time: {response_time_ms:.0f}ms"
                if response_time_ms >= 1000:
                    details += " (SLOW)"
                
            except Exception as e:
                success = False
                details = str(e)
            
            self.log_test(f"Performance: {endpoint}", success, details)
    
    def print_summary(self):
        """Print test summary."""
        total = self.test_results["passed"] + self.test_results["failed"]
        success_rate = (self.test_results["passed"] / total) * 100 if total > 0 else 0
        
        print("\n" + "="*60)
        print("DASHBOARD API TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {total}")
        print(f"Passed: {self.test_results['passed']}")
        print(f"Failed: {self.test_results['failed']}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if self.test_results["errors"]:
            print("\nFAILED TESTS:")
            for error in self.test_results["errors"]:
                print(f"  - {error}")
        
        print("\nCRITICAL SYSTEM STATUS:")
        if success_rate >= 90:
            print("🟢 Dashboard APIs are READY for production")
        elif success_rate >= 70:
            print("🟡 Dashboard APIs have issues but are functional") 
        else:
            print("🔴 Dashboard APIs have CRITICAL issues")
        
        print("="*60)
        
        return success_rate


async def run_comprehensive_dashboard_tests():
    """Run all dashboard API tests."""
    print("🚀 Starting Comprehensive Dashboard API Tests")
    print("="*60)
    
    async with DashboardAPITester() as tester:
        # Test basic connectivity first
        if not await tester.test_basic_connectivity():
            print("❌ Basic connectivity failed. Check if the server is running.")
            return
        
        # Run all test suites
        print("\n📊 Testing Agent Status & Health APIs...")
        await tester.test_agent_status_apis()
        
        print("\n🔄 Testing Coordination Monitoring APIs...")
        await tester.test_coordination_monitoring_apis()
        
        print("\n📋 Testing Task Distribution APIs...")
        await tester.test_task_distribution_apis()
        
        print("\n🏥 Testing System Health APIs...")
        await tester.test_system_health_apis()
        
        print("\n📈 Testing Prometheus Metrics APIs...")
        await tester.test_prometheus_metrics()
        
        print("\n🔌 Testing WebSocket Connectivity...")
        await tester.test_websocket_connectivity()
        
        print("\n🎛️ Testing Control APIs...")
        await tester.test_control_apis()
        
        print("\n⚡ Testing API Performance...")
        await tester.test_api_performance()
        
        # Print summary
        success_rate = tester.print_summary()
        return success_rate


def test_specific_endpoint(endpoint: str):
    """Test a specific endpoint quickly."""
    async def _test():
        async with DashboardAPITester() as tester:
            try:
                response = await tester.client.get(f"{BASE_URL}{endpoint}")
                print(f"Status: {response.status_code}")
                
                if response.status_code == 200:
                    if endpoint.endswith("/metrics") or "/metrics/" in endpoint:
                        print(f"Content length: {len(response.text)} characters")
                        print("Content sample:")
                        print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                    else:
                        data = response.json()
                        print(f"Response keys: {list(data.keys())}")
                        print(json.dumps(data, indent=2)[:1000] + "..." if len(str(data)) > 1000 else json.dumps(data, indent=2))
                else:
                    print(f"Error: {response.text}")
                    
            except Exception as e:
                print(f"Error: {e}")
    
    asyncio.run(_test())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific endpoint
        endpoint = sys.argv[1]
        print(f"Testing endpoint: {endpoint}")
        test_specific_endpoint(endpoint)
    else:
        # Run comprehensive tests
        success_rate = asyncio.run(run_comprehensive_dashboard_tests())
        
        # Set exit code based on success rate
        if success_rate >= 90:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure