"""
Performance & Load Testing Suite for Multi-Agent Coordination Dashboard

Tests performance requirements and system behavior under load conditions.
Validates that the dashboard can handle high coordination failure rates and 
multiple concurrent users while maintaining <100ms WebSocket latency and
responsive UI performance.

CRITICAL CONTEXT: During the 20% coordination success rate crisis, the dashboard 
must remain responsive and functional even under high stress conditions with
multiple operators trying to resolve the crisis simultaneously.

Test Coverage:
1. WebSocket Latency Validation - <100ms requirement under load
2. API Response Time Testing - Performance under stress
3. Concurrent User Load Testing - Multiple dashboard instances
4. High Coordination Failure Rate Testing - System behavior under crisis
5. Mobile Performance Testing - Touch responsiveness under load
6. Memory and Resource Usage Testing - Resource efficiency validation
"""

import asyncio
import json
import pytest
import time
import statistics
import uuid
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import websockets
import httpx
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Performance Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"
WEBSOCKET_URL = "ws://localhost:8000"

# Performance Thresholds
WEBSOCKET_LATENCY_THRESHOLD_MS = 100  # <100ms WebSocket latency requirement
API_RESPONSE_THRESHOLD_MS = 1000      # <1s API response time requirement
UI_LOAD_THRESHOLD_MS = 1000           # <1s UI load time requirement
MEMORY_USAGE_THRESHOLD_MB = 512       # <512MB memory usage per dashboard instance

# Load Testing Configuration
CONCURRENT_USERS = [5, 10, 20, 50]    # Concurrent user scenarios
STRESS_DURATION_SECONDS = 30          # Duration for sustained load tests
WEBSOCKET_PING_FREQUENCY = 1          # Ping every second during load tests


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    test_name: str
    latency_ms: List[float] = field(default_factory=list)
    response_times_ms: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    concurrent_users: int = 1
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latency_ms) if self.latency_ms else 0.0
    
    @property
    def p95_latency(self) -> float:
        return statistics.quantiles(self.latency_ms, n=20)[18] if len(self.latency_ms) >= 20 else max(self.latency_ms) if self.latency_ms else 0.0
    
    @property
    def p99_latency(self) -> float:
        return statistics.quantiles(self.latency_ms, n=100)[98] if len(self.latency_ms) >= 100 else max(self.latency_ms) if self.latency_ms else 0.0
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times_ms) if self.response_times_ms else 0.0
    
    @property
    def max_memory_usage(self) -> float:
        return max(self.memory_usage_mb) if self.memory_usage_mb else 0.0
    
    @property
    def avg_cpu_usage(self) -> float:
        return statistics.mean(self.cpu_usage_percent) if self.cpu_usage_percent else 0.0


@dataclass
class LoadTestResult:
    """Load test result tracking."""
    test_name: str
    success: bool
    metrics: PerformanceMetrics
    details: str = ""
    critical: bool = False


@dataclass
class PerformanceTestSuite:
    """Performance test suite aggregator."""
    name: str
    results: List[LoadTestResult] = field(default_factory=list)
    
    def add_result(self, result: LoadTestResult):
        self.results.append(result)
    
    @property
    def passed(self) -> int:
        return len([r for r in self.results if r.success])
    
    @property
    def failed(self) -> int:
        return len([r for r in self.results if not r.success])
    
    @property
    def critical_failures(self) -> int:
        return len([r for r in self.results if not r.success and r.critical])
    
    @property
    def success_rate(self) -> float:
        total = len(self.results)
        return (self.passed / total * 100) if total > 0 else 0.0


class DashboardPerformanceTester:
    """Comprehensive performance tester for dashboard systems."""
    
    def __init__(self, backend_url: str = BACKEND_URL, frontend_url: str = FRONTEND_URL, websocket_url: str = WEBSOCKET_URL):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.websocket_url = websocket_url
        self.test_suites: Dict[str, PerformanceTestSuite] = {}
        self.browser: Optional[Browser] = None
    
    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    def get_suite(self, suite_name: str) -> PerformanceTestSuite:
        """Get or create test suite."""
        if suite_name not in self.test_suites:
            self.test_suites[suite_name] = PerformanceTestSuite(suite_name)
        return self.test_suites[suite_name]
    
    def collect_system_metrics(self) -> Tuple[float, float]:
        """Collect current system metrics."""
        try:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return memory_mb, cpu_percent
        except:
            return 0.0, 0.0
    
    async def measure_websocket_latency(self, websocket_url: str, duration_seconds: int = 10) -> List[float]:
        """Measure WebSocket latency over a period."""
        latencies = []
        
        try:
            async with websockets.connect(websocket_url, timeout=10) as websocket:
                end_time = time.time() + duration_seconds
                
                while time.time() < end_time:
                    ping_start = time.perf_counter()
                    
                    ping_msg = {
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    await websocket.send(json.dumps(ping_msg))
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    
                    ping_end = time.perf_counter()
                    latency_ms = (ping_end - ping_start) * 1000
                    latencies.append(latency_ms)
                    
                    # Wait before next ping
                    await asyncio.sleep(WEBSOCKET_PING_FREQUENCY)
                    
        except Exception as e:
            # Return empty list on failure
            pass
        
        return latencies
    
    # ===== WEBSOCKET PERFORMANCE TESTING =====
    
    async def test_websocket_latency_under_load(self):
        """Test WebSocket latency under various load conditions."""
        suite = self.get_suite("WebSocket Performance")
        
        websocket_endpoints = [
            ("/api/dashboard/ws/coordination", "Coordination WebSocket", True),
            ("/api/dashboard/ws/agents", "Agent Status WebSocket", False),
            ("/api/dashboard/ws/system", "System Health WebSocket", False),
            ("/api/dashboard/ws/dashboard", "Main Dashboard WebSocket", True)
        ]
        
        for endpoint, name, is_critical in websocket_endpoints:
            for concurrent_connections in [1, 5, 10, 25]:
                try:
                    websocket_url = f"{self.websocket_url}{endpoint}"
                    
                    # Run concurrent WebSocket connections
                    tasks = []
                    for i in range(concurrent_connections):
                        connection_url = f"{websocket_url}?connection_id={uuid.uuid4()}"
                        tasks.append(self.measure_websocket_latency(connection_url, duration_seconds=10))
                    
                    # Collect system metrics during test
                    start_memory, start_cpu = self.collect_system_metrics()
                    start_time = time.perf_counter()
                    
                    # Execute all connections concurrently
                    latency_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    end_time = time.perf_counter()
                    end_memory, end_cpu = self.collect_system_metrics()
                    
                    # Process results
                    all_latencies = []
                    successful_connections = 0
                    
                    for result in latency_results:
                        if isinstance(result, list) and result:
                            all_latencies.extend(result)
                            successful_connections += 1
                    
                    # Calculate metrics
                    metrics = PerformanceMetrics(
                        test_name=f"{name} - {concurrent_connections} connections",
                        latency_ms=all_latencies,
                        memory_usage_mb=[start_memory, end_memory],
                        cpu_usage_percent=[start_cpu, end_cpu],
                        concurrent_users=concurrent_connections
                    )
                    
                    # Determine success
                    avg_latency = metrics.avg_latency
                    p95_latency = metrics.p95_latency
                    connection_success_rate = successful_connections / concurrent_connections
                    
                    success = (
                        avg_latency <= WEBSOCKET_LATENCY_THRESHOLD_MS and
                        p95_latency <= WEBSOCKET_LATENCY_THRESHOLD_MS * 2 and
                        connection_success_rate >= 0.9  # 90% connection success rate
                    )
                    
                    details = f"Avg: {avg_latency:.1f}ms, P95: {p95_latency:.1f}ms, Connections: {successful_connections}/{concurrent_connections}"
                    
                    if avg_latency > WEBSOCKET_LATENCY_THRESHOLD_MS:
                        details += f" (EXCEEDS {WEBSOCKET_LATENCY_THRESHOLD_MS}ms THRESHOLD)"
                    
                except Exception as e:
                    success = False
                    metrics = PerformanceMetrics(test_name=f"{name} - {concurrent_connections} connections")
                    details = f"WebSocket latency test failed: {str(e)}"
                
                result = LoadTestResult(
                    test_name=f"{name} Latency ({concurrent_connections} concurrent)",
                    success=success,
                    metrics=metrics,
                    details=details,
                    critical=is_critical
                )
                suite.add_result(result)
    
    async def test_api_performance_under_load(self):
        """Test API performance under concurrent load conditions."""
        suite = self.get_suite("API Performance Under Load")
        
        # Critical API endpoints for performance testing
        api_endpoints = [
            ("/api/dashboard/coordination/success-rate", "GET", "Coordination Success Rate", True),
            ("/api/dashboard/agents/status", "GET", "Agent Status", True),
            ("/api/dashboard/tasks/queue", "GET", "Task Queue Status", False),
            ("/api/dashboard/system/health", "GET", "System Health", False),
            ("/api/dashboard/metrics/coordination", "GET", "Coordination Metrics", False)
        ]
        
        for endpoint, method, name, is_critical in api_endpoints:
            for concurrent_requests in [10, 25, 50, 100]:
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        
                        async def make_request():
                            start_time = time.perf_counter()
                            try:
                                response = await client.get(f"{self.backend_url}{endpoint}")
                                end_time = time.perf_counter()
                                response_time_ms = (end_time - start_time) * 1000
                                return {
                                    "success": response.status_code == 200,
                                    "response_time_ms": response_time_ms,
                                    "status_code": response.status_code
                                }
                            except Exception as e:
                                end_time = time.perf_counter()
                                return {
                                    "success": False,
                                    "response_time_ms": (end_time - start_time) * 1000,
                                    "error": str(e)
                                }
                        
                        # Collect system metrics
                        start_memory, start_cpu = self.collect_system_metrics()
                        
                        # Execute concurrent requests
                        start_time = time.perf_counter()
                        tasks = [make_request() for _ in range(concurrent_requests)]
                        results = await asyncio.gather(*tasks)
                        end_time = time.perf_counter()
                        
                        end_memory, end_cpu = self.collect_system_metrics()
                        
                        # Process results
                        successful_requests = len([r for r in results if r["success"]])
                        response_times = [r["response_time_ms"] for r in results]
                        error_rate = (concurrent_requests - successful_requests) / concurrent_requests
                        
                        total_duration = end_time - start_time
                        throughput_rps = concurrent_requests / total_duration if total_duration > 0 else 0
                        
                        # Calculate metrics
                        metrics = PerformanceMetrics(
                            test_name=f"{name} - {concurrent_requests} concurrent requests",
                            response_times_ms=response_times,
                            memory_usage_mb=[start_memory, end_memory],
                            cpu_usage_percent=[start_cpu, end_cpu],
                            error_rate=error_rate,
                            throughput_rps=throughput_rps,
                            concurrent_users=concurrent_requests
                        )
                        
                        # Determine success
                        avg_response_time = metrics.avg_response_time
                        success = (
                            avg_response_time <= API_RESPONSE_THRESHOLD_MS and
                            error_rate <= 0.05 and  # 5% error rate tolerance
                            successful_requests >= concurrent_requests * 0.95  # 95% success rate
                        )
                        
                        details = f"Avg: {avg_response_time:.1f}ms, Success: {successful_requests}/{concurrent_requests}, RPS: {throughput_rps:.1f}"
                        
                        if avg_response_time > API_RESPONSE_THRESHOLD_MS:
                            details += f" (EXCEEDS {API_RESPONSE_THRESHOLD_MS}ms THRESHOLD)"
                        if error_rate > 0.05:
                            details += f" (HIGH ERROR RATE: {error_rate*100:.1f}%)"
                
                except Exception as e:
                    success = False
                    metrics = PerformanceMetrics(test_name=f"{name} - {concurrent_requests} concurrent requests")
                    details = f"API performance test failed: {str(e)}"
                
                result = LoadTestResult(
                    test_name=f"{name} Load Test ({concurrent_requests} concurrent)",
                    success=success,
                    metrics=metrics,
                    details=details,
                    critical=is_critical
                )
                suite.add_result(result)
    
    async def test_concurrent_dashboard_instances(self):
        """Test multiple concurrent dashboard instances (simulating multiple operators)."""
        suite = self.get_suite("Concurrent Dashboard Users")
        
        for user_count in CONCURRENT_USERS:
            try:
                # Create multiple browser contexts (simulating different users)
                contexts = []
                pages = []
                
                for i in range(user_count):
                    context = await self.browser.new_context(
                        viewport={"width": 1920, "height": 1080}
                    )
                    contexts.append(context)
                    
                    page = await context.new_page()
                    pages.append(page)
                
                # Collect baseline metrics
                start_memory, start_cpu = self.collect_system_metrics()
                start_time = time.perf_counter()
                
                # Load dashboard on all instances simultaneously
                load_tasks = []
                for page in pages:
                    load_tasks.append(page.goto(self.frontend_url, wait_until="networkidle"))
                
                await asyncio.gather(*load_tasks, return_exceptions=True)
                
                # Simulate user interactions on all dashboards
                interaction_tasks = []
                for i, page in enumerate(pages):
                    # Simulate different user behaviors
                    if i % 3 == 0:
                        # User focusing on agent monitoring
                        interaction_tasks.append(self._simulate_agent_monitoring_user(page))
                    elif i % 3 == 1:
                        # User focusing on task management
                        interaction_tasks.append(self._simulate_task_management_user(page))
                    else:
                        # User focusing on emergency controls
                        interaction_tasks.append(self._simulate_emergency_control_user(page))
                
                # Run interactions for test duration
                await asyncio.wait_for(
                    asyncio.gather(*interaction_tasks, return_exceptions=True),
                    timeout=STRESS_DURATION_SECONDS
                )
                
                end_time = time.perf_counter()
                end_memory, end_cpu = self.collect_system_metrics()
                
                # Calculate metrics
                total_duration_ms = (end_time - start_time) * 1000
                memory_per_user = (end_memory - start_memory) / user_count if user_count > 0 else 0
                
                metrics = PerformanceMetrics(
                    test_name=f"Concurrent Dashboard Users - {user_count} users",
                    memory_usage_mb=[start_memory, end_memory],
                    cpu_usage_percent=[start_cpu, end_cpu],
                    concurrent_users=user_count
                )
                
                # Determine success
                success = (
                    end_memory < MEMORY_USAGE_THRESHOLD_MB * user_count and
                    memory_per_user < MEMORY_USAGE_THRESHOLD_MB and
                    end_cpu < 80.0  # CPU usage should be reasonable
                )
                
                details = f"Users: {user_count}, Memory: {end_memory:.1f}MB ({memory_per_user:.1f}MB/user), CPU: {end_cpu:.1f}%"
                
                if memory_per_user > MEMORY_USAGE_THRESHOLD_MB:
                    details += f" (EXCEEDS {MEMORY_USAGE_THRESHOLD_MB}MB/USER THRESHOLD)"
                
                # Cleanup
                for context in contexts:
                    await context.close()
                
            except Exception as e:
                success = False
                metrics = PerformanceMetrics(test_name=f"Concurrent Dashboard Users - {user_count} users")
                details = f"Concurrent user test failed: {str(e)}"
                
                # Cleanup on error
                for context in contexts:
                    try:
                        await context.close()
                    except:
                        pass
            
            result = LoadTestResult(
                test_name=f"Concurrent Dashboard Users ({user_count} users)",
                success=success,
                metrics=metrics,
                details=details,
                critical=user_count >= 10  # Critical for 10+ users
            )
            suite.add_result(result)
    
    async def _simulate_agent_monitoring_user(self, page: Page):
        """Simulate user focusing on agent monitoring."""
        try:
            for _ in range(10):  # 10 interactions over test period
                # Click on agent cards
                agent_cards = page.locator(".agent-card, [data-testid*='agent']")
                if await agent_cards.count() > 0:
                    await agent_cards.first.click()
                
                # Check agent details
                await asyncio.sleep(2)
                
                # Scroll through agent list
                await page.keyboard.press("ArrowDown")
                await asyncio.sleep(1)
                
        except Exception:
            pass  # Ignore errors in simulation
    
    async def _simulate_task_management_user(self, page: Page):
        """Simulate user focusing on task management."""
        try:
            for _ in range(8):  # 8 interactions over test period
                # Check task queue
                task_elements = page.locator(".task-card, [data-testid*='task']")
                if await task_elements.count() > 0:
                    await task_elements.first.click()
                
                await asyncio.sleep(2)
                
                # Scroll through tasks
                await page.mouse.wheel(0, 100)
                await asyncio.sleep(1.5)
                
        except Exception:
            pass  # Ignore errors in simulation
    
    async def _simulate_emergency_control_user(self, page: Page):
        """Simulate user focusing on emergency controls."""
        try:
            for _ in range(5):  # 5 interactions over test period
                # Check emergency controls (without actually executing)
                emergency_controls = page.locator("[data-testid*='emergency'], .emergency-controls")
                if await emergency_controls.count() > 0:
                    # Hover over emergency controls
                    await emergency_controls.first.hover()
                
                await asyncio.sleep(3)
                
                # Check system health
                health_elements = page.locator("[data-testid*='health'], .system-health")
                if await health_elements.count() > 0:
                    await health_elements.first.click()
                
                await asyncio.sleep(3)
                
        except Exception:
            pass  # Ignore errors in simulation
    
    async def test_high_coordination_failure_simulation(self):
        """Test system performance during high coordination failure rates."""
        suite = self.get_suite("High Failure Rate Performance")
        
        # Simulate different crisis scenarios
        crisis_scenarios = [
            {"name": "Moderate Crisis", "simulated_success_rate": 60, "expected_load": "medium"},
            {"name": "Severe Crisis", "simulated_success_rate": 30, "expected_load": "high"},
            {"name": "Critical Crisis", "simulated_success_rate": 15, "expected_load": "critical"}
        ]
        
        for scenario in crisis_scenarios:
            try:
                # Create dashboard instance
                context = await self.browser.new_context(viewport={"width": 1920, "height": 1080})
                page = await context.new_page()
                
                start_memory, start_cpu = self.collect_system_metrics()
                start_time = time.perf_counter()
                
                # Load dashboard
                await page.goto(self.frontend_url, wait_until="networkidle")
                
                # Simulate crisis scenario by rapidly checking coordination status
                crisis_interactions = []
                for _ in range(20):  # High frequency monitoring during crisis
                    crisis_interactions.append(self._simulate_crisis_monitoring(page))
                    await asyncio.sleep(1)  # 1 second intervals
                
                await asyncio.gather(*crisis_interactions, return_exceptions=True)
                
                end_time = time.perf_counter()
                end_memory, end_cpu = self.collect_system_metrics()
                
                # Calculate metrics
                total_duration_ms = (end_time - start_time) * 1000
                
                metrics = PerformanceMetrics(
                    test_name=scenario["name"],
                    memory_usage_mb=[start_memory, end_memory],
                    cpu_usage_percent=[start_cpu, end_cpu]
                )
                
                # Determine success based on system remaining responsive
                success = (
                    end_memory < MEMORY_USAGE_THRESHOLD_MB * 2 and  # Allow more memory during crisis
                    end_cpu < 90.0 and  # Allow higher CPU during crisis
                    total_duration_ms < 30000  # Should complete within 30 seconds
                )
                
                details = f"Scenario: {scenario['name']}, Memory: {end_memory:.1f}MB, CPU: {end_cpu:.1f}%, Duration: {total_duration_ms:.1f}ms"
                
                await context.close()
                
            except Exception as e:
                success = False
                metrics = PerformanceMetrics(test_name=scenario["name"])
                details = f"Crisis simulation failed: {str(e)}"
                
                try:
                    await context.close()
                except:
                    pass
            
            result = LoadTestResult(
                test_name=f"High Failure Rate - {scenario['name']}",
                success=success,
                metrics=metrics,
                details=details,
                critical=scenario["name"] == "Critical Crisis"
            )
            suite.add_result(result)
    
    async def _simulate_crisis_monitoring(self, page: Page):
        """Simulate intensive crisis monitoring behavior."""
        try:
            # Check success rate frequently
            success_elements = page.locator("[data-testid*='success-rate'], .success-rate")
            if await success_elements.count() > 0:
                await success_elements.first.click()
            
            # Check emergency controls
            emergency_elements = page.locator("[data-testid*='emergency'], .emergency-controls")
            if await emergency_elements.count() > 0:
                await emergency_elements.first.hover()
            
            # Check system health
            health_elements = page.locator("[data-testid*='health'], .system-health")
            if await health_elements.count() > 0:
                await health_elements.first.click()
                
        except Exception:
            pass  # Ignore errors in simulation
    
    async def test_mobile_performance_under_load(self):
        """Test mobile dashboard performance under load conditions."""
        suite = self.get_suite("Mobile Performance Under Load")
        
        mobile_viewports = [
            {"width": 375, "height": 667, "name": "iPhone SE"},
            {"width": 414, "height": 896, "name": "iPhone XR"},
            {"width": 768, "height": 1024, "name": "iPad"}
        ]
        
        for viewport in mobile_viewports:
            try:
                # Create mobile context
                mobile_context = await self.browser.new_context(
                    viewport={"width": viewport["width"], "height": viewport["height"]},
                    user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
                )
                
                page = await mobile_context.new_page()
                
                start_memory, start_cpu = self.collect_system_metrics()
                start_time = time.perf_counter()
                
                # Load mobile dashboard
                await page.goto(self.frontend_url, wait_until="networkidle")
                
                load_end_time = time.perf_counter()
                load_time_ms = (load_end_time - start_time) * 1000
                
                # Simulate mobile touch interactions
                touch_interactions = []
                for _ in range(10):
                    touch_interactions.append(self._simulate_mobile_touch_interactions(page))
                    await asyncio.sleep(2)  # 2 second intervals
                
                await asyncio.gather(*touch_interactions, return_exceptions=True)
                
                end_time = time.perf_counter()
                end_memory, end_cpu = self.collect_system_metrics()
                
                # Calculate metrics
                metrics = PerformanceMetrics(
                    test_name=f"Mobile {viewport['name']}",
                    response_times_ms=[load_time_ms],
                    memory_usage_mb=[start_memory, end_memory],
                    cpu_usage_percent=[start_cpu, end_cpu]
                )
                
                # Determine success
                success = (
                    load_time_ms <= UI_LOAD_THRESHOLD_MS and
                    end_memory < MEMORY_USAGE_THRESHOLD_MB and
                    end_cpu < 70.0  # Lower CPU threshold for mobile
                )
                
                details = f"Device: {viewport['name']}, Load: {load_time_ms:.1f}ms, Memory: {end_memory:.1f}MB, CPU: {end_cpu:.1f}%"
                
                if load_time_ms > UI_LOAD_THRESHOLD_MS:
                    details += f" (EXCEEDS {UI_LOAD_THRESHOLD_MS}ms LOAD THRESHOLD)"
                
                await mobile_context.close()
                
            except Exception as e:
                success = False
                metrics = PerformanceMetrics(test_name=f"Mobile {viewport['name']}")
                details = f"Mobile performance test failed: {str(e)}"
                
                try:
                    await mobile_context.close()
                except:
                    pass
            
            result = LoadTestResult(
                test_name=f"Mobile Performance - {viewport['name']}",
                success=success,
                metrics=metrics,
                details=details,
                critical=viewport["name"] == "iPhone SE"  # Critical for smallest viewport
            )
            suite.add_result(result)
    
    async def _simulate_mobile_touch_interactions(self, page: Page):
        """Simulate mobile touch interactions."""
        try:
            # Tap on dashboard elements
            touchable_elements = await page.locator("button, .card, [role='button']").all()
            if touchable_elements:
                await touchable_elements[0].tap()
            
            # Swipe gestures
            await page.touch_screen.swipe(200, 300, 200, 100)  # Swipe up
            await asyncio.sleep(0.5)
            
            # Pinch to zoom (if supported)
            await page.touch_screen.tap(400, 400)
            
        except Exception:
            pass  # Ignore errors in simulation
    
    # ===== COMPREHENSIVE TEST EXECUTION =====
    
    async def run_all_performance_tests(self) -> Dict[str, PerformanceTestSuite]:
        """Run all comprehensive performance tests."""
        print("⚡ Starting Comprehensive Performance Testing")
        print("="*50)
        print(f"Thresholds: WebSocket <{WEBSOCKET_LATENCY_THRESHOLD_MS}ms, API <{API_RESPONSE_THRESHOLD_MS}ms, UI <{UI_LOAD_THRESHOLD_MS}ms")
        print(f"Load Testing: {CONCURRENT_USERS} concurrent users, {STRESS_DURATION_SECONDS}s duration")
        print("="*50)
        
        # Test 1: WebSocket Latency Under Load
        print("\n🔌 Testing WebSocket Latency Under Load...")
        await self.test_websocket_latency_under_load()
        
        # Test 2: API Performance Under Load
        print("\n📡 Testing API Performance Under Load...")
        await self.test_api_performance_under_load()
        
        # Test 3: Concurrent Dashboard Instances
        print("\n👥 Testing Concurrent Dashboard Users...")
        await self.test_concurrent_dashboard_instances()
        
        # Test 4: High Coordination Failure Simulation
        print("\n🚨 Testing High Coordination Failure Performance...")
        await self.test_high_coordination_failure_simulation()
        
        # Test 5: Mobile Performance Under Load
        print("\n📱 Testing Mobile Performance Under Load...")
        await self.test_mobile_performance_under_load()
        
        return self.test_suites
    
    def print_performance_test_report(self):
        """Print comprehensive performance test report."""
        print("\n" + "="*70)
        print("PERFORMANCE & LOAD TESTING REPORT")
        print("="*70)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_critical_failures = 0
        
        for suite_name, suite in self.test_suites.items():
            print(f"\n⚡ {suite.name}")
            print(f"   Tests: {len(suite.results)} | Passed: {suite.passed} | Failed: {suite.failed}")
            print(f"   Success Rate: {suite.success_rate:.1f}%")
            
            if suite.critical_failures > 0:
                print(f"   🔴 CRITICAL FAILURES: {suite.critical_failures}")
            
            # Show performance statistics
            all_metrics = [r.metrics for r in suite.results]
            if all_metrics:
                avg_latencies = [m.avg_latency for m in all_metrics if m.latency_ms]
                avg_response_times = [m.avg_response_time for m in all_metrics if m.response_times_ms]
                max_memories = [m.max_memory_usage for m in all_metrics if m.memory_usage_mb]
                
                if avg_latencies:
                    print(f"   Avg WebSocket Latency: {statistics.mean(avg_latencies):.1f}ms")
                if avg_response_times:
                    print(f"   Avg API Response Time: {statistics.mean(avg_response_times):.1f}ms")
                if max_memories:
                    print(f"   Max Memory Usage: {max(max_memories):.1f}MB")
            
            # Show failed tests
            failed_tests = [r for r in suite.results if not r.success]
            if failed_tests:
                print("   Failed Tests:")
                for test in failed_tests[:2]:  # Show first 2 failures
                    prefix = "🔴" if test.critical else "❌"
                    print(f"     {prefix} {test.test_name}: {test.details[:70]}...")
                if len(failed_tests) > 2:
                    print(f"     ... and {len(failed_tests) - 2} more")
            
            total_tests += len(suite.results)
            total_passed += suite.passed
            total_failed += suite.failed
            total_critical_failures += suite.critical_failures
        
        # Overall summary
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n{'='*70}")
        print("PERFORMANCE SYSTEM STATUS")
        print(f"{'='*70}")
        print(f"Total Performance Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {overall_success_rate:.1f}%")
        print(f"Critical Failures: {total_critical_failures}")
        
        # Performance readiness assessment
        print(f"\n🎯 PERFORMANCE PRODUCTION READINESS:")
        
        if total_critical_failures == 0 and overall_success_rate >= 85:
            print("🟢 PERFORMANCE READY - All critical performance requirements met")
            status = "READY"
        elif total_critical_failures <= 2 and overall_success_rate >= 75:
            print("🟡 PERFORMANCE ACCEPTABLE - Most requirements met")
            status = "ACCEPTABLE"
        elif total_critical_failures <= 5 and overall_success_rate >= 65:
            print("🟠 PERFORMANCE NEEDS OPTIMIZATION - Some issues present")
            status = "NEEDS_OPTIMIZATION"
        else:
            print("🔴 PERFORMANCE NOT READY - Critical performance failures")
            status = "NOT_READY"
        
        print(f"\n📊 COORDINATION CRISIS PERFORMANCE STATUS:")
        
        # Check crisis-specific performance
        websocket_suite = self.test_suites.get("WebSocket Performance")
        crisis_suite = self.test_suites.get("High Failure Rate Performance")
        concurrent_suite = self.test_suites.get("Concurrent Dashboard Users")
        
        crisis_performance_ready = True
        if websocket_suite and websocket_suite.critical_failures > 0:
            crisis_performance_ready = False
        if crisis_suite and crisis_suite.critical_failures > 0:
            crisis_performance_ready = False
        if concurrent_suite and concurrent_suite.critical_failures > 2:
            crisis_performance_ready = False
        
        if crisis_performance_ready:
            print("✅ Dashboard performance ready for coordination crisis management")
        else:
            print("❌ Performance issues may hinder coordination crisis resolution")
        
        print(f"{'='*70}")
        
        return {
            "overall_success_rate": overall_success_rate,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "critical_failures": total_critical_failures,
            "performance_status": status,
            "crisis_performance_ready": crisis_performance_ready,
            "suites": {name: {
                "success_rate": suite.success_rate,
                "tests": len(suite.results),
                "passed": suite.passed,
                "failed": suite.failed,
                "critical_failures": suite.critical_failures
            } for name, suite in self.test_suites.items()}
        }


# ===== TEST EXECUTION FUNCTIONS =====

async def run_performance_load_validation():
    """Run comprehensive performance and load validation."""
    async with DashboardPerformanceTester() as tester:
        await tester.run_all_performance_tests()
        return tester.print_performance_test_report()


async def run_websocket_performance_only():
    """Run only WebSocket performance tests."""
    print("🔌 Running WebSocket Performance Tests Only")
    print("="*40)
    
    async with DashboardPerformanceTester() as tester:
        await tester.test_websocket_latency_under_load()
        return tester.print_performance_test_report()


async def run_crisis_performance_only():
    """Run only crisis performance tests."""
    print("🚨 Running Crisis Performance Tests Only")
    print("="*35)
    
    async with DashboardPerformanceTester() as tester:
        await tester.test_high_coordination_failure_simulation()
        await tester.test_concurrent_dashboard_instances()
        return tester.print_performance_test_report()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "websocket":
            result = asyncio.run(run_websocket_performance_only())
        elif test_type == "crisis":
            result = asyncio.run(run_crisis_performance_only())
        else:
            print(f"Unknown test type: {test_type}")
            print("Available types: websocket, crisis")
            sys.exit(1)
    else:
        # Run comprehensive validation
        result = asyncio.run(run_performance_load_validation())
    
    # Set exit code based on results
    if result and result.get("critical_failures", 1) <= 2 and result.get("overall_success_rate", 0) >= 75:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure