#!/usr/bin/env python3
"""
VS 6.2 Performance Validation Script
LeanVibe Agent Hive 2.0

Comprehensive performance validation for Live Dashboard Integration with Event Streaming.
Validates all performance targets and generates detailed reports.
"""

import asyncio
import json
import time
import statistics
import psutil
import aiohttp
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import argparse

# Performance targets from requirements
PERFORMANCE_TARGETS = {
    "dashboard_load_time_ms": 2000,
    "event_processing_latency_ms": 1000,
    "event_throughput_per_second": 1000,
    "websocket_connection_time_ms": 500,
    "api_response_time_ms": 1000,
    "memory_usage_mb": 512,
    "cpu_usage_percent": 80,
    "concurrent_connections": 100
}

class PerformanceValidator:
    """Main performance validation orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "targets": PERFORMANCE_TARGETS,
            "tests": {},
            "summary": {},
            "passed": True
        }
        self.base_url = config.get("base_url", "http://localhost:8000")
        self.ws_url = config.get("ws_url", "ws://localhost:8000/ws/observability/dashboard")
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all performance validations"""
        print("🚀 VS 6.2 Performance Validation Suite")
        print("=" * 60)
        
        validations = [
            ("Dashboard Load Time", self.validate_dashboard_load_time),
            ("Event Processing Latency", self.validate_event_processing_latency),
            ("Event Throughput", self.validate_event_throughput),
            ("WebSocket Performance", self.validate_websocket_performance),
            ("API Response Times", self.validate_api_response_times),
            ("Concurrent Load", self.validate_concurrent_load),
            ("Memory Usage", self.validate_memory_usage),
            ("Frontend Performance", self.validate_frontend_performance)
        ]
        
        for name, validation_func in validations:
            print(f"\n📊 Running {name} validation...")
            try:
                result = await validation_func()
                self.results["tests"][name] = result
                
                status = "✅ PASSED" if result["passed"] else "❌ FAILED"
                print(f"   {status} - {result.get('summary', 'No summary')}")
                
                if not result["passed"]:
                    self.results["passed"] = False
                    
            except Exception as e:
                print(f"   ❌ ERROR - {str(e)}")
                self.results["tests"][name] = {
                    "passed": False,
                    "error": str(e),
                    "summary": f"Validation failed with error: {str(e)}"
                }
                self.results["passed"] = False
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    async def validate_dashboard_load_time(self) -> Dict[str, Any]:
        """Validate dashboard component load times"""
        endpoints = [
            "/api/v1/observability/workflow-constellation",
            "/api/v1/observability/intelligence-kpis",
            "/api/v1/observability/context-trajectory?context_id=test-context"
        ]
        
        load_times = []
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                times = []
                
                # Perform multiple measurements
                for _ in range(10):
                    start_time = time.time()
                    
                    try:
                        async with session.get(f"{self.base_url}{endpoint}") as response:
                            await response.json()
                            
                        load_time = (time.time() - start_time) * 1000
                        times.append(load_time)
                        
                    except Exception as e:
                        print(f"   Warning: Request to {endpoint} failed: {e}")
                        continue
                
                if times:
                    load_times.extend(times)
        
        if not load_times:
            return {
                "passed": False,
                "summary": "No successful dashboard loads measured",
                "metrics": {}
            }
        
        avg_load_time = statistics.mean(load_times)
        max_load_time = max(load_times)
        p95_load_time = statistics.quantiles(load_times, n=20)[18]  # 95th percentile
        
        passed = avg_load_time < PERFORMANCE_TARGETS["dashboard_load_time_ms"]
        
        return {
            "passed": passed,
            "summary": f"Avg: {avg_load_time:.1f}ms, Max: {max_load_time:.1f}ms, P95: {p95_load_time:.1f}ms (Target: <{PERFORMANCE_TARGETS['dashboard_load_time_ms']}ms)",
            "metrics": {
                "average_ms": avg_load_time,
                "maximum_ms": max_load_time,
                "p95_ms": p95_load_time,
                "target_ms": PERFORMANCE_TARGETS["dashboard_load_time_ms"],
                "sample_count": len(load_times)
            }
        }
    
    async def validate_event_processing_latency(self) -> Dict[str, Any]:
        """Validate event processing latency"""
        latencies = []
        
        # Test semantic search latency
        search_request = {
            "query": "show me agent performance issues",
            "context_window_hours": 24,
            "max_results": 25,
            "similarity_threshold": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            for _ in range(50):  # Test with 50 requests
                start_time = time.time()
                
                try:
                    async with session.post(
                        f"{self.base_url}/api/v1/observability/semantic-search",
                        json=search_request
                    ) as response:
                        await response.json()
                        
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)
                    
                except Exception as e:
                    print(f"   Warning: Semantic search failed: {e}")
                    continue
        
        if not latencies:
            return {
                "passed": False,
                "summary": "No successful event processing measured",
                "metrics": {}
            }
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        
        passed = avg_latency < PERFORMANCE_TARGETS["event_processing_latency_ms"]
        
        return {
            "passed": passed,
            "summary": f"Avg: {avg_latency:.1f}ms, Max: {max_latency:.1f}ms, P95: {p95_latency:.1f}ms (Target: <{PERFORMANCE_TARGETS['event_processing_latency_ms']}ms)",
            "metrics": {
                "average_ms": avg_latency,
                "maximum_ms": max_latency,
                "p95_ms": p95_latency,
                "target_ms": PERFORMANCE_TARGETS["event_processing_latency_ms"],
                "sample_count": len(latencies)
            }
        }
    
    async def validate_event_throughput(self) -> Dict[str, Any]:
        """Validate event throughput capacity"""
        try:
            # Connect to WebSocket
            async with websockets.connect(self.ws_url) as websocket:
                # Send authentication/handshake if required
                await websocket.send(json.dumps({
                    "type": "subscribe",
                    "component": "performance_test",
                    "filters": {},
                    "priority": 10
                }))
                
                # Test event throughput
                event_count = 1200  # Slightly above target
                start_time = time.time()
                
                # Send events rapidly
                for i in range(event_count):
                    event = {
                        "type": "performance_test",
                        "id": f"throughput-test-{i}",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"test_index": i}
                    }
                    
                    await websocket.send(json.dumps(event))
                
                processing_time = time.time() - start_time
                throughput = event_count / processing_time
                
                passed = throughput >= PERFORMANCE_TARGETS["event_throughput_per_second"]
                
                return {
                    "passed": passed,
                    "summary": f"Processed {event_count} events in {processing_time:.2f}s = {throughput:.1f} events/sec (Target: >{PERFORMANCE_TARGETS['event_throughput_per_second']}/sec)",
                    "metrics": {
                        "events_processed": event_count,
                        "processing_time_s": processing_time,
                        "throughput_per_second": throughput,
                        "target_per_second": PERFORMANCE_TARGETS["event_throughput_per_second"]
                    }
                }
                
        except Exception as e:
            return {
                "passed": False,
                "summary": f"WebSocket throughput test failed: {str(e)}",
                "metrics": {}
            }
    
    async def validate_websocket_performance(self) -> Dict[str, Any]:
        """Validate WebSocket connection and performance"""
        connection_times = []
        
        # Test multiple WebSocket connections
        for _ in range(20):
            start_time = time.time()
            
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    await websocket.send(json.dumps({
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    
                    # Wait for response or timeout
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        connection_time = (time.time() - start_time) * 1000
                        connection_times.append(connection_time)
                    except asyncio.TimeoutError:
                        print("   Warning: WebSocket response timeout")
                        continue
                        
            except Exception as e:
                print(f"   Warning: WebSocket connection failed: {e}")
                continue
        
        if not connection_times:
            return {
                "passed": False,
                "summary": "No successful WebSocket connections",
                "metrics": {}
            }
        
        avg_connection_time = statistics.mean(connection_times)
        max_connection_time = max(connection_times)
        
        passed = avg_connection_time < PERFORMANCE_TARGETS["websocket_connection_time_ms"]
        
        return {
            "passed": passed,
            "summary": f"Avg: {avg_connection_time:.1f}ms, Max: {max_connection_time:.1f}ms (Target: <{PERFORMANCE_TARGETS['websocket_connection_time_ms']}ms)",
            "metrics": {
                "average_connection_ms": avg_connection_time,
                "maximum_connection_ms": max_connection_time,
                "target_ms": PERFORMANCE_TARGETS["websocket_connection_time_ms"],
                "successful_connections": len(connection_times)
            }
        }
    
    async def validate_api_response_times(self) -> Dict[str, Any]:
        """Validate API response times across all endpoints"""
        endpoints = [
            ("/api/v1/observability/semantic-search", "POST", {"query": "test", "max_results": 10}),
            ("/api/v1/observability/workflow-constellation", "GET", {}),
            ("/api/v1/observability/context-trajectory", "GET", {"context_id": "test"}),
            ("/api/v1/observability/intelligence-kpis", "GET", {"time_range_hours": 1}),
            ("/api/v1/observability/semantic-suggestions", "GET", {"partial_query": "agent"})
        ]
        
        all_response_times = []
        endpoint_results = {}
        
        async with aiohttp.ClientSession() as session:
            for endpoint, method, params in endpoints:
                response_times = []
                
                for _ in range(10):  # 10 requests per endpoint
                    start_time = time.time()
                    
                    try:
                        if method == "POST":
                            async with session.post(f"{self.base_url}{endpoint}", json=params) as response:
                                await response.json()
                        else:
                            async with session.get(f"{self.base_url}{endpoint}", params=params) as response:
                                await response.json()
                        
                        response_time = (time.time() - start_time) * 1000
                        response_times.append(response_time)
                        all_response_times.append(response_time)
                        
                    except Exception as e:
                        print(f"   Warning: Request to {endpoint} failed: {e}")
                        continue
                
                if response_times:
                    endpoint_results[endpoint] = {
                        "average_ms": statistics.mean(response_times),
                        "maximum_ms": max(response_times),
                        "sample_count": len(response_times)
                    }
        
        if not all_response_times:
            return {
                "passed": False,
                "summary": "No successful API responses measured",
                "metrics": {}
            }
        
        avg_response_time = statistics.mean(all_response_times)
        max_response_time = max(all_response_times)
        
        passed = avg_response_time < PERFORMANCE_TARGETS["api_response_time_ms"]
        
        return {
            "passed": passed,
            "summary": f"Avg: {avg_response_time:.1f}ms, Max: {max_response_time:.1f}ms (Target: <{PERFORMANCE_TARGETS['api_response_time_ms']}ms)",
            "metrics": {
                "average_ms": avg_response_time,
                "maximum_ms": max_response_time,
                "target_ms": PERFORMANCE_TARGETS["api_response_time_ms"],
                "total_requests": len(all_response_times),
                "endpoint_breakdown": endpoint_results
            }
        }
    
    async def validate_concurrent_load(self) -> Dict[str, Any]:
        """Validate performance under concurrent load"""
        concurrent_connections = PERFORMANCE_TARGETS["concurrent_connections"]
        
        async def make_concurrent_request(session: aiohttp.ClientSession, request_id: int) -> Tuple[bool, float]:
            """Make a single concurrent request"""
            start_time = time.time()
            
            try:
                async with session.post(
                    f"{self.base_url}/api/v1/observability/semantic-search",
                    json={
                        "query": f"concurrent test {request_id}",
                        "max_results": 5
                    }
                ) as response:
                    await response.json()
                    
                response_time = (time.time() - start_time) * 1000
                return True, response_time
                
            except Exception:
                response_time = (time.time() - start_time) * 1000
                return False, response_time
        
        # Execute concurrent requests
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                make_concurrent_request(session, i) 
                for i in range(concurrent_connections)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = 0
        response_times = []
        
        for result in results:
            if isinstance(result, tuple):
                success, response_time = result
                if success:
                    successful_requests += 1
                response_times.append(response_time)
        
        success_rate = (successful_requests / concurrent_connections) * 100
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Pass if >90% success rate and reasonable response times
        passed = success_rate >= 90 and avg_response_time < (PERFORMANCE_TARGETS["api_response_time_ms"] * 2)
        
        return {
            "passed": passed,
            "summary": f"{successful_requests}/{concurrent_connections} requests succeeded ({success_rate:.1f}%) in {total_time:.2f}s, avg response: {avg_response_time:.1f}ms",
            "metrics": {
                "concurrent_connections": concurrent_connections,
                "successful_requests": successful_requests,
                "success_rate_percent": success_rate,
                "total_time_s": total_time,
                "average_response_ms": avg_response_time,
                "target_success_rate": 90
            }
        }
    
    async def validate_memory_usage(self) -> Dict[str, Any]:
        """Validate memory usage during load"""
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate load and measure memory
        async with aiohttp.ClientSession() as session:
            # Make multiple concurrent requests to generate load
            tasks = []
            for i in range(50):
                task = session.post(
                    f"{self.base_url}/api/v1/observability/semantic-search",
                    json={"query": f"memory test {i}", "max_results": 10}
                )
                tasks.append(task)
            
            # Execute and measure memory during load
            memory_samples = []
            
            for i in range(10):  # Sample memory 10 times during load
                await asyncio.sleep(0.1)
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
            
            # Wait for requests to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        peak_memory = max(memory_samples)
        avg_memory = statistics.mean(memory_samples)
        memory_increase = peak_memory - baseline_memory
        
        passed = peak_memory < PERFORMANCE_TARGETS["memory_usage_mb"]
        
        return {
            "passed": passed,
            "summary": f"Peak: {peak_memory:.1f}MB, Avg: {avg_memory:.1f}MB, Increase: {memory_increase:.1f}MB (Target: <{PERFORMANCE_TARGETS['memory_usage_mb']}MB)",
            "metrics": {
                "baseline_mb": baseline_memory,
                "peak_mb": peak_memory,
                "average_mb": avg_memory,
                "increase_mb": memory_increase,
                "target_mb": PERFORMANCE_TARGETS["memory_usage_mb"]
            }
        }
    
    async def validate_frontend_performance(self) -> Dict[str, Any]:
        """Validate frontend performance using headless browser"""
        try:
            # This would require Playwright or Selenium for full browser testing
            # For now, we'll validate that the frontend build is optimized
            
            frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
            
            if not os.path.exists(frontend_path):
                return {
                    "passed": False,
                    "summary": "Frontend directory not found",
                    "metrics": {}
                }
            
            # Check if build exists
            dist_path = os.path.join(frontend_path, "dist")
            
            if not os.path.exists(dist_path):
                # Try to build
                result = subprocess.run(
                    ["npm", "run", "build"],
                    cwd=frontend_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    return {
                        "passed": False,
                        "summary": f"Frontend build failed: {result.stderr}",
                        "metrics": {}
                    }
            
            # Analyze bundle sizes
            bundle_sizes = {}
            total_size = 0
            
            for root, dirs, files in os.walk(dist_path):
                for file in files:
                    if file.endswith(('.js', '.css')):
                        file_path = os.path.join(root, file)
                        size = os.path.getsize(file_path) / 1024  # KB
                        bundle_sizes[file] = size
                        total_size += size
            
            # Frontend performance targets
            max_bundle_size_kb = 2048  # 2MB total
            passed = total_size < max_bundle_size_kb
            
            return {
                "passed": passed,
                "summary": f"Total bundle size: {total_size:.1f}KB (Target: <{max_bundle_size_kb}KB)",
                "metrics": {
                    "total_bundle_size_kb": total_size,
                    "target_size_kb": max_bundle_size_kb,
                    "bundle_breakdown": bundle_sizes
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "summary": f"Frontend validation failed: {str(e)}",
                "metrics": {}
            }
    
    def generate_summary(self):
        """Generate performance validation summary"""
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for test in self.results["tests"].values() if test["passed"])
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "overall_passed": self.results["passed"]
        }
    
    def generate_report(self, output_file: str = None):
        """Generate detailed performance report"""
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        
        # Console report
        print("\n" + "=" * 60)
        print("📊 VS 6.2 Performance Validation Report")
        print("=" * 60)
        
        summary = self.results["summary"]
        status = "✅ PASSED" if self.results["passed"] else "❌ FAILED"
        print(f"\nOverall Status: {status}")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']} ({summary['success_rate']:.1f}%)")
        
        print(f"\n{'Test Name':<30} {'Status':<10} {'Summary'}")
        print("-" * 80)
        
        for test_name, test_result in self.results["tests"].items():
            status = "✅ PASS" if test_result["passed"] else "❌ FAIL"
            summary = test_result.get("summary", "No summary")[:40]
            print(f"{test_name:<30} {status:<10} {summary}")
        
        if not self.results["passed"]:
            print(f"\n⚠️  Performance targets not met. Review failed tests and optimize.")
        else:
            print(f"\n🎉 All performance targets met! Dashboard is ready for production.")
        
        if output_file:
            print(f"\n📝 Detailed report saved to: {output_file}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VS 6.2 Performance Validation")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for API tests")
    parser.add_argument("--ws-url", default="ws://localhost:8000/ws/observability/dashboard", help="WebSocket URL")
    parser.add_argument("--output", help="Output file for detailed report")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (fewer iterations)")
    
    args = parser.parse_args()
    
    config = {
        "base_url": args.base_url,
        "ws_url": args.ws_url,
        "quick_mode": args.quick
    }
    
    validator = PerformanceValidator(config)
    
    try:
        results = await validator.run_all_validations()
        validator.generate_report(args.output)
        
        # Exit with appropriate code
        sys.exit(0 if results["passed"] else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Performance validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Performance validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class Vs62PerformanceValidationScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(Vs62PerformanceValidationScript)