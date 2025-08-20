#!/usr/bin/env python3
"""
Focused Performance Testing for LeanVibe Agent Hive 2.0

Tests actual available endpoints and system performance using the existing
infrastructure and services.
"""

import asyncio
import time
import json
import statistics
import httpx
import psutil
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess

import structlog

logger = structlog.get_logger()


class SystemPerformanceTester:
    """Test actual system performance with available endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        
        # Performance targets (enterprise requirements)
        self.targets = {
            'health_check_response_time': 100,  # 100ms
            'api_root_response_time': 50,  # 50ms
            'status_endpoint_response_time': 500,  # 500ms
            'concurrent_requests_rps': 100,  # 100 RPS minimum
            'system_availability': 99.9,  # 99.9% availability
            'max_cpu_usage': 80,  # 80% max CPU
            'max_memory_mb': 4096,  # 4GB max memory
        }
    
    async def run_comprehensive_tests(self) -> Dict:
        """Run comprehensive performance tests."""
        logger.info("🚀 Starting Focused Performance Testing")
        
        client_timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(
            timeout=client_timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        ) as client:
            
            # Test 1: Basic endpoint response times
            await self._test_endpoint_response_times(client)
            
            # Test 2: Load testing with concurrent requests
            await self._test_concurrent_load(client)
            
            # Test 3: System resource utilization
            await self._test_resource_utilization(client)
            
            # Test 4: Error handling and recovery
            await self._test_error_handling(client)
            
            # Test 5: Database and Redis performance
            await self._test_infrastructure_performance(client)
            
            # Test 6: System stress testing
            await self._test_system_stress(client)
        
        return self._generate_report()
    
    async def _test_endpoint_response_times(self, client: httpx.AsyncClient):
        """Test response times of available endpoints."""
        logger.info("📊 Testing endpoint response times")
        
        endpoints = [
            ("/health", "health_check"),
            ("/api/v1/", "api_root"),
            ("/status", "system_status"),
            ("/metrics", "metrics"),
            ("/debug-agents", "debug_agents")
        ]
        
        endpoint_results = {}
        
        for endpoint, name in endpoints:
            times = []
            success_count = 0
            error_count = 0
            
            # Test each endpoint 20 times
            for _ in range(20):
                start_time = time.time()
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    if response.status_code == 200:
                        times.append(response_time)
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Endpoint {endpoint} error: {e}")
            
            if times:
                endpoint_results[name] = {
                    "avg_response_time": statistics.mean(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "p95_response_time": sorted(times)[int(len(times) * 0.95)] if times else 0,
                    "success_count": success_count,
                    "error_count": error_count,
                    "success_rate": success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0
                }
                
                logger.info(f"{name}: {endpoint_results[name]['avg_response_time']:.2f}ms avg, {endpoint_results[name]['success_rate']:.1%} success")
        
        self.results["endpoint_performance"] = endpoint_results
    
    async def _test_concurrent_load(self, client: httpx.AsyncClient):
        """Test concurrent request handling."""
        logger.info("🔄 Testing concurrent load handling")
        
        async def make_request():
            """Make a single request."""
            try:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200, time.time()
            except Exception:
                return False, time.time()
        
        # Test different concurrency levels
        concurrency_levels = [10, 25, 50, 100]
        load_results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = [asyncio.create_task(make_request()) for _ in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Analyze results
            successful_requests = sum(1 for r in results if isinstance(r, tuple) and r[0])
            total_requests = len(results)
            rps = total_requests / duration if duration > 0 else 0
            
            load_results[f"concurrency_{concurrency}"] = {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": total_requests - successful_requests,
                "duration_seconds": duration,
                "requests_per_second": rps,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0
            }
            
            logger.info(f"Concurrency {concurrency}: {rps:.2f} RPS, {successful_requests}/{total_requests} successful")
        
        self.results["load_testing"] = load_results
    
    async def _test_resource_utilization(self, client: httpx.AsyncClient):
        """Monitor system resource utilization."""
        logger.info("📈 Testing system resource utilization")
        
        # Monitor resources for 30 seconds while making requests
        resource_data = {
            "cpu_usage": [],
            "memory_usage_mb": [],
            "disk_io_mb": [],
            "network_io_mb": []
        }
        
        start_time = time.time()
        
        # Create background load
        async def background_load():
            while time.time() - start_time < 30:
                try:
                    await client.get(f"{self.base_url}/health")
                    await asyncio.sleep(0.1)  # 10 RPS background load
                except Exception:
                    pass
        
        load_task = asyncio.create_task(background_load())
        
        # Monitor resources
        while time.time() - start_time < 30:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                resource_data["cpu_usage"].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                resource_data["memory_usage_mb"].append(memory_mb)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_io_mb = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)
                    resource_data["disk_io_mb"].append(disk_io_mb)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    net_io_mb = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
                    resource_data["network_io_mb"].append(net_io_mb)
                
            except Exception as e:
                logger.debug(f"Resource monitoring error: {e}")
                
            await asyncio.sleep(1)
        
        # Stop background load
        load_task.cancel()
        try:
            await load_task
        except asyncio.CancelledError:
            pass
        
        # Calculate statistics
        resource_stats = {}
        for metric, values in resource_data.items():
            if values:
                resource_stats[metric] = {
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "p95": sorted(values)[int(len(values) * 0.95)] if values else 0
                }
        
        self.results["resource_utilization"] = resource_stats
        
        logger.info(f"Resource utilization - CPU: {resource_stats.get('cpu_usage', {}).get('avg', 0):.1f}% avg, Memory: {resource_stats.get('memory_usage_mb', {}).get('avg', 0):.1f}MB avg")
    
    async def _test_error_handling(self, client: httpx.AsyncClient):
        """Test error handling and recovery."""
        logger.info("🚧 Testing error handling and recovery")
        
        # Test invalid endpoints
        invalid_endpoints = [
            "/nonexistent",
            "/api/v1/invalid",
            "/health/invalid"
        ]
        
        error_handling_results = {}
        
        for endpoint in invalid_endpoints:
            try:
                response = await client.get(f"{self.base_url}{endpoint}")
                error_handling_results[endpoint] = {
                    "status_code": response.status_code,
                    "handled_gracefully": response.status_code in [404, 405, 422]  # Expected error codes
                }
            except Exception as e:
                error_handling_results[endpoint] = {
                    "status_code": "exception",
                    "handled_gracefully": False,
                    "error": str(e)
                }
        
        # Test system recovery under rapid requests
        recovery_start = time.time()
        rapid_requests = []
        
        # Make 100 rapid requests
        tasks = []
        for _ in range(100):
            task = asyncio.create_task(client.get(f"{self.base_url}/health"))
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        recovery_time = time.time() - recovery_start
        
        successful_responses = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
        
        error_handling_results["rapid_recovery"] = {
            "total_requests": len(responses),
            "successful_requests": successful_responses,
            "recovery_time_seconds": recovery_time,
            "success_rate": successful_responses / len(responses) if responses else 0
        }
        
        self.results["error_handling"] = error_handling_results
    
    async def _test_infrastructure_performance(self, client: httpx.AsyncClient):
        """Test database and Redis performance."""
        logger.info("🗄️ Testing infrastructure performance")
        
        # Test health endpoint which checks DB and Redis
        infrastructure_times = []
        db_redis_status = {"healthy": 0, "unhealthy": 0}
        
        for _ in range(50):
            start_time = time.time()
            try:
                response = await client.get(f"{self.base_url}/health")
                response_time = (time.time() - start_time) * 1000
                infrastructure_times.append(response_time)
                
                if response.status_code == 200:
                    health_data = response.json()
                    db_status = health_data.get("components", {}).get("database", {}).get("status")
                    redis_status = health_data.get("components", {}).get("redis", {}).get("status")
                    
                    if db_status == "healthy" and redis_status == "healthy":
                        db_redis_status["healthy"] += 1
                    else:
                        db_redis_status["unhealthy"] += 1
                else:
                    db_redis_status["unhealthy"] += 1
                    
            except Exception:
                db_redis_status["unhealthy"] += 1
        
        if infrastructure_times:
            infrastructure_stats = {
                "avg_response_time": statistics.mean(infrastructure_times),
                "min_response_time": min(infrastructure_times),
                "max_response_time": max(infrastructure_times),
                "p95_response_time": sorted(infrastructure_times)[int(len(infrastructure_times) * 0.95)],
                "db_redis_availability": db_redis_status["healthy"] / (db_redis_status["healthy"] + db_redis_status["unhealthy"]) if (db_redis_status["healthy"] + db_redis_status["unhealthy"]) > 0 else 0
            }
            
            self.results["infrastructure_performance"] = infrastructure_stats
            
            logger.info(f"Infrastructure - Avg response: {infrastructure_stats['avg_response_time']:.2f}ms, Availability: {infrastructure_stats['db_redis_availability']:.1%}")
    
    async def _test_system_stress(self, client: httpx.AsyncClient):
        """Test system under stress conditions."""
        logger.info("⚡ Testing system under stress")
        
        # Create sustained load for 60 seconds
        stress_start_time = time.time()
        stress_results = {
            "requests_made": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "response_times": []
        }
        
        async def stress_worker():
            """Worker that makes continuous requests."""
            while time.time() - stress_start_time < 60:  # 60 seconds of stress
                request_start = time.time()
                try:
                    response = await client.get(f"{self.base_url}/health")
                    response_time = (time.time() - request_start) * 1000
                    
                    stress_results["requests_made"] += 1
                    stress_results["response_times"].append(response_time)
                    
                    if response.status_code == 200:
                        stress_results["requests_successful"] += 1
                    else:
                        stress_results["requests_failed"] += 1
                        
                except Exception:
                    stress_results["requests_made"] += 1
                    stress_results["requests_failed"] += 1
                
                await asyncio.sleep(0.05)  # 20 RPS per worker
        
        # Start 10 stress workers
        stress_tasks = [asyncio.create_task(stress_worker()) for _ in range(10)]
        
        # Monitor system during stress
        stress_cpu_usage = []
        stress_memory_usage = []
        
        monitor_start = time.time()
        while time.time() - monitor_start < 60:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                
                stress_cpu_usage.append(cpu_percent)
                stress_memory_usage.append(memory_mb)
                
            except Exception:
                pass
        
        # Wait for stress tasks to complete
        await asyncio.gather(*stress_tasks, return_exceptions=True)
        
        # Calculate stress test results
        if stress_results["response_times"]:
            stress_results["avg_response_time"] = statistics.mean(stress_results["response_times"])
            stress_results["p95_response_time"] = sorted(stress_results["response_times"])[int(len(stress_results["response_times"]) * 0.95)]
        
        stress_results["success_rate"] = stress_results["requests_successful"] / stress_results["requests_made"] if stress_results["requests_made"] > 0 else 0
        stress_results["requests_per_second"] = stress_results["requests_made"] / 60
        
        if stress_cpu_usage:
            stress_results["max_cpu_usage"] = max(stress_cpu_usage)
            stress_results["avg_cpu_usage"] = statistics.mean(stress_cpu_usage)
        
        if stress_memory_usage:
            stress_results["max_memory_usage_mb"] = max(stress_memory_usage)
            stress_results["avg_memory_usage_mb"] = statistics.mean(stress_memory_usage)
        
        self.results["stress_testing"] = stress_results
        
        logger.info(f"Stress test - {stress_results['requests_per_second']:.1f} RPS, {stress_results['success_rate']:.1%} success rate")
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive performance report."""
        
        # Analyze results against targets
        test_results = []
        
        # Health check response time
        if "endpoint_performance" in self.results and "health_check" in self.results["endpoint_performance"]:
            health_check = self.results["endpoint_performance"]["health_check"]
            avg_time = health_check["avg_response_time"]
            
            test_results.append({
                "test_name": "Health Check Response Time",
                "target": self.targets["health_check_response_time"],
                "actual": avg_time,
                "unit": "ms",
                "passed": avg_time <= self.targets["health_check_response_time"]
            })
        
        # API root response time
        if "endpoint_performance" in self.results and "api_root" in self.results["endpoint_performance"]:
            api_root = self.results["endpoint_performance"]["api_root"]
            avg_time = api_root["avg_response_time"]
            
            test_results.append({
                "test_name": "API Root Response Time",
                "target": self.targets["api_root_response_time"],
                "actual": avg_time,
                "unit": "ms",
                "passed": avg_time <= self.targets["api_root_response_time"]
            })
        
        # Concurrent requests RPS
        if "load_testing" in self.results:
            max_rps = 0
            for concurrency_test in self.results["load_testing"].values():
                rps = concurrency_test.get("requests_per_second", 0)
                max_rps = max(max_rps, rps)
            
            test_results.append({
                "test_name": "Concurrent Request Handling",
                "target": self.targets["concurrent_requests_rps"],
                "actual": max_rps,
                "unit": " RPS",
                "passed": max_rps >= self.targets["concurrent_requests_rps"]
            })
        
        # System availability
        if "infrastructure_performance" in self.results:
            availability = self.results["infrastructure_performance"].get("db_redis_availability", 0) * 100
            
            test_results.append({
                "test_name": "System Availability",
                "target": self.targets["system_availability"],
                "actual": availability,
                "unit": "%",
                "passed": availability >= self.targets["system_availability"]
            })
        
        # CPU Usage
        if "resource_utilization" in self.results and "cpu_usage" in self.results["resource_utilization"]:
            max_cpu = self.results["resource_utilization"]["cpu_usage"]["max"]
            
            test_results.append({
                "test_name": "Maximum CPU Usage",
                "target": self.targets["max_cpu_usage"],
                "actual": max_cpu,
                "unit": "%",
                "passed": max_cpu <= self.targets["max_cpu_usage"]
            })
        
        # Memory Usage
        if "resource_utilization" in self.results and "memory_usage_mb" in self.results["resource_utilization"]:
            max_memory = self.results["resource_utilization"]["memory_usage_mb"]["max"]
            
            test_results.append({
                "test_name": "Maximum Memory Usage",
                "target": self.targets["max_memory_mb"],
                "actual": max_memory,
                "unit": "MB",
                "passed": max_memory <= self.targets["max_memory_mb"]
            })
        
        # Calculate overall results
        passed_tests = sum(1 for test in test_results if test["passed"])
        total_tests = len(test_results)
        overall_pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        return {
            "summary": {
                "overall_status": "PASS" if overall_pass_rate >= 0.8 else "FAIL",  # 80% pass rate
                "pass_rate": overall_pass_rate,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": test_results,
            "detailed_results": self.results,
            "targets": self.targets,
            "recommendations": self._generate_recommendations(test_results)
        }
    
    def _generate_recommendations(self, test_results: List[Dict]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [test for test in test_results if not test["passed"]]
        
        for test in failed_tests:
            if "Response Time" in test["test_name"]:
                recommendations.append(f"Optimize {test['test_name'].lower()} - consider caching, database optimization, or infrastructure scaling")
            elif "CPU Usage" in test["test_name"]:
                recommendations.append("High CPU usage detected - consider code optimization, load balancing, or scaling up compute resources")
            elif "Memory Usage" in test["test_name"]:
                recommendations.append("High memory usage detected - investigate memory leaks, optimize data structures, or increase available memory")
            elif "Concurrent Request" in test["test_name"]:
                recommendations.append("Improve concurrent request handling - consider connection pooling, async optimization, or horizontal scaling")
            elif "Availability" in test["test_name"]:
                recommendations.append("Improve system availability - implement redundancy, health checks, and graceful failure handling")
        
        if not recommendations:
            recommendations.append("All performance targets met! System ready for enterprise deployment.")
        
        return recommendations


async def main():
    """Main execution function."""
    tester = SystemPerformanceTester()
    
    logger.info("🚀 Starting Focused Performance Testing")
    
    # Run comprehensive tests
    report = await tester.run_comprehensive_tests()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/tmp/focused_performance_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("FOCUSED PERFORMANCE TEST RESULTS")
    print("="*80)
    
    summary = report["summary"]
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
    
    print("\nTEST RESULTS:")
    for test in report["test_results"]:
        status = "✅ PASS" if test["passed"] else "❌ FAIL"
        print(f"  {status} {test['test_name']}: {test['actual']:.2f}{test['unit']} (target: {test['target']:.2f}{test['unit']})")
    
    print("\nRECOMMENDATIONS:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    print(f"\nFull report saved to: {report_file}")
    print("="*80)
    
    return summary["overall_status"] == "PASS"


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class FocusedPerformanceTest(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            import sys

            # Run focused performance testing
            await main()

            # Exit with appropriate code
            sys.exit(0 if success else 1)
            
            return {"status": "completed"}
    
    script_main(FocusedPerformanceTest)