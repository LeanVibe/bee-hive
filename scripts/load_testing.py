#!/usr/bin/env python3
"""
Load Testing for LeanVibe Agent Hive 2.0

Tests system performance under high concurrent load, multiple client connections,
and maximum capacity conditions.
"""

import asyncio
import time
import json
import httpx
import statistics
from datetime import datetime
from typing import Dict, List, Tuple
import structlog

logger = structlog.get_logger()


class LoadTester:
    """Comprehensive load testing for the agent hive system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        
        # Load testing targets
        self.targets = {
            'max_concurrent_requests': 500,  # Maximum concurrent requests
            'min_sustained_rps': 200,  # Minimum sustained requests per second
            'max_response_time_p95': 1000,  # Max P95 response time in ms
            'min_success_rate': 0.99,  # 99% success rate under load
            'max_agent_spawn_time': 10000,  # Max 10 seconds to spawn agents
            'concurrent_client_capacity': 100,  # Support 100 concurrent clients
        }
    
    async def run_load_tests(self) -> Dict:
        """Run comprehensive load testing."""
        logger.info("🔄 Starting Load Testing")
        
        # Test 1: Concurrent request capacity
        await self._test_concurrent_request_capacity()
        
        # Test 2: Sustained load testing
        await self._test_sustained_load()
        
        # Test 3: Multiple client connections
        await self._test_multiple_client_connections()
        
        # Test 4: Agent system load testing
        await self._test_agent_system_load()
        
        # Test 5: Mixed workload testing
        await self._test_mixed_workload()
        
        # Test 6: Maximum capacity testing
        await self._test_maximum_capacity()
        
        return self._generate_load_test_report()
    
    async def _test_concurrent_request_capacity(self):
        """Test maximum concurrent request handling capacity."""
        logger.info("Testing concurrent request capacity")
        
        capacity_results = {}
        
        # Test increasing levels of concurrency
        concurrency_levels = [10, 25, 50, 100, 200, 500]
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing {concurrency} concurrent requests")
            
            async def make_request():
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        start_time = time.time()
                        response = await client.get(f"{self.base_url}/health")
                        response_time = (time.time() - start_time) * 1000
                        return response.status_code == 200, response_time
                except Exception as e:
                    return False, 0
            
            start_time = time.time()
            tasks = [asyncio.create_task(make_request()) for _ in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze results
            successful = sum(1 for r in results if isinstance(r, tuple) and r[0])
            response_times = [r[1] for r in results if isinstance(r, tuple)]
            
            if response_times:
                capacity_results[f"concurrency_{concurrency}"] = {
                    "total_requests": concurrency,
                    "successful_requests": successful,
                    "failed_requests": concurrency - successful,
                    "success_rate": successful / concurrency,
                    "duration_seconds": total_time,
                    "requests_per_second": concurrency / total_time,
                    "avg_response_time": statistics.mean(response_times),
                    "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                    "p99_response_time": sorted(response_times)[int(len(response_times) * 0.99)] if response_times else 0
                }
                
                logger.info(f"Concurrency {concurrency}: {successful}/{concurrency} successful, {successful/concurrency:.1%} success rate")
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        self.results["concurrent_capacity"] = capacity_results
    
    async def _test_sustained_load(self):
        """Test sustained load over time."""
        logger.info("Testing sustained load performance")
        
        # Run sustained load for 2 minutes at different RPS levels
        rps_levels = [50, 100, 200]
        sustained_results = {}
        
        for target_rps in rps_levels:
            logger.info(f"Testing sustained {target_rps} RPS for 120 seconds")
            
            request_interval = 1.0 / target_rps  # Time between requests
            test_duration = 120  # 2 minutes
            
            start_time = time.time()
            results = []
            
            async def sustained_worker():
                while time.time() - start_time < test_duration:
                    request_start = time.time()
                    
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            response = await client.get(f"{self.base_url}/health")
                            response_time = (time.time() - request_start) * 1000
                            results.append({
                                "success": response.status_code == 200,
                                "response_time": response_time,
                                "timestamp": time.time()
                            })
                    except Exception:
                        results.append({
                            "success": False,
                            "response_time": 0,
                            "timestamp": time.time()
                        })
                    
                    # Wait for next request
                    next_request_time = request_start + request_interval
                    wait_time = next_request_time - time.time()
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
            
            # Run multiple workers to achieve target RPS
            num_workers = min(10, max(1, target_rps // 20))  # 1-10 workers
            worker_tasks = [asyncio.create_task(sustained_worker()) for _ in range(num_workers)]
            
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            actual_duration = time.time() - start_time
            
            # Analyze sustained test results
            if results:
                successful = sum(1 for r in results if r["success"])
                response_times = [r["response_time"] for r in results if r["success"]]
                actual_rps = len(results) / actual_duration
                
                sustained_results[f"target_rps_{target_rps}"] = {
                    "target_rps": target_rps,
                    "actual_rps": actual_rps,
                    "total_requests": len(results),
                    "successful_requests": successful,
                    "success_rate": successful / len(results),
                    "duration_seconds": actual_duration,
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0
                }
                
                logger.info(f"Sustained {target_rps} RPS: {actual_rps:.1f} actual RPS, {successful/len(results):.1%} success")
            
            # Brief pause between sustained tests
            await asyncio.sleep(5)
        
        self.results["sustained_load"] = sustained_results
    
    async def _test_multiple_client_connections(self):
        """Test handling multiple concurrent client connections."""
        logger.info("Testing multiple client connections")
        
        # Simulate multiple clients making requests simultaneously
        client_counts = [10, 25, 50, 100]
        client_results = {}
        
        for num_clients in client_counts:
            logger.info(f"Testing {num_clients} concurrent clients")
            
            async def client_session(client_id: int):
                """Simulate a client making multiple requests."""
                session_results = {
                    "requests": 0,
                    "successful": 0,
                    "response_times": []
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Each client makes 20 requests over 60 seconds
                    for _ in range(20):
                        start_time = time.time()
                        try:
                            response = await client.get(f"{self.base_url}/health")
                            response_time = (time.time() - start_time) * 1000
                            
                            session_results["requests"] += 1
                            session_results["response_times"].append(response_time)
                            
                            if response.status_code == 200:
                                session_results["successful"] += 1
                                
                        except Exception:
                            session_results["requests"] += 1
                        
                        # Wait 3 seconds between requests (simulating user behavior)
                        await asyncio.sleep(3)
                
                return session_results
            
            # Start all clients simultaneously
            start_time = time.time()
            client_tasks = [
                asyncio.create_task(client_session(i)) 
                for i in range(num_clients)
            ]
            
            client_session_results = await asyncio.gather(*client_tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Aggregate client results
            total_requests = 0
            total_successful = 0
            all_response_times = []
            
            for result in client_session_results:
                if isinstance(result, dict):
                    total_requests += result["requests"]
                    total_successful += result["successful"]
                    all_response_times.extend(result["response_times"])
            
            client_results[f"clients_{num_clients}"] = {
                "num_clients": num_clients,
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "success_rate": total_successful / total_requests if total_requests > 0 else 0,
                "duration_seconds": total_time,
                "avg_response_time": statistics.mean(all_response_times) if all_response_times else 0,
                "p95_response_time": sorted(all_response_times)[int(len(all_response_times) * 0.95)] if all_response_times else 0
            }
            
            logger.info(f"{num_clients} clients: {total_successful}/{total_requests} successful, {total_successful/total_requests:.1%} success rate")
        
        self.results["multiple_clients"] = client_results
    
    async def _test_agent_system_load(self):
        """Test agent system under load."""
        logger.info("Testing agent system load")
        
        agent_load_results = {}
        
        # Test agent activation under different loads
        async def test_agent_activation_load(background_load_rps: int):
            """Test agent activation while system is under background load."""
            
            # Start background load
            async def background_load():
                start_time = time.time()
                request_count = 0
                
                while time.time() - start_time < 30:  # 30 seconds of background load
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            await client.get(f"{self.base_url}/health")
                            request_count += 1
                    except Exception:
                        pass
                    
                    await asyncio.sleep(1.0 / background_load_rps)
                
                return request_count
            
            # Start background load tasks
            num_background_workers = max(1, background_load_rps // 50)
            background_tasks = [
                asyncio.create_task(background_load()) 
                for _ in range(num_background_workers)
            ]
            
            # Test agent activation under load
            activation_start = time.time()
            activation_successful = False
            activation_response_time = 0
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/agents/activate",
                        json={"team_size": 3, "auto_start_tasks": False}
                    )
                    activation_response_time = (time.time() - activation_start) * 1000
                    activation_successful = response.status_code == 200
                    
                    if activation_successful:
                        activation_data = response.json()
                        logger.info(f"Agent activation successful: {activation_data.get('message', 'No message')}")
                    
            except Exception as e:
                logger.error(f"Agent activation failed: {e}")
                activation_response_time = (time.time() - activation_start) * 1000
            
            # Wait for background tasks to complete
            background_results = await asyncio.gather(*background_tasks, return_exceptions=True)
            total_background_requests = sum(r for r in background_results if isinstance(r, int))
            
            return {
                "background_load_rps": background_load_rps,
                "background_requests": total_background_requests,
                "activation_successful": activation_successful,
                "activation_response_time": activation_response_time
            }
        
        # Test agent system under different background loads
        background_loads = [0, 10, 50, 100]
        
        for bg_load in background_loads:
            logger.info(f"Testing agent activation with {bg_load} RPS background load")
            
            result = await test_agent_activation_load(bg_load)
            agent_load_results[f"bg_load_{bg_load}"] = result
            
            logger.info(f"Background load {bg_load}: Activation {'✅' if result['activation_successful'] else '❌'} in {result['activation_response_time']:.0f}ms")
            
            # Brief pause between tests
            await asyncio.sleep(5)
        
        self.results["agent_system_load"] = agent_load_results
    
    async def _test_mixed_workload(self):
        """Test system with mixed workload patterns."""
        logger.info("Testing mixed workload patterns")
        
        # Create mixed workload: health checks, agent status, metrics
        endpoints = [
            ("/health", 0.6),  # 60% health checks
            ("/status", 0.2),  # 20% status checks
            ("/metrics", 0.2)  # 20% metrics requests
        ]
        
        mixed_results = {}
        
        async def mixed_workload_worker(duration: int, rps: int):
            """Worker that generates mixed workload."""
            start_time = time.time()
            results = []
            
            while time.time() - start_time < duration:
                # Select endpoint based on distribution
                import random
                rand = random.random()
                cumulative = 0
                selected_endpoint = endpoints[0][0]
                
                for endpoint, probability in endpoints:
                    cumulative += probability
                    if rand <= cumulative:
                        selected_endpoint = endpoint
                        break
                
                # Make request
                request_start = time.time()
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(f"{self.base_url}{selected_endpoint}")
                        response_time = (time.time() - request_start) * 1000
                        results.append({
                            "endpoint": selected_endpoint,
                            "success": response.status_code == 200,
                            "response_time": response_time
                        })
                except Exception:
                    results.append({
                        "endpoint": selected_endpoint,
                        "success": False,
                        "response_time": 0
                    })
                
                # Wait for next request
                await asyncio.sleep(1.0 / rps)
            
            return results
        
        # Test mixed workload at different intensities
        workload_levels = [25, 50, 100, 200]
        
        for rps in workload_levels:
            logger.info(f"Testing mixed workload at {rps} RPS for 60 seconds")
            
            # Run mixed workload
            num_workers = max(1, rps // 25)
            worker_rps = rps // num_workers
            
            worker_tasks = [
                asyncio.create_task(mixed_workload_worker(60, worker_rps))
                for _ in range(num_workers)
            ]
            
            worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            
            # Aggregate results
            all_results = []
            for worker_result in worker_results:
                if isinstance(worker_result, list):
                    all_results.extend(worker_result)
            
            if all_results:
                successful = sum(1 for r in all_results if r["success"])
                response_times = [r["response_time"] for r in all_results if r["success"]]
                
                # Group by endpoint
                endpoint_stats = {}
                for endpoint, _ in endpoints:
                    endpoint_results = [r for r in all_results if r["endpoint"] == endpoint]
                    if endpoint_results:
                        endpoint_successful = sum(1 for r in endpoint_results if r["success"])
                        endpoint_response_times = [r["response_time"] for r in endpoint_results if r["success"]]
                        
                        endpoint_stats[endpoint] = {
                            "requests": len(endpoint_results),
                            "successful": endpoint_successful,
                            "success_rate": endpoint_successful / len(endpoint_results),
                            "avg_response_time": statistics.mean(endpoint_response_times) if endpoint_response_times else 0
                        }
                
                mixed_results[f"rps_{rps}"] = {
                    "target_rps": rps,
                    "total_requests": len(all_results),
                    "successful_requests": successful,
                    "success_rate": successful / len(all_results),
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                    "endpoint_breakdown": endpoint_stats
                }
                
                logger.info(f"Mixed workload {rps} RPS: {successful}/{len(all_results)} successful, {successful/len(all_results):.1%} success")
        
        self.results["mixed_workload"] = mixed_results
    
    async def _test_maximum_capacity(self):
        """Test system at maximum capacity to find breaking point."""
        logger.info("Testing maximum capacity")
        
        # Gradually increase load until we find the breaking point
        max_capacity_results = {}
        breaking_point = None
        
        # Start with a reasonable load and increase gradually
        current_rps = 100
        step_size = 100
        max_rps_to_test = 1000
        
        while current_rps <= max_rps_to_test and breaking_point is None:
            logger.info(f"Testing capacity at {current_rps} RPS")
            
            async def capacity_test_worker(test_rps: int, duration: int = 30):
                """Worker for capacity testing."""
                start_time = time.time()
                results = []
                
                request_interval = 1.0 / test_rps
                
                while time.time() - start_time < duration:
                    request_start = time.time()
                    
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            response = await client.get(f"{self.base_url}/health")
                            response_time = (time.time() - request_start) * 1000
                            results.append({
                                "success": response.status_code == 200,
                                "response_time": response_time
                            })
                    except Exception:
                        results.append({
                            "success": False,
                            "response_time": 0
                        })
                    
                    # Control request rate
                    elapsed = time.time() - request_start
                    sleep_time = max(0, request_interval - elapsed)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                
                return results
            
            # Run capacity test
            num_workers = max(1, current_rps // 50)
            worker_rps = current_rps // num_workers
            
            capacity_tasks = [
                asyncio.create_task(capacity_test_worker(worker_rps))
                for _ in range(num_workers)
            ]
            
            capacity_results = await asyncio.gather(*capacity_tasks, return_exceptions=True)
            
            # Analyze capacity test results
            all_results = []
            for result in capacity_results:
                if isinstance(result, list):
                    all_results.extend(result)
            
            if all_results:
                successful = sum(1 for r in all_results if r["success"])
                success_rate = successful / len(all_results)
                response_times = [r["response_time"] for r in all_results if r["success"]]
                p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)] if response_times else float('inf')
                
                max_capacity_results[f"rps_{current_rps}"] = {
                    "rps": current_rps,
                    "total_requests": len(all_results),
                    "successful_requests": successful,
                    "success_rate": success_rate,
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "p95_response_time": p95_response_time
                }
                
                logger.info(f"Capacity {current_rps} RPS: {success_rate:.1%} success, P95: {p95_response_time:.0f}ms")
                
                # Check if we've reached the breaking point
                if success_rate < 0.95 or p95_response_time > 5000:  # 95% success or P95 > 5s
                    breaking_point = current_rps
                    logger.info(f"Breaking point found at {current_rps} RPS")
                    break
            
            current_rps += step_size
            await asyncio.sleep(5)  # Brief pause between capacity tests
        
        max_capacity_results["breaking_point"] = breaking_point
        max_capacity_results["max_tested_rps"] = current_rps - step_size
        
        self.results["maximum_capacity"] = max_capacity_results
    
    def _generate_load_test_report(self) -> Dict:
        """Generate comprehensive load test report."""
        
        test_results = []
        
        # Concurrent request capacity
        if "concurrent_capacity" in self.results:
            capacity_data = self.results["concurrent_capacity"]
            max_successful_concurrency = 0
            
            for test_name, data in capacity_data.items():
                if data["success_rate"] >= 0.95:  # 95% success rate
                    concurrency = data["total_requests"]
                    max_successful_concurrency = max(max_successful_concurrency, concurrency)
            
            test_results.append({
                "test_name": "Maximum Concurrent Requests",
                "target": self.targets["max_concurrent_requests"],
                "actual": max_successful_concurrency,
                "unit": " requests",
                "passed": max_successful_concurrency >= self.targets["max_concurrent_requests"]
            })
        
        # Sustained RPS
        if "sustained_load" in self.results:
            sustained_data = self.results["sustained_load"]
            max_sustained_rps = 0
            
            for test_name, data in sustained_data.items():
                if data["success_rate"] >= 0.99:  # 99% success rate
                    rps = data["actual_rps"]
                    max_sustained_rps = max(max_sustained_rps, rps)
            
            test_results.append({
                "test_name": "Sustained Requests Per Second",
                "target": self.targets["min_sustained_rps"],
                "actual": max_sustained_rps,
                "unit": " RPS",
                "passed": max_sustained_rps >= self.targets["min_sustained_rps"]
            })
        
        # P95 Response Time
        if "concurrent_capacity" in self.results:
            capacity_data = self.results["concurrent_capacity"]
            worst_p95 = 0
            
            for test_name, data in capacity_data.items():
                p95 = data.get("p95_response_time", 0)
                worst_p95 = max(worst_p95, p95)
            
            test_results.append({
                "test_name": "P95 Response Time Under Load",
                "target": self.targets["max_response_time_p95"],
                "actual": worst_p95,
                "unit": "ms",
                "passed": worst_p95 <= self.targets["max_response_time_p95"]
            })
        
        # Multiple client handling
        if "multiple_clients" in self.results:
            client_data = self.results["multiple_clients"]
            max_clients_supported = 0
            
            for test_name, data in client_data.items():
                if data["success_rate"] >= 0.99:  # 99% success rate
                    clients = data["num_clients"]
                    max_clients_supported = max(max_clients_supported, clients)
            
            test_results.append({
                "test_name": "Concurrent Client Connections",
                "target": self.targets["concurrent_client_capacity"],
                "actual": max_clients_supported,
                "unit": " clients",
                "passed": max_clients_supported >= self.targets["concurrent_client_capacity"]
            })
        
        # Agent system performance
        if "agent_system_load" in self.results:
            agent_data = self.results["agent_system_load"]
            agent_activation_successful = True
            max_activation_time = 0
            
            for test_name, data in agent_data.items():
                if not data["activation_successful"]:
                    agent_activation_successful = False
                max_activation_time = max(max_activation_time, data["activation_response_time"])
            
            test_results.append({
                "test_name": "Agent System Load Performance",
                "target": self.targets["max_agent_spawn_time"],
                "actual": max_activation_time,
                "unit": "ms",
                "passed": agent_activation_successful and max_activation_time <= self.targets["max_agent_spawn_time"]
            })
        
        # Calculate overall results
        passed_tests = sum(1 for test in test_results if test["passed"])
        total_tests = len(test_results)
        overall_pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Extract key performance metrics
        performance_summary = {}
        
        if "maximum_capacity" in self.results:
            max_cap = self.results["maximum_capacity"]
            performance_summary["breaking_point_rps"] = max_cap.get("breaking_point", "Not found")
            performance_summary["max_tested_rps"] = max_cap.get("max_tested_rps", 0)
        
        return {
            "summary": {
                "overall_status": "PASS" if overall_pass_rate >= 0.8 else "FAIL",  # 80% pass rate
                "pass_rate": overall_pass_rate,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": test_results,
            "performance_summary": performance_summary,
            "detailed_results": self.results,
            "targets": self.targets,
            "recommendations": self._generate_load_recommendations(test_results)
        }
    
    def _generate_load_recommendations(self, test_results: List[Dict]) -> List[str]:
        """Generate recommendations for load testing improvements."""
        recommendations = []
        
        failed_tests = [test for test in test_results if not test["passed"]]
        
        for test in failed_tests:
            if "Concurrent Requests" in test["test_name"]:
                recommendations.append("Increase connection pooling, implement request queuing, or scale horizontally to handle more concurrent requests")
            elif "Sustained Requests" in test["test_name"]:
                recommendations.append("Optimize application performance, implement caching, or add load balancing for sustained high throughput")
            elif "Response Time" in test["test_name"]:
                recommendations.append("Optimize database queries, implement response caching, or reduce processing overhead to improve response times")
            elif "Client Connections" in test["test_name"]:
                recommendations.append("Implement connection pooling, optimize session management, or increase server capacity for multiple clients")
            elif "Agent System" in test["test_name"]:
                recommendations.append("Optimize agent spawning process, implement resource pre-allocation, or improve system initialization")
        
        if not recommendations:
            recommendations.append("All load tests passed! System demonstrates excellent scalability and performance under load.")
        
        return recommendations


async def main():
    """Main execution function."""
    tester = LoadTester()
    
    logger.info("🔄 Starting Load Testing")
    
    # Run load tests
    report = await tester.run_load_tests()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/tmp/load_test_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("LOAD TESTING RESULTS")
    print("="*80)
    
    summary = report["summary"]
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
    
    print("\nPERFORMANCE SUMMARY:")
    perf_summary = report["performance_summary"]
    for key, value in perf_summary.items():
        print(f"  {key}: {value}")
    
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
    import sys
    
    # Run load testing
    success = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)