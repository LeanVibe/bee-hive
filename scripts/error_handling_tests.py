#!/usr/bin/env python3
"""
Error Handling and Recovery Tests for LeanVibe Agent Hive 2.0

Tests system resilience, error recovery, and graceful degradation
under various failure conditions.
"""

import asyncio
import time
import json
import httpx
import subprocess
from datetime import datetime
from typing import Dict, List
import structlog

logger = structlog.get_logger()


class ErrorHandlingTester:
    """Test error handling, recovery, and system resilience."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
    
    async def run_error_handling_tests(self) -> Dict:
        """Run comprehensive error handling tests."""
        logger.info("🚧 Starting Error Handling and Recovery Tests")
        
        # Test 1: Database connection resilience
        await self._test_database_resilience()
        
        # Test 2: Redis connection resilience
        await self._test_redis_resilience()
        
        # Test 3: Network failure handling
        await self._test_network_failure_handling()
        
        # Test 4: Resource exhaustion handling
        await self._test_resource_exhaustion()
        
        # Test 5: Graceful error responses
        await self._test_graceful_error_responses()
        
        # Test 6: System recovery time
        await self._test_system_recovery()
        
        return self._generate_error_handling_report()
    
    async def _test_database_resilience(self):
        """Test system behavior when database is under stress or fails."""
        logger.info("Testing database resilience")
        
        # Create database stress by making many concurrent requests
        async def db_stress_request():
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(f"{self.base_url}/health")
                    return response.status_code == 200, response.json() if response.status_code == 200 else None
            except Exception as e:
                return False, str(e)
        
        # Test with increasing concurrent DB operations
        db_results = {}
        
        for concurrency in [10, 25, 50, 100]:
            logger.info(f"Testing database under {concurrency} concurrent operations")
            
            start_time = time.time()
            tasks = [asyncio.create_task(db_stress_request()) for _ in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            
            successful = sum(1 for r in results if isinstance(r, tuple) and r[0])
            failed = len(results) - successful
            
            # Check if database is still healthy in successful responses
            db_healthy_count = 0
            for r in results:
                if isinstance(r, tuple) and r[0] and r[1]:
                    health_data = r[1]
                    if health_data.get("components", {}).get("database", {}).get("status") == "healthy":
                        db_healthy_count += 1
            
            db_results[f"concurrency_{concurrency}"] = {
                "total_requests": len(results),
                "successful_requests": successful,
                "failed_requests": failed,
                "db_healthy_responses": db_healthy_count,
                "duration_seconds": duration,
                "success_rate": successful / len(results) if results else 0,
                "db_health_rate": db_healthy_count / successful if successful > 0 else 0
            }
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        self.results["database_resilience"] = db_results
    
    async def _test_redis_resilience(self):
        """Test system behavior when Redis is under stress."""
        logger.info("Testing Redis resilience")
        
        # Similar to database test, but focus on Redis-dependent operations
        redis_results = {}
        
        async def redis_dependent_request():
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        health_data = response.json()
                        redis_status = health_data.get("components", {}).get("redis", {}).get("status")
                        return True, redis_status == "healthy"
                    return False, False
            except Exception:
                return False, False
        
        # Test Redis under load
        for concurrency in [20, 50, 100]:
            logger.info(f"Testing Redis under {concurrency} concurrent operations")
            
            start_time = time.time()
            tasks = [asyncio.create_task(redis_dependent_request()) for _ in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            
            successful = sum(1 for r in results if isinstance(r, tuple) and r[0])
            redis_healthy = sum(1 for r in results if isinstance(r, tuple) and r[0] and r[1])
            
            redis_results[f"concurrency_{concurrency}"] = {
                "total_requests": len(results),
                "successful_requests": successful,
                "redis_healthy_responses": redis_healthy,
                "duration_seconds": duration,
                "success_rate": successful / len(results) if results else 0,
                "redis_health_rate": redis_healthy / successful if successful > 0 else 0
            }
            
            await asyncio.sleep(2)
        
        self.results["redis_resilience"] = redis_results
    
    async def _test_network_failure_handling(self):
        """Test handling of network failures and timeouts."""
        logger.info("Testing network failure handling")
        
        network_results = {}
        
        # Test with very short timeouts to simulate network issues
        timeout_values = [0.1, 0.5, 1.0, 5.0]  # seconds
        
        for timeout in timeout_values:
            logger.info(f"Testing with {timeout}s timeout")
            
            successful = 0
            timeout_errors = 0
            other_errors = 0
            total_requests = 20
            
            for _ in range(total_requests):
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.get(f"{self.base_url}/health")
                        if response.status_code == 200:
                            successful += 1
                        else:
                            other_errors += 1
                except httpx.TimeoutException:
                    timeout_errors += 1
                except Exception:
                    other_errors += 1
            
            network_results[f"timeout_{timeout}s"] = {
                "total_requests": total_requests,
                "successful_requests": successful,
                "timeout_errors": timeout_errors,
                "other_errors": other_errors,
                "success_rate": successful / total_requests,
                "timeout_rate": timeout_errors / total_requests
            }
        
        self.results["network_failure_handling"] = network_results
    
    async def _test_resource_exhaustion(self):
        """Test system behavior under resource exhaustion."""
        logger.info("Testing resource exhaustion handling")
        
        # Create high load to potentially exhaust resources
        async def resource_intensive_request():
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Make multiple rapid requests to increase load
                    tasks = []
                    for _ in range(5):
                        task = asyncio.create_task(client.get(f"{self.base_url}/health"))
                        tasks.append(task)
                    
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
                    return successful, len(responses)
            except Exception:
                return 0, 5
        
        # Run resource exhaustion test
        start_time = time.time()
        
        # Create 50 concurrent resource-intensive operations
        tasks = [asyncio.create_task(resource_intensive_request()) for _ in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        
        total_requests = 0
        successful_requests = 0
        
        for result in results:
            if isinstance(result, tuple):
                successful, total = result
                successful_requests += successful
                total_requests += total
        
        resource_results = {
            "total_operations": len(tasks),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "duration_seconds": duration,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "operations_per_second": len(tasks) / duration,
            "requests_per_second": total_requests / duration
        }
        
        self.results["resource_exhaustion"] = resource_results
    
    async def _test_graceful_error_responses(self):
        """Test that the system returns graceful error responses."""
        logger.info("Testing graceful error responses")
        
        error_endpoints = [
            ("/nonexistent", "404 Not Found"),
            ("/api/v1/nonexistent", "404 Not Found"),
            ("/health/invalid", "404 Not Found"),
            ("/api/invalid", "404 Not Found")
        ]
        
        error_response_results = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for endpoint, expected_error in error_endpoints:
                logger.info(f"Testing error endpoint: {endpoint}")
                
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    
                    # Check if error is handled gracefully
                    graceful = True
                    error_details = {}
                    
                    # Response should have appropriate status code
                    if response.status_code not in [404, 405, 422, 500]:
                        graceful = False
                        error_details["unexpected_status"] = response.status_code
                    
                    # Response should have valid JSON (if applicable)
                    try:
                        response_data = response.json()
                        error_details["has_json_response"] = True
                        error_details["response_data"] = response_data
                    except Exception:
                        error_details["has_json_response"] = False
                    
                    # Response should be fast
                    response_time = response.elapsed.total_seconds() * 1000
                    error_details["response_time_ms"] = response_time
                    
                    if response_time > 5000:  # 5 seconds
                        graceful = False
                        error_details["slow_response"] = True
                    
                    error_response_results[endpoint] = {
                        "status_code": response.status_code,
                        "graceful": graceful,
                        "details": error_details
                    }
                    
                except Exception as e:
                    error_response_results[endpoint] = {
                        "status_code": "exception",
                        "graceful": False,
                        "details": {"exception": str(e)}
                    }
        
        self.results["error_responses"] = error_response_results
    
    async def _test_system_recovery(self):
        """Test system recovery time after stress."""
        logger.info("Testing system recovery time")
        
        # First, stress the system
        logger.info("Creating system stress...")
        
        async def stress_request():
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{self.base_url}/health")
                    return response.status_code == 200
            except Exception:
                return False
        
        # Create stress with 100 concurrent requests
        stress_tasks = [asyncio.create_task(stress_request()) for _ in range(100)]
        stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)
        
        # Now test recovery
        logger.info("Testing recovery time...")
        
        recovery_start = time.time()
        recovery_times = []
        consecutive_successes = 0
        
        # Test recovery by making requests until we get 10 consecutive successes
        while consecutive_successes < 10 and (time.time() - recovery_start) < 60:  # Max 60 seconds
            request_start = time.time()
            
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{self.base_url}/health")
                    request_time = (time.time() - request_start) * 1000
                    recovery_times.append(request_time)
                    
                    if response.status_code == 200:
                        consecutive_successes += 1
                    else:
                        consecutive_successes = 0
                        
            except Exception:
                request_time = (time.time() - request_start) * 1000
                recovery_times.append(request_time)
                consecutive_successes = 0
            
            await asyncio.sleep(0.5)  # 500ms between recovery checks
        
        total_recovery_time = time.time() - recovery_start
        
        recovery_results = {
            "stress_requests": len(stress_tasks),
            "stress_successful": sum(1 for r in stress_results if r is True),
            "recovery_time_seconds": total_recovery_time,
            "recovery_successful": consecutive_successes >= 10,
            "recovery_attempts": len(recovery_times),
            "avg_response_time_during_recovery": sum(recovery_times) / len(recovery_times) if recovery_times else 0
        }
        
        self.results["system_recovery"] = recovery_results
    
    def _generate_error_handling_report(self) -> Dict:
        """Generate comprehensive error handling report."""
        
        # Analyze results
        test_results = []
        
        # Database resilience
        if "database_resilience" in self.results:
            db_results = self.results["database_resilience"]
            avg_success_rate = sum(r["success_rate"] for r in db_results.values()) / len(db_results)
            
            test_results.append({
                "test_name": "Database Resilience",
                "target": 0.95,  # 95% success rate under stress
                "actual": avg_success_rate,
                "unit": " success rate",
                "passed": avg_success_rate >= 0.95
            })
        
        # Redis resilience
        if "redis_resilience" in self.results:
            redis_results = self.results["redis_resilience"]
            avg_success_rate = sum(r["success_rate"] for r in redis_results.values()) / len(redis_results)
            
            test_results.append({
                "test_name": "Redis Resilience",
                "target": 0.95,  # 95% success rate under stress
                "actual": avg_success_rate,
                "unit": " success rate",
                "passed": avg_success_rate >= 0.95
            })
        
        # Network failure handling
        if "network_failure_handling" in self.results:
            network_results = self.results["network_failure_handling"]
            # Check that reasonable timeouts (>= 1s) have good success rates
            reasonable_timeout_results = [r for k, r in network_results.items() if "1.0" in k or "5.0" in k]
            if reasonable_timeout_results:
                avg_success_rate = sum(r["success_rate"] for r in reasonable_timeout_results) / len(reasonable_timeout_results)
                
                test_results.append({
                    "test_name": "Network Failure Handling",
                    "target": 0.90,  # 90% success rate with reasonable timeouts
                    "actual": avg_success_rate,
                    "unit": " success rate",
                    "passed": avg_success_rate >= 0.90
                })
        
        # Resource exhaustion handling
        if "resource_exhaustion" in self.results:
            resource_result = self.results["resource_exhaustion"]
            success_rate = resource_result["success_rate"]
            
            test_results.append({
                "test_name": "Resource Exhaustion Handling",
                "target": 0.80,  # 80% success rate under resource stress
                "actual": success_rate,
                "unit": " success rate",
                "passed": success_rate >= 0.80
            })
        
        # Graceful error responses
        if "error_responses" in self.results:
            error_results = self.results["error_responses"]
            graceful_responses = sum(1 for r in error_results.values() if r["graceful"])
            total_error_tests = len(error_results)
            graceful_rate = graceful_responses / total_error_tests if total_error_tests > 0 else 0
            
            test_results.append({
                "test_name": "Graceful Error Responses",
                "target": 1.0,  # 100% of errors should be handled gracefully
                "actual": graceful_rate,
                "unit": " graceful rate",
                "passed": graceful_rate >= 1.0
            })
        
        # System recovery
        if "system_recovery" in self.results:
            recovery_result = self.results["system_recovery"]
            recovery_time = recovery_result["recovery_time_seconds"]
            recovery_successful = recovery_result["recovery_successful"]
            
            test_results.append({
                "test_name": "System Recovery Time",
                "target": 30.0,  # Should recover within 30 seconds
                "actual": recovery_time,
                "unit": " seconds",
                "passed": recovery_successful and recovery_time <= 30.0
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
            "recommendations": self._generate_error_recommendations(test_results)
        }
    
    def _generate_error_recommendations(self, test_results: List[Dict]) -> List[str]:
        """Generate recommendations for error handling improvements."""
        recommendations = []
        
        failed_tests = [test for test in test_results if not test["passed"]]
        
        for test in failed_tests:
            if "Database Resilience" in test["test_name"]:
                recommendations.append("Improve database connection pooling, implement circuit breakers, and add database failover mechanisms")
            elif "Redis Resilience" in test["test_name"]:
                recommendations.append("Implement Redis clustering, connection pooling, and graceful degradation when Redis is unavailable")
            elif "Network Failure" in test["test_name"]:
                recommendations.append("Implement retry mechanisms with exponential backoff and improve timeout handling")
            elif "Resource Exhaustion" in test["test_name"]:
                recommendations.append("Implement rate limiting, resource monitoring, and graceful degradation under high load")
            elif "Graceful Error" in test["test_name"]:
                recommendations.append("Improve error handling middleware to ensure all errors return appropriate status codes and messages")
            elif "Recovery Time" in test["test_name"]:
                recommendations.append("Optimize system recovery mechanisms and implement health checks for faster recovery detection")
        
        if not recommendations:
            recommendations.append("All error handling tests passed! System demonstrates excellent resilience and recovery capabilities.")
        
        return recommendations


async def main():
    """Main execution function."""
    tester = ErrorHandlingTester()
    
    logger.info("🚧 Starting Error Handling and Recovery Tests")
    
    # Run error handling tests
    report = await tester.run_error_handling_tests()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/tmp/error_handling_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("ERROR HANDLING AND RECOVERY TEST RESULTS")
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
    
    class ErrorHandlingTests(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            import sys

            # Run error handling tests
            await main()

            # Exit with appropriate code
            sys.exit(0 if success else 1)
            
            return {"status": "completed"}
    
    script_main(ErrorHandlingTests)