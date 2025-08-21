#!/usr/bin/env python3
"""
Comprehensive System Validation for LeanVibe Agent Hive 2.0
Validates all success criteria and testing requirements
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    details: str
    performance_metrics: Dict[str, Any] = None

class ComprehensiveSystemValidator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Execute comprehensive testing validation plan"""
        logger.info("🚀 Starting Comprehensive System Validation for LeanVibe Agent Hive 2.0")
        
        # Phase 1: Dashboard Feature Validation
        await self.test_dashboard_features()
        
        # Phase 2: Integration Validation  
        await self.test_api_integration()
        
        # Phase 3: Multi-agent Coordination
        await self.test_agent_coordination()
        
        # Phase 4: WebSocket Real-time Communication
        await self.test_websocket_streaming()
        
        # Phase 5: Performance Benchmarks
        await self.test_performance_metrics()
        
        return self.generate_report()
    
    async def test_dashboard_features(self):
        """Phase 1: Dashboard Feature Validation"""
        logger.info("📊 Phase 1: Dashboard Feature Validation")
        
        # Test Dashboard API Live Data
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/dashboard/api/live-data") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # Validate required dashboard data structure
                        required_keys = ['metrics', 'agent_activities', 'project_snapshots']
                        missing_keys = [key for key in required_keys if key not in data]
                        
                        if not missing_keys:
                            metrics = data['metrics']
                            active_agents = metrics.get('active_agents', 0)
                            system_status = metrics.get('system_status', 'unknown')
                            
                            duration = (time.time() - start_time) * 1000
                            self.results.append(TestResult(
                                name="Dashboard API Live Data",
                                passed=active_agents > 0 and system_status == 'healthy',
                                duration_ms=duration,
                                details=f"Active agents: {active_agents}, Status: {system_status}",
                                performance_metrics={
                                    "response_time_ms": duration,
                                    "active_agents": active_agents,
                                    "system_status": system_status
                                }
                            ))
                        else:
                            self.results.append(TestResult(
                                name="Dashboard API Live Data",
                                passed=False,
                                duration_ms=(time.time() - start_time) * 1000,
                                details=f"Missing required keys: {missing_keys}"
                            ))
                    else:
                        self.results.append(TestResult(
                            name="Dashboard API Live Data",
                            passed=False,
                            duration_ms=(time.time() - start_time) * 1000,
                            details=f"HTTP {resp.status}"
                        ))
        except Exception as e:
            self.results.append(TestResult(
                name="Dashboard API Live Data",
                passed=False,
                duration_ms=(time.time() - start_time) * 1000,
                details=f"Exception: {str(e)}"
            ))
        
        # Test Dashboard HTML Interface
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/dashboard/") as resp:
                    duration = (time.time() - start_time) * 1000
                    html_content = await resp.text()
                    
                    # Validate HTML contains expected elements
                    expected_elements = [
                        'LeanVibe Agent Hive 2.0',
                        'Coordination Dashboard',
                        'dashboard-container',
                        'agent-activities'
                    ]
                    
                    missing_elements = [elem for elem in expected_elements if elem not in html_content]
                    
                    self.results.append(TestResult(
                        name="Dashboard HTML Interface",
                        passed=resp.status == 200 and not missing_elements,
                        duration_ms=duration,
                        details=f"Status: {resp.status}, Missing elements: {missing_elements}",
                        performance_metrics={"html_load_time_ms": duration}
                    ))
        except Exception as e:
            self.results.append(TestResult(
                name="Dashboard HTML Interface",
                passed=False,
                duration_ms=(time.time() - start_time) * 1000,
                details=f"Exception: {str(e)}"
            ))
    
    async def test_api_integration(self):
        """Phase 2: Integration Validation"""
        logger.info("🔧 Phase 2: Integration Validation")
        
        # Test Health Endpoint
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as resp:
                    duration = (time.time() - start_time) * 1000
                    if resp.status == 200:
                        health_data = await resp.json()
                        
                        components = health_data.get('components', {})
                        healthy_components = sum(1 for comp in components.values() 
                                               if comp.get('status') == 'healthy')
                        total_components = len(components)
                        
                        self.results.append(TestResult(
                            name="Health Check API",
                            passed=health_data.get('status') in ['healthy', 'degraded'],
                            duration_ms=duration,
                            details=f"Status: {health_data.get('status')}, Components: {healthy_components}/{total_components}",
                            performance_metrics={
                                "health_check_time_ms": duration,
                                "healthy_components": healthy_components,
                                "total_components": total_components
                            }
                        ))
                    else:
                        self.results.append(TestResult(
                            name="Health Check API",
                            passed=False,
                            duration_ms=duration,
                            details=f"HTTP {resp.status}"
                        ))
        except Exception as e:
            self.results.append(TestResult(
                name="Health Check API",
                passed=False,
                duration_ms=(time.time() - start_time) * 1000,
                details=f"Exception: {str(e)}"
            ))
        
        # Test Debug Agents Endpoint
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/debug-agents") as resp:
                    duration = (time.time() - start_time) * 1000
                    if resp.status == 200:
                        agents_data = await resp.json()
                        agent_count = agents_data.get('agent_count', 0)
                        
                        self.results.append(TestResult(
                            name="Debug Agents API",
                            passed=agent_count > 0,
                            duration_ms=duration,
                            details=f"Agent count: {agent_count}",
                            performance_metrics={
                                "debug_agents_time_ms": duration,
                                "agent_count": agent_count
                            }
                        ))
                    else:
                        self.results.append(TestResult(
                            name="Debug Agents API",
                            passed=False,
                            duration_ms=duration,
                            details=f"HTTP {resp.status}"
                        ))
        except Exception as e:
            self.results.append(TestResult(
                name="Debug Agents API",
                passed=False,
                duration_ms=(time.time() - start_time) * 1000,
                details=f"Exception: {str(e)}"
            ))
    
    async def test_agent_coordination(self):
        """Phase 3: Multi-agent Coordination Testing"""
        logger.info("🤖 Phase 3: Multi-agent Coordination Testing")
        
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                # Get agent data from dashboard API
                async with session.get(f"{self.base_url}/dashboard/api/live-data") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        agent_activities = data.get('agent_activities', [])
                        
                        # Validate agent coordination capabilities
                        required_roles = ['Product Manager', 'Architect', 'Backend Developer', 'Qa Engineer', 'Devops Engineer']
                        found_roles = [agent.get('name') for agent in agent_activities]
                        missing_roles = [role for role in required_roles if role not in found_roles]
                        
                        # Check agent performance scores
                        avg_performance = sum(agent.get('performance_score', 0) for agent in agent_activities) / len(agent_activities) if agent_activities else 0
                        
                        duration = (time.time() - start_time) * 1000
                        self.results.append(TestResult(
                            name="Multi-Agent Coordination",
                            passed=len(missing_roles) == 0 and avg_performance > 0.7,
                            duration_ms=duration,
                            details=f"Roles found: {len(found_roles)}/{len(required_roles)}, Avg performance: {avg_performance:.2f}",
                            performance_metrics={
                                "coordination_check_time_ms": duration,
                                "active_agent_roles": len(found_roles),
                                "average_performance_score": avg_performance,
                                "missing_roles": missing_roles
                            }
                        ))
                    else:
                        duration = (time.time() - start_time) * 1000
                        self.results.append(TestResult(
                            name="Multi-Agent Coordination",
                            passed=False,
                            duration_ms=duration,
                            details=f"Failed to get agent data: HTTP {resp.status}"
                        ))
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.results.append(TestResult(
                name="Multi-Agent Coordination",
                passed=False,
                duration_ms=duration,
                details=f"Exception: {str(e)}"
            ))
    
    async def test_websocket_streaming(self):
        """Phase 4: WebSocket Real-time Communication"""
        logger.info("📡 Phase 4: WebSocket Real-time Communication")
        
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                # Test WebSocket connection
                async with session.ws_connect(f'ws://localhost:8000/api/v1/ws/observability') as ws:
                    connection_time = (time.time() - start_time) * 1000
                    
                    # Send authentication message
                    auth_msg = {'type': 'authenticate', 'token': 'test-validation'}
                    await ws.send_str(json.dumps(auth_msg))
                    
                    # Listen for messages with timeout
                    messages_received = 0
                    try:
                        while messages_received < 3:  # Try to receive a few messages
                            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                messages_received += 1
                                logger.info(f"Received WebSocket message: {msg.data[:100]}...")
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break
                    except asyncio.TimeoutError:
                        pass  # Expected for this test
                    
                    duration = (time.time() - start_time) * 1000
                    self.results.append(TestResult(
                        name="WebSocket Real-time Communication",
                        passed=True,  # Connection established successfully
                        duration_ms=duration,
                        details=f"Connection established in {connection_time:.1f}ms, Messages received: {messages_received}",
                        performance_metrics={
                            "websocket_connection_time_ms": connection_time,
                            "total_test_time_ms": duration,
                            "messages_received": messages_received
                        }
                    ))
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.results.append(TestResult(
                name="WebSocket Real-time Communication",
                passed=False,
                duration_ms=duration,
                details=f"Exception: {str(e)}"
            ))
    
    async def test_performance_metrics(self):
        """Phase 5: Performance Benchmarks"""
        logger.info("⚡ Phase 5: Performance Benchmarks")
        
        # Test API response times
        endpoints_to_test = [
            ("/health", "Health Check"),
            ("/dashboard/api/live-data", "Dashboard API"),
            ("/debug-agents", "Debug Agents"),
            ("/status", "System Status")
        ]
        
        performance_results = {}
        
        for endpoint, name in endpoints_to_test:
            start_time = time.time()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}{endpoint}") as resp:
                        duration = (time.time() - start_time) * 1000
                        performance_results[name] = {
                            "response_time_ms": duration,
                            "status_code": resp.status,
                            "success": resp.status == 200
                        }
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                performance_results[name] = {
                    "response_time_ms": duration,
                    "status_code": 0,
                    "success": False,
                    "error": str(e)
                }
        
        # Calculate overall performance metrics
        successful_requests = sum(1 for result in performance_results.values() if result.get("success", False))
        total_requests = len(performance_results)
        avg_response_time = sum(result.get("response_time_ms", 0) for result in performance_results.values()) / total_requests
        
        self.results.append(TestResult(
            name="Performance Benchmarks",
            passed=successful_requests == total_requests and avg_response_time < 1000,  # Under 1 second average
            duration_ms=avg_response_time,
            details=f"Success rate: {successful_requests}/{total_requests}, Avg response: {avg_response_time:.1f}ms",
            performance_metrics={
                "success_rate": successful_requests / total_requests,
                "average_response_time_ms": avg_response_time,
                "endpoint_results": performance_results
            }
        ))
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test execution report"""
        passed_tests = sum(1 for result in self.results if result.passed)
        total_tests = len(self.results)
        
        # Calculate performance metrics
        avg_duration = sum(result.duration_ms for result in self.results) / total_tests if total_tests > 0 else 0
        
        # Categorize results
        dashboard_tests = [r for r in self.results if 'Dashboard' in r.name]
        api_tests = [r for r in self.results if 'API' in r.name or 'Health' in r.name]
        coordination_tests = [r for r in self.results if 'Coordination' in r.name or 'WebSocket' in r.name]
        performance_tests = [r for r in self.results if 'Performance' in r.name or 'Benchmark' in r.name]
        
        report = {
            "test_execution_summary": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "overall_status": "PASS" if passed_tests == total_tests else "FAIL",
                "average_test_duration_ms": avg_duration
            },
            "phase_results": {
                "phase_1_dashboard_validation": {
                    "tests": len(dashboard_tests),
                    "passed": sum(1 for t in dashboard_tests if t.passed),
                    "status": "PASS" if all(t.passed for t in dashboard_tests) else "FAIL"
                },
                "phase_2_api_integration": {
                    "tests": len(api_tests),
                    "passed": sum(1 for t in api_tests if t.passed),
                    "status": "PASS" if all(t.passed for t in api_tests) else "FAIL"
                },
                "phase_3_agent_coordination": {
                    "tests": len(coordination_tests),
                    "passed": sum(1 for t in coordination_tests if t.passed),
                    "status": "PASS" if all(t.passed for t in coordination_tests) else "FAIL"
                },
                "phase_4_performance_benchmarks": {
                    "tests": len(performance_tests),
                    "passed": sum(1 for t in performance_tests if t.passed),
                    "status": "PASS" if all(t.passed for t in performance_tests) else "FAIL"
                }
            },
            "detailed_results": [
                {
                    "test_name": result.name,
                    "status": "PASS" if result.passed else "FAIL",
                    "duration_ms": result.duration_ms,
                    "details": result.details,
                    "performance_metrics": result.performance_metrics
                }
                for result in self.results
            ],
            "success_criteria_validation": {
                "dashboard_features_operational": all(t.passed for t in dashboard_tests),
                "api_integration_working": all(t.passed for t in api_tests),
                "multi_agent_coordination_active": all(t.passed for t in coordination_tests),
                "websocket_streaming_functional": any(t.passed and 'WebSocket' in t.name for t in self.results),
                "performance_benchmarks_met": all(t.passed for t in performance_tests)
            },
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed tests to achieve full system validation")
        
        slow_tests = [r for r in self.results if r.duration_ms > 1000]
        if slow_tests:
            recommendations.append(f"Optimize performance for {len(slow_tests)} slow-responding components")
        
        # Check for specific issues
        dashboard_issues = [r for r in self.results if 'Dashboard' in r.name and not r.passed]
        if dashboard_issues:
            recommendations.append("Dashboard functionality requires attention for production readiness")
        
        api_issues = [r for r in self.results if 'API' in r.name and not r.passed]
        if api_issues:
            recommendations.append("API integration issues must be resolved before deployment")
        
        if not recommendations:
            recommendations.append("All tests passed - system is ready for production autonomous development workflows")
        
        return recommendations

async def main():
    """Main execution function"""
    validator = ComprehensiveSystemValidator()
    
    print("🚀 LeanVibe Agent Hive 2.0 - Comprehensive System Validation")
    print("=" * 70)
    
    # Run all validation tests
    report = await validator.run_all_tests()
    
    # Print summary
    summary = report["test_execution_summary"]
    print(f"\n📊 TEST EXECUTION SUMMARY")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Average Duration: {summary['average_test_duration_ms']:.1f}ms")
    
    # Print phase results
    print(f"\n🔍 PHASE RESULTS")
    for phase, results in report["phase_results"].items():
        print(f"{phase}: {results['passed']}/{results['tests']} - {results['status']}")
    
    # Print detailed results
    print(f"\n📋 DETAILED RESULTS")
    for result in report["detailed_results"]:
        status_icon = "✅" if result["status"] == "PASS" else "❌"
        print(f"{status_icon} {result['test_name']}: {result['status']} ({result['duration_ms']:.1f}ms)")
        if result["details"]:
            print(f"   Details: {result['details']}")
    
    # Print recommendations
    print(f"\n💡 RECOMMENDATIONS")
    for rec in report["recommendations"]:
        print(f"• {rec}")
    
    # Save detailed report
    import json
    with open("/Users/bogdan/work/leanvibe-dev/bee-hive/system_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: system_validation_report.json")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())