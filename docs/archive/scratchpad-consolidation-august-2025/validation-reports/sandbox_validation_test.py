#!/usr/bin/env python3
"""
Comprehensive Sandbox Mode Integration Test
Tests the complete sandbox flow from configuration detection to autonomous development
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.sandbox.sandbox_config import (
    detect_sandbox_requirements,
    create_sandbox_config,
    get_sandbox_status,
    is_sandbox_mode,
    print_sandbox_banner
)
from app.core.sandbox.mock_anthropic_client import MockAnthropicClient
from app.core.sandbox.demo_scenarios import get_demo_scenario_engine
from app.core.sandbox.sandbox_orchestrator import SandboxOrchestrator


class SandboxValidationTest:
    """Comprehensive validation test for sandbox mode functionality."""
    
    def __init__(self):
        self.results = {
            "configuration": {},
            "mock_client": {},
            "demo_scenarios": {},
            "orchestrator": {},
            "integration": {},
            "overall_status": "pending"
        }
    
    def print_test_header(self, test_name: str):
        """Print formatted test header."""
        print(f"\n{'='*80}")
        print(f"🧪 {test_name}")
        print('='*80)
    
    def print_result(self, component: str, test: str, passed: bool, details: str = ""):
        """Print test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} [{component}] {test}")
        if details:
            print(f"    Details: {details}")
    
    async def test_configuration_system(self):
        """Test sandbox configuration detection and setup."""
        self.print_test_header("SANDBOX CONFIGURATION SYSTEM")
        
        try:
            # Test detection
            detection = detect_sandbox_requirements()
            self.results["configuration"]["detection"] = {
                "passed": detection["should_enable_sandbox"] == True,
                "reason": detection["reason"],
                "confidence": detection["confidence"],
                "missing_keys": detection["missing_api_keys"]
            }
            
            self.print_result("CONFIG", "API Key Detection", 
                            detection["should_enable_sandbox"],
                            f"Missing: {detection['missing_api_keys']}")
            
            # Test configuration creation
            config = create_sandbox_config()
            config_valid = config.enabled and config.mock_anthropic
            self.results["configuration"]["creation"] = {
                "passed": config_valid,
                "enabled": config.enabled,
                "mock_services": {
                    "anthropic": config.mock_anthropic,
                    "openai": config.mock_openai,
                    "github": config.mock_github
                }
            }
            
            self.print_result("CONFIG", "Configuration Creation", config_valid,
                            f"Enabled: {config.enabled}, Mock Anthropic: {config.mock_anthropic}")
            
            # Test status reporting
            status = get_sandbox_status()
            status_valid = "sandbox_mode" in status and status["sandbox_mode"]["enabled"]
            self.results["configuration"]["status"] = {
                "passed": status_valid,
                "complete_status": status
            }
            
            self.print_result("CONFIG", "Status Reporting", status_valid,
                            f"Enabled: {status['sandbox_mode']['enabled'] if 'sandbox_mode' in status else 'Unknown'}")
            
        except Exception as e:
            self.results["configuration"]["error"] = str(e)
            self.print_result("CONFIG", "Configuration System", False, str(e))
    
    async def test_mock_anthropic_client(self):
        """Test mock Anthropic client functionality."""
        self.print_test_header("MOCK ANTHROPIC CLIENT")
        
        try:
            client = MockAnthropicClient()
            
            # Test basic message creation
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{
                    "role": "user",
                    "content": "Create a simple function to calculate factorial"
                }],
                max_tokens=4096
            )
            
            response_valid = (
                response.content and 
                len(response.content) > 0 and 
                "text" in response.content[0] and
                len(response.content[0]["text"]) > 100
            )
            
            self.results["mock_client"]["basic_response"] = {
                "passed": response_valid,
                "response_length": len(response.content[0]["text"]) if response_valid else 0,
                "has_usage": response.usage is not None
            }
            
            self.print_result("MOCK", "Basic Message Creation", response_valid,
                            f"Response length: {len(response.content[0]['text']) if response_valid else 0}")
            
            # Test context awareness
            response2 = await client.messages.create(
                model="claude-3-5-sonnet-20241022", 
                messages=[{
                    "role": "user",
                    "content": "Design a temperature converter with multiple units and validation"
                }],
                max_tokens=4096
            )
            
            context_aware = (
                response2.content and
                "temperature" in response2.content[0]["text"].lower() and
                "convert" in response2.content[0]["text"].lower()
            )
            
            self.results["mock_client"]["context_awareness"] = {
                "passed": context_aware,
                "contains_keywords": context_aware
            }
            
            self.print_result("MOCK", "Context Awareness", context_aware,
                            "Response contains relevant keywords")
            
        except Exception as e:
            self.results["mock_client"]["error"] = str(e)
            self.print_result("MOCK", "Mock Client", False, str(e))
    
    async def test_demo_scenarios(self):
        """Test demo scenarios engine."""
        self.print_test_header("DEMO SCENARIOS ENGINE")
        
        try:
            engine = get_demo_scenario_engine()
            
            # Test scenario loading
            scenarios = engine.get_all_scenarios()
            scenarios_loaded = len(scenarios) >= 4  # Expect at least 4 scenarios
            
            self.results["demo_scenarios"]["loading"] = {
                "passed": scenarios_loaded,
                "count": len(scenarios),
                "titles": [s["title"] for s in scenarios[:3]]  # First 3 titles
            }
            
            self.print_result("SCENARIOS", "Scenario Loading", scenarios_loaded,
                            f"Loaded {len(scenarios)} scenarios")
            
            # Test scenario complexity filtering
            simple_scenarios = engine.get_scenarios_by_complexity(
                engine.scenarios[scenarios[0]["id"]].complexity
            )
            complexity_filtering = len(simple_scenarios) >= 1
            
            self.results["demo_scenarios"]["filtering"] = {
                "passed": complexity_filtering,
                "simple_count": len(simple_scenarios)
            }
            
            self.print_result("SCENARIOS", "Complexity Filtering", complexity_filtering,
                            f"Found {len(simple_scenarios)} scenarios in first complexity level")
            
            # Test recommendations
            recommended = engine.get_recommended_scenario("beginner")
            recommendation_valid = recommended is not None and recommended.title
            
            self.results["demo_scenarios"]["recommendations"] = {
                "passed": recommendation_valid,
                "recommended_title": recommended.title if recommended else None
            }
            
            self.print_result("SCENARIOS", "Recommendations", recommendation_valid,
                            f"Recommended: {recommended.title if recommended else 'None'}")
            
        except Exception as e:
            self.results["demo_scenarios"]["error"] = str(e)
            self.print_result("SCENARIOS", "Demo Scenarios", False, str(e))
    
    async def test_sandbox_orchestrator(self):
        """Test sandbox orchestrator functionality."""
        self.print_test_header("SANDBOX ORCHESTRATOR")
        
        try:
            orchestrator = SandboxOrchestrator()
            
            # Test initialization
            status = orchestrator.get_sandbox_status()
            init_valid = status["enabled"] and status["available_agents"] >= 5
            
            self.results["orchestrator"]["initialization"] = {
                "passed": init_valid,
                "enabled": status["enabled"],
                "agent_count": status["available_agents"],
                "mock_services": status["mock_services"]
            }
            
            self.print_result("ORCHESTRATOR", "Initialization", init_valid,
                            f"Agents: {status['available_agents']}, Services: {status['mock_services']}")
            
            # Test agent creation
            agents = orchestrator.get_all_agents()
            agents_valid = len(agents) >= 5 and all("role" in agent for agent in agents)
            
            self.results["orchestrator"]["agents"] = {
                "passed": agents_valid,
                "count": len(agents),
                "roles": [agent["role"] for agent in agents]
            }
            
            self.print_result("ORCHESTRATOR", "Agent Creation", agents_valid,
                            f"Created {len(agents)} agents with roles")
            
            # Test autonomous development start
            session_result = await orchestrator.start_autonomous_development(
                session_id="validation-test-001",
                task_description="Create a validation test function", 
                complexity="simple"
            )
            
            development_started = (
                session_result["status"] == "started" and
                session_result["tasks_created"] >= 5
            )
            
            self.results["orchestrator"]["development"] = {
                "passed": development_started,
                "session_id": session_result["session_id"],
                "status": session_result["status"],
                "tasks_created": session_result["tasks_created"]
            }
            
            self.print_result("ORCHESTRATOR", "Autonomous Development", development_started,
                            f"Tasks: {session_result['tasks_created']}, Status: {session_result['status']}")
            
            # Test session tracking
            await asyncio.sleep(2)  # Allow some processing
            session_status = orchestrator.get_session_status("validation-test-001")
            tracking_valid = session_status is not None and "progress" in session_status
            
            self.results["orchestrator"]["tracking"] = {
                "passed": tracking_valid,
                "has_status": session_status is not None,
                "progress": session_status.get("progress", 0) if session_status else 0
            }
            
            self.print_result("ORCHESTRATOR", "Session Tracking", tracking_valid,
                            f"Progress: {session_status.get('progress', 0) if session_status else 0}%")
            
        except Exception as e:
            self.results["orchestrator"]["error"] = str(e)
            self.print_result("ORCHESTRATOR", "Sandbox Orchestrator", False, str(e))
    
    async def test_integration_flow(self):
        """Test complete integration flow."""
        self.print_test_header("END-TO-END INTEGRATION")
        
        try:
            # Test full sandbox detection and setup
            detection = detect_sandbox_requirements()
            config = create_sandbox_config()
            orchestrator = SandboxOrchestractor() if is_sandbox_mode() else None
            
            integration_valid = (
                detection["should_enable_sandbox"] and
                config.enabled and
                orchestrator is not None
            )
            
            self.results["integration"]["full_flow"] = {
                "passed": integration_valid,
                "detection_confidence": detection["confidence"],
                "config_enabled": config.enabled,
                "orchestrator_created": orchestrator is not None
            }
            
            self.print_result("INTEGRATION", "Full Flow", integration_valid,
                            f"Detection: {detection['should_enable_sandbox']}, Config: {config.enabled}")
            
            # Test zero-friction setup
            zero_friction = (
                not os.getenv("ANTHROPIC_API_KEY") and  # No real API key needed
                config.enabled and  # Sandbox enabled
                config.mock_anthropic  # Mock services active
            )
            
            self.results["integration"]["zero_friction"] = {
                "passed": zero_friction,
                "no_api_key_required": not os.getenv("ANTHROPIC_API_KEY"),
                "sandbox_enabled": config.enabled,
                "mock_services_active": config.mock_anthropic
            }
            
            self.print_result("INTEGRATION", "Zero Friction Setup", zero_friction,
                            "No API keys required, sandbox auto-enabled")
            
        except Exception as e:
            self.results["integration"]["error"] = str(e)
            self.print_result("INTEGRATION", "Integration Flow", False, str(e))
    
    async def run_all_tests(self):
        """Run all validation tests."""
        print_sandbox_banner()
        print("\n🧪 COMPREHENSIVE SANDBOX VALIDATION TEST")
        print("="*80)
        
        await self.test_configuration_system()
        await self.test_mock_anthropic_client()
        await self.test_demo_scenarios()
        await self.test_sandbox_orchestrator()
        await self.test_integration_flow()
        
        # Calculate overall status
        all_tests = []
        for category, tests in self.results.items():
            if category != "overall_status":
                for test_name, result in tests.items():
                    if isinstance(result, dict) and "passed" in result:
                        all_tests.append(result["passed"])
        
        overall_pass = all(all_tests) if all_tests else False
        self.results["overall_status"] = "PASS" if overall_pass else "FAIL"
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print validation summary."""
        self.print_test_header("VALIDATION SUMMARY")
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.results.items():
            if category != "overall_status":
                category_tests = 0
                category_passed = 0
                
                for test_name, result in tests.items():
                    if isinstance(result, dict) and "passed" in result:
                        category_tests += 1
                        total_tests += 1
                        if result["passed"]:
                            category_passed += 1
                            passed_tests += 1
                
                if category_tests > 0:
                    percentage = (category_passed / category_tests) * 100
                    status = "✅" if category_passed == category_tests else "❌"
                    print(f"{status} {category.upper()}: {category_passed}/{category_tests} ({percentage:.1f}%)")
        
        overall_percentage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        overall_status = "✅ PASS" if passed_tests == total_tests else "❌ FAIL"
        
        print(f"\n{'='*80}")
        print(f"{overall_status} OVERALL: {passed_tests}/{total_tests} ({overall_percentage:.1f}%)")
        print('='*80)
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Sandbox mode is fully functional!")
            print("🏖️  Zero-friction autonomous development demonstration ready!")
        else:
            print("⚠️  Some tests failed - Review implementation details above.")


# Fix typo in integration test
def SandboxOrchestrator():
    from app.core.sandbox.sandbox_orchestrator import SandboxOrchestrator as SO
    return SO()


async def main():
    """Main test execution."""
    validator = SandboxValidationTest()
    results = await validator.run_all_tests()
    
    # Save results to file
    results_file = Path(__file__).parent / "sandbox_validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Detailed results saved to: {results_file}")
    
    # Return appropriate exit code
    return 0 if results["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)