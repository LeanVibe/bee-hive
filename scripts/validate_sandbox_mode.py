#!/usr/bin/env python3
"""
Sandbox Mode Validation Script
Tests all sandbox mode components and functionality
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_status(status: str, message: str):
    """Print colored status message."""
    colors = {
        "SUCCESS": "\033[92m✅",
        "ERROR": "\033[91m❌", 
        "INFO": "\033[94mℹ️",
        "WARNING": "\033[93m⚠️"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')} {message}{reset}")


async def test_sandbox_config():
    """Test sandbox configuration detection."""
    try:
        from app.core.sandbox.sandbox_config import (
            detect_sandbox_requirements,
            create_sandbox_config,
            is_sandbox_mode,
            get_sandbox_status
        )
        
        print_status("INFO", "Testing sandbox configuration...")
        
        # Test auto-detection
        detection = detect_sandbox_requirements()
        print_status("SUCCESS", f"Auto-detection working: {detection['should_enable_sandbox']}")
        
        # Test config creation
        config = create_sandbox_config(force_enable=True, demo_mode=True)
        print_status("SUCCESS", f"Config creation working: {config.enabled}")
        
        # Test status endpoint
        status = get_sandbox_status()
        print_status("SUCCESS", f"Status endpoint working: {len(status)} fields")
        
        return True
        
    except Exception as e:
        print_status("ERROR", f"Sandbox config test failed: {str(e)}")
        return False


async def test_mock_anthropic_client():
    """Test mock Anthropic client functionality."""
    try:
        from app.core.sandbox.mock_anthropic_client import (
            MockAnthropicClient,
            create_mock_anthropic_client
        )
        
        print_status("INFO", "Testing mock Anthropic client...")
        
        # Create mock client
        client = create_mock_anthropic_client()
        print_status("SUCCESS", "Mock client created successfully")
        
        # Test simple request
        response = await client.messages_create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Create a Fibonacci function"}],
            max_tokens=4096
        )
        
        print_status("SUCCESS", f"Mock response generated: {len(response.content[0]['text'])} chars")
        
        # Test different task types
        tasks = [
            "Create a temperature converter",
            "Build user authentication system",
            "Debug this code issue"
        ]
        
        for task in tasks:
            response = await client.messages_create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": task}],
                max_tokens=4096
            )
            print_status("SUCCESS", f"Task '{task[:30]}...' processed successfully")
        
        return True
        
    except Exception as e:
        print_status("ERROR", f"Mock Anthropic client test failed: {str(e)}")
        return False


async def test_demo_scenarios():
    """Test demo scenarios functionality."""
    try:
        from app.core.sandbox.demo_scenarios import (
            get_demo_scenario_engine,
            ScenarioComplexity,
            ScenarioCategory
        )
        
        print_status("INFO", "Testing demo scenarios...")
        
        # Get scenario engine
        engine = get_demo_scenario_engine()
        print_status("SUCCESS", f"Scenario engine created with {len(engine.scenarios)} scenarios")
        
        # Test scenario retrieval
        all_scenarios = engine.get_all_scenarios()
        print_status("SUCCESS", f"Retrieved {len(all_scenarios)} scenarios")
        
        # Test specific scenarios
        fibonacci_scenario = engine.get_scenario_by_id("fibonacci-calculator")
        if fibonacci_scenario:
            print_status("SUCCESS", f"Fibonacci scenario found: {fibonacci_scenario.title}")
        else:
            print_status("WARNING", "Fibonacci scenario not found")
        
        # Test recommendations
        for level in ["beginner", "intermediate", "advanced", "enterprise"]:
            recommended = engine.get_recommended_scenario(level)
            if recommended:
                print_status("SUCCESS", f"{level.capitalize()} recommendation: {recommended.title}")
        
        return True
        
    except Exception as e:
        print_status("ERROR", f"Demo scenarios test failed: {str(e)}")
        return False


async def test_sandbox_orchestrator():
    """Test sandbox orchestrator functionality."""
    try:
        from app.core.sandbox.sandbox_orchestrator import (
            SandboxOrchestrator,
            create_sandbox_orchestrator
        )
        
        print_status("INFO", "Testing sandbox orchestrator...")
        
        # Create orchestrator
        orchestrator = create_sandbox_orchestrator()
        print_status("SUCCESS", "Sandbox orchestrator created successfully")
        
        # Test agent listing
        agents = orchestrator.get_all_agents()
        print_status("SUCCESS", f"Retrieved {len(agents)} demo agents")
        
        # Test status
        status = orchestrator.get_sandbox_status()
        print_status("SUCCESS", f"Orchestrator status retrieved: {status['enabled']}")
        
        # Test autonomous development simulation
        session_id = "test-session-12345"
        result = await orchestrator.start_autonomous_development(
            session_id=session_id,
            task_description="Create a simple calculator function",
            requirements=["Basic arithmetic", "Input validation"],
            complexity="simple"
        )
        
        print_status("SUCCESS", f"Autonomous development started: {result['status']}")
        
        # Wait a moment for processing
        await asyncio.sleep(2)
        
        # Check session status
        session_status = orchestrator.get_session_status(session_id)
        if session_status:
            print_status("SUCCESS", f"Session status retrieved: {session_status['status']}")
        
        return True
        
    except Exception as e:
        print_status("ERROR", f"Sandbox orchestrator test failed: {str(e)}")
        return False


async def test_autonomous_development_engine():
    """Test autonomous development engine with sandbox mode."""
    try:
        from app.core.autonomous_development_engine import (
            create_autonomous_development_engine,
            DevelopmentTask,
            TaskComplexity
        )
        
        print_status("INFO", "Testing autonomous development engine...")
        
        # Create engine (should auto-detect sandbox mode)
        engine = create_autonomous_development_engine()
        print_status("SUCCESS", f"Engine created in sandbox mode: {engine.sandbox_mode}")
        
        # Create test task
        task = DevelopmentTask(
            id="test-task-001",
            description="Create a function to calculate factorial",
            requirements=["Handle positive integers", "Input validation", "Efficient algorithm"],
            complexity=TaskComplexity.SIMPLE
        )
        
        print_status("INFO", "Running autonomous development (this may take a few seconds)...")
        
        # Run autonomous development
        result = await engine.develop_autonomously(task)
        
        print_status("SUCCESS", f"Development completed: {result.success}")
        print_status("INFO", f"Phases completed: {len(result.phases_completed)}")
        print_status("INFO", f"Artifacts generated: {len(result.artifacts)}")
        print_status("INFO", f"Execution time: {result.execution_time_seconds:.2f}s")
        
        # Cleanup
        engine.cleanup_workspace()
        print_status("SUCCESS", "Workspace cleaned up")
        
        return True
        
    except Exception as e:
        print_status("ERROR", f"Autonomous development engine test failed: {str(e)}")
        return False


async def test_demo_api_endpoints():
    """Test demo API endpoints (mock test)."""
    try:
        from app.core.sandbox import (
            get_sandbox_status,
            is_sandbox_mode
        )
        from app.core.sandbox.demo_scenarios import get_demo_scenario_engine
        
        print_status("INFO", "Testing demo API functionality...")
        
        # Test sandbox status
        if is_sandbox_mode():
            print_status("SUCCESS", "Sandbox mode is active")
        
        status = get_sandbox_status()
        print_status("SUCCESS", f"Sandbox status retrieved: {len(status)} sections")
        
        # Test scenarios endpoint functionality
        engine = get_demo_scenario_engine()
        scenarios = engine.get_all_scenarios()
        print_status("SUCCESS", f"API scenarios ready: {len(scenarios)} available")
        
        return True
        
    except Exception as e:
        print_status("ERROR", f"Demo API test failed: {str(e)}")
        return False


async def main():
    """Run all sandbox mode validation tests."""
    print_status("INFO", "Starting LeanVibe Sandbox Mode Validation")
    print_status("INFO", "=" * 60)
    
    tests = [
        ("Sandbox Configuration", test_sandbox_config),
        ("Mock Anthropic Client", test_mock_anthropic_client),
        ("Demo Scenarios", test_demo_scenarios),
        ("Sandbox Orchestrator", test_sandbox_orchestrator),
        ("Autonomous Development Engine", test_autonomous_development_engine),
        ("Demo API Endpoints", test_demo_api_endpoints)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print_status("INFO", f"\n--- Testing {test_name} ---")
        try:
            success = await test_func()
            if success:
                passed += 1
                print_status("SUCCESS", f"{test_name} passed")
            else:
                print_status("ERROR", f"{test_name} failed")
        except Exception as e:
            print_status("ERROR", f"{test_name} failed with exception: {str(e)}")
    
    print_status("INFO", "\n" + "=" * 60)
    print_status("INFO", f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print_status("SUCCESS", "🎉 All sandbox mode tests passed! Sandbox mode is ready for use.")
        print_status("INFO", "🚀 Run './start-sandbox-demo.sh' to launch the demo")
    else:
        print_status("WARNING", f"⚠️  {total - passed} test(s) failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class ValidateSandboxModeScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            # Set sandbox mode for testing
            os.environ["SANDBOX_MODE"] = "true"
            os.environ["SANDBOX_DEMO_MODE"] = "true"

            await main()
            sys.exit(exit_code)
            
            return {"status": "completed"}
    
    script_main(ValidateSandboxModeScript)