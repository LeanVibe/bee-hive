#!/usr/bin/env python3
"""
Manual Test Suite - Phase 2 PWA Backend Testing
Validates the implemented system without pytest dependencies.
"""

import asyncio
import sys
import time
from typing import List, Dict, Any


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def record_test(self, name: str, passed: bool, error: str = None):
        self.tests.append({"name": name, "passed": passed, "error": error})
        if passed:
            self.passed += 1
            print(f"✅ {name}")
        else:
            self.failed += 1
            print(f"❌ {name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/total)*100:.1f}%")
        
        if self.failed > 0:
            print(f"\nFAILED TESTS:")
            for test in self.tests:
                if not test["passed"]:
                    print(f"  - {test['name']}: {test['error']}")


def test_core_imports(results: TestResults):
    """Test that all core modules import successfully."""
    print("\n🧪 Testing Core Imports...")
    
    # Core configuration
    try:
        from app.core.configuration_service import ConfigurationService
        config_service = ConfigurationService()
        results.record_test("Core Configuration Import", True)
    except Exception as e:
        results.record_test("Core Configuration Import", False, str(e))
    
    # Simple Orchestrator
    try:
        from app.core.simple_orchestrator import SimpleOrchestrator
        orchestrator = SimpleOrchestrator()
        results.record_test("SimpleOrchestrator Import", True)
    except Exception as e:
        results.record_test("SimpleOrchestrator Import", False, str(e))
    
    # PWA Backend
    try:
        from app.api.pwa_backend import router, agents_router
        results.record_test("PWA Backend Import", True)
    except Exception as e:
        results.record_test("PWA Backend Import", False, str(e))
    
    # Main App
    try:
        from app.main import create_app
        results.record_test("Main App Import", True)
    except Exception as e:
        results.record_test("Main App Import", False, str(e))


def test_pwa_backend_functionality(results: TestResults):
    """Test PWA backend mock data generation and core functionality."""
    print("\n🧪 Testing PWA Backend Functionality...")
    
    # Mock data generation
    try:
        from app.api.pwa_backend import (
            generate_mock_system_metrics, 
            generate_mock_agent_activities,
            generate_mock_project_snapshots
        )
        
        metrics = generate_mock_system_metrics()
        assert metrics.active_agents == 3
        assert metrics.system_status == "healthy"
        results.record_test("Mock System Metrics", True)
        
        activities = generate_mock_agent_activities()
        assert len(activities) == 3
        assert all(activity.name for activity in activities)
        results.record_test("Mock Agent Activities", True)
        
        projects = generate_mock_project_snapshots()
        assert len(projects) == 2
        results.record_test("Mock Project Snapshots", True)
        
    except Exception as e:
        results.record_test("PWA Backend Mock Data", False, str(e))
    
    # Pydantic model validation
    try:
        from app.api.pwa_backend import LiveDataResponse, SystemMetrics, AgentActivity
        
        live_data = LiveDataResponse(
            metrics=SystemMetrics(),
            agent_activities=[],
            project_snapshots=[],
            conflict_snapshots=[]
        )
        assert live_data.metrics.system_status == "healthy"
        results.record_test("PWA Pydantic Models", True)
        
    except Exception as e:
        results.record_test("PWA Pydantic Models", False, str(e))


async def test_orchestrator_functionality(results: TestResults):
    """Test SimpleOrchestrator async functionality."""
    print("\n🧪 Testing SimpleOrchestrator Functionality...")
    
    try:
        from app.core.simple_orchestrator import SimpleOrchestrator
        
        orchestrator = SimpleOrchestrator()
        status = await orchestrator.get_system_status()
        
        assert isinstance(status, dict)
        assert "health" in status
        assert "agents" in status
        assert "timestamp" in status
        results.record_test("Orchestrator System Status", True)
        
    except Exception as e:
        results.record_test("Orchestrator System Status", False, str(e))


async def test_pwa_integration(results: TestResults):
    """Test PWA backend integration with orchestrator."""
    print("\n🧪 Testing PWA-Orchestrator Integration...")
    
    try:
        from app.api.pwa_backend import convert_orchestrator_data_to_pwa
        from app.core.simple_orchestrator import SimpleOrchestrator
        
        orchestrator = SimpleOrchestrator()
        system_data = await orchestrator.get_system_status()
        
        # Convert to PWA format
        pwa_data = await convert_orchestrator_data_to_pwa(system_data)
        
        assert pwa_data.metrics is not None
        assert isinstance(pwa_data.agent_activities, list)
        assert isinstance(pwa_data.project_snapshots, list)
        results.record_test("PWA-Orchestrator Integration", True)
        
    except Exception as e:
        results.record_test("PWA-Orchestrator Integration", False, str(e))


def test_phase_2_implementation(results: TestResults):
    """Test Phase 2 specific implementations.""" 
    print("\n🧪 Testing Phase 2 Implementation...")
    
    # Phase 2.1: Critical PWA endpoints
    try:
        from app.api.pwa_backend import (
            AgentStatusResponse,
            AgentActivationRequest,
            AgentActivationResponse,
            pwa_connection_manager
        )
        
        # Test response models
        status_response = AgentStatusResponse(
            active=True,
            total_agents=3,
            agents=[],
            system_health="healthy"
        )
        assert status_response.active is True
        results.record_test("Phase 2.1 Response Models", True)
        
    except Exception as e:
        results.record_test("Phase 2.1 Response Models", False, str(e))
    
    # Phase 2.2: WebSocket connection manager
    try:
        from app.api.pwa_backend import PWAConnectionManager
        
        manager = PWAConnectionManager()
        assert hasattr(manager, 'active_connections')
        assert hasattr(manager, 'client_subscriptions')
        results.record_test("Phase 2.2 WebSocket Manager", True)
        
    except Exception as e:
        results.record_test("Phase 2.2 WebSocket Manager", False, str(e))


def test_phase_1_consolidation(results: TestResults):
    """Test Phase 1 orchestrator consolidation."""
    print("\n🧪 Testing Phase 1 Consolidation...")
    
    # Test that consolidated orchestrator works
    try:
        from app.core.simple_orchestrator import SimpleOrchestrator, create_simple_orchestrator
        
        orchestrator1 = SimpleOrchestrator()
        orchestrator2 = create_simple_orchestrator()
        
        assert orchestrator1 is not None
        assert orchestrator2 is not None
        results.record_test("Phase 1 Orchestrator Consolidation", True)
        
    except Exception as e:
        results.record_test("Phase 1 Orchestrator Consolidation", False, str(e))


def performance_benchmark(results: TestResults):
    """Basic performance benchmarks."""
    print("\n🧪 Running Performance Benchmarks...")
    
    try:
        from app.api.pwa_backend import generate_mock_agent_activities
        
        # Test mock data generation performance
        start_time = time.time()
        for _ in range(100):
            activities = generate_mock_agent_activities()
        generation_time = (time.time() - start_time) * 1000
        
        # Should be very fast for mock data
        if generation_time < 100:  # Less than 100ms for 100 iterations
            results.record_test("Mock Data Performance", True)
        else:
            results.record_test("Mock Data Performance", False, f"Took {generation_time:.1f}ms")
            
    except Exception as e:
        results.record_test("Mock Data Performance", False, str(e))


async def main():
    """Run all manual tests."""
    print("🚀 Starting Manual Test Suite - Phase 2 PWA Backend Validation")
    print("="*60)
    
    results = TestResults()
    
    # Run all test suites
    test_core_imports(results)
    test_pwa_backend_functionality(results)
    await test_orchestrator_functionality(results)
    await test_pwa_integration(results)
    test_phase_2_implementation(results)
    test_phase_1_consolidation(results)
    performance_benchmark(results)
    
    # Print results
    results.summary()
    
    # Return success/failure
    return results.failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)