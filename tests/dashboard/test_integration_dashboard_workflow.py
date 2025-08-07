"""
Integration Testing Suite for End-to-End Dashboard Functionality

Tests complete workflows from frontend dashboard interactions through backend APIs 
to system state changes. Validates the critical coordination crisis management 
workflow from detection through resolution.

CRITICAL CONTEXT: This integration testing validates that the dashboard can 
successfully help operators identify and resolve the 20% coordination success 
rate crisis through complete end-to-end workflows.

Test Coverage:
1. End-to-End Dashboard Workflow - Complete crisis detection to resolution
2. Frontend-Backend Integration - Dashboard UI to API coordination
3. Real-time Data Flow - WebSocket updates and reactive UI changes
4. Emergency Recovery Workflows - Complete recovery action sequences
5. Cross-Component Communication - Component interaction validation
6. Data Accuracy Validation - Ensure UI displays match backend reality
"""

import asyncio
import json
import pytest
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import websockets
import httpx

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"
WEBSOCKET_URL = "ws://localhost:8000"
INTEGRATION_TIMEOUT = 30  # seconds for integration tests
WORKFLOW_TIMEOUT = 60    # seconds for complete workflow tests


@dataclass
class IntegrationTestResult:
    """Integration test result tracking."""
    workflow: str
    test_name: str
    success: bool
    details: str = ""
    duration_ms: float = 0.0
    critical: bool = False
    components_tested: List[str] = field(default_factory=list)
    data_accuracy: bool = True


@dataclass
class WorkflowTestSuite:
    """Workflow test suite aggregator."""
    name: str
    results: List[IntegrationTestResult] = field(default_factory=list)
    
    def add_result(self, result: IntegrationTestResult):
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
    
    @property
    def average_duration(self) -> float:
        durations = [r.duration_ms for r in self.results if r.duration_ms > 0]
        return sum(durations) / len(durations) if durations else 0.0


class DashboardIntegrationTester:
    """Comprehensive integration tester for dashboard workflows."""
    
    def __init__(self, backend_url: str = BACKEND_URL, frontend_url: str = FRONTEND_URL, websocket_url: str = WEBSOCKET_URL):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.websocket_url = websocket_url
        self.test_suites: Dict[str, WorkflowTestSuite] = {}
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
    
    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    def get_suite(self, suite_name: str) -> WorkflowTestSuite:
        """Get or create test suite."""
        if suite_name not in self.test_suites:
            self.test_suites[suite_name] = WorkflowTestSuite(suite_name)
        return self.test_suites[suite_name]
    
    async def validate_backend_state(self, expected_state: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate backend system state matches expectations."""
        try:
            validation_results = []
            
            # Check coordination success rate
            if "success_rate" in expected_state:
                response = await self.http_client.get(f"{self.backend_url}/api/dashboard/coordination/success-rate")
                if response.status_code == 200:
                    data = response.json()
                    actual_rate = data.get("current_metrics", {}).get("success_rate", 0)
                    expected_rate = expected_state["success_rate"]
                    
                    if abs(actual_rate - expected_rate) <= 5:  # 5% tolerance
                        validation_results.append(f"Success rate: ✓ {actual_rate}%")
                    else:
                        validation_results.append(f"Success rate: ✗ Expected {expected_rate}%, got {actual_rate}%")
                        return False, "; ".join(validation_results)
            
            # Check agent health
            if "agent_health" in expected_state:
                response = await self.http_client.get(f"{self.backend_url}/api/dashboard/agents/status")
                if response.status_code == 200:
                    data = response.json()
                    agents = data.get("agents", [])
                    healthy_count = len([a for a in agents if a.get("health_score", 0) > 70])
                    expected_healthy = expected_state["agent_health"]
                    
                    if healthy_count >= expected_healthy:
                        validation_results.append(f"Agent health: ✓ {healthy_count} healthy")
                    else:
                        validation_results.append(f"Agent health: ✗ Expected {expected_healthy}, got {healthy_count}")
                        return False, "; ".join(validation_results)
            
            # Check system health
            if "system_health" in expected_state:
                response = await self.http_client.get(f"{self.backend_url}/api/dashboard/system/health")
                if response.status_code == 200:
                    data = response.json()
                    health_score = data.get("overall_health", {}).get("score", 0)
                    expected_health = expected_state["system_health"]
                    
                    if health_score >= expected_health:
                        validation_results.append(f"System health: ✓ {health_score}/100")
                    else:
                        validation_results.append(f"System health: ✗ Expected {expected_health}, got {health_score}")
                        return False, "; ".join(validation_results)
            
            return True, "; ".join(validation_results)
            
        except Exception as e:
            return False, f"Backend validation error: {str(e)}"
    
    async def validate_frontend_state(self, page: Page, expected_ui_state: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate frontend UI state matches expectations."""
        try:
            validation_results = []
            
            # Check success rate display
            if "success_rate_displayed" in expected_ui_state:
                success_rate_text = await page.evaluate("""() => {
                    const elements = document.querySelectorAll('[data-testid*="success-rate"], .success-rate, .coordination-success');
                    for (const el of elements) {
                        if (el.textContent.includes('%')) {
                            return el.textContent;
                        }
                    }
                    return null;
                }""")
                
                if success_rate_text and "%" in success_rate_text:
                    validation_results.append(f"Success rate UI: ✓ {success_rate_text}")
                else:
                    validation_results.append("Success rate UI: ✗ Not displayed")
                    return False, "; ".join(validation_results)
            
            # Check emergency controls visibility
            if "emergency_controls_visible" in expected_ui_state:
                controls_visible = await page.evaluate("""() => {
                    const controls = document.querySelectorAll('[data-testid*="emergency"], .emergency-controls, [data-testid*="reset"]');
                    return Array.from(controls).some(el => el.offsetHeight > 0 && el.offsetWidth > 0);
                }""")
                
                expected_visible = expected_ui_state["emergency_controls_visible"]
                if controls_visible == expected_visible:
                    validation_results.append(f"Emergency controls: ✓ {'Visible' if controls_visible else 'Hidden'}")
                else:
                    validation_results.append(f"Emergency controls: ✗ Expected {'visible' if expected_visible else 'hidden'}")
                    return False, "; ".join(validation_results)
            
            # Check agent status display
            if "agents_displayed" in expected_ui_state:
                agent_count = await page.evaluate("""() => {
                    const agents = document.querySelectorAll('.agent-card, [data-testid*="agent"], .agent-status');
                    return agents.length;
                }""")
                
                expected_count = expected_ui_state["agents_displayed"]
                if agent_count >= expected_count:
                    validation_results.append(f"Agent display: ✓ {agent_count} agents shown")
                else:
                    validation_results.append(f"Agent display: ✗ Expected {expected_count}, got {agent_count}")
                    return False, "; ".join(validation_results)
            
            return True, "; ".join(validation_results)
            
        except Exception as e:
            return False, f"Frontend validation error: {str(e)}"
    
    # ===== CORE INTEGRATION WORKFLOWS =====
    
    async def test_coordination_crisis_detection_workflow(self):
        """Test complete crisis detection workflow from backend data to UI display."""
        suite = self.get_suite("Crisis Detection Workflow")
        
        start_time = time.perf_counter()
        
        try:
            # Step 1: Get initial backend state
            response = await self.http_client.get(f"{self.backend_url}/api/dashboard/coordination/success-rate")
            if response.status_code != 200:
                raise Exception("Backend not available")
            
            initial_data = response.json()
            initial_success_rate = initial_data.get("current_metrics", {}).get("success_rate", 100)
            
            # Step 2: Load dashboard UI
            page = await self.context.new_page()
            await page.goto(self.frontend_url, wait_until="networkidle")
            
            # Step 3: Wait for real-time data to load
            await asyncio.sleep(3)
            
            # Step 4: Validate backend-frontend data consistency
            backend_valid, backend_details = await self.validate_backend_state({
                "success_rate": initial_success_rate
            })
            
            frontend_valid, frontend_details = await self.validate_frontend_state(page, {
                "success_rate_displayed": True,
                "emergency_controls_visible": initial_success_rate < 50  # Show emergency controls if critical
            })
            
            # Step 5: Test crisis threshold detection
            crisis_detected = initial_success_rate < 50
            
            if crisis_detected:
                # Validate crisis UI elements
                crisis_ui_visible = await page.evaluate("""() => {
                    const crisis_elements = document.querySelectorAll('[data-testid*="crisis"], .crisis-alert, .critical-warning');
                    return Array.from(crisis_elements).some(el => el.offsetHeight > 0);
                }""")
                
                if not crisis_ui_visible:
                    frontend_valid = False
                    frontend_details += "; Crisis UI not shown for low success rate"
            
            success = backend_valid and frontend_valid
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            details = f"Backend: {backend_details}; Frontend: {frontend_details}; Crisis: {'Detected' if crisis_detected else 'None'}"
            
            await page.close()
            
        except Exception as e:
            success = False
            duration_ms = (time.perf_counter() - start_time) * 1000
            details = f"Workflow failed: {str(e)}"
        
        result = IntegrationTestResult(
            workflow="Crisis Detection",
            test_name="Backend-Frontend Coordination Detection",
            success=success,
            details=details,
            duration_ms=duration_ms,
            critical=True,
            components_tested=["Backend API", "Frontend UI", "Real-time Updates"]
        )
        suite.add_result(result)
    
    async def test_real_time_websocket_integration(self):
        """Test real-time WebSocket integration between backend and frontend."""
        suite = self.get_suite("Real-time Integration")
        
        start_time = time.perf_counter()
        
        try:
            # Step 1: Establish WebSocket connection
            connection_id = str(uuid.uuid4())
            websocket_url = f"{self.websocket_url}/api/dashboard/ws/coordination?connection_id={connection_id}"
            
            # Step 2: Load dashboard UI  
            page = await self.context.new_page()
            await page.goto(self.frontend_url, wait_until="networkidle")
            
            # Step 3: Test WebSocket communication
            async with websockets.connect(websocket_url, timeout=10) as websocket:
                # Send test message
                test_message = {
                    "type": "ping",
                    "timestamp": datetime.utcnow().isoformat(),
                    "connection_id": connection_id
                }
                
                await websocket.send(json.dumps(test_message))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                response_data = json.loads(response)
                
                websocket_communication_success = response_data.get("type") in ["pong", "ping_response"]
                
                # Step 4: Test real-time data updates
                # Monitor for any real-time updates in the UI
                initial_page_state = await page.evaluate("""() => {
                    return document.body.innerHTML.length;
                }""")
                
                await asyncio.sleep(2)  # Wait for potential updates
                
                final_page_state = await page.evaluate("""() => {
                    return document.body.innerHTML.length;
                }""")
                
                # Step 5: Check for WebSocket status in UI
                websocket_status_shown = await page.evaluate("""() => {
                    const status_elements = document.querySelectorAll('[data-testid*="websocket"], .connection-status, .realtime-status');
                    return Array.from(status_elements).some(el => el.offsetHeight > 0);
                }""")
                
                success = websocket_communication_success
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                details = f"WebSocket: {'✓' if websocket_communication_success else '✗'}"
                details += f", UI Status: {'✓' if websocket_status_shown else '✗'}"
                details += f", Page updates: {abs(final_page_state - initial_page_state)}"
            
            await page.close()
            
        except Exception as e:
            success = False
            duration_ms = (time.perf_counter() - start_time) * 1000
            details = f"WebSocket integration failed: {str(e)}"
        
        result = IntegrationTestResult(
            workflow="Real-time Updates",
            test_name="WebSocket Backend-Frontend Integration",
            success=success,
            details=details,
            duration_ms=duration_ms,
            critical=True,
            components_tested=["WebSocket Server", "Frontend WebSocket Client", "Real-time UI Updates"]
        )
        suite.add_result(result)
    
    async def test_emergency_recovery_workflow(self):
        """Test complete emergency recovery workflow from UI action to backend execution."""
        suite = self.get_suite("Emergency Recovery Workflow")
        
        recovery_workflows = [
            {
                "name": "Coordination Reset",
                "ui_trigger": "[data-testid='system-reset-btn'], .reset-coordination-btn",
                "backend_endpoint": "/api/dashboard/coordination/reset",
                "expected_confirmation": True,
                "critical": True
            },
            {
                "name": "Agent Restart",
                "ui_trigger": "[data-testid='agent-restart-btn'], .restart-agents-btn", 
                "backend_endpoint": "/api/dashboard/recovery/auto-heal",
                "expected_confirmation": True,
                "critical": True
            },
            {
                "name": "Task Reassignment",
                "ui_trigger": "[data-testid='reassign-tasks-btn'], .reassign-task-btn",
                "backend_endpoint": "/api/dashboard/tasks/reassign",
                "expected_confirmation": False,
                "critical": False
            }
        ]
        
        for workflow in recovery_workflows:
            start_time = time.perf_counter()
            
            try:
                # Step 1: Load dashboard UI
                page = await self.context.new_page()
                await page.goto(self.frontend_url, wait_until="networkidle")
                
                # Step 2: Find and test UI trigger
                trigger_exists = await page.locator(workflow["ui_trigger"]).count() > 0
                
                if not trigger_exists:
                    success = False
                    details = "UI trigger not found"
                else:
                    # Step 3: Click trigger
                    await page.locator(workflow["ui_trigger"]).click()
                    
                    # Step 4: Handle confirmation if expected
                    if workflow["expected_confirmation"]:
                        # Check for confirmation dialog
                        confirmation_visible = await page.is_visible(".confirmation-dialog, .modal, [role='dialog']")
                        
                        if confirmation_visible:
                            # Cancel to avoid actual execution
                            cancel_btn = page.locator("[data-testid='cancel-btn'], .cancel-btn, button:has-text('Cancel')")
                            if await cancel_btn.count() > 0:
                                await cancel_btn.click()
                            
                            success = True
                            details = "UI workflow complete with confirmation (safe)"
                        else:
                            success = False
                            details = "No confirmation dialog (unsafe)"
                    else:
                        # For non-confirmation workflows, just check that UI responds
                        await asyncio.sleep(1)
                        success = True
                        details = "UI workflow triggered successfully"
                    
                    # Step 5: Verify backend endpoint is available (dry run)
                    try:
                        if workflow["name"] == "Coordination Reset":
                            backend_response = await self.http_client.post(
                                f"{self.backend_url}{workflow['backend_endpoint']}",
                                params={"reset_type": "soft", "confirm": False}
                            )
                        elif workflow["name"] == "Agent Restart":
                            backend_response = await self.http_client.post(
                                f"{self.backend_url}{workflow['backend_endpoint']}",
                                params={"recovery_type": "smart", "dry_run": True}
                            )
                        else:
                            # For other workflows, just check endpoint exists
                            backend_response = await self.http_client.get(
                                f"{self.backend_url}/api/dashboard/tasks/queue"
                            )
                        
                        backend_available = backend_response.status_code == 200
                        details += f", Backend: {'✓' if backend_available else '✗'}"
                        
                    except Exception as be:
                        details += f", Backend error: {str(be)[:30]}"
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                await page.close()
                
            except Exception as e:
                success = False
                duration_ms = (time.perf_counter() - start_time) * 1000
                details = f"Recovery workflow failed: {str(e)}"
            
            result = IntegrationTestResult(
                workflow=workflow["name"],
                test_name="Emergency Recovery Workflow",
                success=success,
                details=details,
                duration_ms=duration_ms,
                critical=workflow["critical"],
                components_tested=["Frontend UI", "User Interactions", "Backend API", "Confirmation System"]
            )
            suite.add_result(result)
    
    async def test_cross_component_communication(self):
        """Test communication between different dashboard components."""
        suite = self.get_suite("Cross-Component Communication")
        
        start_time = time.perf_counter()
        
        try:
            # Step 1: Load dashboard UI
            page = await self.context.new_page()
            await page.goto(self.frontend_url, wait_until="networkidle")
            
            # Step 2: Test component interaction scenarios
            interaction_tests = []
            
            # Test: Agent selection updates task panel
            agent_cards = await page.locator(".agent-card, [data-testid*='agent']").count()
            if agent_cards > 0:
                await page.locator(".agent-card, [data-testid*='agent']").first.click()
                await asyncio.sleep(1)
                
                # Check if task panel shows agent-specific tasks
                task_panel_updated = await page.evaluate("""() => {
                    const taskPanel = document.querySelector('.task-distribution-panel, [data-testid*="task"]');
                    return taskPanel && taskPanel.textContent.length > 50;
                }""")
                
                interaction_tests.append({
                    "name": "Agent Selection → Task Panel",
                    "success": task_panel_updated,
                    "details": "Task panel updated" if task_panel_updated else "No task panel update"
                })
            
            # Test: Success rate change affects emergency controls
            current_success_rate = await page.evaluate(r"""() => {
                const successElements = document.querySelectorAll('[data-testid*="success-rate"], .success-rate');
                for (const el of successElements) {
                    const match = el.textContent.match(/(\d+)%/);
                    if (match) return parseInt(match[1]);
                }
                return 100;
            }""")
            
            emergency_controls_visible = await page.evaluate("""() => {
                const controls = document.querySelectorAll('[data-testid*="emergency"], .emergency-controls');
                return Array.from(controls).some(el => el.offsetHeight > 0);
            }""")
            
            expected_emergency_visible = current_success_rate < 50
            emergency_logic_correct = emergency_controls_visible == expected_emergency_visible
            
            interaction_tests.append({
                "name": "Success Rate → Emergency Controls",
                "success": emergency_logic_correct,
                "details": f"Success rate: {current_success_rate}%, Emergency visible: {emergency_controls_visible}"
            })
            
            # Test: System health affects overall dashboard theme/status
            system_health_indicated = await page.evaluate("""() => {
                const healthElements = document.querySelectorAll('[data-testid*="health"], .system-health, .health-indicator');
                return Array.from(healthElements).some(el => {
                    const text = el.textContent.toLowerCase();
                    return text.includes('healthy') || text.includes('critical') || text.includes('warning');
                });
            }""")
            
            interaction_tests.append({
                "name": "System Health → Dashboard Status",
                "success": system_health_indicated,
                "details": "Health status indicated" if system_health_indicated else "No health indication"
            })
            
            # Aggregate results
            successful_interactions = len([test for test in interaction_tests if test["success"]])
            total_interactions = len(interaction_tests)
            
            success = successful_interactions >= total_interactions * 0.7  # 70% success rate
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            details = f"Interactions: {successful_interactions}/{total_interactions}"
            for test in interaction_tests:
                status = "✓" if test["success"] else "✗"
                details += f"; {test['name']}: {status}"
            
            await page.close()
            
        except Exception as e:
            success = False
            duration_ms = (time.perf_counter() - start_time) * 1000
            details = f"Cross-component test failed: {str(e)}"
        
        result = IntegrationTestResult(
            workflow="Component Communication",
            test_name="Cross-Component Interaction",
            success=success,
            details=details,
            duration_ms=duration_ms,
            critical=False,
            components_tested=["Agent Panel", "Task Panel", "Emergency Controls", "System Health Display"]
        )
        suite.add_result(result)
    
    async def test_data_accuracy_validation(self):
        """Test that UI displays match backend data reality."""
        suite = self.get_suite("Data Accuracy Validation")
        
        start_time = time.perf_counter()
        
        try:
            # Step 1: Get backend ground truth data
            backend_data = {}
            
            # Get coordination data
            response = await self.http_client.get(f"{self.backend_url}/api/dashboard/coordination/success-rate")
            if response.status_code == 200:
                coord_data = response.json()
                backend_data["success_rate"] = coord_data.get("current_metrics", {}).get("success_rate", 0)
            
            # Get agent data
            response = await self.http_client.get(f"{self.backend_url}/api/dashboard/agents/status")
            if response.status_code == 200:
                agent_data = response.json()
                backend_data["agents"] = len(agent_data.get("agents", []))
                backend_data["healthy_agents"] = len([a for a in agent_data.get("agents", []) if a.get("health_score", 0) > 70])
            
            # Get task data
            response = await self.http_client.get(f"{self.backend_url}/api/dashboard/tasks/queue")
            if response.status_code == 200:
                task_data = response.json()
                backend_data["active_tasks"] = task_data.get("distribution_metrics", {}).get("total_active_tasks", 0)
            
            # Step 2: Load dashboard UI and extract displayed data
            page = await self.context.new_page()
            await page.goto(self.frontend_url, wait_until="networkidle")
            await asyncio.sleep(3)  # Wait for data to load
            
            # Extract UI data
            ui_data = await page.evaluate(r"""() => {
                const data = {};
                
                // Extract success rate
                const successElements = document.querySelectorAll('[data-testid*="success-rate"], .success-rate, .coordination-success');
                for (const el of successElements) {
                    const match = el.textContent.match(/(\d+(?:\.\d+)?)%/);
                    if (match) {
                        data.success_rate = parseFloat(match[1]);
                        break;
                    }
                }
                
                // Extract agent count
                const agentElements = document.querySelectorAll('[data-testid*="agent-count"], .agent-count, .agent-total');
                for (const el of agentElements) {
                    const match = el.textContent.match(/(\\d+)/);
                    if (match) {
                        data.agents = parseInt(match[1]);
                        break;
                    }
                }
                
                // Count visible agent cards as fallback
                if (!data.agents) {
                    data.agents = document.querySelectorAll('.agent-card, [data-testid*="agent-card"]').length;
                }
                
                // Extract task count
                const taskElements = document.querySelectorAll('[data-testid*="task-count"], .task-count, .active-tasks');
                for (const el of taskElements) {
                    const match = el.textContent.match(/(\\d+)/);
                    if (match) {
                        data.active_tasks = parseInt(match[1]);
                        break;
                    }
                }
                
                return data;
            }""")
            
            # Step 3: Compare backend vs UI data
            accuracy_tests = []
            
            # Success rate accuracy
            if "success_rate" in backend_data and "success_rate" in ui_data:
                diff = abs(backend_data["success_rate"] - ui_data["success_rate"])
                accurate = diff <= 2.0  # 2% tolerance
                accuracy_tests.append({
                    "metric": "Success Rate",
                    "accurate": accurate,
                    "backend": backend_data["success_rate"],
                    "ui": ui_data["success_rate"],
                    "diff": diff
                })
            
            # Agent count accuracy
            if "agents" in backend_data and "agents" in ui_data:
                diff = abs(backend_data["agents"] - ui_data["agents"])
                accurate = diff <= 1  # 1 agent tolerance
                accuracy_tests.append({
                    "metric": "Agent Count",
                    "accurate": accurate,
                    "backend": backend_data["agents"],
                    "ui": ui_data["agents"],
                    "diff": diff
                })
            
            # Task count accuracy  
            if "active_tasks" in backend_data and "active_tasks" in ui_data:
                diff = abs(backend_data["active_tasks"] - ui_data["active_tasks"])
                accurate = diff <= 2  # 2 task tolerance
                accuracy_tests.append({
                    "metric": "Task Count",
                    "accurate": accurate,
                    "backend": backend_data["active_tasks"],
                    "ui": ui_data["active_tasks"],
                    "diff": diff
                })
            
            # Aggregate accuracy results
            accurate_metrics = len([test for test in accuracy_tests if test["accurate"]])
            total_metrics = len(accuracy_tests)
            
            success = accurate_metrics >= total_metrics * 0.8  # 80% accuracy required
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            details = f"Accurate metrics: {accurate_metrics}/{total_metrics}"
            for test in accuracy_tests:
                status = "✓" if test["accurate"] else "✗"
                details += f"; {test['metric']}: {status} (Backend: {test['backend']}, UI: {test['ui']})"
            
            await page.close()
            
        except Exception as e:
            success = False
            duration_ms = (time.perf_counter() - start_time) * 1000
            details = f"Data accuracy test failed: {str(e)}"
        
        result = IntegrationTestResult(
            workflow="Data Accuracy",
            test_name="Backend-UI Data Consistency",
            success=success,
            details=details,
            duration_ms=duration_ms,
            critical=True,
            components_tested=["Backend APIs", "Frontend Data Display", "Real-time Updates"],
            data_accuracy=success
        )
        suite.add_result(result)
    
    # ===== COMPREHENSIVE TEST EXECUTION =====
    
    async def run_all_integration_tests(self) -> Dict[str, WorkflowTestSuite]:
        """Run all comprehensive integration tests."""
        print("🔗 Starting Integration Dashboard Testing")
        print("="*50)
        print(f"Timeout: {INTEGRATION_TIMEOUT}s per test, {WORKFLOW_TIMEOUT}s per workflow")
        print("="*50)
        
        # Test 1: Crisis Detection Workflow
        print("\n🚨 Testing Coordination Crisis Detection Workflow...")
        await self.test_coordination_crisis_detection_workflow()
        
        # Test 2: Real-time WebSocket Integration
        print("\n🔌 Testing Real-time WebSocket Integration...")
        await self.test_real_time_websocket_integration()
        
        # Test 3: Emergency Recovery Workflows
        print("\n🆘 Testing Emergency Recovery Workflows...")
        await self.test_emergency_recovery_workflow()
        
        # Test 4: Cross-Component Communication
        print("\n🔄 Testing Cross-Component Communication...")
        await self.test_cross_component_communication()
        
        # Test 5: Data Accuracy Validation
        print("\n📊 Testing Data Accuracy Validation...")
        await self.test_data_accuracy_validation()
        
        return self.test_suites
    
    def print_integration_test_report(self):
        """Print comprehensive integration test report."""
        print("\n" + "="*60)
        print("INTEGRATION DASHBOARD TEST REPORT")
        print("="*60)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_critical_failures = 0
        
        for suite_name, suite in self.test_suites.items():
            print(f"\n🔗 {suite.name}")
            print(f"   Tests: {len(suite.results)} | Passed: {suite.passed} | Failed: {suite.failed}")
            print(f"   Success Rate: {suite.success_rate:.1f}% | Avg Duration: {suite.average_duration:.1f}ms")
            
            if suite.critical_failures > 0:
                print(f"   🔴 CRITICAL FAILURES: {suite.critical_failures}")
            
            # Show component coverage
            all_components = set()
            for result in suite.results:
                all_components.update(result.components_tested)
            if all_components:
                print(f"   Components: {', '.join(all_components)}")
            
            # Show failed tests
            failed_tests = [r for r in suite.results if not r.success]
            if failed_tests:
                print("   Failed Tests:")
                for test in failed_tests[:3]:  # Show first 3 failures
                    prefix = "🔴" if test.critical else "❌"
                    print(f"     {prefix} {test.workflow}: {test.details[:60]}...")
                if len(failed_tests) > 3:
                    print(f"     ... and {len(failed_tests) - 3} more")
            
            total_tests += len(suite.results)
            total_passed += suite.passed
            total_failed += suite.failed
            total_critical_failures += suite.critical_failures
        
        # Overall summary
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n{'='*60}")
        print("INTEGRATION SYSTEM STATUS")
        print(f"{'='*60}")
        print(f"Total Integration Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {overall_success_rate:.1f}%")
        print(f"Critical Failures: {total_critical_failures}")
        
        # Production readiness assessment
        print(f"\n🎯 INTEGRATION PRODUCTION READINESS:")
        
        if total_critical_failures == 0 and overall_success_rate >= 90:
            print("🟢 READY FOR PRODUCTION - All critical workflows operational")
            status = "READY"
        elif total_critical_failures <= 1 and overall_success_rate >= 80:
            print("🟡 READY WITH MINOR ISSUES - Most workflows operational")
            status = "READY_WITH_ISSUES"
        elif total_critical_failures <= 2 and overall_success_rate >= 70:
            print("🟠 NEEDS ATTENTION - Some critical workflow issues present")
            status = "NEEDS_ATTENTION"
        else:
            print("🔴 NOT READY - Critical integration failures present")
            status = "NOT_READY"
        
        print(f"\n📊 COORDINATION CRISIS WORKFLOW STATUS:")
        
        # Check crisis workflow specific results
        crisis_suite = self.test_suites.get("Crisis Detection Workflow")
        emergency_suite = self.test_suites.get("Emergency Recovery Workflow")
        data_suite = self.test_suites.get("Data Accuracy Validation")
        
        crisis_workflow_ready = True
        if crisis_suite and crisis_suite.critical_failures > 0:
            crisis_workflow_ready = False
        if emergency_suite and emergency_suite.critical_failures > 0:
            crisis_workflow_ready = False
        if data_suite and data_suite.critical_failures > 0:
            crisis_workflow_ready = False
        
        if crisis_workflow_ready:
            print("✅ Complete workflow ready to resolve 20% success rate crisis")
        else:
            print("❌ Workflow cannot adequately support coordination crisis resolution")
        
        print(f"{'='*60}")
        
        return {
            "overall_success_rate": overall_success_rate,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "critical_failures": total_critical_failures,
            "production_status": status,
            "crisis_workflow_ready": crisis_workflow_ready,
            "suites": {name: {
                "success_rate": suite.success_rate,
                "tests": len(suite.results),
                "passed": suite.passed,
                "failed": suite.failed,
                "critical_failures": suite.critical_failures,
                "average_duration": suite.average_duration
            } for name, suite in self.test_suites.items()}
        }


# ===== TEST EXECUTION FUNCTIONS =====

async def run_integration_workflow_validation():
    """Run comprehensive integration workflow validation."""
    async with DashboardIntegrationTester() as tester:
        await tester.run_all_integration_tests()
        return tester.print_integration_test_report()


async def run_crisis_workflow_only():
    """Run only crisis detection and recovery workflows."""
    print("🚨 Running Crisis Workflow Tests Only")
    print("="*40)
    
    async with DashboardIntegrationTester() as tester:
        await tester.test_coordination_crisis_detection_workflow()
        await tester.test_emergency_recovery_workflow()
        await tester.test_data_accuracy_validation()
        return tester.print_integration_test_report()


async def run_realtime_integration_only():
    """Run only real-time integration tests."""
    print("🔄 Running Real-time Integration Tests")
    print("="*35)
    
    async with DashboardIntegrationTester() as tester:
        await tester.test_real_time_websocket_integration()
        await tester.test_cross_component_communication()
        return tester.print_integration_test_report()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "crisis":
            result = asyncio.run(run_crisis_workflow_only())
        elif test_type == "realtime":
            result = asyncio.run(run_realtime_integration_only())
        else:
            print(f"Unknown test type: {test_type}")
            print("Available types: crisis, realtime")
            sys.exit(1)
    else:
        # Run comprehensive validation
        result = asyncio.run(run_integration_workflow_validation())
    
    # Set exit code based on results
    if result and result.get("critical_failures", 1) == 0 and result.get("overall_success_rate", 0) >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure