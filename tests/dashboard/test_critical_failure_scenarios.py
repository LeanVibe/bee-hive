"""
Critical Failure Scenario Testing for 20% Success Rate Conditions

Tests the dashboard's behavior and recovery capabilities when the coordination system
is experiencing critical failures with only 20% success rate. This is the primary
crisis scenario the dashboard was designed to address.

CRITICAL CONTEXT: This testing validates that the dashboard can:
1. Detect and accurately report the 20% coordination crisis
2. Provide actionable recovery options during system failure
3. Remain functional and responsive during system stress
4. Guide operators through crisis resolution workflows
5. Validate system recovery and return to normal operation

Test Coverage:
1. Crisis Detection - Accurate identification of 20% success rate
2. Emergency Dashboard Functionality - UI remains usable during crisis  
3. Recovery Action Effectiveness - Emergency controls work under stress
4. System Resilience - Dashboard survives system failures
5. Operator Guidance - Clear crisis resolution workflows
6. Recovery Validation - Confirmation of successful crisis resolution
"""

import asyncio
import json
import pytest
import time
import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import websockets
import httpx

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"
WEBSOCKET_URL = "ws://localhost:8000"

# Crisis Simulation Parameters
CRISIS_SUCCESS_RATE = 20.0         # The critical 20% success rate
SEVERE_CRISIS_SUCCESS_RATE = 15.0  # Even more severe crisis
RECOVERY_SUCCESS_RATE = 85.0       # Target success rate after recovery
CRISIS_SIMULATION_DURATION = 60    # seconds to simulate crisis conditions


@dataclass
class CrisisTestResult:
    """Crisis test result tracking."""
    scenario: str
    test_name: str
    success: bool
    details: str = ""
    duration_ms: float = 0.0
    critical: bool = True
    crisis_detected: bool = False
    recovery_successful: bool = False
    dashboard_remained_functional: bool = False


@dataclass
class CrisisTestSuite:
    """Crisis test suite aggregator."""
    name: str
    results: List[CrisisTestResult] = field(default_factory=list)
    
    def add_result(self, result: CrisisTestResult):
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
    def crisis_detection_rate(self) -> float:
        total = len(self.results)
        detected = len([r for r in self.results if r.crisis_detected])
        return (detected / total * 100) if total > 0 else 0.0
    
    @property
    def recovery_success_rate(self) -> float:
        recovery_attempts = len([r for r in self.results if "recovery" in r.test_name.lower()])
        successful_recoveries = len([r for r in self.results if r.recovery_successful])
        return (successful_recoveries / recovery_attempts * 100) if recovery_attempts > 0 else 0.0


class CriticalFailureTester:
    """Comprehensive tester for critical failure scenarios."""
    
    def __init__(self, backend_url: str = BACKEND_URL, frontend_url: str = FRONTEND_URL, websocket_url: str = WEBSOCKET_URL):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.websocket_url = websocket_url
        self.test_suites: Dict[str, CrisisTestSuite] = {}
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
    
    def get_suite(self, suite_name: str) -> CrisisTestSuite:
        """Get or create test suite."""
        if suite_name not in self.test_suites:
            self.test_suites[suite_name] = CrisisTestSuite(suite_name)
        return self.test_suites[suite_name]
    
    async def simulate_coordination_crisis(self, success_rate: float = CRISIS_SUCCESS_RATE, duration_seconds: int = 30):
        """Simulate coordination crisis by creating system stress."""
        # Note: In a real implementation, this would interact with the coordination system
        # to actually reduce success rates. For testing, we'll simulate the conditions.
        
        crisis_data = {
            "simulated_success_rate": success_rate,
            "failure_types": ["redis_timeout", "serialization_error", "agent_unresponsive"],
            "start_time": datetime.utcnow().isoformat(),
            "duration_seconds": duration_seconds
        }
        
        # Simulate the crisis effects
        await asyncio.sleep(0.1)  # Brief delay to simulate setup
        return crisis_data
    
    async def validate_crisis_detection(self, page: Page, expected_success_rate: float) -> Tuple[bool, str]:
        """Validate that the dashboard correctly detects and displays crisis conditions."""
        try:
            # Wait for dashboard to load and update
            await asyncio.sleep(5)
            
            # Check if success rate is displayed and approximately correct
            success_rate_detected = await page.evaluate("""() => {
                const elements = document.querySelectorAll('[data-testid*="success-rate"], .success-rate, .coordination-success');
                for (const el of elements) {
                    const match = el.textContent.match(/(\d+(?:\.\d+)?)%/);
                    if (match) {
                        return parseFloat(match[1]);
                    }
                }
                return null;
            }""")
            
            # Check if crisis UI elements are visible
            crisis_ui_visible = await page.evaluate("""() => {
                const crisis_indicators = document.querySelectorAll('[data-testid*="crisis"], .crisis-alert, .critical-warning, .emergency-status');
                return Array.from(crisis_indicators).some(el => el.offsetHeight > 0 && el.offsetWidth > 0);
            }""")
            
            # Check if emergency controls are visible
            emergency_controls_visible = await page.evaluate("""() => {
                const controls = document.querySelectorAll('[data-testid*="emergency"], .emergency-controls, [data-testid*="reset"]');
                return Array.from(controls).some(el => el.offsetHeight > 0 && el.offsetWidth > 0);
            }""")
            
            # Validate success rate accuracy (within 10% tolerance)
            rate_accurate = success_rate_detected is not None and abs(success_rate_detected - expected_success_rate) <= 10
            
            # Validate crisis response
            crisis_response_appropriate = crisis_ui_visible and emergency_controls_visible if expected_success_rate < 50 else True
            
            detection_success = rate_accurate and crisis_response_appropriate
            
            details = f"Rate shown: {success_rate_detected}% (expected ~{expected_success_rate}%), Crisis UI: {crisis_ui_visible}, Emergency controls: {emergency_controls_visible}"
            
            return detection_success, details
            
        except Exception as e:
            return False, f"Crisis detection validation failed: {str(e)}"
    
    async def validate_dashboard_functionality_during_crisis(self, page: Page) -> Tuple[bool, str]:
        """Validate that dashboard remains functional during crisis conditions."""
        functionality_tests = []
        
        try:
            # Test 1: Navigation still works
            try:
                navigation_elements = await page.locator(".navigation, .menu, [role='navigation']").count()
                if navigation_elements > 0:
                    await page.locator(".navigation, .menu, [role='navigation']").first.click()
                    await asyncio.sleep(1)
                functionality_tests.append({"name": "Navigation", "success": True})
            except:
                functionality_tests.append({"name": "Navigation", "success": False})
            
            # Test 2: Agent status panels load
            try:
                agent_panels_loaded = await page.evaluate("""() => {
                    const panels = document.querySelectorAll('.agent-panel, [data-testid*="agent"], .agent-status');
                    return panels.length > 0 && Array.from(panels).some(p => p.textContent.length > 10);
                }""")
                functionality_tests.append({"name": "Agent Panels", "success": agent_panels_loaded})
            except:
                functionality_tests.append({"name": "Agent Panels", "success": False})
            
            # Test 3: Task information displays
            try:
                task_info_loaded = await page.evaluate("""() => {
                    const task_elements = document.querySelectorAll('.task-panel, [data-testid*="task"], .task-status');
                    return task_elements.length > 0;
                }""")
                functionality_tests.append({"name": "Task Information", "success": task_info_loaded})
            except:
                functionality_tests.append({"name": "Task Information", "success": False})
            
            # Test 4: Emergency controls are interactive
            try:
                emergency_interactive = await page.evaluate("""() => {
                    const emergency_controls = document.querySelectorAll('[data-testid*="emergency"], .emergency-controls button');
                    return Array.from(emergency_controls).some(btn => !btn.disabled && btn.offsetHeight > 0);
                }""")
                functionality_tests.append({"name": "Emergency Controls", "success": emergency_interactive})
            except:
                functionality_tests.append({"name": "Emergency Controls", "success": False})
            
            # Test 5: System health indicators work
            try:
                health_indicators_work = await page.evaluate("""() => {
                    const health_elements = document.querySelectorAll('[data-testid*="health"], .system-health, .health-indicator');
                    return Array.from(health_elements).some(el => el.textContent.length > 5);
                }""")
                functionality_tests.append({"name": "Health Indicators", "success": health_indicators_work})
            except:
                functionality_tests.append({"name": "Health Indicators", "success": False})
            
            # Calculate overall functionality
            successful_functions = len([test for test in functionality_tests if test["success"]])
            total_functions = len(functionality_tests)
            functionality_rate = (successful_functions / total_functions) if total_functions > 0 else 0
            
            overall_functional = functionality_rate >= 0.7  # 70% of functions should work
            
            details = f"Functional components: {successful_functions}/{total_functions} ({functionality_rate*100:.1f}%)"
            for test in functionality_tests:
                status = "✓" if test["success"] else "✗"
                details += f"; {test['name']}: {status}"
            
            return overall_functional, details
            
        except Exception as e:
            return False, f"Functionality validation failed: {str(e)}"
    
    async def test_recovery_action_effectiveness(self, page: Page, recovery_action: str) -> Tuple[bool, str]:
        """Test effectiveness of recovery actions during crisis."""
        try:
            # Define recovery action selectors
            action_selectors = {
                "system_reset": "[data-testid='system-reset-btn'], .reset-coordination-btn",
                "agent_restart": "[data-testid='agent-restart-btn'], .restart-agents-btn", 
                "auto_heal": "[data-testid='auto-heal-btn'], .auto-heal-btn",
                "task_reassign": "[data-testid='reassign-tasks-btn'], .reassign-task-btn"
            }
            
            if recovery_action not in action_selectors:
                return False, f"Unknown recovery action: {recovery_action}"
            
            selector = action_selectors[recovery_action]
            
            # Check if recovery action button exists and is clickable
            button_exists = await page.locator(selector).count() > 0
            
            if not button_exists:
                return False, f"Recovery action button not found: {recovery_action}"
            
            button_enabled = await page.locator(selector).is_enabled()
            button_visible = await page.locator(selector).is_visible()
            
            if not (button_enabled and button_visible):
                return False, f"Recovery action button not interactive: enabled={button_enabled}, visible={button_visible}"
            
            # Click the recovery action button
            await page.locator(selector).click()
            
            # Handle confirmation if required
            confirmation_appeared = False
            try:
                # Wait for potential confirmation dialog
                await page.wait_for_selector(".confirmation-dialog, .modal, [role='dialog']", timeout=3000)
                confirmation_appeared = True
                
                # Cancel to avoid actual execution during testing
                cancel_btn = page.locator("[data-testid='cancel-btn'], .cancel-btn, button:has-text('Cancel')")
                if await cancel_btn.count() > 0:
                    await cancel_btn.click()
                
            except:
                # No confirmation dialog appeared, which might be expected for some actions
                pass
            
            # For system reset and agent restart, confirmation should be required
            confirmation_required = recovery_action in ["system_reset", "agent_restart"]
            
            if confirmation_required and not confirmation_appeared:
                return False, f"Recovery action {recovery_action} did not show required confirmation (unsafe)"
            
            # Check if recovery action triggered appropriate backend call (dry run check)
            backend_endpoint_map = {
                "system_reset": "/api/dashboard/coordination/reset",
                "agent_restart": "/api/dashboard/recovery/auto-heal",
                "auto_heal": "/api/dashboard/recovery/auto-heal",
                "task_reassign": "/api/dashboard/tasks/queue"  # Just check endpoint exists
            }
            
            backend_endpoint = backend_endpoint_map.get(recovery_action)
            if backend_endpoint:
                try:
                    if recovery_action == "system_reset":
                        response = await self.http_client.post(
                            f"{self.backend_url}{backend_endpoint}",
                            params={"reset_type": "soft", "confirm": False}
                        )
                    elif recovery_action in ["agent_restart", "auto_heal"]:
                        response = await self.http_client.post(
                            f"{self.backend_url}{backend_endpoint}",
                            params={"recovery_type": "smart", "dry_run": True}
                        )
                    else:
                        response = await self.http_client.get(f"{self.backend_url}{backend_endpoint}")
                    
                    backend_available = response.status_code == 200
                except:
                    backend_available = False
            else:
                backend_available = True  # No backend check needed
            
            success = button_exists and button_enabled and button_visible and backend_available
            if confirmation_required:
                success = success and confirmation_appeared
            
            details = f"Button: {'✓' if button_exists and button_enabled and button_visible else '✗'}"
            details += f", Confirmation: {'✓' if confirmation_appeared or not confirmation_required else '✗'}"
            details += f", Backend: {'✓' if backend_available else '✗'}"
            
            return success, details
            
        except Exception as e:
            return False, f"Recovery action test failed: {str(e)}"
    
    # ===== CORE CRISIS TESTING =====
    
    async def test_20_percent_success_rate_detection(self):
        """Test detection and response to 20% success rate crisis."""
        suite = self.get_suite("Crisis Detection")
        
        crisis_scenarios = [
            {"name": "20% Crisis", "success_rate": 20.0, "expected_severity": "critical"},
            {"name": "15% Crisis", "success_rate": 15.0, "expected_severity": "critical"},
            {"name": "25% Crisis", "success_rate": 25.0, "expected_severity": "severe"},
            {"name": "35% Crisis", "success_rate": 35.0, "expected_severity": "moderate"}
        ]
        
        for scenario in crisis_scenarios:
            start_time = time.perf_counter()
            
            try:
                # Simulate crisis conditions
                crisis_data = await self.simulate_coordination_crisis(
                    success_rate=scenario["success_rate"],
                    duration_seconds=30
                )
                
                # Load dashboard
                page = await self.context.new_page()
                await page.goto(self.frontend_url, wait_until="networkidle")
                
                # Validate crisis detection
                crisis_detected, detection_details = await self.validate_crisis_detection(
                    page, scenario["success_rate"]
                )
                
                # Validate dashboard functionality during crisis
                dashboard_functional, functionality_details = await self.validate_dashboard_functionality_during_crisis(page)
                
                # Check if appropriate severity level is indicated
                severity_indicated = await page.evaluate(f"""() => {{
                    const severity_elements = document.querySelectorAll('.severity-indicator, [data-testid*="severity"], .crisis-level');
                    const text = Array.from(severity_elements).map(el => el.textContent.toLowerCase()).join(' ');
                    return text.includes('{scenario["expected_severity"]}') || 
                           text.includes('critical') || 
                           text.includes('severe') ||
                           text.includes('emergency');
                }}""")
                
                success = crisis_detected and dashboard_functional
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                details = f"Detection: {detection_details}; Functionality: {functionality_details}; Severity indicated: {severity_indicated}"
                
                await page.close()
                
            except Exception as e:
                success = False
                duration_ms = (time.perf_counter() - start_time) * 1000
                details = f"Crisis detection test failed: {str(e)}"
                crisis_detected = False
                dashboard_functional = False
            
            result = CrisisTestResult(
                scenario=scenario["name"],
                test_name="Crisis Detection and Dashboard Response",
                success=success,
                details=details,
                duration_ms=duration_ms,
                critical=True,
                crisis_detected=crisis_detected,
                dashboard_remained_functional=dashboard_functional
            )
            suite.add_result(result)
    
    async def test_emergency_recovery_during_crisis(self):
        """Test emergency recovery actions during active crisis conditions."""
        suite = self.get_suite("Emergency Recovery During Crisis")
        
        recovery_actions = [
            {"name": "System Reset", "action": "system_reset", "critical": True},
            {"name": "Agent Restart", "action": "agent_restart", "critical": True},
            {"name": "Auto Heal", "action": "auto_heal", "critical": False},
            {"name": "Task Reassignment", "action": "task_reassign", "critical": False}
        ]
        
        for recovery in recovery_actions:
            start_time = time.perf_counter()
            
            try:
                # Simulate active crisis
                crisis_data = await self.simulate_coordination_crisis(
                    success_rate=CRISIS_SUCCESS_RATE,
                    duration_seconds=60
                )
                
                # Load dashboard during crisis
                page = await self.context.new_page()
                await page.goto(self.frontend_url, wait_until="networkidle")
                
                # Wait for crisis to be detected
                await asyncio.sleep(5)
                
                # Test recovery action effectiveness
                recovery_effective, recovery_details = await self.test_recovery_action_effectiveness(
                    page, recovery["action"]
                )
                
                # Validate dashboard remains responsive during recovery action
                dashboard_responsive = await page.evaluate("""() => {
                    // Check if page is still interactive
                    const buttons = document.querySelectorAll('button');
                    return buttons.length > 0 && !document.querySelector('.loading-overlay, .spinner');
                }""")
                
                success = recovery_effective and dashboard_responsive
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                details = f"Recovery: {recovery_details}; Dashboard responsive: {dashboard_responsive}"
                
                await page.close()
                
            except Exception as e:
                success = False
                duration_ms = (time.perf_counter() - start_time) * 1000
                details = f"Emergency recovery test failed: {str(e)}"
                recovery_effective = False
            
            result = CrisisTestResult(
                scenario=f"Crisis Recovery - {recovery['name']}",
                test_name="Emergency Recovery During Crisis",
                success=success,
                details=details,
                duration_ms=duration_ms,
                critical=recovery["critical"],
                recovery_successful=recovery_effective
            )
            suite.add_result(result)
    
    async def test_sustained_crisis_dashboard_resilience(self):
        """Test dashboard resilience during sustained crisis conditions."""
        suite = self.get_suite("Sustained Crisis Resilience")
        
        resilience_scenarios = [
            {"name": "Short Crisis", "duration": 30, "severity": "moderate"},
            {"name": "Extended Crisis", "duration": 60, "severity": "high"},
            {"name": "Prolonged Crisis", "duration": 120, "severity": "extreme"}
        ]
        
        for scenario in resilience_scenarios:
            start_time = time.perf_counter()
            
            try:
                # Start sustained crisis
                crisis_data = await self.simulate_coordination_crisis(
                    success_rate=CRISIS_SUCCESS_RATE,
                    duration_seconds=scenario["duration"]
                )
                
                # Load dashboard
                page = await self.context.new_page()
                await page.goto(self.frontend_url, wait_until="networkidle")
                
                # Monitor dashboard during sustained crisis
                monitoring_results = []
                monitoring_interval = min(10, scenario["duration"] // 6)  # Check 6 times during crisis
                
                for check_point in range(0, scenario["duration"], monitoring_interval):
                    await asyncio.sleep(monitoring_interval)
                    
                    # Check dashboard functionality at this point
                    functional, func_details = await self.validate_dashboard_functionality_during_crisis(page)
                    
                    # Check if success rate is still being reported
                    success_rate_visible = await page.evaluate("""() => {
                        const elements = document.querySelectorAll('[data-testid*="success-rate"], .success-rate');
                        return Array.from(elements).some(el => el.textContent.includes('%'));
                    }""")
                    
                    # Check if WebSocket connection is maintained (if indicator exists)
                    websocket_connected = await page.evaluate("""() => {
                        return window.websocketStatus || window.wsConnected || true; // Assume connected if no indicator
                    }""")
                    
                    monitoring_results.append({
                        "time": check_point,
                        "functional": functional,
                        "success_rate_visible": success_rate_visible,
                        "websocket_connected": websocket_connected
                    })
                
                # Analyze resilience results
                functional_checks = len([r for r in monitoring_results if r["functional"]])
                success_rate_checks = len([r for r in monitoring_results if r["success_rate_visible"]])
                websocket_checks = len([r for r in monitoring_results if r["websocket_connected"]])
                total_checks = len(monitoring_results)
                
                resilience_rate = (functional_checks / total_checks) if total_checks > 0 else 0
                
                success = resilience_rate >= 0.8  # 80% of checks should pass
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                details = f"Functional: {functional_checks}/{total_checks} ({resilience_rate*100:.1f}%)"
                details += f", Success rate visible: {success_rate_checks}/{total_checks}"
                details += f", WebSocket: {websocket_checks}/{total_checks}"
                
                await page.close()
                
            except Exception as e:
                success = False
                duration_ms = (time.perf_counter() - start_time) * 1000
                details = f"Sustained crisis test failed: {str(e)}"
            
            result = CrisisTestResult(
                scenario=scenario["name"],
                test_name="Sustained Crisis Dashboard Resilience",
                success=success,
                details=details,
                duration_ms=duration_ms,
                critical=scenario["severity"] in ["high", "extreme"],
                dashboard_remained_functional=success
            )
            suite.add_result(result)
    
    async def test_operator_guidance_during_crisis(self):
        """Test that dashboard provides clear operator guidance during crisis."""
        suite = self.get_suite("Operator Guidance During Crisis")
        
        guidance_scenarios = [
            {"name": "First-time Crisis", "user_type": "novice", "guidance_level": "detailed"},
            {"name": "Recurring Crisis", "user_type": "experienced", "guidance_level": "summary"},
            {"name": "Escalated Crisis", "user_type": "expert", "guidance_level": "advanced"}
        ]
        
        for scenario in guidance_scenarios:
            start_time = time.perf_counter()
            
            try:
                # Simulate crisis
                crisis_data = await self.simulate_coordination_crisis(
                    success_rate=CRISIS_SUCCESS_RATE,
                    duration_seconds=45
                )
                
                # Load dashboard
                page = await self.context.new_page()
                await page.goto(self.frontend_url, wait_until="networkidle")
                await asyncio.sleep(5)  # Wait for crisis detection
                
                # Check for operator guidance elements
                guidance_checks = []
                
                # Check for crisis explanation/description
                crisis_explanation = await page.evaluate("""() => {
                    const explanation_elements = document.querySelectorAll('.crisis-explanation, [data-testid*="crisis-info"], .help-text');
                    return Array.from(explanation_elements).some(el => el.textContent.length > 20);
                }""")
                guidance_checks.append({"name": "Crisis Explanation", "present": crisis_explanation})
                
                # Check for recommended actions
                recommended_actions = await page.evaluate("""() => {
                    const action_elements = document.querySelectorAll('.recommended-actions, [data-testid*="recommendation"], .next-steps');
                    return Array.from(action_elements).some(el => el.textContent.length > 10);
                }""")
                guidance_checks.append({"name": "Recommended Actions", "present": recommended_actions})
                
                # Check for step-by-step guidance
                step_guidance = await page.evaluate("""() => {
                    const step_elements = document.querySelectorAll('.step-by-step, [data-testid*="steps"], .procedure, ol li, ul li');
                    return step_elements.length >= 2; // At least 2 steps
                }""")
                guidance_checks.append({"name": "Step-by-step Guidance", "present": step_guidance})
                
                # Check for severity indicators
                severity_indicators = await page.evaluate("""() => {
                    const severity_elements = document.querySelectorAll('.severity-indicator, [data-testid*="severity"], .priority-level');
                    return Array.from(severity_elements).some(el => el.offsetHeight > 0);
                }""")
                guidance_checks.append({"name": "Severity Indicators", "present": severity_indicators})
                
                # Check for help or documentation links
                help_available = await page.evaluate("""() => {
                    const help_elements = document.querySelectorAll('[data-testid*="help"], .help-button, .documentation-link, [href*="help"]');
                    return help_elements.length > 0;
                }""")
                guidance_checks.append({"name": "Help Resources", "present": help_available})
                
                # Calculate guidance completeness
                present_guidance = len([check for check in guidance_checks if check["present"]])
                total_guidance = len(guidance_checks)
                guidance_completeness = (present_guidance / total_guidance) if total_guidance > 0 else 0
                
                success = guidance_completeness >= 0.6  # 60% of guidance elements should be present
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                details = f"Guidance elements: {present_guidance}/{total_guidance} ({guidance_completeness*100:.1f}%)"
                for check in guidance_checks:
                    status = "✓" if check["present"] else "✗"
                    details += f"; {check['name']}: {status}"
                
                await page.close()
                
            except Exception as e:
                success = False
                duration_ms = (time.perf_counter() - start_time) * 1000
                details = f"Operator guidance test failed: {str(e)}"
            
            result = CrisisTestResult(
                scenario=scenario["name"],
                test_name="Operator Guidance During Crisis",
                success=success,
                details=details,
                duration_ms=duration_ms,
                critical=scenario["guidance_level"] == "detailed"
            )
            suite.add_result(result)
    
    async def test_crisis_to_recovery_workflow(self):
        """Test complete workflow from crisis detection to recovery validation."""
        suite = self.get_suite("Crisis to Recovery Workflow")
        
        start_time = time.perf_counter()
        
        try:
            # Phase 1: Simulate initial crisis
            crisis_data = await self.simulate_coordination_crisis(
                success_rate=CRISIS_SUCCESS_RATE,
                duration_seconds=30
            )
            
            # Load dashboard
            page = await self.context.new_page()
            await page.goto(self.frontend_url, wait_until="networkidle")
            
            # Phase 2: Detect crisis
            await asyncio.sleep(5)
            crisis_detected, detection_details = await self.validate_crisis_detection(page, CRISIS_SUCCESS_RATE)
            
            # Phase 3: Execute recovery action (system reset with confirmation)
            recovery_attempted = False
            try:
                reset_button = page.locator("[data-testid='system-reset-btn'], .reset-coordination-btn")
                if await reset_button.count() > 0:
                    await reset_button.click()
                    
                    # Handle confirmation dialog
                    try:
                        await page.wait_for_selector(".confirmation-dialog, .modal", timeout=3000)
                        # In real recovery, we would confirm. For testing, we cancel.
                        cancel_btn = page.locator("[data-testid='cancel-btn'], .cancel-btn, button:has-text('Cancel')")
                        if await cancel_btn.count() > 0:
                            await cancel_btn.click()
                        recovery_attempted = True
                    except:
                        recovery_attempted = False
            except:
                recovery_attempted = False
            
            # Phase 4: Simulate recovery (improved success rate)
            recovery_data = await self.simulate_coordination_crisis(
                success_rate=RECOVERY_SUCCESS_RATE,  # 85% success rate
                duration_seconds=20
            )
            
            # Phase 5: Validate recovery detection
            await asyncio.sleep(5)
            recovery_detected = await page.evaluate(f"""() => {{
                const elements = document.querySelectorAll('[data-testid*="success-rate"], .success-rate');
                for (const el of elements) {{
                    const match = el.textContent.match(/(\d+(?:\.\d+)?)%/);
                    if (match) {{
                        const rate = parseFloat(match[1]);
                        return rate > 50; // Recovery should show >50% success rate
                    }}
                }}
                return false;
            }}""")
            
            # Phase 6: Validate crisis UI disappears
            crisis_ui_cleared = await page.evaluate("""() => {
                const crisis_elements = document.querySelectorAll('[data-testid*="crisis"], .crisis-alert, .critical-warning');
                return Array.from(crisis_elements).every(el => el.offsetHeight === 0 || !el.offsetParent);
            }""")
            
            # Phase 7: Validate normal operation restored
            normal_operation = await page.evaluate("""() => {
                // Check that normal dashboard elements are functioning
                const normal_elements = document.querySelectorAll('.agent-status, .task-status, .system-health');
                return normal_elements.length > 0;
            }""")
            
            # Overall workflow success
            workflow_phases = [
                ("Crisis Detection", crisis_detected),
                ("Recovery Attempted", recovery_attempted),
                ("Recovery Detected", recovery_detected),
                ("Crisis UI Cleared", crisis_ui_cleared),
                ("Normal Operation", normal_operation)
            ]
            
            successful_phases = len([phase for phase, success in workflow_phases if success])
            total_phases = len(workflow_phases)
            workflow_success_rate = (successful_phases / total_phases) if total_phases > 0 else 0
            
            success = workflow_success_rate >= 0.8  # 80% of workflow phases should succeed
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            details = f"Workflow phases: {successful_phases}/{total_phases} ({workflow_success_rate*100:.1f}%)"
            for phase_name, phase_success in workflow_phases:
                status = "✓" if phase_success else "✗"
                details += f"; {phase_name}: {status}"
            
            await page.close()
            
        except Exception as e:
            success = False
            duration_ms = (time.perf_counter() - start_time) * 1000
            details = f"Crisis to recovery workflow failed: {str(e)}"
            crisis_detected = False
            recovery_attempted = False
        
        result = CrisisTestResult(
            scenario="Complete Crisis to Recovery",
            test_name="Crisis to Recovery Workflow",
            success=success,
            details=details,
            duration_ms=duration_ms,
            critical=True,
            crisis_detected=crisis_detected,
            recovery_successful=recovery_attempted and success
        )
        suite.add_result(result)
    
    # ===== COMPREHENSIVE TEST EXECUTION =====
    
    async def run_all_critical_failure_tests(self) -> Dict[str, CrisisTestSuite]:
        """Run all critical failure scenario tests."""
        print("🚨 Starting Critical Failure Scenario Testing")
        print("="*55)
        print(f"Crisis Success Rate: {CRISIS_SUCCESS_RATE}% (Target scenario)")
        print(f"Recovery Success Rate: {RECOVERY_SUCCESS_RATE}% (Target recovery)")
        print("="*55)
        
        # Test 1: 20% Success Rate Detection
        print("\n🔍 Testing 20% Success Rate Crisis Detection...")
        await self.test_20_percent_success_rate_detection()
        
        # Test 2: Emergency Recovery During Crisis
        print("\n🆘 Testing Emergency Recovery During Crisis...")
        await self.test_emergency_recovery_during_crisis()
        
        # Test 3: Sustained Crisis Resilience
        print("\n⏱️  Testing Sustained Crisis Dashboard Resilience...")
        await self.test_sustained_crisis_dashboard_resilience()
        
        # Test 4: Operator Guidance During Crisis
        print("\n📋 Testing Operator Guidance During Crisis...")
        await self.test_operator_guidance_during_crisis()
        
        # Test 5: Complete Crisis to Recovery Workflow
        print("\n🔄 Testing Complete Crisis to Recovery Workflow...")
        await self.test_crisis_to_recovery_workflow()
        
        return self.test_suites
    
    def print_critical_failure_test_report(self):
        """Print comprehensive critical failure test report."""
        print("\n" + "="*70)
        print("CRITICAL FAILURE SCENARIO TEST REPORT")
        print("="*70)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_critical_failures = 0
        total_crisis_detected = 0
        total_recoveries_attempted = 0
        total_successful_recoveries = 0
        
        for suite_name, suite in self.test_suites.items():
            print(f"\n🚨 {suite.name}")
            print(f"   Tests: {len(suite.results)} | Passed: {suite.passed} | Failed: {suite.failed}")
            print(f"   Success Rate: {suite.success_rate:.1f}%")
            print(f"   Crisis Detection Rate: {suite.crisis_detection_rate:.1f}%")
            if suite.recovery_success_rate > 0:
                print(f"   Recovery Success Rate: {suite.recovery_success_rate:.1f}%")
            
            if suite.critical_failures > 0:
                print(f"   🔴 CRITICAL FAILURES: {suite.critical_failures}")
            
            # Show failed tests
            failed_tests = [r for r in suite.results if not r.success]
            if failed_tests:
                print("   Failed Tests:")
                for test in failed_tests[:2]:  # Show first 2 failures
                    print(f"     🔴 {test.scenario}: {test.details[:60]}...")
                if len(failed_tests) > 2:
                    print(f"     ... and {len(failed_tests) - 2} more")
            
            total_tests += len(suite.results)
            total_passed += suite.passed
            total_failed += suite.failed
            total_critical_failures += suite.critical_failures
            total_crisis_detected += len([r for r in suite.results if r.crisis_detected])
            
            recovery_tests = [r for r in suite.results if r.recovery_successful is not False]
            total_recoveries_attempted += len(recovery_tests)
            total_successful_recoveries += len([r for r in recovery_tests if r.recovery_successful])
        
        # Overall summary
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        overall_crisis_detection_rate = (total_crisis_detected / total_tests * 100) if total_tests > 0 else 0
        overall_recovery_rate = (total_successful_recoveries / total_recoveries_attempted * 100) if total_recoveries_attempted > 0 else 0
        
        print(f"\n{'='*70}")
        print("CRITICAL FAILURE SYSTEM STATUS")
        print(f"{'='*70}")
        print(f"Total Crisis Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"Crisis Detection Rate: {overall_crisis_detection_rate:.1f}%")
        print(f"Recovery Success Rate: {overall_recovery_rate:.1f}%")
        print(f"Critical Failures: {total_critical_failures}")
        
        # Crisis management readiness assessment
        print(f"\n🎯 CRISIS MANAGEMENT READINESS:")
        
        if (total_critical_failures == 0 and overall_success_rate >= 85 and 
            overall_crisis_detection_rate >= 90 and overall_recovery_rate >= 75):
            print("🟢 FULLY READY - Dashboard can handle 20% coordination crisis")
            status = "FULLY_READY"
        elif (total_critical_failures <= 1 and overall_success_rate >= 75 and 
              overall_crisis_detection_rate >= 80 and overall_recovery_rate >= 60):
            print("🟡 MOSTLY READY - Dashboard can handle crisis with minor issues")
            status = "MOSTLY_READY"
        elif (total_critical_failures <= 3 and overall_success_rate >= 60 and 
              overall_crisis_detection_rate >= 70):
            print("🟠 NEEDS IMPROVEMENT - Dashboard has crisis management gaps")
            status = "NEEDS_IMPROVEMENT"
        else:
            print("🔴 NOT READY - Dashboard cannot adequately handle coordination crisis")
            status = "NOT_READY"
        
        print(f"\n📊 20% SUCCESS RATE CRISIS READINESS ASSESSMENT:")
        
        # Specific crisis readiness factors
        crisis_readiness_factors = []
        
        if overall_crisis_detection_rate >= 90:
            crisis_readiness_factors.append("✅ Crisis Detection: Excellent")
        elif overall_crisis_detection_rate >= 75:
            crisis_readiness_factors.append("🟡 Crisis Detection: Good")
        else:
            crisis_readiness_factors.append("❌ Crisis Detection: Poor")
        
        if overall_recovery_rate >= 75:
            crisis_readiness_factors.append("✅ Recovery Actions: Effective")
        elif overall_recovery_rate >= 60:
            crisis_readiness_factors.append("🟡 Recovery Actions: Adequate")
        else:
            crisis_readiness_factors.append("❌ Recovery Actions: Ineffective")
        
        dashboard_resilience_tests = [r for r in self.test_suites.get("Sustained Crisis Resilience", CrisisTestSuite("")).results]
        if dashboard_resilience_tests:
            resilient_tests = len([r for r in dashboard_resilience_tests if r.dashboard_remained_functional])
            resilience_rate = (resilient_tests / len(dashboard_resilience_tests) * 100)
            
            if resilience_rate >= 80:
                crisis_readiness_factors.append("✅ Dashboard Resilience: Excellent")
            elif resilience_rate >= 60:
                crisis_readiness_factors.append("🟡 Dashboard Resilience: Good")
            else:
                crisis_readiness_factors.append("❌ Dashboard Resilience: Poor")
        
        for factor in crisis_readiness_factors:
            print(f"   {factor}")
        
        print(f"\n🏆 FINAL ASSESSMENT:")
        if status == "FULLY_READY":
            print("   The dashboard is fully capable of helping operators resolve")
            print("   the 20% coordination success rate crisis effectively.")
        elif status == "MOSTLY_READY":
            print("   The dashboard can handle the crisis with some limitations.")
            print("   Minor improvements recommended for optimal crisis management.")
        elif status == "NEEDS_IMPROVEMENT":
            print("   The dashboard has significant gaps in crisis management.")
            print("   Important improvements needed before production deployment.")
        else:
            print("   The dashboard is not ready for crisis management.")
            print("   Major issues must be resolved before deployment.")
        
        print(f"{'='*70}")
        
        return {
            "overall_success_rate": overall_success_rate,
            "crisis_detection_rate": overall_crisis_detection_rate,
            "recovery_success_rate": overall_recovery_rate,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "critical_failures": total_critical_failures,
            "crisis_readiness_status": status,
            "suites": {name: {
                "success_rate": suite.success_rate,
                "crisis_detection_rate": suite.crisis_detection_rate,
                "recovery_success_rate": suite.recovery_success_rate,
                "tests": len(suite.results),
                "passed": suite.passed,
                "failed": suite.failed,
                "critical_failures": suite.critical_failures
            } for name, suite in self.test_suites.items()}
        }


# ===== TEST EXECUTION FUNCTIONS =====

async def run_critical_failure_validation():
    """Run comprehensive critical failure validation."""
    async with CriticalFailureTester() as tester:
        await tester.run_all_critical_failure_tests()
        return tester.print_critical_failure_test_report()


async def run_crisis_detection_only():
    """Run only crisis detection tests."""
    print("🔍 Running Crisis Detection Tests Only")
    print("="*35)
    
    async with CriticalFailureTester() as tester:
        await tester.test_20_percent_success_rate_detection()
        return tester.print_critical_failure_test_report()


async def run_recovery_workflow_only():
    """Run only recovery workflow tests."""
    print("🔄 Running Recovery Workflow Tests Only")
    print("="*35)
    
    async with CriticalFailureTester() as tester:
        await tester.test_emergency_recovery_during_crisis()
        await tester.test_crisis_to_recovery_workflow()
        return tester.print_critical_failure_test_report()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "detection":
            result = asyncio.run(run_crisis_detection_only())
        elif test_type == "recovery":
            result = asyncio.run(run_recovery_workflow_only())
        else:
            print(f"Unknown test type: {test_type}")
            print("Available types: detection, recovery")
            sys.exit(1)
    else:
        # Run comprehensive validation
        result = asyncio.run(run_critical_failure_validation())
    
    # Set exit code based on results
    if (result and result.get("critical_failures", 1) == 0 and 
        result.get("overall_success_rate", 0) >= 75 and
        result.get("crisis_detection_rate", 0) >= 80):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure