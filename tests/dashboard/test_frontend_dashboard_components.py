"""
Frontend Dashboard Component Testing Framework

Tests the Vue.js components that comprise the multi-agent coordination monitoring dashboard.
Validates real-time monitoring functionality, user interface responsiveness, and 
coordination crisis management capabilities.

CRITICAL CONTEXT: These components must provide reliable operational visibility 
to help operators resolve the 20% coordination success rate crisis.

Test Coverage:
1. Vue.js Component Validation - All 5 core dashboard components
2. Real-time Data Binding - WebSocket integration and reactive updates
3. User Interface Responsiveness - Load times <1 second
4. Mobile-Responsive Design - Touch controls and responsive layouts
5. Emergency Control Interface - Critical recovery action buttons
6. Data Visualization - Charts, graphs, and real-time metrics display
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import httpx

# Configuration
DASHBOARD_URL = "http://localhost:8000"
MOBILE_PWA_URL = "http://localhost:5173"  # Vite dev server
COMPONENT_LOAD_THRESHOLD_MS = 1000  # 1 second load time requirement
MOBILE_VIEWPORTS = [
    {"width": 375, "height": 667, "name": "iPhone SE"},
    {"width": 414, "height": 896, "name": "iPhone XR"},
    {"width": 768, "height": 1024, "name": "iPad"},
    {"width": 1024, "height": 768, "name": "iPad Landscape"}
]


@dataclass
class ComponentTestResult:
    """Component test result tracking."""
    component: str
    test_name: str
    success: bool
    details: str = ""
    load_time_ms: float = 0.0
    critical: bool = False
    viewport: str = "desktop"


@dataclass
class FrontendTestSuite:
    """Frontend test suite aggregator."""
    name: str
    results: List[ComponentTestResult] = field(default_factory=list)
    
    def add_result(self, result: ComponentTestResult):
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
    def average_load_time(self) -> float:
        load_times = [r.load_time_ms for r in self.results if r.load_time_ms > 0]
        return sum(load_times) / len(load_times) if load_times else 0.0


class DashboardComponentTester:
    """Comprehensive tester for dashboard Vue.js components."""
    
    def __init__(self, dashboard_url: str = DASHBOARD_URL, mobile_pwa_url: str = MOBILE_PWA_URL):
        self.dashboard_url = dashboard_url
        self.mobile_pwa_url = mobile_pwa_url
        self.test_suites: Dict[str, FrontendTestSuite] = {}
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
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    def get_suite(self, suite_name: str) -> FrontendTestSuite:
        """Get or create test suite."""
        if suite_name not in self.test_suites:
            self.test_suites[suite_name] = FrontendTestSuite(suite_name)
        return self.test_suites[suite_name]
    
    async def measure_page_load(self, page: Page, url: str) -> float:
        """Measure page load time."""
        start_time = time.perf_counter()
        await page.goto(url, wait_until="networkidle", timeout=10000)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000
    
    async def wait_for_vue_component(self, page: Page, component_selector: str, timeout: int = 5000) -> bool:
        """Wait for Vue component to be rendered."""
        try:
            await page.wait_for_selector(component_selector, timeout=timeout)
            
            # Wait for Vue reactive data to load
            await page.wait_for_function(
                f"""() => {{
                    const element = document.querySelector('{component_selector}');
                    return element && element.innerHTML.trim() !== '';
                }}""",
                timeout=timeout
            )
            return True
        except:
            return False
    
    # ===== CORE COMPONENT TESTING =====
    
    async def test_vue_component_loading(self):
        """Test all 5 core Vue.js dashboard components."""
        suite = self.get_suite("Vue.js Component Loading")
        
        # Core dashboard components to test
        components = [
            {
                "name": "Coordination Success Panel",
                "url": f"{self.mobile_pwa_url}",
                "selector": ".coordination-success-panel",
                "critical": True,
                "test_data_binding": True
            },
            {
                "name": "Agent Health Panel", 
                "url": f"{self.mobile_pwa_url}",
                "selector": ".agent-health-panel",
                "critical": True,
                "test_data_binding": True
            },
            {
                "name": "Task Distribution Panel",
                "url": f"{self.mobile_pwa_url}",
                "selector": ".task-distribution-panel", 
                "critical": False,
                "test_data_binding": True
            },
            {
                "name": "Recovery Controls Panel",
                "url": f"{self.mobile_pwa_url}",
                "selector": ".recovery-controls-panel",
                "critical": True,
                "test_data_binding": False
            },
            {
                "name": "Real-time Agent Status Panel",
                "url": f"{self.mobile_pwa_url}",
                "selector": ".realtime-agent-status-panel",
                "critical": True,
                "test_data_binding": True
            }
        ]
        
        for component in components:
            try:
                page = await self.context.new_page()
                
                # Measure load time
                load_time_ms = await self.measure_page_load(page, component["url"])
                
                # Check if component loads
                component_loaded = await self.wait_for_vue_component(
                    page, component["selector"], timeout=5000
                )
                
                success = component_loaded and load_time_ms < COMPONENT_LOAD_THRESHOLD_MS
                
                details = f"Loaded: {component_loaded}, Load time: {load_time_ms:.1f}ms"
                
                if load_time_ms >= COMPONENT_LOAD_THRESHOLD_MS:
                    details += f" (SLOW - threshold: {COMPONENT_LOAD_THRESHOLD_MS}ms)"
                
                # Test data binding if specified
                if component_loaded and component["test_data_binding"]:
                    try:
                        # Check for reactive data elements
                        has_data = await page.evaluate(f"""() => {{
                            const element = document.querySelector('{component["selector"]}');
                            const textContent = element ? element.textContent : '';
                            return textContent.length > 10 && !textContent.includes('Loading');
                        }}""")
                        
                        if not has_data:
                            success = False
                            details += " (No reactive data)"
                        else:
                            details += " (Data binding active)"
                            
                    except Exception as e:
                        details += f" (Data binding test failed: {str(e)[:50]})"
                
                await page.close()
                
            except Exception as e:
                success = False
                details = f"Component test failed: {str(e)[:100]}"
                load_time_ms = 0
            
            result = ComponentTestResult(
                component=component["name"],
                test_name="Component Loading",
                success=success,
                details=details,
                load_time_ms=load_time_ms,
                critical=component["critical"],
                viewport="desktop"
            )
            suite.add_result(result)
    
    async def test_real_time_data_binding(self):
        """Test real-time data binding and WebSocket integration."""
        suite = self.get_suite("Real-time Data Binding")
        
        # Components that should have real-time updates
        realtime_components = [
            {
                "name": "Coordination Success Rate",
                "selector": ".coordination-success-rate",
                "data_selector": "[data-testid='success-rate-value']",
                "update_expected": True
            },
            {
                "name": "Agent Status Updates",
                "selector": ".agent-status-list",
                "data_selector": "[data-testid='agent-count']",
                "update_expected": True
            },
            {
                "name": "Task Queue Status",
                "selector": ".task-queue-status", 
                "data_selector": "[data-testid='queue-length']",
                "update_expected": True
            },
            {
                "name": "System Health Indicators",
                "selector": ".system-health",
                "data_selector": "[data-testid='health-score']",
                "update_expected": True
            }
        ]
        
        try:
            page = await self.context.new_page()
            load_time_ms = await self.measure_page_load(page, self.mobile_pwa_url)
            
            for component in realtime_components:
                try:
                    # Wait for component to load
                    component_loaded = await self.wait_for_vue_component(
                        page, component["selector"], timeout=10000
                    )
                    
                    if not component_loaded:
                        result = ComponentTestResult(
                            component=component["name"],
                            test_name="Real-time Data Binding",
                            success=False,
                            details="Component not found",
                            critical=True
                        )
                        suite.add_result(result)
                        continue
                    
                    # Get initial data value
                    initial_value = await page.evaluate(f"""() => {{
                        const element = document.querySelector('{component["data_selector"]}');
                        return element ? element.textContent : null;
                    }}""")
                    
                    # Wait for potential updates (simulate real-time activity)
                    await asyncio.sleep(2)
                    
                    # Check if value has been updated or if component is reactive
                    final_value = await page.evaluate(f"""() => {{
                        const element = document.querySelector('{component["data_selector"]}');
                        return element ? element.textContent : null;
                    }}""")
                    
                    # Test WebSocket connection status
                    websocket_connected = await page.evaluate("""() => {
                        return window.websocketStatus || window.wsConnected || false;
                    }""")
                    
                    success = initial_value is not None and (
                        final_value != initial_value or websocket_connected
                    )
                    
                    details = f"Initial: '{initial_value}', Final: '{final_value}'"
                    if websocket_connected:
                        details += ", WebSocket: Connected"
                    else:
                        details += ", WebSocket: Unknown"
                        
                    if success:
                        details += " (Reactive)"
                    else:
                        details += " (Not reactive)"
                    
                except Exception as e:
                    success = False
                    details = f"Real-time test error: {str(e)[:100]}"
                
                result = ComponentTestResult(
                    component=component["name"],
                    test_name="Real-time Data Binding",
                    success=success,
                    details=details,
                    critical=True
                )
                suite.add_result(result)
            
            await page.close()
            
        except Exception as e:
            # Add a general failure result
            result = ComponentTestResult(
                component="Real-time System",
                test_name="Data Binding Setup",
                success=False,
                details=f"Setup failed: {str(e)[:100]}",
                critical=True
            )
            suite.add_result(result)
    
    async def test_mobile_responsive_design(self):
        """Test mobile-responsive design and touch controls."""
        suite = self.get_suite("Mobile Responsive Design")
        
        for viewport in MOBILE_VIEWPORTS:
            try:
                # Create mobile context
                mobile_context = await self.browser.new_context(
                    viewport={"width": viewport["width"], "height": viewport["height"]},
                    user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
                )
                
                page = await mobile_context.new_page()
                load_time_ms = await self.measure_page_load(page, self.mobile_pwa_url)
                
                # Test responsive layout
                responsive_tests = [
                    {
                        "name": "Navigation Menu",
                        "selector": ".mobile-navigation, .bottom-navigation",
                        "critical": False
                    },
                    {
                        "name": "Dashboard Grid",
                        "selector": ".dashboard-grid, .mobile-dashboard",
                        "critical": True
                    },
                    {
                        "name": "Emergency Controls",
                        "selector": ".emergency-controls, .mobile-emergency",
                        "critical": True
                    },
                    {
                        "name": "Agent Status Cards",
                        "selector": ".agent-cards, .mobile-agent-status",
                        "critical": True
                    }
                ]
                
                for test in responsive_tests:
                    try:
                        # Check if element exists and is visible
                        element_visible = await page.is_visible(test["selector"])
                        
                        if element_visible:
                            # Check if element has appropriate mobile styling
                            mobile_optimized = await page.evaluate(f"""() => {{
                                const element = document.querySelector('{test["selector"]}');
                                if (!element) return false;
                                
                                const styles = window.getComputedStyle(element);
                                const width = parseInt(styles.width);
                                const height = parseInt(styles.height);
                                
                                // Check if element is sized appropriately for mobile
                                return width <= {viewport["width"]} && height > 0;
                            }}""")
                            
                            success = mobile_optimized
                            details = f"Visible: {element_visible}, Mobile optimized: {mobile_optimized}"
                        else:
                            success = False
                            details = "Element not visible on mobile"
                        
                    except Exception as e:
                        success = False
                        details = f"Mobile test error: {str(e)[:50]}"
                    
                    result = ComponentTestResult(
                        component=test["name"],
                        test_name="Mobile Responsive Design",
                        success=success,
                        details=details,
                        load_time_ms=load_time_ms,
                        critical=test["critical"],
                        viewport=viewport["name"]
                    )
                    suite.add_result(result)
                
                await mobile_context.close()
                
            except Exception as e:
                # Add failure result for this viewport
                result = ComponentTestResult(
                    component=f"Mobile Layout ({viewport['name']})",
                    test_name="Viewport Test",
                    success=False,
                    details=f"Viewport test failed: {str(e)[:100]}",
                    critical=True,
                    viewport=viewport["name"]
                )
                suite.add_result(result)
    
    async def test_emergency_control_interface(self):
        """Test emergency control interface and critical recovery actions."""
        suite = self.get_suite("Emergency Control Interface")
        
        emergency_controls = [
            {
                "name": "System Reset Button",
                "selector": "[data-testid='system-reset-btn']",
                "action": "click",
                "confirmation_required": True,
                "critical": True
            },
            {
                "name": "Agent Restart Controls",
                "selector": "[data-testid='agent-restart-btn']",
                "action": "click", 
                "confirmation_required": True,
                "critical": True
            },
            {
                "name": "Emergency Override",
                "selector": "[data-testid='emergency-override-btn']",
                "action": "click",
                "confirmation_required": True,
                "critical": True
            },
            {
                "name": "Task Reassignment",
                "selector": "[data-testid='reassign-tasks-btn']",
                "action": "click",
                "confirmation_required": False,
                "critical": False
            }
        ]
        
        try:
            page = await self.context.new_page()
            load_time_ms = await self.measure_page_load(page, self.mobile_pwa_url)
            
            for control in emergency_controls:
                try:
                    # Check if control element exists
                    element_exists = await page.locator(control["selector"]).count() > 0
                    
                    if not element_exists:
                        result = ComponentTestResult(
                            component=control["name"],
                            test_name="Emergency Control Interface",
                            success=False,
                            details="Control element not found",
                            critical=control["critical"]
                        )
                        suite.add_result(result)
                        continue
                    
                    # Check if control is enabled and clickable
                    is_enabled = await page.locator(control["selector"]).is_enabled()
                    is_visible = await page.locator(control["selector"]).is_visible()
                    
                    if is_enabled and is_visible:
                        # Test click action (without actually confirming dangerous actions)
                        try:
                            await page.locator(control["selector"]).click()
                            
                            if control["confirmation_required"]:
                                # Check for confirmation dialog
                                confirmation_visible = await page.is_visible(
                                    ".confirmation-dialog, .modal, [role='dialog']"
                                )
                                
                                if confirmation_visible:
                                    success = True
                                    details = "Control clickable, confirmation dialog shown (safe)"
                                    
                                    # Close confirmation dialog
                                    cancel_btn = page.locator(
                                        "[data-testid='cancel-btn'], .cancel-btn, button:has-text('Cancel')"
                                    )
                                    if await cancel_btn.count() > 0:
                                        await cancel_btn.click()
                                else:
                                    success = False
                                    details = "Control clicked but no confirmation (unsafe)"
                            else:
                                success = True
                                details = "Control clickable, action executed"
                                
                        except Exception as e:
                            success = False
                            details = f"Click test failed: {str(e)[:50]}"
                    else:
                        success = False
                        details = f"Control not interactive (enabled: {is_enabled}, visible: {is_visible})"
                    
                except Exception as e:
                    success = False
                    details = f"Control test error: {str(e)[:100]}"
                
                result = ComponentTestResult(
                    component=control["name"],
                    test_name="Emergency Control Interface",
                    success=success,
                    details=details,
                    critical=control["critical"]
                )
                suite.add_result(result)
            
            await page.close()
            
        except Exception as e:
            result = ComponentTestResult(
                component="Emergency Control System",
                test_name="Interface Setup",
                success=False,
                details=f"Setup failed: {str(e)[:100]}",
                critical=True
            )
            suite.add_result(result)
    
    async def test_data_visualization_components(self):
        """Test charts, graphs, and real-time metrics display."""
        suite = self.get_suite("Data Visualization")
        
        visualization_components = [
            {
                "name": "Success Rate Chart",
                "selector": ".success-rate-chart, [data-testid='success-rate-chart']",
                "chart_type": "line",
                "critical": True
            },
            {
                "name": "Agent Distribution Pie Chart", 
                "selector": ".agent-distribution-chart, [data-testid='agent-chart']",
                "chart_type": "pie",
                "critical": False
            },
            {
                "name": "Task Queue Timeline",
                "selector": ".task-timeline, [data-testid='task-timeline']", 
                "chart_type": "timeline",
                "critical": False
            },
            {
                "name": "System Health Gauge",
                "selector": ".health-gauge, [data-testid='health-gauge']",
                "chart_type": "gauge",
                "critical": True
            }
        ]
        
        try:
            page = await self.context.new_page()
            load_time_ms = await self.measure_page_load(page, self.mobile_pwa_url)
            
            for viz in visualization_components:
                try:
                    # Check if visualization element exists
                    element_exists = await self.wait_for_vue_component(
                        page, viz["selector"], timeout=5000
                    )
                    
                    if not element_exists:
                        result = ComponentTestResult(
                            component=viz["name"],
                            test_name="Data Visualization",
                            success=False,
                            details="Visualization component not found",
                            critical=viz["critical"]
                        )
                        suite.add_result(result)
                        continue
                    
                    # Check if chart has data
                    has_data = await page.evaluate(f"""() => {{
                        const element = document.querySelector('{viz["selector"]}');
                        if (!element) return false;
                        
                        // Check for SVG elements (common in charts)
                        const svgElements = element.querySelectorAll('svg, canvas, .chart-data');
                        if (svgElements.length === 0) return false;
                        
                        // Check for data attributes or non-empty content
                        const hasContent = element.textContent.trim().length > 0 || 
                                         element.children.length > 0;
                        
                        return hasContent;
                    }}""")
                    
                    # Check if chart is responsive
                    is_responsive = await page.evaluate(f"""() => {{
                        const element = document.querySelector('{viz["selector"]}');
                        if (!element) return false;
                        
                        const styles = window.getComputedStyle(element);
                        const width = styles.width;
                        const height = styles.height;
                        
                        return width.includes('%') || width.includes('vw') || 
                               height.includes('%') || height.includes('vh');
                    }}""")
                    
                    success = has_data and element_exists
                    details = f"Exists: {element_exists}, Has data: {has_data}, Responsive: {is_responsive}"
                    
                    if viz["critical"] and viz["name"] == "Success Rate Chart":
                        # Special validation for critical success rate chart
                        success_rate_visible = await page.evaluate(f"""() => {{
                            const element = document.querySelector('{viz["selector"]}');
                            const text = element ? element.textContent : '';
                            return text.includes('%') || text.includes('success') || text.includes('rate');
                        }}""")
                        
                        if success_rate_visible:
                            details += " (Success rate data visible)"
                        else:
                            success = False
                            details += " (No success rate data)"
                    
                except Exception as e:
                    success = False
                    details = f"Visualization test error: {str(e)[:100]}"
                
                result = ComponentTestResult(
                    component=viz["name"],
                    test_name="Data Visualization",
                    success=success,
                    details=details,
                    critical=viz["critical"]
                )
                suite.add_result(result)
            
            await page.close()
            
        except Exception as e:
            result = ComponentTestResult(
                component="Data Visualization System", 
                test_name="Setup",
                success=False,
                details=f"Setup failed: {str(e)[:100]}",
                critical=True
            )
            suite.add_result(result)
    
    # ===== COMPREHENSIVE TEST EXECUTION =====
    
    async def run_all_frontend_tests(self) -> Dict[str, FrontendTestSuite]:
        """Run all comprehensive frontend dashboard tests."""
        print("🎨 Starting Frontend Dashboard Component Testing")
        print("="*60)
        print(f"Target Load Time: <{COMPONENT_LOAD_THRESHOLD_MS}ms")
        print(f"Testing {len(MOBILE_VIEWPORTS)} mobile viewports")
        print("="*60)
        
        # Test 1: Vue.js Component Loading
        print("\n⚡ Testing Vue.js Component Loading...")
        await self.test_vue_component_loading()
        
        # Test 2: Real-time Data Binding  
        print("\n🔄 Testing Real-time Data Binding...")
        await self.test_real_time_data_binding()
        
        # Test 3: Mobile Responsive Design
        print("\n📱 Testing Mobile Responsive Design...")
        await self.test_mobile_responsive_design()
        
        # Test 4: Emergency Control Interface
        print("\n🆘 Testing Emergency Control Interface...")
        await self.test_emergency_control_interface()
        
        # Test 5: Data Visualization
        print("\n📊 Testing Data Visualization Components...")
        await self.test_data_visualization_components()
        
        return self.test_suites
    
    def print_frontend_test_report(self):
        """Print comprehensive frontend test report."""
        print("\n" + "="*60)
        print("FRONTEND DASHBOARD COMPONENT TEST REPORT")
        print("="*60)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_critical_failures = 0
        
        for suite_name, suite in self.test_suites.items():
            print(f"\n🎨 {suite.name}")
            print(f"   Tests: {len(suite.results)} | Passed: {suite.passed} | Failed: {suite.failed}")
            print(f"   Success Rate: {suite.success_rate:.1f}% | Avg Load Time: {suite.average_load_time:.1f}ms")
            
            if suite.critical_failures > 0:
                print(f"   🔴 CRITICAL FAILURES: {suite.critical_failures}")
            
            # Show viewport breakdown for mobile tests
            mobile_results = [r for r in suite.results if r.viewport != "desktop"]
            if mobile_results:
                viewports = set(r.viewport for r in mobile_results)
                print(f"   📱 Mobile Viewports: {', '.join(viewports)}")
            
            # Show failed tests
            failed_tests = [r for r in suite.results if not r.success]
            if failed_tests:
                print("   Failed Tests:")
                for test in failed_tests[:3]:  # Show first 3 failures
                    prefix = "🔴" if test.critical else "❌"
                    viewport_info = f" ({test.viewport})" if test.viewport != "desktop" else ""
                    print(f"     {prefix} {test.component}{viewport_info}: {test.details[:50]}...")
                if len(failed_tests) > 3:
                    print(f"     ... and {len(failed_tests) - 3} more")
            
            total_tests += len(suite.results)
            total_passed += suite.passed
            total_failed += suite.failed
            total_critical_failures += suite.critical_failures
        
        # Overall summary
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n{'='*60}")
        print("FRONTEND SYSTEM STATUS")
        print(f"{'='*60}")
        print(f"Total Component Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {overall_success_rate:.1f}%")
        print(f"Critical Failures: {total_critical_failures}")
        
        # Production readiness assessment
        print(f"\n🎯 FRONTEND PRODUCTION READINESS:")
        
        if total_critical_failures == 0 and overall_success_rate >= 90:
            print("🟢 READY FOR PRODUCTION - All critical UI components operational")
            status = "READY"
        elif total_critical_failures <= 1 and overall_success_rate >= 80:
            print("🟡 READY WITH MINOR ISSUES - Most components operational")
            status = "READY_WITH_ISSUES"  
        elif total_critical_failures <= 3 and overall_success_rate >= 70:
            print("🟠 NEEDS ATTENTION - Some critical UI issues present")
            status = "NEEDS_ATTENTION"
        else:
            print("🔴 NOT READY - Critical UI failures present")
            status = "NOT_READY"
        
        print(f"\n📊 COORDINATION CRISIS UI STATUS:")
        
        # Check coordination UI specific results
        coord_suite = self.test_suites.get("Real-time Data Binding")
        emergency_suite = self.test_suites.get("Emergency Control Interface")
        
        crisis_ui_ready = True
        if coord_suite and coord_suite.critical_failures > 0:
            crisis_ui_ready = False
        if emergency_suite and emergency_suite.critical_failures > 0:
            crisis_ui_ready = False
        
        if crisis_ui_ready:
            print("✅ Dashboard UI ready to help resolve 20% success rate crisis")
        else:
            print("❌ Dashboard UI cannot adequately support coordination crisis management")
        
        print(f"{'='*60}")
        
        return {
            "overall_success_rate": overall_success_rate,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "critical_failures": total_critical_failures,
            "production_status": status,
            "crisis_ui_ready": crisis_ui_ready,
            "suites": {name: {
                "success_rate": suite.success_rate,
                "tests": len(suite.results),
                "passed": suite.passed,
                "failed": suite.failed,
                "critical_failures": suite.critical_failures,
                "average_load_time": suite.average_load_time
            } for name, suite in self.test_suites.items()}
        }


# ===== TEST EXECUTION FUNCTIONS =====

async def run_frontend_component_validation():
    """Run comprehensive frontend component validation."""
    async with DashboardComponentTester() as tester:
        await tester.run_all_frontend_tests()
        return tester.print_frontend_test_report()


async def run_mobile_only_tests():
    """Run only mobile-responsive design tests."""
    print("📱 Running Mobile-Only Component Tests")
    print("="*40)
    
    async with DashboardComponentTester() as tester:
        await tester.test_mobile_responsive_design()
        return tester.print_frontend_test_report()


async def run_emergency_ui_tests():
    """Run only emergency control interface tests."""
    print("🆘 Running Emergency Control UI Tests")
    print("="*40)
    
    async with DashboardComponentTester() as tester:
        await tester.test_emergency_control_interface()
        await tester.test_real_time_data_binding()
        return tester.print_frontend_test_report()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "mobile":
            result = asyncio.run(run_mobile_only_tests())
        elif test_type == "emergency":
            result = asyncio.run(run_emergency_ui_tests())
        else:
            print(f"Unknown test type: {test_type}")
            print("Available types: mobile, emergency")
            sys.exit(1)
    else:
        # Run comprehensive validation
        result = asyncio.run(run_frontend_component_validation())
    
    # Set exit code based on results
    if result and result.get("critical_failures", 1) == 0 and result.get("overall_success_rate", 0) >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure