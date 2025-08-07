"""
Mobile Compatibility & Responsive Design Testing Suite

Tests the mobile dashboard implementation for touch controls, responsive layouts,
and accessibility across various mobile devices. Validates that operators can
effectively manage the 20% coordination crisis from mobile devices in emergency
situations.

CRITICAL CONTEXT: During coordination crises, operators may need to access the
dashboard remotely via mobile devices. The dashboard must provide full crisis
management capabilities with touch-optimized controls and mobile-responsive
layouts.

Test Coverage:
1. Mobile Device Compatibility - Various screen sizes and orientations
2. Touch Interface Validation - Touch controls and gesture support
3. Responsive Layout Testing - Adaptive UI across breakpoints
4. Mobile Performance Testing - Load times and memory usage
5. Offline Functionality Testing - PWA capabilities and offline access
6. Emergency Mobile Controls - Critical functions accessible via touch
"""

import asyncio
import json
import pytest
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import httpx

# Mobile Configuration
FRONTEND_URL = "http://localhost:5173"
BACKEND_URL = "http://localhost:8000"

# Mobile Device Configurations
MOBILE_DEVICES = [
    {
        "name": "iPhone SE",
        "viewport": {"width": 375, "height": 667},
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
        "device_pixel_ratio": 2,
        "has_touch": True,
        "critical": True  # Critical device for testing
    },
    {
        "name": "iPhone 12",
        "viewport": {"width": 390, "height": 844},
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
        "device_pixel_ratio": 3,
        "has_touch": True,
        "critical": True
    },
    {
        "name": "iPhone XR",
        "viewport": {"width": 414, "height": 896},
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
        "device_pixel_ratio": 2,
        "has_touch": True,
        "critical": False
    },
    {
        "name": "iPad",
        "viewport": {"width": 768, "height": 1024},
        "user_agent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
        "device_pixel_ratio": 2,
        "has_touch": True,
        "critical": True
    },
    {
        "name": "iPad Pro",
        "viewport": {"width": 1024, "height": 1366},
        "user_agent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
        "device_pixel_ratio": 2,
        "has_touch": True,
        "critical": False
    },
    {
        "name": "Android Phone",
        "viewport": {"width": 360, "height": 640},
        "user_agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36",
        "device_pixel_ratio": 3,
        "has_touch": True,
        "critical": True
    },
    {
        "name": "Android Tablet",
        "viewport": {"width": 800, "height": 1280},
        "user_agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36",
        "device_pixel_ratio": 2,
        "has_touch": True,
        "critical": False
    }
]

# Performance Thresholds for Mobile
MOBILE_LOAD_THRESHOLD_MS = 3000    # 3 second load time for mobile
TOUCH_RESPONSE_THRESHOLD_MS = 200  # 200ms touch response time
MOBILE_MEMORY_THRESHOLD_MB = 256   # 256MB memory usage for mobile


@dataclass
class MobileTestResult:
    """Mobile test result tracking."""
    device_name: str
    test_name: str
    success: bool
    details: str = ""
    load_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    touch_responsive: bool = True
    critical: bool = False
    orientation: str = "portrait"


@dataclass
class MobileTestSuite:
    """Mobile test suite aggregator."""
    name: str
    results: List[MobileTestResult] = field(default_factory=list)
    
    def add_result(self, result: MobileTestResult):
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
        return statistics.mean(load_times) if load_times else 0.0
    
    @property
    def touch_responsiveness_rate(self) -> float:
        total = len(self.results)
        responsive = len([r for r in self.results if r.touch_responsive])
        return (responsive / total * 100) if total > 0 else 0.0


class MobileDashboardTester:
    """Comprehensive mobile compatibility tester for dashboard."""
    
    def __init__(self, frontend_url: str = FRONTEND_URL, backend_url: str = BACKEND_URL):
        self.frontend_url = frontend_url
        self.backend_url = backend_url
        self.test_suites: Dict[str, MobileTestSuite] = {}
        self.browser: Optional[Browser] = None
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    def get_suite(self, suite_name: str) -> MobileTestSuite:
        """Get or create test suite."""
        if suite_name not in self.test_suites:
            self.test_suites[suite_name] = MobileTestSuite(suite_name)
        return self.test_suites[suite_name]
    
    async def create_mobile_context(self, device: Dict[str, Any]) -> BrowserContext:
        """Create mobile browser context for testing."""
        return await self.browser.new_context(
            viewport=device["viewport"],
            user_agent=device["user_agent"],
            device_scale_factor=device["device_pixel_ratio"],
            has_touch=device["has_touch"],
            is_mobile=True
        )
    
    async def measure_mobile_load_time(self, page: Page) -> float:
        """Measure mobile page load time."""
        start_time = time.perf_counter()
        await page.goto(self.frontend_url, wait_until="networkidle", timeout=15000)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000
    
    async def test_touch_responsiveness(self, page: Page) -> Tuple[bool, str, float]:
        """Test touch responsiveness and interaction."""
        touch_tests = []
        
        try:
            # Test 1: Touch targets are appropriately sized
            touch_target_test = await page.evaluate("""() => {
                const touchElements = document.querySelectorAll('button, .card, [role="button"], a, input');
                const appropriatelySized = Array.from(touchElements).filter(el => {
                    const rect = el.getBoundingClientRect();
                    return rect.width >= 44 && rect.height >= 44; // Apple's 44pt minimum
                });
                return {
                    total: touchElements.length,
                    appropriatelySized: appropriatelySized.length
                };
            }""")
            
            touch_size_rate = (touch_target_test["appropriatelySized"] / touch_target_test["total"]) if touch_target_test["total"] > 0 else 0
            touch_tests.append({
                "name": "Touch Target Sizes",
                "success": touch_size_rate >= 0.8,  # 80% should be appropriately sized
                "details": f"{touch_target_test['appropriatelySized']}/{touch_target_test['total']} appropriate"
            })
            
            # Test 2: Touch interactions work
            touch_interaction_test = True
            touch_response_times = []
            
            # Find clickable elements
            clickable_elements = await page.locator("button, .card, [role='button']").all()
            
            if clickable_elements:
                for i, element in enumerate(clickable_elements[:5]):  # Test first 5 elements
                    try:
                        start_touch = time.perf_counter()
                        await element.tap()
                        end_touch = time.perf_counter()
                        
                        response_time = (end_touch - start_touch) * 1000
                        touch_response_times.append(response_time)
                        
                        if response_time > TOUCH_RESPONSE_THRESHOLD_MS:
                            touch_interaction_test = False
                    except:
                        touch_interaction_test = False
                        break
            
            avg_touch_response = statistics.mean(touch_response_times) if touch_response_times else 0
            
            touch_tests.append({
                "name": "Touch Interactions",
                "success": touch_interaction_test and avg_touch_response <= TOUCH_RESPONSE_THRESHOLD_MS,
                "details": f"Avg response: {avg_touch_response:.1f}ms"
            })
            
            # Test 3: Scroll and swipe gestures
            try:
                # Test vertical scroll
                await page.evaluate("window.scrollTo(0, 100)")
                await asyncio.sleep(0.5)
                scroll_position = await page.evaluate("window.pageYOffset")
                scroll_works = scroll_position > 0
                
                # Test touch scroll back to top
                await page.touch_screen.swipe(400, 400, 400, 600)  # Swipe up
                await asyncio.sleep(0.5)
                
                touch_tests.append({
                    "name": "Scroll and Swipe",
                    "success": scroll_works,
                    "details": "Scroll gestures functional" if scroll_works else "Scroll gestures failed"
                })
            except:
                touch_tests.append({
                    "name": "Scroll and Swipe",
                    "success": False,
                    "details": "Gesture test failed"
                })
            
            # Calculate overall touch responsiveness
            successful_tests = len([test for test in touch_tests if test["success"]])
            total_tests = len(touch_tests)
            touch_success_rate = (successful_tests / total_tests) if total_tests > 0 else 0
            
            overall_success = touch_success_rate >= 0.7  # 70% of touch tests should pass
            
            details = f"Touch tests: {successful_tests}/{total_tests}"
            for test in touch_tests:
                status = "✓" if test["success"] else "✗"
                details += f"; {test['name']}: {status} ({test['details']})"
            
            return overall_success, details, avg_touch_response
            
        except Exception as e:
            return False, f"Touch responsiveness test failed: {str(e)}", 0.0
    
    # ===== CORE MOBILE TESTING =====
    
    async def test_mobile_device_compatibility(self):
        """Test compatibility across various mobile devices."""
        suite = self.get_suite("Mobile Device Compatibility")
        
        for device in MOBILE_DEVICES:
            for orientation in ["portrait", "landscape"]:
                try:
                    # Create mobile context
                    mobile_context = await self.create_mobile_context(device)
                    page = await mobile_context.new_page()
                    
                    # Set orientation
                    if orientation == "landscape":
                        viewport = {"width": device["viewport"]["height"], "height": device["viewport"]["width"]}
                        await page.set_viewport_size(viewport)
                    
                    # Measure load time
                    load_time_ms = await self.measure_mobile_load_time(page)
                    
                    # Test basic functionality
                    dashboard_loads = await page.locator("body").is_visible()
                    
                    # Check for mobile-specific elements
                    mobile_navigation = await page.evaluate("""() => {
                        const mobileNav = document.querySelectorAll('.mobile-navigation, .bottom-navigation, .hamburger-menu');
                        return mobileNav.length > 0;
                    }""")
                    
                    # Check responsive layout
                    layout_responsive = await page.evaluate(f"""() => {{
                        const viewport_width = window.innerWidth;
                        const main_content = document.querySelector('main, .dashboard-main, .content');
                        if (!main_content) return false;
                        
                        const content_width = main_content.getBoundingClientRect().width;
                        return content_width <= viewport_width && content_width > viewport_width * 0.8;
                    }}""")
                    
                    # Test critical dashboard elements visibility
                    critical_elements_visible = await page.evaluate("""() => {
                        const critical = [
                            '.coordination-success, [data-testid*="success-rate"]',
                            '.agent-status, [data-testid*="agent"]',
                            '.emergency-controls, [data-testid*="emergency"]'
                        ];
                        
                        return critical.every(selector => {
                            const elements = document.querySelectorAll(selector);
                            return Array.from(elements).some(el => el.offsetHeight > 0);
                        });
                    }""")
                    
                    success = (
                        dashboard_loads and
                        load_time_ms <= MOBILE_LOAD_THRESHOLD_MS and
                        layout_responsive and
                        critical_elements_visible
                    )
                    
                    details = f"Load: {load_time_ms:.1f}ms, Mobile nav: {mobile_navigation}, Layout: {layout_responsive}, Critical elements: {critical_elements_visible}"
                    
                    if load_time_ms > MOBILE_LOAD_THRESHOLD_MS:
                        details += f" (SLOW - threshold: {MOBILE_LOAD_THRESHOLD_MS}ms)"
                    
                    await mobile_context.close()
                    
                except Exception as e:
                    success = False
                    load_time_ms = 0
                    details = f"Device compatibility test failed: {str(e)}"
                    
                    try:
                        await mobile_context.close()
                    except:
                        pass
                
                result = MobileTestResult(
                    device_name=device["name"],
                    test_name="Device Compatibility",
                    success=success,
                    details=details,
                    load_time_ms=load_time_ms,
                    critical=device["critical"],
                    orientation=orientation
                )
                suite.add_result(result)
    
    async def test_touch_interface_validation(self):
        """Test touch interface controls and responsiveness."""
        suite = self.get_suite("Touch Interface Validation")
        
        # Test on critical mobile devices
        critical_devices = [device for device in MOBILE_DEVICES if device["critical"]]
        
        for device in critical_devices:
            try:
                mobile_context = await self.create_mobile_context(device)
                page = await mobile_context.new_page()
                
                # Load dashboard
                await self.measure_mobile_load_time(page)
                
                # Test touch responsiveness
                touch_responsive, touch_details, avg_response_time = await self.test_touch_responsiveness(page)
                
                # Test emergency control touch accessibility
                emergency_touch_test = await page.evaluate("""() => {
                    const emergency_controls = document.querySelectorAll('[data-testid*="emergency"], .emergency-controls button');
                    const touch_friendly = Array.from(emergency_controls).filter(btn => {
                        const rect = btn.getBoundingClientRect();
                        return rect.width >= 44 && rect.height >= 44 && !btn.disabled;
                    });
                    
                    return {
                        total: emergency_controls.length,
                        touch_friendly: touch_friendly.length
                    };
                }""")
                
                emergency_touch_rate = (emergency_touch_test["touch_friendly"] / emergency_touch_test["total"]) if emergency_touch_test["total"] > 0 else 1
                
                # Test drag and drop functionality (if present)
                drag_drop_works = True
                try:
                    # Look for draggable elements
                    draggable_elements = await page.locator("[draggable='true'], .draggable").count()
                    if draggable_elements > 0:
                        # Test drag and drop on mobile
                        draggable = page.locator("[draggable='true'], .draggable").first
                        await draggable.drag_to(page.locator("body"))
                except:
                    drag_drop_works = False
                
                success = (
                    touch_responsive and
                    emergency_touch_rate >= 0.9 and  # 90% of emergency controls should be touch-friendly
                    avg_response_time <= TOUCH_RESPONSE_THRESHOLD_MS
                )
                
                details = f"Touch: {touch_details}, Emergency controls: {emergency_touch_test['touch_friendly']}/{emergency_touch_test['total']} touch-friendly"
                if drag_drop_works:
                    details += ", Drag/drop: ✓"
                
                await mobile_context.close()
                
            except Exception as e:
                success = False
                details = f"Touch interface test failed: {str(e)}"
                avg_response_time = 0
                touch_responsive = False
                
                try:
                    await mobile_context.close()
                except:
                    pass
            
            result = MobileTestResult(
                device_name=device["name"],
                test_name="Touch Interface Validation",
                success=success,
                details=details,
                touch_responsive=touch_responsive,
                critical=True
            )
            suite.add_result(result)
    
    async def test_responsive_breakpoints(self):
        """Test responsive design across different screen breakpoints."""
        suite = self.get_suite("Responsive Layout Testing")
        
        # Define breakpoints to test
        breakpoints = [
            {"name": "Small Mobile", "width": 320, "critical": True},
            {"name": "Mobile", "width": 375, "critical": True},
            {"name": "Large Mobile", "width": 414, "critical": True},
            {"name": "Small Tablet", "width": 768, "critical": True},
            {"name": "Tablet", "width": 1024, "critical": False},
            {"name": "Desktop", "width": 1200, "critical": False}
        ]
        
        for breakpoint in breakpoints:
            try:
                context = await self.browser.new_context(
                    viewport={"width": breakpoint["width"], "height": 800},
                    is_mobile=breakpoint["width"] < 768
                )
                page = await context.new_page()
                
                load_time_ms = await self.measure_mobile_load_time(page)
                
                # Test layout adaptation
                layout_tests = []
                
                # Test 1: Navigation adapts to screen size
                navigation_adapted = await page.evaluate(f"""() => {{
                    const width = {breakpoint["width"]};
                    const nav = document.querySelector('nav, .navigation, .header-nav');
                    if (!nav) return true; // No navigation to test
                    
                    if (width < 768) {{
                        // Mobile should have hamburger or bottom nav
                        const mobile_nav = document.querySelectorAll('.hamburger, .mobile-menu, .bottom-navigation');
                        return mobile_nav.length > 0;
                    }} else {{
                        // Desktop should have full navigation
                        const nav_items = nav.querySelectorAll('a, button');
                        return nav_items.length >= 3; // At least 3 nav items visible
                    }}
                }}""")
                layout_tests.append({"name": "Navigation", "success": navigation_adapted})
                
                # Test 2: Dashboard grid adapts
                grid_adapted = await page.evaluate(f"""() => {{
                    const width = {breakpoint["width"]};
                    const grid_containers = document.querySelectorAll('.dashboard-grid, .grid, .cards-container');
                    if (grid_containers.length === 0) return true;
                    
                    const grid = grid_containers[0];
                    const styles = window.getComputedStyle(grid);
                    
                    if (width < 768) {{
                        // Mobile should stack items (single column)
                        return styles.flexDirection === 'column' || 
                               styles.gridTemplateColumns.includes('1fr') ||
                               !styles.gridTemplateColumns.includes('repeat');
                    }} else {{
                        // Desktop should have multiple columns
                        return styles.gridTemplateColumns.includes('repeat') ||
                               styles.gridTemplateColumns.split(' ').length > 1;
                    }}
                }}""")
                layout_tests.append({"name": "Grid Layout", "success": grid_adapted})
                
                # Test 3: Typography scales appropriately
                typography_scaled = await page.evaluate(f"""() => {{
                    const width = {breakpoint["width"]};
                    const headings = document.querySelectorAll('h1, h2, h3');
                    if (headings.length === 0) return true;
                    
                    const heading = headings[0];
                    const fontSize = parseFloat(window.getComputedStyle(heading).fontSize);
                    
                    if (width < 375) {{
                        return fontSize >= 16 && fontSize <= 24; // Small mobile
                    }} else if (width < 768) {{
                        return fontSize >= 18 && fontSize <= 28; // Mobile
                    }} else {{
                        return fontSize >= 20 && fontSize <= 36; // Desktop
                    }}
                }}""")
                layout_tests.append({"name": "Typography", "success": typography_scaled})
                
                # Test 4: Content doesn't overflow
                no_overflow = await page.evaluate("""() => {
                    const body = document.body;
                    return body.scrollWidth <= window.innerWidth + 20; // 20px tolerance
                }""")
                layout_tests.append({"name": "No Horizontal Overflow", "success": no_overflow})
                
                # Test 5: Critical elements remain visible
                critical_visible = await page.evaluate("""() => {
                    const critical_selectors = [
                        '[data-testid*="success-rate"], .success-rate',
                        '[data-testid*="emergency"], .emergency-controls',
                        '.agent-status, [data-testid*="agent"]'
                    ];
                    
                    return critical_selectors.every(selector => {
                        const elements = document.querySelectorAll(selector);
                        return elements.length === 0 || Array.from(elements).some(el => el.offsetHeight > 0);
                    });
                }""")
                layout_tests.append({"name": "Critical Elements Visible", "success": critical_visible})
                
                # Calculate layout success rate
                successful_layouts = len([test for test in layout_tests if test["success"]])
                total_layouts = len(layout_tests)
                layout_success_rate = (successful_layouts / total_layouts) if total_layouts > 0 else 0
                
                success = layout_success_rate >= 0.8 and load_time_ms <= MOBILE_LOAD_THRESHOLD_MS
                
                details = f"Layout tests: {successful_layouts}/{total_layouts} ({layout_success_rate*100:.1f}%), Load: {load_time_ms:.1f}ms"
                for test in layout_tests:
                    status = "✓" if test["success"] else "✗"
                    details += f"; {test['name']}: {status}"
                
                await context.close()
                
            except Exception as e:
                success = False
                load_time_ms = 0
                details = f"Responsive breakpoint test failed: {str(e)}"
                
                try:
                    await context.close()
                except:
                    pass
            
            result = MobileTestResult(
                device_name=f"Breakpoint {breakpoint['width']}px",
                test_name="Responsive Layout",
                success=success,
                details=details,
                load_time_ms=load_time_ms,
                critical=breakpoint["critical"]
            )
            suite.add_result(result)
    
    async def test_mobile_performance(self):
        """Test mobile performance including memory usage and load times."""
        suite = self.get_suite("Mobile Performance Testing")
        
        performance_devices = [device for device in MOBILE_DEVICES if device["critical"]]
        
        for device in performance_devices:
            try:
                mobile_context = await self.create_mobile_context(device)
                page = await mobile_context.new_page()
                
                # Enable performance monitoring
                await page.route("**/*", lambda route: route.continue_())
                
                # Measure initial load
                start_time = time.perf_counter()
                await page.goto(self.frontend_url, wait_until="domcontentloaded")
                dom_load_time = (time.perf_counter() - start_time) * 1000
                
                await page.wait_for_load_state("networkidle")
                full_load_time = (time.perf_counter() - start_time) * 1000
                
                # Measure runtime performance
                performance_metrics = await page.evaluate("""() => {
                    const performance = window.performance;
                    const navigation = performance.getEntriesByType('navigation')[0];
                    
                    return {
                        dom_content_loaded: navigation ? navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart : 0,
                        dom_interactive: navigation ? navigation.domInteractive - navigation.fetchStart : 0,
                        load_complete: navigation ? navigation.loadEventEnd - navigation.fetchStart : 0,
                        first_paint: performance.getEntriesByType('paint').find(p => p.name === 'first-paint')?.startTime || 0,
                        first_contentful_paint: performance.getEntriesByType('paint').find(p => p.name === 'first-contentful-paint')?.startTime || 0
                    };
                }""")
                
                # Test memory usage (simulated)
                memory_usage_mb = 0
                try:
                    # Simulate memory measurement
                    js_heap_size = await page.evaluate("performance.memory ? performance.memory.usedJSHeapSize : 0")
                    memory_usage_mb = js_heap_size / 1024 / 1024 if js_heap_size > 0 else 128  # Fallback estimate
                except:
                    memory_usage_mb = 128  # Default estimate for mobile
                
                # Test scrolling performance
                scrolling_smooth = True
                try:
                    start_scroll = time.perf_counter()
                    for _ in range(5):
                        await page.evaluate("window.scrollBy(0, 100)")
                        await asyncio.sleep(0.1)
                    scroll_time = (time.perf_counter() - start_scroll) * 1000
                    scrolling_smooth = scroll_time < 1000  # Should complete in under 1 second
                except:
                    scrolling_smooth = False
                
                # Test interaction responsiveness
                interaction_responsive = True
                try:
                    button = page.locator("button").first
                    if await button.count() > 0:
                        start_click = time.perf_counter()
                        await button.click()
                        click_time = (time.perf_counter() - start_click) * 1000
                        interaction_responsive = click_time < TOUCH_RESPONSE_THRESHOLD_MS
                except:
                    interaction_responsive = False
                
                # Performance assessment
                performance_score = 0
                performance_criteria = [
                    ("Load Time", full_load_time <= MOBILE_LOAD_THRESHOLD_MS),
                    ("DOM Load", dom_load_time <= MOBILE_LOAD_THRESHOLD_MS * 0.7),
                    ("Memory Usage", memory_usage_mb <= MOBILE_MEMORY_THRESHOLD_MB),
                    ("Smooth Scrolling", scrolling_smooth),
                    ("Touch Response", interaction_responsive)
                ]
                
                performance_score = len([criteria for criteria, passed in performance_criteria if passed]) / len(performance_criteria)
                
                success = performance_score >= 0.8  # 80% of performance criteria should pass
                
                details = f"Load: {full_load_time:.1f}ms, DOM: {dom_load_time:.1f}ms, Memory: {memory_usage_mb:.1f}MB, Score: {performance_score*100:.1f}%"
                for criteria, passed in performance_criteria:
                    status = "✓" if passed else "✗"
                    details += f"; {criteria}: {status}"
                
                await mobile_context.close()
                
            except Exception as e:
                success = False
                full_load_time = 0
                memory_usage_mb = 0
                details = f"Mobile performance test failed: {str(e)}"
                
                try:
                    await mobile_context.close()
                except:
                    pass
            
            result = MobileTestResult(
                device_name=device["name"],
                test_name="Mobile Performance",
                success=success,
                details=details,
                load_time_ms=full_load_time,
                memory_usage_mb=memory_usage_mb,
                critical=True
            )
            suite.add_result(result)
    
    async def test_pwa_offline_functionality(self):
        """Test PWA capabilities and offline functionality."""
        suite = self.get_suite("PWA & Offline Functionality")
        
        try:
            # Test with primary mobile device
            device = MOBILE_DEVICES[0]  # iPhone SE
            mobile_context = await self.create_mobile_context(device)
            page = await mobile_context.new_page()
            
            # Load dashboard initially
            await self.measure_mobile_load_time(page)
            
            # Test PWA manifest
            manifest_valid = await page.evaluate("""() => {
                const manifest_link = document.querySelector('link[rel="manifest"]');
                return manifest_link && manifest_link.href.length > 0;
            }""")
            
            # Test service worker registration
            service_worker_registered = False
            try:
                service_worker_registered = await page.evaluate("""async () => {
                    if ('serviceWorker' in navigator) {
                        const registration = await navigator.serviceWorker.getRegistration();
                        return registration !== undefined;
                    }
                    return false;
                }""")
            except:
                service_worker_registered = False
            
            # Test offline capability (simulate network failure)
            offline_works = False
            try:
                # Go offline
                await mobile_context.set_offline(True)
                
                # Try to navigate or reload
                await page.reload(wait_until="domcontentloaded", timeout=5000)
                
                # Check if offline page or cached content loads
                offline_content_available = await page.evaluate("""() => {
                    return document.body.textContent.length > 100; // Some meaningful content available
                }""")
                
                offline_works = offline_content_available
                
                # Go back online
                await mobile_context.set_offline(False)
            except:
                offline_works = False
                await mobile_context.set_offline(False)
            
            # Test app-like behavior
            app_like_behavior = await page.evaluate("""() => {
                const checks = [];
                
                // Check for app-like styling (no browser UI indicators)
                checks.push(!document.querySelector('body').style.includes('browser'));
                
                // Check for mobile-first design
                const viewport_meta = document.querySelector('meta[name="viewport"]');
                checks.push(viewport_meta && viewport_meta.content.includes('width=device-width'));
                
                // Check for touch-friendly interface
                const touch_elements = document.querySelectorAll('button, [role="button"]');
                const large_touch_targets = Array.from(touch_elements).filter(el => {
                    const rect = el.getBoundingClientRect();
                    return rect.width >= 44 && rect.height >= 44;
                });
                checks.push(large_touch_targets.length >= touch_elements.length * 0.7);
                
                return checks.filter(check => check).length / checks.length;
            }""")
            
            # Test installability hints
            install_prompt_available = await page.evaluate("""() => {
                return window.deferredPrompt !== undefined || 
                       document.querySelector('[data-testid*="install"], .install-prompt, .add-to-home') !== null;
            }""")
            
            # PWA assessment
            pwa_criteria = [
                ("Manifest Valid", manifest_valid),
                ("Service Worker", service_worker_registered),
                ("Offline Functionality", offline_works),
                ("App-like Behavior", app_like_behavior >= 0.7),
                ("Install Prompt", install_prompt_available)
            ]
            
            pwa_score = len([criteria for criteria, passed in pwa_criteria if passed]) / len(pwa_criteria)
            success = pwa_score >= 0.6  # 60% of PWA features should work
            
            details = f"PWA score: {pwa_score*100:.1f}%"
            for criteria, passed in pwa_criteria:
                status = "✓" if passed else "✗"
                details += f"; {criteria}: {status}"
            
            await mobile_context.close()
            
        except Exception as e:
            success = False
            details = f"PWA functionality test failed: {str(e)}"
            
            try:
                await mobile_context.close()
            except:
                pass
        
        result = MobileTestResult(
            device_name="PWA Test Device",
            test_name="PWA & Offline Functionality",
            success=success,
            details=details,
            critical=False  # PWA is enhancement, not critical
        )
        suite.add_result(result)
    
    async def test_emergency_mobile_controls(self):
        """Test emergency controls accessibility and functionality on mobile."""
        suite = self.get_suite("Emergency Mobile Controls")
        
        critical_devices = [device for device in MOBILE_DEVICES if device["critical"]]
        
        for device in critical_devices:
            try:
                mobile_context = await self.create_mobile_context(device)
                page = await mobile_context.new_page()
                
                # Load dashboard
                await self.measure_mobile_load_time(page)
                
                # Test emergency control accessibility
                emergency_tests = []
                
                # Test 1: Emergency controls are visible and accessible
                emergency_visibility = await page.evaluate("""() => {
                    const emergency_controls = document.querySelectorAll('[data-testid*="emergency"], .emergency-controls, [data-testid*="crisis"]');
                    const visible_controls = Array.from(emergency_controls).filter(el => el.offsetHeight > 0 && el.offsetWidth > 0);
                    
                    return {
                        total: emergency_controls.length,
                        visible: visible_controls.length
                    };
                }""")
                
                emergency_visible_rate = (emergency_visibility["visible"] / emergency_visibility["total"]) if emergency_visibility["total"] > 0 else 0
                emergency_tests.append({
                    "name": "Emergency Visibility",
                    "success": emergency_visible_rate >= 0.8,
                    "details": f"{emergency_visibility['visible']}/{emergency_visibility['total']} visible"
                })
                
                # Test 2: Emergency controls are touch-friendly
                emergency_touch_friendly = await page.evaluate("""() => {
                    const emergency_buttons = document.querySelectorAll('[data-testid*="emergency"] button, .emergency-controls button, [data-testid*="reset"]');
                    const touch_friendly = Array.from(emergency_buttons).filter(btn => {
                        const rect = btn.getBoundingClientRect();
                        return rect.width >= 44 && rect.height >= 44; // Minimum touch target size
                    });
                    
                    return {
                        total: emergency_buttons.length,
                        touch_friendly: touch_friendly.length
                    };
                }""")
                
                touch_friendly_rate = (emergency_touch_friendly["touch_friendly"] / emergency_touch_friendly["total"]) if emergency_touch_friendly["total"] > 0 else 1
                emergency_tests.append({
                    "name": "Touch-Friendly Size",
                    "success": touch_friendly_rate >= 0.9,
                    "details": f"{emergency_touch_friendly['touch_friendly']}/{emergency_touch_friendly['total']} appropriate size"
                })
                
                # Test 3: Emergency controls respond to touch
                emergency_responsive = True
                try:
                    emergency_buttons = await page.locator("[data-testid*='emergency'] button, .emergency-controls button").all()
                    if emergency_buttons:
                        start_touch = time.perf_counter()
                        await emergency_buttons[0].tap()
                        touch_time = (time.perf_counter() - start_touch) * 1000
                        emergency_responsive = touch_time <= TOUCH_RESPONSE_THRESHOLD_MS
                        
                        # Check for confirmation dialog (cancel it)
                        try:
                            await page.wait_for_selector(".confirmation-dialog, .modal", timeout=2000)
                            cancel_btn = page.locator("[data-testid='cancel-btn'], .cancel-btn, button:has-text('Cancel')")
                            if await cancel_btn.count() > 0:
                                await cancel_btn.tap()
                        except:
                            pass  # No confirmation dialog
                except:
                    emergency_responsive = False
                
                emergency_tests.append({
                    "name": "Touch Responsiveness",
                    "success": emergency_responsive,
                    "details": "Emergency controls respond to touch" if emergency_responsive else "Unresponsive"
                })
                
                # Test 4: Critical information remains visible
                critical_info_mobile = await page.evaluate("""() => {
                    const critical_selectors = [
                        '[data-testid*="success-rate"], .success-rate',
                        '[data-testid*="health"], .system-health',
                        '.coordination-status, [data-testid*="coordination"]'
                    ];
                    
                    const visible_info = critical_selectors.filter(selector => {
                        const elements = document.querySelectorAll(selector);
                        return Array.from(elements).some(el => el.offsetHeight > 0);
                    });
                    
                    return {
                        total: critical_selectors.length,
                        visible: visible_info.length
                    };
                }""")
                
                critical_info_rate = (critical_info_mobile["visible"] / critical_info_mobile["total"]) if critical_info_mobile["total"] > 0 else 0
                emergency_tests.append({
                    "name": "Critical Information Visible",
                    "success": critical_info_rate >= 0.8,
                    "details": f"{critical_info_mobile['visible']}/{critical_info_mobile['total']} info types visible"
                })
                
                # Calculate overall emergency controls success
                successful_emergency_tests = len([test for test in emergency_tests if test["success"]])
                total_emergency_tests = len(emergency_tests)
                emergency_success_rate = (successful_emergency_tests / total_emergency_tests) if total_emergency_tests > 0 else 0
                
                success = emergency_success_rate >= 0.8  # 80% of emergency tests should pass
                
                details = f"Emergency tests: {successful_emergency_tests}/{total_emergency_tests}"
                for test in emergency_tests:
                    status = "✓" if test["success"] else "✗"
                    details += f"; {test['name']}: {status} ({test['details']})"
                
                await mobile_context.close()
                
            except Exception as e:
                success = False
                details = f"Emergency mobile controls test failed: {str(e)}"
                
                try:
                    await mobile_context.close()
                except:
                    pass
            
            result = MobileTestResult(
                device_name=device["name"],
                test_name="Emergency Mobile Controls",
                success=success,
                details=details,
                critical=True
            )
            suite.add_result(result)
    
    # ===== COMPREHENSIVE TEST EXECUTION =====
    
    async def run_all_mobile_tests(self) -> Dict[str, MobileTestSuite]:
        """Run all comprehensive mobile tests."""
        print("📱 Starting Mobile Compatibility & Responsive Design Testing")
        print("="*60)
        print(f"Testing {len(MOBILE_DEVICES)} mobile devices")
        print(f"Thresholds: Load <{MOBILE_LOAD_THRESHOLD_MS}ms, Touch <{TOUCH_RESPONSE_THRESHOLD_MS}ms, Memory <{MOBILE_MEMORY_THRESHOLD_MB}MB")
        print("="*60)
        
        # Test 1: Mobile Device Compatibility
        print("\n📱 Testing Mobile Device Compatibility...")
        await self.test_mobile_device_compatibility()
        
        # Test 2: Touch Interface Validation
        print("\n👆 Testing Touch Interface Validation...")
        await self.test_touch_interface_validation()
        
        # Test 3: Responsive Layout Testing
        print("\n📐 Testing Responsive Layout Breakpoints...")
        await self.test_responsive_breakpoints()
        
        # Test 4: Mobile Performance Testing
        print("\n⚡ Testing Mobile Performance...")
        await self.test_mobile_performance()
        
        # Test 5: PWA & Offline Functionality
        print("\n🔄 Testing PWA & Offline Functionality...")
        await self.test_pwa_offline_functionality()
        
        # Test 6: Emergency Mobile Controls
        print("\n🆘 Testing Emergency Mobile Controls...")
        await self.test_emergency_mobile_controls()
        
        return self.test_suites
    
    def print_mobile_test_report(self):
        """Print comprehensive mobile test report."""
        print("\n" + "="*70)
        print("MOBILE COMPATIBILITY & RESPONSIVE DESIGN REPORT")
        print("="*70)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_critical_failures = 0
        
        for suite_name, suite in self.test_suites.items():
            print(f"\n📱 {suite.name}")
            print(f"   Tests: {len(suite.results)} | Passed: {suite.passed} | Failed: {suite.failed}")
            print(f"   Success Rate: {suite.success_rate:.1f}%")
            
            if hasattr(suite, 'average_load_time') and suite.average_load_time > 0:
                print(f"   Avg Load Time: {suite.average_load_time:.1f}ms")
            if hasattr(suite, 'touch_responsiveness_rate') and suite.touch_responsiveness_rate > 0:
                print(f"   Touch Responsiveness: {suite.touch_responsiveness_rate:.1f}%")
            
            if suite.critical_failures > 0:
                print(f"   🔴 CRITICAL FAILURES: {suite.critical_failures}")
            
            # Show device/breakpoint breakdown
            devices = set(r.device_name for r in suite.results)
            if devices:
                print(f"   Devices/Breakpoints: {', '.join(devices)}")
            
            # Show failed tests
            failed_tests = [r for r in suite.results if not r.success]
            if failed_tests:
                print("   Failed Tests:")
                for test in failed_tests[:3]:  # Show first 3 failures
                    prefix = "🔴" if test.critical else "❌"
                    device_info = f" ({test.device_name})" if test.device_name else ""
                    print(f"     {prefix} {test.test_name}{device_info}: {test.details[:50]}...")
                if len(failed_tests) > 3:
                    print(f"     ... and {len(failed_tests) - 3} more")
            
            total_tests += len(suite.results)
            total_passed += suite.passed
            total_failed += suite.failed
            total_critical_failures += suite.critical_failures
        
        # Overall summary
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n{'='*70}")
        print("MOBILE SYSTEM STATUS")
        print(f"{'='*70}")
        print(f"Total Mobile Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {overall_success_rate:.1f}%")
        print(f"Critical Failures: {total_critical_failures}")
        
        # Mobile readiness assessment
        print(f"\n🎯 MOBILE PRODUCTION READINESS:")
        
        if total_critical_failures == 0 and overall_success_rate >= 85:
            print("🟢 MOBILE READY - Dashboard fully functional on mobile devices")
            status = "MOBILE_READY"
        elif total_critical_failures <= 2 and overall_success_rate >= 75:
            print("🟡 MOSTLY MOBILE READY - Dashboard functional with minor mobile issues")
            status = "MOSTLY_READY"
        elif total_critical_failures <= 5 and overall_success_rate >= 65:
            print("🟠 NEEDS MOBILE OPTIMIZATION - Some mobile compatibility issues")
            status = "NEEDS_OPTIMIZATION"
        else:
            print("🔴 NOT MOBILE READY - Critical mobile functionality issues")
            status = "NOT_READY"
        
        print(f"\n📊 MOBILE CRISIS MANAGEMENT READINESS:")
        
        # Check mobile crisis management specific results
        emergency_suite = self.test_suites.get("Emergency Mobile Controls")
        touch_suite = self.test_suites.get("Touch Interface Validation")
        performance_suite = self.test_suites.get("Mobile Performance Testing")
        
        mobile_crisis_ready = True
        if emergency_suite and emergency_suite.critical_failures > 0:
            mobile_crisis_ready = False
        if touch_suite and touch_suite.critical_failures > 0:
            mobile_crisis_ready = False
        if performance_suite and performance_suite.critical_failures > 2:
            mobile_crisis_ready = False
        
        if mobile_crisis_ready:
            print("✅ Mobile dashboard ready for coordination crisis management")
            print("   - Emergency controls accessible via touch")
            print("   - Critical information visible on mobile screens")
            print("   - Performance adequate for crisis response")
        else:
            print("❌ Mobile dashboard not ready for crisis management")
            print("   - Emergency controls may not be accessible")
            print("   - Performance or functionality issues present")
        
        print(f"{'='*70}")
        
        return {
            "overall_success_rate": overall_success_rate,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "critical_failures": total_critical_failures,
            "mobile_readiness_status": status,
            "mobile_crisis_ready": mobile_crisis_ready,
            "suites": {name: {
                "success_rate": suite.success_rate,
                "tests": len(suite.results),
                "passed": suite.passed,
                "failed": suite.failed,
                "critical_failures": suite.critical_failures,
                "average_load_time": getattr(suite, 'average_load_time', 0),
                "touch_responsiveness": getattr(suite, 'touch_responsiveness_rate', 0)
            } for name, suite in self.test_suites.items()}
        }


# ===== TEST EXECUTION FUNCTIONS =====

async def run_mobile_compatibility_validation():
    """Run comprehensive mobile compatibility validation."""
    async with MobileDashboardTester() as tester:
        await tester.run_all_mobile_tests()
        return tester.print_mobile_test_report()


async def run_touch_interface_only():
    """Run only touch interface validation tests."""
    print("👆 Running Touch Interface Tests Only")
    print("="*35)
    
    async with MobileDashboardTester() as tester:
        await tester.test_touch_interface_validation()
        await tester.test_emergency_mobile_controls()
        return tester.print_mobile_test_report()


async def run_responsive_design_only():
    """Run only responsive design tests."""
    print("📐 Running Responsive Design Tests Only")
    print("="*35)
    
    async with MobileDashboardTester() as tester:
        await tester.test_responsive_breakpoints()
        return tester.print_mobile_test_report()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "touch":
            result = asyncio.run(run_touch_interface_only())
        elif test_type == "responsive":
            result = asyncio.run(run_responsive_design_only())
        else:
            print(f"Unknown test type: {test_type}")
            print("Available types: touch, responsive")
            sys.exit(1)
    else:
        # Run comprehensive validation
        result = asyncio.run(run_mobile_compatibility_validation())
    
    # Set exit code based on results
    if (result and result.get("critical_failures", 1) <= 3 and 
        result.get("overall_success_rate", 0) >= 70 and
        result.get("mobile_crisis_ready", False)):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure