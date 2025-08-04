/**
 * Dashboard Visibility Debug Test
 */

import { test, expect } from '@playwright/test';

test.describe('Dashboard Visibility Debug', () => {
  test('Debug why dashboard-view is hidden', async ({ page }) => {
    const consoleLogs: string[] = [];
    const errors: string[] = [];
    
    page.on('console', msg => {
      const text = msg.text();
      consoleLogs.push(text);
      if (msg.type() === 'error') {
        console.log(`ERROR:`, text);
      }
    });
    
    page.on('pageerror', error => {
      errors.push(error.message);
      console.log('JS ERROR:', error.message);
    });

    console.log('🔍 Loading PWA to debug dashboard visibility...');
    await page.goto('http://localhost:3002/', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    await page.waitForTimeout(3000);
    
    // Check if dashboard-view element exists
    const dashboardExists = await page.locator('dashboard-view').count();
    console.log('Dashboard-view element count:', dashboardExists);
    
    if (dashboardExists > 0) {
      // Get detailed info about the element
      const dashboardInfo = await page.evaluate(() => {
        const element = document.querySelector('dashboard-view');
        if (!element) return null;
        
        const styles = window.getComputedStyle(element);
        return {
          display: styles.display,
          visibility: styles.visibility,
          opacity: styles.opacity,
          width: styles.width,
          height: styles.height,
          position: styles.position,
          innerHTML: element.innerHTML.length,
          hasContent: element.children.length > 0,
          isConnected: element.isConnected,
          shadowRoot: element.shadowRoot !== null,
          classList: Array.from(element.classList),
          offsetWidth: element.offsetWidth,
          offsetHeight: element.offsetHeight
        };
      });
      
      console.log('\n📐 Dashboard Element Analysis:');
      console.log('  - Display:', dashboardInfo?.display);
      console.log('  - Visibility:', dashboardInfo?.visibility);
      console.log('  - Opacity:', dashboardInfo?.opacity);
      console.log('  - Width:', dashboardInfo?.width);
      console.log('  - Height:', dashboardInfo?.height);
      console.log('  - Offset Width:', dashboardInfo?.offsetWidth);
      console.log('  - Offset Height:', dashboardInfo?.offsetHeight);
      console.log('  - Has shadow root:', dashboardInfo?.shadowRoot);
      console.log('  - Inner HTML length:', dashboardInfo?.innerHTML);
      console.log('  - Has children:', dashboardInfo?.hasContent);
      console.log('  - Is connected:', dashboardInfo?.isConnected);
      console.log('  - CSS Classes:', dashboardInfo?.classList);
      
      // Check if element is in viewport
      const isInViewport = await page.locator('dashboard-view').isInViewport();
      console.log('  - Is in viewport:', isInViewport);
      
      // Check parent containers
      const parentInfo = await page.evaluate(() => {
        const element = document.querySelector('dashboard-view');
        if (!element || !element.parentElement) return null;
        
        const parent = element.parentElement;
        const styles = window.getComputedStyle(parent);
        return {
          tagName: parent.tagName,
          display: styles.display,
          visibility: styles.visibility,
          opacity: styles.opacity,
          width: styles.width,
          height: styles.height,
          overflow: styles.overflow
        };
      });
      
      console.log('\n🏠 Parent Container Analysis:');
      console.log('  - Tag:', parentInfo?.tagName);
      console.log('  - Display:', parentInfo?.display);
      console.log('  - Visibility:', parentInfo?.visibility);
      console.log('  - Opacity:', parentInfo?.opacity);
      console.log('  - Width:', parentInfo?.width);
      console.log('  - Height:', parentInfo?.height);
      console.log('  - Overflow:', parentInfo?.overflow);
    }
    
    // Look for specific errors that might explain the issue
    const initErrors = errors.filter(error => 
      error.includes('addEventListener') ||
      error.includes('DashboardView') ||
      error.includes('initialization') ||
      error.includes('constructor')
    );
    
    console.log('\n❌ Initialization Errors:');
    if (initErrors.length > 0) {
      initErrors.forEach(error => console.log('  -', error));
    } else {
      console.log('  - No initialization errors found');
    }
    
    // Check if services are properly initialized
    const serviceStatus = await page.evaluate(() => {
      const services = (window as any).appServices;
      return {
        hasServices: services !== undefined,
        hasOffline: services?.offline !== undefined,
        hasAuth: services?.auth !== undefined,
        hasBackendAdapter: services?.backendAdapter !== undefined
      };
    });
    
    console.log('\n🔧 Service Status:');
    console.log('  - Has app services:', serviceStatus.hasServices);
    console.log('  - Has offline service:', serviceStatus.hasOffline);
    console.log('  - Has auth service:', serviceStatus.hasAuth);
    console.log('  - Has backend adapter:', serviceStatus.hasBackendAdapter);
    
    // Take screenshot for visual analysis
    await page.screenshot({ 
      path: 'test-results/dashboard-visibility-debug.png', 
      fullPage: true 
    });
    
    // The test should pass, we're just debugging
    expect(dashboardExists).toBeGreaterThan(0);
  });
});