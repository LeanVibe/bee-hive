/**
 * Router Debug Test - Check router initialization and navigation
 */

import { test, expect } from '@playwright/test';

test.describe('Router Debug', () => {
  test('Check router initialization and navigation flow', async ({ page }) => {
    const consoleLogs: string[] = [];
    const errors: string[] = [];
    
    // Capture all console messages
    page.on('console', msg => {
      const text = msg.text();
      consoleLogs.push(text);
      console.log(`PWA Console [${msg.type()}]:`, text);
    });
    
    // Capture errors
    page.on('pageerror', error => {
      const errorText = error.message;
      errors.push(errorText);
      console.log('PWA Error:', errorText);
    });

    console.log('🔍 Loading PWA to debug router...');
    await page.goto('http://localhost:3002/', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait for initialization
    await page.waitForTimeout(5000);
    
    // Check if router exists in window
    const routerExists = await page.evaluate(() => {
      return window.router !== undefined;
    });
    
    console.log('Router exists in window:', routerExists);
    
    // Check router state by calling debug methods
    const routerDebugInfo = await page.evaluate(() => {
      const app = document.querySelector('agent-hive-app');
      if (app && app.router) {
        return {
          isStarted: app.router.isStarted,
          currentRoute: app.router.getCurrentRoute(),
          registeredRoutes: app.router.getRegisteredRoutes(),
          debugInfo: app.router.debugInfo()
        };
      }
      return null;
    });
    
    console.log('Router debug info:', JSON.stringify(routerDebugInfo, null, 2));
    
    // Look for specific router logs in console
    const routerLogs = consoleLogs.filter(log => 
      log.includes('Router') || 
      log.includes('router') ||
      log.includes('🛣️') ||
      log.includes('Navigating') ||
      log.includes('started')
    );
    
    console.log('\n🛣️ All Router-related Logs:');
    routerLogs.forEach(log => console.log('  -', log));
    
    // Check app initialization logs
    const initLogs = consoleLogs.filter(log =>
      log.includes('App initialization') ||
      log.includes('initializeApp') ||
      log.includes('Starting') ||
      log.includes('Started')
    );
    
    console.log('\n🚀 App Initialization Logs:');
    initLogs.forEach(log => console.log('  -', log));
    
    // Check current DOM state
    const appHTML = await page.locator('#app').innerHTML();
    const currentURL = page.url();
    
    console.log('\n🏗️ Final State:');
    console.log('  - Current URL:', currentURL);
    console.log('  - App HTML length:', appHTML.length);
    console.log('  - Router started:', routerDebugInfo?.isStarted);
    console.log('  - Current route:', routerDebugInfo?.currentRoute);
    
    if (appHTML.length < 200) {
      console.log('  - App HTML:', appHTML);
    }
    
    // Take screenshot
    await page.screenshot({ path: 'test-results/router-debug-screenshot.png', fullPage: true });
    
    // Basic assertions
    expect(errors.length).toBe(0);
    expect(routerDebugInfo).not.toBeNull();
  });
});