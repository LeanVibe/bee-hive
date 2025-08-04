/**
 * PWA Debug Test - Check console logs and app state
 */

import { test, expect } from '@playwright/test';

test.describe('PWA Debug', () => {
  test('Check PWA console logs and app state', async ({ page }) => {
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

    console.log('🔍 Loading PWA and checking logs...');
    await page.goto('http://localhost:3002/', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait for initialization
    await page.waitForTimeout(8000);
    
    console.log('\n📋 Console Logs Summary:');
    console.log(`Total logs: ${consoleLogs.length}`);
    console.log(`Total errors: ${errors.length}`);
    
    // Look for specific authentication logs
    const authLogs = consoleLogs.filter(log => 
      log.includes('Development mode') || 
      log.includes('authentication') ||
      log.includes('authenticated')
    );
    
    console.log('\n🔐 Authentication Logs:');
    authLogs.forEach(log => console.log('  -', log));
    
    // Look for backend adapter logs
    const backendLogs = consoleLogs.filter(log => 
      log.includes('Backend adapter') ||
      log.includes('Syncing data') ||
      log.includes('live data')
    );
    
    // Look for router logs
    const routerLogs = consoleLogs.filter(log => 
      log.includes('Router started') ||
      log.includes('Navigating to') ||
      log.includes('🛣️')
    );
    
    console.log('\n🔄 Backend Adapter Logs:');
    backendLogs.forEach(log => console.log('  -', log));
    
    console.log('\n🛣️ Router Logs:');
    routerLogs.forEach(log => console.log('  -', log));
    
    // Check DOM state
    const appElement = await page.locator('#app');
    const appExists = await appElement.count() > 0;
    const appVisible = appExists ? await appElement.isVisible() : false;
    const appHTML = appExists ? await appElement.innerHTML() : 'Not found';
    
    // Check for specific dashboard elements
    const dashboardView = await page.locator('dashboard-view').count();
    const loginView = await page.locator('login-view').count();
    const currentURL = page.url();
    
    console.log('\n🏗️ App DOM State:');
    console.log('  - App element exists:', appExists);
    console.log('  - App element visible:', appVisible);
    console.log('  - App HTML length:', appHTML.length);
    console.log('  - Dashboard view count:', dashboardView);
    console.log('  - Login view count:', loginView);
    console.log('  - Current URL:', currentURL);
    
    if (appHTML.length < 100) {
      console.log('  - App HTML:', appHTML);
    }
    
    // Check for loading states
    const loadingElements = await page.locator('.loading, .spinner, loading-spinner').count();
    console.log('  - Loading elements:', loadingElements);
    
    // Take screenshot for manual inspection
    await page.screenshot({ path: 'test-results/pwa-debug-screenshot.png', fullPage: true });
    
    // Basic assertions
    expect(errors.length).toBe(0); // No JavaScript errors
    expect(appExists).toBe(true); // App element should exist
    
    // If we have authentication logs, that's good
    if (authLogs.length > 0) {
      console.log('✅ Authentication system is initializing');
    }
  });
});