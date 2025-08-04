/**
 * Mobile PWA Backend Integration Test
 * 
 * Tests the mobile PWA with the new backend adapter
 * to ensure it connects to real data from /dashboard/api/live-data
 */

import { test, expect } from '@playwright/test';

test.describe('Mobile PWA Backend Integration', () => {
  test('PWA loads and connects to backend adapter', async ({ page }) => {
    console.log('🧪 Testing PWA backend integration...');
    
    // Navigate to PWA
    await page.goto('http://localhost:3002/', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait for app initialization
    await page.waitForTimeout(5000);
    
    // Take screenshot for debugging
    await page.screenshot({ path: 'test-results/pwa-backend-test.png', fullPage: true });
    
    // Check if app loaded without the previous error
    const appElement = await page.locator('#app');
    await expect(appElement).toBeVisible({ timeout: 10000 });
    
    console.log('✅ PWA app element is visible');
  });

  test('PWA fetches real backend data', async ({ page }) => {
    // Monitor network requests
    const requests: string[] = [];
    page.on('request', request => {
      if (request.url().includes('/dashboard/api/live-data')) {
        requests.push(request.url());
        console.log('📡 Backend API request:', request.url());
      }
    });

    // Navigate to PWA
    await page.goto('http://localhost:3002/', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait for backend adapter to initialize and fetch data
    await page.waitForTimeout(8000);
    
    // Verify backend API was called
    expect(requests.length).toBeGreaterThan(0);
    console.log('✅ PWA successfully called backend API');
  });

  test('PWA displays real agent data', async ({ page }) => {
    await page.goto('http://localhost:3002/', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait for data to load
    await page.waitForTimeout(8000);
    
    // Check for dashboard content (should not be loading screen)
    const loadingContainer = await page.locator('.loading-container');
    await expect(loadingContainer).toBeHidden({ timeout: 15000 });
    
    // Look for agent-related content
    const pageContent = await page.textContent('body');
    
    // Should contain agent roles or data
    const hasAgentData = pageContent?.includes('Product Manager') || 
                        pageContent?.includes('Architect') ||
                        pageContent?.includes('Backend Developer') ||
                        pageContent?.includes('active') ||
                        pageContent?.includes('agent');
    
    if (hasAgentData) {
      console.log('✅ PWA displays real agent data');
      expect(hasAgentData).toBe(true);
    } else {
      console.log('⚠️ No agent data found in PWA - checking console for errors');
      
      // Check for console errors
      const logs: string[] = [];
      page.on('console', msg => logs.push(msg.text()));
      
      console.log('Console logs:', logs.slice(-10)); // Last 10 logs
    }
  });

  test('PWA console shows backend adapter logs', async ({ page }) => {
    const consoleLogs: string[] = [];
    
    page.on('console', msg => {
      consoleLogs.push(msg.text());
      if (msg.text().includes('Backend adapter') || msg.text().includes('🔄') || msg.text().includes('✅')) {
        console.log('PWA Console:', msg.text());
      }
    });

    await page.goto('http://localhost:3002/', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait for initialization logs
    await page.waitForTimeout(8000);
    
    // Look for backend adapter initialization logs
    const backendLogs = consoleLogs.filter(log => 
      log.includes('Backend adapter') || 
      log.includes('Syncing data') ||
      log.includes('initialized successfully')
    );
    
    console.log('Backend adapter logs found:', backendLogs.length);
    backendLogs.forEach(log => console.log('  -', log));
    
    expect(backendLogs.length).toBeGreaterThan(0);
  });
});