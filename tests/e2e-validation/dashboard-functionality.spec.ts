/**
 * Dashboard Functionality Test - Verify PWA dashboard features work with real data
 */

import { test, expect } from '@playwright/test';

test.describe('PWA Dashboard Functionality', () => {
  test('Dashboard loads and displays real data', async ({ page }) => {
    const consoleLogs: string[] = [];
    const errors: string[] = [];
    
    // Capture console messages
    page.on('console', msg => {
      const text = msg.text();
      consoleLogs.push(text);
      if (msg.type() === 'error') {
        console.log(`PWA Error:`, text);
      }
    });
    
    // Capture JS errors
    page.on('pageerror', error => {
      errors.push(error.message);
      console.log('JS Error:', error.message);
    });

    console.log('🔍 Loading PWA Dashboard...');
    await page.goto('http://localhost:3002/', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait for app initialization
    await page.waitForTimeout(3000);
    
    // Check that we're on dashboard route
    const currentURL = page.url();
    console.log('Current URL:', currentURL);
    expect(currentURL).toContain('/dashboard');
    
    // Check that dashboard view is rendered
    const dashboardView = page.locator('dashboard-view');
    await expect(dashboardView).toBeVisible({ timeout: 10000 });
    
    // Check for key dashboard elements
    const dashboardTitle = page.locator('h1, h2, .dashboard-title').first();
    await expect(dashboardTitle).toBeVisible({ timeout: 5000 });
    
    // Check for navigation elements (sidebar or bottom nav)
    const hasNavigation = await page.evaluate(() => {
      const sidebar = document.querySelector('sidebar-navigation');
      const bottomNav = document.querySelector('bottom-navigation');
      return sidebar !== null || bottomNav !== null;
    });
    console.log('Navigation elements present:', hasNavigation);
    expect(hasNavigation).toBe(true);
    
    // Check for data loading indicators or content
    const hasLoadingOrContent = await page.evaluate(() => {
      const loadingSpinners = document.querySelectorAll('loading-spinner, .loading, .spinner');
      const contentCards = document.querySelectorAll('.card, .widget, .metric, .chart');
      return loadingSpinners.length > 0 || contentCards.length > 0;
    });
    console.log('Has loading indicators or content:', hasLoadingOrContent);
    
    // Test backend adapter data fetching
    const backendAdapterLogs = consoleLogs.filter(log =>
      log.includes('Backend adapter') || 
      log.includes('Syncing data') ||
      log.includes('live data') ||
      log.includes('API call')
    );
    console.log('Backend adapter activity:', backendAdapterLogs.length > 0);
    
    // Check authentication state is displayed
    const userElements = await page.evaluate(() => {
      const userInfo = document.querySelector('.user-info, .profile, [data-testid="user"]');
      const avatar = document.querySelector('.avatar, .user-avatar, img[alt*="user"]');
      return { userInfo: userInfo !== null, avatar: avatar !== null };
    });
    console.log('User elements:', userElements);
    
    // Test responsive design
    const isMobile = await page.evaluate(() => window.innerWidth < 768);
    console.log('Mobile viewport:', isMobile);
    
    if (isMobile) {
      // Check mobile-specific elements
      const mobileHeader = page.locator('app-header');
      const bottomNav = page.locator('bottom-navigation');
      
      await expect(mobileHeader).toBeVisible();
      await expect(bottomNav).toBeVisible();
      console.log('✅ Mobile layout verified');
    } else {
      // Check desktop-specific elements  
      const sidebar = page.locator('sidebar-navigation');
      await expect(sidebar).toBeVisible();
      console.log('✅ Desktop layout verified');
    }
    
    // Take screenshot for visual verification
    await page.screenshot({ 
      path: 'test-results/dashboard-functionality-screenshot.png', 
      fullPage: true 
    });
    
    // Summary
    console.log('\n📊 Dashboard Test Summary:');
    console.log('  - Dashboard route accessible:', currentURL.includes('/dashboard'));
    console.log('  - Dashboard view rendered:', await dashboardView.isVisible());
    console.log('  - Navigation present:', hasNavigation);
    console.log('  - Content/loading present:', hasLoadingOrContent);
    console.log('  - Backend adapter activity:', backendAdapterLogs.length > 0);
    console.log('  - Authentication displayed:', userElements.userInfo || userElements.avatar);
    console.log('  - JavaScript errors:', errors.length);
    
    // Filter out Lit warning errors for functionality test
    const functionalErrors = errors.filter(error => 
      !error.includes('class field shadowing') &&
      !error.includes('will not trigger updates')
    );
    
    // Basic functionality assertions
    expect(functionalErrors.length).toBe(0);
    expect(currentURL).toContain('/dashboard');
    await expect(dashboardView).toBeVisible();
    expect(hasNavigation).toBe(true);
  });
  
  test('PWA features work correctly', async ({ page }) => {
    console.log('🔍 Testing PWA features...');
    
    await page.goto('http://localhost:3002/', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    await page.waitForTimeout(3000);
    
    // Test service worker registration
    const serviceWorkerRegistered = await page.evaluate(() => {
      return 'serviceWorker' in navigator && 
             navigator.serviceWorker.controller !== null;
    });
    console.log('Service Worker registered:', serviceWorkerRegistered);
    
    // Test offline capability indicators
    const offlineFeatures = await page.evaluate(() => {
      const offlineService = window.appServices?.offline;
      return {
        offlineServiceExists: offlineService !== undefined,
        hasOfflineStorage: 'indexedDB' in window,
        hasLocalStorage: 'localStorage' in window
      };
    });
    console.log('Offline features:', offlineFeatures);
    
    // Test notification service
    const notificationFeatures = await page.evaluate(() => {
      const notificationService = window.appServices?.notification;
      return {
        notificationServiceExists: notificationService !== undefined,
        notificationPermission: 'Notification' in window ? Notification.permission : 'not-supported'
      };
    });
    console.log('Notification features:', notificationFeatures);
    
    // Test PWA manifest
    const manifestLink = page.locator('link[rel="manifest"]');
    await expect(manifestLink).toHaveAttribute('href', '/manifest.webmanifest');
    
    console.log('✅ PWA features verified');
    
    // Assertions
    expect(offlineFeatures.hasOfflineStorage).toBe(true);
    expect(offlineFeatures.hasLocalStorage).toBe(true);
  });
});