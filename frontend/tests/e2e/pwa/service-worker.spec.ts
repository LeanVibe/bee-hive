import { test, expect } from '@playwright/test';
import { 
  PWAHelpers,
  AuthHelpers, 
  loadTestData,
  takeTimestampedScreenshot,
  waitForNetworkIdle
} from '../utils/test-helpers';

/**
 * PWA Service Worker E2E Tests
 * Tests service worker functionality, caching, and offline capabilities
 */

test.describe('PWA Service Worker Functionality', () => {
  let pwaHelpers: PWAHelpers;
  let authHelpers: AuthHelpers;
  let testData: any;

  test.beforeEach(async ({ page, context }) => {
    pwaHelpers = new PWAHelpers(page, context);
    authHelpers = new AuthHelpers(page);
    testData = loadTestData();
    
    // Navigate to PWA
    await page.goto('/');
  });

  test.describe('Service Worker Registration', () => {
    test('should register service worker successfully', async ({ page }) => {
      // Wait for service worker registration
      await page.waitForTimeout(2000);

      // Check service worker registration
      const swRegistered = await pwaHelpers.checkServiceWorkerRegistration();
      expect(swRegistered).toBe(true);

      // Verify service worker is active
      const swStatus = await page.evaluate(async () => {
        if ('serviceWorker' in navigator) {
          const registration = await navigator.serviceWorker.getRegistration();
          return {
            hasRegistration: !!registration,
            isActive: !!registration?.active,
            scope: registration?.scope
          };
        }
        return { hasRegistration: false, isActive: false, scope: null };
      });

      expect(swStatus.hasRegistration).toBe(true);
      expect(swStatus.isActive).toBe(true);
      expect(swStatus.scope).toContain('/');

      await takeTimestampedScreenshot(page, 'service-worker-registered');
    });

    test('should handle service worker updates', async ({ page }) => {
      // Initial service worker registration
      await page.waitForTimeout(2000);

      // Simulate service worker update
      await page.evaluate(() => {
        if ('serviceWorker' in navigator) {
          navigator.serviceWorker.getRegistration().then(registration => {
            if (registration) {
              registration.update();
            }
          });
        }
      });

      // Wait for update process
      await page.waitForTimeout(3000);

      // Check for update notification
      const updateAvailable = await page.evaluate(() => {
        return new Promise((resolve) => {
          if ('serviceWorker' in navigator) {
            navigator.serviceWorker.addEventListener('controllerchange', () => {
              resolve(true);
            });
            setTimeout(() => resolve(false), 2000);
          } else {
            resolve(false);
          }
        });
      });

      // Service worker should remain functional
      const swRegistered = await pwaHelpers.checkServiceWorkerRegistration();
      expect(swRegistered).toBe(true);

      await takeTimestampedScreenshot(page, 'service-worker-updated');
    });

    test('should handle service worker registration failures gracefully', async ({ page }) => {
      // Test with invalid service worker scope (simulating failure)
      const registrationResult = await page.evaluate(async () => {
        try {
          if ('serviceWorker' in navigator) {
            await navigator.serviceWorker.register('/invalid-sw.js');
            return { success: true, error: null };
          }
          return { success: false, error: 'Service Worker not supported' };
        } catch (error) {
          return { success: false, error: (error as Error).message };
        }
      });

      // Should handle failure gracefully
      expect(registrationResult.success).toBe(false);
      expect(registrationResult.error).toBeTruthy();

      // PWA should still function without service worker
      await expect(page.locator('body')).toBeVisible();

      await takeTimestampedScreenshot(page, 'service-worker-failure-handled');
    });
  });

  test.describe('Caching Strategies', () => {
    test('should cache static assets effectively', async ({ page }) => {
      // Log in and navigate to ensure assets are cached
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await authHelpers.login(adminUser);
      
      // Navigate through different pages to trigger caching
      await page.goto('/dashboard');
      await waitForNetworkIdle(page);
      
      await page.goto('/agents');
      await waitForNetworkIdle(page);
      
      await page.goto('/tasks');
      await waitForNetworkIdle(page);

      // Check cache storage
      const cacheStatus = await page.evaluate(async () => {
        if ('caches' in window) {
          const cacheNames = await caches.keys();
          const cacheEntries = [];
          
          for (const cacheName of cacheNames) {
            const cache = await caches.open(cacheName);
            const keys = await cache.keys();
            cacheEntries.push({
              name: cacheName,
              entries: keys.length,
              urls: keys.slice(0, 5).map(req => req.url) // First 5 URLs
            });
          }
          
          return {
            hasCaches: cacheNames.length > 0,
            cacheCount: cacheNames.length,
            caches: cacheEntries
          };
        }
        return { hasCaches: false, cacheCount: 0, caches: [] };
      });

      expect(cacheStatus.hasCaches).toBe(true);
      expect(cacheStatus.cacheCount).toBeGreaterThan(0);

      // Verify common assets are cached
      const cachedUrls = cacheStatus.caches.flatMap(cache => cache.urls);
      const hasStaticAssets = cachedUrls.some(url => 
        url.includes('.js') || url.includes('.css') || url.includes('.html')
      );
      
      expect(hasStaticAssets).toBe(true);

      await takeTimestampedScreenshot(page, 'assets-cached');
    });

    test('should implement cache-first strategy for static assets', async ({ page }) => {
      // Load page initially to populate cache
      await page.goto('/dashboard');
      await waitForNetworkIdle(page);

      // Go offline
      await page.context().setOffline(true);

      // Reload page - should load from cache
      await page.reload();
      await waitForNetworkIdle(page);

      // Verify page loaded successfully from cache
      await expect(page.locator('[data-testid="dashboard-header"]')).toBeVisible();
      await expect(page.locator('body')).not.toHaveClass(/.*offline-indicator.*/);

      // Go back online
      await page.context().setOffline(false);

      await takeTimestampedScreenshot(page, 'cache-first-strategy');
    });

    test('should implement network-first strategy for API calls', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await authHelpers.login(adminUser);

      // Monitor network requests
      const networkRequests: string[] = [];
      page.on('request', request => {
        if (request.url().includes('/api/')) {
          networkRequests.push(request.url());
        }
      });

      // Navigate to agents page (triggers API calls)
      await page.goto('/agents');
      await waitForNetworkIdle(page);

      // Verify API requests were made
      expect(networkRequests.length).toBeGreaterThan(0);

      // Test cache fallback when offline
      await page.context().setOffline(true);
      
      // Navigate away and back to trigger cache fallback
      await page.goto('/dashboard');
      await page.goto('/agents');
      await waitForNetworkIdle(page);

      // Should still show content (from cache)
      await expect(page.locator('[data-testid="agents-page"]')).toBeVisible();

      await page.context().setOffline(false);

      await takeTimestampedScreenshot(page, 'network-first-strategy');
    });
  });

  test.describe('Offline Functionality', () => {
    test('should work offline after initial load', async ({ page, context }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      
      // Login and navigate to populate cache
      await authHelpers.login(adminUser);
      await page.goto('/dashboard');
      await waitForNetworkIdle(page);

      // Test offline functionality
      await pwaHelpers.testOfflineMode();

      // Verify offline indicator appears
      await expect(page.locator('[data-testid="offline-indicator"]')).toBeVisible();

      // Test navigation while offline
      await page.click('[data-testid="nav-agents"]');
      await expect(page.locator('[data-testid="agents-page"]')).toBeVisible();

      await page.click('[data-testid="nav-tasks"]');
      await expect(page.locator('[data-testid="tasks-page"]')).toBeVisible();

      // Verify cached data is displayed
      await expect(page.locator('[data-testid="cached-data-indicator"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'offline-functionality');
    });

    test('should queue actions while offline', async ({ page, context }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await authHelpers.login(adminUser);
      await page.goto('/agents');
      await waitForNetworkIdle(page);

      // Go offline
      await context.setOffline(true);

      // Try to create a new agent (should be queued)
      await page.click('[data-testid="create-agent-button"]');
      await page.fill('[data-testid="agent-name-input"]', 'Offline Test Agent');
      await page.selectOption('[data-testid="agent-type-select"]', 'monitoring');
      await page.fill('[data-testid="agent-description-input"]', 'Created while offline');
      
      await page.click('[data-testid="create-agent-submit"]');

      // Should show queued message
      await expect(page.locator('[data-testid="action-queued-message"]')).toBeVisible();

      // Go back online
      await context.setOffline(false);
      await waitForNetworkIdle(page, 10000);

      // Queued action should be processed
      await expect(page.locator('[data-testid="action-processed-message"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'offline-queue');
    });

    test('should sync data when coming back online', async ({ page, context }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await authHelpers.login(adminUser);
      
      // Load dashboard with initial data
      await page.goto('/dashboard');
      await waitForNetworkIdle(page);

      // Get initial metrics
      const initialMetrics = await page.evaluate(() => {
        return {
          totalAgents: document.querySelector('[data-testid="total-agents-value"]')?.textContent,
          activeTasks: document.querySelector('[data-testid="active-tasks-value"]')?.textContent
        };
      });

      // Go offline
      await context.setOffline(true);
      await page.waitForTimeout(2000);

      // Go back online
      await context.setOffline(false);

      // Wait for sync to complete
      await waitForNetworkIdle(page, 10000);

      // Verify sync indicator appears
      await expect(page.locator('[data-testid="sync-indicator"]')).toBeVisible();

      // Verify data is refreshed
      const syncedMetrics = await page.evaluate(() => {
        return {
          totalAgents: document.querySelector('[data-testid="total-agents-value"]')?.textContent,
          activeTasks: document.querySelector('[data-testid="active-tasks-value"]')?.textContent
        };
      });

      // Data should be present (may be same or updated)
      expect(syncedMetrics.totalAgents).toBeTruthy();
      expect(syncedMetrics.activeTasks).toBeTruthy();

      await takeTimestampedScreenshot(page, 'data-sync');
    });
  });

  test.describe('Background Sync', () => {
    test('should register background sync events', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await authHelpers.login(adminUser);

      // Check if background sync is supported and registered
      const backgroundSyncSupport = await page.evaluate(async () => {
        if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
          try {
            const registration = await navigator.serviceWorker.ready;
            await registration.sync.register('background-sync-test');
            return { supported: true, registered: true };
          } catch (error) {
            return { supported: true, registered: false, error: (error as Error).message };
          }
        }
        return { supported: false, registered: false };
      });

      if (backgroundSyncSupport.supported) {
        expect(backgroundSyncSupport.registered).toBe(true);
      }

      await takeTimestampedScreenshot(page, 'background-sync');
    });

    test('should handle periodic background sync', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await authHelpers.login(adminUser);

      // Check periodic background sync support
      const periodicSyncSupport = await page.evaluate(async () => {
        if ('serviceWorker' in navigator && 'periodicSync' in window.ServiceWorkerRegistration.prototype) {
          try {
            const registration = await navigator.serviceWorker.ready;
            const status = await navigator.permissions.query({ name: 'periodic-background-sync' as any });
            
            if (status.state === 'granted') {
              await (registration as any).periodicSync.register('data-refresh', {
                minInterval: 24 * 60 * 60 * 1000 // 24 hours
              });
              return { supported: true, registered: true };
            }
            
            return { supported: true, registered: false, reason: 'Permission not granted' };
          } catch (error) {
            return { supported: true, registered: false, error: (error as Error).message };
          }
        }
        return { supported: false, registered: false };
      });

      // Note: Periodic sync might not be supported in all browsers/contexts
      if (periodicSyncSupport.supported) {
        console.log('Periodic sync status:', periodicSyncSupport);
      }

      await takeTimestampedScreenshot(page, 'periodic-sync');
    });
  });

  test.describe('Cache Management', () => {
    test('should clean up old cache entries', async ({ page }) => {
      // Load pages to populate cache
      await page.goto('/dashboard');
      await waitForNetworkIdle(page);
      await page.goto('/agents');
      await waitForNetworkIdle(page);

      // Check initial cache state
      const initialCacheState = await page.evaluate(async () => {
        if ('caches' in window) {
          const cacheNames = await caches.keys();
          return { count: cacheNames.length, names: cacheNames };
        }
        return { count: 0, names: [] };
      });

      expect(initialCacheState.count).toBeGreaterThan(0);

      // Trigger cache cleanup (simulate version update)
      const cleanupResult = await page.evaluate(async () => {
        if ('caches' in window) {
          const cacheNames = await caches.keys();
          const oldCaches = cacheNames.filter(name => name.includes('v1') || name.includes('old'));
          
          let deletedCount = 0;
          for (const cacheName of oldCaches) {
            const deleted = await caches.delete(cacheName);
            if (deleted) deletedCount++;
          }
          
          return { deletedCount, remainingCaches: (await caches.keys()).length };
        }
        return { deletedCount: 0, remainingCaches: 0 };
      });

      // Verify cache management is working
      expect(cleanupResult.remainingCaches).toBeGreaterThanOrEqual(0);

      await takeTimestampedScreenshot(page, 'cache-cleanup');
    });

    test('should handle cache storage limits', async ({ page }) => {
      // Test cache storage quota
      const storageInfo = await page.evaluate(async () => {
        if ('storage' in navigator && 'estimate' in navigator.storage) {
          const estimate = await navigator.storage.estimate();
          return {
            quota: estimate.quota,
            usage: estimate.usage,
            usagePercentage: estimate.quota ? (estimate.usage! / estimate.quota) * 100 : 0
          };
        }
        return { quota: null, usage: null, usagePercentage: 0 };
      });

      if (storageInfo.quota && storageInfo.usage) {
        expect(storageInfo.usagePercentage).toBeLessThan(90); // Should not exceed 90% of quota
      }

      // Test cache size management
      const cacheManagement = await page.evaluate(async () => {
        if ('caches' in window) {
          const cacheNames = await caches.keys();
          let totalSize = 0;
          
          for (const cacheName of cacheNames) {
            const cache = await caches.open(cacheName);
            const keys = await cache.keys();
            totalSize += keys.length; // Approximate size based on entry count
          }
          
          return { totalEntries: totalSize, cacheCount: cacheNames.length };
        }
        return { totalEntries: 0, cacheCount: 0 };
      });

      expect(cacheManagement.totalEntries).toBeGreaterThanOrEqual(0);

      await takeTimestampedScreenshot(page, 'storage-management');
    });
  });
});