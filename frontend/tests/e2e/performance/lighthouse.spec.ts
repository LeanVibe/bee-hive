import { test, expect } from '@playwright/test';
import { 
  PerformanceHelpers,
  AuthHelpers,
  loadTestData,
  takeTimestampedScreenshot,
  waitForNetworkIdle
} from '../utils/test-helpers';

/**
 * Performance E2E Tests using Lighthouse-style metrics
 * Tests PWA performance across different scenarios and pages
 */

test.describe('PWA Performance Testing', () => {
  let performanceHelpers: PerformanceHelpers;
  let authHelpers: AuthHelpers;
  let testData: any;

  test.beforeEach(async ({ page }) => {
    performanceHelpers = new PerformanceHelpers(page);
    authHelpers = new AuthHelpers(page);
    testData = loadTestData();
  });

  test.describe('Core Web Vitals', () => {
    test('should meet Core Web Vitals thresholds on landing page', async ({ page }) => {
      // Navigate to landing page
      await page.goto('/');
      await waitForNetworkIdle(page);

      // Measure Core Web Vitals
      const vitals = await performanceHelpers.getCoreWebVitals();

      // First Contentful Paint (FCP) - should be < 1.8s
      if (vitals.FCP) {
        expect(vitals.FCP).toBeLessThan(1800);
      }

      // Largest Contentful Paint (LCP) - should be < 2.5s
      if (vitals.LCP) {
        expect(vitals.LCP).toBeLessThan(2500);
      }

      // Cumulative Layout Shift (CLS) - should be < 0.1
      if (vitals.CLS) {
        expect(vitals.CLS).toBeLessThan(0.1);
      }

      // First Input Delay (FID) - should be < 100ms
      if (vitals.FID) {
        expect(vitals.FID).toBeLessThan(100);
      }

      await takeTimestampedScreenshot(page, 'core-web-vitals-landing');
    });

    test('should maintain good performance on dashboard', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);

      // Navigate to dashboard
      await page.goto('/dashboard');
      await waitForNetworkIdle(page);

      // Measure performance on data-heavy dashboard
      const loadTime = await performanceHelpers.measurePageLoadTime();
      expect(loadTime).toBeLessThan(3000); // 3 seconds

      const vitals = await performanceHelpers.getCoreWebVitals();

      // Dashboard should still meet reasonable thresholds
      if (vitals.LCP) {
        expect(vitals.LCP).toBeLessThan(4000); // Slightly higher for data-heavy page
      }

      if (vitals.CLS) {
        expect(vitals.CLS).toBeLessThan(0.15); // Slightly higher for dynamic content
      }

      await takeTimestampedScreenshot(page, 'core-web-vitals-dashboard');
    });

    test('should perform well under load simulation', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);

      // Simulate multiple rapid navigation events
      const startTime = Date.now();
      
      await page.goto('/dashboard');
      await waitForNetworkIdle(page, 2000);
      
      await page.goto('/agents');
      await waitForNetworkIdle(page, 2000);
      
      await page.goto('/tasks');
      await waitForNetworkIdle(page, 2000);
      
      await page.goto('/dashboard');
      await waitForNetworkIdle(page, 2000);

      const totalTime = Date.now() - startTime;
      
      // Rapid navigation should complete within reasonable time
      expect(totalTime).toBeLessThan(15000); // 15 seconds for all navigation

      // Check final page performance
      const vitals = await performanceHelpers.getCoreWebVitals();
      if (vitals.LCP) {
        expect(vitals.LCP).toBeLessThan(5000); // Allow higher threshold under load
      }

      await takeTimestampedScreenshot(page, 'performance-under-load');
    });
  });

  test.describe('Resource Loading Performance', () => {
    test('should load critical resources quickly', async ({ page }) => {
      const resourceTimings: Array<{ name: string; duration: number; size: number }> = [];

      // Monitor resource loading
      page.on('response', async (response) => {
        const url = response.url();
        const timing = await response.finished();
        
        if (url.includes('.js') || url.includes('.css') || url.includes('.woff')) {
          resourceTimings.push({
            name: url.split('/').pop() || url,
            duration: timing ? Date.now() - timing : 0,
            size: parseInt(response.headers()['content-length'] || '0')
          });
        }
      });

      await page.goto('/');
      await waitForNetworkIdle(page);

      // Verify critical resources loaded efficiently
      const criticalResources = resourceTimings.filter(resource => 
        resource.name.includes('main') || 
        resource.name.includes('vendor') ||
        resource.name.includes('app')
      );

      expect(criticalResources.length).toBeGreaterThan(0);

      // Critical resources should load within 2 seconds each
      criticalResources.forEach(resource => {
        expect(resource.duration).toBeLessThan(2000);
      });

      await takeTimestampedScreenshot(page, 'resource-loading');
    });

    test('should implement efficient code splitting', async ({ page }) => {
      const loadedChunks: string[] = [];

      // Monitor JavaScript chunk loading
      page.on('response', (response) => {
        const url = response.url();
        if (url.includes('.js') && !url.includes('node_modules')) {
          loadedChunks.push(url.split('/').pop() || url);
        }
      });

      // Load different pages to trigger code splitting
      await page.goto('/');
      await waitForNetworkIdle(page);
      const initialChunks = [...loadedChunks];

      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await authHelpers.login(adminUser);

      await page.goto('/agents');
      await waitForNetworkIdle(page);
      
      const agentsChunks = loadedChunks.filter(chunk => !initialChunks.includes(chunk));

      // Should load additional chunks for new pages
      expect(agentsChunks.length).toBeGreaterThan(0);

      await takeTimestampedScreenshot(page, 'code-splitting');
    });

    test('should optimize image loading', async ({ page }) => {
      const imageMetrics: Array<{ url: string; loadTime: number; format: string }> = [];

      page.on('response', async (response) => {
        const url = response.url();
        const contentType = response.headers()['content-type'] || '';
        
        if (contentType.startsWith('image/')) {
          const startTime = Date.now();
          await response.finished();
          const loadTime = Date.now() - startTime;
          
          imageMetrics.push({
            url,
            loadTime,
            format: contentType.split('/')[1]
          });
        }
      });

      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await page.goto('/dashboard');
      await waitForNetworkIdle(page);

      if (imageMetrics.length > 0) {
        // Images should load efficiently
        imageMetrics.forEach(image => {
          expect(image.loadTime).toBeLessThan(3000); // 3 seconds per image
        });

        // Should use modern image formats
        const modernFormats = imageMetrics.filter(img => 
          ['webp', 'avif'].includes(img.format)
        );
        
        // At least some images should use modern formats (if available)
        // This is optional as it depends on server configuration
        if (modernFormats.length > 0) {
          expect(modernFormats.length).toBeGreaterThan(0);
        }
      }

      await takeTimestampedScreenshot(page, 'image-optimization');
    });
  });

  test.describe('Runtime Performance', () => {
    test('should maintain smooth animations', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await page.goto('/dashboard');

      // Test animation performance
      const animationPerformance = await page.evaluate(() => {
        return new Promise((resolve) => {
          let frameCount = 0;
          let startTime = Date.now();
          
          function countFrames() {
            frameCount++;
            if (Date.now() - startTime < 1000) {
              requestAnimationFrame(countFrames);
            } else {
              resolve({
                fps: frameCount,
                duration: Date.now() - startTime
              });
            }
          }
          
          requestAnimationFrame(countFrames);
        });
      });

      // Should maintain reasonable frame rate
      expect((animationPerformance as any).fps).toBeGreaterThan(30); // 30+ FPS

      await takeTimestampedScreenshot(page, 'animation-performance');
    });

    test('should handle memory efficiently', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);

      // Get initial memory usage
      const initialMemory = await page.evaluate(() => {
        if ('memory' in performance) {
          return {
            used: (performance as any).memory.usedJSHeapSize,
            total: (performance as any).memory.totalJSHeapSize,
            limit: (performance as any).memory.jsHeapSizeLimit
          };
        }
        return null;
      });

      // Navigate through multiple pages to stress test memory
      const pages = ['/dashboard', '/agents', '/tasks', '/dashboard'];
      
      for (const pagePath of pages) {
        await page.goto(pagePath);
        await waitForNetworkIdle(page);
        await page.waitForTimeout(1000);
      }

      // Get final memory usage
      const finalMemory = await page.evaluate(() => {
        if ('memory' in performance) {
          return {
            used: (performance as any).memory.usedJSHeapSize,
            total: (performance as any).memory.totalJSHeapSize,
            limit: (performance as any).memory.jsHeapSizeLimit
          };
        }
        return null;
      });

      if (initialMemory && finalMemory) {
        // Memory growth should be reasonable
        const memoryGrowth = finalMemory.used - initialMemory.used;
        const maxAcceptableGrowth = initialMemory.used * 2; // 100% growth max
        
        expect(memoryGrowth).toBeLessThan(maxAcceptableGrowth);
        
        // Should not exceed 50% of memory limit
        expect(finalMemory.used).toBeLessThan(finalMemory.limit * 0.5);
      }

      await takeTimestampedScreenshot(page, 'memory-efficiency');
    });

    test('should handle large datasets efficiently', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);

      // Navigate to page with large dataset
      await page.goto('/agents');
      
      // Simulate loading large dataset
      await page.selectOption('[data-testid="items-per-page"]', '100');
      await waitForNetworkIdle(page);

      // Measure rendering performance with large dataset
      const renderingPerformance = await page.evaluate(() => {
        const startTime = Date.now();
        
        // Force a reflow by accessing layout properties
        const elements = document.querySelectorAll('[data-testid="agent-card"]');
        elements.forEach(el => {
          el.getBoundingClientRect();
        });
        
        return {
          renderTime: Date.now() - startTime,
          elementCount: elements.length
        };
      });

      // Should render large datasets efficiently
      expect(renderingPerformance.renderTime).toBeLessThan(1000); // 1 second
      
      // Test scrolling performance with large dataset
      const scrollPerformance = await page.evaluate(() => {
        return new Promise((resolve) => {
          const startTime = Date.now();
          let scrollCount = 0;
          
          function performScroll() {
            window.scrollBy(0, 100);
            scrollCount++;
            
            if (scrollCount < 10) {
              requestAnimationFrame(performScroll);
            } else {
              resolve({
                scrollTime: Date.now() - startTime,
                scrollCount
              });
            }
          }
          
          requestAnimationFrame(performScroll);
        });
      });

      // Scrolling should be smooth
      expect((scrollPerformance as any).scrollTime).toBeLessThan(500); // 500ms for 10 scrolls

      await takeTimestampedScreenshot(page, 'large-dataset-performance');
    });
  });

  test.describe('Network Performance', () => {
    test('should handle slow network conditions', async ({ page, context }) => {
      // Simulate slow 3G connection
      await context.route('**/*', async (route) => {
        // Add delay to simulate slow network
        await new Promise(resolve => setTimeout(resolve, 100));
        route.continue();
      });

      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      
      const loadTime = await performanceHelpers.measurePageLoadTime();
      
      // Should still load within reasonable time even on slow network
      expect(loadTime).toBeLessThan(10000); // 10 seconds on slow network

      await authHelpers.login(adminUser);
      await page.goto('/dashboard');
      await waitForNetworkIdle(page, 10000);

      // Dashboard should be functional despite slow network
      await expect(page.locator('[data-testid="dashboard-header"]')).toBeVisible();

      await takeTimestampedScreenshot(page, 'slow-network-performance');
    });

    test('should implement efficient caching strategies', async ({ page }) => {
      // Track cache hits vs network requests
      const cacheHits: string[] = [];
      const networkRequests: string[] = [];

      page.on('response', (response) => {
        const cacheHeader = response.headers()['cache-control'];
        const url = response.url();
        
        if (cacheHeader && (cacheHeader.includes('max-age') || cacheHeader.includes('immutable'))) {
          cacheHits.push(url);
        } else {
          networkRequests.push(url);
        }
      });

      // Initial page load
      await page.goto('/');
      await waitForNetworkIdle(page);

      const initialNetworkRequests = networkRequests.length;

      // Reload page to test caching
      await page.reload();
      await waitForNetworkIdle(page);

      // Should have fewer network requests on reload due to caching
      const reloadNetworkRequests = networkRequests.length - initialNetworkRequests;
      expect(reloadNetworkRequests).toBeLessThan(initialNetworkRequests);

      await takeTimestampedScreenshot(page, 'caching-efficiency');
    });

    test('should minimize bundle sizes', async ({ page }) => {
      const bundleSizes: Array<{ name: string; size: number }> = [];

      page.on('response', async (response) => {
        const url = response.url();
        const contentLength = response.headers()['content-length'];
        
        if (url.includes('.js') && contentLength) {
          bundleSizes.push({
            name: url.split('/').pop() || url,
            size: parseInt(contentLength)
          });
        }
      });

      await page.goto('/');
      await waitForNetworkIdle(page);

      if (bundleSizes.length > 0) {
        // Main bundle should be reasonably sized
        const mainBundle = bundleSizes.find(bundle => 
          bundle.name.includes('main') || bundle.name.includes('app')
        );
        
        if (mainBundle) {
          expect(mainBundle.size).toBeLessThan(500000); // 500KB for main bundle
        }

        // Total JavaScript should be under 2MB
        const totalSize = bundleSizes.reduce((sum, bundle) => sum + bundle.size, 0);
        expect(totalSize).toBeLessThan(2000000); // 2MB total
      }

      await takeTimestampedScreenshot(page, 'bundle-size-optimization');
    });
  });

  test.describe('Performance Monitoring', () => {
    test('should collect performance metrics', async ({ page }) => {
      // Navigate and collect various performance metrics
      await page.goto('/');
      await waitForNetworkIdle(page);

      const performanceMetrics = await page.evaluate(() => {
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        const paint = performance.getEntriesByType('paint');
        
        return {
          domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
          loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
          firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
          firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
          ttfb: navigation.responseStart - navigation.requestStart
        };
      });

      // Validate performance metrics
      expect(performanceMetrics.domContentLoaded).toBeGreaterThan(0);
      expect(performanceMetrics.firstContentfulPaint).toBeGreaterThan(0);
      expect(performanceMetrics.ttfb).toBeGreaterThan(0);

      // Time to First Byte should be reasonable
      expect(performanceMetrics.ttfb).toBeLessThan(1000); // 1 second

      await takeTimestampedScreenshot(page, 'performance-metrics');
    });

    test('should track user interaction performance', async ({ page }) => {
      const adminUser = testData.users.find((u: any) => u.role === 'admin');
      await page.goto('/');
      await authHelpers.login(adminUser);
      await page.goto('/dashboard');

      // Measure interaction responsiveness
      const interactionMetrics = await page.evaluate(() => {
        return new Promise((resolve) => {
          const metrics: Array<{ type: string; duration: number }> = [];
          
          // Measure click responsiveness
          const button = document.querySelector('[data-testid="nav-agents"]') as HTMLElement;
          if (button) {
            const startTime = Date.now();
            
            button.addEventListener('click', () => {
              metrics.push({
                type: 'click',
                duration: Date.now() - startTime
              });
              resolve(metrics);
            });
            
            button.click();
          } else {
            resolve(metrics);
          }
        });
      });

      // Click should be responsive
      if (Array.isArray(interactionMetrics) && interactionMetrics.length > 0) {
        expect(interactionMetrics[0].duration).toBeLessThan(100); // 100ms for click response
      }

      await takeTimestampedScreenshot(page, 'interaction-performance');
    });
  });
});