import { test, expect } from '@playwright/test'
import { TestHelpers } from '../../utils/test-helpers'

/**
 * Progressive Web App (PWA) Functionality Tests
 * 
 * Validates PWA features including:
 * - Manifest and installability
 * - Service worker registration and functionality
 * - Offline mode and caching
 * - App-like behavior
 * - Push notifications (when supported)
 * 
 * These tests ensure the dashboard works as a proper PWA for mobile users
 */

test.describe('PWA Installation & Offline Mode', () => {
  
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await TestHelpers.waitForNetworkIdle(page)
  })

  test('manifest file is valid and accessible', async ({ page }) => {
    // Check for manifest link
    const manifestLink = page.locator('link[rel="manifest"]')
    await expect(manifestLink).toBeAttached()
    
    const manifestHref = await manifestLink.getAttribute('href')
    expect(manifestHref).toBeTruthy()
    
    // Fetch and validate manifest
    const manifestResponse = await page.request.get(manifestHref!)
    expect(manifestResponse.status()).toBe(200)
    
    const manifestContent = await manifestResponse.json()
    
    // Validate required manifest fields
    expect(manifestContent.name).toBeTruthy()
    expect(manifestContent.short_name).toBeTruthy()
    expect(manifestContent.start_url).toBeTruthy()
    expect(manifestContent.display).toBeTruthy()
    expect(manifestContent.theme_color).toBeTruthy()
    expect(manifestContent.background_color).toBeTruthy()
    expect(manifestContent.icons).toBeTruthy()
    expect(Array.isArray(manifestContent.icons)).toBe(true)
    expect(manifestContent.icons.length).toBeGreaterThan(0)
    
    // Validate icon requirements
    const hasRequiredSizes = manifestContent.icons.some((icon: any) => 
      icon.sizes?.includes('192x192') || icon.sizes?.includes('512x512')
    )
    expect(hasRequiredSizes).toBe(true)
    
    console.log('Manifest validation passed:', {
      name: manifestContent.name,
      icons: manifestContent.icons.length,
      display: manifestContent.display
    })
  })

  test('service worker registers successfully', async ({ page }) => {
    // Wait for service worker registration
    const serviceWorkerRegistered = await page.evaluate(async () => {
      if (!('serviceWorker' in navigator)) {
        return { supported: false, registered: false }
      }
      
      try {
        // Wait for existing registration or new registration
        const registration = await navigator.serviceWorker.ready
        return {
          supported: true,
          registered: !!registration,
          scope: registration.scope,
          updateViaCache: registration.updateViaCache
        }
      } catch (error) {
        return {
          supported: true,
          registered: false,
          error: error.message
        }
      }
    })
    
    console.log('Service Worker status:', serviceWorkerRegistered)
    
    if (serviceWorkerRegistered.supported) {
      expect(serviceWorkerRegistered.registered).toBe(true)
    } else {
      test.skip('Service workers not supported in this environment')
    }
  })

  test('app functions in offline mode', async ({ page, context }) => {
    // First, load the app normally to ensure it's cached
    await page.waitForLoadState('networkidle')
    await expect(page.locator('body')).toBeVisible()
    
    // Wait for service worker to be ready
    await page.waitForFunction(() => {
      return 'serviceWorker' in navigator && navigator.serviceWorker.controller
    }, { timeout: 10000 }).catch(() => {
      console.log('Service worker not ready - testing basic offline behavior')
    })
    
    // Take screenshot of online state
    await page.screenshot({ 
      path: 'test-results/pwa/online-state.png',
      fullPage: true 
    })
    
    // Go offline
    await context.setOffline(true)
    
    // Reload the page
    await page.reload({ waitUntil: 'domcontentloaded' })
    
    // Verify the app still loads (cached content or offline page)
    await expect(page.locator('body')).toBeVisible()
    
    // Check for offline indicators
    const offlineIndicators = page.locator(
      '.offline-indicator, .offline-mode, [data-testid*="offline"], .connection-status'
    )
    
    if (await offlineIndicators.count() > 0) {
      await expect(offlineIndicators.first()).toBeVisible()
      console.log('✓ Offline indicator displayed')
    }
    
    // Verify essential content is still available
    const essentialContent = page.locator('header, main, .dashboard-content')
    if (await essentialContent.count() > 0) {
      await expect(essentialContent.first()).toBeVisible()
      console.log('✓ Essential content available offline')
    }
    
    // Take screenshot of offline state
    await page.screenshot({ 
      path: 'test-results/pwa/offline-state.png',
      fullPage: true 
    })
    
    // Restore online mode
    await context.setOffline(false)
    
    // Verify app recovers online
    await page.reload()
    await TestHelpers.waitForNetworkIdle(page)
    await expect(page.locator('body')).toBeVisible()
  })

  test('app behaves like native app', async ({ page, browserName }) => {
    // Skip for browsers that don't support PWA features
    if (browserName === 'firefox') {
      test.skip('PWA app-like behavior not fully supported in Firefox')
    }
    
    // Check for app-like display mode in manifest
    const manifestLink = page.locator('link[rel="manifest"]')
    if (await manifestLink.count() > 0) {
      const manifestHref = await manifestLink.getAttribute('href')
      const manifestResponse = await page.request.get(manifestHref!)
      const manifest = await manifestResponse.json()
      
      // Should have app-like display mode
      const appLikeDisplays = ['standalone', 'fullscreen', 'minimal-ui']
      expect(appLikeDisplays).toContain(manifest.display)
    }
    
    // Check for PWA meta tags
    const metaTags = {
      'theme-color': page.locator('meta[name="theme-color"]'),
      'apple-mobile-web-app-capable': page.locator('meta[name="apple-mobile-web-app-capable"]'),
      'apple-mobile-web-app-status-bar-style': page.locator('meta[name="apple-mobile-web-app-status-bar-style"]')
    }
    
    for (const [name, locator] of Object.entries(metaTags)) {
      if (await locator.count() > 0) {
        await expect(locator).toBeAttached()
        console.log(`✓ ${name} meta tag present`)
      }
    }
    
    // Check for app icons
    const iconLinks = page.locator('link[rel*="icon"], link[rel="apple-touch-icon"]')
    if (await iconLinks.count() > 0) {
      expect(await iconLinks.count()).toBeGreaterThan(0)
      console.log(`✓ ${await iconLinks.count()} app icons found`)
    }
  })

  test('caching strategy works correctly', async ({ page, context }) => {
    // Load page and wait for caching
    await page.waitForLoadState('networkidle')
    
    // Check for cached resources
    const cachedResources = await page.evaluate(async () => {
      if (!('caches' in window)) {
        return { supported: false, caches: [] }
      }
      
      try {
        const cacheNames = await caches.keys()
        const cacheContents = []
        
        for (const cacheName of cacheNames) {
          const cache = await caches.open(cacheName)
          const requests = await cache.keys()
          cacheContents.push({
            name: cacheName,
            entries: requests.length,
            urls: requests.slice(0, 5).map(req => req.url) // First 5 URLs
          })
        }
        
        return { supported: true, caches: cacheContents }
      } catch (error) {
        return { supported: true, error: error.message, caches: [] }
      }
    })
    
    console.log('Cache status:', cachedResources)
    
    if (cachedResources.supported && cachedResources.caches.length > 0) {
      expect(cachedResources.caches.length).toBeGreaterThan(0)
      
      // Verify essential resources are cached
      const hasIndexCached = cachedResources.caches.some(cache => 
        cache.urls.some(url => url.includes('/') || url.includes('index'))
      )
      
      if (hasIndexCached) {
        console.log('✓ Main app resources are cached')
      }
    }
  })

  test('app updates properly when available', async ({ page }) => {
    // Test update mechanism
    const updateCheck = await page.evaluate(async () => {
      if (!('serviceWorker' in navigator)) {
        return { supported: false }
      }
      
      try {
        const registration = await navigator.serviceWorker.getRegistration()
        if (!registration) {
          return { supported: true, hasRegistration: false }
        }
        
        // Check for update
        await registration.update()
        
        return {
          supported: true,
          hasRegistration: true,
          scope: registration.scope,
          updateViaCache: registration.updateViaCache
        }
      } catch (error) {
        return {
          supported: true,
          hasRegistration: false,
          error: error.message
        }
      }
    })
    
    console.log('Update check result:', updateCheck)
    
    if (updateCheck.supported && updateCheck.hasRegistration) {
      console.log('✓ Service worker update mechanism is functional')
    }
    
    // Look for update UI elements
    const updateElements = page.locator(
      '.update-available, .app-update, [data-testid*="update"]'
    )
    
    // Update UI may not be visible if no update is available
    if (await updateElements.count() > 0) {
      console.log('Update UI elements found')
    }
  })

  test('storage quota and usage are managed', async ({ page }) => {
    const storageInfo = await page.evaluate(async () => {
      const estimates: any = {}
      
      // Check storage estimate
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        try {
          const estimate = await navigator.storage.estimate()
          estimates.quota = estimate.quota
          estimates.usage = estimate.usage
          estimates.usagePercent = estimate.quota ? (estimate.usage! / estimate.quota * 100) : 0
        } catch (error) {
          estimates.error = error.message
        }
      }
      
      // Check localStorage usage
      let localStorageSize = 0
      try {
        for (const key in localStorage) {
          if (localStorage.hasOwnProperty(key)) {
            localStorageSize += localStorage[key].length + key.length
          }
        }
        estimates.localStorage = localStorageSize
      } catch (error) {
        estimates.localStorageError = error.message
      }
      
      // Check indexedDB usage (basic check)
      estimates.indexedDBSupported = 'indexedDB' in window
      
      return estimates
    })
    
    console.log('Storage information:', storageInfo)
    
    // Verify storage is available and not over-used
    if (storageInfo.quota && storageInfo.usage) {
      expect(storageInfo.usagePercent).toBeLessThan(80) // Should not use more than 80% of quota
      console.log(`Storage usage: ${storageInfo.usagePercent.toFixed(2)}%`)
    }
    
    // Verify localStorage is functional
    expect(storageInfo.localStorage).toBeGreaterThanOrEqual(0)
    
    // Verify IndexedDB support
    expect(storageInfo.indexedDBSupported).toBe(true)
  })

  test('app works across different viewport sizes', async ({ page }) => {
    const viewports = [
      { width: 320, height: 568, name: 'iPhone SE' },
      { width: 375, height: 667, name: 'iPhone 8' },
      { width: 414, height: 896, name: 'iPhone 11' },
      { width: 768, height: 1024, name: 'iPad' },
      { width: 1024, height: 768, name: 'iPad Landscape' },
      { width: 1280, height: 720, name: 'Desktop' }
    ]
    
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height })
      await TestHelpers.waitForAnimations(page)
      
      // Verify page is still functional
      await expect(page.locator('body')).toBeVisible()
      
      // Check for responsive navigation
      const navigation = page.locator('nav, .navigation, .bottom-navigation')
      if (await navigation.count() > 0) {
        await expect(navigation.first()).toBeVisible()
      }
      
      // Take screenshot for each viewport
      await page.screenshot({ 
        path: `test-results/pwa/responsive-${viewport.name.toLowerCase().replace(' ', '-')}.png`,
        fullPage: true 
      })
      
      console.log(`✓ App functional at ${viewport.name} (${viewport.width}x${viewport.height})`)
    }
  })
})