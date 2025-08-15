import { test, expect } from '@playwright/test'
import { TestHelpers } from '../../utils/test-helpers'

/**
 * Performance Validation Tests - Core Web Vitals & Load Times
 * 
 * Validates performance requirements:
 * - Core Web Vitals (FCP, LCP, CLS, FID)
 * - Page load times and bundle size
 * - Memory usage and resource optimization
 * - Mobile performance specifically
 * - Network condition tolerance
 * - Performance regression detection
 */

test.describe('Performance Validation - Core Web Vitals', () => {
  
  test.beforeEach(async ({ page }) => {
    // Clear cache to ensure fresh measurements
    await page.context().clearCookies()
    await page.context().clearPermissions()
  })

  test('dashboard loads within 2 seconds as per PRD requirement', async ({ page }) => {
    const startTime = Date.now()
    
    // Navigate and measure total load time
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    
    const totalLoadTime = Date.now() - startTime
    
    // PRD requirement: Dashboard loads <2s
    expect(totalLoadTime).toBeLessThan(2000)
    
    console.log(`Dashboard load time: ${totalLoadTime}ms (requirement: <2000ms)`)
    
    // Also verify the page is actually functional
    await expect(page.locator('body')).toBeVisible()
    await expect(page.locator('text=HiveOps, text=Agent Dashboard')).toBeVisible()
  })

  test('core web vitals meet performance standards', async ({ page }) => {
    await page.goto('/')
    
    // Wait for page to fully load
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(3000) // Allow for LCP measurement
    
    // Collect Core Web Vitals
    const vitals = await page.evaluate(() => {
      return new Promise((resolve) => {
        const vitals: any = {}
        
        // First Contentful Paint (FCP)
        const fcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries()
          const fcpEntry = entries.find(entry => entry.name === 'first-contentful-paint')
          if (fcpEntry) {
            vitals.fcp = fcpEntry.startTime
          }
        })
        fcpObserver.observe({ entryTypes: ['paint'] })
        
        // Largest Contentful Paint (LCP)
        const lcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries()
          const lastEntry = entries[entries.length - 1]
          vitals.lcp = lastEntry.startTime
        })
        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] })
        
        // Cumulative Layout Shift (CLS)
        let clsValue = 0
        const clsObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (!(entry as any).hadRecentInput) {
              clsValue += (entry as any).value
            }
          }
          vitals.cls = clsValue
        })
        clsObserver.observe({ entryTypes: ['layout-shift'] })
        
        // First Input Delay (FID) - simulated
        const fidObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            vitals.fid = (entry as any).processingStart - entry.startTime
          }
        })
        fidObserver.observe({ entryTypes: ['first-input'] })
        
        // Navigation timing
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
        vitals.domContentLoaded = navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart
        vitals.loadComplete = navigation.loadEventEnd - navigation.loadEventStart
        vitals.ttfb = navigation.responseStart - navigation.requestStart
        
        // Give observers time to collect data
        setTimeout(() => {
          resolve(vitals)
        }, 2000)
      })
    })
    
    console.log('Core Web Vitals:', vitals)
    
    // Validate Core Web Vitals against Google's thresholds
    if (vitals.fcp) {
      expect(vitals.fcp).toBeLessThan(1800) // Good: <1.8s
      console.log(`✓ FCP: ${vitals.fcp.toFixed(0)}ms (Good: <1800ms)`)
    }
    
    if (vitals.lcp) {
      expect(vitals.lcp).toBeLessThan(2500) // Good: <2.5s
      console.log(`✓ LCP: ${vitals.lcp.toFixed(0)}ms (Good: <2500ms)`)
    }
    
    if (vitals.cls !== undefined) {
      expect(vitals.cls).toBeLessThan(0.1) // Good: <0.1
      console.log(`✓ CLS: ${vitals.cls.toFixed(3)} (Good: <0.1)`)
    }
    
    if (vitals.fid) {
      expect(vitals.fid).toBeLessThan(100) // Good: <100ms
      console.log(`✓ FID: ${vitals.fid.toFixed(0)}ms (Good: <100ms)`)
    }
    
    // Additional metrics
    if (vitals.ttfb) {
      expect(vitals.ttfb).toBeLessThan(600) // Good TTFB: <600ms
      console.log(`✓ TTFB: ${vitals.ttfb.toFixed(0)}ms (Good: <600ms)`)
    }
  })

  test('bundle size is optimized and under limits', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    
    // Analyze resource loading
    const resourceStats = await page.evaluate(() => {
      const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
      
      const stats = {
        totalSize: 0,
        jsSize: 0,
        cssSize: 0,
        imageSize: 0,
        resourceCount: resources.length,
        resources: []
      }
      
      resources.forEach(resource => {
        const size = resource.transferSize || resource.encodedBodySize || 0
        stats.totalSize += size
        
        const resourceInfo = {
          name: resource.name,
          size: size,
          type: resource.initiatorType,
          duration: resource.duration
        }
        
        stats.resources.push(resourceInfo)
        
        if (resource.name.includes('.js')) {
          stats.jsSize += size
        } else if (resource.name.includes('.css')) {
          stats.cssSize += size
        } else if (resource.initiatorType === 'img') {
          stats.imageSize += size
        }
      })
      
      return stats
    })
    
    console.log('Resource statistics:', {
      totalSize: `${(resourceStats.totalSize / 1024).toFixed(1)} KB`,
      jsSize: `${(resourceStats.jsSize / 1024).toFixed(1)} KB`,
      cssSize: `${(resourceStats.cssSize / 1024).toFixed(1)} KB`,
      imageSize: `${(resourceStats.imageSize / 1024).toFixed(1)} KB`,
      resourceCount: resourceStats.resourceCount
    })
    
    // Bundle size targets (adjustable based on requirements)
    expect(resourceStats.totalSize).toBeLessThan(2 * 1024 * 1024) // <2MB total
    expect(resourceStats.jsSize).toBeLessThan(1 * 1024 * 1024) // <1MB JS
    expect(resourceStats.cssSize).toBeLessThan(200 * 1024) // <200KB CSS
    
    // Check for largest resources
    const largestResources = resourceStats.resources
      .sort((a, b) => b.size - a.size)
      .slice(0, 5)
    
    console.log('Largest resources:', largestResources.map(r => ({
      name: r.name.split('/').pop(),
      size: `${(r.size / 1024).toFixed(1)} KB`
    })))
  })

  test('memory usage remains within acceptable limits', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    
    // Get initial memory baseline
    const initialMemory = await page.evaluate(() => {
      if ('memory' in performance) {
        const memory = (performance as any).memory
        return {
          usedJSHeapSize: memory.usedJSHeapSize,
          totalJSHeapSize: memory.totalJSHeapSize,
          jsHeapSizeLimit: memory.jsHeapSizeLimit
        }
      }
      return null
    })
    
    if (initialMemory) {
      console.log('Initial memory usage:', {
        used: `${(initialMemory.usedJSHeapSize / 1024 / 1024).toFixed(1)} MB`,
        total: `${(initialMemory.totalJSHeapSize / 1024 / 1024).toFixed(1)} MB`,
        limit: `${(initialMemory.jsHeapSizeLimit / 1024 / 1024).toFixed(1)} MB`
      })
    }
    
    // Simulate some user interactions to stress test memory
    const interactions = [
      () => page.reload(),
      () => page.mouse.move(100, 100),
      () => page.mouse.move(200, 200),
      () => page.keyboard.press('Tab'),
      () => page.evaluate(() => window.scrollTo(0, 100))
    ]
    
    for (const interaction of interactions) {
      await interaction()
      await page.waitForTimeout(500)
    }
    
    // Check memory after interactions
    const finalMemory = await page.evaluate(() => {
      if ('memory' in performance) {
        const memory = (performance as any).memory
        return {
          usedJSHeapSize: memory.usedJSHeapSize,
          totalJSHeapSize: memory.totalJSHeapSize,
          jsHeapSizeLimit: memory.jsHeapSizeLimit
        }
      }
      return null
    })
    
    if (initialMemory && finalMemory) {
      const memoryIncrease = finalMemory.usedJSHeapSize - initialMemory.usedJSHeapSize
      const memoryIncreasePercent = (memoryIncrease / initialMemory.usedJSHeapSize) * 100
      
      console.log('Memory change after interactions:', {
        increase: `${(memoryIncrease / 1024 / 1024).toFixed(1)} MB`,
        increasePercent: `${memoryIncreasePercent.toFixed(1)}%`,
        finalUsage: `${(finalMemory.usedJSHeapSize / 1024 / 1024).toFixed(1)} MB`
      })
      
      // Memory shouldn't increase dramatically during normal usage
      expect(memoryIncreasePercent).toBeLessThan(100) // <100% increase
      
      // Memory usage should be reasonable
      const memoryUsagePercent = (finalMemory.usedJSHeapSize / finalMemory.jsHeapSizeLimit) * 100
      expect(memoryUsagePercent).toBeLessThan(50) // <50% of heap limit
    }
  })

  test('mobile performance meets standards', async ({ page }) => {
    // Set mobile viewport and simulate mobile conditions
    await page.setViewportSize({ width: 375, height: 667 })
    
    // Simulate mobile CPU throttling
    const client = await page.context().newCDPSession(page)
    await client.send('Emulation.setCPUThrottlingRate', { rate: 4 }) // 4x slower
    
    const startTime = Date.now()
    
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    
    const mobileLoadTime = Date.now() - startTime
    
    // Mobile should still load reasonably fast even with throttling
    expect(mobileLoadTime).toBeLessThan(4000) // <4s on throttled mobile
    
    console.log(`Mobile load time (throttled): ${mobileLoadTime}ms`)
    
    // Check mobile-specific performance
    const mobileMetrics = await page.evaluate(() => {
      return {
        touchEvents: 'ontouchstart' in window,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight
        },
        devicePixelRatio: window.devicePixelRatio,
        connection: (navigator as any).connection?.effectiveType || 'unknown'
      }
    })
    
    console.log('Mobile environment:', mobileMetrics)
    
    // Verify touch support
    expect(mobileMetrics.touchEvents).toBe(true)
    
    // Disable CPU throttling
    await client.send('Emulation.setCPUThrottlingRate', { rate: 1 })
  })

  test('performance under slow network conditions', async ({ page }) => {
    // Simulate slow 3G connection
    const client = await page.context().newCDPSession(page)
    await client.send('Network.emulateNetworkConditions', {
      offline: false,
      downloadThroughput: 1.5 * 1024 * 1024 / 8, // 1.5 Mbps in bytes/sec
      uploadThroughput: 750 * 1024 / 8, // 750 Kbps in bytes/sec
      latency: 300 // 300ms latency
    })
    
    const startTime = Date.now()
    
    await page.goto('/')
    await page.waitForLoadState('domcontentloaded')
    
    const slowNetworkLoadTime = Date.now() - startTime
    
    console.log(`Load time on slow network: ${slowNetworkLoadTime}ms`)
    
    // Should still be usable on slow networks (graceful degradation)
    expect(slowNetworkLoadTime).toBeLessThan(10000) // <10s on slow network
    
    // Verify basic functionality works
    await expect(page.locator('body')).toBeVisible()
    
    // Check for loading states
    const loadingIndicators = page.locator('.loading, .spinner, .skeleton')
    if (await loadingIndicators.count() > 0) {
      console.log('✓ Loading indicators present for slow network')
    }
    
    // Restore normal network
    await client.send('Network.emulateNetworkConditions', {
      offline: false,
      downloadThroughput: -1,
      uploadThroughput: -1,
      latency: 0
    })
  })

  test('resource caching is effective', async ({ page }) => {
    // First visit - measure uncached performance
    const firstVisitStart = Date.now()
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    const firstVisitTime = Date.now() - firstVisitStart
    
    // Count resources loaded
    const firstVisitResources = await page.evaluate(() => {
      return performance.getEntriesByType('resource').length
    })
    
    // Second visit - should use cache
    const secondVisitStart = Date.now()
    await page.reload()
    await page.waitForLoadState('networkidle')
    const secondVisitTime = Date.now() - secondVisitStart
    
    const secondVisitResources = await page.evaluate(() => {
      return performance.getEntriesByType('resource').length
    })
    
    // Check cache usage
    const cachedResources = await page.evaluate(() => {
      const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
      return resources.filter(resource => 
        resource.transferSize === 0 || // Served from cache
        resource.transferSize < resource.encodedBodySize // Partial cache hit
      ).length
    })
    
    console.log('Caching effectiveness:', {
      firstVisit: `${firstVisitTime}ms (${firstVisitResources} resources)`,
      secondVisit: `${secondVisitTime}ms (${secondVisitResources} resources)`,
      cachedResources: cachedResources,
      speedImprovement: `${((firstVisitTime - secondVisitTime) / firstVisitTime * 100).toFixed(1)}%`
    })
    
    // Second visit should be faster due to caching
    expect(secondVisitTime).toBeLessThan(firstVisitTime)
    
    // Should have some cached resources
    expect(cachedResources).toBeGreaterThan(0)
  })

  test('javascript execution performance is optimized', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    
    // Measure JavaScript execution performance
    const jsPerformance = await page.evaluate(() => {
      const start = performance.now()
      
      // Simulate some typical operations
      const tests = []
      
      // DOM query performance
      const domStart = performance.now()
      for (let i = 0; i < 100; i++) {
        document.querySelectorAll('div, span, button')
      }
      const domTime = performance.now() - domStart
      tests.push({ test: 'DOM queries', time: domTime })
      
      // Array operations
      const arrayStart = performance.now()
      const arr = Array.from({ length: 1000 }, (_, i) => i)
      arr.map(x => x * 2).filter(x => x % 2 === 0).reduce((a, b) => a + b, 0)
      const arrayTime = performance.now() - arrayStart
      tests.push({ test: 'Array operations', time: arrayTime })
      
      // Object operations
      const objStart = performance.now()
      const obj = {}
      for (let i = 0; i < 1000; i++) {
        obj[`key${i}`] = `value${i}`
      }
      Object.keys(obj).forEach(key => obj[key])
      const objTime = performance.now() - objStart
      tests.push({ test: 'Object operations', time: objTime })
      
      const totalTime = performance.now() - start
      
      return { tests, totalTime }
    })
    
    console.log('JavaScript performance tests:', jsPerformance.tests.map(t => 
      `${t.test}: ${t.time.toFixed(2)}ms`
    ))
    
    // All operations should complete quickly
    expect(jsPerformance.totalTime).toBeLessThan(100) // <100ms total
    
    // Individual operations should be fast
    jsPerformance.tests.forEach(test => {
      expect(test.time).toBeLessThan(50) // <50ms per test
    })
  })

  test('performance monitoring and metrics collection works', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    
    // Check if performance monitoring is implemented
    const performanceData = await page.evaluate(() => {
      const data: any = {}
      
      // Check for Performance Observer support
      data.performanceObserverSupported = 'PerformanceObserver' in window
      
      // Check for User Timing API
      data.userTimingSupported = 'mark' in performance && 'measure' in performance
      
      // Check for Resource Timing API
      data.resourceTimingSupported = 'getEntriesByType' in performance
      
      // Check for Navigation Timing API
      data.navigationTimingSupported = 'getEntriesByType' in performance && 
        performance.getEntriesByType('navigation').length > 0
      
      // Look for custom performance marks
      const marks = performance.getEntriesByType('mark')
      data.customMarks = marks.length
      
      // Look for custom measures
      const measures = performance.getEntriesByType('measure')
      data.customMeasures = measures.length
      
      // Check for web vitals measurement
      data.webVitalsImplemented = !!(window as any).webVitals || 
        !!(window as any).gtag || 
        marks.some(mark => mark.name.includes('vitals'))
      
      return data
    })
    
    console.log('Performance monitoring capabilities:', performanceData)
    
    // Modern browsers should support these APIs
    expect(performanceData.performanceObserverSupported).toBe(true)
    expect(performanceData.resourceTimingSupported).toBe(true)
    expect(performanceData.navigationTimingSupported).toBe(true)
    
    // Application should implement some performance monitoring
    if (performanceData.customMarks > 0 || performanceData.customMeasures > 0) {
      console.log('✓ Custom performance monitoring detected')
    }
  })
})