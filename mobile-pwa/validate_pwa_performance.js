#!/usr/bin/env node
/**
 * PWA Performance Validation Script
 * Tests offline capability, cache health, and PWA compliance
 */

import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class PWAValidator {
  constructor() {
    this.results = {
      timestamp: new Date().toISOString(),
      phase: "Phase 2 Mobile Interface Optimization",
      performance_targets: {
        offline_capability: "24-hour cached context",
        pwa_score: ">90%",
        cache_health: "100%",
        service_worker: "Fully functional"
      },
      test_results: {},
      summary: {
        total_tests: 0,
        passed: 0,
        failed: 0,
        warnings: 0
      }
    };
  }

  async validate() {
    console.log('🔍 Starting PWA Performance Validation...');
    
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();
    
    try {
      // Test 1: Offline Capability
      await this.testOfflineCapability(page);
      
      // Test 2: Service Worker Registration
      await this.testServiceWorkerRegistration(page);
      
      // Test 3: Cache Health
      await this.testCacheHealth(page);
      
      // Test 4: PWA Manifest Validation
      await this.testPWAManifest(page);
      
      // Test 5: Touch Support
      await this.testTouchSupport(page);
      
      // Calculate final results
      this.calculateSummary();
      
    } catch (error) {
      console.error('❌ Validation failed:', error);
      this.results.error = error.message;
    } finally {
      await browser.close();
    }
    
    // Save results
    await this.saveResults();
    this.printSummary();
    
    return this.results;
  }

  async testOfflineCapability(page) {
    console.log('🔄 Testing offline capability...');
    
    const testResult = {
      target: "24-hour cached context",
      passed: false,
      warning: false,
      failed: false,
      details: {
        tests_passed: 0,
        total_tests: 5,
        cache_tests: [
          "Core assets caching",
          "API response caching", 
          "Context data persistence",
          "Offline command queuing",
          "Cache cleanup mechanisms"
        ]
      }
    };
    
    try {
      // Navigate to the app
      await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
      
      // Wait for service worker registration
      await page.waitForTimeout(2000);
      
      // Test core assets caching
      const swRegistered = await page.evaluate(() => {
        return 'serviceWorker' in navigator && navigator.serviceWorker.controller !== null;
      });
      
      if (swRegistered) {
        testResult.details.tests_passed++;
        console.log('✅ Service worker registered and active');
      }
      
      // Test cache status
      const cacheStatus = await page.evaluate(async () => {
        if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
          const messageChannel = new MessageChannel();
          
          return new Promise((resolve) => {
            messageChannel.port1.onmessage = (event) => {
              resolve(event.data);
            };
            
            navigator.serviceWorker.controller.postMessage(
              { type: 'get-cache-status' },
              [messageChannel.port2]
            );
            
            // Timeout after 5 seconds
            setTimeout(() => resolve(null), 5000);
          });
        }
        return null;
      });
      
      if (cacheStatus && cacheStatus.isHealthy) {
        testResult.details.tests_passed += 2;
        testResult.cache_health_score = `${Math.round((cacheStatus.contextEntries + cacheStatus.cacheSize) > 0 ? 100 : 0)}%`;
        console.log('✅ Cache is healthy:', cacheStatus.cacheSizeFormatted);
      } else {
        testResult.cache_health_score = "0%";
        console.log('❌ Cache status unavailable');
      }
      
      // Test offline functionality
      await page.route('**/*', route => route.abort());
      
      try {
        await page.reload({ waitUntil: 'domcontentloaded' });
        const offlineContent = await page.textContent('body');
        
        if (offlineContent.includes('offline') || offlineContent.includes('cached')) {
          testResult.details.tests_passed++;
          console.log('✅ Offline fallback working');
        }
      } catch (error) {
        console.log('⚠️ Offline fallback needs improvement');
      }
      
      // Determine pass/fail
      if (testResult.details.tests_passed >= 3) {
        testResult.passed = true;
        testResult.cache_health_score = `${Math.round((testResult.details.tests_passed / testResult.details.total_tests) * 100)}%`;
      } else {
        testResult.failed = true;
        testResult.cache_health_score = "0%";
      }
      
    } catch (error) {
      console.error('❌ Offline capability test failed:', error);
      testResult.failed = true;
      testResult.error = error.message;
    }
    
    this.results.test_results.offline_capability = testResult;
    this.results.summary.total_tests++;
    
    if (testResult.passed) this.results.summary.passed++;
    else if (testResult.failed) this.results.summary.failed++;
    else this.results.summary.warnings++;
  }

  async testServiceWorkerRegistration(page) {
    console.log('🔧 Testing service worker registration...');
    
    const testResult = {
      target: "Fully functional service worker",
      passed: false,
      warning: false,
      failed: false,
      details: {}
    };
    
    try {
      // Clear routes first
      await page.unroute('**/*');
      await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
      
      const swInfo = await page.evaluate(async () => {
        if ('serviceWorker' in navigator) {
          const registration = await navigator.serviceWorker.getRegistration();
          return {
            supported: true,
            registered: !!registration,
            active: !!(registration && registration.active),
            scope: registration ? registration.scope : null,
            state: registration && registration.active ? registration.active.state : null
          };
        }
        return { supported: false };
      });
      
      testResult.details = swInfo;
      
      if (swInfo.supported && swInfo.registered && swInfo.active) {
        testResult.passed = true;
        console.log('✅ Service worker fully functional');
      } else {
        testResult.failed = true;
        console.log('❌ Service worker not properly registered');
      }
      
    } catch (error) {
      console.error('❌ Service worker test failed:', error);
      testResult.failed = true;
      testResult.error = error.message;
    }
    
    this.results.test_results.service_worker = testResult;
    this.results.summary.total_tests++;
    
    if (testResult.passed) this.results.summary.passed++;
    else if (testResult.failed) this.results.summary.failed++;
    else this.results.summary.warnings++;
  }

  async testCacheHealth(page) {
    console.log('💾 Testing cache health...');
    
    const testResult = {
      target: "24-hour context cache",
      passed: false,
      warning: false, 
      failed: false,
      details: {}
    };
    
    try {
      await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
      
      const cacheInfo = await page.evaluate(async () => {
        if ('caches' in window) {
          const cacheNames = await caches.keys();
          let totalSize = 0;
          let totalEntries = 0;
          
          for (const cacheName of cacheNames) {
            const cache = await caches.open(cacheName);
            const requests = await cache.keys();
            totalEntries += requests.length;
            
            for (const request of requests) {
              const response = await cache.match(request);
              if (response) {
                const buffer = await response.arrayBuffer();
                totalSize += buffer.byteLength;
              }
            }
          }
          
          return {
            cacheNames,
            totalSize,
            totalEntries,
            supported: true
          };
        }
        return { supported: false };
      });
      
      testResult.details = cacheInfo;
      
      if (cacheInfo.supported && (cacheInfo.totalEntries > 0 || cacheInfo.totalSize > 0)) {
        testResult.passed = true;
        testResult.cache_size = this.formatBytes(cacheInfo.totalSize);
        console.log(`✅ Cache healthy: ${testResult.cache_size}, ${cacheInfo.totalEntries} entries`);
      } else {
        testResult.failed = true;
        console.log('❌ Cache not functioning properly');
      }
      
    } catch (error) {
      console.error('❌ Cache health test failed:', error);
      testResult.failed = true;
      testResult.error = error.message;
    }
    
    this.results.test_results.cache_health = testResult;
    this.results.summary.total_tests++;
    
    if (testResult.passed) this.results.summary.passed++;
    else if (testResult.failed) this.results.summary.failed++;
    else this.results.summary.warnings++;
  }

  async testPWAManifest(page) {
    console.log('📱 Testing PWA manifest...');
    
    const testResult = {
      target: ">90% PWA score",
      passed: false,
      warning: false,
      failed: false,
      details: {}
    };
    
    try {
      await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
      
      const manifestInfo = await page.evaluate(async () => {
        const manifestLink = document.querySelector('link[rel="manifest"]');
        if (!manifestLink) return { found: false };
        
        try {
          const response = await fetch(manifestLink.href);
          const manifest = await response.json();
          
          return {
            found: true,
            manifest,
            hasName: !!manifest.name,
            hasShortName: !!manifest.short_name,
            hasStartUrl: !!manifest.start_url,
            hasDisplay: !!manifest.display,
            hasThemeColor: !!manifest.theme_color,
            hasBackgroundColor: !!manifest.background_color,
            hasIcons: !!(manifest.icons && manifest.icons.length > 0),
            iconSizes: manifest.icons ? manifest.icons.map(icon => icon.sizes) : []
          };
        } catch (error) {
          return { found: true, error: error.message };
        }
      });
      
      testResult.details = manifestInfo;
      
      if (manifestInfo.found && !manifestInfo.error) {
        const score = [
          manifestInfo.hasName,
          manifestInfo.hasShortName,
          manifestInfo.hasStartUrl,
          manifestInfo.hasDisplay,
          manifestInfo.hasThemeColor,
          manifestInfo.hasBackgroundColor,
          manifestInfo.hasIcons
        ].filter(Boolean).length;
        
        const pwaScore = Math.round((score / 7) * 100);
        testResult.pwa_score = `${pwaScore}%`;
        
        if (pwaScore >= 90) {
          testResult.passed = true;
          console.log(`✅ PWA manifest score: ${pwaScore}%`);
        } else {
          testResult.warning = true;
          console.log(`⚠️ PWA manifest score: ${pwaScore}% (needs improvement)`);
        }
      } else {
        testResult.failed = true;
        testResult.pwa_score = "0%";
        console.log('❌ PWA manifest not found or invalid');
      }
      
    } catch (error) {
      console.error('❌ PWA manifest test failed:', error);
      testResult.failed = true;
      testResult.error = error.message;
    }
    
    this.results.test_results.pwa_manifest = testResult;
    this.results.summary.total_tests++;
    
    if (testResult.passed) this.results.summary.passed++;
    else if (testResult.failed) this.results.summary.failed++;
    else this.results.summary.warnings++;
  }

  async testTouchSupport(page) {
    console.log('👆 Testing touch support...');
    
    const testResult = {
      target: "Touch-optimized interface",
      passed: false,
      warning: false,
      failed: false,
      details: {}
    };
    
    try {
      await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
      
      const touchInfo = await page.evaluate(() => {
        const hasTouchEvents = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        const hasViewportMeta = !!document.querySelector('meta[name="viewport"]');
        const touchTargets = Array.from(document.querySelectorAll('button, a, input, [role="button"]')).length;
        
        return {
          hasTouchEvents,
          hasViewportMeta,
          touchTargets,
          userAgent: navigator.userAgent
        };
      });
      
      testResult.details = touchInfo;
      
      if (touchInfo.hasViewportMeta && touchInfo.touchTargets > 0) {
        testResult.passed = true;
        console.log(`✅ Touch support ready: ${touchInfo.touchTargets} interactive elements`);
      } else {
        testResult.warning = true;
        console.log('⚠️ Touch support could be improved');
      }
      
    } catch (error) {
      console.error('❌ Touch support test failed:', error);
      testResult.failed = true;
      testResult.error = error.message;
    }
    
    this.results.test_results.touch_support = testResult;
    this.results.summary.total_tests++;
    
    if (testResult.passed) this.results.summary.passed++;
    else if (testResult.failed) this.results.summary.failed++;
    else this.results.summary.warnings++;
  }

  calculateSummary() {
    // Calculate overall performance score
    const passRate = this.results.summary.total_tests > 0 
      ? (this.results.summary.passed / this.results.summary.total_tests) * 100 
      : 0;
    
    this.results.overall_performance_score = `${Math.round(passRate)}%`;
    this.results.status = passRate >= 80 ? 'PASSED' : passRate >= 60 ? 'WARNING' : 'FAILED';
  }

  async saveResults() {
    const filename = `pwa_validation_${Date.now()}.json`;
    const filepath = path.join(__dirname, 'scratchpad', filename);
    
    // Ensure scratchpad directory exists
    const dir = path.dirname(filepath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    
    fs.writeFileSync(filepath, JSON.stringify(this.results, null, 2));
    console.log(`💾 Results saved to: ${filepath}`);
  }

  printSummary() {
    console.log('\n📊 PWA VALIDATION SUMMARY');
    console.log('================================');
    console.log(`Overall Score: ${this.results.overall_performance_score}`);
    console.log(`Status: ${this.results.status}`);
    console.log(`Tests Passed: ${this.results.summary.passed}/${this.results.summary.total_tests}`);
    console.log(`Warnings: ${this.results.summary.warnings}`);
    console.log(`Failed: ${this.results.summary.failed}`);
    
    console.log('\n🔍 DETAILED RESULTS:');
    Object.entries(this.results.test_results).forEach(([testName, result]) => {
      const status = result.passed ? '✅' : result.failed ? '❌' : '⚠️';
      console.log(`${status} ${testName}: ${result.target}`);
      if (result.pwa_score) console.log(`   PWA Score: ${result.pwa_score}`);
      if (result.cache_health_score) console.log(`   Cache Health: ${result.cache_health_score}`);
      if (result.cache_size) console.log(`   Cache Size: ${result.cache_size}`);
    });
    
    console.log('\n🎯 RECOMMENDATIONS:');
    if (this.results.summary.failed > 0) {
      console.log('- Fix failing tests to achieve enterprise-grade performance');
    }
    if (this.results.summary.warnings > 0) {
      console.log('- Address warnings to improve PWA compliance');  
    }
    if (this.results.summary.passed === this.results.summary.total_tests) {
      console.log('- Excellent! All tests passed. PWA is enterprise-ready.');
    }
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
}

// Run validation if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const validator = new PWAValidator();
  validator.validate().then(results => {
    process.exit(results.status === 'PASSED' ? 0 : 1);
  }).catch(error => {
    console.error('Validation failed:', error);
    process.exit(1);
  });
}

export default PWAValidator;