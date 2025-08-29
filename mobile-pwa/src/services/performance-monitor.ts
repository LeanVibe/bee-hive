/**
 * Advanced Performance Monitoring for Mobile PWA
 * Tracks Core Web Vitals and custom PWA metrics
 */

import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

export class PWAPerformanceMonitor {
  private static instance: PWAPerformanceMonitor;
  private metrics: Map<string, any> = new Map();
  private observers: PerformanceObserver[] = [];

  static getInstance() {
    if (!this.instance) {
      this.instance = new PWAPerformanceMonitor();
    }
    return this.instance;
  }

  async initialize() {
    console.log('Initializing PWA performance monitoring...');
    
    // Track Core Web Vitals
    this.trackCoreWebVitals();
    
    // Track custom PWA metrics
    this.trackPWAMetrics();
    
    // Track offline/online transitions
    this.trackConnectionQuality();
    
    // Track service worker performance
    this.trackServiceWorkerMetrics();
  }

  private trackCoreWebVitals() {
    // Cumulative Layout Shift
    getCLS((metric) => {
      this.recordMetric('CLS', metric.value);
      this.reportToAnalytics('web-vital', {
        name: 'CLS',
        value: metric.value,
        rating: metric.value <= 0.1 ? 'good' : metric.value <= 0.25 ? 'needs-improvement' : 'poor'
      });
    });

    // First Input Delay
    getFID((metric) => {
      this.recordMetric('FID', metric.value);
      this.reportToAnalytics('web-vital', {
        name: 'FID',
        value: metric.value,
        rating: metric.value <= 100 ? 'good' : metric.value <= 300 ? 'needs-improvement' : 'poor'
      });
    });

    // First Contentful Paint
    getFCP((metric) => {
      this.recordMetric('FCP', metric.value);
      this.reportToAnalytics('web-vital', {
        name: 'FCP',
        value: metric.value,
        rating: metric.value <= 1800 ? 'good' : metric.value <= 3000 ? 'needs-improvement' : 'poor'
      });
    });

    // Largest Contentful Paint
    getLCP((metric) => {
      this.recordMetric('LCP', metric.value);
      this.reportToAnalytics('web-vital', {
        name: 'LCP',
        value: metric.value,
        rating: metric.value <= 2500 ? 'good' : metric.value <= 4000 ? 'needs-improvement' : 'poor'
      });
    });

    // Time to First Byte
    getTTFB((metric) => {
      this.recordMetric('TTFB', metric.value);
      this.reportToAnalytics('web-vital', {
        name: 'TTFB',
        value: metric.value,
        rating: metric.value <= 800 ? 'good' : metric.value <= 1800 ? 'needs-improvement' : 'poor'
      });
    });
  }

  private trackPWAMetrics() {
    // App Shell Load Time
    const navigationEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    if (navigationEntry) {
      const appShellTime = navigationEntry.loadEventEnd - navigationEntry.fetchStart;
      this.recordMetric('AppShellLoadTime', appShellTime);
    }

    // Service Worker Registration Time
    if ('serviceWorker' in navigator) {
      const swRegistrationStart = performance.now();
      navigator.serviceWorker.ready.then(() => {
        const swRegistrationTime = performance.now() - swRegistrationStart;
        this.recordMetric('ServiceWorkerRegistrationTime', swRegistrationTime);
      });
    }

    // IndexedDB Performance
    this.trackIndexedDBPerformance();
  }

  private trackConnectionQuality() {
    // Network Information API
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      
      const recordConnectionMetrics = () => {
        this.recordMetric('ConnectionType', connection.effectiveType);
        this.recordMetric('ConnectionDownlink', connection.downlink);
        this.recordMetric('ConnectionRTT', connection.rtt);
      };

      recordConnectionMetrics();
      connection.addEventListener('change', recordConnectionMetrics);
    }

    // Online/Offline tracking
    window.addEventListener('online', () => {
      this.recordMetric('ConnectionStatus', 'online');
      this.reportToAnalytics('connection-change', { status: 'online', timestamp: Date.now() });
    });

    window.addEventListener('offline', () => {
      this.recordMetric('ConnectionStatus', 'offline');
      this.reportToAnalytics('connection-change', { status: 'offline', timestamp: Date.now() });
    });
  }

  private trackServiceWorkerMetrics() {
    if ('serviceWorker' in navigator) {
      // Track service worker update cycles
      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data && event.data.type === 'performance-metric') {
          this.recordMetric(`SW_${event.data.name}`, event.data.value);
        }
      });
    }
  }

  private async trackIndexedDBPerformance() {
    try {
      const start = performance.now();
      
      // Test IndexedDB read performance
      const request = indexedDB.open('PerformanceTestDB', 1);
      request.onsuccess = () => {
        const dbOpenTime = performance.now() - start;
        this.recordMetric('IndexedDBOpenTime', dbOpenTime);
        request.result.close();
      };
    } catch (error) {
      console.error('Failed to test IndexedDB performance:', error);
    }
  }

  private recordMetric(name: string, value: any) {
    this.metrics.set(name, {
      value,
      timestamp: Date.now(),
      url: window.location.pathname
    });
    
    console.log(`Performance Metric - ${name}:`, value);
  }

  private reportToAnalytics(event: string, data: any) {
    // In a real implementation, this would send to analytics service
    console.log('Analytics Event:', event, data);
    
    // Could integrate with Google Analytics, custom analytics endpoint, etc.
    // Example: gtag('event', event, data);
  }

  // Public API for accessing metrics
  getMetric(name: string) {
    return this.metrics.get(name);
  }

  getAllMetrics() {
    return Object.fromEntries(this.metrics);
  }

  // Performance budget checking
  checkPerformanceBudget() {
    const budget = {
      FCP: 1800,  // First Contentful Paint < 1.8s
      LCP: 2500,  // Largest Contentful Paint < 2.5s
      FID: 100,   // First Input Delay < 100ms
      CLS: 0.1,   // Cumulative Layout Shift < 0.1
      TTFB: 800   // Time to First Byte < 800ms
    };

    const results = {};
    for (const [metricName, budgetValue] of Object.entries(budget)) {
      const metric = this.getMetric(metricName);
      if (metric) {
        results[metricName] = {
          value: metric.value,
          budget: budgetValue,
          withinBudget: metric.value <= budgetValue
        };
      }
    }

    return results;
  }
}

// Initialize performance monitoring
export const initializePerformanceMonitoring = () => {
  if (typeof window !== 'undefined') {
    const monitor = PWAPerformanceMonitor.getInstance();
    monitor.initialize();
    return monitor;
  }
};