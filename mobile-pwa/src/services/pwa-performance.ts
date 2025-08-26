/**
 * PWA Performance Optimization Service
 * 
 * Provides advanced performance monitoring and optimization features
 * specifically designed for Phase 4 enterprise mobile PWA requirements:
 * 
 * - Core Web Vitals monitoring
 * - Resource loading optimization
 * - Memory usage tracking
 * - Network performance analysis
 * - Battery usage optimization
 * - Background sync management
 */

import { WebVitals } from 'web-vitals';

interface PerformanceMetrics {
  // Core Web Vitals
  lcp: number | null; // Largest Contentful Paint
  fid: number | null; // First Input Delay
  cls: number | null; // Cumulative Layout Shift
  fcp: number | null; // First Contentful Paint
  ttfb: number | null; // Time to First Byte
  
  // Custom metrics
  memoryUsage: number;
  networkLatency: number;
  batteryLevel: number | null;
  connectionType: string;
  renderTime: number;
  bundleSize: number;
  cacheHitRate: number;
  
  timestamp: number;
}

interface PerformanceAlert {
  id: string;
  type: 'warning' | 'error' | 'info';
  metric: string;
  message: string;
  value: number;
  threshold: number;
  timestamp: number;
}

interface PWAOptimization {
  preload: string[];
  prefetch: string[];
  defer: string[];
  lazyLoad: boolean;
  imageOptimization: boolean;
  bundleSplitting: boolean;
  serviceWorkerCaching: boolean;
}

class PWAPerformanceService {
  private metrics: PerformanceMetrics[] = [];
  private alerts: PerformanceAlert[] = [];
  private observers: Map<string, PerformanceObserver> = new Map();
  private optimizations: PWAOptimization = {
    preload: [],
    prefetch: [],
    defer: [],
    lazyLoad: true,
    imageOptimization: true,
    bundleSplitting: true,
    serviceWorkerCaching: true
  };

  // Performance thresholds (Lighthouse recommendations)
  private thresholds = {
    lcp: 2500, // Good: < 2.5s
    fid: 100,  // Good: < 100ms
    cls: 0.1,  // Good: < 0.1
    fcp: 1800, // Good: < 1.8s
    ttfb: 800, // Good: < 0.8s
    memoryUsage: 50 * 1024 * 1024, // 50MB
    networkLatency: 300, // 300ms
    batteryLevel: 20 // 20%
  };

  private eventListeners: Map<string, Function[]> = new Map();

  constructor() {
    this.initialize();
  }

  private async initialize() {
    console.log('🚀 PWA Performance Service initializing...');
    
    // Initialize Web Vitals monitoring
    this.initializeWebVitals();
    
    // Initialize performance observers
    this.initializePerformanceObservers();
    
    // Initialize network monitoring
    this.initializeNetworkMonitoring();
    
    // Initialize battery monitoring
    this.initializeBatteryMonitoring();
    
    // Initialize memory monitoring
    this.initializeMemoryMonitoring();
    
    // Initialize resource optimization
    this.initializeResourceOptimization();
    
    // Start performance monitoring loop
    this.startMonitoringLoop();
    
    console.log('✅ PWA Performance Service initialized');
  }

  private initializeWebVitals() {
    try {
      // Note: web-vitals library would be imported at the top
      // For now, we'll simulate the metrics collection
      
      // LCP - Largest Contentful Paint
      this.observeMetric('largest-contentful-paint', (entry) => {
        const lcp = entry.startTime;
        this.updateMetric('lcp', lcp);
        if (lcp > this.thresholds.lcp) {
          this.createAlert('warning', 'lcp', 'LCP exceeds recommended threshold', lcp, this.thresholds.lcp);
        }
      });

      // FID - First Input Delay  
      this.observeMetric('first-input', (entry) => {
        const fid = entry.processingStart - entry.startTime;
        this.updateMetric('fid', fid);
        if (fid > this.thresholds.fid) {
          this.createAlert('warning', 'fid', 'FID exceeds recommended threshold', fid, this.thresholds.fid);
        }
      });

      // CLS - Cumulative Layout Shift
      this.observeMetric('layout-shift', (entry) => {
        if (!entry.hadRecentInput) {
          const cls = entry.value;
          this.updateMetric('cls', cls);
          if (cls > this.thresholds.cls) {
            this.createAlert('warning', 'cls', 'CLS exceeds recommended threshold', cls, this.thresholds.cls);
          }
        }
      });

    } catch (error) {
      console.warn('Web Vitals monitoring setup failed:', error);
    }
  }

  private observeMetric(entryType: string, callback: (entry: any) => void) {
    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          callback(entry);
        }
      });
      
      observer.observe({ entryTypes: [entryType] });
      this.observers.set(entryType, observer);
      
    } catch (error) {
      console.warn(`Failed to observe metric ${entryType}:`, error);
    }
  }

  private initializePerformanceObservers() {
    // Navigation timing
    this.observeMetric('navigation', (entry) => {
      const ttfb = entry.responseStart - entry.fetchStart;
      this.updateMetric('ttfb', ttfb);
      
      if (ttfb > this.thresholds.ttfb) {
        this.createAlert('warning', 'ttfb', 'TTFB exceeds recommended threshold', ttfb, this.thresholds.ttfb);
      }
    });

    // Paint timing
    this.observeMetric('paint', (entry) => {
      if (entry.name === 'first-contentful-paint') {
        const fcp = entry.startTime;
        this.updateMetric('fcp', fcp);
        
        if (fcp > this.thresholds.fcp) {
          this.createAlert('warning', 'fcp', 'FCP exceeds recommended threshold', fcp, this.thresholds.fcp);
        }
      }
    });

    // Resource timing for bundle size analysis
    this.observeMetric('resource', (entry) => {
      if (entry.name.includes('.js') || entry.name.includes('.css')) {
        const size = entry.transferSize || 0;
        this.updateMetric('bundleSize', size);
      }
    });
  }

  private initializeNetworkMonitoring() {
    // Connection monitoring
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      
      const updateConnectionInfo = () => {
        const connectionType = connection.effectiveType || 'unknown';
        const downlink = connection.downlink || 0;
        const rtt = connection.rtt || 0;
        
        this.updateMetric('networkLatency', rtt);
        this.updateCustomMetric('connectionType', connectionType);
        this.updateCustomMetric('downlink', downlink);
        
        if (rtt > this.thresholds.networkLatency) {
          this.createAlert('info', 'networkLatency', 'High network latency detected', rtt, this.thresholds.networkLatency);
        }
      };

      connection.addEventListener('change', updateConnectionInfo);
      updateConnectionInfo();
    }

    // Online/offline monitoring
    window.addEventListener('online', () => {
      this.createAlert('info', 'network', 'Network connection restored', 1, 1);
      this.emit('network-online');
    });

    window.addEventListener('offline', () => {
      this.createAlert('warning', 'network', 'Network connection lost', 0, 1);
      this.emit('network-offline');
    });
  }

  private async initializeBatteryMonitoring() {
    try {
      if ('getBattery' in navigator) {
        const battery = await (navigator as any).getBattery();
        
        const updateBatteryInfo = () => {
          const batteryLevel = Math.round(battery.level * 100);
          this.updateMetric('batteryLevel', batteryLevel);
          
          if (batteryLevel < this.thresholds.batteryLevel) {
            this.createAlert('warning', 'batteryLevel', 'Low battery level detected', batteryLevel, this.thresholds.batteryLevel);
          }
        };

        battery.addEventListener('levelchange', updateBatteryInfo);
        battery.addEventListener('chargingchange', updateBatteryInfo);
        updateBatteryInfo();
      }
    } catch (error) {
      console.warn('Battery monitoring not available:', error);
    }
  }

  private initializeMemoryMonitoring() {
    if ('memory' in performance) {
      const memInfo = (performance as any).memory;
      
      setInterval(() => {
        const memoryUsage = memInfo.usedJSHeapSize;
        this.updateMetric('memoryUsage', memoryUsage);
        
        if (memoryUsage > this.thresholds.memoryUsage) {
          this.createAlert('warning', 'memoryUsage', 'High memory usage detected', memoryUsage, this.thresholds.memoryUsage);
        }
      }, 10000); // Check every 10 seconds
    }
  }

  private initializeResourceOptimization() {
    // Preload critical resources
    this.optimizations.preload = [
      '/manifest.json',
      '/icons/icon-192x192.png',
      '/offline.html'
    ];

    // Prefetch likely needed resources
    this.optimizations.prefetch = [
      '/api/dashboard/live-data',
      '/api/agents',
      '/api/tasks'
    ];

    // Apply optimizations
    this.applyResourceOptimizations();
  }

  private applyResourceOptimizations() {
    const head = document.head;

    // Add preload links
    this.optimizations.preload.forEach(href => {
      const link = document.createElement('link');
      link.rel = 'preload';
      link.href = href;
      link.as = this.getResourceType(href);
      head.appendChild(link);
    });

    // Add prefetch links
    this.optimizations.prefetch.forEach(href => {
      const link = document.createElement('link');
      link.rel = 'prefetch';
      link.href = href;
      head.appendChild(link);
    });
  }

  private getResourceType(href: string): string {
    if (href.includes('.css')) return 'style';
    if (href.includes('.js')) return 'script';
    if (href.includes('.png') || href.includes('.jpg') || href.includes('.webp')) return 'image';
    if (href.includes('.woff') || href.includes('.woff2')) return 'font';
    return 'fetch';
  }

  private startMonitoringLoop() {
    // Collect comprehensive metrics every 30 seconds
    setInterval(() => {
      this.collectComprehensiveMetrics();
    }, 30000);

    // Emit performance updates every 10 seconds
    setInterval(() => {
      this.emit('performance-update', this.getCurrentMetrics());
    }, 10000);
  }

  private collectComprehensiveMetrics() {
    const now = Date.now();
    const renderStart = performance.now();
    
    // Force a small render to measure render time
    requestAnimationFrame(() => {
      const renderTime = performance.now() - renderStart;
      this.updateMetric('renderTime', renderTime);
      
      // Calculate cache hit rate from resource timing
      const cacheHitRate = this.calculateCacheHitRate();
      this.updateMetric('cacheHitRate', cacheHitRate);
      
      // Store comprehensive metrics snapshot
      const metrics = this.getCurrentMetrics();
      this.metrics.push({ ...metrics, timestamp: now });
      
      // Keep only last 100 measurements
      if (this.metrics.length > 100) {
        this.metrics = this.metrics.slice(-100);
      }
      
      // Clean up old alerts (keep last 50)
      if (this.alerts.length > 50) {
        this.alerts = this.alerts.slice(-50);
      }
    });
  }

  private calculateCacheHitRate(): number {
    const resourceEntries = performance.getEntriesByType('resource');
    if (resourceEntries.length === 0) return 100;
    
    const cacheHits = resourceEntries.filter((entry: any) => 
      entry.transferSize === 0 || entry.transferSize < entry.decodedBodySize
    ).length;
    
    return Math.round((cacheHits / resourceEntries.length) * 100);
  }

  private updateMetric(name: keyof PerformanceMetrics, value: number) {
    const currentMetrics = this.getCurrentMetrics();
    (currentMetrics as any)[name] = value;
    
    this.emit('metric-updated', { name, value });
  }

  private updateCustomMetric(name: string, value: any) {
    this.emit('custom-metric-updated', { name, value });
  }

  private createAlert(type: PerformanceAlert['type'], metric: string, message: string, value: number, threshold: number) {
    const alert: PerformanceAlert = {
      id: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      metric,
      message,
      value,
      threshold,
      timestamp: Date.now()
    };
    
    this.alerts.unshift(alert);
    this.emit('performance-alert', alert);
    
    console.warn(`PWA Performance Alert [${type.toUpperCase()}]:`, message, `(${value} > ${threshold})`);
  }

  getCurrentMetrics(): PerformanceMetrics {
    return {
      lcp: null,
      fid: null,
      cls: null,
      fcp: null,
      ttfb: null,
      memoryUsage: (performance as any).memory?.usedJSHeapSize || 0,
      networkLatency: (navigator as any).connection?.rtt || 0,
      batteryLevel: null,
      connectionType: (navigator as any).connection?.effectiveType || 'unknown',
      renderTime: 0,
      bundleSize: 0,
      cacheHitRate: 100,
      timestamp: Date.now()
    };
  }

  getMetricsHistory(): PerformanceMetrics[] {
    return [...this.metrics];
  }

  getAlerts(): PerformanceAlert[] {
    return [...this.alerts];
  }

  getOptimizationSuggestions(): string[] {
    const suggestions: string[] = [];
    const latest = this.metrics[this.metrics.length - 1];
    
    if (!latest) return suggestions;

    if (latest.lcp && latest.lcp > this.thresholds.lcp) {
      suggestions.push('Consider optimizing largest contentful paint by preloading critical resources');
    }
    
    if (latest.cls && latest.cls > this.thresholds.cls) {
      suggestions.push('Reduce cumulative layout shift by specifying image dimensions');
    }
    
    if (latest.bundleSize > 500000) { // 500KB
      suggestions.push('Consider code splitting to reduce bundle size');
    }
    
    if (latest.cacheHitRate < 80) {
      suggestions.push('Improve cache hit rate by implementing better caching strategies');
    }
    
    if (latest.memoryUsage > this.thresholds.memoryUsage) {
      suggestions.push('Memory usage is high - consider optimizing component lifecycle');
    }

    return suggestions;
  }

  // Event system
  on(event: string, callback: Function) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event)!.push(callback);
  }

  off(event: string, callback: Function) {
    if (this.eventListeners.has(event)) {
      const callbacks = this.eventListeners.get(event)!;
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  private emit(event: string, data?: any) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event)!.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  // PWA-specific optimizations
  enableLazyLoading() {
    this.optimizations.lazyLoad = true;
    
    // Implement intersection observer for lazy loading
    if ('IntersectionObserver' in window) {
      const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLImageElement;
            if (img.dataset.src) {
              img.src = img.dataset.src;
              img.removeAttribute('data-src');
              imageObserver.unobserve(img);
            }
          }
        });
      });

      // Observe all images with data-src
      document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
      });
    }
  }

  optimizeImages() {
    // Convert images to WebP if supported
    if (this.supportsWebP()) {
      document.querySelectorAll('img').forEach(img => {
        if (img.src && !img.src.includes('.webp')) {
          const webpSrc = img.src.replace(/\.(jpg|jpeg|png)$/, '.webp');
          // Check if WebP version exists
          this.checkImageExists(webpSrc).then(exists => {
            if (exists) {
              img.src = webpSrc;
            }
          });
        }
      });
    }
  }

  private supportsWebP(): boolean {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    return canvas.toDataURL('image/webp').indexOf('webp') > -1;
  }

  private async checkImageExists(src: string): Promise<boolean> {
    return new Promise(resolve => {
      const img = new Image();
      img.onload = () => resolve(true);
      img.onerror = () => resolve(false);
      img.src = src;
    });
  }

  // Background sync optimization
  registerBackgroundSync(tag: string) {
    if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
      navigator.serviceWorker.ready.then(registration => {
        return registration.sync.register(tag);
      }).catch(error => {
        console.warn('Background sync registration failed:', error);
      });
    }
  }

  // Cleanup
  destroy() {
    this.observers.forEach(observer => {
      observer.disconnect();
    });
    this.observers.clear();
    this.eventListeners.clear();
  }
}

// Singleton instance
const pwaPerformanceService = new PWAPerformanceService();

export default pwaPerformanceService;
export type { PerformanceMetrics, PerformanceAlert, PWAOptimization };