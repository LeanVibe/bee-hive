/**
 * Network Transition Performance Optimizer
 * Optimizes offline-to-online transitions for mobile PWA excellence
 */

export class NetworkTransitionOptimizer {
  private connectionQuality: 'fast' | 'slow' | 'offline' = 'fast';
  private transitionQueue: Array<() => Promise<void>> = [];
  private isProcessingTransitions = false;

  constructor() {
    this.initializeNetworkMonitoring();
    this.optimizeTransitionHandling();
  }

  private initializeNetworkMonitoring() {
    // Monitor connection changes
    window.addEventListener('online', () => this.handleOnlineTransition());
    window.addEventListener('offline', () => this.handleOfflineTransition());
    
    // Monitor connection quality
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      connection.addEventListener('change', () => this.assessConnectionQuality());
      this.assessConnectionQuality();
    }
  }

  private assessConnectionQuality() {
    if (!navigator.onLine) {
      this.connectionQuality = 'offline';
      return;
    }

    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      const effectiveType = connection.effectiveType;
      
      if (effectiveType === '4g' || connection.downlink > 10) {
        this.connectionQuality = 'fast';
      } else if (effectiveType === '3g' || connection.downlink > 1.5) {
        this.connectionQuality = 'slow';
      } else {
        this.connectionQuality = 'slow';
      }
    } else {
      // Fallback: measure connection speed
      this.measureConnectionSpeed().then(speed => {
        this.connectionQuality = speed > 5 ? 'fast' : 'slow';
      });
    }
  }

  private async measureConnectionSpeed(): Promise<number> {
    try {
      const startTime = performance.now();
      await fetch('/manifest.json', { cache: 'no-cache' });
      const endTime = performance.now();
      
      // Rough speed estimation in Mbps
      const duration = endTime - startTime;
      return duration < 100 ? 10 : duration < 500 ? 5 : 1;
    } catch {
      return 0;
    }
  }

  private async handleOnlineTransition() {
    console.log('Network transition: Online detected');
    
    // Prioritize critical data sync
    await this.syncCriticalData();
    
    // Batch process queued transitions
    await this.processTransitionQueue();
    
    // Warm important caches
    this.warmCaches();
    
    // Notify components of online status
    this.broadcastNetworkStatus('online');
  }

  private handleOfflineTransition() {
    console.log('Network transition: Offline detected');
    
    // Enable aggressive caching
    this.enableAggressiveCaching();
    
    // Queue outgoing requests
    this.enableRequestQueueing();
    
    // Notify components of offline status
    this.broadcastNetworkStatus('offline');
  }

  private async syncCriticalData() {
    // Sync with service worker
    if ('serviceWorker' in navigator) {
      const registration = await navigator.serviceWorker.ready;
      registration.active?.postMessage({
        type: 'sync-critical-data',
        priority: 'high'
      });
    }
  }

  private async processTransitionQueue() {
    if (this.isProcessingTransitions || this.transitionQueue.length === 0) {
      return;
    }

    this.isProcessingTransitions = true;
    
    try {
      // Process based on connection quality
      const batchSize = this.connectionQuality === 'fast' ? 10 : 3;
      
      while (this.transitionQueue.length > 0) {
        const batch = this.transitionQueue.splice(0, batchSize);
        
        if (this.connectionQuality === 'fast') {
          // Process in parallel for fast connections
          await Promise.all(batch.map(fn => fn()));
        } else {
          // Process sequentially for slow connections
          for (const fn of batch) {
            await fn();
          }
        }
        
        // Add delay for slow connections to prevent overwhelming
        if (this.connectionQuality === 'slow') {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
    } finally {
      this.isProcessingTransitions = false;
    }
  }

  private warmCaches() {
    // Send cache warming request to service worker
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then(registration => {
        registration.active?.postMessage({
          type: 'warm-caches',
          resources: ['/', '/dashboard', '/tasks', '/agents']
        });
      });
    }
  }

  private enableAggressiveCaching() {
    // Configure for offline mode
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then(registration => {
        registration.active?.postMessage({
          type: 'enable-aggressive-caching',
          mode: 'offline'
        });
      });
    }
  }

  private enableRequestQueueing() {
    // Enable request queueing in service worker
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then(registration => {
        registration.active?.postMessage({
          type: 'enable-request-queueing',
          enabled: true
        });
      });
    }
  }

  private broadcastNetworkStatus(status: 'online' | 'offline') {
    // Broadcast to all components
    window.dispatchEvent(new CustomEvent('network-status-change', {
      detail: {
        status,
        quality: this.connectionQuality,
        timestamp: Date.now()
      }
    }));
  }

  // Public API
  queueTransition(fn: () => Promise<void>) {
    this.transitionQueue.push(fn);
    
    if (navigator.onLine && !this.isProcessingTransitions) {
      this.processTransitionQueue();
    }
  }

  getConnectionQuality() {
    return this.connectionQuality;
  }

  isOnline() {
    return navigator.onLine;
  }
}

// Initialize the optimizer
export const networkTransitionOptimizer = new NetworkTransitionOptimizer();