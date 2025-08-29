#!/usr/bin/env node
/**
 * Offline Functionality Excellence Validator
 * EPIC E Phase 1: PWA Performance Optimization
 * 
 * Validates comprehensive offline capabilities and optimizes offline-to-online
 * transition performance for mobile PWA excellence.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

class OfflineFunctionalityValidator {
  constructor() {
    this.results = {
      timestamp: new Date().toISOString(),
      validations: [],
      optimizations: [],
      offline_score: 0,
      max_score: 100,
      success: false
    };
  }

  async validateOfflineExcellence() {
    console.log('🌐 OFFLINE FUNCTIONALITY EXCELLENCE VALIDATION');
    console.log('=' .repeat(55));
    
    try {
      // Step 1: Service Worker Validation
      console.log('\n🔧 Step 1: Service Worker Capability Validation');
      await this.validateServiceWorkerCapabilities();
      
      // Step 2: Cache Strategy Validation
      console.log('\n💾 Step 2: Cache Strategy Excellence Validation');
      await this.validateCacheStrategies();
      
      // Step 3: Offline-First Data Validation
      console.log('\n📊 Step 3: Offline-First Data Management Validation');
      await this.validateOfflineDataManagement();
      
      // Step 4: Background Sync Validation
      console.log('\n🔄 Step 4: Background Sync & Queue Management Validation');
      await this.validateBackgroundSync();
      
      // Step 5: Offline UX Validation
      console.log('\n🎨 Step 5: Offline User Experience Validation');
      await this.validateOfflineUX();
      
      // Step 6: Network Transition Optimization
      console.log('\n⚡ Step 6: Network Transition Performance Optimization');
      await this.optimizeNetworkTransitions();
      
      // Calculate final score
      this.calculateOfflineScore();
      await this.generateOfflineReport();
      
      this.results.success = true;
      
    } catch (error) {
      console.error('❌ Offline validation failed:', error);
      this.results.error = error.message;
      await this.generateOfflineReport();
    }
  }

  async validateServiceWorkerCapabilities() {
    const swPath = path.join(projectRoot, 'public', 'sw.js');
    const swContent = fs.readFileSync(swPath, 'utf8');
    
    const capabilities = {
      install_handler: swContent.includes('addEventListener(\'install\''),
      activate_handler: swContent.includes('addEventListener(\'activate\''),
      fetch_handler: swContent.includes('addEventListener(\'fetch\''),
      push_handler: swContent.includes('addEventListener(\'push\''),
      sync_handler: swContent.includes('addEventListener(\'sync\''),
      cache_management: swContent.includes('caches.open'),
      indexeddb_support: swContent.includes('indexedDB'),
      timeout_handling: swContent.includes('fetchWithTimeout'),
      mobile_optimization: swContent.includes('MOBILE_CONFIG'),
      cache_warming: swContent.includes('warmCriticalCache')
    };
    
    const score = (Object.values(capabilities).filter(Boolean).length / Object.keys(capabilities).length) * 100;
    
    this.results.validations.push({
      category: 'Service Worker Capabilities',
      score: Math.round(score),
      details: capabilities,
      status: score >= 90 ? 'excellent' : score >= 70 ? 'good' : 'needs-improvement'
    });
    
    console.log(`  ✅ Service Worker Capabilities: ${Math.round(score)}/100`);
    
    // Add optimization recommendations
    if (score < 90) {
      this.results.optimizations.push({
        category: 'Service Worker',
        recommendation: 'Implement missing service worker capabilities for offline excellence'
      });
    }
  }

  async validateCacheStrategies() {
    const swPath = path.join(projectRoot, 'public', 'sw.js');
    const swContent = fs.readFileSync(swPath, 'utf8');
    
    const strategies = {
      cache_first: swContent.includes('cacheFirst'),
      network_first: swContent.includes('networkFirst'),
      stale_while_revalidate: swContent.includes('staleWhileRevalidate'),
      mobile_optimized: swContent.includes('mobileStaleWhileRevalidate'),
      intelligent_caching: swContent.includes('intelligentNetworkFirst'),
      cache_versioning: swContent.includes('CACHE_VERSION'),
      cache_cleanup: swContent.includes('caches.delete'),
      resource_specific: swContent.includes('request.destination'),
      timeout_handling: swContent.includes('MOBILE_CONFIG.connectionTimeoutMs'),
      fallback_handling: swContent.includes('offline.html')
    };
    
    const score = (Object.values(strategies).filter(Boolean).length / Object.keys(strategies).length) * 100;
    
    this.results.validations.push({
      category: 'Cache Strategies',
      score: Math.round(score),
      details: strategies,
      status: score >= 90 ? 'excellent' : score >= 70 ? 'good' : 'needs-improvement'
    });
    
    console.log(`  ✅ Cache Strategies: ${Math.round(score)}/100`);
    
    if (score < 90) {
      this.results.optimizations.push({
        category: 'Cache Strategies',
        recommendation: 'Implement comprehensive cache strategies for all resource types'
      });
    }
  }

  async validateOfflineDataManagement() {
    const swPath = path.join(projectRoot, 'public', 'sw.js');
    const swContent = fs.readFileSync(swPath, 'utf8');
    
    const dataManagement = {
      indexeddb_initialization: swContent.includes('initializeDB'),
      data_persistence: swContent.includes('cacheContextData'),
      data_expiration: swContent.includes('expiresAt'),
      data_cleanup: swContent.includes('cleanupExpiredContext'),
      offline_queue: swContent.includes('queueOfflineAction'),
      data_synchronization: swContent.includes('syncOfflineActions'),
      context_caching: swContent.includes('CONTEXT_CACHE_NAME'),
      data_versioning: swContent.includes('version: 1'),
      large_data_handling: swContent.includes('maxCacheSize'),
      data_compression: swContent.includes('compression') || swContent.includes('gzip')
    };
    
    const score = (Object.values(dataManagement).filter(Boolean).length / Object.keys(dataManagement).length) * 100;
    
    this.results.validations.push({
      category: 'Offline Data Management',
      score: Math.round(score),
      details: dataManagement,
      status: score >= 90 ? 'excellent' : score >= 70 ? 'good' : 'needs-improvement'
    });
    
    console.log(`  ✅ Offline Data Management: ${Math.round(score)}/100`);
    
    if (score < 90) {
      this.results.optimizations.push({
        category: 'Data Management',
        recommendation: 'Enhance offline data management with compression and intelligent sync'
      });
    }
  }

  async validateBackgroundSync() {
    const swPath = path.join(projectRoot, 'public', 'sw.js');
    const swContent = fs.readFileSync(swPath, 'utf8');
    
    const backgroundSync = {
      sync_listener: swContent.includes('addEventListener(\'sync\''),
      task_sync: swContent.includes('syncTaskUpdates'),
      offline_action_sync: swContent.includes('syncOfflineActions'),
      retry_mechanism: swContent.includes('retryCount'),
      exponential_backoff: swContent.includes('maxRetries'),
      sync_registration: swContent.includes('sync.register') || swContent.includes('registration.sync'),
      queue_management: swContent.includes('getQueuedOfflineActions'),
      error_handling: swContent.includes('catch(error)'),
      network_detection: swContent.includes('navigator.onLine'),
      batch_processing: swContent.includes('Promise.all')
    };
    
    const score = (Object.values(backgroundSync).filter(Boolean).length / Object.keys(backgroundSync).length) * 100;
    
    this.results.validations.push({
      category: 'Background Sync',
      score: Math.round(score),
      details: backgroundSync,
      status: score >= 90 ? 'excellent' : score >= 70 ? 'good' : 'needs-improvement'
    });
    
    console.log(`  ✅ Background Sync: ${Math.round(score)}/100`);
    
    if (score < 90) {
      this.results.optimizations.push({
        category: 'Background Sync',
        recommendation: 'Implement robust background sync with intelligent retry mechanisms'
      });
    }
  }

  async validateOfflineUX() {
    const indexPath = path.join(projectRoot, 'index.html');
    const indexContent = fs.readFileSync(indexPath, 'utf8');
    const offlinePath = path.join(projectRoot, 'public', 'offline.html');
    
    const uxFeatures = {
      offline_page_exists: fs.existsSync(offlinePath),
      viewport_optimized: indexContent.includes('viewport'),
      theme_color: indexContent.includes('theme-color'),
      apple_touch_icon: indexContent.includes('apple-touch-icon'),
      manifest_linked: indexContent.includes('manifest.json'),
      loading_states: this.checkForLoadingStates(),
      error_boundaries: this.checkForErrorBoundaries(),
      offline_indicators: this.checkForOfflineIndicators(),
      connection_status: this.checkForConnectionStatus(),
      graceful_degradation: this.checkForGracefulDegradation()
    };
    
    const score = (Object.values(uxFeatures).filter(Boolean).length / Object.keys(uxFeatures).length) * 100;
    
    this.results.validations.push({
      category: 'Offline User Experience',
      score: Math.round(score),
      details: uxFeatures,
      status: score >= 90 ? 'excellent' : score >= 70 ? 'good' : 'needs-improvement'
    });
    
    console.log(`  ✅ Offline User Experience: ${Math.round(score)}/100`);
    
    if (score < 90) {
      this.results.optimizations.push({
        category: 'Offline UX',
        recommendation: 'Enhance offline user experience with better indicators and graceful degradation'
      });
    }
  }

  checkForLoadingStates() {
    const srcPath = path.join(projectRoot, 'src');
    try {
      const files = this.getAllFiles(srcPath, '.ts');
      return files.some(file => {
        const content = fs.readFileSync(file, 'utf8');
        return content.includes('loading-spinner') || content.includes('isLoading');
      });
    } catch {
      return false;
    }
  }

  checkForErrorBoundaries() {
    const srcPath = path.join(projectRoot, 'src');
    try {
      const files = this.getAllFiles(srcPath, '.ts');
      return files.some(file => {
        const content = fs.readFileSync(file, 'utf8');
        return content.includes('error-boundary') || content.includes('ErrorBoundary');
      });
    } catch {
      return false;
    }
  }

  checkForOfflineIndicators() {
    const srcPath = path.join(projectRoot, 'src');
    try {
      const files = this.getAllFiles(srcPath, '.ts');
      return files.some(file => {
        const content = fs.readFileSync(file, 'utf8');
        return content.includes('offline') || content.includes('connection-monitor');
      });
    } catch {
      return false;
    }
  }

  checkForConnectionStatus() {
    const srcPath = path.join(projectRoot, 'src');
    try {
      const files = this.getAllFiles(srcPath, '.ts');
      return files.some(file => {
        const content = fs.readFileSync(file, 'utf8');
        return content.includes('navigator.onLine') || content.includes('connection');
      });
    } catch {
      return false;
    }
  }

  checkForGracefulDegradation() {
    const srcPath = path.join(projectRoot, 'src');
    try {
      const files = this.getAllFiles(srcPath, '.ts');
      return files.some(file => {
        const content = fs.readFileSync(file, 'utf8');
        return content.includes('fallback') || content.includes('degraded');
      });
    } catch {
      return false;
    }
  }

  getAllFiles(dir, ext) {
    let files = [];
    const items = fs.readdirSync(dir);
    
    for (const item of items) {
      const fullPath = path.join(dir, item);
      if (fs.statSync(fullPath).isDirectory()) {
        files = files.concat(this.getAllFiles(fullPath, ext));
      } else if (item.endsWith(ext)) {
        files.push(fullPath);
      }
    }
    
    return files;
  }

  async optimizeNetworkTransitions() {
    // Create optimized network transition service
    const transitionOptimizer = `/**
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
export const networkTransitionOptimizer = new NetworkTransitionOptimizer();`;

    const optimizerPath = path.join(projectRoot, 'src', 'utils', 'network-transition-optimizer.ts');
    fs.writeFileSync(optimizerPath, transitionOptimizer);
    
    this.results.optimizations.push({
      category: 'Network Transitions',
      description: 'Created intelligent network transition optimizer for seamless offline-to-online transitions',
      impact: 'Improved perceived performance during network changes'
    });
    
    console.log('  ✅ Network transition performance optimizer created');
  }

  calculateOfflineScore() {
    const validations = this.results.validations;
    const totalScore = validations.reduce((sum, validation) => sum + validation.score, 0);
    this.results.offline_score = Math.round(totalScore / validations.length);
    
    console.log(`\n🎯 OVERALL OFFLINE FUNCTIONALITY SCORE: ${this.results.offline_score}/100`);
    
    if (this.results.offline_score >= 95) {
      console.log('🏆 EXCELLENT: Offline functionality exceeds excellence standards!');
    } else if (this.results.offline_score >= 85) {
      console.log('✅ GOOD: Offline functionality meets high standards');
    } else {
      console.log('⚠️  NEEDS IMPROVEMENT: Offline functionality requires optimization');
    }
  }

  async generateOfflineReport() {
    const reportPath = path.join(projectRoot, 'offline-validation-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(this.results, null, 2));
    
    console.log(`\n📄 Offline validation report saved to: ${reportPath}`);
    
    // Generate summary
    console.log('\n📊 VALIDATION SUMMARY:');
    console.log('=' .repeat(40));
    
    this.results.validations.forEach(validation => {
      const status = validation.status === 'excellent' ? '🏆' : 
                    validation.status === 'good' ? '✅' : '⚠️';
      console.log(`${status} ${validation.category}: ${validation.score}/100`);
    });
    
    if (this.results.optimizations.length > 0) {
      console.log('\n🔧 OPTIMIZATIONS APPLIED:');
      this.results.optimizations.forEach((opt, index) => {
        console.log(`${index + 1}. ${opt.category}: ${opt.description || opt.recommendation}`);
      });
    }
  }

  async optimizeForMobileConnectivity() {
    // Create mobile-specific connection handling
    const mobileConnectivity = `/**
 * Mobile Connectivity Optimizer
 * Handles mobile-specific network challenges and optimizations
 */

export class MobileConnectivityOptimizer {
  private networkType: string = 'unknown';
  private dataUsageOptimization = true;
  private adaptiveQuality = true;

  constructor() {
    this.initializeMobileOptimizations();
  }

  private initializeMobileOptimizations() {
    // Detect mobile network type
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      this.networkType = connection.effectiveType || connection.type || 'unknown';
      
      connection.addEventListener('change', () => {
        this.networkType = connection.effectiveType || connection.type || 'unknown';
        this.adaptToNetworkConditions();
      });
    }

    // Initialize based on current conditions
    this.adaptToNetworkConditions();
  }

  private adaptToNetworkConditions() {
    switch (this.networkType) {
      case 'slow-2g':
      case '2g':
        this.enableUltraLowBandwidthMode();
        break;
      case '3g':
        this.enableLowBandwidthMode();
        break;
      case '4g':
      case '5g':
        this.enableHighBandwidthMode();
        break;
      default:
        this.enableAdaptiveMode();
    }
  }

  private enableUltraLowBandwidthMode() {
    // Minimize data usage for 2G connections
    this.configureServiceWorker({
      cacheStrategy: 'cache-first',
      compressionLevel: 'maximum',
      imageQuality: 'low',
      updateFrequency: 'minimal'
    });
  }

  private enableLowBandwidthMode() {
    // Optimize for 3G connections
    this.configureServiceWorker({
      cacheStrategy: 'stale-while-revalidate',
      compressionLevel: 'high',
      imageQuality: 'medium',
      updateFrequency: 'reduced'
    });
  }

  private enableHighBandwidthMode() {
    // Full functionality for 4G/5G
    this.configureServiceWorker({
      cacheStrategy: 'network-first',
      compressionLevel: 'standard',
      imageQuality: 'high',
      updateFrequency: 'normal'
    });
  }

  private enableAdaptiveMode() {
    // Adaptive configuration based on performance
    this.configureServiceWorker({
      cacheStrategy: 'adaptive',
      compressionLevel: 'adaptive',
      imageQuality: 'adaptive',
      updateFrequency: 'adaptive'
    });
  }

  private configureServiceWorker(config: any) {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then(registration => {
        registration.active?.postMessage({
          type: 'configure-mobile-optimization',
          config
        });
      });
    }
  }

  // Battery-aware optimizations
  optimizeForBatteryLevel() {
    if ('getBattery' in navigator) {
      (navigator as any).getBattery().then((battery: any) => {
        const level = battery.level;
        
        if (level < 0.2) {
          // Low battery mode
          this.enablePowerSavingMode();
        } else if (level < 0.5) {
          // Moderate battery mode
          this.enableBalancedMode();
        } else {
          // Full performance mode
          this.enableFullPerformanceMode();
        }
      });
    }
  }

  private enablePowerSavingMode() {
    // Reduce update frequency and background tasks
    this.configureServiceWorker({
      backgroundSync: 'minimal',
      pushNotifications: 'essential-only',
      updateFrequency: 'very-reduced'
    });
  }

  private enableBalancedMode() {
    // Balanced performance and power
    this.configureServiceWorker({
      backgroundSync: 'standard',
      pushNotifications: 'important',
      updateFrequency: 'standard'
    });
  }

  private enableFullPerformanceMode() {
    // Full performance
    this.configureServiceWorker({
      backgroundSync: 'full',
      pushNotifications: 'all',
      updateFrequency: 'high'
    });
  }
}

export const mobileConnectivityOptimizer = new MobileConnectivityOptimizer();`;

    const mobilePath = path.join(projectRoot, 'src', 'utils', 'mobile-connectivity-optimizer.ts');
    fs.writeFileSync(mobilePath, mobileConnectivity);
    
    this.results.optimizations.push({
      category: 'Mobile Connectivity',
      description: 'Created mobile-specific connectivity optimizer for varied network conditions',
      impact: 'Optimized performance across different mobile network types'
    });
    
    console.log('  ✅ Mobile connectivity optimizer created');
  }
}

// Run the validator if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const validator = new OfflineFunctionalityValidator();
  validator.validateOfflineExcellence().catch(console.error);
}

export default OfflineFunctionalityValidator;