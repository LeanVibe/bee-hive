import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../services/websocket'
import '../components/mobile-gesture-interface'
import '../components/mobile-notification-system'
import '../components/mobile-context-explorer'
import './mobile-enhanced-dashboard-view'

interface DashboardConfig {
  enableGestures: boolean
  enableNotifications: boolean
  enableContextExploration: boolean
  theme: 'light' | 'dark' | 'auto'
  layout: 'mobile' | 'tablet' | 'desktop'
}

interface PerformanceMetrics {
  gestureResponseTime: number
  notificationDeliveryTime: number
  contextSwitchTime: number
  cacheHitRate: number
}

@customElement('mobile-decision-dashboard')
export class MobileDecisionDashboard extends LitElement {
  @property({ type: String }) declare role: 'developer' | 'manager' | 'architect'
  @property({ type: Boolean }) declare mobile: boolean
  @property({ type: String }) declare initialContext: string

  @state() private declare config: DashboardConfig
  @state() private declare currentView: 'dashboard' | 'context' | 'settings'
  @state() private declare isOnline: boolean
  @state() private declare performanceMetrics: PerformanceMetrics
  @state() private declare showPerformancePanel: boolean
  @state() private declare gestureTrainingActive: boolean
  @state() private declare contextHistory: string[]
  @state() private declare lastActivity: Date | null

  private websocketService: WebSocketService
  private performanceMonitor: PerformanceObserver | null = null
  private activityTimer: number | null = null
  private swipeStartX = 0
  private swipeStartY = 0
  private swipeThreshold = 100

  static styles = css`
    :host {
      display: block;
      height: 100vh;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      overflow: hidden;
      position: relative;
    }

    .dashboard-container {
      height: 100%;
      display: flex;
      flex-direction: column;
      position: relative;
    }

    .dashboard-header {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(0, 0, 0, 0.1);
      padding: 0.75rem 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 100;
      flex-shrink: 0;
    }

    .header-title {
      font-size: 1.125rem;
      font-weight: 700;
      color: #111827;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .online-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 2s infinite;
    }

    .online-indicator.offline {
      background: #ef4444;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .header-button {
      background: none;
      border: none;
      padding: 0.5rem;
      border-radius: 8px;
      cursor: pointer;
      color: #6b7280;
      transition: all 0.2s;
      font-size: 1.25rem;
    }

    .header-button:hover {
      background: rgba(0, 0, 0, 0.05);
      color: #374151;
    }

    .header-button.active {
      background: #eff6ff;
      color: #3b82f6;
    }

    .main-content {
      flex: 1;
      overflow: hidden;
      position: relative;
    }

    .view-container {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      transition: transform 0.3s ease-out;
    }

    .view-container.slide-left {
      transform: translateX(-100%);
    }

    .view-container.slide-right {
      transform: translateX(100%);
    }

    .bottom-navigation {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      border-top: 1px solid rgba(0, 0, 0, 0.1);
      padding: 0.75rem;
      display: flex;
      justify-content: space-around;
      z-index: 100;
      flex-shrink: 0;
    }

    .nav-button {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.25rem;
      background: none;
      border: none;
      padding: 0.5rem;
      cursor: pointer;
      color: #6b7280;
      transition: all 0.2s;
      border-radius: 8px;
      min-width: 60px;
    }

    .nav-button:hover {
      background: rgba(0, 0, 0, 0.05);
    }

    .nav-button.active {
      color: #3b82f6;
      background: #eff6ff;
    }

    .nav-icon {
      font-size: 1.25rem;
    }

    .nav-label {
      font-size: 0.75rem;
      font-weight: 500;
    }

    .performance-panel {
      position: fixed;
      bottom: 80px;
      right: 1rem;
      background: rgba(17, 24, 39, 0.95);
      color: white;
      border-radius: 12px;
      padding: 1rem;
      min-width: 280px;
      backdrop-filter: blur(10px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
      z-index: 200;
      transform: translateY(100%);
      transition: transform 0.3s ease-out;
    }

    .performance-panel.visible {
      transform: translateY(0);
    }

    .performance-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .performance-title {
      font-weight: 600;
      font-size: 0.9rem;
    }

    .close-button {
      background: none;
      border: none;
      color: #9ca3af;
      cursor: pointer;
      padding: 0.25rem;
      border-radius: 4px;
      transition: all 0.2s;
    }

    .close-button:hover {
      background: rgba(255, 255, 255, 0.1);
      color: white;
    }

    .performance-metrics {
      display: grid;
      gap: 0.75rem;
    }

    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .metric-label {
      font-size: 0.8rem;
      color: #d1d5db;
    }

    .metric-value {
      font-weight: 700;
      font-size: 0.9rem;
    }

    .metric-value.good {
      color: #10b981;
    }

    .metric-value.warning {
      color: #f59e0b;
    }

    .metric-value.error {
      color: #ef4444;
    }

    .gesture-training-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      z-index: 300;
      display: flex;
      align-items: center;
      justify-content: center;
      backdrop-filter: blur(5px);
    }

    .training-panel {
      background: white;
      border-radius: 20px;
      padding: 2rem;
      max-width: 90%;
      width: 400px;
      text-align: center;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }

    .training-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin-bottom: 1rem;
    }

    .training-message {
      color: #6b7280;
      margin-bottom: 2rem;
      line-height: 1.5;
    }

    .training-actions {
      display: flex;
      gap: 1rem;
      justify-content: center;
    }

    .training-button {
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      border: none;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }

    .training-button.primary {
      background: #3b82f6;
      color: white;
    }

    .training-button.primary:hover {
      background: #2563eb;
    }

    .training-button.secondary {
      background: #f3f4f6;
      color: #374151;
    }

    .training-button.secondary:hover {
      background: #e5e7eb;
    }

    .swipe-hint {
      position: fixed;
      bottom: 50%;
      left: 50%;
      transform: translate(-50%, 50%);
      background: rgba(59, 130, 246, 0.9);
      color: white;
      padding: 0.75rem 1.5rem;
      border-radius: 12px;
      font-size: 0.9rem;
      font-weight: 500;
      z-index: 150;
      animation: fadeInOut 3s ease-in-out;
      pointer-events: none;
    }

    /* Animations */
    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.5;
      }
    }

    @keyframes fadeInOut {
      0%, 100% {
        opacity: 0;
        transform: translate(-50%, 50%) scale(0.8);
      }
      20%, 80% {
        opacity: 1;
        transform: translate(-50%, 50%) scale(1);
      }
    }

    /* Touch optimizations */
    @media (hover: none) and (pointer: coarse) {
      .header-button,
      .nav-button {
        min-height: 44px;
        min-width: 44px;
      }
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      :host {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
      }

      .dashboard-header,
      .bottom-navigation {
        background: rgba(31, 41, 55, 0.95);
        border-color: rgba(255, 255, 255, 0.1);
      }

      .header-title {
        color: #f9fafb;
      }

      .header-button,
      .nav-button {
        color: #d1d5db;
      }

      .header-button:hover,
      .nav-button:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #f9fafb;
      }

      .training-panel {
        background: #1f2937;
        color: #f9fafb;
      }

      .training-title {
        color: #f9fafb;
      }
    }

    /* Accessibility */
    @media (prefers-reduced-motion: reduce) {
      * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }
  `

  constructor() {
    super()
    this.role = 'developer'
    this.mobile = true
    this.initialContext = ''
    this.config = {
      enableGestures: true,
      enableNotifications: true,
      enableContextExploration: true,
      theme: 'auto',
      layout: 'mobile'
    }
    this.currentView = 'dashboard'
    this.isOnline = navigator.onLine
    this.performanceMetrics = {
      gestureResponseTime: 0,
      notificationDeliveryTime: 0,
      contextSwitchTime: 0,
      cacheHitRate: 0
    }
    this.showPerformancePanel = false
    this.gestureTrainingActive = false
    this.contextHistory = []
    this.lastActivity = null
    this.websocketService = WebSocketService.getInstance()
  }

  connectedCallback() {
    super.connectedCallback()
    this.setupEventListeners()
    this.initializePerformanceMonitoring()
    this.startActivityTracking()
    this.loadConfiguration()
    this.checkFirstTimeUser()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.cleanupEventListeners()
    if (this.performanceMonitor) {
      this.performanceMonitor.disconnect()
    }
    if (this.activityTimer) {
      clearInterval(this.activityTimer)
    }
  }

  private setupEventListeners() {
    // Online/offline status
    window.addEventListener('online', () => {
      this.isOnline = true
      this.syncOfflineData()
    })
    
    window.addEventListener('offline', () => {
      this.isOnline = false
    })

    // Gesture events from mobile-gesture-interface
    this.addEventListener('gesture-executed', (event: CustomEvent) => {
      const { action, gestureType } = event.detail
      this.handleGestureAction(action, gestureType)
    })

    // Notification events
    this.addEventListener('show-feedback', (event: CustomEvent) => {
      this.showFeedback(event.detail.message, event.detail.type)
    })

    // Context explorer events
    this.addEventListener('action-executed', (event: CustomEvent) => {
      this.handleContextAction(event.detail.action, event.detail.result)
    })

    // Touch events for swipe navigation
    this.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: true })
    this.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: true })

    // Command execution events
    this.addEventListener('execute-command', (event: CustomEvent) => {
      this.executeCommand(event.detail.command, event.detail.title)
    })
  }

  private cleanupEventListeners() {
    window.removeEventListener('online', () => {})
    window.removeEventListener('offline', () => {})
  }

  private initializePerformanceMonitoring() {
    if ('PerformanceObserver' in window) {
      this.performanceMonitor = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          this.updatePerformanceMetrics(entry)
        }
      })
      
      this.performanceMonitor.observe({ entryTypes: ['measure', 'navigation'] })
    }

    // Start measuring initial metrics
    this.measureGesturePerformance()
    this.measureNotificationPerformance()
    this.measureContextPerformance()
  }

  private startActivityTracking() {
    const updateActivity = () => {
      this.lastActivity = new Date()
    }

    ['touchstart', 'touchmove', 'touchend', 'click', 'scroll'].forEach(event => {
      document.addEventListener(event, updateActivity, { passive: true })
    })

    // Check for inactivity every 5 minutes
    this.activityTimer = window.setInterval(() => {
      if (this.lastActivity) {
        const inactive = Date.now() - this.lastActivity.getTime()
        if (inactive > 10 * 60 * 1000) { // 10 minutes
          this.handleInactivity()
        }
      }
    }, 5 * 60 * 1000)
  }

  private loadConfiguration() {
    const saved = localStorage.getItem('mobile-dashboard-config')
    if (saved) {
      try {
        this.config = { ...this.config, ...JSON.parse(saved) }
      } catch (error) {
        console.warn('Failed to load dashboard configuration:', error)
      }
    }
  }

  private saveConfiguration() {
    localStorage.setItem('mobile-dashboard-config', JSON.stringify(this.config))
  }

  private checkFirstTimeUser() {
    const hasUsedGestures = localStorage.getItem('mobile-gestures-used')
    if (!hasUsedGestures && this.config.enableGestures) {
      setTimeout(() => {
        this.gestureTrainingActive = true
      }, 2000) // Show training after 2 seconds
    }
  }

  private handleTouchStart(event: TouchEvent) {
    if (event.touches.length === 1) {
      this.swipeStartX = event.touches[0].clientX
      this.swipeStartY = event.touches[0].clientY
    }
  }

  private handleTouchEnd(event: TouchEvent) {
    if (event.changedTouches.length === 1) {
      const deltaX = event.changedTouches[0].clientX - this.swipeStartX
      const deltaY = event.changedTouches[0].clientY - this.swipeStartY
      
      if (Math.abs(deltaX) > this.swipeThreshold && Math.abs(deltaX) > Math.abs(deltaY)) {
        if (deltaX > 0) {
          this.handleSwipeRight()
        } else {
          this.handleSwipeLeft()
        }
      }
    }
  }

  private handleSwipeLeft() {
    if (this.currentView === 'dashboard') {
      this.switchView('context')
      this.showSwipeHint('Swipe right to go back')
    }
  }

  private handleSwipeRight() {
    if (this.currentView === 'context') {
      this.switchView('dashboard')
      this.showSwipeHint('Swipe left for context explorer')
    }
  }

  private showSwipeHint(message: string) {
    const hint = document.createElement('div')
    hint.className = 'swipe-hint'
    hint.textContent = message
    this.shadowRoot?.appendChild(hint)
    
    setTimeout(() => {
      hint.remove()
    }, 3000)
  }

  private handleGestureAction(action: any, gestureType: string) {
    const startTime = performance.now()
    
    // Record gesture usage
    localStorage.setItem('mobile-gestures-used', 'true')
    
    // Execute the action
    console.log('Gesture executed:', gestureType, action)
    
    // Measure response time
    const responseTime = performance.now() - startTime
    this.performanceMetrics.gestureResponseTime = Math.round(responseTime)
    
    this.requestUpdate()
  }

  private handleContextAction(action: any, result: any) {
    console.log('Context action executed:', action, result)
    this.contextHistory.push(action.id)
    
    // Keep only last 10 context actions
    if (this.contextHistory.length > 10) {
      this.contextHistory = this.contextHistory.slice(-10)
    }
  }

  private async executeCommand(command: string, title?: string) {
    const startTime = performance.now()
    
    try {
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command })
      })
      
      const result = await response.json()
      const executionTime = performance.now() - startTime
      
      if (result.success) {
        this.showFeedback(`${title || 'Command'} executed successfully`, 'success')
      } else {
        this.showFeedback(`${title || 'Command'} failed: ${result.error}`, 'error')
      }
      
      console.log(`Command executed in ${Math.round(executionTime)}ms:`, command)
    } catch (error) {
      this.showFeedback('Command execution failed - queued for offline sync', 'warning')
      
      // Queue for offline execution
      if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
        navigator.serviceWorker.controller.postMessage({
          type: 'store-offline-command',
          data: { command, timestamp: Date.now() }
        })
      }
    }
  }

  private showFeedback(message: string, type: 'success' | 'error' | 'warning' = 'success') {
    // Create and show a toast notification
    const toast = document.createElement('div')
    toast.style.cssText = `
      position: fixed;
      bottom: 100px;
      left: 50%;
      transform: translateX(-50%);
      background: ${
        type === 'success' ? '#10b981' : 
        type === 'error' ? '#ef4444' : '#f59e0b'
      };
      color: white;
      padding: 1rem 1.5rem;
      border-radius: 12px;
      font-weight: 500;
      z-index: 250;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
      animation: slideUp 0.3s ease-out;
    `
    toast.textContent = message
    
    document.body.appendChild(toast)
    
    setTimeout(() => {
      toast.style.transform = 'translateX(-50%) translateY(20px)'
      toast.style.opacity = '0'
      setTimeout(() => toast.remove(), 300)
    }, 3000)
  }

  private switchView(view: typeof this.currentView) {
    const startTime = performance.now()
    this.currentView = view
    
    // Measure context switch time
    setTimeout(() => {
      const switchTime = performance.now() - startTime
      this.performanceMetrics.contextSwitchTime = Math.round(switchTime)
      this.requestUpdate()
    }, 100)
  }

  private handleInactivity() {
    // Put system in low-power mode or show activity prompt
    console.log('User inactive for 10+ minutes')
  }

  private async syncOfflineData() {
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      try {
        // Sync any queued offline commands
        await navigator.serviceWorker.ready
        if ('sync' in window.ServiceWorkerRegistration.prototype) {
          const registration = await navigator.serviceWorker.ready
          await registration.sync.register('command-sync')
        }
      } catch (error) {
        console.warn('Failed to sync offline data:', error)
      }
    }
  }

  private measureGesturePerformance() {
    // This would measure actual gesture response times
    // For now, simulate good performance
    this.performanceMetrics.gestureResponseTime = Math.random() * 50 + 30 // 30-80ms
  }

  private measureNotificationPerformance() {
    // This would measure notification delivery times
    this.performanceMetrics.notificationDeliveryTime = Math.random() * 2000 + 1000 // 1-3s
  }

  private measureContextPerformance() {
    // This would measure context switching performance
    this.performanceMetrics.contextSwitchTime = Math.random() * 100 + 50 // 50-150ms
  }

  private updatePerformanceMetrics(entry: PerformanceEntry) {
    // Update metrics based on actual performance measurements
    if (entry.name.includes('gesture')) {
      this.performanceMetrics.gestureResponseTime = Math.round(entry.duration)
    } else if (entry.name.includes('context')) {
      this.performanceMetrics.contextSwitchTime = Math.round(entry.duration)
    }
  }

  private getPerformanceStatus(metric: number, thresholds: { good: number; warning: number }): string {
    if (metric <= thresholds.good) return 'good'
    if (metric <= thresholds.warning) return 'warning'
    return 'error'
  }

  togglePerformancePanel() {
    this.showPerformancePanel = !this.showPerformancePanel
  }

  toggleGestureTraining() {
    this.gestureTrainingActive = !this.gestureTrainingActive
  }

  startGestureTraining() {
    const gestureInterface = this.shadowRoot?.querySelector('mobile-gesture-interface') as any
    if (gestureInterface) {
      gestureInterface.showTrainingModal()
    }
    this.gestureTrainingActive = false
  }

  render() {
    return html`
      <div class="dashboard-container">
        ${this.renderHeader()}
        ${this.renderMainContent()}
        ${this.renderBottomNavigation()}
        ${this.renderPerformancePanel()}
        ${this.renderGestureTrainingOverlay()}
        ${this.renderComponents()}
      </div>
    `
  }

  private renderHeader() {
    return html`
      <div class="dashboard-header">
        <div class="header-title">
          <div class="online-indicator ${this.isOnline ? '' : 'offline'}"></div>
          📱 Mobile Decision Interface
        </div>
        
        <div class="header-actions">
          <button class="header-button ${this.showPerformancePanel ? 'active' : ''}" 
                  @click=${this.togglePerformancePanel}
                  title="Performance Metrics">
            📊
          </button>
          <button class="header-button" 
                  @click=${this.toggleGestureTraining}
                  title="Gesture Training">
            ✋
          </button>
          <button class="header-button" 
                  @click=${() => this.switchView('settings')}
                  title="Settings">
            ⚙️
          </button>
        </div>
      </div>
    `
  }

  private renderMainContent() {
    return html`
      <div class="main-content">
        <div class="view-container ${this.currentView !== 'dashboard' ? 'slide-left' : ''}">
          <mobile-enhanced-dashboard-view 
            .mobile=${this.mobile}
            .decisionMode=${true}>
          </mobile-enhanced-dashboard-view>
        </div>
        
        <div class="view-container ${this.currentView !== 'context' ? 'slide-right' : ''}">
          <mobile-context-explorer 
            .mobile=${this.mobile}
            .initialContext=${this.initialContext}>
          </mobile-context-explorer>
        </div>
      </div>
    `
  }

  private renderBottomNavigation() {
    return html`
      <div class="bottom-navigation">
        <button class="nav-button ${this.currentView === 'dashboard' ? 'active' : ''}" 
                @click=${() => this.switchView('dashboard')}>
          <div class="nav-icon">🏠</div>
          <div class="nav-label">Dashboard</div>
        </button>
        
        <button class="nav-button ${this.currentView === 'context' ? 'active' : ''}" 
                @click=${() => this.switchView('context')}>
          <div class="nav-icon">🔍</div>
          <div class="nav-label">Context</div>
        </button>
        
        <button class="nav-button" 
                @click=${() => {
                  const notificationSystem = this.shadowRoot?.querySelector('mobile-notification-system') as any
                  if (notificationSystem) {
                    notificationSystem.openSettings()
                  }
                }}>
          <div class="nav-icon">🔔</div>
          <div class="nav-label">Alerts</div>
        </button>
        
        <button class="nav-button ${this.currentView === 'settings' ? 'active' : ''}" 
                @click=${() => this.switchView('settings')}>
          <div class="nav-icon">⚙️</div>
          <div class="nav-label">Settings</div>
        </button>
      </div>
    `
  }

  private renderPerformancePanel() {
    if (!this.showPerformancePanel) return html``

    return html`
      <div class="performance-panel visible">
        <div class="performance-header">
          <div class="performance-title">📊 Performance Metrics</div>
          <button class="close-button" @click=${this.togglePerformancePanel}>✕</button>
        </div>
        
        <div class="performance-metrics">
          <div class="metric-row">
            <div class="metric-label">Gesture Response</div>
            <div class="metric-value ${this.getPerformanceStatus(this.performanceMetrics.gestureResponseTime, { good: 100, warning: 200 })}">
              ${this.performanceMetrics.gestureResponseTime}ms
            </div>
          </div>
          
          <div class="metric-row">
            <div class="metric-label">Notification Delivery</div>
            <div class="metric-value ${this.getPerformanceStatus(this.performanceMetrics.notificationDeliveryTime, { good: 5000, warning: 10000 })}">
              ${Math.round(this.performanceMetrics.notificationDeliveryTime / 1000)}s
            </div>
          </div>
          
          <div class="metric-row">
            <div class="metric-label">Context Switch</div>
            <div class="metric-value ${this.getPerformanceStatus(this.performanceMetrics.contextSwitchTime, { good: 200, warning: 500 })}">
              ${this.performanceMetrics.contextSwitchTime}ms
            </div>
          </div>
          
          <div class="metric-row">
            <div class="metric-label">Cache Hit Rate</div>
            <div class="metric-value good">
              ${Math.round(this.performanceMetrics.cacheHitRate)}%
            </div>
          </div>
        </div>
      </div>
    `
  }

  private renderGestureTrainingOverlay() {
    if (!this.gestureTrainingActive) return html``

    return html`
      <div class="gesture-training-overlay">
        <div class="training-panel">
          <div class="training-title">📱 Welcome to Mobile Gestures!</div>
          <div class="training-message">
            Control your agents with intuitive swipes and taps:
            <br><br>
            • Swipe right to approve tasks<br>
            • Swipe left to pause for review<br>
            • Swipe up to escalate to human<br>
            • Long press for detailed context
          </div>
          
          <div class="training-actions">
            <button class="training-button secondary" @click=${this.toggleGestureTraining}>
              Skip Training
            </button>
            <button class="training-button primary" @click=${this.startGestureTraining}>
              Learn Gestures
            </button>
          </div>
        </div>
      </div>
    `
  }

  private renderComponents() {
    return html`
      <!-- Mobile Gesture Interface -->
      ${this.config.enableGestures ? html`
        <mobile-gesture-interface 
          .enabled=${true}
          .trainingMode=${false}
          .gestureThreshold=${50}
          .longPressTimeout=${700}>
        </mobile-gesture-interface>
      ` : ''}
      
      <!-- Mobile Notification System -->
      ${this.config.enableNotifications ? html`
        <mobile-notification-system 
          .enabled=${true}
          .role=${this.role}>
        </mobile-notification-system>
      ` : ''}
    `
  }
}
