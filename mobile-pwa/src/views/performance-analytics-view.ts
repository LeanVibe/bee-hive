/**
 * Performance Analytics View
 * 
 * Main view component for the Performance Analytics Dashboard
 * Integrates the enhanced performance analytics panel with real-time data
 */

import { LitElement, html, css } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { performanceAnalyticsService } from '../services/performance-analytics'
import type { PerformanceData, TimeRange } from '../components/dashboard/enhanced-performance-analytics-panel'
import '../components/dashboard/enhanced-performance-analytics-panel'
import '../components/common/loading-spinner'
import '../components/common/error-boundary'

@customElement('performance-analytics-view')
export class PerformanceAnalyticsView extends LitElement {
  @state() private performanceData: PerformanceData | null = null
  @state() private isLoading: boolean = true
  @state() private error: string | null = null
  @state() private connectionStatus: string = 'disconnected'
  @state() private timeRange: TimeRange = '1h'
  @state() private autoRefresh: boolean = true
  @state() private lastUpdate: Date | null = null
  
  static styles = css`
    :host {
      display: block;
      height: 100%;
      width: 100%;
      background: #f8fafc;
      overflow: hidden;
    }
    
    .view-container {
      height: 100%;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    
    .view-header {
      flex-shrink: 0;
      background: white;
      border-bottom: 1px solid #e5e7eb;
      padding: 1rem 1.5rem;
      box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    
    .header-content {
      display: flex;
      justify-content: space-between;
      align-items: center;
      max-width: 1200px;
      margin: 0 auto;
    }
    
    .view-title {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    .title-icon {
      width: 24px;
      height: 24px;
      color: #3b82f6;
    }
    
    .title-text {
      font-size: 1.5rem;
      font-weight: 700;
      color: #1f2937;
      margin: 0;
    }
    
    .title-subtitle {
      font-size: 0.875rem;
      color: #6b7280;
      margin: 0.25rem 0 0 0;
    }
    
    .header-actions {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    
    .status-indicator {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      background: #f3f4f6;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      font-weight: 500;
    }
    
    .status-indicator.connected {
      background: #d1fae5;
      color: #065f46;
    }
    
    .status-indicator.connecting {
      background: #fef3c7;
      color: #92400e;
    }
    
    .status-indicator.disconnected {
      background: #fee2e2;
      color: #991b1b;
    }
    
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
    }
    
    .status-dot.connected {
      animation: pulse 2s infinite;
    }
    
    .refresh-button {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      background: #3b82f6;
      color: white;
      border: none;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    
    .refresh-button:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }
    
    .refresh-button:disabled {
      background: #9ca3af;
      cursor: not-allowed;
      transform: none;
    }
    
    .refresh-button.loading {
      pointer-events: none;
    }
    
    .refresh-icon {
      width: 16px;
      height: 16px;
    }
    
    .refresh-icon.spinning {
      animation: spin 1s linear infinite;
    }
    
    .view-content {
      flex: 1;
      overflow: hidden;
      position: relative;
    }
    
    .dashboard-container {
      height: 100%;
      max-width: 1200px;
      margin: 0 auto;
      padding: 1.5rem;
      overflow: hidden;
    }
    
    .error-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      text-align: center;
      color: #6b7280;
      padding: 2rem;
    }
    
    .error-icon {
      width: 64px;
      height: 64px;
      color: #ef4444;
      margin-bottom: 1rem;
    }
    
    .error-title {
      font-size: 1.25rem;
      font-weight: 600;
      color: #1f2937;
      margin-bottom: 0.5rem;
    }
    
    .error-message {
      font-size: 0.875rem;
      margin-bottom: 1.5rem;
      max-width: 400px;
    }
    
    .error-actions {
      display: flex;
      gap: 1rem;
    }
    
    .retry-button {
      padding: 0.75rem 1.5rem;
      background: #3b82f6;
      color: white;
      border: none;
      border-radius: 0.5rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    
    .retry-button:hover {
      background: #2563eb;
    }
    
    .loading-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(248, 250, 252, 0.8);
      backdrop-filter: blur(4px);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10;
    }
    
    .loading-content {
      text-align: center;
      color: #6b7280;
    }
    
    .loading-text {
      margin-top: 1rem;
      font-size: 0.875rem;
      font-weight: 500;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
      .view-header {
        padding: 1rem;
      }
      
      .header-content {
        flex-direction: column;
        gap: 1rem;
        align-items: stretch;
      }
      
      .header-actions {
        justify-content: space-between;
      }
      
      .title-subtitle {
        font-size: 0.75rem;
      }
      
      .dashboard-container {
        padding: 1rem;
      }
      
      .error-actions {
        flex-direction: column;
        width: 100%;
        max-width: 200px;
      }
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `
  
  constructor() {
    super()
    this.setupServiceListeners()
  }
  
  async connectedCallback() {
    super.connectedCallback()
    await this.initializeView()
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    this.cleanup()
  }
  
  private async initializeView() {
    try {
      this.isLoading = true
      this.error = null
      
      // Initialize service if not already done
      if (!performanceAnalyticsService.getLastMetrics()) {
        await performanceAnalyticsService.initialize()
      }
      
      // Load initial data
      await this.loadPerformanceData()
      
      this.isLoading = false
      
    } catch (error) {
      console.error('❌ Error initializing performance analytics view:', error)
      this.error = error instanceof Error ? error.message : 'Failed to load performance data'
      this.isLoading = false
    }
  }
  
  private setupServiceListeners() {
    // Real-time data updates
    performanceAnalyticsService.on('real-time-update', (data: PerformanceData) => {
      this.performanceData = data
      this.lastUpdate = new Date()
      this.error = null
    })
    
    performanceAnalyticsService.on('data-updated', (data: PerformanceData) => {
      this.performanceData = data
      this.lastUpdate = new Date()
      this.error = null
    })
    
    // Connection status updates
    performanceAnalyticsService.on('connection-status-changed', (status: string) => {
      this.connectionStatus = status
    })
    
    // Error handling
    performanceAnalyticsService.on('error', (error: Error) => {
      console.error('Performance analytics service error:', error)
      if (!this.performanceData) {
        this.error = error.message
      }
    })
    
    // WebSocket events
    performanceAnalyticsService.on('websocket-connected', () => {
      console.log('✅ Performance WebSocket connected')
    })
    
    performanceAnalyticsService.on('websocket-disconnected', () => {
      console.log('🔌 Performance WebSocket disconnected')
    })
    
    // Alert events
    performanceAnalyticsService.on('real-time-alert', (alert: any) => {
      this.showAlert(alert)
    })
    
    performanceAnalyticsService.on('anomaly-detected', (anomaly: any) => {
      this.showAnomalyAlert(anomaly)
    })
    
    performanceAnalyticsService.on('regression-detected', (regression: any) => {
      this.showRegressionAlert(regression)
    })
  }
  
  private async loadPerformanceData(forceRefresh = false) {
    try {
      console.log(`📊 Loading performance data for time range: ${this.timeRange}`)
      
      this.performanceData = await performanceAnalyticsService.fetchPerformanceData(
        this.timeRange,
        forceRefresh
      )
      
      this.lastUpdate = new Date()
      this.error = null
      
    } catch (error) {
      console.error('❌ Error loading performance data:', error)
      throw error
    }
  }
  
  private async handleRefresh() {
    if (this.isLoading) return
    
    try {
      this.isLoading = true
      await this.loadPerformanceData(true)
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to refresh data'
    } finally {
      this.isLoading = false
    }
  }
  
  private async handleTimeRangeChange(event: CustomEvent) {
    const newTimeRange = event.detail.timeRange as TimeRange
    if (newTimeRange === this.timeRange) return
    
    this.timeRange = newTimeRange
    
    try {
      this.isLoading = true
      await this.loadPerformanceData()
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to load data for time range'
    } finally {
      this.isLoading = false
    }
  }
  
  private handleAutoRefreshToggle(event: CustomEvent) {
    this.autoRefresh = event.detail.enabled
    performanceAnalyticsService.setAutoRefresh(this.autoRefresh)
  }
  
  private async handleRetry() {
    await this.initializeView()
  }
  
  private showAlert(alert: any) {
    // Show toast notification for real-time alerts
    console.log('🚨 Real-time alert:', alert)
    
    this.dispatchEvent(new CustomEvent('performance-alert', {
      detail: { alert },
      bubbles: true,
      composed: true
    }))
  }
  
  private showAnomalyAlert(anomaly: any) {
    console.log('🔍 Anomaly detected:', anomaly)
    
    this.dispatchEvent(new CustomEvent('performance-anomaly', {
      detail: { anomaly },
      bubbles: true,
      composed: true
    }))
  }
  
  private showRegressionAlert(regression: any) {
    console.log('📉 Regression detected:', regression)
    
    this.dispatchEvent(new CustomEvent('performance-regression', {
      detail: { regression },
      bubbles: true,
      composed: true
    }))
  }
  
  private cleanup() {
    // Clean up service listeners if needed
    // Service handles its own lifecycle
  }
  
  private renderHeader() {
    return html`
      <div class="view-header">
        <div class="header-content">
          <div class="view-title">
            <svg class="title-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>
            <div>
              <h1 class="title-text">Performance Analytics</h1>
              <p class="title-subtitle">Real-time monitoring and optimization insights</p>
            </div>
          </div>
          
          <div class="header-actions">
            <div class="status-indicator ${this.connectionStatus}">
              <div class="status-dot ${this.connectionStatus}"></div>
              ${this.connectionStatus === 'connected' ? 'Live Data' :
                this.connectionStatus === 'connecting' ? 'Connecting...' : 'Offline Mode'}
            </div>
            
            <button 
              class="refresh-button ${this.isLoading ? 'loading' : ''}"
              @click=${this.handleRefresh}
              ?disabled=${this.isLoading}
            >
              <svg class="refresh-icon ${this.isLoading ? 'spinning' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
              </svg>
              ${this.isLoading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>
      </div>
    `
  }
  
  private renderErrorState() {
    return html`
      <div class="error-state">
        <svg class="error-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <h2 class="error-title">Unable to Load Performance Data</h2>
        <p class="error-message">${this.error}</p>
        <div class="error-actions">
          <button class="retry-button" @click=${this.handleRetry}>
            Try Again
          </button>
        </div>
      </div>
    `
  }
  
  private renderLoadingOverlay() {
    return html`
      <div class="loading-overlay">
        <div class="loading-content">
          <loading-spinner size="large"></loading-spinner>
          <div class="loading-text">Loading performance analytics...</div>
        </div>
      </div>
    `
  }
  
  private renderDashboard() {
    if (this.error && !this.performanceData) {
      return this.renderErrorState()
    }
    
    return html`
      <div class="dashboard-container">
        <error-boundary>
          <enhanced-performance-analytics-panel
            .performanceData=${this.performanceData}
            .realtime=${this.connectionStatus === 'connected'}
            .timeRange=${this.timeRange}
            .mobile=${window.innerWidth < 768}
            @time-range-changed=${this.handleTimeRangeChange}
            @auto-refresh-toggled=${this.handleAutoRefreshToggle}
            @export-requested=${this.handleExportRequest}
            @alert-dismissed=${this.handleAlertDismiss}
          ></enhanced-performance-analytics-panel>
        </error-boundary>
      </div>
    `
  }
  
  private handleExportRequest(event: CustomEvent) {
    // Handle chart/data export requests
    console.log('📊 Export requested:', event.detail)
  }
  
  private handleAlertDismiss(event: CustomEvent) {
    // Handle alert dismissal
    console.log('✅ Alert dismissed:', event.detail)
  }
  
  render() {
    return html`
      <div class="view-container">
        ${this.renderHeader()}
        
        <div class="view-content">
          ${this.renderDashboard()}
          ${this.isLoading ? this.renderLoadingOverlay() : ''}
        </div>
      </div>
    `
  }
}