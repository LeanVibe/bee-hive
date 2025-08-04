import { LitElement, html, css } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { getSystemHealthService, getMetricsService } from '../services'
import type { SystemHealth, PerformanceSnapshot } from '../services'
import '../components/common/loading-spinner'

interface SystemMetric {
  id: string
  name: string
  value: number
  unit: string
  status: 'healthy' | 'warning' | 'critical'
  trend: 'up' | 'down' | 'stable'
  history: number[]
  threshold: {
    warning: number
    critical: number
  }
}

interface ServiceStatus {
  id: string
  name: string
  status: 'online' | 'offline' | 'degraded'
  uptime: number
  lastCheck: string
  responseTime: number
  errorRate: number
  description: string
}

@customElement('system-health-view')
export class SystemHealthView extends LitElement {
  @state() private metrics: SystemMetric[] = []
  @state() private services: ServiceStatus[] = []
  @state() private systemHealth: SystemHealth | null = null
  @state() private performanceData: PerformanceSnapshot | null = null
  @state() private isLoading: boolean = true
  @state() private error: string = ''
  @state() private lastRefresh: Date | null = null
  @state() private autoRefresh: boolean = true
  @state() private systemHealthService = getSystemHealthService()
  @state() private metricsService = getMetricsService()
  @state() private monitoringActive: boolean = false

  private refreshInterval?: number

  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: #f9fafb;
    }

    .health-container {
      display: flex;
      flex-direction: column;
      height: 100%;
      max-width: 1400px;
      margin: 0 auto;
    }

    .health-header {
      background: white;
      border-bottom: 1px solid #e5e7eb;
      padding: 2rem;
      position: sticky;
      top: 0;
      z-index: 10;
    }

    .header-content {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
    }

    .page-title {
      font-size: 1.875rem;
      font-weight: 700;
      color: #111827;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .title-icon {
      width: 32px;
      height: 32px;
      color: #3b82f6;
    }

    .header-actions {
      display: flex;
      gap: 1rem;
      align-items: center;
      flex-wrap: wrap;
    }

    .refresh-button {
      background: #3b82f6;
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .refresh-button:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }

    .refresh-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }

    .auto-refresh-toggle {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.875rem;
      color: #6b7280;
    }

    .toggle-switch {
      position: relative;
      width: 44px;
      height: 24px;
      background: #d1d5db;
      border-radius: 12px;
      cursor: pointer;
      transition: background 0.2s ease;
    }

    .toggle-switch.active {
      background: #3b82f6;
    }

    .toggle-switch::after {
      content: '';
      position: absolute;
      top: 2px;
      left: 2px;
      width: 20px;
      height: 20px;
      background: white;
      border-radius: 50%;
      transition: transform 0.2s ease;
    }

    .toggle-switch.active::after {
      transform: translateX(20px);
    }

    .last-refresh {
      font-size: 0.75rem;
      color: #9ca3af;
    }

    .health-content {
      flex: 1;
      overflow-y: auto;
      padding: 2rem;
    }

    .health-overview {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .overview-card {
      background: white;
      border-radius: 0.75rem;
      border: 1px solid #e5e7eb;
      padding: 1.5rem;
      text-align: center;
      position: relative;
      overflow: hidden;
    }

    .overview-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 4px;
    }

    .overview-card.healthy::before {
      background: #10b981;
    }

    .overview-card.warning::before {
      background: #f59e0b;
    }

    .overview-card.critical::before {
      background: #ef4444;
    }

    .overview-value {
      font-size: 2rem;
      font-weight: 700;
      margin: 0 0 0.25rem 0;
    }

    .overview-card.healthy .overview-value {
      color: #10b981;
    }

    .overview-card.warning .overview-value {
      color: #f59e0b;
    }

    .overview-card.critical .overview-value {
      color: #ef4444;
    }

    .overview-label {
      font-size: 0.875rem;
      color: #6b7280;
      margin: 0;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }

    .health-sections {
      display: grid;
      grid-template-columns: 1fr;
      gap: 2rem;
    }

    .section-card {
      background: white;
      border-radius: 0.75rem;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }

    .section-header {
      background: #f8fafc;
      border-bottom: 1px solid #e5e7eb;
      padding: 1rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .section-title {
      font-size: 1.125rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .section-icon {
      width: 20px;
      height: 20px;
      color: #6b7280;
    }

    .section-status {
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 500;
    }

    .section-status.healthy {
      background: #d1fae5;
      color: #065f46;
    }

    .section-status.warning {
      background: #fef3c7;
      color: #92400e;
    }

    .section-status.critical {
      background: #fee2e2;
      color: #991b1b;
    }

    .section-content {
      padding: 1.5rem;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1rem;
    }

    .metric-item {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      position: relative;
    }

    .metric-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }

    .metric-name {
      font-size: 0.875rem;
      font-weight: 500;
      color: #374151;
      margin: 0;
    }

    .metric-status {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .metric-status.healthy {
      background: #10b981;
    }

    .metric-status.warning {
      background: #f59e0b;
    }

    .metric-status.critical {
      background: #ef4444;
      animation: pulse 2s infinite;
    }

    .metric-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin: 0 0 0.25rem 0;
    }

    .metric-unit {
      font-size: 0.75rem;
      color: #6b7280;
    }

    .metric-trend {
      position: absolute;
      top: 1rem;
      right: 1rem;
      width: 16px;
      height: 16px;
    }

    .metric-trend.up {
      color: #10b981;
    }

    .metric-trend.down {
      color: #ef4444;
    }

    .metric-trend.stable {
      color: #6b7280;
    }

    .services-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .service-item {
      display: flex;
      align-items: center;
      justify-content: between;
      padding: 1rem;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      transition: all 0.2s ease;
    }

    .service-item:hover {
      border-color: #3b82f6;
      box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    }

    .service-info {
      flex: 1;
    }

    .service-name {
      font-size: 1rem;
      font-weight: 600;
      color: #111827;
      margin: 0 0 0.25rem 0;
    }

    .service-description {
      font-size: 0.875rem;
      color: #6b7280;
      margin: 0;
    }

    .service-metrics {
      display: flex;
      gap: 1.5rem;
      align-items: center;
      margin-right: 1rem;
    }

    .service-metric {
      text-align: center;
    }

    .service-metric-value {
      font-size: 0.875rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
    }

    .service-metric-label {
      font-size: 0.625rem;
      color: #9ca3af;
      margin: 0;
      text-transform: uppercase;
    }

    .service-status-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .service-status-indicator.online {
      background: #10b981;
      animation: pulse 2s infinite;
    }

    .service-status-indicator.offline {
      background: #ef4444;
    }

    .service-status-indicator.degraded {
      background: #f59e0b;
    }

    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 60vh;
      text-align: center;
      color: #6b7280;
    }

    .loading-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 60vh;
      gap: 1rem;
    }

    .error-state {
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #dc2626;
      padding: 1rem;
      border-radius: 0.5rem;
      margin: 1rem;
      text-align: center;
    }

    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.5;
      }
    }

    /* Responsive Design */
    @media (max-width: 1024px) {
      .health-sections {
        grid-template-columns: 1fr;
      }

      .metrics-grid {
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      }
    }

    @media (max-width: 768px) {
      .health-header {
        padding: 1rem;
      }

      .page-title {
        font-size: 1.5rem;
      }

      .health-content {
        padding: 1rem;
      }

      .header-content {
        flex-direction: column;
        align-items: flex-start;
      }

      .header-actions {
        width: 100%;
        justify-content: space-between;
      }

      .overview-card {
        padding: 1rem;
      }

      .service-metrics {
        flex-direction: column;
        gap: 0.5rem;
        margin-right: 0.5rem;
      }

      .service-item {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.75rem;
      }
    }

    @media (max-width: 640px) {
      .health-overview {
        grid-template-columns: repeat(2, 1fr);
      }

      .metrics-grid {
        grid-template-columns: 1fr;
      }
    }
  `

  async connectedCallback() {
    super.connectedCallback()
    await this.initializeHealthServices()
    await this.loadHealthData()
    this.setupAutoRefresh()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.clearAutoRefresh()
    this.stopMonitoring()
  }
  
  /**
   * Initialize health services with real-time monitoring
   */
  private async initializeHealthServices() {
    try {
      // Set up event listeners for real-time updates
      this.systemHealthService.addEventListener('healthChanged', this.handleHealthChanged.bind(this))
      this.metricsService.addEventListener('metricsUpdated', this.handleMetricsUpdated.bind(this))
      
      // Start monitoring for real-time updates
      this.startMonitoring()
      
      console.log('Health services initialized successfully')
      
    } catch (error) {
      console.error('Failed to initialize health services:', error)
      this.error = 'Failed to initialize health services'
    }
  }
  
  /**
   * Start real-time monitoring
   */
  private startMonitoring() {
    if (this.monitoringActive) return
    
    try {
      this.systemHealthService.startMonitoring()
      this.metricsService.startMonitoring()
      this.monitoringActive = true
      console.log('Health monitoring started')
      
    } catch (error) {
      console.error('Failed to start health monitoring:', error)
    }
  }
  
  /**
   * Stop monitoring
   */
  private stopMonitoring() {
    if (!this.monitoringActive) return
    
    try {
      this.systemHealthService.stopMonitoring()
      this.metricsService.stopMonitoring()
      this.monitoringActive = false
      console.log('Health monitoring stopped')
      
    } catch (error) {
      console.error('Failed to stop health monitoring:', error)
    }
  }
  
  /**
   * Real-time event handlers
   */
  private handleHealthChanged = (event: CustomEvent) => {
    this.systemHealth = event.detail.health
    this.transformHealthDataToUI()
    console.log('System health updated')
  }
  
  private handleMetricsUpdated = (event: CustomEvent) => {
    this.performanceData = event.detail.metrics
    this.transformMetricsDataToUI()
    console.log('Performance metrics updated')
  }

  private async loadHealthData() {
    this.isLoading = true
    this.error = ''

    try {
      // Load real data using integrated services
      const [healthData, metricsData] = await Promise.all([
        this.systemHealthService.getSystemHealth(),
        this.metricsService.getCurrentMetrics()
      ])
      
      this.systemHealth = healthData  
      this.performanceData = metricsData
      
      // Transform data to UI format
      this.transformHealthDataToUI()
      this.transformMetricsDataToUI()
      
      this.lastRefresh = new Date()
      
      console.log('Health data loaded from services')

    } catch (error) {
      console.error('Failed to load health data:', error)
      this.error = error instanceof Error ? error.message : 'Failed to load health data'
      
      // Fall back to mock data for demonstration
      this.loadMockData()
    } finally {
      this.isLoading = false
    }
  }
  
  /**
   * Transform SystemHealth data to UI format
   */
  private transformHealthDataToUI() {
    if (!this.systemHealth) return
    
    this.services = this.systemHealth.components.map(component => ({
      id: component.name.toLowerCase().replace(/\s+/g, '-'),
      name: component.name,
      status: component.status === 'healthy' ? 'online' : 
             component.status === 'degraded' ? 'degraded' : 'offline',
      uptime: component.uptime || 99.0,
      lastCheck: component.last_check || new Date().toISOString(),
      responseTime: component.response_time || 0,
      errorRate: component.error_rate || 0,
      description: component.details || ''
    }))
  }
  
  /**
   * Transform PerformanceSnapshot data to UI format
   */
  private transformMetricsDataToUI() {
    if (!this.performanceData) return
    
    const systemMetrics = this.performanceData.system_metrics
    
    this.metrics = [
      {
        id: 'cpu',
        name: 'CPU Usage',
        value: systemMetrics.cpu_usage,
        unit: '%',
        status: systemMetrics.cpu_usage > 90 ? 'critical' : 
               systemMetrics.cpu_usage > 70 ? 'warning' : 'healthy',
        trend: 'stable', // Could be derived from historical data
        history: [systemMetrics.cpu_usage - 5, systemMetrics.cpu_usage - 2, systemMetrics.cpu_usage],
        threshold: { warning: 70, critical: 90 }
      },
      {
        id: 'memory',
        name: 'Memory Usage',
        value: systemMetrics.memory_usage,
        unit: '%',
        status: systemMetrics.memory_usage > 95 ? 'critical' : 
               systemMetrics.memory_usage > 80 ? 'warning' : 'healthy',
        trend: 'stable',
        history: [systemMetrics.memory_usage - 3, systemMetrics.memory_usage - 1, systemMetrics.memory_usage],
        threshold: { warning: 80, critical: 95 }
      },
      {
        id: 'disk',
        name: 'Disk Usage',
        value: systemMetrics.disk_usage,
        unit: '%',
        status: systemMetrics.disk_usage > 95 ? 'critical' : 
               systemMetrics.disk_usage > 80 ? 'warning' : 'healthy',
        trend: 'stable',
        history: [systemMetrics.disk_usage - 2, systemMetrics.disk_usage - 1, systemMetrics.disk_usage],
        threshold: { warning: 80, critical: 95 }
      },
      {
        id: 'network',
        name: 'Network I/O',
        value: systemMetrics.network_io,
        unit: 'MB/s',
        status: systemMetrics.network_io > 800 ? 'critical' : 
               systemMetrics.network_io > 500 ? 'warning' : 'healthy',
        trend: 'stable',
        history: [systemMetrics.network_io - 10, systemMetrics.network_io - 5, systemMetrics.network_io],
        threshold: { warning: 500, critical: 800 }
      }
    ]
  }
  
  /**
   * Fallback mock data for demonstration
   */
  private loadMockData() {
    this.metrics = [
        {
          id: 'cpu',
          name: 'CPU Usage',
          value: 45.2,
          unit: '%',
          status: 'healthy',
          trend: 'stable',
          history: [42, 44, 45, 43, 45],
          threshold: { warning: 70, critical: 90 }
        },
        {
          id: 'memory',
          name: 'Memory Usage',
          value: 67.8,
          unit: '%',
          status: 'healthy',
          trend: 'up',
          history: [60, 62, 65, 66, 68],
          threshold: { warning: 80, critical: 95 }
        },
        {
          id: 'disk',
          name: 'Disk Usage',
          value: 34.1,
          unit: '%',
          status: 'healthy',
          trend: 'stable',
          history: [32, 33, 34, 35, 34],
          threshold: { warning: 80, critical: 95 }
        },
        {
          id: 'network',
          name: 'Network I/O',
          value: 125.6,
          unit: 'MB/s',
          status: 'healthy',
          trend: 'down',
          history: [140, 135, 130, 128, 126],
          threshold: { warning: 500, critical: 800 }
        }
      ]

      this.services = [
        {
          id: 'api-server',
          name: 'API Server',
          status: 'online',
          uptime: 99.8,
          lastCheck: new Date().toISOString(),
          responseTime: 145,
          errorRate: 0.1,
          description: 'Main REST API service'
        },
        {
          id: 'database',
          name: 'PostgreSQL Database',
          status: 'online',
          uptime: 99.9,
          lastCheck: new Date().toISOString(),
          responseTime: 25,
          errorRate: 0.0,
          description: 'Primary application database'
        },
        {
          id: 'redis',
          name: 'Redis Cache',
          status: 'online',
          uptime: 99.7,
          lastCheck: new Date().toISOString(),
          responseTime: 5,
          errorRate: 0.0,
          description: 'In-memory data store and cache'
        },
        {
          id: 'websocket',
          name: 'WebSocket Service',
          status: 'degraded',
          uptime: 98.5,
          lastCheck: new Date().toISOString(),
          responseTime: 230,
          errorRate: 1.2,
          description: 'Real-time communication service'
        }
      ]

      this.lastRefresh = new Date()
  }

  private setupAutoRefresh() {
    if (this.autoRefresh) {
      this.refreshInterval = window.setInterval(() => {
        this.loadHealthData()
      }, 30000) // Refresh every 30 seconds
    }
  }

  private clearAutoRefresh() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval)
      this.refreshInterval = undefined
    }
  }

  private async handleRefresh() {
    await this.loadHealthData()
  }

  private handleAutoRefreshToggle() {
    this.autoRefresh = !this.autoRefresh
    
    if (this.autoRefresh) {
      this.setupAutoRefresh()
    } else {
      this.clearAutoRefresh()
    }
  }

  private get systemStatus() {
    const criticalMetrics = this.metrics.filter(m => m.status === 'critical').length
    const warningMetrics = this.metrics.filter(m => m.status === 'warning').length
    const offlineServices = this.services.filter(s => s.status === 'offline').length
    const degradedServices = this.services.filter(s => s.status === 'degraded').length

    if (criticalMetrics > 0 || offlineServices > 0) {
      return 'critical'
    } else if (warningMetrics > 0 || degradedServices > 0) {
      return 'warning'
    } else {
      return 'healthy'
    }
  }

  private formatUptime(uptime: number): string {
    return `${uptime.toFixed(1)}%`
  }

  private formatLastRefresh(): string {
    if (!this.lastRefresh) return 'Never'
    
    const now = new Date()
    const diffMs = now.getTime() - this.lastRefresh.getTime()
    const diffSecs = Math.floor(diffMs / 1000)
    
    if (diffSecs < 60) return `${diffSecs}s ago`
    
    const diffMins = Math.floor(diffSecs / 60)
    return `${diffMins}m ago`
  }

  private renderTrendIcon(trend: string) {
    const icons = {
      up: html`<svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"/>
      </svg>`,
      down: html`<svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"/>
      </svg>`,
      stable: html`<svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/>
      </svg>`
    }
    return icons[trend as keyof typeof icons]
  }

  render() {
    if (this.isLoading && this.metrics.length === 0) {
      return html`
        <div class="loading-state">
          <loading-spinner size="large"></loading-spinner>
          <p>Loading system health data...</p>
        </div>
      `
    }

    if (this.error) {
      return html`
        <div class="error-state">
          <p><strong>Error:</strong> ${this.error}</p>
          <button class="refresh-button" @click=${this.handleRefresh}>
            Try Again
          </button>
        </div>
      `
    }

    const systemStatus = this.systemStatus
    const healthyServices = this.services.filter(s => s.status === 'online').length
    const totalServices = this.services.length

    return html`
      <div class="health-container">
        <div class="health-header">
          <div class="header-content">
            <h1 class="page-title">
              <svg class="title-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
              </svg>
              System Health
            </h1>
            
            <div class="header-actions">
              <div class="auto-refresh-toggle">
                <span>Auto-refresh</span>
                <div 
                  class="toggle-switch ${this.autoRefresh ? 'active' : ''}"
                  @click=${this.handleAutoRefreshToggle}
                ></div>
              </div>
              
              <div class="last-refresh">
                Last updated: ${this.formatLastRefresh()}
              </div>
              
              <button 
                class="refresh-button" 
                @click=${this.handleRefresh}
                ?disabled=${this.isLoading}
              >
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                </svg>
                Refresh
              </button>
            </div>
          </div>
        </div>

        <div class="health-content">
          <div class="health-overview">
            <div class="overview-card ${systemStatus}">
              <p class="overview-value">
                ${systemStatus === 'healthy' ? '✓' : systemStatus === 'warning' ? '⚠' : '✗'}
              </p>
              <p class="overview-label">System Status</p>
            </div>
            
            <div class="overview-card healthy">
              <p class="overview-value">${healthyServices}/${totalServices}</p>
              <p class="overview-label">Services Online</p>
            </div>
            
            <div class="overview-card healthy">
              <p class="overview-value">${this.metrics.filter(m => m.status === 'healthy').length}</p>
              <p class="overview-label">Healthy Metrics</p>
            </div>
            
            <div class="overview-card ${this.metrics.some(m => m.status === 'critical') ? 'critical' : 'healthy'}">
              <p class="overview-value">${this.metrics.filter(m => m.status === 'critical').length}</p>
              <p class="overview-label">Critical Alerts</p>
            </div>
          </div>

          <div class="health-sections">
            <div class="section-card">
              <div class="section-header">
                <h2 class="section-title">
                  <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                  </svg>
                  System Metrics
                </h2>
                <span class="section-status ${systemStatus}">
                  ${systemStatus}
                </span>
              </div>
              
              <div class="section-content">
                <div class="metrics-grid">
                  ${this.metrics.map(metric => html`
                    <div class="metric-item">
                      <div class="metric-header">
                        <h3 class="metric-name">${metric.name}</h3>
                        <div class="metric-status ${metric.status}"></div>
                      </div>
                      <div class="metric-value">
                        ${metric.value}
                        <span class="metric-unit">${metric.unit}</span>
                      </div>
                      <div class="metric-trend ${metric.trend}">
                        ${this.renderTrendIcon(metric.trend)}
                      </div>
                    </div>
                  `)}
                </div>
              </div>
            </div>

            <div class="section-card">
              <div class="section-header">
                <h2 class="section-title">
                  <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2"/>
                  </svg>
                  Services
                </h2>
                <span class="section-status ${healthyServices === totalServices ? 'healthy' : 'warning'}">
                  ${healthyServices}/${totalServices} Online
                </span>
              </div>
              
              <div class="section-content">
                <div class="services-list">
                  ${this.services.map(service => html`
                    <div class="service-item">
                      <div class="service-info">
                        <h3 class="service-name">${service.name}</h3>
                        <p class="service-description">${service.description}</p>
                      </div>
                      
                      <div class="service-metrics">
                        <div class="service-metric">
                          <p class="service-metric-value">${this.formatUptime(service.uptime)}</p>
                          <p class="service-metric-label">Uptime</p>
                        </div>
                        <div class="service-metric">
                          <p class="service-metric-value">${service.responseTime}ms</p>
                          <p class="service-metric-label">Response</p>
                        </div>
                        <div class="service-metric">
                          <p class="service-metric-value">${service.errorRate}%</p>
                          <p class="service-metric-label">Error Rate</p>
                        </div>
                      </div>
                      
                      <div class="service-status-indicator ${service.status}"></div>
                    </div>
                  `)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `
  }
}