/**
 * Enhanced Performance Analytics Dashboard - Phase 2.1 Implementation
 * 
 * Advanced performance monitoring dashboard with real-time visualizations,
 * automated performance validation, and comprehensive analytics framework
 * as defined in the LeanVibe Agent Hive dashboard enhancement strategic plan.
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { Chart, registerables } from 'chart.js'
import 'chartjs-adapter-date-fns'

// Register Chart.js components
Chart.register(...registerables)

export interface PerformanceData {
  timestamp: string
  system_metrics: {
    cpu_usage: number
    memory_usage: number
    network_usage: number
    disk_usage: number
  }
  response_times: {
    api_response_time: number
    api_p95_response_time: number
    api_p99_response_time: number
    websocket_latency: number
    database_query_time: number
  }
  throughput: {
    requests_per_second: number
    peak_rps: number
    tasks_completed_per_hour: number
    agent_operations_per_minute: number
  }
  error_rates: {
    http_4xx_rate: number
    http_5xx_rate: number
    system_error_rate: number
    total_error_rate: number
  }
  capacity_metrics: {
    queue_length: number
    connection_pool_usage: number
    thread_pool_usage: number
    bottlenecks: string[]
  }
  alerts: Array<{
    id: string
    type: 'performance' | 'threshold' | 'anomaly' | 'regression'
    severity: 'critical' | 'warning' | 'info'
    message: string
    timestamp: string
    metric: string
    current_value: number
    threshold_value: number
    impact_assessment?: string
  }>
}

export type TimeRange = '1m' | '5m' | '15m' | '1h' | '6h' | '24h' | '7d'

@customElement('enhanced-performance-analytics-panel')
export class EnhancedPerformanceAnalyticsPanel extends LitElement {
  @property({ type: Object }) declare performanceData: PerformanceData | null
  @property({ type: Boolean }) declare realtime: boolean
  @property({ type: String }) declare timeRange: TimeRange
  @property({ type: Boolean }) declare mobile: boolean
  
  @state() private selectedTab: string = 'overview'
  @state() private autoRefresh: boolean = true
  @state() private lastUpdate: Date | null = null
  @state() private connectionStatus: 'connected' | 'disconnected' | 'connecting' = 'disconnected'
  @state() private chartInstances: Map<string, Chart> = new Map()
  @state() private historicalData: PerformanceData[] = []
  @state() private expandedCharts: Set<string> = new Set()
  @state() private showRegressionAlerts: boolean = true
  @state() private performanceTargets = {
    api_response_time: { warning: 500, critical: 1000 },
    cpu_usage: { warning: 70, critical: 90 },
    memory_usage: { warning: 80, critical: 95 },
    error_rate: { warning: 1, critical: 5 },
    requests_per_second: { target: 1000, warning: 800 }
  }
  
  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: white;
      border-radius: 0.75rem;
      border: 1px solid #e5e7eb;
      overflow: hidden;
      box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    .analytics-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1.25rem;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-bottom: 1px solid #e5e7eb;
    }
    
    .header-title {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 1.25rem;
      font-weight: 700;
    }
    
    .analytics-icon {
      width: 24px;
      height: 24px;
      flex-shrink: 0;
    }
    
    .header-controls {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    
    .time-range-selector {
      display: flex;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 0.5rem;
      padding: 0.25rem;
      gap: 0.25rem;
    }
    
    .time-range-btn {
      background: transparent;
      border: none;
      color: white;
      padding: 0.375rem 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s ease;
      opacity: 0.7;
    }
    
    .time-range-btn:hover {
      opacity: 1;
      background: rgba(255, 255, 255, 0.1);
    }
    
    .time-range-btn.active {
      background: rgba(255, 255, 255, 0.2);
      opacity: 1;
      font-weight: 600;
    }
    
    .connection-indicator {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.875rem;
    }
    
    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 2s infinite;
    }
    
    .status-dot.connecting {
      background: #f59e0b;
      animation: pulse 1s infinite;
    }
    
    .status-dot.disconnected {
      background: #ef4444;
      animation: none;
    }
    
    .refresh-controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .control-btn {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: white;
      padding: 0.5rem;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .control-btn:hover {
      background: rgba(255, 255, 255, 0.2);
    }
    
    .control-btn.active {
      background: rgba(255, 255, 255, 0.3);
      border-color: rgba(255, 255, 255, 0.4);
    }
    
    .analytics-content {
      height: calc(100% - 85px);
      overflow-y: auto;
    }
    
    .performance-tabs {
      display: flex;
      border-bottom: 2px solid #f1f5f9;
      background: #f8fafc;
      overflow-x: auto;
      scrollbar-width: none;
      -ms-overflow-style: none;
    }
    
    .performance-tabs::-webkit-scrollbar {
      display: none;
    }
    
    .tab-btn {
      background: none;
      border: none;
      padding: 1rem 1.5rem;
      cursor: pointer;
      transition: all 0.2s ease;
      font-size: 0.875rem;
      font-weight: 500;
      color: #64748b;
      white-space: nowrap;
      border-bottom: 3px solid transparent;
      position: relative;
    }
    
    .tab-btn:hover {
      color: #3b82f6;
      background: rgba(59, 130, 246, 0.05);
    }
    
    .tab-btn.active {
      color: #3b82f6;
      background: white;
      border-bottom-color: #3b82f6;
      font-weight: 600;
    }
    
    .tab-badge {
      position: absolute;
      top: 0.25rem;
      right: 0.25rem;
      background: #ef4444;
      color: white;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.125rem 0.375rem;
      border-radius: 1rem;
      min-width: 1.25rem;
      text-align: center;
    }
    
    .analytics-panel {
      padding: 1.5rem;
    }
    
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.5rem;
      margin-bottom: 2rem;
    }
    
    .metric-card {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 0.75rem;
      padding: 1.25rem;
      box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }
    
    .metric-card:hover {
      box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
      transform: translateY(-1px);
    }
    
    .metric-card.critical {
      border-left: 4px solid #ef4444;
      background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
    }
    
    .metric-card.warning {
      border-left: 4px solid #f59e0b;
      background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
    }
    
    .metric-card.healthy {
      border-left: 4px solid #10b981;
      background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    }
    
    .metric-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 1rem;
    }
    
    .metric-label {
      font-size: 0.875rem;
      font-weight: 600;
      color: #374151;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .metric-icon {
      width: 16px;
      height: 16px;
      opacity: 0.7;
    }
    
    .metric-trend {
      display: flex;
      align-items: center;
      gap: 0.375rem;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.25rem 0.5rem;
      border-radius: 0.375rem;
    }
    
    .trend-up {
      color: #dc2626;
      background: rgba(220, 38, 38, 0.1);
    }
    
    .trend-down {
      color: #059669;
      background: rgba(5, 150, 105, 0.1);
    }
    
    .trend-stable {
      color: #6b7280;
      background: rgba(107, 114, 128, 0.1);
    }
    
    .metric-value {
      font-size: 2rem;
      font-weight: 800;
      line-height: 1;
      margin-bottom: 0.5rem;
    }
    
    .metric-value.critical {
      color: #dc2626;
    }
    
    .metric-value.warning {
      color: #d97706;
    }
    
    .metric-value.healthy {
      color: #059669;
    }
    
    .metric-subtitle {
      font-size: 0.75rem;
      color: #6b7280;
      margin-bottom: 1rem;
    }
    
    .metric-progress {
      position: relative;
      height: 6px;
      background: #f1f5f9;
      border-radius: 3px;
      overflow: hidden;
    }
    
    .metric-progress-fill {
      height: 100%;
      border-radius: 3px;
      transition: all 0.5s ease;
      position: relative;
      overflow: hidden;
    }
    
    .metric-progress-fill.critical {
      background: linear-gradient(90deg, #dc2626, #ef4444);
    }
    
    .metric-progress-fill.warning {
      background: linear-gradient(90deg, #d97706, #f59e0b);
    }
    
    .metric-progress-fill.healthy {
      background: linear-gradient(90deg, #059669, #10b981);
    }
    
    .metric-progress-fill::after {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
      animation: shimmer 2s infinite;
    }
    
    .chart-container {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 0.75rem;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid #f1f5f9;
    }
    
    .chart-title {
      font-size: 1.125rem;
      font-weight: 700;
      color: #1f2937;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .chart-controls {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    .chart-toggle {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      color: #374151;
      padding: 0.5rem;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .chart-toggle:hover {
      background: #f1f5f9;
      border-color: #cbd5e1;
    }
    
    .chart-toggle.expanded {
      background: #3b82f6;
      color: white;
      border-color: #3b82f6;
    }
    
    .chart-canvas {
      height: 300px;
      position: relative;
    }
    
    .chart-canvas.expanded {
      height: 500px;
    }
    
    .alerts-section {
      margin-top: 2rem;
    }
    
    .alerts-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }
    
    .alerts-title {
      font-size: 1.125rem;
      font-weight: 700;
      color: #1f2937;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .alerts-filters {
      display: flex;
      gap: 0.5rem;
    }
    
    .alert-filter-btn {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      color: #6b7280;
      padding: 0.375rem 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    
    .alert-filter-btn:hover {
      background: #f1f5f9;
      color: #374151;
    }
    
    .alert-filter-btn.active {
      background: #3b82f6;
      color: white;
      border-color: #3b82f6;
    }
    
    .alerts-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    
    .alert-item {
      display: flex;
      align-items: flex-start;
      gap: 1rem;
      padding: 1rem;
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 0.75rem;
      transition: all 0.2s ease;
      position: relative;
      overflow: hidden;
    }
    
    .alert-item:hover {
      box-shadow: 0 2px 4px -1px rgb(0 0 0 / 0.1);
    }
    
    .alert-item.critical {
      border-left: 4px solid #ef4444;
      background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
    }
    
    .alert-item.warning {
      border-left: 4px solid #f59e0b;
      background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
    }
    
    .alert-item.info {
      border-left: 4px solid #3b82f6;
      background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
    }
    
    .alert-icon {
      width: 20px;
      height: 20px;
      flex-shrink: 0;
      margin-top: 0.125rem;
    }
    
    .alert-content {
      flex: 1;
      min-width: 0;
    }
    
    .alert-message {
      font-weight: 600;
      color: #1f2937;
      margin-bottom: 0.25rem;
      font-size: 0.875rem;
    }
    
    .alert-details {
      font-size: 0.75rem;
      color: #6b7280;
      margin-bottom: 0.5rem;
    }
    
    .alert-impact {
      font-size: 0.75rem;
      font-weight: 500;
      color: #374151;
      background: rgba(107, 114, 128, 0.1);
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      display: inline-block;
    }
    
    .alert-time {
      font-size: 0.75rem;
      color: #9ca3af;
      white-space: nowrap;
    }
    
    .empty-state {
      text-align: center;
      padding: 3rem 1rem;
      color: #6b7280;
    }
    
    .empty-state-icon {
      width: 48px;
      height: 48px;
      margin: 0 auto 1rem;
      opacity: 0.3;
    }
    
    .loading-state {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 200px;
      gap: 1rem;
      color: #6b7280;
    }
    
    .spinner {
      width: 24px;
      height: 24px;
      border: 2px solid #e5e7eb;
      border-top: 2px solid #3b82f6;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    
    .performance-summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
      padding: 1.5rem;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      border-radius: 0.75rem;
      border: 1px solid #e2e8f0;
    }
    
    .summary-item {
      text-align: center;
    }
    
    .summary-value {
      font-size: 1.5rem;
      font-weight: 800;
      color: #1f2937;
    }
    
    .summary-label {
      font-size: 0.75rem;
      color: #6b7280;
      font-weight: 500;
      margin-top: 0.25rem;
    }
    
    /* Mobile Responsive Styles */
    @media (max-width: 768px) {
      .analytics-header {
        flex-direction: column;
        gap: 1rem;
        align-items: stretch;
      }
      
      .header-controls {
        justify-content: space-between;
      }
      
      .time-range-selector {
        flex-wrap: wrap;
      }
      
      .metrics-grid {
        grid-template-columns: 1fr;
        gap: 1rem;
      }
      
      .analytics-panel {
        padding: 1rem;
      }
      
      .chart-container {
        padding: 1rem;
      }
      
      .chart-canvas {
        height: 250px;
      }
      
      .chart-canvas.expanded {
        height: 400px;
      }
      
      .performance-summary {
        grid-template-columns: repeat(2, 1fr);
        padding: 1rem;
      }
      
      .alert-item {
        flex-direction: column;
        gap: 0.75rem;
      }
      
      .alert-time {
        align-self: flex-end;
      }
    }
    
    /* Animation Keyframes */
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    @keyframes shimmer {
      0% { left: -100%; }
      100% { left: 100%; }
    }
    
    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    
    .metric-card,
    .chart-container,
    .alert-item {
      animation: slideIn 0.3s ease-out;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      :host {
        background: #1f2937;
        border-color: #374151;
      }
      
      .metric-card,
      .chart-container,
      .alert-item {
        background: #1f2937;
        border-color: #374151;
        color: #f9fafb;
      }
      
      .performance-tabs {
        background: #374151;
        border-color: #4b5563;
      }
      
      .tab-btn {
        color: #d1d5db;
      }
      
      .tab-btn.active {
        background: #1f2937;
        color: #3b82f6;
      }
    }
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {
      .metric-card,
      .chart-container,
      .alert-item {
        border-width: 2px;
      }
      
      .control-btn,
      .tab-btn {
        border-width: 2px;
      }
    }
    
    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
      
      .status-dot {
        animation: none;
      }
    }
  `
  
  constructor() {
    super()
    this.performanceData = null
    this.realtime = true
    this.timeRange = '1h'
    this.mobile = false
  }
  
  connectedCallback() {
    super.connectedCallback()
    this.initializeCharts()
    if (this.realtime && this.autoRefresh) {
      this.startRealTimeUpdates()
    }
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    this.destroyCharts()
    this.stopRealTimeUpdates()
  }
  
  private async initializeCharts() {
    // Charts will be initialized after first render when DOM elements are available
    await this.updateComplete
    this.createCharts()
  }
  
  private createCharts() {
    // Response Time Trends Chart
    this.createResponseTimeChart()
    
    // Throughput Chart
    this.createThroughputChart()
    
    // Error Rate Chart
    this.createErrorRateChart()
    
    // Resource Utilization Chart
    this.createResourceChart()
  }
  
  private createResponseTimeChart() {
    const canvas = this.shadowRoot?.querySelector('#response-time-chart') as HTMLCanvasElement
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.getTimeLabels(),
        datasets: [
          {
            label: 'API Response Time (ms)',
            data: this.getResponseTimeData(),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.4,
            fill: true
          },
          {
            label: 'P95 Response Time (ms)',
            data: this.getP95ResponseTimeData(),
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            tension: 0.4,
            borderDash: [5, 5]
          },
          {
            label: 'P99 Response Time (ms)',
            data: this.getP99ResponseTimeData(),
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            tension: 0.4,
            borderDash: [10, 5]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: {
            position: 'top'
          },
          tooltip: {
            backgroundColor: 'rgba(17, 24, 39, 0.9)',
            titleColor: '#f9fafb',
            bodyColor: '#f9fafb',
            borderColor: '#374151',
            borderWidth: 1
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              displayFormats: {
                minute: 'HH:mm',
                hour: 'HH:mm'
              }
            },
            title: {
              display: true,
              text: 'Time'
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Response Time (ms)'
            },
            grid: {
              color: 'rgba(107, 114, 128, 0.1)'
            }
          }
        },
        elements: {
          point: {
            radius: 2,
            hoverRadius: 6
          }
        }
      }
    })
    
    this.chartInstances.set('response-time', chart)
  }
  
  private createThroughputChart() {
    const canvas = this.shadowRoot?.querySelector('#throughput-chart') as HTMLCanvasElement
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: this.getTimeLabels(),
        datasets: [
          {
            label: 'Requests/Second',
            data: this.getThroughputData(),
            backgroundColor: 'rgba(16, 185, 129, 0.8)',
            borderColor: '#10b981',
            borderWidth: 1
          },
          {
            label: 'Peak RPS',
            data: this.getPeakRPSData(),
            type: 'line',
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            tension: 0.4,
            borderDash: [5, 5]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: {
            position: 'top'
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              displayFormats: {
                minute: 'HH:mm',
                hour: 'HH:mm'
              }
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Requests per Second'
            }
          }
        }
      }
    })
    
    this.chartInstances.set('throughput', chart)
  }
  
  private createErrorRateChart() {
    const canvas = this.shadowRoot?.querySelector('#error-rate-chart') as HTMLCanvasElement
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.getTimeLabels(),
        datasets: [
          {
            label: 'Total Error Rate (%)',
            data: this.getErrorRateData(),
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            tension: 0.4,
            fill: true
          },
          {
            label: '4xx Errors (%)',
            data: this.get4xxErrorData(),
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            tension: 0.4
          },
          {
            label: '5xx Errors (%)',
            data: this.get5xxErrorData(),
            borderColor: '#dc2626',
            backgroundColor: 'rgba(220, 38, 38, 0.1)',
            tension: 0.4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: {
            position: 'top'
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              displayFormats: {
                minute: 'HH:mm',
                hour: 'HH:mm'
              }
            }
          },
          y: {
            beginAtZero: true,
            max: 10,
            title: {
              display: true,
              text: 'Error Rate (%)'
            }
          }
        }
      }
    })
    
    this.chartInstances.set('error-rate', chart)
  }
  
  private createResourceChart() {
    const canvas = this.shadowRoot?.querySelector('#resource-chart') as HTMLCanvasElement
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.getTimeLabels(),
        datasets: [
          {
            label: 'CPU Usage (%)',
            data: this.getCPUData(),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.4
          },
          {
            label: 'Memory Usage (%)',
            data: this.getMemoryData(),
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            tension: 0.4
          },
          {
            label: 'Network Usage (%)',
            data: this.getNetworkData(),
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            tension: 0.4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: {
            position: 'top'
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              displayFormats: {
                minute: 'HH:mm',
                hour: 'HH:mm'
              }
            }
          },
          y: {
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: 'Usage (%)'
            }
          }
        }
      }
    })
    
    this.chartInstances.set('resource', chart)
  }
  
  private destroyCharts() {
    this.chartInstances.forEach(chart => chart.destroy())
    this.chartInstances.clear()
  }
  
  private startRealTimeUpdates() {
    // Implementation for WebSocket connection will be added
    this.connectionStatus = 'connecting'
    
    // Simulate connection for now
    setTimeout(() => {
      this.connectionStatus = 'connected'
    }, 1000)
  }
  
  private stopRealTimeUpdates() {
    this.connectionStatus = 'disconnected'
  }
  
  // Data generation methods (will be replaced with real data)
  private getTimeLabels(): string[] {
    const labels: string[] = []
    const now = new Date()
    const points = this.getDataPoints()
    
    for (let i = points - 1; i >= 0; i--) {
      const time = new Date(now.getTime() - (i * this.getTimeInterval()))
      labels.push(time.toISOString())
    }
    
    return labels
  }
  
  private getDataPoints(): number {
    switch (this.timeRange) {
      case '1m': return 60
      case '5m': return 60
      case '15m': return 60
      case '1h': return 60
      case '6h': return 72
      case '24h': return 144
      case '7d': return 168
      default: return 60
    }
  }
  
  private getTimeInterval(): number {
    switch (this.timeRange) {
      case '1m': return 1000 // 1 second
      case '5m': return 5000 // 5 seconds
      case '15m': return 15000 // 15 seconds
      case '1h': return 60000 // 1 minute
      case '6h': return 300000 // 5 minutes
      case '24h': return 600000 // 10 minutes
      case '7d': return 3600000 // 1 hour
      default: return 60000
    }
  }
  
  private getResponseTimeData(): number[] {
    return this.generateMockData(250, 50, this.getDataPoints())
  }
  
  private getP95ResponseTimeData(): number[] {
    return this.generateMockData(350, 75, this.getDataPoints())
  }
  
  private getP99ResponseTimeData(): number[] {
    return this.generateMockData(450, 100, this.getDataPoints())
  }
  
  private getThroughputData(): number[] {
    return this.generateMockData(850, 200, this.getDataPoints())
  }
  
  private getPeakRPSData(): number[] {
    return this.generateMockData(1050, 150, this.getDataPoints())
  }
  
  private getErrorRateData(): number[] {
    return this.generateMockData(2, 1, this.getDataPoints())
  }
  
  private get4xxErrorData(): number[] {
    return this.generateMockData(1.5, 0.5, this.getDataPoints())
  }
  
  private get5xxErrorData(): number[] {
    return this.generateMockData(0.5, 0.3, this.getDataPoints())
  }
  
  private getCPUData(): number[] {
    return this.generateMockData(65, 15, this.getDataPoints())
  }
  
  private getMemoryData(): number[] {
    return this.generateMockData(75, 10, this.getDataPoints())
  }
  
  private getNetworkData(): number[] {
    return this.generateMockData(45, 20, this.getDataPoints())
  }
  
  private generateMockData(base: number, variance: number, points: number): number[] {
    const data: number[] = []
    let current = base
    
    for (let i = 0; i < points; i++) {
      // Add some trend and randomness
      const trend = Math.sin(i / points * Math.PI * 2) * (variance * 0.3)
      const noise = (Math.random() - 0.5) * variance
      current = Math.max(0, base + trend + noise)
      data.push(Math.round(current * 10) / 10)
    }
    
    return data
  }
  
  private getMetricStatus(value: number, thresholds: { warning: number; critical: number }): 'healthy' | 'warning' | 'critical' {
    if (value >= thresholds.critical) return 'critical'
    if (value >= thresholds.warning) return 'warning'
    return 'healthy'
  }
  
  private getTrendDirection(current: number, previous: number): 'up' | 'down' | 'stable' {
    const change = ((current - previous) / previous) * 100
    if (Math.abs(change) < 2) return 'stable'
    return change > 0 ? 'up' : 'down'
  }
  
  private handleTimeRangeChange(newRange: TimeRange) {
    this.timeRange = newRange
    this.updateCharts()
    
    this.dispatchEvent(new CustomEvent('time-range-changed', {
      detail: { timeRange: newRange },
      bubbles: true,
      composed: true
    }))
  }
  
  private handleTabChange(tab: string) {
    this.selectedTab = tab
    
    this.dispatchEvent(new CustomEvent('tab-changed', {
      detail: { tab },
      bubbles: true,
      composed: true
    }))
  }
  
  private toggleAutoRefresh() {
    this.autoRefresh = !this.autoRefresh
    
    if (this.autoRefresh) {
      this.startRealTimeUpdates()
    } else {
      this.stopRealTimeUpdates()
    }
    
    this.dispatchEvent(new CustomEvent('auto-refresh-toggled', {
      detail: { enabled: this.autoRefresh },
      bubbles: true,
      composed: true
    }))
  }
  
  private toggleChartExpansion(chartId: string) {
    if (this.expandedCharts.has(chartId)) {
      this.expandedCharts.delete(chartId)
    } else {
      this.expandedCharts.add(chartId)
    }
    
    this.requestUpdate()
    
    // Resize chart after DOM update
    setTimeout(() => {
      const chart = this.chartInstances.get(chartId)
      if (chart) {
        chart.resize()
      }
    }, 100)
  }
  
  private updateCharts() {
    this.chartInstances.forEach(chart => {
      chart.data.labels = this.getTimeLabels()
      // Update datasets with new data based on time range
      chart.update('none')
    })
  }
  
  private exportChart(chartId: string) {
    const chart = this.chartInstances.get(chartId)
    if (chart) {
      const url = chart.toBase64Image()
      const link = document.createElement('a')
      link.download = `${chartId}-${new Date().toISOString().split('T')[0]}.png`
      link.href = url
      link.click()
    }
  }
  
  private renderTimeRangeSelector() {
    const ranges: { value: TimeRange; label: string }[] = [
      { value: '1m', label: '1M' },
      { value: '5m', label: '5M' },
      { value: '15m', label: '15M' },
      { value: '1h', label: '1H' },
      { value: '6h', label: '6H' },
      { value: '24h', label: '24H' },
      { value: '7d', label: '7D' }
    ]
    
    return html`
      <div class="time-range-selector">
        ${ranges.map(range => html`
          <button
            class="time-range-btn ${this.timeRange === range.value ? 'active' : ''}"
            @click=${() => this.handleTimeRangeChange(range.value)}
          >
            ${range.label}
          </button>
        `)}
      </div>
    `
  }
  
  private renderPerformanceOverview() {
    if (!this.performanceData) return this.renderLoadingState()
    
    const { system_metrics, response_times, throughput, error_rates } = this.performanceData
    
    const apiStatus = this.getMetricStatus(response_times.api_response_time, this.performanceTargets.api_response_time)
    const cpuStatus = this.getMetricStatus(system_metrics.cpu_usage, this.performanceTargets.cpu_usage)
    const memoryStatus = this.getMetricStatus(system_metrics.memory_usage, this.performanceTargets.memory_usage)
    const errorStatus = this.getMetricStatus(error_rates.total_error_rate, this.performanceTargets.error_rate)
    
    return html`
      <div class="performance-summary">
        <div class="summary-item">
          <div class="summary-value ${apiStatus}">${Math.round(response_times.api_response_time)}ms</div>
          <div class="summary-label">Avg Response Time</div>
        </div>
        <div class="summary-item">
          <div class="summary-value">${Math.round(throughput.requests_per_second)}</div>
          <div class="summary-label">Requests/Second</div>
        </div>
        <div class="summary-item">
          <div class="summary-value ${errorStatus}">${error_rates.total_error_rate.toFixed(2)}%</div>
          <div class="summary-label">Error Rate</div>
        </div>
        <div class="summary-item">
          <div class="summary-value ${cpuStatus}">${Math.round(system_metrics.cpu_usage)}%</div>
          <div class="summary-label">CPU Usage</div>
        </div>
      </div>
      
      <div class="metrics-grid">
        <div class="metric-card ${apiStatus}">
          <div class="metric-header">
            <div class="metric-label">
              <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
              </svg>
              API Response Time
            </div>
            <div class="metric-trend trend-stable">
              <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14"/>
              </svg>
              ±2ms
            </div>
          </div>
          <div class="metric-value ${apiStatus}">${Math.round(response_times.api_response_time)}ms</div>
          <div class="metric-subtitle">P95: ${Math.round(response_times.api_p95_response_time)}ms | P99: ${Math.round(response_times.api_p99_response_time)}ms</div>
          <div class="metric-progress">
            <div class="metric-progress-fill ${apiStatus}" style="width: ${Math.min((response_times.api_response_time / 1000) * 100, 100)}%"></div>
          </div>
        </div>
        
        <div class="metric-card healthy">
          <div class="metric-header">
            <div class="metric-label">
              <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/>
              </svg>
              Throughput
            </div>
            <div class="metric-trend trend-up">
              <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 17l9.2-9.2M17 17V7H7"/>
              </svg>
              +5%
            </div>
          </div>
          <div class="metric-value healthy">${Math.round(throughput.requests_per_second)}</div>
          <div class="metric-subtitle">Peak: ${Math.round(throughput.peak_rps)} RPS</div>
          <div class="metric-progress">
            <div class="metric-progress-fill healthy" style="width: ${Math.min((throughput.requests_per_second / 1200) * 100, 100)}%"></div>
          </div>
        </div>
        
        <div class="metric-card ${errorStatus}">
          <div class="metric-header">
            <div class="metric-label">
              <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
              Error Rate
            </div>
            <div class="metric-trend trend-down">
              <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 7l-9.2 9.2M7 7v10h10"/>
              </svg>
              -0.1%
            </div>
          </div>
          <div class="metric-value ${errorStatus}">${error_rates.total_error_rate.toFixed(2)}%</div>
          <div class="metric-subtitle">4xx: ${error_rates.http_4xx_rate.toFixed(2)}% | 5xx: ${error_rates.http_5xx_rate.toFixed(2)}%</div>
          <div class="metric-progress">
            <div class="metric-progress-fill ${errorStatus}" style="width: ${Math.min((error_rates.total_error_rate / 10) * 100, 100)}%"></div>
          </div>
        </div>
        
        <div class="metric-card ${cpuStatus}">
          <div class="metric-header">
            <div class="metric-label">
              <svg class="metric-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/>
              </svg>
              System Resources
            </div>
            <div class="metric-trend trend-stable">
              <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14"/>
              </svg>
              ±1%
            </div>
          </div>
          <div class="metric-value ${cpuStatus}">${Math.round(system_metrics.cpu_usage)}%</div>
          <div class="metric-subtitle">Memory: ${Math.round(system_metrics.memory_usage)}% | Disk: ${Math.round(system_metrics.disk_usage)}%</div>
          <div class="metric-progress">
            <div class="metric-progress-fill ${cpuStatus}" style="width: ${system_metrics.cpu_usage}%"></div>
          </div>
        </div>
      </div>
      
      <!-- Real-time Charts -->
      <div class="chart-container">
        <div class="chart-header">
          <h3 class="chart-title">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>
            Response Time Trends
          </h3>
          <div class="chart-controls">
            <button 
              class="chart-toggle ${this.expandedCharts.has('response-time') ? 'expanded' : ''}"
              @click=${() => this.toggleChartExpansion('response-time')}
              title="Toggle full screen"
            >
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/>
              </svg>
            </button>
            <button 
              class="chart-toggle"
              @click=${() => this.exportChart('response-time')}
              title="Export chart"
            >
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
              </svg>
            </button>
          </div>
        </div>
        <div class="chart-canvas ${this.expandedCharts.has('response-time') ? 'expanded' : ''}">
          <canvas id="response-time-chart"></canvas>
        </div>
      </div>
    `
  }
  
  private renderLoadingState() {
    return html`
      <div class="loading-state">
        <div class="spinner"></div>
        <span>Loading performance analytics...</span>
      </div>
    `
  }
  
  private renderCurrentTab() {
    switch (this.selectedTab) {
      case 'throughput':
        return this.renderThroughputAnalytics()
      case 'errors':
        return this.renderErrorAnalytics()
      case 'resources':
        return this.renderResourceAnalytics()
      case 'alerts':
        return this.renderAlertsAnalytics()
      case 'regression':
        return this.renderRegressionAnalytics()
      default:
        return this.renderPerformanceOverview()
    }
  }
  
  private renderThroughputAnalytics() {
    return html`
      <div class="chart-container">
        <div class="chart-header">
          <h3 class="chart-title">Throughput & Capacity Monitoring</h3>
          <div class="chart-controls">
            <button 
              class="chart-toggle ${this.expandedCharts.has('throughput') ? 'expanded' : ''}"
              @click=${() => this.toggleChartExpansion('throughput')}
            >
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/>
              </svg>
            </button>
          </div>
        </div>
        <div class="chart-canvas ${this.expandedCharts.has('throughput') ? 'expanded' : ''}">
          <canvas id="throughput-chart"></canvas>
        </div>
      </div>
      
      ${this.performanceData ? html`
        <div class="metrics-grid">
          <div class="metric-card healthy">
            <div class="metric-header">
              <div class="metric-label">Queue Length</div>
            </div>
            <div class="metric-value healthy">${this.performanceData.capacity_metrics.queue_length}</div>
            <div class="metric-subtitle">Tasks waiting for processing</div>
          </div>
          
          <div class="metric-card ${this.getMetricStatus(this.performanceData.capacity_metrics.connection_pool_usage, { warning: 80, critical: 95 })}">
            <div class="metric-header">
              <div class="metric-label">Connection Pool</div>
            </div>
            <div class="metric-value">${this.performanceData.capacity_metrics.connection_pool_usage}%</div>
            <div class="metric-subtitle">Database connections in use</div>
          </div>
        </div>
      ` : ''}
    `
  }
  
  private renderErrorAnalytics() {
    return html`
      <div class="chart-container">
        <div class="chart-header">
          <h3 class="chart-title">Error Rate & Reliability Tracking</h3>
          <div class="chart-controls">
            <button 
              class="chart-toggle ${this.expandedCharts.has('error-rate') ? 'expanded' : ''}"
              @click=${() => this.toggleChartExpansion('error-rate')}
            >
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/>
              </svg>
            </button>
          </div>
        </div>
        <div class="chart-canvas ${this.expandedCharts.has('error-rate') ? 'expanded' : ''}">
          <canvas id="error-rate-chart"></canvas>
        </div>
      </div>
      
      ${this.renderAlertsSection()}
    `
  }
  
  private renderResourceAnalytics() {
    return html`
      <div class="chart-container">
        <div class="chart-header">
          <h3 class="chart-title">Resource Utilization Monitoring</h3>
          <div class="chart-controls">
            <button 
              class="chart-toggle ${this.expandedCharts.has('resource') ? 'expanded' : ''}"
              @click=${() => this.toggleChartExpansion('resource')}
            >
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/>
              </svg>
            </button>
          </div>
        </div>
        <div class="chart-canvas ${this.expandedCharts.has('resource') ? 'expanded' : ''}">
          <canvas id="resource-chart"></canvas>
        </div>
      </div>
    `
  }
  
  private renderAlertsAnalytics() {
    return this.renderAlertsSection()
  }
  
  private renderRegressionAnalytics() {
    const regressionAlerts = this.performanceData?.alerts?.filter(alert => alert.type === 'regression') || []
    
    return html`
      <div class="alerts-section">
        <div class="alerts-header">
          <h3 class="alerts-title">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z"/>
            </svg>
            Performance Regression Detection
          </h3>
          <div class="alerts-filters">
            <button class="alert-filter-btn active">All</button>
            <button class="alert-filter-btn">Critical</button>
            <button class="alert-filter-btn">Recent</button>
          </div>
        </div>
        
        ${regressionAlerts.length > 0 ? html`
          <div class="alerts-list">
            ${regressionAlerts.map(alert => html`
              <div class="alert-item ${alert.severity}">
                <svg class="alert-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
                </svg>
                <div class="alert-content">
                  <div class="alert-message">${alert.message}</div>
                  <div class="alert-details">
                    ${alert.metric}: ${alert.current_value} (baseline: ${alert.threshold_value})
                  </div>
                  ${alert.impact_assessment ? html`
                    <div class="alert-impact">Impact: ${alert.impact_assessment}</div>
                  ` : ''}
                </div>
                <div class="alert-time">
                  ${new Date(alert.timestamp).toLocaleTimeString()}
                </div>
              </div>
            `)}
          </div>
        ` : html`
          <div class="empty-state">
            <svg class="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z"/>
            </svg>
            <h3>No Performance Regressions</h3>
            <p>System performance is stable with no detected regressions.</p>
          </div>
        `}
      </div>
    `
  }
  
  private renderAlertsSection() {
    if (!this.performanceData?.alerts?.length) {
      return html`
        <div class="empty-state">
          <svg class="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6-4a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <h3>No Performance Issues</h3>
          <p>All systems are operating within normal parameters.</p>
        </div>
      `
    }
    
    return html`
      <div class="alerts-section">
        <div class="alerts-header">
          <h3 class="alerts-title">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-5 5v-5zM4.5 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971L6.167 19.334c-.75.412-1.667-.13-1.667-.986V5.653z"/>
            </svg>
            Performance Alerts
          </h3>
        </div>
        
        <div class="alerts-list">
          ${this.performanceData.alerts.map(alert => html`
            <div class="alert-item ${alert.severity}">
              <svg class="alert-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                ${alert.severity === 'critical' ? html`
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
                ` : alert.severity === 'warning' ? html`
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                ` : html`
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                `}
              </svg>
              <div class="alert-content">
                <div class="alert-message">${alert.message}</div>
                <div class="alert-details">
                  ${alert.metric}: ${alert.current_value} (threshold: ${alert.threshold_value})
                </div>
              </div>
              <div class="alert-time">
                ${new Date(alert.timestamp).toLocaleTimeString()}
              </div>
            </div>
          `)}
        </div>
      </div>
    `
  }
  
  updated(changedProperties: Map<string, any>) {
    super.updated(changedProperties)
    
    if (changedProperties.has('performanceData') && this.performanceData) {
      this.lastUpdate = new Date()
      this.updateCharts()
    }
    
    if (changedProperties.has('timeRange')) {
      this.updateCharts()
    }
  }
  
  render() {
    const alertCount = this.performanceData?.alerts?.length || 0
    const regressionCount = this.performanceData?.alerts?.filter(a => a.type === 'regression').length || 0
    
    return html`
      <div class="analytics-header">
        <div class="header-title">
          <svg class="analytics-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
          </svg>
          Performance Analytics
        </div>
        <div class="header-controls">
          ${this.renderTimeRangeSelector()}
          
          <div class="connection-indicator">
            <div class="status-dot ${this.connectionStatus}"></div>
            ${this.connectionStatus === 'connected' ? 'Live' : 
              this.connectionStatus === 'connecting' ? 'Connecting...' : 'Offline'}
          </div>
          
          <div class="refresh-controls">
            <button 
              class="control-btn ${this.autoRefresh ? 'active' : ''}"
              @click=${this.toggleAutoRefresh}
              title="${this.autoRefresh ? 'Disable' : 'Enable'} auto-refresh"
            >
              ${this.autoRefresh ? '⏸️' : '▶️'}
            </button>
            
            ${this.lastUpdate ? html`
              <span style="font-size: 0.75rem; opacity: 0.9;">
                ${this.lastUpdate.toLocaleTimeString()}
              </span>
            ` : ''}
          </div>
        </div>
      </div>
      
      <div class="analytics-content">
        <div class="performance-tabs">
          <button 
            class="tab-btn ${this.selectedTab === 'overview' ? 'active' : ''}"
            @click=${() => this.handleTabChange('overview')}
          >
            Overview
          </button>
          <button 
            class="tab-btn ${this.selectedTab === 'throughput' ? 'active' : ''}"
            @click=${() => this.handleTabChange('throughput')}
          >
            Throughput
          </button>
          <button 
            class="tab-btn ${this.selectedTab === 'errors' ? 'active' : ''}"
            @click=${() => this.handleTabChange('errors')}
          >
            Error Analysis
          </button>
          <button 
            class="tab-btn ${this.selectedTab === 'resources' ? 'active' : ''}"
            @click=${() => this.handleTabChange('resources')}
          >
            Resources
          </button>
          <button 
            class="tab-btn ${this.selectedTab === 'regression' ? 'active' : ''}"
            @click=${() => this.handleTabChange('regression')}
          >
            Regression
            ${regressionCount > 0 ? html`<span class="tab-badge">${regressionCount}</span>` : ''}
          </button>
          <button 
            class="tab-btn ${this.selectedTab === 'alerts' ? 'active' : ''}"
            @click=${() => this.handleTabChange('alerts')}
          >
            Alerts
            ${alertCount > 0 ? html`<span class="tab-badge">${alertCount}</span>` : ''}
          </button>
        </div>
        
        <div class="analytics-panel">
          ${this.renderCurrentTab()}
        </div>
      </div>
    `
  }
}