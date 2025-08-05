import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../services/websocket'

interface ContextNode {
  id: string
  type: 'agent' | 'task' | 'system' | 'metric' | 'alert' | 'session'
  title: string
  summary: string
  details: any
  children?: ContextNode[]
  metadata: {
    timestamp: string
    priority: 'critical' | 'high' | 'medium' | 'low'
    status: 'active' | 'completed' | 'failed' | 'pending'
    tags: string[]
  }
  metrics?: {
    performance: number
    reliability: number
    efficiency: number
  }
  actions: ContextAction[]
}

interface ContextAction {
  id: string
  label: string
  command: string
  icon: string
  description: string
  requiresConfirmation?: boolean
}

interface NavigationHistory {
  nodeId: string
  title: string
  timestamp: number
}

interface FilterOptions {
  type: string[]
  priority: string[]
  status: string[]
  timeRange: string
  searchQuery: string
}

@customElement('mobile-context-explorer')
export class MobileContextExplorer extends LitElement {
  @property({ type: Boolean }) declare mobile: boolean
  @property({ type: String }) declare initialContext: string

  @state() private declare currentNode: ContextNode | null
  @state() private declare breadcrumbs: NavigationHistory[]
  @state() private declare loading: boolean
  @state() private declare error: string | null
  @state() private declare expandedSections: Set<string>
  @state() private declare activeTab: 'overview' | 'details' | 'metrics' | 'related'
  @state() private declare filterOptions: FilterOptions
  @state() private declare showFilters: boolean
  @state() private declare relatedNodes: ContextNode[]
  @state() private declare chartData: any
  @state() private declare refreshing: boolean

  private websocketService: WebSocketService
  private refreshInterval: number | null = null
  private touchStartY = 0
  private pullToRefreshThreshold = 80
  private pullToRefreshActive = false

  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      overflow: hidden;
    }

    .explorer-container {
      display: flex;
      flex-direction: column;
      height: 100%;
    }

    .explorer-header {
      background: white;
      border-bottom: 1px solid #e5e7eb;
      padding: 0.75rem 1rem;
      flex-shrink: 0;
    }

    .breadcrumb-nav {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      padding: 0.5rem 0;
      margin-bottom: 0.5rem;
    }

    .breadcrumb-item {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      white-space: nowrap;
      font-size: 0.875rem;
      color: #6b7280;
      cursor: pointer;
      padding: 0.25rem 0.5rem;
      border-radius: 6px;
      transition: all 0.2s;
    }

    .breadcrumb-item:hover {
      background: #f3f4f6;
      color: #374151;
    }

    .breadcrumb-item.current {
      color: #3b82f6;
      font-weight: 600;
    }

    .breadcrumb-separator {
      color: #d1d5db;
      font-size: 0.75rem;
    }

    .search-bar {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 0.5rem 1rem;
    }

    .search-input {
      flex: 1;
      border: none;
      background: none;
      outline: none;
      font-size: 1rem;
      color: #374151;
    }

    .search-input::placeholder {
      color: #9ca3af;
    }

    .filter-button {
      background: none;
      border: none;
      color: #6b7280;
      cursor: pointer;
      padding: 0.25rem;
      border-radius: 4px;
      transition: all 0.2s;
    }

    .filter-button:hover {
      background: #e5e7eb;
      color: #374151;
    }

    .filter-button.active {
      color: #3b82f6;
      background: #eff6ff;
    }

    .tab-navigation {
      display: flex;
      background: white;
      border-bottom: 1px solid #e5e7eb;
      flex-shrink: 0;
    }

    .tab-button {
      flex: 1;
      padding: 1rem;
      border: none;
      background: none;
      font-size: 0.875rem;
      font-weight: 500;
      color: #6b7280;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
    }

    .tab-button:hover {
      color: #374151;
      background: #f9fafb;
    }

    .tab-button.active {
      color: #3b82f6;
      border-bottom-color: #3b82f6;
      background: #fefefe;
    }

    .content-area {
      flex: 1;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      padding: 1rem;
    }

    .pull-to-refresh-indicator {
      text-align: center;
      padding: 1rem;
      color: #6b7280;
      font-size: 0.875rem;
      transform: translateY(-100%);
      transition: transform 0.3s;
    }

    .pull-to-refresh-indicator.active {
      transform: translateY(0);
    }

    .context-overview {
      background: white;
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 1rem;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .context-header {
      display: flex;
      align-items: flex-start;
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .context-icon {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.5rem;
      background: #eff6ff;
      color: #3b82f6;
    }

    .context-icon.agent {
      background: #f0f9ff;
      color: #0ea5e9;
    }

    .context-icon.task {
      background: #fef3c7;
      color: #f59e0b;
    }

    .context-icon.system {
      background: #f3e8ff;
      color: #8b5cf6;
    }

    .context-icon.alert {
      background: #fee2e2;
      color: #ef4444;
    }

    .context-meta {
      flex: 1;
      min-width: 0;
    }

    .context-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: #111827;
      margin-bottom: 0.5rem;
      line-height: 1.3;
    }

    .context-summary {
      color: #6b7280;
      line-height: 1.4;
      margin-bottom: 1rem;
    }

    .context-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }

    .context-tag {
      padding: 0.25rem 0.75rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 500;
      background: #f3f4f6;
      color: #374151;
    }

    .context-tag.priority-critical {
      background: #fee2e2;
      color: #dc2626;
    }

    .context-tag.priority-high {
      background: #fef3c7;
      color: #d97706;
    }

    .context-tag.status-active {
      background: #d1fae5;
      color: #065f46;
    }

    .context-tag.status-failed {
      background: #fee2e2;
      color: #dc2626;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
      gap: 1rem;
      margin: 1rem 0;
    }

    .metric-card {
      text-align: center;
      padding: 1rem;
      background: #f9fafb;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
    }

    .metric-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin-bottom: 0.25rem;
    }

    .metric-label {
      font-size: 0.8rem;
      color: #6b7280;
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.05em;
    }

    .expandable-section {
      margin-bottom: 1rem;
    }

    .section-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1rem;
      background: white;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.2s;
      border: 1px solid #e5e7eb;
    }

    .section-header:hover {
      background: #f9fafb;
    }

    .section-title {
      font-weight: 600;
      color: #111827;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .section-toggle {
      color: #6b7280;
      transition: transform 0.2s;
    }

    .section-toggle.expanded {
      transform: rotate(180deg);
    }

    .section-content {
      padding: 1rem;
      background: white;
      border: 1px solid #e5e7eb;
      border-top: none;
      border-radius: 0 0 12px 12px;
    }

    .details-grid {
      display: grid;
      gap: 0.75rem;
    }

    .detail-item {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 1rem;
      padding: 0.75rem 0;
      border-bottom: 1px solid #f3f4f6;
    }

    .detail-item:last-child {
      border-bottom: none;
    }

    .detail-label {
      font-weight: 600;
      color: #374151;
      font-size: 0.875rem;
    }

    .detail-value {
      color: #6b7280;
      font-size: 0.875rem;
      word-break: break-word;
    }

    .json-view {
      background: #1f2937;
      color: #f9fafb;
      padding: 1rem;
      border-radius: 8px;
      font-family: 'SF Mono', Monaco, monospace;
      font-size: 0.8rem;
      overflow-x: auto;
      max-height: 300px;
      overflow-y: auto;
    }

    .related-items {
      display: grid;
      gap: 0.75rem;
    }

    .related-item {
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 1rem;
      background: white;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
      cursor: pointer;
      transition: all 0.2s;
    }

    .related-item:hover {
      background: #f9fafb;
      border-color: #d1d5db;
    }

    .related-item:active {
      transform: translateY(1px);
    }

    .related-icon {
      width: 40px;
      height: 40px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
      background: #f3f4f6;
      color: #6b7280;
    }

    .related-content {
      flex: 1;
      min-width: 0;
    }

    .related-title {
      font-weight: 600;
      color: #111827;
      margin-bottom: 0.25rem;
      font-size: 0.9rem;
    }

    .related-description {
      color: #6b7280;
      font-size: 0.8rem;
      line-height: 1.3;
    }

    .action-bar {
      background: white;
      border-top: 1px solid #e5e7eb;
      padding: 1rem;
      display: flex;
      gap: 0.75rem;
      flex-shrink: 0;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }

    .action-button {
      padding: 0.75rem 1.5rem;
      border-radius: 12px;
      border: none;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.875rem;
    }

    .action-button.primary {
      background: #3b82f6;
      color: white;
    }

    .action-button.primary:hover {
      background: #2563eb;
    }

    .action-button.secondary {
      background: #f3f4f6;
      color: #374151;
    }

    .action-button.secondary:hover {
      background: #e5e7eb;
    }

    .action-button.danger {
      background: #ef4444;
      color: white;
    }

    .action-button.danger:hover {
      background: #dc2626;
    }

    .loading-state {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 3rem;
      color: #6b7280;
    }

    .error-state {
      text-align: center;
      padding: 3rem 1rem;
      color: #dc2626;
    }

    .error-message {
      font-size: 1rem;
      margin-bottom: 1rem;
    }

    .retry-button {
      background: #3b82f6;
      color: white;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }

    .chart-container {
      background: white;
      border-radius: 12px;
      padding: 1rem;
      margin-bottom: 1rem;
      height: 250px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid #e5e7eb;
    }

    .chart-placeholder {
      color: #6b7280;
      text-align: center;
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
      .details-grid {
        grid-template-columns: 1fr;
      }
      
      .detail-item {
        grid-template-columns: 1fr;
        gap: 0.5rem;
      }
      
      .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }

    /* Touch-friendly improvements */
    @media (hover: none) and (pointer: coarse) {
      .tab-button {
        padding: 1.25rem;
      }
      
      .action-button {
        padding: 1rem 1.5rem;
        min-height: 44px;
      }
      
      .related-item {
        padding: 1.25rem;
      }
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      :host {
        background: #111827;
      }
      
      .explorer-header,
      .tab-navigation,
      .context-overview,
      .section-header,
      .section-content,
      .related-item,
      .action-bar {
        background: #1f2937;
        border-color: #374151;
        color: #f9fafb;
      }
      
      .context-title {
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
    this.mobile = true
    this.initialContext = ''
    this.currentNode = null
    this.breadcrumbs = []
    this.loading = false
    this.error = null
    this.expandedSections = new Set()
    this.activeTab = 'overview'
    this.filterOptions = {
      type: [],
      priority: [],
      status: [],
      timeRange: 'all',
      searchQuery: ''
    }
    this.showFilters = false
    this.relatedNodes = []
    this.chartData = null
    this.refreshing = false
    this.websocketService = WebSocketService.getInstance()
  }

  connectedCallback() {
    super.connectedCallback()
    this.setupTouchHandlers()
    this.setupWebSocketListeners()
    if (this.initialContext) {
      this.navigateToContext(this.initialContext)
    } else {
      this.loadRootContext()
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval)
    }
  }

  private setupTouchHandlers() {
    const contentArea = this.shadowRoot?.querySelector('.content-area')
    if (!contentArea) return

    contentArea.addEventListener('touchstart', (e) => {
      if (contentArea.scrollTop === 0) {
        this.touchStartY = e.touches[0].clientY
      }
    }, { passive: true })

    contentArea.addEventListener('touchmove', (e) => {
      if (contentArea.scrollTop === 0 && !this.refreshing) {
        const touchY = e.touches[0].clientY
        const pullDistance = touchY - this.touchStartY
        
        if (pullDistance > 0) {
          e.preventDefault()
          
          if (pullDistance > this.pullToRefreshThreshold) {
            this.pullToRefreshActive = true
          }
          
          const indicator = this.shadowRoot?.querySelector('.pull-to-refresh-indicator')
          if (indicator) {
            (indicator as HTMLElement).style.transform = `translateY(${Math.min(pullDistance - 100, 0)}px)`
          }
        }
      }
    })

    contentArea.addEventListener('touchend', () => {
      if (this.pullToRefreshActive && !this.refreshing) {
        this.handleRefresh()
      }
      
      this.pullToRefreshActive = false
      const indicator = this.shadowRoot?.querySelector('.pull-to-refresh-indicator')
      if (indicator) {
        (indicator as HTMLElement).style.transform = 'translateY(-100%)'
      }
    }, { passive: true })
  }

  private setupWebSocketListeners() {
    this.websocketService.on('context-update', (data) => {
      if (this.currentNode && data.nodeId === this.currentNode.id) {
        this.handleContextUpdate(data)
      }
    })

    this.websocketService.on('related-context-change', (data) => {
      this.loadRelatedNodes()
    })
  }

  private async loadRootContext() {
    this.loading = true
    this.error = null
    
    try {
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:context-root --mobile'
        })
      })
      
      const result = await response.json()
      if (result.success && result.result.context) {
        this.currentNode = result.result.context
        this.breadcrumbs = [{ 
          nodeId: this.currentNode.id, 
          title: this.currentNode.title, 
          timestamp: Date.now() 
        }]
        this.loadRelatedNodes()
      } else {
        this.generateMockRootContext()
      }
    } catch (error) {
      console.error('Failed to load root context:', error)
      this.generateMockRootContext()
    } finally {
      this.loading = false
    }
  }

  private generateMockRootContext() {
    this.currentNode = {
      id: 'root',
      type: 'system',
      title: 'LeanVibe Agent Hive System',
      summary: 'Multi-agent autonomous development platform with 5 active agents coordinating development tasks.',
      details: {
        platform_version: '2.0.0',
        uptime: '2h 34m',
        total_agents: 5,
        active_sessions: 3,
        completed_tasks: 12,
        current_project: 'Mobile PWA Enhancement',
        resource_usage: {
          cpu: '24%',
          memory: '1.2GB',
          storage: '45GB'
        }
      },
      metadata: {
        timestamp: new Date().toISOString(),
        priority: 'high',
        status: 'active',
        tags: ['system', 'autonomous', 'development', 'mobile']
      },
      metrics: {
        performance: 87,
        reliability: 94,
        efficiency: 78
      },
      actions: [
        {
          id: 'overview',
          label: 'System Overview',
          command: '/hive:overview --detailed',
          icon: '📊',
          description: 'View complete system status and metrics'
        },
        {
          id: 'agents',
          label: 'Manage Agents',
          command: '/hive:agents --manage',
          icon: '🤖',
          description: 'View and control active agents'
        },
        {
          id: 'tasks',
          label: 'Task Queue',
          command: '/hive:tasks --queue',
          icon: '📋',
          description: 'View active and pending tasks'
        }
      ]
    }
    
    this.breadcrumbs = [{ 
      nodeId: this.currentNode.id, 
      title: this.currentNode.title, 
      timestamp: Date.now() 
    }]
    
    this.generateMockRelatedNodes()
  }

  private generateMockRelatedNodes() {
    this.relatedNodes = [
      {
        id: 'agent-frontend',
        type: 'agent',
        title: 'Frontend Developer Agent',
        summary: 'Implementing mobile gesture interface components',
        details: {},
        metadata: {
          timestamp: new Date().toISOString(),
          priority: 'high',
          status: 'active',
          tags: ['frontend', 'mobile', 'gestures']
        },
        actions: []
      },
      {
        id: 'task-notifications',
        type: 'task',
        title: 'Push Notification System',
        summary: 'Building critical developer alerts with smart filtering',
        details: {},
        metadata: {
          timestamp: new Date().toISOString(),
          priority: 'high',
          status: 'active',
          tags: ['notifications', 'mobile', 'alerts']
        },
        actions: []
      },
      {
        id: 'metric-performance',
        type: 'metric',
        title: 'Performance Dashboard',
        summary: 'Real-time system performance and agent utilization metrics',
        details: {},
        metadata: {
          timestamp: new Date().toISOString(),
          priority: 'medium',
          status: 'active',
          tags: ['performance', 'metrics', 'monitoring']
        },
        actions: []
      }
    ]
  }

  private async navigateToContext(nodeId: string) {
    if (this.currentNode?.id === nodeId) return
    
    this.loading = true
    this.error = null
    
    try {
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: `/hive:context-drill --node-id=${nodeId} --mobile`
        })
      })
      
      const result = await response.json()
      if (result.success && result.result.context) {
        const newNode = result.result.context
        this.currentNode = newNode
        
        // Update breadcrumbs
        if (!this.breadcrumbs.find(b => b.nodeId === nodeId)) {
          this.breadcrumbs.push({ 
            nodeId, 
            title: newNode.title, 
            timestamp: Date.now() 
          })
        }
        
        this.loadRelatedNodes()
        this.loadChartData()
      } else {
        this.error = 'Failed to load context details'
      }
    } catch (error) {
      this.error = `Navigation failed: ${error}`
    } finally {
      this.loading = false
    }
  }

  private async loadRelatedNodes() {
    if (!this.currentNode) return
    
    try {
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: `/hive:context-related --node-id=${this.currentNode.id} --mobile`
        })
      })
      
      const result = await response.json()
      if (result.success && result.result.related) {
        this.relatedNodes = result.result.related
      }
    } catch (error) {
      console.warn('Failed to load related nodes:', error)
    }
  }

  private async loadChartData() {
    if (!this.currentNode || this.activeTab !== 'metrics') return
    
    try {
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: `/hive:context-metrics --node-id=${this.currentNode.id} --chart-data`
        })
      })
      
      const result = await response.json()
      if (result.success && result.result.chart_data) {
        this.chartData = result.result.chart_data
      }
    } catch (error) {
      console.warn('Failed to load chart data:', error)
    }
  }

  private handleContextUpdate(data: any) {
    if (this.currentNode) {
      this.currentNode = { ...this.currentNode, ...data.updates }
      this.requestUpdate()
    }
  }

  private async handleRefresh() {
    this.refreshing = true
    
    try {
      if (this.currentNode) {
        await this.navigateToContext(this.currentNode.id)
      } else {
        await this.loadRootContext()
      }
    } finally {
      this.refreshing = false
    }
  }

  private navigateToBreadcrumb(index: number) {
    if (index === this.breadcrumbs.length - 1) return // Already at this location
    
    const targetBreadcrumb = this.breadcrumbs[index]
    this.breadcrumbs = this.breadcrumbs.slice(0, index + 1)
    this.navigateToContext(targetBreadcrumb.nodeId)
  }

  private toggleSection(sectionId: string) {
    if (this.expandedSections.has(sectionId)) {
      this.expandedSections.delete(sectionId)
    } else {
      this.expandedSections.add(sectionId)
    }
    this.requestUpdate()
  }

  private switchTab(tab: typeof this.activeTab) {
    this.activeTab = tab
    if (tab === 'metrics') {
      this.loadChartData()
    }
  }

  private async executeAction(action: ContextAction) {
    if (action.requiresConfirmation) {
      const confirmed = confirm(`Execute action: ${action.label}?`)
      if (!confirmed) return
    }
    
    try {
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: action.command,
          context: { nodeId: this.currentNode?.id }
        })
      })
      
      const result = await response.json()
      if (result.success) {
        this.dispatchEvent(new CustomEvent('action-executed', {
          detail: { action, result },
          bubbles: true,
          composed: true
        }))
        
        // Refresh current context
        if (this.currentNode) {
          this.navigateToContext(this.currentNode.id)
        }
      } else {
        this.error = `Action failed: ${action.label}`
      }
    } catch (error) {
      this.error = `Action execution error: ${error}`
    }
  }

  private getContextIcon(type: ContextNode['type']): string {
    const icons = {
      agent: '🤖',
      task: '📋',
      system: '⚙️',
      metric: '📊',
      alert: '⚠️',
      session: '🔄'
    }
    return icons[type] || '📎'
  }

  private formatDetailValue(value: any): string {
    if (typeof value === 'object') {
      return JSON.stringify(value, null, 2)
    }
    return String(value)
  }

  render() {
    if (this.loading && !this.currentNode) {
      return html`
        <div class="loading-state">
          <div>Loading context...</div>
        </div>
      `
    }

    if (this.error && !this.currentNode) {
      return html`
        <div class="error-state">
          <div class="error-message">${this.error}</div>
          <button class="retry-button" @click=${this.loadRootContext}>
            Retry
          </button>
        </div>
      `
    }

    if (!this.currentNode) return html``

    return html`
      <div class="explorer-container">
        ${this.renderHeader()}
        ${this.renderTabNavigation()}
        ${this.renderContent()}
        ${this.renderActionBar()}
      </div>
    `
  }

  private renderHeader() {
    return html`
      <div class="explorer-header">
        <div class="breadcrumb-nav">
          ${this.breadcrumbs.map((breadcrumb, index) => html`
            <div class="breadcrumb-item ${index === this.breadcrumbs.length - 1 ? 'current' : ''}" 
                 @click=${() => this.navigateToBreadcrumb(index)}>
              ${breadcrumb.title}
            </div>
            ${index < this.breadcrumbs.length - 1 ? html`
              <div class="breadcrumb-separator">›</div>
            ` : ''}
          `)}
        </div>
        
        <div class="search-bar">
          <input type="text" class="search-input" placeholder="Search context..."
                 .value=${this.filterOptions.searchQuery}
                 @input=${(e: InputEvent) => {
                   this.filterOptions.searchQuery = (e.target as HTMLInputElement).value
                 }}>
          <button class="filter-button ${this.showFilters ? 'active' : ''}" 
                  @click=${() => { this.showFilters = !this.showFilters }}>
            🔍
          </button>
        </div>
      </div>
    `
  }

  private renderTabNavigation() {
    return html`
      <div class="tab-navigation">
        <button class="tab-button ${this.activeTab === 'overview' ? 'active' : ''}" 
                @click=${() => this.switchTab('overview')}>
          Overview
        </button>
        <button class="tab-button ${this.activeTab === 'details' ? 'active' : ''}" 
                @click=${() => this.switchTab('details')}>
          Details
        </button>
        <button class="tab-button ${this.activeTab === 'metrics' ? 'active' : ''}" 
                @click=${() => this.switchTab('metrics')}>
          Metrics
        </button>
        <button class="tab-button ${this.activeTab === 'related' ? 'active' : ''}" 
                @click=${() => this.switchTab('related')}>
          Related
        </button>
      </div>
    `
  }

  private renderContent() {
    return html`
      <div class="content-area">
        <div class="pull-to-refresh-indicator ${this.pullToRefreshActive ? 'active' : ''}">
          ${this.refreshing ? 'Refreshing...' : 'Pull to refresh'}
        </div>
        
        ${this.activeTab === 'overview' ? this.renderOverviewTab() :
          this.activeTab === 'details' ? this.renderDetailsTab() :
          this.activeTab === 'metrics' ? this.renderMetricsTab() :
          this.renderRelatedTab()}
      </div>
    `
  }

  private renderOverviewTab() {
    if (!this.currentNode) return html``

    return html`
      <div class="context-overview">
        <div class="context-header">
          <div class="context-icon ${this.currentNode.type}">
            ${this.getContextIcon(this.currentNode.type)}
          </div>
          <div class="context-meta">
            <div class="context-title">${this.currentNode.title}</div>
            <div class="context-summary">${this.currentNode.summary}</div>
          </div>
        </div>
        
        <div class="context-tags">
          ${this.currentNode.metadata.tags.map(tag => html`
            <span class="context-tag">${tag}</span>
          `)}
          <span class="context-tag priority-${this.currentNode.metadata.priority}">
            ${this.currentNode.metadata.priority}
          </span>
          <span class="context-tag status-${this.currentNode.metadata.status}">
            ${this.currentNode.metadata.status}
          </span>
        </div>
        
        ${this.currentNode.metrics ? html`
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-value">${this.currentNode.metrics.performance}%</div>
              <div class="metric-label">Performance</div>
            </div>
            <div class="metric-card">
              <div class="metric-value">${this.currentNode.metrics.reliability}%</div>
              <div class="metric-label">Reliability</div>
            </div>
            <div class="metric-card">
              <div class="metric-value">${this.currentNode.metrics.efficiency}%</div>
              <div class="metric-label">Efficiency</div>
            </div>
          </div>
        ` : ''}
      </div>
    `
  }

  private renderDetailsTab() {
    if (!this.currentNode) return html``

    return html`
      <div class="expandable-section">
        <div class="section-header" @click=${() => this.toggleSection('details')}>
          <div class="section-title">
            📎 Context Details
          </div>
          <div class="section-toggle ${this.expandedSections.has('details') ? 'expanded' : ''}">
            ▼
          </div>
        </div>
        ${this.expandedSections.has('details') ? html`
          <div class="section-content">
            <div class="details-grid">
              ${Object.entries(this.currentNode.details).map(([key, value]) => html`
                <div class="detail-item">
                  <div class="detail-label">${key}</div>
                  <div class="detail-value">
                    ${typeof value === 'object' ? html`
                      <div class="json-view">${JSON.stringify(value, null, 2)}</div>
                    ` : this.formatDetailValue(value)}
                  </div>
                </div>
              `)}
            </div>
          </div>
        ` : ''}
      </div>
      
      <div class="expandable-section">
        <div class="section-header" @click=${() => this.toggleSection('metadata')}>
          <div class="section-title">
            🏷️ Metadata
          </div>
          <div class="section-toggle ${this.expandedSections.has('metadata') ? 'expanded' : ''}">
            ▼
          </div>
        </div>
        ${this.expandedSections.has('metadata') ? html`
          <div class="section-content">
            <div class="details-grid">
              <div class="detail-item">
                <div class="detail-label">Type</div>
                <div class="detail-value">${this.currentNode.metadata.timestamp}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">Priority</div>
                <div class="detail-value">${this.currentNode.metadata.priority}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">Status</div>
                <div class="detail-value">${this.currentNode.metadata.status}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">Tags</div>
                <div class="detail-value">${this.currentNode.metadata.tags.join(', ')}</div>
              </div>
            </div>
          </div>
        ` : ''}
      </div>
    `
  }

  private renderMetricsTab() {
    return html`
      ${this.currentNode?.metrics ? html`
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-value">${this.currentNode.metrics.performance}%</div>
            <div class="metric-label">Performance</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${this.currentNode.metrics.reliability}%</div>
            <div class="metric-label">Reliability</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${this.currentNode.metrics.efficiency}%</div>
            <div class="metric-label">Efficiency</div>
          </div>
        </div>
      ` : ''}
      
      <div class="chart-container">
        ${this.chartData ? html`
          <div>Chart visualization would render here</div>
        ` : html`
          <div class="chart-placeholder">
            <div>📊</div>
            <div>Chart data loading...</div>
          </div>
        `}
      </div>
    `
  }

  private renderRelatedTab() {
    return html`
      <div class="related-items">
        ${this.relatedNodes.map(node => html`
          <div class="related-item" @click=${() => this.navigateToContext(node.id)}>
            <div class="related-icon">
              ${this.getContextIcon(node.type)}
            </div>
            <div class="related-content">
              <div class="related-title">${node.title}</div>
              <div class="related-description">${node.summary}</div>
            </div>
          </div>
        `)}
        
        ${this.relatedNodes.length === 0 ? html`
          <div class="chart-placeholder">
            <div>🔗</div>
            <div>No related contexts found</div>
          </div>
        ` : ''}
      </div>
    `
  }

  private renderActionBar() {
    if (!this.currentNode?.actions?.length) return html``

    return html`
      <div class="action-bar">
        ${this.currentNode.actions.map(action => html`
          <button class="action-button primary" @click=${() => this.executeAction(action)}>
            <span>${action.icon}</span>
            <span>${action.label}</span>
          </button>
        `)}
      </div>
    `
  }
}
