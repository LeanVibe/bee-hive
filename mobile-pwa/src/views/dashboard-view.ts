import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { Task, TaskStatus, TaskMoveEvent } from '../types/task'
import { AgentStatus } from '../components/dashboard/agent-health-panel'
import { TimelineEvent } from '../components/dashboard/event-timeline'
import { WebSocketService } from '../services/websocket'
import { OfflineService } from '../services/offline'
import { NotificationService } from '../services/notification'
import { backendAdapter } from '../services/backend-adapter'
import type {
  SystemHealth,
  Agent,
  SystemEvent,
  PerformanceSnapshot,
  HealthSummary
} from '../services'
import '../components/kanban/kanban-board'
import '../components/dashboard/agent-health-panel'
import '../components/dashboard/event-timeline'
import '../components/dashboard/coordination-success-panel'
import '../components/dashboard/realtime-agent-status-panel'
import '../components/dashboard/task-distribution-panel'
import '../components/dashboard/recovery-controls-panel'
import '../components/dashboard/communication-monitoring-panel'
import '../components/dashboard/connection-monitor'
import '../components/autonomous-development/multi-agent-oversight-dashboard'
import '../components/autonomous-development/remote-control-center'
import '../components/common/enhanced-loading-spinner'
import '../components/context-compression/CompressionWidget'
import '../components/context-compression/CompressionDashboard'
import '../components/dashboard/technical-debt-panel'
import TechnicalDebtService from '../services/technical-debt-service'
import type { ProjectDebtStatus } from '../components/dashboard/technical-debt-panel'

@customElement('dashboard-view')
export class DashboardView extends LitElement {
  @property({ type: Boolean }) declare offline: boolean
  
  @state() private declare tasks: Task[]
  @state() private declare agents: AgentStatus[]
  @state() private declare events: TimelineEvent[]
  @state() private declare systemHealth: SystemHealth | null
  @state() private declare performanceMetrics: PerformanceSnapshot | null
  @state() private declare healthSummary: HealthSummary | null
  @state() private declare isLoading: boolean
  @state() private declare error: string
  @state() private declare lastSync: Date | null
  @state() private declare selectedView: 'overview' | 'kanban' | 'agents' | 'events' | 'performance' | 'security' | 'oversight' | 'control' | 'coordination' | 'tools' | 'debt'
  @state() private declare servicesInitialized: boolean
  @state() private declare wsConnected: boolean
  @state() private declare realtimeEnabled: boolean
  @state() private declare connectionQuality: 'excellent' | 'good' | 'poor' | 'offline'
  @state() private declare updateQueue: any[]
  @state() private declare lastUpdateTimestamp: Date | null
  @state() private declare comprehensivePerformanceMetrics: any | null
  @state() private declare securityMetrics: any | null
  @state() private declare securityAlerts: any[]
  @state() private declare technicalDebtProjects: ProjectDebtStatus[]
  @state() private declare debtAnalysisInProgress: boolean
  
  private websocketService: WebSocketService
  private offlineService: OfflineService
  private notificationService: NotificationService
  private technicalDebtService: TechnicalDebtService
  
  // Backend adapter for real data integration
  private backendAdapter = backendAdapter
  
  private monitoringActive: boolean = false
  private stopRealtimeUpdates: (() => void) | null = null
  
  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: #f9fafb;
    }
    
    .dashboard-header {
      background: white;
      border-bottom: 1px solid #e5e7eb;
      padding: 1rem;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    
    .view-tabs {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    
    .tab-button {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      color: #6b7280;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
      font-size: 0.875rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 0.375rem;
      min-height: 44px;
      position: relative;
    }
    
    .tab-button:focus {
      outline: none;
      box-shadow: 0 0 0 2px #3b82f6, 0 0 0 4px rgba(59, 130, 246, 0.2);
      z-index: 1;
    }
    
    .tab-button:focus-visible {
      outline: 2px solid #3b82f6;
      outline-offset: 2px;
    }
    
    .tab-button:hover {
      background: #f3f4f6;
      border-color: #d1d5db;
    }
    
    .tab-button.active {
      background: #3b82f6;
      border-color: #3b82f6;
      color: white;
    }
    
    .overview-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
    }
    
    .overview-summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
      padding: 1rem;
      background: white;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
      margin-bottom: 1rem;
    }
    
    .summary-card {
      text-align: center;
      padding: 0.75rem;
    }
    
    /* Enhanced metric cards with visual indicators */
    .enhanced-metric-card {
      position: relative;
      overflow: hidden;
    }
    
    .metric-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.5rem;
    }
    
    .metric-trend {
      opacity: 0.7;
      transition: all 0.2s ease;
    }
    
    .trend-healthy {
      color: #10b981;
    }
    
    .trend-warning {
      color: #f59e0b;
    }
    
    .trend-critical {
      color: #ef4444;
      animation: pulse 2s infinite;
    }
    
    .metric-bar {
      height: 4px;
      background: rgba(148, 163, 184, 0.2);
      border-radius: 2px;
      margin-top: 0.5rem;
      overflow: hidden;
    }
    
    .metric-fill {
      height: 100%;
      transition: width 0.6s ease-in-out;
      border-radius: 2px;
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 0.7; }
      50% { opacity: 1; }
    }
    
    .summary-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin-bottom: 0.25rem;
    }
    
    .summary-label {
      font-size: 0.875rem;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }
    
    .overview-panels {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
      padding: 0 1rem;
    }
    
    .panel-section {
      height: 400px;
    }
    
    .panel-section.full-height {
      height: calc(100vh - 300px);
    }
    
    @media (min-width: 768px) {
      .overview-grid {
        grid-template-columns: 2fr 1fr;
      }
      
      .overview-panels {
        grid-template-columns: 1fr 1fr;
      }
      
      .panel-section {
        height: 350px;
      }
    }
    
    @media (min-width: 1024px) {
      .overview-panels {
        grid-template-columns: 2fr 1fr;
      }
      
      .panel-section {
        height: 400px;
      }
    }
    
    .header-content {
      display: flex;
      align-items: center;
      justify-content: space-between;
      max-width: 1200px;
      margin: 0 auto;
    }
    
    .page-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .sync-status {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.875rem;
      color: #6b7280;
    }
    
    .sync-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
    }
    
    .sync-indicator.offline {
      background: #f59e0b;
      animation: pulse 2s infinite;
    }
    
    .sync-indicator.error {
      background: #ef4444;
    }
    
    .dashboard-content {
      height: calc(100vh - 120px);
      max-width: 1200px;
      margin: 0 auto;
      padding: 0;
    }
    
    .loading-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 50vh;
      gap: 1rem;
    }
    
    .spinner {
      width: 40px;
      height: 40px;
      border: 4px solid #e5e7eb;
      border-top: 4px solid #3b82f6;
      border-radius: 50%;
      animation: spin 1s linear infinite;
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
    
    .error-actions {
      margin-top: 1rem;
      display: flex;
      gap: 0.5rem;
      justify-content: center;
    }
    
    .btn {
      padding: 0.5rem 1rem;
      border: none;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .btn-primary {
      background: #3b82f6;
      color: white;
    }
    
    .btn-primary:hover {
      background: #2563eb;
    }
    
    .refresh-button {
      background: none;
      border: 1px solid #d1d5db;
      color: #374151;
      padding: 0.5rem;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
    
    .refresh-button:hover {
      background: #f9fafb;
      border-color: #9ca3af;
    }
    
    .refresh-icon {
      width: 16px;
      height: 16px;
    }
    
    .refresh-button.spinning .refresh-icon {
      animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    @media (max-width: 768px) {
      .dashboard-header {
        padding: 0.75rem;
      }
      
      .page-title {
        font-size: 1.25rem;
      }
      
      .sync-status {
        font-size: 0.8125rem;
      }
      
      .dashboard-content {
        height: calc(100vh - 100px);
      }
    }
    
    /* Accessibility utilities */
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    
    .skip-link {
      position: absolute;
      top: -40px;
      left: 6px;
      background: #000;
      color: #fff;
      padding: 8px;
      text-decoration: none;
      border-radius: 4px;
      z-index: 9999;
    }
    
    .skip-link:focus {
      top: 6px;
    }
    
    /* Real-time update indicators */
    .realtime-indicator {
      position: relative;
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
    }
    
    .realtime-pulse {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      opacity: 0;
      transition: opacity 0.2s ease;
    }
    
    .realtime-indicator.active .realtime-pulse {
      opacity: 1;
      animation: realtimePulse 2s ease-in-out;
    }
    
    .connection-status {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.75rem;
      color: #6b7280;
    }
    
    .connection-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      transition: all 0.2s ease;
    }
    
    .connection-dot.excellent {
      background: #10b981;
      box-shadow: 0 0 4px rgba(16, 185, 129, 0.5);
    }
    
    .connection-dot.good {
      background: #f59e0b;
      box-shadow: 0 0 4px rgba(245, 158, 11, 0.5);
    }
    
    .connection-dot.poor {
      background: #ef4444;
      box-shadow: 0 0 4px rgba(239, 68, 68, 0.5);
    }
    
    .connection-dot.offline {
      background: #6b7280;
    }
    
    .update-queue-indicator {
      position: absolute;
      top: 0.25rem;
      right: 0.25rem;
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      color: white;
      border-radius: 50%;
      width: 18px;
      height: 18px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.625rem;
      font-weight: 600;
      opacity: 0;
      transform: scale(0.8);
      transition: all 0.2s ease;
    }
    
    .update-queue-indicator.visible {
      opacity: 1;
      transform: scale(1);
    }
    
    .realtime-controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-left: 1rem;
    }
    
    .realtime-toggle {
      background: none;
      border: 1px solid #d1d5db;
      color: #374151;
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .realtime-toggle.active {
      background: #10b981;
      border-color: #10b981;
      color: white;
    }
    
    .realtime-toggle:hover {
      border-color: #9ca3af;
    }
    
    @keyframes realtimePulse {
      0%, 100% {
        transform: scale(1);
        opacity: 1;
      }
      50% {
        transform: scale(1.2);
        opacity: 0.7;
      }
    }
    
    /* High contrast mode enhancements */
    @media (prefers-contrast: high) {
      .tab-button {
        border-width: 2px;
      }
      
      .tab-button:focus {
        box-shadow: 0 0 0 3px #000, 0 0 0 6px #fff;
      }
      
      .tab-button.active {
        border-color: #000;
        background: #000;
        color: #fff;
      }
      
      .connection-dot.excellent,
      .connection-dot.good,
      .connection-dot.poor {
        box-shadow: none;
        border: 2px solid #000;
      }
    }
  `
  
  constructor() {
    super()
    
    // Initialize reactive properties
    this.offline = false
    this.tasks = []
    this.agents = []
    this.events = []
    this.systemHealth = null
    this.performanceMetrics = null
    this.healthSummary = null
    this.isLoading = true
    this.error = ''
    this.lastSync = null
    this.selectedView = 'overview'
    this.servicesInitialized = false
    this.wsConnected = false
    this.realtimeEnabled = true
    this.connectionQuality = 'offline'
    this.updateQueue = []
    this.lastUpdateTimestamp = null
    this.comprehensivePerformanceMetrics = null
    this.securityMetrics = null
    this.securityAlerts = []
    this.technicalDebtProjects = []
    this.debtAnalysisInProgress = false
    
    this.websocketService = WebSocketService.getInstance()
    this.offlineService = OfflineService.getInstance()
    this.notificationService = NotificationService.getInstance()
    this.technicalDebtService = TechnicalDebtService.getInstance()
    
    // Backend adapter handles all data integration
    
    // Listen for offline state changes
    this.offlineService.on('online', () => {
      this.offline = false
      this.syncTasks()
    })
    
    this.offlineService.on('offline', () => {
      this.offline = true
    })
    
    // Enhanced WebSocket event handlers with real-time updates
    this.websocketService.on('connected', () => {
      this.wsConnected = true
      this.connectionQuality = 'excellent'
      console.log('✅ WebSocket connected - Real-time updates enabled')
      this.makeLiveAnnouncement('Real-time updates connected')
    })
    
    this.websocketService.on('disconnected', () => {
      this.wsConnected = false
      this.connectionQuality = 'offline'
      console.log('❌ WebSocket disconnected - Falling back to polling')
      this.makeLiveAnnouncement('Real-time updates disconnected')
    })
    
    this.websocketService.on('connection-quality', (event: any) => {
      this.connectionQuality = event.detail.quality || 'good'
    })
    
    this.websocketService.on('task-updated', (event: any) => {
      this.handleRealtimeTaskUpdate(event.detail || event)
    })
    
    this.websocketService.on('task-created', (event: any) => {
      this.handleRealtimeTaskCreated(event.detail || event)
    })
    
    this.websocketService.on('task-deleted', (event: any) => {
      this.handleRealtimeTaskDeleted(event.detail || event)
    })
    
    this.websocketService.on('agent-status-changed', (event: any) => {
      this.handleRealtimeAgentUpdate(event.detail || event)
    })
    
    this.websocketService.on('system-event', (event: any) => {
      this.handleRealtimeSystemEvent(event.detail || event)
    })
    
    this.websocketService.on('metrics-updated', (event: any) => {
      this.handleRealtimeMetricsUpdate(event.detail || event)
    })
  }
  
  async connectedCallback() {
    super.connectedCallback()
    await this.initializeIntegratedServices()
    await this.loadAllData()
    
    // Connect WebSocket if online
    if (!this.offline) {
      await this.websocketService.connect()
    }
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    this.websocketService.disconnect()
    
    // Stop real-time updates from backend adapter
    if (this.stopRealtimeUpdates) {
      this.stopRealtimeUpdates()
      this.stopRealtimeUpdates = null
    }
  }
  
  /**
   * Initialize backend adapter with event listeners
   */
  private async initializeIntegratedServices() {
    if (this.servicesInitialized) return
    
    try {
      // Set up event listeners for real-time updates from backend adapter
      this.backendAdapter.on('liveDataUpdated', this.handleLiveDataUpdated.bind(this))
      this.backendAdapter.on('error', this.handleBackendError.bind(this))
      
      // Start real-time monitoring if online
      if (!this.offline) {
        this.stopRealtimeUpdates = this.backendAdapter.startRealtimeUpdates()
      }
      
      this.servicesInitialized = true
      console.log('✅ Backend adapter initialized successfully')
      
    } catch (error) {
      console.error('❌ Failed to initialize backend adapter:', error)
      this.error = 'Failed to connect to backend services'
    }
  }
  
  // Monitoring methods removed - now handled by backend adapter
  
  /**
   * Load all dashboard data using integrated services
   */
  private async loadAllData() {
    this.isLoading = true
    this.error = ''
    
    try {
      // Try to load from cache first
      const cachedTasks = await this.offlineService.getTasks()
      if (cachedTasks.length > 0) {
        // Convert OfflineTask format back to UI format
        this.tasks = cachedTasks.map(task => ({
          id: task.id,
          title: task.title,
          status: task.status,
          priority: task.priority,
          createdAt: new Date(task.created_at).toISOString(),
          updatedAt: new Date(task.updated_at).toISOString(),
          syncStatus: task.synced ? 'synced' : 'pending'
        }))
        this.isLoading = false
      }
      
      // If online, fetch fresh data using integrated services
      if (!this.offline) {
        await this.syncAllIntegratedData()
        
        // Enable mobile dashboard optimizations
        this.websocketService.enableMobileDashboardMode()
      }
      
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
      this.error = error instanceof Error ? error.message : 'Failed to load dashboard data'
    } finally {
      this.isLoading = false
    }
  }
  
  /**
   * Sync all data using backend adapter
   */
  private async syncAllIntegratedData() {
    try {
      console.log('🔄 Syncing data from backend adapter...')
      
      // Get all data from backend adapter with fallback to mock data
      const [taskData, agentData, systemHealthData, eventData, metricsData] = await Promise.allSettled([
        this.backendAdapter.getTasksFromLiveData(),
        this.backendAdapter.getAgentsFromLiveData(),
        this.backendAdapter.getSystemHealthFromLiveData(),
        this.backendAdapter.getEventsFromLiveData(100),
        this.backendAdapter.getPerformanceMetricsFromLiveData()
      ])
      
      // Extract values or use mock data for failed requests
      const tasks = taskData.status === 'fulfilled' ? taskData.value : this.getMockTasks()
      const agents = agentData.status === 'fulfilled' ? agentData.value : this.getMockAgents()
      const systemHealth = systemHealthData.status === 'fulfilled' ? systemHealthData.value : this.getMockSystemHealth()
      const events = eventData.status === 'fulfilled' ? eventData.value : this.getMockEvents()
      const metrics = metricsData.status === 'fulfilled' ? metricsData.value : this.getMockMetrics()
      
      // Update system health
      this.systemHealth = systemHealth
      this.healthSummary = {
        overall: systemHealth.overall,
        components: systemHealth.components
      }
      
      // Transform agent data to UI format
      this.agents = agents.map(this.transformAgentToUIFormat)
      
      // Update tasks - convert to OfflineTask format for saving
      this.tasks = tasks
      const offlineTasks = tasks.map(task => ({
        id: task.id,
        title: task.title,
        status: task.status,
        priority: task.priority || 'medium',
        created_at: Date.now(),
        updated_at: Date.now(),
        synced: true
      }))
      await this.offlineService.saveTasks(offlineTasks)
      
      // Transform events to UI format  
      this.events = events.map(this.transformEventToUIFormat)
      
      // Update performance metrics
      this.performanceMetrics = metrics
      
      this.lastSync = new Date()
      this.error = ''
      
      console.log('✅ All data synced successfully from backend adapter')
      
    } catch (error) {
      console.error('❌ Failed to sync data from backend adapter:', error)
      this.error = error instanceof Error ? error.message : 'Failed to sync data'
    }
  }
  
  /**
   * Transform Agent API data to UI AgentStatus format
   */
  private transformAgentToUIFormat = (agent: Agent): AgentStatus => {
    return {
      id: agent.id,
      name: agent.name || `Agent ${agent.id}`,
      status: agent.status === 'active' ? 'active' : 
             agent.status === 'idle' ? 'idle' : 
             agent.status === 'error' ? 'error' : 'offline',
      uptime: agent.performance_metrics?.uptime || 0,
      lastSeen: agent.last_seen || new Date().toISOString(),
      currentTask: agent.current_task_id || undefined,
      metrics: {
        cpuUsage: agent.performance_metrics?.cpu_usage || [],
        memoryUsage: agent.performance_metrics?.memory_usage || [],
        tokenUsage: agent.performance_metrics?.token_usage || [],
        tasksCompleted: agent.performance_metrics?.tasks_completed || [],
        errorRate: agent.performance_metrics?.error_rate || [],
        responseTime: agent.performance_metrics?.response_time || [],
        timestamps: agent.performance_metrics?.timestamps || []
      },
      performance: {
        score: agent.performance_metrics?.overall_score || 85,
        trend: agent.performance_metrics?.trend || 'stable'
      }
    }
  }
  
  /**
   * Transform SystemEvent API data to UI TimelineEvent format
   */
  private transformEventToUIFormat = (event: SystemEvent): TimelineEvent => {
    return {
      id: event.id,
      type: event.event_type,
      title: event.summary || event.event_type,
      description: event.description || '',
      agent: event.agent_id,
      timestamp: event.created_at,
      metadata: event.metadata || {},
      severity: event.severity === 'high' ? 'error' : 
               event.severity === 'medium' ? 'warning' : 'info'
    }
  }
  
  /**
   * Real-time event handlers for backend adapter
   */
  private handleLiveDataUpdated = async (event: CustomEvent) => {
    console.log('🔄 Live data updated from backend')
    // Refresh all data when backend sends updates
    await this.syncAllIntegratedData()
  }
  
  private handleBackendError = (event: CustomEvent) => {
    console.error('❌ Backend adapter error:', event.detail)
    this.error = 'Connection to backend lost - using cached data'
  }
  
  private async syncAllData() {
    // Legacy method - now redirect to integrated sync
    await this.syncAllIntegratedData()
  }
  
  // Legacy sync methods removed - using integrated services instead
  
  private async handleTaskMove(event: CustomEvent<TaskMoveEvent>) {
    const { taskId, newStatus, newIndex, offline } = event.detail
    
    try {
      // Update local state immediately for better UX
      const taskIndex = this.tasks.findIndex(t => t.id === taskId)
      if (taskIndex >= 0) {
        const updatedTasks = [...this.tasks]
        updatedTasks[taskIndex] = {
          ...updatedTasks[taskIndex],
          status: newStatus,
          updatedAt: new Date().toISOString(),
          syncStatus: offline ? 'pending' : 'synced'
        }
        this.tasks = updatedTasks
      }
      
      // If offline, queue for later sync
      if (offline) {
        await this.offlineService.queueTaskUpdate(taskId, { status: newStatus })
        await this.offlineService.cacheTasks(this.tasks)
        return
      }
      
      // Use backend adapter for task updates (mock operation)
      const updatedTask = await this.backendAdapter.mockWriteOperation('updateTask', { 
        id: taskId,
        status: newStatus,
        updated_at: new Date() 
      })
      
      // Update with service response
      const finalTaskIndex = this.tasks.findIndex(t => t.id === taskId)
      if (finalTaskIndex >= 0) {
        const finalTasks = [...this.tasks]
        finalTasks[finalTaskIndex] = { ...updatedTask, syncStatus: 'synced' }
        this.tasks = finalTasks
        await this.offlineService.cacheTasks(this.tasks)
      }
      
    } catch (error) {
      console.error('Failed to move task:', error)
      
      // Mark task as having sync error
      const errorTaskIndex = this.tasks.findIndex(t => t.id === taskId)
      if (errorTaskIndex >= 0) {
        const errorTasks = [...this.tasks]
        errorTasks[errorTaskIndex] = {
          ...errorTasks[errorTaskIndex],
          syncStatus: 'error'
        }
        this.tasks = errorTasks
      }
      
      // Show notification
      this.notificationService.showError('Failed to update task')
    }
  }
  
  // Enhanced real-time event handlers
  private handleRealtimeTaskUpdate(taskData: Task) {
    this.lastUpdateTimestamp = new Date()
    
    const index = this.tasks.findIndex(t => t.id === taskData.id)
    if (index >= 0) {
      const updated = [...this.tasks]
      updated[index] = { ...taskData, syncStatus: 'synced' }
      this.tasks = updated
      
      // Announce update for screen readers
      this.makeLiveAnnouncement(`Task "${taskData.title}" updated`)
      
      // Add visual update indicator
      this.showUpdateIndicator('task-updated')
    }
  }
  
  private handleRealtimeTaskCreated(taskData: Task) {
    this.lastUpdateTimestamp = new Date()
    this.tasks = [...this.tasks, { ...taskData, syncStatus: 'synced' }]
    
    this.makeLiveAnnouncement(`New task "${taskData.title}" created`)
    this.showUpdateIndicator('task-created')
  }
  
  private handleRealtimeTaskDeleted(taskId: string) {
    const task = this.tasks.find(t => t.id === taskId)
    this.tasks = this.tasks.filter(t => t.id !== taskId)
    
    if (task) {
      this.makeLiveAnnouncement(`Task "${task.title}" deleted`)
      this.showUpdateIndicator('task-deleted')
    }
  }
  
  private handleRealtimeAgentUpdate(agentData: any) {
    this.lastUpdateTimestamp = new Date()
    
    const index = this.agents.findIndex(a => a.id === agentData.id)
    if (index >= 0) {
      const updated = [...this.agents]
      updated[index] = this.transformAgentToUIFormat(agentData)
      this.agents = updated
      
      const statusChanged = this.agents[index].status !== agentData.status
      if (statusChanged) {
        this.makeLiveAnnouncement(`Agent ${agentData.name} status changed to ${agentData.status}`)
        this.showUpdateIndicator('agent-status-changed')
      }
    }
  }
  
  private handleRealtimeSystemEvent(eventData: any) {
    this.lastUpdateTimestamp = new Date()
    
    const newEvent = this.transformEventToUIFormat(eventData)
    this.events = [newEvent, ...this.events].slice(0, 100) // Keep only latest 100 events
    
    if (eventData.severity === 'high' || eventData.severity === 'critical') {
      this.makeLiveAnnouncement(`Critical system event: ${eventData.summary}`)
      this.showUpdateIndicator('system-alert')
    }
  }
  
  private handleRealtimeMetricsUpdate(metricsData: any) {
    this.lastUpdateTimestamp = new Date()
    this.performanceMetrics = metricsData
    
    // Update health summary if included
    if (metricsData.healthSummary) {
      this.healthSummary = metricsData.healthSummary
    }
    
    this.showUpdateIndicator('metrics-updated')
  }
  
  private showUpdateIndicator(updateType: string) {
    // Add a visual indicator that shows real-time updates
    const indicator = this.shadowRoot?.querySelector('.realtime-indicator')
    if (indicator) {
      indicator.classList.add('active', updateType)
      setTimeout(() => {
        indicator.classList.remove('active', updateType)
      }, 2000)
    }
    
    // Add to update queue for batching
    this.updateQueue.push({
      type: updateType,
      timestamp: new Date(),
      id: Math.random().toString(36).substr(2, 9)
    })
    
    // Keep only recent updates
    this.updateQueue = this.updateQueue.slice(-10)
  }
  
  // Legacy handlers for backward compatibility
  private handleTaskUpdate(taskData: Task) {
    this.handleRealtimeTaskUpdate(taskData)
  }
  
  private handleTaskCreated(taskData: Task) {
    this.handleRealtimeTaskCreated(taskData)
  }
  
  private handleTaskDeleted(taskId: string) {
    this.handleRealtimeTaskDeleted(taskId)
  }
  
  private handleTaskClick(event: CustomEvent) {
    const { task } = event.detail
    
    // Dispatch event for parent to handle (could open modal, navigate, etc.)
    const clickEvent = new CustomEvent('task-details', {
      detail: { task },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(clickEvent)
  }
  
  private async handleRefresh() {
    console.log('Manual refresh triggered')
    await this.syncAllIntegratedData()
  }
  
  private get syncStatusText() {
    if (this.offline) {
      return 'Offline mode'
    }
    
    if (this.lastSync) {
      const mins = Math.floor((Date.now() - this.lastSync.getTime()) / (1000 * 60))
      return `Last sync: ${mins < 1 ? 'just now' : `${mins}m ago`}`
    }
    
    return 'Not synced'
  }
  
  // Accessibility and keyboard navigation methods
  private handleTabClick(view: 'overview' | 'kanban' | 'agents' | 'events' | 'performance' | 'security' | 'oversight' | 'control' | 'coordination' | 'debt') {
    this.selectedView = view
    this.announceViewChange(view)
  }
  
  private handleTabKeydown(event: KeyboardEvent) {
    const tabs = this.shadowRoot?.querySelectorAll('[role="tab"]') as NodeListOf<HTMLElement>
    if (!tabs) return
    
    const currentIndex = Array.from(tabs).findIndex(tab => tab.getAttribute('aria-selected') === 'true')
    let newIndex = currentIndex
    
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        event.preventDefault()
        newIndex = (currentIndex + 1) % tabs.length
        break
      case 'ArrowLeft':
      case 'ArrowUp':
        event.preventDefault()
        newIndex = (currentIndex - 1 + tabs.length) % tabs.length
        break
      case 'Home':
        event.preventDefault()
        newIndex = 0
        break
      case 'End':
        event.preventDefault()
        newIndex = tabs.length - 1
        break
      case 'Enter':
      case ' ':
        event.preventDefault()
        const view = tabs[currentIndex].getAttribute('aria-controls')?.replace('-panel', '') as any
        if (view) this.handleTabClick(view)
        return
      default:
        return
    }
    
    // Update focus and selection
    tabs[currentIndex].setAttribute('tabindex', '-1')
    tabs[currentIndex].setAttribute('aria-selected', 'false')
    tabs[newIndex].setAttribute('tabindex', '0')
    tabs[newIndex].setAttribute('aria-selected', 'true')
    tabs[newIndex].focus()
    
    // Update selected view
    const newView = tabs[newIndex].getAttribute('aria-controls')?.replace('-panel', '') as any
    if (newView) {
      this.selectedView = newView
      this.announceViewChange(newView)
    }
  }
  
  private announceViewChange(view: string) {
    const viewNames = {
      overview: 'Overview dashboard',
      kanban: 'Task management board',
      agents: 'Agent health and management',
      events: 'System events timeline',
      debt: 'Technical debt analysis and monitoring'
    }
    
    const announcement = `Switched to ${viewNames[view as keyof typeof viewNames] || view} view`
    
    // Create a live region announcement
    const announcer = document.createElement('div')
    announcer.setAttribute('aria-live', 'polite')
    announcer.setAttribute('aria-atomic', 'true')
    announcer.className = 'sr-only'
    announcer.textContent = announcement
    
    document.body.appendChild(announcer)
    setTimeout(() => document.body.removeChild(announcer), 1000)
  }
  
  private get dashboardSummary() {
    const activeTasks = this.tasks.filter(t => t.status === 'in-progress').length
    const completedTasks = this.tasks.filter(t => t.status === 'done').length
    const activeAgents = this.agents.filter(a => a.status === 'active').length
    const errorAgents = this.agents.filter(a => a.status === 'error').length
    const recentEvents = this.events.filter(e => {
      const eventTime = new Date(e.timestamp)
      const hourAgo = new Date(Date.now() - 60 * 60 * 1000)
      return eventTime > hourAgo
    }).length
    
    // Include system health information
    const systemStatus = this.healthSummary?.overall || 'unknown'
    const healthyComponents = this.healthSummary?.components.healthy || 0
    const unhealthyComponents = (this.healthSummary?.components.degraded || 0) + 
                               (this.healthSummary?.components.unhealthy || 0)
    
    return {
      activeTasks,
      completedTasks,
      totalTasks: this.tasks.length,
      activeAgents,
      errorAgents,
      totalAgents: this.agents.length,
      recentEvents,
      systemStatus,
      healthyComponents,
      unhealthyComponents,
      cpuUsage: this.performanceMetrics?.system_metrics?.cpu_usage || 0,
      memoryUsage: this.performanceMetrics?.system_metrics?.memory_usage || 0
    }
  }
  
  private renderOverviewView() {
    const summary = this.dashboardSummary
    
    return html`
      <div id="overview-panel" role="tabpanel" aria-labelledby="overview-tab">
        <h2 class="sr-only">Dashboard Overview</h2>
        
        <section class="overview-summary" aria-label="System metrics summary" role="region">
          <div class="summary-card" role="img" aria-label="Active tasks: ${summary.activeTasks}">
            <div class="summary-value" aria-hidden="true">${summary.activeTasks}</div>
            <div class="summary-label">Active Tasks</div>
          </div>
          <div class="summary-card" role="img" aria-label="Completed tasks: ${summary.completedTasks}">
            <div class="summary-value" aria-hidden="true">${summary.completedTasks}</div>
            <div class="summary-label">Completed Tasks</div>
          </div>
          <div class="summary-card" role="img" aria-label="Active agents: ${summary.activeAgents}">
            <div class="summary-value" aria-hidden="true">${summary.activeAgents}</div>
            <div class="summary-label">Active Agents</div>
          </div>
          <div class="summary-card" role="img" aria-label="System health: ${summary.systemStatus}">
            <div class="summary-value" 
                 style="color: ${summary.systemStatus === 'healthy' ? '#10b981' : summary.systemStatus === 'degraded' ? '#f59e0b' : '#ef4444'}"
                 aria-hidden="true">
              ${summary.systemStatus.toUpperCase()}
            </div>
            <div class="summary-label">System Health</div>
          </div>
          <div class="summary-card enhanced-metric-card" role="img" aria-label="CPU usage: ${Math.round(summary.cpuUsage)} percent">
            <div class="metric-header">
              <div class="summary-value" aria-hidden="true">${Math.round(summary.cpuUsage)}%</div>
              <div class="metric-trend ${summary.cpuUsage > 80 ? 'trend-critical' : summary.cpuUsage > 60 ? 'trend-warning' : 'trend-healthy'}">
                <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${summary.cpuUsage > 60 ? 'M7 17l9.2-9.2M17 17V7H7' : 'M5 12h14'}"/>
                </svg>
              </div>
            </div>
            <div class="summary-label">CPU Usage</div>
            <div class="metric-bar">
              <div class="metric-fill" style="width: ${Math.round(summary.cpuUsage)}%; background-color: ${summary.cpuUsage > 80 ? '#ef4444' : summary.cpuUsage > 60 ? '#f59e0b' : '#10b981'}"></div>
            </div>
          </div>
          <div class="summary-card enhanced-metric-card" role="img" aria-label="Memory usage: ${Math.round(summary.memoryUsage)} percent">
            <div class="metric-header">
              <div class="summary-value" aria-hidden="true">${Math.round(summary.memoryUsage)}%</div>
              <div class="metric-trend ${summary.memoryUsage > 85 ? 'trend-critical' : summary.memoryUsage > 70 ? 'trend-warning' : 'trend-healthy'}">
                <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${summary.memoryUsage > 70 ? 'M7 17l9.2-9.2M17 17V7H7' : 'M5 12h14'}"/>
                </svg>
              </div>
            </div>
            <div class="summary-label">Memory Usage</div>
            <div class="metric-bar">
              <div class="metric-fill" style="width: ${Math.round(summary.memoryUsage)}%; background-color: ${summary.memoryUsage > 85 ? '#ef4444' : summary.memoryUsage > 70 ? '#f59e0b' : '#10b981'}"></div>
            </div>
          </div>
          <div class="summary-card" role="img" aria-label="Recent events: ${summary.recentEvents}">
            <div class="summary-value" aria-hidden="true">${summary.recentEvents}</div>
            <div class="summary-label">Recent Events</div>
          </div>
          <div class="summary-card" role="img" aria-label="Healthy components: ${summary.healthyComponents}">
            <div class="summary-value" aria-hidden="true">${summary.healthyComponents}</div>
            <div class="summary-label">Healthy Components</div>
          </div>
        </section>
        
        <div class="overview-panels">
          <section class="panel-section" aria-label="Agent health status">
            <agent-health-panel 
              .agents=${this.agents}
              .compact=${true}
              @refresh-agents=${this.handleRefresh}
              role="region"
              aria-label="Agent health and status information"
            ></agent-health-panel>
          </section>
          
          <section class="panel-section" aria-label="System events timeline">
            <event-timeline
              .events=${this.events}
              .maxEvents=${20}
              .realtime=${!this.offline}
              .compact=${true}
              role="region"
              aria-label="Recent system events and activities"
            ></event-timeline>
          </section>
        </div>
      </div>
    `
  }
  
  private renderKanbanView() {
    return html`
      <div id="kanban-panel" role="tabpanel" aria-labelledby="kanban-tab" style="height: calc(100vh - 140px); padding: 0;">
        <h2 class="sr-only">Task Management Board</h2>
        <kanban-board
          .tasks=${this.tasks}
          .offline=${this.offline}
          @task-move=${this.handleTaskMove}
          @task-click=${this.handleTaskClick}
          @tasks-updated=${(e: CustomEvent) => {
            this.tasks = e.detail.tasks
          }}
          role="application"
          aria-label="Kanban task management board with drag and drop functionality"
        ></kanban-board>
      </div>
    `
  }
  
  private renderAgentsView() {
    return html`
      <div id="agents-panel" role="tabpanel" aria-labelledby="agents-tab" style="height: calc(100vh - 140px); padding: 1rem;">
        <h2 class="sr-only">Agent Health and Management</h2>
        <agent-health-panel 
          .agents=${this.agents}
          .compact=${false}
          @refresh-agents=${this.handleRefresh}
          @agent-selected=${(e: CustomEvent) => {
            console.log('Agent selected:', e.detail.agent)
            this.announceAgentSelection(e.detail.agent)
          }}
          role="region"
          aria-label="Detailed agent health monitoring and management controls"
        ></agent-health-panel>
      </div>
    `
  }
  
  private renderEventsView() {
    return html`
      <div id="events-panel" role="tabpanel" aria-labelledby="events-tab" style="height: calc(100vh - 140px);">
        <h2 class="sr-only">System Events Timeline</h2>
        <event-timeline
          .events=${this.events}
          .maxEvents=${100}
          .realtime=${!this.offline}
          .compact=${false}
          @event-selected=${(e: CustomEvent) => {
            console.log('Event selected:', e.detail.event)
            this.announceEventSelection(e.detail.event)
          }}
          role="log"
          aria-label="System events and activities timeline"
          aria-live="polite"
        ></event-timeline>
      </div>
    `
  }

  private renderOversightView() {
    return html`
      <div id="oversight-panel" role="tabpanel" aria-labelledby="oversight-tab" style="height: calc(100vh - 140px);">
        <h2 class="sr-only">Multi-Agent Oversight Dashboard</h2>
        <multi-agent-oversight-dashboard
          .fullscreen=${false}
          .viewMode=${'grid'}
          role="application"
          aria-label="Advanced multi-agent oversight with real-time monitoring and control"
        ></multi-agent-oversight-dashboard>
      </div>
    `
  }

  private renderControlView() {
    return html`
      <div id="control-panel" role="tabpanel" aria-labelledby="control-tab" style="height: calc(100vh - 140px);">
        <h2 class="sr-only">Remote Control Center</h2>
        <remote-control-center
          .expanded=${true}
          .selectedAgents=${[]}
          .emergencyMode=${this.error ? true : false}
          role="application"
          aria-label="Remote agent control center with voice commands and emergency controls"
        ></remote-control-center>
      </div>
    `
  }

  private renderCoordinationView() {
    return html`
      <div id="coordination-panel" role="tabpanel" aria-labelledby="coordination-tab" style="height: calc(100vh - 140px); padding: 1rem; overflow-y: auto;">
        <h2 class="sr-only">Multi-Agent Coordination Monitoring</h2>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1.5rem;">
          <!-- Critical Success Rate Tracking -->
          <section aria-label="Coordination success rate monitoring">
            <coordination-success-panel
              .realtime=${this.realtimeEnabled}
              .compact=${false}
              .timeRange=${'1h'}
              role="region"
              aria-label="Real-time coordination success rate with failure analysis and recovery actions"
            ></coordination-success-panel>
          </section>

          <!-- Agent Status Grid -->
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
            <section aria-label="Real-time agent status monitoring">
              <realtime-agent-status-panel
                .realtime=${this.realtimeEnabled}
                .compact=${false}
                .viewMode=${'grid'}
                role="region"
                aria-label="Live agent health monitoring with performance metrics and recovery controls"
              ></realtime-agent-status-panel>
            </section>

            <section aria-label="Task distribution and queue management">
              <task-distribution-panel
                .realtime=${this.realtimeEnabled}
                .enableDragDrop=${true}
                .agents=${this.agents}
                role="region"
                aria-label="Interactive task queue with drag-and-drop reassignment capabilities"
              ></task-distribution-panel>
            </section>
          </div>

          <!-- Communication and Recovery -->
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
            <section aria-label="Communication health monitoring">
              <communication-monitoring-panel
                .realtime=${this.realtimeEnabled}
                .compact=${false}
                .timeRange=${'1h'}
                role="region"
                aria-label="Agent communication monitoring with Redis health and latency tracking"
              ></communication-monitoring-panel>
            </section>

            <section aria-label="Emergency recovery controls">
              <recovery-controls-panel
                .agents=${this.agents}
                .systemHealth=${this.systemHealth}
                .emergencyMode=${this.error ? true : false}
                role="region"
                aria-label="Emergency recovery actions and system diagnostic tools"
              ></recovery-controls-panel>
            </section>
          </div>
        </div>
      </div>
    `
  }

  private renderToolsView() {
    return html`
      <div id="tools-panel" role="tabpanel" aria-labelledby="tools-tab" style="height: calc(100vh - 140px); padding: 1rem; overflow-y: auto;">
        <h2 class="sr-only">Development Tools and Utilities</h2>
        
        <div style="display: grid; grid-template-columns: 1fr; gap: 1.5rem; max-width: 1200px; margin: 0 auto;">
          <!-- Context Compression Tools -->
          <section aria-label="Context compression and optimization tools">
            <h3 style="margin: 0 0 1rem 0; font-size: 1.25rem; font-weight: 600; color: var(--text-primary-color, #1f2937);">
              Context Compression
            </h3>
            <compression-dashboard
              .contextInfo=${{
                tokenCount: 45000, // Example token count
                sessionType: 'development_session',
                priority: 'quality'
              }}
              role="region"
              aria-label="Context compression dashboard with real-time progress monitoring and analytics"
            ></compression-dashboard>
          </section>

          <!-- Quick Tools Grid -->
          <section aria-label="Quick access development tools">
            <h3 style="margin: 0 0 1rem 0; font-size: 1.25rem; font-weight: 600; color: var(--text-primary-color, #1f2937);">
              Quick Tools
            </h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
              <compression-widget
                .contextInfo=${{
                  tokenCount: 45000,
                  sessionType: 'development_session',
                  priority: 'quality'
                }}
                @compression-started=${(e: CustomEvent) => {
                  console.log('Compression started:', e.detail)
                  this.makeLiveAnnouncement('Context compression started')
                }}
                @compression-error=${(e: CustomEvent) => {
                  console.error('Compression error:', e.detail)
                  this.makeLiveAnnouncement('Context compression failed')
                }}
                role="region"
                aria-label="Quick context compression widget with smart recommendations"
              ></compression-widget>
              
              <!-- Placeholder for additional tools -->
              <div style="background: var(--surface-color, #ffffff); border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); border: 1px solid var(--border-color, #e0e0e0); display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 200px; color: var(--text-secondary-color, #666);">
                <div style="font-size: 48px; margin-bottom: 16px; opacity: 0.5;">🛠️</div>
                <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px;">More Tools Coming Soon</div>
                <div style="font-size: 14px; text-align: center; line-height: 1.5;">
                  Additional development tools and utilities will be added here.
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    `
  }
  
  private announceAgentSelection(agent: any) {
    const announcement = `Agent ${agent.name} selected. Status: ${agent.status}. Performance: ${agent.performance?.score || 'unknown'}%`
    this.makeLiveAnnouncement(announcement)
  }
  
  private announceEventSelection(event: any) {
    const announcement = `Event selected: ${event.title}. Type: ${event.type}. Time: ${new Date(event.timestamp).toLocaleTimeString()}`
    this.makeLiveAnnouncement(announcement)
  }
  
  private makeLiveAnnouncement(message: string) {
    const liveRegion = this.shadowRoot?.querySelector('#live-region')
    if (liveRegion) {
      liveRegion.textContent = message
      setTimeout(() => {
        liveRegion.textContent = ''
      }, 3000)
    }
  }
  
  private toggleRealtime() {
    this.realtimeEnabled = !this.realtimeEnabled
    
    if (this.realtimeEnabled) {
      // Re-enable WebSocket if offline
      if (!this.wsConnected && !this.offline) {
        this.websocketService.connect()
      }
      this.makeLiveAnnouncement('Real-time updates enabled')
    } else {
      this.makeLiveAnnouncement('Real-time updates paused')
    }
    
    // Dispatch event for other components
    const toggleEvent = new CustomEvent('realtime-toggled', {
      detail: { enabled: this.realtimeEnabled },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(toggleEvent)
  }
  
  private renderPerformanceView() {
    return html`
      <div id="performance-panel" role="tabpanel" aria-labelledby="performance-tab" style="height: calc(100vh - 140px); padding: 0;">
        <h2 class="sr-only">Performance Monitoring Dashboard</h2>
        <performance-metrics-panel
          .metrics=${this.comprehensivePerformanceMetrics}
          .realtime=${!this.offline && this.realtimeEnabled}
          .compact=${false}
          .timeRange=${"1h"}
          @tab-changed=${(e: CustomEvent) => {
            console.log('Performance tab changed:', e.detail.tab)
          }}
          @auto-refresh-toggled=${(e: CustomEvent) => {
            console.log('Auto-refresh toggled:', e.detail.enabled)
          }}
          role="application"
          aria-label="Real-time performance monitoring with metrics, alerts, and system health data"
        ></performance-metrics-panel>
      </div>
    `
  }
  
  private renderDebtView() {
    return html`
      <div id="debt-panel" role="tabpanel" aria-labelledby="debt-tab" style="height: calc(100vh - 140px); padding: 1rem; overflow-y: auto;">
        <h2 class="sr-only">Technical Debt Analysis and Monitoring</h2>
        
        <div style="max-width: 1200px; margin: 0 auto;">
          <technical-debt-panel
            .projects=${this.technicalDebtProjects}
            .compact=${false}
            .sortBy=${'debt_score'}
            .filterSeverity=${'all'}
            .showRecommendations=${true}
            @analyze-project=${this.handleAnalyzeProject}
            @analyze-all-projects=${this.handleAnalyzeAllProjects}
            role="region"
            aria-label="Technical debt monitoring and analysis dashboard"
          ></technical-debt-panel>
        </div>
      </div>
    `
  }
  
  private async handleAnalyzeProject(event: CustomEvent) {
    const { projectId } = event.detail
    console.log('Analyzing project:', projectId)
    
    try {
      this.debtAnalysisInProgress = true
      
      // Perform debt analysis
      const analysisResult = await this.technicalDebtService.analyzeProject(projectId, {
        include_advanced_patterns: true,
        include_historical_analysis: true,
        analysis_depth: 'comprehensive'
      })
      
      // Get historical data for trend analysis
      let historyResult
      try {
        historyResult = await this.technicalDebtService.getDebtHistory(projectId, 90, 7)
      } catch (error) {
        console.warn('Could not get historical data:', error)
      }
      
      // Update the project in our list
      const projectIndex = this.technicalDebtProjects.findIndex(p => p.project_id === projectId)
      const projectName = projectIndex >= 0 ? this.technicalDebtProjects[projectIndex].project_name : `Project ${projectId}`
      
      const updatedProject = this.technicalDebtService.convertToProjectStatus(
        projectId,
        projectName,
        analysisResult,
        historyResult
      )
      
      if (projectIndex >= 0) {
        this.technicalDebtProjects[projectIndex] = updatedProject
      } else {
        this.technicalDebtProjects.push(updatedProject)
      }
      
      this.requestUpdate()
      this.makeLiveAnnouncement(`Technical debt analysis completed for ${projectName}`)
      
    } catch (error) {
      console.error('Failed to analyze project:', error)
      this.makeLiveAnnouncement(`Technical debt analysis failed: ${error.message}`)
    } finally {
      this.debtAnalysisInProgress = false
    }
  }
  
  private async handleAnalyzeAllProjects(event: CustomEvent) {
    console.log('Analyzing all projects')
    
    try {
      this.debtAnalysisInProgress = true
      
      // For demo purposes, analyze a sample project
      // In real implementation, this would get all projects from the project index
      const sampleProjectId = '550e8400-e29b-41d4-a716-446655440000' // Mock UUID
      await this.handleAnalyzeProject(new CustomEvent('analyze-project', {
        detail: { projectId: sampleProjectId }
      }))
      
    } catch (error) {
      console.error('Failed to analyze all projects:', error)
      this.makeLiveAnnouncement(`Bulk technical debt analysis failed: ${error.message}`)
    }
  }
  
  private renderCurrentView() {
    switch (this.selectedView) {
      case 'overview':
        return this.renderOverviewView()
      case 'kanban':
        return this.renderKanbanView()
      case 'agents':
        return this.renderAgentsView()
      case 'events':
        return this.renderEventsView()
      case 'performance':
        return this.renderPerformanceView()
      case 'oversight':
        return this.renderOversightView()
      case 'control':
        return this.renderControlView()
      case 'coordination':
        return this.renderCoordinationView()
      case 'tools':
        return this.renderToolsView()
      case 'debt':
        return this.renderDebtView()
      default:
        return this.renderOverviewView()
    }
  }
  
  render() {
    if (this.isLoading && this.tasks.length === 0 && this.agents.length === 0) {
      return html`
        <enhanced-loading-spinner
          variant="skeleton"
          message="Loading dashboard data..."
          overlay
        ></enhanced-loading-spinner>
      `
    }
    
    return html`
      <div class="dashboard-header">
        <div class="header-content">
          <h1 class="page-title">
            <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
            Agent Dashboard
          </h1>
          
          <div style="display: flex; align-items: center; gap: 1rem;">
            <!-- Connection Status -->
            <div class="connection-status">
              <div class="connection-dot ${this.connectionQuality}"></div>
              ${this.wsConnected ? 'Live' : 'Offline'}
              ${this.lastUpdateTimestamp ? html`
                <span title="Last update: ${this.lastUpdateTimestamp.toLocaleTimeString()}">
                  • ${Math.floor((Date.now() - this.lastUpdateTimestamp.getTime()) / 1000)}s ago
                </span>
              ` : ''}
            </div>
            
            <!-- Real-time Indicator -->
            <div class="realtime-indicator">
              <div class="realtime-pulse"></div>
              ${this.updateQueue.length > 0 ? html`
                <div class="update-queue-indicator visible" title="${this.updateQueue.length} recent updates">
                  ${this.updateQueue.length}
                </div>
              ` : ''}
            </div>
            
            <!-- Sync Status -->
            <div class="sync-status">
              <div class="sync-indicator ${this.offline ? 'offline' : ''}"></div>
              ${this.syncStatusText}
            </div>
            
            <!-- Connection Monitor -->
            <connection-monitor compact .showControls=${false}></connection-monitor>
            
            <!-- Real-time Controls -->
            <div class="realtime-controls">
              <button
                class="realtime-toggle ${this.realtimeEnabled ? 'active' : ''}"
                @click=${this.toggleRealtime}
                title="${this.realtimeEnabled ? 'Disable' : 'Enable'} real-time updates"
                aria-label="${this.realtimeEnabled ? 'Disable' : 'Enable'} real-time updates"
              >
                ${this.realtimeEnabled ? '🔴 Live' : '⏸️ Paused'}
              </button>
            </div>
            
            <!-- Refresh Button -->
            <button
              class="refresh-button ${this.isLoading ? 'spinning' : ''}"
              @click=${this.handleRefresh}
              ?disabled=${this.isLoading}
              title="Refresh data manually"
              aria-label="Refresh dashboard data"
            >
              <svg class="refresh-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        </div>
        
        <div class="view-tabs" role="tablist" aria-label="Dashboard navigation">
          <button 
            class="tab-button ${this.selectedView === 'overview' ? 'active' : ''}"
            role="tab"
            aria-selected=${this.selectedView === 'overview'}
            aria-controls="overview-panel"
            tabindex=${this.selectedView === 'overview' ? '0' : '-1'}
            @click=${() => this.handleTabClick('overview')}
            @keydown=${this.handleTabKeydown}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
            Overview
          </button>
          
          <button 
            class="tab-button ${this.selectedView === 'kanban' ? 'active' : ''}"
            role="tab"
            aria-selected=${this.selectedView === 'kanban'}
            aria-controls="kanban-panel"
            tabindex=${this.selectedView === 'kanban' ? '0' : '-1'}
            @click=${() => this.handleTabClick('kanban')}
            @keydown=${this.handleTabKeydown}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
            Tasks
          </button>
          
          <button 
            class="tab-button ${this.selectedView === 'agents' ? 'active' : ''}"
            role="tab"
            aria-selected=${this.selectedView === 'agents'}
            aria-controls="agents-panel"
            tabindex=${this.selectedView === 'agents' ? '0' : '-1'}
            @click=${() => this.handleTabClick('agents')}
            @keydown=${this.handleTabKeydown}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Agents
          </button>
          
          <button 
            class="tab-button ${this.selectedView === 'events' ? 'active' : ''}"
            role="tab"
            aria-selected=${this.selectedView === 'events'}
            aria-controls="events-panel"
            tabindex=${this.selectedView === 'events' ? '0' : '-1'}
            @click=${() => this.handleTabClick('events')}
            @keydown=${this.handleTabKeydown}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Events
          </button>

          <button 
            class="tab-button ${this.selectedView === 'performance' ? 'active' : ''}"
            role="tab"
            aria-selected=${this.selectedView === 'performance'}
            aria-controls="performance-panel"
            tabindex=${this.selectedView === 'performance' ? '0' : '-1'}
            @click=${() => this.handleTabClick('performance')}
            @keydown=${this.handleTabKeydown}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Performance
          </button>

          <button 
            class="tab-button ${this.selectedView === 'oversight' ? 'active' : ''}"
            role="tab"
            aria-selected=${this.selectedView === 'oversight'}
            aria-controls="oversight-panel"
            tabindex=${this.selectedView === 'oversight' ? '0' : '-1'}
            @click=${() => this.handleTabClick('oversight')}
            @keydown=${this.handleTabKeydown}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Oversight
          </button>

          <button 
            class="tab-button ${this.selectedView === 'control' ? 'active' : ''}"
            role="tab"
            aria-selected=${this.selectedView === 'control'}
            aria-controls="control-panel"
            tabindex=${this.selectedView === 'control' ? '0' : '-1'}
            @click=${() => this.handleTabClick('control')}
            @keydown=${this.handleTabKeydown}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
            </svg>
            Control
          </button>

          <button 
            class="tab-button ${this.selectedView === 'coordination' ? 'active' : ''}"
            role="tab"
            aria-selected=${this.selectedView === 'coordination'}
            aria-controls="coordination-panel"
            tabindex=${this.selectedView === 'coordination' ? '0' : '-1'}
            @click=${() => this.handleTabClick('coordination')}
            @keydown=${this.handleTabKeydown}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            🚨 Coordination
          </button>
          
          <button 
            class="tab-button ${this.selectedView === 'debt' ? 'active' : ''}"
            role="tab"
            aria-selected=${this.selectedView === 'debt'}
            aria-controls="debt-panel"
            tabindex=${this.selectedView === 'debt' ? '0' : '-1'}
            @click=${() => this.handleTabClick('debt')}
            @keydown=${this.handleTabKeydown}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            📊 Debt Analysis
          </button>
        </div>
      </div>
      
      ${this.error ? html`
        <div class="error-state">
          <p><strong>Error:</strong> ${this.error}</p>
          <div class="error-actions">
            <button class="btn btn-primary" @click=${this.handleRefresh}>
              Try Again
            </button>
          </div>
        </div>
      ` : ''}
      
      <main class="dashboard-content" role="main">
        <!-- Skip link for keyboard navigation -->
        <a href="#main-content" class="skip-link">Skip to main content</a>
        
        <!-- Live region for dynamic announcements -->
        <div id="live-region" aria-live="polite" aria-atomic="true" class="sr-only"></div>
        
        <div id="main-content" tabindex="-1">
          ${this.renderCurrentView()}
        </div>
      </main>
    `
  }

  // Mock data methods for when backend is unavailable
  private getMockTasks(): Task[] {
    return [
      {
        id: 'task-1',
        title: 'Setup CI/CD Pipeline',
        status: 'in-progress' as TaskStatus,
        priority: 'high',
        type: 'feature',
        agent: 'DevOps Agent',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        syncStatus: 'synced'
      },
      {
        id: 'task-2', 
        title: 'Review Mobile Dashboard',
        status: 'todo' as TaskStatus,
        priority: 'medium',
        type: 'enhancement',
        agent: 'QA Agent',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        syncStatus: 'synced'
      },
      {
        id: 'task-3',
        title: 'Deploy to Production',
        status: 'done' as TaskStatus,
        priority: 'high',
        type: 'feature',
        agent: 'DevOps Agent',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        syncStatus: 'synced'
      }
    ]
  }

  private getMockAgents(): Agent[] {
    return [
      {
        id: 'agent-1',
        name: 'Development Agent',
        role: 'developer',
        status: 'active',
        capabilities: ['coding', 'testing', 'deployment'],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        last_activity: new Date().toISOString(),
        performance_metrics: {
          tasks_completed: 15,
          tasks_failed: 1,
          average_completion_time: 45,
          cpu_usage: 65.2,
          memory_usage: 78.5,
          success_rate: 93.8,
          uptime: 99.5
        }
      },
      {
        id: 'agent-2',
        name: 'QA Agent', 
        role: 'qa',
        status: 'active',
        capabilities: ['testing', 'validation', 'reporting'],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        last_activity: new Date().toISOString(),
        performance_metrics: {
          tasks_completed: 12,
          tasks_failed: 0,
          average_completion_time: 32,
          cpu_usage: 45.7,
          memory_usage: 62.3,
          success_rate: 100,
          uptime: 98.2
        }
      },
      {
        id: 'agent-3',
        name: 'DevOps Agent',
        role: 'orchestrator',
        status: 'idle',
        capabilities: ['deployment', 'monitoring', 'maintenance'],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        last_activity: new Date().toISOString(),
        performance_metrics: {
          tasks_completed: 8,
          tasks_failed: 0,
          average_completion_time: 28,
          cpu_usage: 25.1,
          memory_usage: 41.6,
          success_rate: 100,
          uptime: 100
        }
      }
    ]
  }

  private getMockSystemHealth(): SystemHealth {
    return {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      components: {
        database: { status: 'healthy', latency: 45, lastCheck: new Date().toISOString() },
        redis: { status: 'healthy', latency: 25, lastCheck: new Date().toISOString() },
        orchestrator: { status: 'healthy', latency: 120, lastCheck: new Date().toISOString() },
        agents: { status: 'healthy', latency: 30, lastCheck: new Date().toISOString() }
      },
      metrics: {
        cpu_usage: 45.2,
        memory_usage: 62.8,
        disk_usage: 34.1,
        network_latency: 89,
        error_rate: 0.1,
        throughput: 127
      }
    }
  }

  private getMockEvents(): SystemEvent[] {
    return [
      {
        id: 'event-1',
        type: 'task_completed',
        severity: 'info',
        title: 'Task Completed',
        description: 'Development Agent completed "Setup CI/CD Pipeline"',
        source: 'agent_system',
        agent_id: 'agent-1',
        task_id: 'task-1',
        data: { completion_time: 45 },
        timestamp: new Date().toISOString()
      },
      {
        id: 'event-2',
        type: 'agent_activated',
        severity: 'info',
        title: 'Agent Started',
        description: 'QA Agent successfully started and ready for tasks',
        source: 'agent_system',
        agent_id: 'agent-2',
        data: { startup_time: 2.3 },
        timestamp: new Date(Date.now() - 60000).toISOString()
      },
      {
        id: 'event-3',
        type: 'system_healthy',
        severity: 'info',
        title: 'System Healthy',
        description: 'All systems operational - 99.8% uptime',
        source: 'health_monitor',
        data: { uptime: 99.8 },
        timestamp: new Date(Date.now() - 120000).toISOString()
      }
    ]
  }

  private getMockMetrics(): PerformanceSnapshot {
    return {
      timestamp: new Date().toISOString(),
      cpu: 45.2,
      memory: 62.8,
      disk: 34.1,
      network: { in: 127, out: 89 },
      agents: {
        total: 3,
        active: 2,
        busy: 1,
        idle: 1,
        error: 0
      },
      tasks: {
        total: 25,
        pending: 5,
        in_progress: 8,
        completed: 12,
        failed: 0
      }
    }
  }
}