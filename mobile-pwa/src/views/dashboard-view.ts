import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { Task, TaskStatus, TaskMoveEvent } from '../types/task'
import { AgentStatus } from '../components/dashboard/agent-health-panel'
import { TimelineEvent } from '../components/dashboard/event-timeline'
import { WebSocketService } from '../services/websocket'
import { OfflineService } from '../services/offline'
import { NotificationService } from '../services/notification'
import '../components/kanban/kanban-board'
import '../components/dashboard/agent-health-panel'
import '../components/dashboard/event-timeline'

@customElement('dashboard-view')
export class DashboardView extends LitElement {
  @property({ type: Boolean }) offline: boolean = false
  
  @state() private tasks: Task[] = []
  @state() private agents: AgentStatus[] = []
  @state() private events: TimelineEvent[] = []
  @state() private isLoading: boolean = true
  @state() private error: string = ''
  @state() private lastSync: Date | null = null
  @state() private selectedView: 'overview' | 'kanban' | 'agents' | 'events' = 'overview'
  
  private websocketService: WebSocketService
  private offlineService: OfflineService
  private notificationService: NotificationService
  
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
  `
  
  constructor() {
    super()
    this.websocketService = WebSocketService.getInstance()
    this.offlineService = OfflineService.getInstance()
    this.notificationService = NotificationService.getInstance()
    
    // Listen for offline state changes
    this.offlineService.addEventListener('online', () => {
      this.offline = false
      this.syncTasks()
    })
    
    this.offlineService.addEventListener('offline', () => {
      this.offline = true
    })
    
    // Listen for WebSocket events
    this.websocketService.addEventListener('task-updated', (event: CustomEvent) => {
      this.handleTaskUpdate(event.detail)
    })
    
    this.websocketService.addEventListener('task-created', (event: CustomEvent) => {
      this.handleTaskCreated(event.detail)
    })
    
    this.websocketService.addEventListener('task-deleted', (event: CustomEvent) => {
      this.handleTaskDeleted(event.detail)
    })
  }
  
  async connectedCallback() {
    super.connectedCallback()
    await this.loadTasks()
    
    // Connect WebSocket if online
    if (!this.offline) {
      await this.websocketService.connect()
    }
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    this.websocketService.disconnect()
  }
  
  private async loadTasks() {
    this.isLoading = true
    this.error = ''
    
    try {
      // Try to load from cache first
      const cachedTasks = await this.offlineService.getTasks()
      if (cachedTasks.length > 0) {
        this.tasks = cachedTasks
        this.isLoading = false
      }
      
      // If online, fetch fresh data
      if (!this.offline) {
        await this.syncAllData()
      }
      
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
      this.error = error instanceof Error ? error.message : 'Failed to load dashboard data'
    } finally {
      this.isLoading = false
    }
  }
  
  private async syncAllData() {
    await Promise.all([
      this.syncTasks(),
      this.syncAgents(),
      this.syncEvents()
    ])
  }
  
  private async syncAgents() {
    try {
      const response = await fetch('/api/v1/agents/status')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const agentData = await response.json()
      
      // Transform API data to AgentStatus format
      this.agents = agentData.map((agent: any) => ({
        id: agent.id,
        name: agent.name,
        status: agent.status,
        uptime: agent.uptime || 0,
        lastSeen: agent.lastSeen || new Date().toISOString(),
        currentTask: agent.currentTask,
        metrics: {
          cpuUsage: agent.metrics?.cpuUsage || [],
          memoryUsage: agent.metrics?.memoryUsage || [],
          tokenUsage: agent.metrics?.tokenUsage || [],
          tasksCompleted: agent.metrics?.tasksCompleted || [],
          errorRate: agent.metrics?.errorRate || [],
          responseTime: agent.metrics?.responseTime || [],
          timestamps: agent.metrics?.timestamps || []
        },
        performance: {
          score: agent.performance?.score || 85,
          trend: agent.performance?.trend || 'stable'
        }
      }))
      
    } catch (error) {
      console.error('Failed to sync agents:', error)
    }
  }
  
  private async syncEvents() {
    try {
      const response = await fetch('/api/v1/events?limit=100')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const eventData = await response.json()
      
      // Transform API data to TimelineEvent format
      this.events = eventData.map((event: any) => ({
        id: event.id,
        type: event.type,
        title: event.title,
        description: event.description,
        agent: event.agent,
        timestamp: event.timestamp,
        metadata: event.metadata,
        severity: event.severity || 'info'
      }))
      
    } catch (error) {
      console.error('Failed to sync events:', error)
    }
  }
  
  private async syncTasks() {
    try {
      const response = await fetch('/api/v1/tasks')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const freshTasks = await response.json()
      this.tasks = freshTasks
      this.lastSync = new Date()
      
      // Cache for offline use
      await this.offlineService.cacheTasks(freshTasks)
      
      // Clear any error state
      this.error = ''
      
    } catch (error) {
      console.error('Failed to sync tasks:', error)
      if (!this.offline) {
        this.error = error instanceof Error ? error.message : 'Failed to sync tasks'
      }
    }
  }
  
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
      
      // If online, sync immediately
      const response = await fetch(`/api/v1/tasks/${taskId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          status: newStatus,
          updatedAt: new Date().toISOString()
        })
      })
      
      if (!response.ok) {
        throw new Error(`Failed to update task: ${response.statusText}`)
      }
      
      const updatedTask = await response.json()
      
      // Update with server response
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
  
  private handleTaskUpdate(taskData: Task) {
    const index = this.tasks.findIndex(t => t.id === taskData.id)
    if (index >= 0) {
      const updated = [...this.tasks]
      updated[index] = { ...taskData, syncStatus: 'synced' }
      this.tasks = updated
    }
  }
  
  private handleTaskCreated(taskData: Task) {
    this.tasks = [...this.tasks, { ...taskData, syncStatus: 'synced' }]
  }
  
  private handleTaskDeleted(taskId: string) {
    this.tasks = this.tasks.filter(t => t.id !== taskId)
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
    await this.syncAllData()
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
    
    return {
      activeTasks,
      completedTasks,
      totalTasks: this.tasks.length,
      activeAgents,
      errorAgents,
      totalAgents: this.agents.length,
      recentEvents
    }
  }
  
  private renderOverviewView() {
    const summary = this.dashboardSummary
    
    return html`
      <div class="overview-summary">
        <div class="summary-card">
          <div class="summary-value">${summary.activeTasks}</div>
          <div class="summary-label">Active Tasks</div>
        </div>
        <div class="summary-card">
          <div class="summary-value">${summary.completedTasks}</div>
          <div class="summary-label">Completed</div>
        </div>
        <div class="summary-card">
          <div class="summary-value">${summary.activeAgents}</div>
          <div class="summary-label">Active Agents</div>
        </div>
        <div class="summary-card">
          <div class="summary-value">${summary.errorAgents}</div>
          <div class="summary-label">Agent Errors</div>
        </div>
        <div class="summary-card">
          <div class="summary-value">${summary.recentEvents}</div>
          <div class="summary-label">Recent Events</div>
        </div>
      </div>
      
      <div class="overview-panels">
        <div class="panel-section">
          <agent-health-panel 
            .agents=${this.agents}
            .compact=${true}
            @refresh-agents=${this.handleRefresh}
          ></agent-health-panel>
        </div>
        
        <div class="panel-section">
          <event-timeline
            .events=${this.events}
            .maxEvents=${20}
            .realtime=${!this.offline}
            .compact=${true}
          ></event-timeline>
        </div>
      </div>
    `
  }
  
  private renderKanbanView() {
    return html`
      <div style="height: calc(100vh - 140px); padding: 0;">
        <kanban-board
          .tasks=${this.tasks}
          .offline=${this.offline}
          @task-move=${this.handleTaskMove}
          @task-click=${this.handleTaskClick}
          @tasks-updated=${(e: CustomEvent) => {
            this.tasks = e.detail.tasks
          }}
        ></kanban-board>
      </div>
    `
  }
  
  private renderAgentsView() {
    return html`
      <div style="height: calc(100vh - 140px); padding: 1rem;">
        <agent-health-panel 
          .agents=${this.agents}
          .compact=${false}
          @refresh-agents=${this.handleRefresh}
          @agent-selected=${(e: CustomEvent) => {
            console.log('Agent selected:', e.detail.agent)
          }}
        ></agent-health-panel>
      </div>
    `
  }
  
  private renderEventsView() {
    return html`
      <div style="height: calc(100vh - 140px);">
        <event-timeline
          .events=${this.events}
          .maxEvents=${100}
          .realtime=${!this.offline}
          .compact=${false}
          @event-selected=${(e: CustomEvent) => {
            console.log('Event selected:', e.detail.event)
          }}
        ></event-timeline>
      </div>
    `
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
      default:
        return this.renderOverviewView()
    }
  }
  
  render() {
    if (this.isLoading && this.tasks.length === 0 && this.agents.length === 0) {
      return html`
        <div class="loading-state">
          <div class="spinner"></div>
          <p>Loading dashboard data...</p>
        </div>
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
            <div class="sync-status">
              <div class="sync-indicator ${this.offline ? 'offline' : ''}"></div>
              ${this.syncStatusText}
            </div>
            
            <button
              class="refresh-button ${this.isLoading ? 'spinning' : ''}"
              @click=${this.handleRefresh}
              ?disabled=${this.isLoading}
              title="Refresh data"
            >
              <svg class="refresh-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        </div>
        
        <div class="view-tabs">
          <button 
            class="tab-button ${this.selectedView === 'overview' ? 'active' : ''}"
            @click=${() => this.selectedView = 'overview'}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
            Overview
          </button>
          
          <button 
            class="tab-button ${this.selectedView === 'kanban' ? 'active' : ''}"
            @click=${() => this.selectedView = 'kanban'}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
            Tasks
          </button>
          
          <button 
            class="tab-button ${this.selectedView === 'agents' ? 'active' : ''}"
            @click=${() => this.selectedView = 'agents'}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Agents
          </button>
          
          <button 
            class="tab-button ${this.selectedView === 'events' ? 'active' : ''}"
            @click=${() => this.selectedView = 'events'}
          >
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Events
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
      
      <div class="dashboard-content">
        ${this.renderCurrentView()}
      </div>
    `
  }
}