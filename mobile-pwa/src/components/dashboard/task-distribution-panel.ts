import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

interface TaskInfo {
  id: string
  title: string
  priority: string // 'low', 'medium', 'high', 'critical'
  estimated_effort: number
  created_at: string
  assigned_agent?: string
  status?: string
}

interface TaskDistributionData {
  pending_tasks: TaskInfo[]
  assigned_tasks: { [agentId: string]: TaskInfo[] }
  failed_tasks: TaskInfo[]
  queue_depth: number
  average_wait_time: number
}

@customElement('task-distribution-panel')
export class TaskDistributionPanel extends LitElement {
  @property({ type: Boolean }) declare realtime: boolean
  @property({ type: Boolean }) declare enableDragDrop: boolean
  @property({ type: Array }) declare agents: any[]

  @state() private declare taskData: TaskDistributionData | null
  @state() private declare isLoading: boolean
  @state() private declare error: string
  @state() private declare lastUpdate: Date | null
  @state() private declare draggedTask: TaskInfo | null
  @state() private declare selectedTasks: Set<string>
  @state() private declare viewMode: string // 'queue', 'agents', 'failed'
  @state() private declare showBulkActions: boolean

  private updateInterval: number | null = null

  static styles = css`
    :host {
      display: block;
      background: white;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }

    .panel-header {
      background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
      color: white;
      padding: 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .panel-title {
      font-size: 1.125rem;
      font-weight: 600;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .queue-summary {
      display: flex;
      align-items: center;
      gap: 1rem;
      background: rgba(255, 255, 255, 0.1);
      padding: 0.5rem 1rem;
      border-radius: 8px;
    }

    .queue-metric {
      text-align: center;
    }

    .queue-value {
      font-size: 1.25rem;
      font-weight: 700;
      line-height: 1;
    }

    .queue-label {
      font-size: 0.75rem;
      opacity: 0.9;
    }

    .panel-content {
      padding: 1rem;
    }

    .view-tabs {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
      border-bottom: 1px solid #e5e7eb;
    }

    .tab-btn {
      padding: 0.75rem 1rem;
      border: none;
      background: none;
      cursor: pointer;
      font-size: 0.875rem;
      font-weight: 500;
      color: #6b7280;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
    }

    .tab-btn:hover {
      color: #374151;
    }

    .tab-btn.active {
      color: #7c3aed;
      border-bottom-color: #7c3aed;
    }

    .distribution-view {
      min-height: 400px;
    }

    .queue-view {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .agents-view {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 1rem;
    }

    .failed-view {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .task-item {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 1rem;
      cursor: grab;
      transition: all 0.2s;
      position: relative;
      user-select: none;
    }

    .task-item:hover {
      border-color: #7c3aed;
      transform: translateY(-1px);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .task-item.dragging {
      opacity: 0.5;
      cursor: grabbing;
      transform: rotate(5deg);
    }

    .task-item.selected {
      border-color: #3b82f6;
      background: #eff6ff;
    }

    .task-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }

    .task-title {
      font-weight: 600;
      color: #111827;
      margin: 0;
    }

    .task-priority {
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }

    .priority-low {
      background: #e0e7ff;
      color: #3730a3;
    }

    .priority-medium {
      background: #fef3c7;
      color: #92400e;
    }

    .priority-high {
      background: #fed7aa;
      color: #ea580c;
    }

    .priority-critical {
      background: #fee2e2;
      color: #991b1b;
      animation: pulse-critical 2s infinite;
    }

    .task-meta {
      display: flex;
      align-items: center;
      gap: 1rem;
      font-size: 0.875rem;
      color: #6b7280;
      margin-top: 0.5rem;
    }

    .task-effort {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }

    .task-age {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }

    .task-checkbox {
      position: absolute;
      top: 0.5rem;
      left: 0.5rem;
      width: 18px;
      height: 18px;
      cursor: pointer;
    }

    .agent-column {
      background: #f9fafb;
      border: 2px dashed #e5e7eb;
      border-radius: 8px;
      padding: 1rem;
      min-height: 200px;
      transition: all 0.2s;
    }

    .agent-column.drag-over {
      border-color: #7c3aed;
      background: #faf5ff;
    }

    .agent-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid #e5e7eb;
    }

    .agent-name {
      font-weight: 600;
      color: #111827;
    }

    .agent-load {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.875rem;
      color: #6b7280;
    }

    .load-indicator {
      width: 60px;
      height: 4px;
      background: #e5e7eb;
      border-radius: 2px;
      overflow: hidden;
    }

    .load-fill {
      height: 100%;
      background: #7c3aed;
      transition: width 0.3s;
    }

    .agent-tasks {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .drop-zone {
      border: 2px dashed #d1d5db;
      border-radius: 8px;
      padding: 2rem;
      text-align: center;
      color: #6b7280;
      background: #fafafa;
      transition: all 0.2s;
    }

    .drop-zone.active {
      border-color: #7c3aed;
      background: #faf5ff;
      color: #7c3aed;
    }

    .bulk-actions {
      position: sticky;
      bottom: 0;
      background: white;
      border-top: 1px solid #e5e7eb;
      padding: 1rem;
      display: flex;
      align-items: center;
      gap: 1rem;
      margin: -1rem -1rem 0;
      z-index: 10;
    }

    .bulk-actions-info {
      flex: 1;
      font-size: 0.875rem;
      color: #6b7280;
    }

    .bulk-action-btn {
      padding: 0.5rem 1rem;
      border: 1px solid #d1d5db;
      background: white;
      border-radius: 6px;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s;
    }

    .bulk-action-btn:hover {
      background: #f9fafb;
    }

    .bulk-action-btn.primary {
      background: #7c3aed;
      color: white;
      border-color: #7c3aed;
    }

    .bulk-action-btn.primary:hover {
      background: #6d28d9;
    }

    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 200px;
      color: #6b7280;
      gap: 1rem;
    }

    .loading-state {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 200px;
      color: #6b7280;
    }

    .error-state {
      background: #fef2f2;
      color: #dc2626;
      padding: 1rem;
      border-radius: 8px;
      text-align: center;
      margin: 1rem 0;
    }

    @keyframes pulse-critical {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; }
    }

    @media (max-width: 768px) {
      .agents-view {
        grid-template-columns: 1fr;
      }

      .queue-summary {
        flex-direction: column;
        gap: 0.5rem;
      }

      .bulk-actions {
        flex-direction: column;
        align-items: flex-start;
      }

      .bulk-action-btn {
        width: 100%;
      }
    }

    @media (hover: none) {
      .task-item {
        cursor: pointer;
      }
      
      .task-item:active {
        transform: scale(0.98);
      }
    }
  `

  constructor() {
    super()
    this.realtime = true
    this.enableDragDrop = true
    this.agents = []
    this.taskData = null
    this.isLoading = true
    this.error = ''
    this.lastUpdate = null
    this.draggedTask = null
    this.selectedTasks = new Set()
    this.viewMode = 'queue'
    this.showBulkActions = false
  }

  connectedCallback() {
    super.connectedCallback()
    this.loadData()
    
    if (this.realtime) {
      this.startRealtimeUpdates()
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.stopRealtimeUpdates()
  }

  private async loadData() {
    this.isLoading = true
    this.error = ''

    try {
      // Load task distribution data from coordination monitoring API
      const response = await fetch('/api/v1/coordination-monitoring/dashboard')
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const dashboardData = await response.json()
      
      this.taskData = dashboardData.task_distribution
      this.lastUpdate = new Date()

    } catch (error) {
      console.error('Failed to load task distribution data:', error)
      this.error = error instanceof Error ? error.message : 'Unknown error occurred'
    } finally {
      this.isLoading = false
    }
  }

  private startRealtimeUpdates() {
    // Update every 4 seconds for task distribution changes
    this.updateInterval = window.setInterval(() => {
      this.loadData()
    }, 4000)
  }

  private stopRealtimeUpdates() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
      this.updateInterval = null
    }
  }

  private async reassignTask(taskId: string, targetAgentId: string | null) {
    try {
      const endpoint = targetAgentId 
        ? `/api/v1/coordination-monitoring/task-distribution/reassign/${taskId}`
        : `/api/v1/coordination-monitoring/task-distribution/unassign/${taskId}`
        
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ target_agent_id: targetAgentId })
      })

      if (!response.ok) {
        throw new Error(`Reassignment failed: ${response.statusText}`)
      }

      const result = await response.json()
      console.log('Task reassignment completed:', result)

      // Refresh data after reassignment
      setTimeout(() => this.loadData(), 1000)

      // Dispatch event
      this.dispatchEvent(new CustomEvent('task-reassigned', {
        detail: { taskId, targetAgentId, result },
        bubbles: true,
        composed: true
      }))

    } catch (error) {
      console.error('Task reassignment failed:', error)
      this.error = `Reassignment failed: ${error}`
    }
  }

  private handleTaskSelect(taskId: string, selected: boolean) {
    if (selected) {
      this.selectedTasks.add(taskId)
    } else {
      this.selectedTasks.delete(taskId)
    }
    
    this.showBulkActions = this.selectedTasks.size > 0
    this.requestUpdate()
  }

  private async bulkReassign(targetAgentId: string) {
    try {
      const tasks = Array.from(this.selectedTasks)
      const promises = tasks.map(taskId => this.reassignTask(taskId, targetAgentId))
      
      await Promise.all(promises)
      
      this.selectedTasks.clear()
      this.showBulkActions = false
      
    } catch (error) {
      console.error('Bulk reassignment failed:', error)
      this.error = `Bulk reassignment failed: ${error}`
    }
  }

  private handleDragStart(e: DragEvent, task: TaskInfo) {
    if (!this.enableDragDrop) return
    
    this.draggedTask = task
    e.dataTransfer!.effectAllowed = 'move'
    e.dataTransfer!.setData('text/plain', task.id)
    
    const taskElement = e.target as HTMLElement
    taskElement.classList.add('dragging')
  }

  private handleDragEnd(e: DragEvent) {
    if (!this.enableDragDrop) return
    
    const taskElement = e.target as HTMLElement
    taskElement.classList.remove('dragging')
    this.draggedTask = null
  }

  private handleDragOver(e: DragEvent) {
    if (!this.enableDragDrop || !this.draggedTask) return
    
    e.preventDefault()
    e.dataTransfer!.dropEffect = 'move'
  }

  private handleDrop(e: DragEvent, targetAgentId: string | null) {
    if (!this.enableDragDrop || !this.draggedTask) return
    
    e.preventDefault()
    
    const taskId = this.draggedTask.id
    const currentAgent = this.draggedTask.assigned_agent
    
    if (currentAgent !== targetAgentId) {
      this.reassignTask(taskId, targetAgentId)
    }
  }

  private formatTaskAge(createdAt: string): string {
    const created = new Date(createdAt)
    const now = new Date()
    const diff = now.getTime() - created.getTime()
    
    if (diff < 60000) return 'Just created'
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m old`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h old`
    return `${Math.floor(diff / 86400000)}d old`
  }

  private getAgentLoad(agentId: string): number {
    if (!this.taskData) return 0
    
    const tasks = this.taskData.assigned_tasks[agentId] || []
    const totalEffort = tasks.reduce((sum, task) => sum + task.estimated_effort, 0)
    
    // Assume 8 hour workday (480 minutes) as 100% load
    return Math.min((totalEffort / 480) * 100, 100)
  }

  private renderTask(task: TaskInfo, showCheckbox = false) {
    const isSelected = this.selectedTasks.has(task.id)
    
    return html`
      <div 
        class="task-item ${isSelected ? 'selected' : ''}"
        draggable=${this.enableDragDrop}
        @dragstart=${(e: DragEvent) => this.handleDragStart(e, task)}
        @dragend=${this.handleDragEnd}
        @click=${(e: Event) => {
          if (showCheckbox && e.target !== e.currentTarget) return
          this.handleTaskSelect(task.id, !isSelected)
        }}
      >
        ${showCheckbox ? html`
          <input 
            type="checkbox" 
            class="task-checkbox"
            .checked=${isSelected}
            @change=${(e: Event) => {
              const checked = (e.target as HTMLInputElement).checked
              this.handleTaskSelect(task.id, checked)
            }}
          >
        ` : ''}

        <div class="task-header">
          <h4 class="task-title">${task.title}</h4>
          <div class="task-priority priority-${task.priority}">
            ${task.priority}
          </div>
        </div>

        <div class="task-meta">
          <div class="task-effort">
            <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            ${task.estimated_effort}min
          </div>
          <div class="task-age">
            <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            ${this.formatTaskAge(task.created_at)}
          </div>
        </div>
      </div>
    `
  }

  private renderQueueView() {
    if (!this.taskData) return html`<div class="loading-state">Loading...</div>`
    
    const { pending_tasks } = this.taskData
    
    if (pending_tasks.length === 0) {
      return html`
        <div class="empty-state">
          <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          <p>No tasks in queue</p>
        </div>
      `
    }
    
    return html`
      <div class="queue-view">
        ${pending_tasks.map(task => this.renderTask(task, true))}
      </div>
    `
  }

  private renderAgentsView() {
    if (!this.taskData) return html`<div class="loading-state">Loading...</div>`
    
    const { assigned_tasks, pending_tasks } = this.taskData
    const allAgents = [...this.agents, ...Object.keys(assigned_tasks)]
    const uniqueAgents = [...new Set(allAgents.map(a => typeof a === 'string' ? a : a.id))]
    
    return html`
      <div class="agents-view">
        <!-- Unassigned tasks column -->
        <div 
          class="agent-column"
          @dragover=${this.handleDragOver}
          @drop=${(e: DragEvent) => this.handleDrop(e, null)}
        >
          <div class="agent-header">
            <div class="agent-name">📥 Unassigned Queue</div>
            <div class="agent-load">
              ${pending_tasks.length} tasks
            </div>
          </div>
          
          <div class="agent-tasks">
            ${pending_tasks.length === 0 ? html`
              <div class="drop-zone">
                Drop tasks here to unassign
              </div>
            ` : pending_tasks.map(task => this.renderTask(task))}
          </div>
        </div>

        <!-- Agent columns -->
        ${uniqueAgents.map(agentId => {
          const agentTasks = assigned_tasks[agentId] || []
          const load = this.getAgentLoad(agentId)
          const agent = this.agents.find(a => a.id === agentId)
          const agentName = agent?.name || `Agent ${agentId.substring(0, 8)}`
          
          return html`
            <div 
              class="agent-column"
              @dragover=${this.handleDragOver}
              @drop=${(e: DragEvent) => this.handleDrop(e, agentId)}
            >
              <div class="agent-header">
                <div class="agent-name">${agentName}</div>
                <div class="agent-load">
                  <span>${load.toFixed(0)}%</span>
                  <div class="load-indicator">
                    <div class="load-fill" style="width: ${load}%"></div>
                  </div>
                </div>
              </div>
              
              <div class="agent-tasks">
                ${agentTasks.length === 0 ? html`
                  <div class="drop-zone">
                    Drop tasks here to assign
                  </div>
                ` : agentTasks.map(task => this.renderTask(task))}
              </div>
            </div>
          `
        })}
      </div>
    `
  }

  private renderFailedView() {
    if (!this.taskData) return html`<div class="loading-state">Loading...</div>`
    
    const { failed_tasks } = this.taskData
    
    if (failed_tasks.length === 0) {
      return html`
        <div class="empty-state">
          <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p>No failed tasks</p>
        </div>
      `
    }
    
    return html`
      <div class="failed-view">
        ${failed_tasks.map(task => this.renderTask(task, true))}
      </div>
    `
  }

  render() {
    if (this.isLoading && !this.taskData) {
      return html`
        <div class="loading-state">
          <div>Loading task distribution data...</div>
        </div>
      `
    }

    if (this.error && !this.taskData) {
      return html`
        <div class="error-state">
          <p><strong>Error:</strong> ${this.error}</p>
          <button class="bulk-action-btn" @click=${() => this.loadData()}>
            Retry
          </button>
        </div>
      `
    }

    const taskData = this.taskData!
    
    return html`
      <div class="panel-header">
        <div class="panel-title">
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M4 6h16M4 10h16M4 14h16M4 18h16" />
          </svg>
          Task Distribution & Queue
        </div>

        <div class="queue-summary">
          <div class="queue-metric">
            <div class="queue-value">${taskData.queue_depth}</div>
            <div class="queue-label">Pending</div>
          </div>
          <div class="queue-metric">
            <div class="queue-value">
              ${Object.values(taskData.assigned_tasks).reduce((sum, tasks) => sum + tasks.length, 0)}
            </div>
            <div class="queue-label">Assigned</div>
          </div>
          <div class="queue-metric">
            <div class="queue-value">${taskData.failed_tasks.length}</div>
            <div class="queue-label">Failed</div>
          </div>
          <div class="queue-metric">
            <div class="queue-value">${taskData.average_wait_time.toFixed(1)}s</div>
            <div class="queue-label">Avg Wait</div>
          </div>
        </div>
      </div>

      <div class="panel-content">
        ${this.error ? html`
          <div class="error-state" style="margin-bottom: 1rem;">
            <p>${this.error}</p>
          </div>
        ` : ''}

        <div class="view-tabs">
          <button 
            class="tab-btn ${this.viewMode === 'queue' ? 'active' : ''}"
            @click=${() => this.viewMode = 'queue'}
          >
            📋 Queue (${taskData.pending_tasks.length})
          </button>
          <button 
            class="tab-btn ${this.viewMode === 'agents' ? 'active' : ''}"
            @click=${() => this.viewMode = 'agents'}
          >
            👥 By Agent
          </button>
          <button 
            class="tab-btn ${this.viewMode === 'failed' ? 'active' : ''}"
            @click=${() => this.viewMode = 'failed'}
          >
            ❌ Failed (${taskData.failed_tasks.length})
          </button>
        </div>

        <div class="distribution-view">
          ${this.viewMode === 'queue' ? this.renderQueueView() : 
            this.viewMode === 'agents' ? this.renderAgentsView() : 
            this.renderFailedView()}
        </div>

        ${this.showBulkActions ? html`
          <div class="bulk-actions">
            <div class="bulk-actions-info">
              ${this.selectedTasks.size} task${this.selectedTasks.size !== 1 ? 's' : ''} selected
            </div>
            
            <select 
              class="bulk-action-btn"
              @change=${(e: Event) => {
                const select = e.target as HTMLSelectElement
                if (select.value) {
                  this.bulkReassign(select.value === 'unassign' ? null : select.value)
                  select.value = ''
                }
              }}
            >
              <option value="">Assign to...</option>
              <option value="unassign">🚫 Unassign</option>
              ${this.agents.map(agent => html`
                <option value="${agent.id}">${agent.name}</option>
              `)}
            </select>
            
            <button 
              class="bulk-action-btn"
              @click=${() => {
                this.selectedTasks.clear()
                this.showBulkActions = false
                this.requestUpdate()
              }}
            >
              Clear Selection
            </button>
          </div>
        ` : ''}

        ${this.lastUpdate ? html`
          <div style="font-size: 0.75rem; color: #6b7280; text-align: right; margin-top: 1rem;">
            Last updated: ${this.lastUpdate.toLocaleTimeString()}
            ${this.enableDragDrop ? ' • Drag & drop enabled' : ''}
          </div>
        ` : ''}
      </div>
    `
  }
}