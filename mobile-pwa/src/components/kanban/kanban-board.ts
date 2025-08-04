import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { repeat } from 'lit/directives/repeat.js'
import Sortable from 'sortablejs'
import { Task, TaskStatus, TaskPriority } from '../../types/task'
import { Agent, AgentRole } from '../../types/api'
import './kanban-column'
import './task-card'

@customElement('kanban-board')
export class KanbanBoard extends LitElement {
  @property({ type: Array }) declare tasks: Task[]
  @property({ type: Array }) declare agents: Agent[]
  @property({ type: Boolean }) declare offline: boolean
  @property({ type: String }) declare filter: string
  @property({ type: String }) declare agentFilter: string
  @property({ type: String }) declare priorityFilter: string
  @property({ type: String }) declare roleFilter: string
  @property({ type: Boolean }) declare showOnlyUnassigned: boolean
  
  @state() private declare draggedTask: Task | null
  @state() private declare isUpdating: boolean
  @state() private declare selectedTasks: Set<string>
  @state() private declare bulkActionPanel: boolean
  @state() private declare taskAnalytics: any
  
  private sortableInstances: Map<string, Sortable> = new Map()
  
  constructor() {
    super()
    
    // Initialize reactive properties
    this.tasks = []
    this.agents = []
    this.offline = false
    this.filter = ''
    this.agentFilter = ''
    this.priorityFilter = ''
    this.roleFilter = ''
    this.showOnlyUnassigned = false
    this.draggedTask = null
    this.isUpdating = false
    this.selectedTasks = new Set()
    this.bulkActionPanel = false
    this.taskAnalytics = null
  }
  
  static styles = css`
    :host {
      display: block;
      height: 100%;
      overflow: hidden;
    }
    
    .board-container {
      display: flex;
      height: 100%;
      gap: 1rem;
      padding: 1rem;
      overflow-x: auto;
      overflow-y: hidden;
      -webkit-overflow-scrolling: touch;
    }
    
    .board-filters {
      display: flex;
      gap: 0.5rem;
      padding: 1rem;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      border-bottom: 1px solid #e5e7eb;
      flex-wrap: wrap;
      align-items: center;
    }
    
    .advanced-filters {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      align-items: center;
    }
    
    .filter-group {
      display: flex;
      gap: 0.25rem;
      align-items: center;
    }
    
    .filter-label {
      font-size: 0.75rem;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }
    
    .filter-input {
      flex: 1;
      min-width: 200px;
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      font-size: 0.875rem;
    }
    
    .filter-select {
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      background: white;
      min-width: 120px;
    }
    
    .offline-indicator {
      background: #fef3c7;
      border: 1px solid #f59e0b;
      color: #92400e;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .offline-dot {
      width: 8px;
      height: 8px;
      background: #f59e0b;
      border-radius: 50%;
      animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    .updating-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(255, 255, 255, 0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10;
    }
    
    .spinner {
      width: 32px;
      height: 32px;
      border: 3px solid #e5e7eb;
      border-top: 3px solid #3b82f6;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    .bulk-actions-panel {
      background: linear-gradient(135deg, #fef3c7 0%, #f59e0b 100%);
      border: 1px solid #f59e0b;
      border-radius: 0.5rem;
      padding: 0.75rem;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
    }
    
    .bulk-actions-info {
      font-size: 0.875rem;
      font-weight: 600;
      color: #92400e;
    }
    
    .bulk-actions-buttons {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    
    .bulk-action-btn {
      padding: 0.375rem 0.75rem;
      font-size: 0.75rem;
      font-weight: 600;
      border: none;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }
    
    .bulk-action-btn.assign {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      color: white;
    }
    
    .bulk-action-btn.priority {
      background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
      color: white;
    }
    
    .bulk-action-btn.move {
      background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
      color: white;
    }
    
    .bulk-action-btn.clear {
      background: linear-gradient(135deg, #64748b 0%, #475569 100%);
      color: white;
    }
    
    .bulk-action-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .task-analytics-panel {
      background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
      border: 1px solid #0ea5e9;
      border-radius: 0.5rem;
      padding: 1rem;
      margin: 1rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }
    
    .analytics-metric {
      text-align: center;
      padding: 0.75rem;
      background: white;
      border-radius: 0.375rem;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .analytics-metric .metric-value {
      font-size: 1.875rem;
      font-weight: 700;
      color: #0f172a;
      display: block;
    }
    
    .analytics-metric .metric-label {
      font-size: 0.75rem;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.025em;
      margin-top: 0.25rem;
    }
    
    .analytics-metric .metric-trend {
      font-size: 0.75rem;
      font-weight: 500;
      margin-top: 0.25rem;
    }
    
    .analytics-metric .metric-trend.up {
      color: #10b981;
    }
    
    .analytics-metric .metric-trend.down {
      color: #ef4444;
    }
    
    .analytics-metric .metric-trend.neutral {
      color: #64748b;
    }
    
    .priority-badge {
      display: inline-block;
      padding: 0.125rem 0.375rem;
      font-size: 0.625rem;
      font-weight: 600;
      border-radius: 0.25rem;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }
    
    .priority-badge.high {
      background: #fecaca;
      color: #dc2626;
    }
    
    .priority-badge.medium {
      background: #fed7aa;
      color: #ea580c;
    }
    
    .priority-badge.low {
      background: #d1fae5;
      color: #16a34a;
    }
    
    .agent-assignment-indicator {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.75rem;
      font-weight: 500;
      color: #64748b;
    }
    
    .agent-avatar {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 0.625rem;
      font-weight: 600;
    }
    
    .unassigned-indicator {
      color: #f59e0b;
      font-weight: 600;
    }
    
    .toggle-switch {
      position: relative;
      display: inline-block;
      width: 44px;
      height: 24px;
    }
    
    .toggle-switch input {
      opacity: 0;
      width: 0;
      height: 0;
    }
    
    .toggle-slider {
      position: absolute;
      cursor: pointer;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: #cbd5e1;
      transition: .4s;
      border-radius: 24px;
    }
    
    .toggle-slider:before {
      position: absolute;
      content: "";
      height: 18px;
      width: 18px;
      left: 3px;
      bottom: 3px;
      background-color: white;
      transition: .4s;
      border-radius: 50%;
    }
    
    input:checked + .toggle-slider {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    input:checked + .toggle-slider:before {
      transform: translateX(20px);
    }
    
    @media (max-width: 768px) {
      .board-container {
        padding: 0.5rem;
        gap: 0.5rem;
      }
      
      .board-filters {
        padding: 0.5rem;
        gap: 0.25rem;
      }
      
      .filter-input,
      .filter-select {
        font-size: 16px; /* Prevents zoom on iOS */
      }
    }
  `
  
  private get columns() {
    return [
      { id: 'pending', title: 'Backlog', status: 'pending' as TaskStatus },
      { id: 'in-progress', title: 'In Progress', status: 'in-progress' as TaskStatus },
      { id: 'review', title: 'Review', status: 'review' as TaskStatus },
      { id: 'done', title: 'Done', status: 'done' as TaskStatus }
    ]
  }
  
  private get filteredTasks() {
    return this.tasks.filter(task => {
      const matchesSearch = !this.filter || 
        task.title.toLowerCase().includes(this.filter.toLowerCase()) ||
        task.description?.toLowerCase().includes(this.filter.toLowerCase())
      
      const matchesAgent = !this.agentFilter || task.agent === this.agentFilter
      
      const matchesPriority = !this.priorityFilter || task.priority === this.priorityFilter
      
      const matchesRole = !this.roleFilter || this.getAgentRole(task.agent) === this.roleFilter
      
      const matchesUnassigned = !this.showOnlyUnassigned || !task.agent
      
      return matchesSearch && matchesAgent && matchesPriority && matchesRole && matchesUnassigned
    })
  }
  
  private getAgentRole(agentId: string): AgentRole | null {
    const agent = this.agents.find(a => a.id === agentId)
    return agent?.role || null
  }
  
  private get taskAnalyticsData() {
    const tasks = this.filteredTasks
    const total = tasks.length
    const completed = tasks.filter(t => t.status === 'done').length
    const inProgress = tasks.filter(t => t.status === 'in-progress').length
    const pending = tasks.filter(t => t.status === 'pending').length
    const review = tasks.filter(t => t.status === 'review').length
    const unassigned = tasks.filter(t => !t.agent).length
    const highPriority = tasks.filter(t => t.priority === 'high').length
    const overdue = tasks.filter(t => t.dueDate && new Date(t.dueDate) < new Date()).length
    
    const completionRate = total > 0 ? Math.round((completed / total) * 100) : 0
    const assignmentRate = total > 0 ? Math.round(((total - unassigned) / total) * 100) : 0
    
    return {
      total,
      completed,
      inProgress,
      pending,
      review,
      unassigned,
      highPriority,
      overdue,
      completionRate,
      assignmentRate,
      averageTasksPerAgent: this.agents.length > 0 ? Math.round((total - unassigned) / this.agents.length) : 0
    }
  }
  
  private get uniqueAgents() {
    const agents = new Set(this.tasks.map(task => task.agent))
    return Array.from(agents).sort()
  }
  
  private getTasksForColumn(status: TaskStatus) {
    return this.filteredTasks.filter(task => task.status === status)
  }
  
  firstUpdated() {
    this.initializeSortable()
  }
  
  updated(changedProperties: Map<string | number | symbol, unknown>) {
    if (changedProperties.has('tasks')) {
      this.updateSortable()
    }
  }
  
  private initializeSortable() {
    this.columns.forEach(column => {
      const columnElement = this.shadowRoot?.querySelector(`[data-column="${column.status}"]`)
      if (columnElement) {
        const sortable = new Sortable(columnElement as HTMLElement, {
          group: 'kanban',
          animation: 150,
          ghostClass: 'ghost',
          dragClass: 'drag',
          onStart: (evt) => {
            const taskId = evt.item.getAttribute('data-task-id')
            this.draggedTask = this.tasks.find(t => t.id === taskId) || null
          },
          onEnd: async (evt) => {
            const taskId = evt.item.getAttribute('data-task-id')
            const newStatus = evt.to.getAttribute('data-column') as TaskStatus
            
            if (taskId && newStatus && this.draggedTask) {
              await this.handleTaskMove(taskId, newStatus, evt.newIndex || 0)
            }
            
            this.draggedTask = null
          }
        })
        
        this.sortableInstances.set(column.status, sortable)
      }
    })
  }
  
  private updateSortable() {
    // Refresh sortable instances with new task data
    this.sortableInstances.forEach(sortable => {
      sortable.destroy()
    })
    this.sortableInstances.clear()
    
    // Reinitialize after DOM update
    setTimeout(() => {
      this.initializeSortable()
    }, 0)
  }
  
  private async handleTaskMove(taskId: string, newStatus: TaskStatus, newIndex: number) {
    this.isUpdating = true
    
    try {
      const moveEvent = new CustomEvent('task-move', {
        detail: {
          taskId,
          newStatus,
          newIndex,
          offline: this.offline
        },
        bubbles: true,
        composed: true
      })
      
      this.dispatchEvent(moveEvent)
      
      // Optimistic update for better UX
      const taskIndex = this.tasks.findIndex(t => t.id === taskId)
      if (taskIndex >= 0) {
        const updatedTasks = [...this.tasks]
        updatedTasks[taskIndex] = {
          ...updatedTasks[taskIndex],
          status: newStatus,
          updatedAt: new Date().toISOString()
        }
        
        // Dispatch update event
        const updateEvent = new CustomEvent('tasks-updated', {
          detail: { tasks: updatedTasks },
          bubbles: true,
          composed: true
        })
        
        this.dispatchEvent(updateEvent)
      }
      
    } catch (error) {
      console.error('Failed to move task:', error)
      
      // Dispatch error event
      const errorEvent = new CustomEvent('task-move-error', {
        detail: { error, taskId, newStatus },
        bubbles: true,
        composed: true
      })
      
      this.dispatchEvent(errorEvent)
      
    } finally {
      this.isUpdating = false
    }
  }
  
  private handleFilterChange(e: Event) {
    this.filter = (e.target as HTMLInputElement).value
  }
  
  private handleAgentFilterChange(e: Event) {
    this.agentFilter = (e.target as HTMLSelectElement).value
  }
  
  private handlePriorityFilterChange(e: Event) {
    this.priorityFilter = (e.target as HTMLSelectElement).value
  }
  
  private handleRoleFilterChange(e: Event) {
    this.roleFilter = (e.target as HTMLSelectElement).value
  }
  
  private handleUnassignedToggle(e: Event) {
    this.showOnlyUnassigned = (e.target as HTMLInputElement).checked
  }
  
  private handleTaskSelection(taskId: string, selected: boolean) {
    if (selected) {
      this.selectedTasks.add(taskId)
    } else {
      this.selectedTasks.delete(taskId)
    }
    
    this.bulkActionPanel = this.selectedTasks.size > 0
    this.requestUpdate()
  }
  
  private handleSelectAllTasks() {
    const visibleTaskIds = this.filteredTasks.map(t => t.id)
    visibleTaskIds.forEach(id => this.selectedTasks.add(id))
    this.bulkActionPanel = true
    this.requestUpdate()
  }
  
  private handleClearSelection() {
    this.selectedTasks.clear()
    this.bulkActionPanel = false
    this.requestUpdate()
  }
  
  private async handleBulkAssignAgent(agentId: string) {
    const selectedTaskIds = Array.from(this.selectedTasks)
    
    const bulkAssignEvent = new CustomEvent('bulk-assign-agent', {
      detail: {
        taskIds: selectedTaskIds,
        agentId,
        offline: this.offline
      },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(bulkAssignEvent)
    this.handleClearSelection()
  }
  
  private async handleBulkChangePriority(priority: TaskPriority) {
    const selectedTaskIds = Array.from(this.selectedTasks)
    
    const bulkPriorityEvent = new CustomEvent('bulk-change-priority', {
      detail: {
        taskIds: selectedTaskIds,
        priority,
        offline: this.offline
      },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(bulkPriorityEvent)
    this.handleClearSelection()
  }
  
  private async handleBulkMoveStatus(status: TaskStatus) {
    const selectedTaskIds = Array.from(this.selectedTasks)
    
    const bulkMoveEvent = new CustomEvent('bulk-move-status', {
      detail: {
        taskIds: selectedTaskIds,
        status,
        offline: this.offline
      },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(bulkMoveEvent)
    this.handleClearSelection()
  }
  
  private toggleAnalytics() {
    this.taskAnalytics = this.taskAnalytics ? null : this.taskAnalyticsData
  }
  
  private handleTaskClick(task: Task) {
    const clickEvent = new CustomEvent('task-click', {
      detail: { task },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(clickEvent)
  }
  
  render() {
    const analytics = this.taskAnalyticsData
    
    return html`
      <div class="board-filters">
        <input
          type="search"
          class="filter-input"
          placeholder="Search tasks..."
          .value=${this.filter}
          @input=${this.handleFilterChange}
        />
        
        <div class="advanced-filters">
          <div class="filter-group">
            <span class="filter-label">Agent</span>
            <select
              class="filter-select"
              .value=${this.agentFilter}
              @change=${this.handleAgentFilterChange}
            >
              <option value="">All Agents</option>
              ${this.agents.map(agent => html`
                <option value=${agent.id}>${agent.name}</option>
              `)}
            </select>
          </div>
          
          <div class="filter-group">
            <span class="filter-label">Priority</span>
            <select
              class="filter-select"
              .value=${this.priorityFilter}
              @change=${this.handlePriorityFilterChange}
            >
              <option value="">All Priorities</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
          
          <div class="filter-group">
            <span class="filter-label">Role</span>
            <select
              class="filter-select"
              .value=${this.roleFilter}
              @change=${this.handleRoleFilterChange}
            >
              <option value="">All Roles</option>
              ${Object.values(AgentRole).map(role => html`
                <option value=${role}>${role.replace('_', ' ')}</option>
              `)}
            </select>
          </div>
          
          <div class="filter-group">
            <span class="filter-label">Unassigned</span>
            <label class="toggle-switch">
              <input
                type="checkbox"
                .checked=${this.showOnlyUnassigned}
                @change=${this.handleUnassignedToggle}
              />
              <span class="toggle-slider"></span>
            </label>
          </div>
          
          <button
            class="bulk-action-btn analytics"
            @click=${this.toggleAnalytics}
            style="background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%); color: white;"
          >
            📊 Analytics
          </button>
          
          ${this.selectedTasks.size > 0 ? html`
            <button
              class="bulk-action-btn"
              @click=${this.handleSelectAllTasks}
              style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;"
            >
              ✓ Select All (${this.filteredTasks.length})
            </button>
          ` : ''}
        </div>
        
        ${this.offline ? html`
          <div class="offline-indicator">
            <div class="offline-dot"></div>
            Offline Mode
          </div>
        ` : ''}
      </div>
      
      ${this.taskAnalytics ? html`
        <div class="task-analytics-panel">
          <div class="analytics-metric">
            <span class="metric-value">${analytics.total}</span>
            <span class="metric-label">Total Tasks</span>
          </div>
          <div class="analytics-metric">
            <span class="metric-value">${analytics.completionRate}%</span>
            <span class="metric-label">Completion Rate</span>
            <span class="metric-trend ${analytics.completionRate >= 70 ? 'up' : analytics.completionRate >= 40 ? 'neutral' : 'down'}">
              ${analytics.completionRate >= 70 ? '↗ Excellent' : analytics.completionRate >= 40 ? '→ Good' : '↘ Needs Attention'}
            </span>
          </div>
          <div class="analytics-metric">
            <span class="metric-value">${analytics.assignmentRate}%</span>
            <span class="metric-label">Assignment Rate</span>
            <span class="metric-trend ${analytics.assignmentRate >= 80 ? 'up' : analytics.assignmentRate >= 50 ? 'neutral' : 'down'}">
              ${analytics.assignmentRate >= 80 ? '↗ Well Assigned' : analytics.assignmentRate >= 50 ? '→ Moderate' : '↘ Many Unassigned'}
            </span>
          </div>
          <div class="analytics-metric">
            <span class="metric-value">${analytics.inProgress}</span>
            <span class="metric-label">In Progress</span>
          </div>
          <div class="analytics-metric">
            <span class="metric-value">${analytics.highPriority}</span>
            <span class="metric-label">High Priority</span>
            <span class="metric-trend ${analytics.highPriority > analytics.total * 0.3 ? 'down' : 'up'}">
              ${analytics.highPriority > analytics.total * 0.3 ? '↘ High Load' : '↗ Manageable'}
            </span>
          </div>
          <div class="analytics-metric">
            <span class="metric-value">${analytics.overdue}</span>
            <span class="metric-label">Overdue</span>
            <span class="metric-trend ${analytics.overdue === 0 ? 'up' : 'down'}">
              ${analytics.overdue === 0 ? '↗ On Track' : '↘ Attention Needed'}
            </span>
          </div>
        </div>
      ` : ''}
      
      ${this.bulkActionPanel ? html`
        <div class="bulk-actions-panel">
          <div class="bulk-actions-info">
            ${this.selectedTasks.size} task${this.selectedTasks.size > 1 ? 's' : ''} selected
          </div>
          <div class="bulk-actions-buttons">
            <select @change=${(e: Event) => {
              const agentId = (e.target as HTMLSelectElement).value
              if (agentId) {
                this.handleBulkAssignAgent(agentId)
                ;(e.target as HTMLSelectElement).value = ''
              }
            }}>
              <option value="">Assign to Agent...</option>
              ${this.agents.map(agent => html`
                <option value=${agent.id}>${agent.name} (${agent.role.replace('_', ' ')})</option>
              `)}
            </select>
            
            <button
              class="bulk-action-btn priority"
              @click=${() => this.handleBulkChangePriority('high' as TaskPriority)}
            >
              🔴 High Priority
            </button>
            
            <button
              class="bulk-action-btn priority"
              @click=${() => this.handleBulkChangePriority('medium' as TaskPriority)}
            >
              🟡 Medium Priority
            </button>
            
            <button
              class="bulk-action-btn move"
              @click=${() => this.handleBulkMoveStatus('in-progress' as TaskStatus)}
            >
              ▶️ Move to Progress
            </button>
            
            <button
              class="bulk-action-btn move"
              @click=${() => this.handleBulkMoveStatus('review' as TaskStatus)}
            >
              👀 Move to Review
            </button>
            
            <button
              class="bulk-action-btn clear"
              @click=${this.handleClearSelection}
            >
              ✖️ Clear Selection
            </button>
          </div>
        </div>
      ` : ''}
      
      <div class="board-container" style="position: relative;">
        ${this.isUpdating ? html`
          <div class="updating-overlay">
            <div class="spinner"></div>
          </div>
        ` : ''}
        
        ${this.columns.map(column => html`
          <kanban-column
            .title=${column.title}
            .status=${column.status}
            .tasks=${this.getTasksForColumn(column.status)}
            .agents=${this.agents}
            .selectedTasks=${this.selectedTasks}
            .offline=${this.offline}
            data-column=${column.status}
            @task-click=${(e: CustomEvent) => this.handleTaskClick(e.detail.task)}
            @task-select=${(e: CustomEvent) => this.handleTaskSelection(e.detail.taskId, e.detail.selected)}
          ></kanban-column>
        `)}
      </div>
    `
  }
}