/**
 * Advanced Task Distribution Panel
 * 
 * Interactive drag-and-drop task assignment interface with visual workflow coordination
 * Priority: High - Essential for intuitive task management and agent coordination
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

export interface TaskItem {
  id: string
  title: string
  description: string
  status: 'pending' | 'in-progress' | 'review' | 'completed'
  priority: 'low' | 'medium' | 'high' | 'critical'
  assignedTo?: string
  assignedToName?: string
  estimatedHours?: number
  actualHours?: number
  tags: string[]
  dependencies?: string[]
  createdAt: string
  updatedAt: string
  dueDate?: string
}

export interface AgentInfo {
  id: string
  name: string
  status: 'available' | 'busy' | 'offline'
  currentLoad: number // 0-100
  capabilities: string[]
  maxCapacity: number
  currentTasks: number
  performanceScore: number
  specializations: string[]
}

export interface TaskAssignmentEvent {
  taskId: string
  fromAgent?: string
  toAgent?: string
  fromStatus: string
  toStatus: string
}

@customElement('advanced-task-distribution-panel')
export class AdvancedTaskDistributionPanel extends LitElement {
  @property({ type: Array }) declare tasks: TaskItem[]
  @property({ type: Array }) declare agents: AgentInfo[]
  @property({ type: Boolean }) declare realtime: boolean
  @property({ type: Boolean }) declare compact: boolean
  
  @state() private selectedView: 'kanban' | 'assignment' | 'workflow' = 'kanban'
  @state() private draggedTask: TaskItem | null = null
  @state() private draggedOverAgent: string | null = null
  @state() private selectedTask: TaskItem | null = null
  @state() private selectedAgent: AgentInfo | null = null
  @state() private showAssignmentModal: boolean = false
  @state() private filterPriority: string = 'all'
  @state() private filterStatus: string = 'all'
  @state() private searchQuery: string = ''
  @state() private autoAssignMode: boolean = false
  
  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: white;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }
    
    .distribution-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1rem;
      border-bottom: 1px solid #e5e7eb;
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      color: white;
    }
    
    .header-title {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 1.125rem;
      font-weight: 600;
    }
    
    .distribution-icon {
      width: 20px;
      height: 20px;
    }
    
    .header-controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .control-button {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: white;
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .control-button:hover {
      background: rgba(255, 255, 255, 0.2);
    }
    
    .control-button.active {
      background: rgba(255, 255, 255, 0.3);
      border-color: rgba(255, 255, 255, 0.4);
    }
    
    .auto-assign-toggle {
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #10b981;
      font-weight: 600;
    }
    
    .auto-assign-toggle.active {
      background: #10b981;
      color: white;
    }
    
    .distribution-content {
      height: calc(100% - 70px);
      overflow: hidden;
    }
    
    .view-tabs {
      display: flex;
      border-bottom: 1px solid #e5e7eb;
      background: #f9fafb;
      overflow-x: auto;
    }
    
    .tab-button {
      background: none;
      border: none;
      padding: 0.75rem 1rem;
      cursor: pointer;
      transition: all 0.2s;
      font-size: 0.875rem;
      color: #6b7280;
      white-space: nowrap;
      border-bottom: 2px solid transparent;
      display: flex;
      align-items: center;
      gap: 0.375rem;
    }
    
    .tab-button:hover {
      color: #374151;
      background: #f3f4f6;
    }
    
    .tab-button.active {
      color: #3b82f6;
      border-bottom-color: #3b82f6;
      background: white;
    }
    
    .distribution-panel {
      height: calc(100% - 50px);
      overflow-y: auto;
      padding: 1rem;
    }
    
    .filters-bar {
      display: flex;
      gap: 1rem;
      align-items: center;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }
    
    .search-input {
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
    }
    
    .kanban-board {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1rem;
      height: calc(100% - 80px);
    }
    
    .kanban-column {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      overflow: hidden;
    }
    
    .column-header {
      padding: 0.75rem;
      background: #f3f4f6;
      border-bottom: 1px solid #e5e7eb;
      font-weight: 600;
      color: #374151;
      display: flex;
      justify-content: between;
      align-items: center;
    }
    
    .column-count {
      background: #6b7280;
      color: white;
      border-radius: 50%;
      width: 20px;
      height: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
    }
    
    .column-content {
      padding: 0.5rem;
      min-height: 200px;
      max-height: calc(100vh - 400px);
      overflow-y: auto;
    }
    
    .column-content.drag-over {
      background: #dbeafe;
      border: 2px dashed #3b82f6;
    }
    
    .task-card {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.375rem;
      padding: 0.75rem;
      margin-bottom: 0.5rem;
      cursor: grab;
      transition: all 0.2s;
      position: relative;
    }
    
    .task-card:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .task-card.dragging {
      opacity: 0.5;
      transform: rotate(2deg);
    }
    
    .task-card.critical {
      border-left: 4px solid #dc2626;
    }
    
    .task-card.high {
      border-left: 4px solid #f59e0b;
    }
    
    .task-card.medium {
      border-left: 4px solid #3b82f6;
    }
    
    .task-card.low {
      border-left: 4px solid #10b981;
    }
    
    .task-title {
      font-size: 0.875rem;
      font-weight: 600;
      color: #374151;
      margin-bottom: 0.5rem;
      line-height: 1.3;
    }
    
    .task-description {
      font-size: 0.75rem;
      color: #6b7280;
      margin-bottom: 0.5rem;
      line-height: 1.4;
    }
    
    .task-meta {
      display: flex;
      justify-content: between;
      align-items: center;
      font-size: 0.625rem;
      color: #9ca3af;
    }
    
    .task-assignee {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      background: #f3f4f6;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
    }
    
    .task-tags {
      display: flex;
      gap: 0.25rem;
      flex-wrap: wrap;
      margin-top: 0.5rem;
    }
    
    .task-tag {
      background: #eff6ff;
      color: #3b82f6;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-size: 0.625rem;
      font-weight: 500;
    }
    
    .assignment-view {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2rem;
      height: calc(100% - 80px);
    }
    
    .agents-panel {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      overflow: hidden;
    }
    
    .panel-header {
      padding: 1rem;
      background: #f3f4f6;
      border-bottom: 1px solid #e5e7eb;
      font-weight: 600;
      color: #374151;
    }
    
    .panel-content {
      padding: 1rem;
      max-height: calc(100% - 60px);
      overflow-y: auto;
    }
    
    .agent-card {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.375rem;
      padding: 1rem;
      margin-bottom: 1rem;
      transition: all 0.2s;
      position: relative;
    }
    
    .agent-card:hover {
      border-color: #3b82f6;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .agent-card.drag-over {
      background: #dbeafe;
      border-color: #3b82f6;
      transform: scale(1.02);
    }
    
    .agent-card.available {
      border-left: 4px solid #10b981;
    }
    
    .agent-card.busy {
      border-left: 4px solid #f59e0b;
    }
    
    .agent-card.offline {
      border-left: 4px solid #6b7280;
      opacity: 0.6;
    }
    
    .agent-header {
      display: flex;
      justify-content: between;
      align-items: center;
      margin-bottom: 0.5rem;
    }
    
    .agent-name {
      font-size: 0.875rem;
      font-weight: 600;
      color: #374151;
    }
    
    .agent-status {
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-size: 0.625rem;
      font-weight: 500;
      text-transform: uppercase;
    }
    
    .agent-status.available {
      background: #dcfce7;
      color: #166534;
    }
    
    .agent-status.busy {
      background: #fef3c7;
      color: #92400e;
    }
    
    .agent-status.offline {
      background: #f3f4f6;
      color: #6b7280;
    }
    
    .agent-metrics {
      display: flex;
      gap: 1rem;
      margin-bottom: 0.5rem;
      font-size: 0.75rem;
      color: #6b7280;
    }
    
    .agent-load {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
    
    .load-bar {
      width: 50px;
      height: 4px;
      background: #e5e7eb;
      border-radius: 2px;
      overflow: hidden;
    }
    
    .load-fill {
      height: 100%;
      transition: width 0.3s ease;
      border-radius: 2px;
    }
    
    .load-fill.low {
      background: #10b981;
    }
    
    .load-fill.medium {
      background: #f59e0b;
    }
    
    .load-fill.high {
      background: #dc2626;
    }
    
    .agent-capabilities {
      display: flex;
      gap: 0.25rem;
      flex-wrap: wrap;
    }
    
    .capability-tag {
      background: #f0f9ff;
      color: #0369a1;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-size: 0.625rem;
    }
    
    .drop-zone {
      min-height: 60px;
      border: 2px dashed #d1d5db;
      border-radius: 0.375rem;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #9ca3af;
      font-size: 0.75rem;
      margin-top: 0.5rem;
      transition: all 0.2s;
    }
    
    .drop-zone.active {
      border-color: #3b82f6;
      background: #eff6ff;
      color: #3b82f6;
    }
    
    .assignment-modal {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 50;
    }
    
    .modal-content {
      background: white;
      border-radius: 0.5rem;
      padding: 2rem;
      width: 90%;
      max-width: 500px;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    .modal-header {
      display: flex;
      justify-content: between;
      align-items: center;
      margin-bottom: 1.5rem;
    }
    
    .modal-title {
      font-size: 1.125rem;
      font-weight: 600;
      color: #374151;
    }
    
    .close-button {
      background: none;
      border: none;
      padding: 0.5rem;
      cursor: pointer;
      color: #6b7280;
    }
    
    .close-button:hover {
      color: #374151;
    }
    
    .assignment-details {
      margin-bottom: 1.5rem;
    }
    
    .detail-row {
      display: flex;
      justify-content: between;
      align-items: center;
      padding: 0.5rem 0;
      border-bottom: 1px solid #f3f4f6;
    }
    
    .detail-label {
      font-weight: 500;
      color: #6b7280;
    }
    
    .detail-value {
      color: #374151;
    }
    
    .modal-actions {
      display: flex;
      gap: 1rem;
      justify-content: flex-end;
    }
    
    .btn {
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .btn-primary {
      background: #3b82f6;
      color: white;
      border: 1px solid #3b82f6;
    }
    
    .btn-primary:hover {
      background: #2563eb;
      border-color: #2563eb;
    }
    
    .btn-secondary {
      background: white;
      color: #374151;
      border: 1px solid #d1d5db;
    }
    
    .btn-secondary:hover {
      background: #f9fafb;
    }
    
    .empty-state {
      text-align: center;
      padding: 3rem 1rem;
      color: #6b7280;
    }
    
    .loading-state {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 200px;
      gap: 1rem;
    }
    
    .spinner {
      width: 20px;
      height: 20px;
      border: 2px solid #e5e7eb;
      border-top: 2px solid #3b82f6;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    @media (max-width: 768px) {
      .kanban-board {
        grid-template-columns: 1fr;
        overflow-x: auto;
      }
      
      .assignment-view {
        grid-template-columns: 1fr;
        gap: 1rem;
      }
      
      .filters-bar {
        flex-direction: column;
        align-items: stretch;
      }
      
      .distribution-header {
        flex-direction: column;
        gap: 0.5rem;
        align-items: stretch;
      }
    }
  `
  
  constructor() {
    super()
    this.tasks = []
    this.agents = []
    this.realtime = true
    this.compact = false
  }
  
  private get filteredTasks() {
    let filtered = this.tasks
    
    if (this.filterPriority !== 'all') {
      filtered = filtered.filter(task => task.priority === this.filterPriority)
    }
    
    if (this.filterStatus !== 'all') {
      filtered = filtered.filter(task => task.status === this.filterStatus)
    }
    
    if (this.searchQuery) {
      const query = this.searchQuery.toLowerCase()
      filtered = filtered.filter(task => 
        task.title.toLowerCase().includes(query) ||
        task.description.toLowerCase().includes(query) ||
        task.tags.some(tag => tag.toLowerCase().includes(query))
      )
    }
    
    return filtered
  }
  
  private getTasksByStatus(status: string) {
    return this.filteredTasks.filter(task => task.status === status)
  }
  
  private getAgentLoadClass(load: number) {
    if (load < 40) return 'low'
    if (load < 80) return 'medium'
    return 'high'
  }
  
  private handleDragStart(event: DragEvent, task: TaskItem) {
    this.draggedTask = task
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = 'move'
      event.dataTransfer.setData('text/plain', task.id)
    }
    
    // Add visual feedback
    const target = event.target as HTMLElement
    target.classList.add('dragging')
  }
  
  private handleDragEnd(event: DragEvent) {
    this.draggedTask = null
    this.draggedOverAgent = null
    
    // Remove visual feedback
    const target = event.target as HTMLElement
    target.classList.remove('dragging')
    
    // Remove drag-over classes from all elements
    this.shadowRoot?.querySelectorAll('.drag-over').forEach(el => {
      el.classList.remove('drag-over')
    })
  }
  
  private handleDragOver(event: DragEvent) {
    event.preventDefault()
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'move'
    }
  }
  
  private handleDragEnter(event: DragEvent, target: 'column' | 'agent', identifier?: string) {
    event.preventDefault()
    
    const element = event.currentTarget as HTMLElement
    element.classList.add('drag-over')
    
    if (target === 'agent' && identifier) {
      this.draggedOverAgent = identifier
    }
  }
  
  private handleDragLeave(event: DragEvent) {
    const element = event.currentTarget as HTMLElement
    element.classList.remove('drag-over')
    this.draggedOverAgent = null
  }
  
  private handleDrop(event: DragEvent, target: 'column' | 'agent', identifier: string) {
    event.preventDefault()
    
    const element = event.currentTarget as HTMLElement
    element.classList.remove('drag-over')
    
    if (!this.draggedTask) return
    
    let assignmentEvent: TaskAssignmentEvent
    
    if (target === 'column') {
      // Task moved to different status column
      assignmentEvent = {
        taskId: this.draggedTask.id,
        fromStatus: this.draggedTask.status,
        toStatus: identifier,
        fromAgent: this.draggedTask.assignedTo,
        toAgent: this.draggedTask.assignedTo
      }
    } else {
      // Task assigned to agent
      assignmentEvent = {
        taskId: this.draggedTask.id,
        fromAgent: this.draggedTask.assignedTo,
        toAgent: identifier,
        fromStatus: this.draggedTask.status,
        toStatus: this.draggedTask.status
      }
    }
    
    this.dispatchTaskAssignment(assignmentEvent)
    this.draggedTask = null
    this.draggedOverAgent = null
  }
  
  private dispatchTaskAssignment(assignment: TaskAssignmentEvent) {
    console.log('Task assignment:', assignment)
    
    const assignmentEvent = new CustomEvent('task-assigned', {
      detail: assignment,
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(assignmentEvent)
  }
  
  private handleTaskClick(task: TaskItem) {
    this.selectedTask = task
    this.showAssignmentModal = true
  }
  
  private handleAgentClick(agent: AgentInfo) {
    this.selectedAgent = agent
    console.log('Agent selected:', agent)
    
    const agentEvent = new CustomEvent('agent-selected', {
      detail: { agent },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(agentEvent)
  }
  
  private closeAssignmentModal() {
    this.showAssignmentModal = false
    this.selectedTask = null
  }
  
  private confirmAssignment() {
    if (this.selectedTask && this.selectedAgent) {
      const assignment: TaskAssignmentEvent = {
        taskId: this.selectedTask.id,
        fromAgent: this.selectedTask.assignedTo,
        toAgent: this.selectedAgent.id,
        fromStatus: this.selectedTask.status,
        toStatus: this.selectedTask.status
      }
      
      this.dispatchTaskAssignment(assignment)
      this.closeAssignmentModal()
    }
  }
  
  private toggleAutoAssign() {
    this.autoAssignMode = !this.autoAssignMode
    
    const autoAssignEvent = new CustomEvent('auto-assign-toggled', {
      detail: { enabled: this.autoAssignMode },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(autoAssignEvent)
  }
  
  private handleViewChange(view: 'kanban' | 'assignment' | 'workflow') {
    this.selectedView = view
    
    const viewEvent = new CustomEvent('view-changed', {
      detail: { view },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(viewEvent)
  }
  
  private handleFilterChange(event: Event, type: 'priority' | 'status' | 'search') {
    const target = event.target as HTMLInputElement | HTMLSelectElement
    
    switch (type) {
      case 'priority':
        this.filterPriority = target.value
        break
      case 'status':
        this.filterStatus = target.value
        break
      case 'search':
        this.searchQuery = target.value
        break
    }
  }
  
  private renderTaskCard(task: TaskItem) {
    return html`
      <div 
        class=\"task-card ${task.priority}\"
        draggable=\"true\"
        @dragstart=${(e: DragEvent) => this.handleDragStart(e, task)}
        @dragend=${this.handleDragEnd}
        @click=${() => this.handleTaskClick(task)}
      >
        <div class=\"task-title\">${task.title}</div>
        <div class=\"task-description\">${task.description}</div>
        <div class=\"task-meta\">
          <div class=\"task-assignee\">
            ${task.assignedToName ? html`
              <svg width=\"12\" height=\"12\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\">
                <path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z\"/>
              </svg>
              ${task.assignedToName}
            ` : 'Unassigned'}
          </div>
          <div>${task.priority.toUpperCase()}</div>
        </div>
        ${task.tags.length > 0 ? html`
          <div class=\"task-tags\">
            ${task.tags.map(tag => html`<span class=\"task-tag\">${tag}</span>`)}
          </div>
        ` : ''}
      </div>
    `
  }
  
  private renderKanbanView() {
    const statuses = [
      { key: 'pending', title: 'Pending', tasks: this.getTasksByStatus('pending') },
      { key: 'in-progress', title: 'In Progress', tasks: this.getTasksByStatus('in-progress') },
      { key: 'review', title: 'Review', tasks: this.getTasksByStatus('review') },
      { key: 'completed', title: 'Completed', tasks: this.getTasksByStatus('completed') }
    ]
    
    return html`
      <div class=\"kanban-board\">
        ${statuses.map(status => html`
          <div class=\"kanban-column\">
            <div class=\"column-header\">
              <span>${status.title}</span>
              <span class=\"column-count\">${status.tasks.length}</span>
            </div>
            <div 
              class=\"column-content\"
              @dragover=${this.handleDragOver}
              @dragenter=${(e: DragEvent) => this.handleDragEnter(e, 'column')}
              @dragleave=${this.handleDragLeave}
              @drop=${(e: DragEvent) => this.handleDrop(e, 'column', status.key)}
            >
              ${status.tasks.map(task => this.renderTaskCard(task))}
              ${status.tasks.length === 0 ? html`
                <div class=\"drop-zone ${this.draggedTask ? 'active' : ''}\">
                  Drop tasks here
                </div>
              ` : ''}
            </div>
          </div>
        `)}
      </div>
    `
  }
  
  private renderAssignmentView() {
    return html`
      <div class=\"assignment-view\">
        <div class=\"agents-panel\">
          <div class=\"panel-header\">Available Agents</div>
          <div class=\"panel-content\">
            ${this.agents.map(agent => html`
              <div 
                class=\"agent-card ${agent.status}\"
                @click=${() => this.handleAgentClick(agent)}
                @dragover=${this.handleDragOver}
                @dragenter=${(e: DragEvent) => this.handleDragEnter(e, 'agent', agent.id)}
                @dragleave=${this.handleDragLeave}
                @drop=${(e: DragEvent) => this.handleDrop(e, 'agent', agent.id)}
              >
                <div class=\"agent-header\">
                  <div class=\"agent-name\">${agent.name}</div>
                  <div class=\"agent-status ${agent.status}\">${agent.status}</div>
                </div>
                <div class=\"agent-metrics\">
                  <div class=\"agent-load\">
                    Load: ${agent.currentLoad}%
                    <div class=\"load-bar\">
                      <div class=\"load-fill ${this.getAgentLoadClass(agent.currentLoad)}\" style=\"width: ${agent.currentLoad}%\"></div>
                    </div>
                  </div>
                  <div>Tasks: ${agent.currentTasks}/${agent.maxCapacity}</div>
                  <div>Score: ${agent.performanceScore}%</div>
                </div>
                <div class=\"agent-capabilities\">
                  ${agent.capabilities.map(cap => html`<span class=\"capability-tag\">${cap}</span>`)}
                </div>
                <div class=\"drop-zone ${this.draggedOverAgent === agent.id ? 'active' : ''}\">
                  Drop task to assign
                </div>
              </div>
            `)}
          </div>
        </div>
        
        <div class=\"agents-panel\">
          <div class=\"panel-header\">Unassigned Tasks</div>
          <div class=\"panel-content\">
            ${this.filteredTasks.filter(task => !task.assignedTo).map(task => this.renderTaskCard(task))}
            ${this.filteredTasks.filter(task => !task.assignedTo).length === 0 ? html`
              <div class=\"empty-state\">
                <p>No unassigned tasks</p>
                <small>All tasks are currently assigned to agents</small>
              </div>
            ` : ''}
          </div>
        </div>
      </div>
    `
  }
  
  private renderCurrentView() {
    switch (this.selectedView) {
      case 'assignment':
        return this.renderAssignmentView()
      case 'workflow':
        return html`<div class=\"empty-state\">Workflow view coming soon...</div>`
      default:
        return this.renderKanbanView()
    }
  }
  
  render() {
    return html`
      <div class=\"distribution-header\">
        <div class=\"header-title\">
          <svg class=\"distribution-icon\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\">
            <path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M4 6h16M4 10h16M4 14h16M4 18h16\"/>
          </svg>
          Task Distribution
        </div>
        <div class=\"header-controls\">
          <button 
            class=\"auto-assign-toggle ${this.autoAssignMode ? 'active' : ''}\"
            @click=${this.toggleAutoAssign}
            title=\"${this.autoAssignMode ? 'Disable' : 'Enable'} automatic task assignment\"
          >
            ${this.autoAssignMode ? '🤖 Auto-Assign ON' : '🤖 Auto-Assign OFF'}
          </button>
          <button class=\"control-button\">
            ${this.realtime ? '🔴 Live' : '⏸️ Paused'}
          </button>
        </div>
      </div>
      
      <div class=\"distribution-content\">
        <div class=\"view-tabs\">
          <button 
            class=\"tab-button ${this.selectedView === 'kanban' ? 'active' : ''}\"
            @click=${() => this.handleViewChange('kanban')}
          >
            <svg width=\"16\" height=\"16\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\">
              <path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2\"/>
            </svg>
            Kanban Board
          </button>
          <button 
            class=\"tab-button ${this.selectedView === 'assignment' ? 'active' : ''}\"
            @click=${() => this.handleViewChange('assignment')}
          >
            <svg width=\"16\" height=\"16\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\">
              <path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z\"/>
            </svg>
            Agent Assignment
          </button>
          <button 
            class=\"tab-button ${this.selectedView === 'workflow' ? 'active' : ''}\"
            @click=${() => this.handleViewChange('workflow')}
          >
            <svg width=\"16\" height=\"16\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\">
              <path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M13 10V3L4 14h7v7l9-11h-7z\"/>
            </svg>
            Workflow
          </button>
        </div>
        
        <div class=\"distribution-panel\">
          <div class=\"filters-bar\">
            <input 
              class=\"search-input\" 
              type=\"text\" 
              placeholder=\"Search tasks...\"
              .value=${this.searchQuery}
              @input=${(e: Event) => this.handleFilterChange(e, 'search')}
            />
            <select class=\"filter-select\" .value=${this.filterPriority} @change=${(e: Event) => this.handleFilterChange(e, 'priority')}>
              <option value=\"all\">All Priorities</option>
              <option value=\"critical\">Critical</option>
              <option value=\"high\">High</option>
              <option value=\"medium\">Medium</option>
              <option value=\"low\">Low</option>
            </select>
            <select class=\"filter-select\" .value=${this.filterStatus} @change=${(e: Event) => this.handleFilterChange(e, 'status')}>
              <option value=\"all\">All Statuses</option>
              <option value=\"pending\">Pending</option>
              <option value=\"in-progress\">In Progress</option>
              <option value=\"review\">Review</option>
              <option value=\"completed\">Completed</option>
            </select>
          </div>
          
          ${this.renderCurrentView()}
        </div>
      </div>
      
      ${this.showAssignmentModal ? html`
        <div class=\"assignment-modal\" @click=${(e: Event) => e.target === e.currentTarget && this.closeAssignmentModal()}>
          <div class=\"modal-content\">
            <div class=\"modal-header\">
              <h3 class=\"modal-title\">Task Assignment</h3>
              <button class=\"close-button\" @click=${this.closeAssignmentModal}>
                <svg width=\"16\" height=\"16\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\">
                  <path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M6 18L18 6M6 6l12 12\"/>
                </svg>
              </button>
            </div>
            
            ${this.selectedTask ? html`
              <div class=\"assignment-details\">
                <div class=\"detail-row\">
                  <span class=\"detail-label\">Task:</span>
                  <span class=\"detail-value\">${this.selectedTask.title}</span>
                </div>
                <div class=\"detail-row\">
                  <span class=\"detail-label\">Priority:</span>
                  <span class=\"detail-value\">${this.selectedTask.priority.toUpperCase()}</span>
                </div>
                <div class=\"detail-row\">
                  <span class=\"detail-label\">Current Assignee:</span>
                  <span class=\"detail-value\">${this.selectedTask.assignedToName || 'Unassigned'}</span>
                </div>
                ${this.selectedAgent ? html`
                  <div class=\"detail-row\">
                    <span class=\"detail-label\">New Assignee:</span>
                    <span class=\"detail-value\">${this.selectedAgent.name}</span>
                  </div>
                ` : ''}
              </div>
            ` : ''}
            
            <div class=\"modal-actions\">
              <button class=\"btn btn-secondary\" @click=${this.closeAssignmentModal}>Cancel</button>
              <button class=\"btn btn-primary\" @click=${this.confirmAssignment} ?disabled=${!this.selectedAgent}>
                Assign Task
              </button>
            </div>
          </div>
        </div>
      ` : ''}
    `
  }
}