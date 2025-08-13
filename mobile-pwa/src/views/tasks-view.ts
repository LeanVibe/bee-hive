import { LitElement, html, css } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { Task, TaskStatus } from '../types/task'
import { getTaskService, getAgentService } from '../services'
import type { TaskFilters } from '../services'
import type { Agent } from '../services'
import '../components/kanban/kanban-board'
import '../components/common/loading-spinner'
import '../components/modals/task-edit-modal'

@customElement('tasks-view')
export class TasksView extends LitElement {
  @state() private declare tasks: Task[]
  @state() private declare isLoading: boolean
  @state() private declare error: string
  @state() private declare viewMode: 'kanban' | 'list'
  @state() private declare filterStatus: TaskStatus | 'all'
  @state() private declare searchQuery: string
  @state() private declare taskService: any
  @state() private declare agentService: any
  @state() private declare monitoringActive: boolean
  @state() private declare showTaskModal: boolean
  @state() private declare taskModalMode: 'create' | 'edit'
  @state() private declare taskModalTask: Task | undefined
  @state() private declare availableAgents: Agent[]
  @state() private declare selectedTasks: Set<string>
  @state() private declare bulkActionMode: boolean
  @state() private declare draggedTask: Task | null
  @state() private declare draggedOverAgent: string | null

  constructor() {
    super()
    
    // Initialize reactive properties
    this.tasks = []
    this.isLoading = true
    this.error = ''
    this.viewMode = 'kanban'
    this.filterStatus = 'all'
    this.searchQuery = ''
    this.taskService = getTaskService()
    this.agentService = getAgentService()
    this.monitoringActive = false
    this.showTaskModal = false
    this.taskModalMode = 'create'
    this.taskModalTask = undefined
    this.availableAgents = []
    this.selectedTasks = new Set()
    this.bulkActionMode = false
    this.draggedTask = null
    this.draggedOverAgent = null
  }
  
  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: #f9fafb;
    }

    .tasks-container {
      display: flex;
      flex-direction: column;
      height: 100%;
      max-width: 1400px;
      margin: 0 auto;
    }

    .tasks-header {
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
      margin-bottom: 1.5rem;
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

    .view-toggle {
      display: flex;
      background: #f3f4f6;
      border-radius: 0.5rem;
      padding: 0.25rem;
    }

    .view-button {
      background: none;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      font-weight: 500;
      color: #6b7280;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .view-button.active {
      background: white;
      color: #3b82f6;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    .controls-bar {
      display: flex;
      gap: 1rem;
      align-items: center;
      flex-wrap: wrap;
    }

    .search-input {
      background: #f9fafb;
      border: 1px solid #d1d5db;
      border-radius: 0.5rem;
      padding: 0.5rem 0.75rem 0.5rem 2.5rem;
      font-size: 0.875rem;
      min-width: 250px;
      position: relative;
    }

    .search-container {
      position: relative;
    }

    .search-icon {
      position: absolute;
      left: 0.75rem;
      top: 50%;
      transform: translateY(-50%);
      width: 16px;
      height: 16px;
      color: #9ca3af;
    }

    .filter-select {
      background: white;
      border: 1px solid #d1d5db;
      border-radius: 0.5rem;
      padding: 0.5rem 0.75rem;
      font-size: 0.875rem;
      color: #374151;
      cursor: pointer;
      min-width: 120px;
    }

    .filter-select:focus {
      outline: none;
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .add-task-button {
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

    .add-task-button:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }

    .bulk-action-bar {
      background: #f0f9ff;
      border: 1px solid #bae6fd;
      border-radius: 0.75rem;
      padding: 1rem 1.5rem;
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
    }

    .bulk-info {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      color: #0369a1;
      font-weight: 500;
    }

    .bulk-actions {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }

    .action-button {
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

    .action-button:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }

    .action-button.secondary {
      background: white;
      color: #374151;
      border: 1px solid #d1d5db;
    }

    .action-button.secondary:hover {
      background: #f9fafb;
      border-color: #9ca3af;
    }

    .action-button.danger {
      background: #ef4444;
      color: white;
    }

    .action-button.danger:hover {
      background: #dc2626;
    }

    .action-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }

    .agent-assignment-panel {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.75rem;
      padding: 1rem;
      margin-bottom: 1.5rem;
    }

    .assignment-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .assignment-title {
      font-size: 1rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .agent-list-compact {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 0.75rem;
    }

    .agent-drop-zone {
      background: #f9fafb;
      border: 2px dashed #d1d5db;
      border-radius: 0.5rem;
      padding: 1rem;
      text-align: center;
      transition: all 0.2s ease;
      cursor: pointer;
      min-height: 80px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }

    .agent-drop-zone:hover {
      border-color: #9ca3af;
      background: #f3f4f6;
    }

    .agent-drop-zone.drag-over {
      border-color: #3b82f6;
      background: #eff6ff;
      border-style: solid;
    }

    .agent-name {
      font-size: 0.875rem;
      font-weight: 500;
      color: #374151;
      margin: 0;
    }

    .agent-role {
      font-size: 0.75rem;
      color: #6b7280;
      margin: 0;
      text-transform: capitalize;
    }

    .task-count {
      font-size: 0.75rem;
      color: #6b7280;
      background: #f3f4f6;
      padding: 0.25rem 0.5rem;
      border-radius: 9999px;
    }

    .drag-placeholder {
      border: 2px dashed #3b82f6;
      background: #eff6ff;
      border-radius: 0.5rem;
      padding: 1rem;
      margin: 0.5rem 0;
      text-align: center;
      color: #3b82f6;
      font-size: 0.875rem;
    }

    .task-checkbox {
      position: absolute;
      top: 0.75rem;
      left: 0.75rem;
      width: 16px;
      height: 16px;
      cursor: pointer;
      z-index: 2;
    }

    .task-item.bulk-selected {
      border-color: #10b981;
      box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
      background: #f0fdf4 !important;
    }

    .task-item[draggable="true"] {
      cursor: grab;
    }

    .task-item[draggable="true"]:active {
      cursor: grabbing;
    }

    .task-item.dragging {
      opacity: 0.5;
      transform: rotate(5deg);
    }

    .tasks-content {
      flex: 1;
      overflow: hidden;
    }

    .kanban-container {
      height: 100%;
      padding: 1rem;
    }

    .list-container {
      height: 100%;
      overflow-y: auto;
      padding: 1rem;
    }

    .task-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      max-width: 800px;
      margin: 0 auto;
    }

    .task-item {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      transition: all 0.2s ease;
      cursor: pointer;
    }

    .task-item:hover {
      border-color: #3b82f6;
      box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    }

    .task-header {
      display: flex;
      align-items: center;
      justify-content: between;
      margin-bottom: 0.5rem;
    }

    .task-title {
      font-size: 1rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
      flex: 1;
    }

    .task-status-badge {
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 500;
      text-transform: capitalize;
    }

    .task-status-badge.todo {
      background: #f3f4f6;
      color: #374151;
    }

    .task-status-badge.in-progress {
      background: #dbeafe;
      color: #1d4ed8;
    }

    .task-status-badge.review {
      background: #fef3c7;
      color: #d97706;
    }

    .task-status-badge.done {
      background: #d1fae5;
      color: #065f46;
    }

    .task-description {
      font-size: 0.875rem;
      color: #6b7280;
      margin: 0 0 0.75rem 0;
      line-height: 1.5;
    }

    .task-meta {
      display: flex;
      align-items: center;
      justify-content: between;
      gap: 1rem;
      font-size: 0.75rem;
      color: #9ca3af;
    }

    .task-assignee {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .task-date {
      margin-left: auto;
    }

    .tasks-summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .summary-card {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      text-align: center;
    }

    .summary-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin: 0 0 0.25rem 0;
    }

    .summary-label {
      font-size: 0.75rem;
      color: #6b7280;
      margin: 0;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }

    .summary-card.todo .summary-value {
      color: #6b7280;
    }

    .summary-card.in-progress .summary-value {
      color: #3b82f6;
    }

    .summary-card.review .summary-value {
      color: #f59e0b;
    }

    .summary-card.done .summary-value {
      color: #10b981;
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

    .empty-icon {
      width: 64px;
      height: 64px;
      color: #d1d5db;
      margin-bottom: 1rem;
    }

    .empty-title {
      font-size: 1.25rem;
      font-weight: 600;
      color: #374151;
      margin: 0 0 0.5rem 0;
    }

    .empty-description {
      font-size: 0.875rem;
      margin: 0 0 1.5rem 0;
      max-width: 400px;
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

    /* Responsive Design */
    @media (max-width: 768px) {
      .tasks-header {
        padding: 1rem;
      }

      .page-title {
        font-size: 1.5rem;
      }

      .header-content {
        flex-direction: column;
        align-items: flex-start;
      }

      .controls-bar {
        width: 100%;
        flex-direction: column;
        align-items: stretch;
      }

      .search-input {
        min-width: auto;
        width: 100%;
      }

      .tasks-summary {
        grid-template-columns: repeat(2, 1fr);
      }

      .task-meta {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
      }

      .task-date {
        margin-left: 0;
      }
    }

    @media (max-width: 640px) {
      .header-actions {
        width: 100%;
        justify-content: space-between;
      }

      .view-toggle {
        flex: 1;
      }

      .view-button {
        flex: 1;
        justify-content: center;
      }
    }
  `

  async connectedCallback() {
    super.connectedCallback()
    await this.initializeServices()
    await this.loadTasks()
    await this.loadAgents()
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    this.stopMonitoring()
  }
  
  /**
   * Initialize task and agent services with real-time monitoring
   */
  private async initializeServices() {
    try {
      // Set up event listeners for real-time updates
      this.taskService.addEventListener('tasksChanged', this.handleTasksChanged.bind(this))
      this.taskService.addEventListener('taskCreated', this.handleTaskCreated.bind(this))
      this.taskService.addEventListener('taskUpdated', this.handleTaskUpdated.bind(this))
      this.taskService.addEventListener('taskDeleted', this.handleTaskDeleted.bind(this))
      // Offline sync reconciliation indicators
      this.taskService.addEventListener('sync_queued', () => this.requestUpdate())
      this.taskService.addEventListener('task_synced', () => this.requestUpdate())
      this.taskService.addEventListener('sync_retry', () => this.requestUpdate())
      this.taskService.addEventListener('sync_failed', () => this.requestUpdate())
      
      // Start monitoring for real-time updates
      this.startMonitoring()
      
      console.log('Task service initialized successfully')
      
    } catch (error) {
      console.error('Failed to initialize task service:', error)
      this.error = 'Failed to initialize task service'
    }
  }

  /**
   * Load available agents for task assignment
   */
  private async loadAgents() {
    try {
      this.availableAgents = this.agentService.getAgents()
      console.log('Loaded', this.availableAgents.length, 'agents for task assignment')
    } catch (error) {
      console.error('Failed to load agents:', error)
    }
  }
  
  /**
   * Start real-time monitoring
   */
  private startMonitoring() {
    if (this.monitoringActive) return
    
    try {
      this.taskService.startMonitoring()
      this.monitoringActive = true
      console.log('Task monitoring started')
      
    } catch (error) {
      console.error('Failed to start task monitoring:', error)
    }
  }
  
  /**
   * Stop monitoring
   */
  private stopMonitoring() {
    if (!this.monitoringActive) return
    
    try {
      this.taskService.stopMonitoring()
      this.monitoringActive = false
      console.log('Task monitoring stopped')
      
    } catch (error) {
      console.error('Failed to stop task monitoring:', error)
    }
  }
  
  /**
   * Real-time event handlers
   */
  private handleTasksChanged = (event: CustomEvent) => {
    this.tasks = event.detail.tasks
    console.log('Tasks updated via real-time:', event.detail.tasks.length, 'tasks')
  }
  
  private handleTaskCreated = (event: CustomEvent) => {
    this.tasks = [...this.tasks, event.detail.task]
    console.log('Task created:', event.detail.task.title)
  }
  
  private handleTaskUpdated = (event: CustomEvent) => {
    const updatedTask = event.detail.task
    const index = this.tasks.findIndex(t => t.id === updatedTask.id)
    if (index >= 0) {
      const updatedTasks = [...this.tasks]
      updatedTasks[index] = updatedTask
      this.tasks = updatedTasks
    }
    console.log('Task updated:', updatedTask.title)
  }
  
  private handleTaskDeleted = (event: CustomEvent) => {
    const taskId = event.detail.taskId
    this.tasks = this.tasks.filter(t => t.id !== taskId)
    console.log('Task deleted:', taskId)
  }

  private async loadTasks() {
    this.isLoading = true
    this.error = ''

    try {
      // Load real tasks using integrated service
      const taskData = await this.taskService.getTasks()
      this.tasks = taskData
      
      console.log('Loaded', this.tasks.length, 'tasks from service')

    } catch (error) {
      console.error('Failed to load tasks:', error)
      this.error = error instanceof Error ? error.message : 'Failed to load tasks'
      
      // Fall back to mock data for demonstration
      this.tasks = [
        {
          id: 'task-001',
          title: 'Implement user authentication',
          description: 'Set up JWT-based authentication with login, logout, and token refresh functionality',
          status: 'in-progress',
          assignee: 'Developer Agent',
          agent: 'Developer Agent',
          createdAt: new Date(Date.now() - 86400000).toISOString(),
          updatedAt: new Date(Date.now() - 3600000).toISOString(),
          priority: 'high',
          tags: ['backend', 'security'],
          syncStatus: 'synced'
        },
        {
          id: 'task-002',
          title: 'Design database schema',
          description: 'Create comprehensive database schema for user management and application data',
          status: 'review',
          assignee: 'Architect Agent',
          agent: 'Architect Agent',
          createdAt: new Date(Date.now() - 172800000).toISOString(),
          updatedAt: new Date(Date.now() - 7200000).toISOString(),
          priority: 'high',
          tags: ['database', 'architecture'],
          syncStatus: 'synced'
        },
        {
          id: 'task-003',
          title: 'Write unit tests',
          description: 'Implement comprehensive unit tests for core business logic',
          status: 'todo',
          assignee: 'Tester Agent',
          agent: 'Tester Agent',
          createdAt: new Date(Date.now() - 259200000).toISOString(),
          updatedAt: new Date(Date.now() - 259200000).toISOString(),
          priority: 'medium',
          tags: ['testing', 'quality'],
          syncStatus: 'synced'
        },
        {
          id: 'task-004',
          title: 'API documentation',
          description: 'Create comprehensive API documentation with examples and best practices',
          status: 'done',
          assignee: 'Developer Agent',
          agent: 'Developer Agent',
          createdAt: new Date(Date.now() - 345600000).toISOString(),
          updatedAt: new Date(Date.now() - 86400000).toISOString(),
          priority: 'low',
          tags: ['documentation', 'api'],
          syncStatus: 'synced'
        }
      ]
    } finally {
      this.isLoading = false
    }
  }

  private get filteredTasks() {
    let filtered = this.tasks

    // Filter by status
    if (this.filterStatus !== 'all') {
      filtered = filtered.filter(task => task.status === this.filterStatus)
    }

    // Filter by search query
    if (this.searchQuery) {
      const query = this.searchQuery.toLowerCase()
      filtered = filtered.filter(task => 
        task.title.toLowerCase().includes(query) ||
        (task.description || '').toLowerCase().includes(query) ||
        task.assignee?.toLowerCase().includes(query)
      )
    }

    return filtered
  }

  private get tasksSummary() {
    const summary = {
      todo: 0,
      'in-progress': 0,
      review: 0,
      done: 0
    }

    this.tasks.forEach(task => {
      if (task.status in summary) {
        summary[task.status as keyof typeof summary]++
      }
    })

    return summary
  }

  private handleViewModeChange(mode: 'kanban' | 'list') {
    this.viewMode = mode
  }

  private handleFilterChange(event: Event) {
    const target = event.target as HTMLSelectElement
    this.filterStatus = target.value as TaskStatus | 'all'
  }

  private handleSearchChange(event: Event) {
    const target = event.target as HTMLInputElement
    this.searchQuery = target.value
  }

  private async handleTaskMove(event: CustomEvent) {
    const { taskId, newStatus } = event.detail
    
    try {
      // Update task using integrated service
      const updatedTask = await this.taskService.updateTask(taskId, {
        status: newStatus,
        updated_at: new Date()
      })
      
      // Update local state
      const index = this.tasks.findIndex(t => t.id === taskId)
      if (index >= 0) {
        const updatedTasks = [...this.tasks]
        updatedTasks[index] = updatedTask
        this.tasks = updatedTasks
      }
      
      console.log('Task moved via service:', taskId, 'to', newStatus)
      
    } catch (error) {
      console.error('Failed to move task:', error)
      this.error = error instanceof Error ? error.message : 'Failed to move task'
    }
  }

  private handleTaskClick(task: Task) {
    // Handle task click
    console.log('Task clicked:', task)
    
    // Dispatch event for parent components
    this.dispatchEvent(new CustomEvent('task-selected', {
      detail: { task },
      bubbles: true,
      composed: true
    }))
  }

  private async handleAddTask() {
    this.handleShowTaskModal('create')
  }

  /**
   * Show task modal for creation or editing
   */
  private handleShowTaskModal(mode: 'create' | 'edit', task?: Task) {
    this.taskModalMode = mode
    this.taskModalTask = task
    this.showTaskModal = true
  }

  /**
   * Handle task modal save
   */
  private async handleTaskModalSave(event: CustomEvent) {
    const { mode, data } = event.detail

    this.isLoading = true
    this.error = ''

    try {
      if (mode === 'create') {
        const newTask = await this.taskService.createTask({
          title: data.title,
          description: data.description,
          status: data.status,
          priority: data.priority,
          type: data.type,
          assignee: data.assignee,
          tags: data.tags,
          estimatedHours: data.estimatedHours,
          dependencies: data.dependencies,
          acceptanceCriteria: data.acceptanceCriteria,
          dueDate: data.dueDate,
          created_at: new Date(),
          updated_at: new Date()
        })
        
        this.tasks = [...this.tasks, newTask]
        console.log('New task created:', newTask.title)
      } else {
        const updatedTask = await this.taskService.updateTask(this.taskModalTask!.id, {
          ...data,
          updated_at: new Date()
        })
        
        const index = this.tasks.findIndex(t => t.id === updatedTask.id)
        if (index >= 0) {
          const updatedTasks = [...this.tasks]
          updatedTasks[index] = updatedTask
          this.tasks = updatedTasks
        }
        
        console.log('Task updated:', updatedTask.title)
      }

    } catch (error) {
      console.error('Failed to save task:', error)
      this.error = error instanceof Error ? error.message : 'Failed to save task'
    } finally {
      this.isLoading = false
    }
  }

  /**
   * Toggle bulk action mode
   */
  private handleToggleBulkMode() {
    this.bulkActionMode = !this.bulkActionMode
    if (!this.bulkActionMode) {
      this.selectedTasks.clear()
    }
    this.requestUpdate()
  }

  /**
   * Toggle task selection for bulk operations
   */
  private handleTaskSelection(taskId: string, event?: Event) {
    if (event) {
      event.stopPropagation()
    }

    if (this.selectedTasks.has(taskId)) {
      this.selectedTasks.delete(taskId)
    } else {
      this.selectedTasks.add(taskId)
    }
    this.requestUpdate()
  }

  /**
   * Select all tasks
   */
  private handleSelectAll() {
    if (this.selectedTasks.size === this.filteredTasks.length) {
      this.selectedTasks.clear()
    } else {
      this.selectedTasks.clear()
      this.filteredTasks.forEach(task => this.selectedTasks.add(task.id))
    }
    this.requestUpdate()
  }

  /**
   * Bulk assign tasks to agent
   */
  private async handleBulkAssign(agentId: string) {
    if (this.selectedTasks.size === 0) return

    this.isLoading = true
    this.error = ''

    try {
      const promises = Array.from(this.selectedTasks).map(taskId => 
        this.taskService.updateTask(taskId, {
          assignee: agentId,
          updated_at: new Date()
        })
      )
      
      const updatedTasks = await Promise.all(promises)
      
      // Update local state
      updatedTasks.forEach(updatedTask => {
        const index = this.tasks.findIndex(t => t.id === updatedTask.id)
        if (index >= 0) {
          this.tasks[index] = updatedTask
        }
      })
      
      console.log(`Bulk assigned ${this.selectedTasks.size} tasks to agent ${agentId}`)
      this.selectedTasks.clear()
      this.bulkActionMode = false

    } catch (error) {
      console.error('Failed to bulk assign tasks:', error)
      this.error = error instanceof Error ? error.message : 'Failed to bulk assign tasks'
    } finally {
      this.isLoading = false
    }
  }

  /**
   * Delete selected tasks
   */
  private async handleBulkDelete() {
    if (this.selectedTasks.size === 0) return

    this.isLoading = true
    this.error = ''

    try {
      const promises = Array.from(this.selectedTasks).map(taskId => 
        this.taskService.deleteTask(taskId)
      )
      
      await Promise.all(promises)
      
      // Update local state
      this.tasks = this.tasks.filter(task => !this.selectedTasks.has(task.id))
      
      console.log(`Deleted ${this.selectedTasks.size} tasks`)
      this.selectedTasks.clear()
      this.bulkActionMode = false

    } catch (error) {
      console.error('Failed to delete tasks:', error)
      this.error = error instanceof Error ? error.message : 'Failed to delete tasks'
    } finally {
      this.isLoading = false
    }
  }

  /**
   * Handle drag start
   */
  private handleDragStart(event: DragEvent, task: Task) {
    this.draggedTask = task
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = 'move'
      event.dataTransfer.setData('text/plain', task.id)
    }
    
    // Add dragging class
    const target = event.target as HTMLElement
    target.classList.add('dragging')
  }

  /**
   * Handle drag end
   */
  private handleDragEnd(event: DragEvent) {
    this.draggedTask = null
    this.draggedOverAgent = null
    
    // Remove dragging class
    const target = event.target as HTMLElement
    target.classList.remove('dragging')
  }

  /**
   * Handle drag over agent
   */
  private handleDragOver(event: DragEvent, agentId: string) {
    event.preventDefault()
    event.dataTransfer!.dropEffect = 'move'
    this.draggedOverAgent = agentId
  }

  /**
   * Handle drag leave agent
   */
  private handleDragLeave(event: DragEvent) {
    // Only clear if we're actually leaving the drop zone
    const related = event.relatedTarget as HTMLElement
    const currentTarget = event.currentTarget as HTMLElement
    
    if (!currentTarget.contains(related)) {
      this.draggedOverAgent = null
    }
  }

  /**
   * Handle drop on agent
   */
  private async handleDrop(event: DragEvent, agentId: string) {
    event.preventDefault()
    
    if (!this.draggedTask) return

    this.draggedOverAgent = null

    try {
      const updatedTask = await this.taskService.updateTask(this.draggedTask.id, {
        assignee: agentId,
        updated_at: new Date()
      })
      
      // Update local state
      const index = this.tasks.findIndex(t => t.id === updatedTask.id)
      if (index >= 0) {
        this.tasks[index] = updatedTask
      }
      
      console.log(`Task "${this.draggedTask.title}" assigned to agent ${agentId}`)
      
    } catch (error) {
      console.error('Failed to assign task:', error)
      this.error = error instanceof Error ? error.message : 'Failed to assign task'
    }

    this.draggedTask = null
  }

  private async handleRefresh() {
    await this.loadTasks()
  }

  private formatDate(dateString: string): string {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  private renderKanbanView() {
    return html`
      <div class="kanban-container">
        <kanban-board
          .tasks=${this.filteredTasks}
          .offline=${false}
          @task-move=${this.handleTaskMove}
          @task-click=${this.handleTaskClick}
        ></kanban-board>
      </div>
    `
  }

  private renderListView() {
    const tasks = this.filteredTasks

    if (tasks.length === 0) {
      return html`
        <div class="empty-state">
          <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
          </svg>
          <h2 class="empty-title">No Tasks Found</h2>
          <p class="empty-description">
            ${this.searchQuery || this.filterStatus !== 'all' 
              ? 'No tasks match your current filters. Try adjusting your search or filter criteria.'
              : 'No tasks have been created yet. Create your first task to get started.'}
          </p>
          <button class="add-task-button" @click=${this.handleAddTask}>
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
            </svg>
            Add Task
          </button>
        </div>
      `
    }

    return html`
      <div class="list-container">
        <div class="task-list">
          ${tasks.map(task => html`
            <div 
              class="task-item ${this.selectedTasks.has(task.id) ? 'bulk-selected' : ''}"
              draggable=${!this.bulkActionMode}
              @click=${() => this.bulkActionMode ? this.handleTaskSelection(task.id) : this.handleTaskClick(task)}
              @dragstart=${(e: DragEvent) => this.handleDragStart(e, task)}
              @dragend=${this.handleDragEnd}
            >
              ${this.bulkActionMode ? html`
                <input
                  type="checkbox"
                  class="task-checkbox"
                  .checked=${this.selectedTasks.has(task.id)}
                  @click=${(e: Event) => this.handleTaskSelection(task.id, e)}
                />
              ` : ''}

              <div class="task-header">
                <h3 class="task-title">${task.title}</h3>
                <span class="task-status-badge ${task.status}">${task.status}</span>
              </div>
              <p class="task-description">${task.description}</p>
              <div class="task-meta">
                <div class="task-assignee">
                  <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/>
                  </svg>
                  ${task.assignee || 'Unassigned'}
                </div>
                <div class="task-date">Updated ${this.formatDate(task.updatedAt)}</div>
                ${task.syncStatus === 'pending' ? html`
                  <span class="task-status-badge" style="background:#fef3c7;color:#92400e;">Pending Sync</span>
                ` : ''}
              </div>

              ${!this.bulkActionMode ? html`
                <button 
                  class="action-button secondary"
                  style="position: absolute; top: 0.75rem; right: 0.75rem; padding: 0.25rem 0.5rem; font-size: 0.75rem;"
                  @click=${(e: Event) => {
                    e.stopPropagation()
                    this.handleShowTaskModal('edit', task)
                  }}
                >
                  Edit
                </button>
              ` : ''}
            </div>
          `)}
        </div>
      </div>
    `
  }

  render() {
    if (this.isLoading && this.tasks.length === 0) {
      return html`
        <div class="loading-state">
          <loading-spinner size="large"></loading-spinner>
          <p>Loading tasks...</p>
        </div>
      `
    }

    if (this.error) {
      return html`
        <div class="error-state">
          <p><strong>Error:</strong> ${this.error}</p>
          <button class="add-task-button" @click=${this.handleRefresh}>
            Try Again
          </button>
        </div>
      `
    }

    const summary = this.tasksSummary

    return html`
      <div class="tasks-container">
        <div class="tasks-header">
          <div class="header-content">
            <h1 class="page-title">
              <svg class="title-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/>
              </svg>
              Task Management
            </h1>
            
            <div class="header-actions">
              <div class="view-toggle">
                <button 
                  class="view-button ${this.viewMode === 'kanban' ? 'active' : ''}"
                  @click=${() => this.handleViewModeChange('kanban')}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 0v10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2z"/>
                  </svg>
                  Kanban
                </button>
                <button 
                  class="view-button ${this.viewMode === 'list' ? 'active' : ''}"
                  @click=${() => this.handleViewModeChange('list')}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 10h16M4 14h16M4 18h16"/>
                  </svg>
                  List
                </button>
              </div>
              
              <button class="add-task-button" @click=${this.handleAddTask}>
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
                </svg>
                Add Task
              </button>

              <button 
                class="action-button secondary"
                @click=${this.handleToggleBulkMode}
                ?disabled=${this.isLoading || this.filteredTasks.length === 0}
              >
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                ${this.bulkActionMode ? 'Exit Bulk Mode' : 'Bulk Actions'}
              </button>
            </div>
          </div>

          <div class="tasks-summary">
            <div class="summary-card todo">
              <p class="summary-value">${summary.todo}</p>
              <p class="summary-label">To Do</p>
            </div>
            <div class="summary-card in-progress">
              <p class="summary-value">${summary['in-progress']}</p>
              <p class="summary-label">In Progress</p>
            </div>
            <div class="summary-card review">
              <p class="summary-value">${summary.review}</p>
              <p class="summary-label">Review</p>
            </div>
            <div class="summary-card done">
              <p class="summary-value">${summary.done}</p>
              <p class="summary-label">Done</p>
            </div>
          </div>

          ${this.bulkActionMode && this.filteredTasks.length > 0 ? html`
            <div class="bulk-action-bar">
              <div class="bulk-info">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                <span>${this.selectedTasks.size} of ${this.filteredTasks.length} tasks selected</span>
              </div>
              <div class="bulk-actions">
                <button 
                  class="action-button secondary"
                  @click=${this.handleSelectAll}
                  ?disabled=${this.isLoading}
                >
                  ${this.selectedTasks.size === this.filteredTasks.length ? 'Deselect All' : 'Select All'}
                </button>
                <button 
                  class="action-button danger"
                  @click=${this.handleBulkDelete}
                  ?disabled=${this.isLoading || this.selectedTasks.size === 0}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                  </svg>
                  Delete Selected
                </button>
              </div>
            </div>
          ` : ''}

          ${this.availableAgents.length > 0 && (this.bulkActionMode || this.draggedTask) ? html`
            <div class="agent-assignment-panel">
              <div class="assignment-header">
                <h3 class="assignment-title">
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"/>
                  </svg>
                  ${this.bulkActionMode ? 'Assign Selected Tasks' : 'Drag & Drop Assignment'}
                </h3>
              </div>
              <div class="agent-list-compact">
                ${this.availableAgents.map(agent => {
                  const assignedTasks = this.tasks.filter(t => t.assignee === agent.id).length
                  return html`
                    <div 
                      class="agent-drop-zone ${this.draggedOverAgent === agent.id ? 'drag-over' : ''}"
                      @click=${() => this.bulkActionMode && this.selectedTasks.size > 0 ? this.handleBulkAssign(agent.id) : null}
                      @dragover=${(e: DragEvent) => this.handleDragOver(e, agent.id)}
                      @dragleave=${this.handleDragLeave}
                      @drop=${(e: DragEvent) => this.handleDrop(e, agent.id)}
                    >
                      <p class="agent-name">${agent.name}</p>
                      <p class="agent-role">${agent.role.replace(/_/g, ' ')}</p>
                      <div class="task-count">${assignedTasks} tasks</div>
                    </div>
                  `
                })}
              </div>
            </div>
          ` : ''}

          <div class="controls-bar">
            <div class="search-container">
              <svg class="search-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
              </svg>
              <input
                class="search-input"
                type="text"
                placeholder="Search tasks..."
                .value=${this.searchQuery}
                @input=${this.handleSearchChange}
              />
            </div>
            
            <select 
              class="filter-select" 
              .value=${this.filterStatus}
              @change=${this.handleFilterChange}
            >
              <option value="all">All Status</option>
              <option value="todo">To Do</option>
              <option value="in-progress">In Progress</option>
              <option value="review">Review</option>
              <option value="done">Done</option>
            </select>
          </div>
        </div>

        <div class="tasks-content">
          ${this.viewMode === 'kanban' ? this.renderKanbanView() : this.renderListView()}
        </div>

        <!-- Task Edit Modal -->
        <task-edit-modal
          .open=${this.showTaskModal}
          .mode=${this.taskModalMode}
          .task=${this.taskModalTask}
          @close=${() => this.showTaskModal = false}
          @save=${this.handleTaskModalSave}
        ></task-edit-modal>
      </div>
    `
  }
}