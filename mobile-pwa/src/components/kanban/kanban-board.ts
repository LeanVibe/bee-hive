import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { repeat } from 'lit/directives/repeat.js'
import Sortable from 'sortablejs'
import { Task, TaskStatus } from '../../types/task'
import './kanban-column'
import './task-card'

@customElement('kanban-board')
export class KanbanBoard extends LitElement {
  @property({ type: Array }) tasks: Task[] = []
  @property({ type: Boolean }) offline: boolean = false
  @property({ type: String }) filter: string = ''
  @property({ type: String }) agentFilter: string = ''
  
  @state() private draggedTask: Task | null = null
  @state() private isUpdating: boolean = false
  
  private sortableInstances: Map<string, Sortable> = new Map()
  
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
      background: white;
      border-bottom: 1px solid #e5e7eb;
      flex-wrap: wrap;
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
      
      return matchesSearch && matchesAgent
    })
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
  
  private handleTaskClick(task: Task) {
    const clickEvent = new CustomEvent('task-click', {
      detail: { task },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(clickEvent)
  }
  
  render() {
    return html`
      <div class="board-filters">
        <input
          type="search"
          class="filter-input"
          placeholder="Search tasks..."
          .value=${this.filter}
          @input=${this.handleFilterChange}
        />
        
        <select
          class="filter-select"
          .value=${this.agentFilter}
          @change=${this.handleAgentFilterChange}
        >
          <option value="">All Agents</option>
          ${this.uniqueAgents.map(agent => html`
            <option value=${agent}>${agent}</option>
          `)}
        </select>
        
        ${this.offline ? html`
          <div class="offline-indicator">
            <div class="offline-dot"></div>
            Offline Mode
          </div>
        ` : ''}
      </div>
      
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
            .offline=${this.offline}
            data-column=${column.status}
            @task-click=${(e: CustomEvent) => this.handleTaskClick(e.detail.task)}
          ></kanban-column>
        `)}
      </div>
    `
  }
}