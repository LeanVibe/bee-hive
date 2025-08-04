import { LitElement, html, css } from 'lit'
import { customElement, property } from 'lit/decorators.js'
import { repeat } from 'lit/directives/repeat.js'
import { Task, TaskStatus } from '../../types/task'

@customElement('kanban-column')
export class KanbanColumn extends LitElement {
  @property({ type: String }) declare title: string
  @property({ type: String }) declare status: TaskStatus
  @property({ type: Array }) declare tasks: Task[]
  @property({ type: Boolean }) declare offline: boolean
  
  constructor() {
    super()
    
    // Initialize reactive properties
    this.title = ''
    this.status = 'pending'
    this.tasks = []
    this.offline = false
  }
  
  static styles = css`
    :host {
      display: flex;
      flex-direction: column;
      min-width: 280px;
      max-width: 320px;
      height: 100%;
      background: #f9fafb;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
    }
    
    .column-header {
      padding: 1rem;
      border-bottom: 1px solid #e5e7eb;
      background: white;
      border-radius: 0.5rem 0.5rem 0 0;
      position: relative;
    }
    
    .column-title {
      font-weight: 600;
      font-size: 0.875rem;
      color: #374151;
      margin: 0 0 0.25rem 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .task-count {
      background: #e5e7eb;
      color: #6b7280;
      font-size: 0.75rem;
      padding: 0.125rem 0.375rem;
      border-radius: 0.75rem;
      font-weight: 500;
    }
    
    .column-tasks {
      flex: 1;
      padding: 1rem;
      overflow-y: auto;
      min-height: 200px;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2rem 1rem;
      color: #9ca3af;
      font-size: 0.875rem;
      text-align: center;
    }
    
    .empty-icon {
      width: 48px;
      height: 48px;
      margin-bottom: 0.75rem;
      opacity: 0.5;
    }
    
    .drop-zone {
      min-height: 100px;
      border: 2px dashed transparent;
      border-radius: 0.375rem;
      transition: all 0.2s;
    }
    
    .drop-zone.drag-over {
      border-color: #3b82f6;
      background: rgba(59, 130, 246, 0.05);
    }
    
    /* Status-specific styling */
    :host([status="pending"]) .column-header {
      border-left: 4px solid #6b7280;
    }
    
    :host([status="in-progress"]) .column-header {
      border-left: 4px solid #f59e0b;
    }
    
    :host([status="review"]) .column-header {
      border-left: 4px solid #3b82f6;
    }
    
    :host([status="done"]) .column-header {
      border-left: 4px solid #10b981;
    }
    
    :host([status="pending"]) .task-count {
      background: #f3f4f6;
      color: #6b7280;
    }
    
    :host([status="in-progress"]) .task-count {
      background: #fef3c7;
      color: #92400e;
    }
    
    :host([status="review"]) .task-count {
      background: #dbeafe;
      color: #1e40af;
    }
    
    :host([status="done"]) .task-count {
      background: #d1fae5;
      color: #065f46;
    }
    
    /* Sortable styling */
    :host(.sortable-ghost) {
      opacity: 0.4;
    }
    
    :host(.sortable-drag) {
      transform: rotate(5deg);
    }
    
    @media (max-width: 768px) {
      :host {
        min-width: 260px;
        max-width: 300px;
      }
      
      .column-header {
        padding: 0.75rem;
      }
      
      .column-tasks {
        padding: 0.75rem;
        gap: 0.5rem;
      }
    }
  `
  
  private get statusIcon() {
    switch (this.status) {
      case 'pending':
        return html`
          <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        `
      case 'in-progress':
        return html`
          <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        `
      case 'review':
        return html`
          <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
        `
      case 'done':
        return html`
          <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        `
      default:
        return html``
    }
  }
  
  private get emptyStateContent() {
    switch (this.status) {
      case 'pending':
        return {
          icon: html`
            <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
          `,
          message: 'No tasks in backlog'
        }
      case 'in-progress':
        return {
          icon: html`
            <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          `,
          message: 'No active tasks'
        }
      case 'review':
        return {
          icon: html`
            <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          `,
          message: 'No tasks under review'
        }
      case 'done':
        return {
          icon: html`
            <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          `,
          message: 'No completed tasks'
        }
      default:
        return {
          icon: html``,
          message: 'No tasks'
        }
    }
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
    const emptyState = this.emptyStateContent
    
    return html`
      <div class="column-header">
        <h3 class="column-title">
          ${this.statusIcon}
          ${this.title}
          <span class="task-count">${this.tasks.length}</span>
        </h3>
      </div>
      
      <div class="column-tasks drop-zone">
        ${this.tasks.length === 0 ? html`
          <div class="empty-state">
            ${emptyState.icon}
            ${emptyState.message}
          </div>
        ` : repeat(
          this.tasks,
          task => task.id,
          task => html`
            <task-card
              .task=${task}
              .offline=${this.offline}
              data-task-id=${task.id}
              @click=${() => this.handleTaskClick(task)}
            ></task-card>
          `
        )}
      </div>
    `
  }
}