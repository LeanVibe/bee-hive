import { LitElement, html, css } from 'lit'
import { customElement, property } from 'lit/decorators.js'
import { Task, TaskPriority } from '../../types/task'

// Simple date formatting utility
function formatDistanceToNow(date: Date): string {
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / (1000 * 60))
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
  
  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`
  if (diffDays < 365) return `${Math.floor(diffDays / 30)}mo ago`
  return `${Math.floor(diffDays / 365)}y ago`
}

@customElement('task-card')
export class TaskCard extends LitElement {
  @property({ type: Object }) task!: Task
  @property({ type: Boolean }) offline: boolean = false
  
  static styles = css`
    :host {
      display: block;
      cursor: pointer;
      user-select: none;
    }
    
    .task-card {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      transition: all 0.2s;
      position: relative;
      overflow: hidden;
    }
    
    .task-card:hover {
      border-color: #d1d5db;
      box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
      transform: translateY(-1px);
    }
    
    .task-card:active {
      transform: translateY(0);
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .task-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 0.75rem;
      gap: 0.5rem;
    }
    
    .task-title {
      font-weight: 600;
      font-size: 0.875rem;
      line-height: 1.25;
      color: #111827;
      margin: 0;
      flex: 1;
      word-break: break-word;
    }
    
    .priority-badge {
      padding: 0.125rem 0.375rem;
      border-radius: 0.75rem;
      font-size: 0.75rem;
      font-weight: 500;
      white-space: nowrap;
      flex-shrink: 0;
    }
    
    .priority-high {
      background: #fee2e2;
      color: #dc2626;
    }
    
    .priority-medium {
      background: #fef3c7;
      color: #d97706;
    }
    
    .priority-low {
      background: #ecfdf5;
      color: #059669;
    }
    
    .task-description {
      font-size: 0.8125rem;
      line-height: 1.4;
      color: #6b7280;
      margin-bottom: 0.75rem;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    
    .task-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.75rem;
      color: #9ca3af;
      gap: 0.5rem;
    }
    
    .agent-badge {
      background: #f3f4f6;
      color: #374151;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
    
    .agent-status {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #10b981;
    }
    
    .agent-status.offline {
      background: #6b7280;
    }
    
    .timestamp {
      white-space: nowrap;
    }
    
    .task-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 0.25rem;
      margin-bottom: 0.5rem;
    }
    
    .tag {
      background: #f3f4f6;
      color: #374151;
      padding: 0.125rem 0.25rem;
      border-radius: 0.125rem;
      font-size: 0.625rem;
      font-weight: 500;
    }
    
    .offline-indicator {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      width: 8px;
      height: 8px;
      background: #f59e0b;
      border-radius: 50%;
      opacity: 0.7;
    }
    
    .sync-status {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.625rem;
      color: #6b7280;
    }
    
    .sync-pending {
      color: #f59e0b;
    }
    
    .sync-error {
      color: #dc2626;
    }
    
    .sync-icon {
      width: 12px;
      height: 12px;
    }
    
    /* Drag styling */
    :host(.sortable-ghost) {
      opacity: 0.4;
    }
    
    :host(.sortable-drag) .task-card {
      transform: rotate(3deg);
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    @media (max-width: 768px) {
      .task-card {
        padding: 0.75rem;
      }
      
      .task-title {
        font-size: 0.8125rem;
      }
      
      .task-description {
        font-size: 0.75rem;
      }
      
      .task-meta {
        font-size: 0.6875rem;
      }
    }
  `
  
  private get priorityClass() {
    return `priority-${this.task.priority || 'medium'}`
  }
  
  private get formattedTimestamp() {
    try {
      return formatDistanceToNow(new Date(this.task.updatedAt))
    } catch {
      return 'Invalid date'
    }
  }
  
  private get syncStatus() {
    if (this.offline) {
      return { icon: 'offline', text: 'Offline', class: 'sync-pending' }
    }
    
    if (this.task.syncStatus === 'pending') {
      return { icon: 'sync', text: 'Syncing...', class: 'sync-pending' }
    }
    
    if (this.task.syncStatus === 'error') {
      return { icon: 'error', text: 'Sync failed', class: 'sync-error' }
    }
    
    return null
  }
  
  private renderSyncIcon(type: string) {
    switch (type) {
      case 'sync':
        return html`
          <svg class="sync-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        `
      case 'error':
        return html`
          <svg class="sync-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        `
      case 'offline':
        return html`
          <svg class="sync-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 5.636l-12.728 12.728m0 0L12 12m-6.364 6.364L12 12m6.364-6.364L12 12" />
          </svg>
        `
      default:
        return html``
    }
  }
  
  render() {
    const sync = this.syncStatus
    
    return html`
      <div class="task-card">
        ${this.offline ? html`<div class="offline-indicator"></div>` : ''}
        
        <div class="task-header">
          <h4 class="task-title">${this.task.title}</h4>
          <span class="priority-badge ${this.priorityClass}">
            ${this.task.priority || 'medium'}
          </span>
        </div>
        
        ${this.task.description ? html`
          <p class="task-description">${this.task.description}</p>
        ` : ''}
        
        ${this.task.tags && this.task.tags.length > 0 ? html`
          <div class="task-tags">
            ${this.task.tags.map(tag => html`
              <span class="tag">${tag}</span>
            `)}
          </div>
        ` : ''}
        
        <div class="task-meta">
          <div class="agent-badge">
            <div class="agent-status ${this.offline ? 'offline' : ''}"></div>
            ${this.task.agent}
          </div>
          
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            ${sync ? html`
              <div class="sync-status ${sync.class}">
                ${this.renderSyncIcon(sync.icon)}
                ${sync.text}
              </div>
            ` : ''}
            
            <span class="timestamp">${this.formattedTimestamp}</span>
          </div>
        </div>
      </div>
    `
  }
}