/**
 * Sprint Planning Component for LeanVibe Agent Hive
 * 
 * Provides comprehensive sprint planning with:
 * - Backlog management and prioritization
 * - Sprint capacity planning with agent workload
 * - Velocity estimation and sprint goal setting
 * - Multi-agent task assignment and coordination
 * - Real-time sprint analytics and progress tracking
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { repeat } from 'lit/directives/repeat.js'
import { Task, TaskStatus, TaskPriority } from '../../types/task'
import { Agent, AgentRole } from '../../types/api'

export interface Sprint {
  id: string
  name: string
  startDate: string
  endDate: string
  status: 'planning' | 'active' | 'completed' | 'cancelled'
  goal: string
  capacity: number
  tasks: Task[]
  metrics: SprintMetrics
}

export interface SprintMetrics {
  plannedStoryPoints: number
  completedStoryPoints: number
  velocityTarget: number
  burndownData: Array<{ date: string; remaining: number }>
  teamUtilization: Record<string, number>
}

export interface BacklogItem extends Task {
  storyPoints: number
  businessValue: number
  effort: number
  risk: 'low' | 'medium' | 'high'
  dependencies: string[]
}

@customElement('sprint-planner')
export class SprintPlanner extends LitElement {
  @property({ type: Array }) backlog: BacklogItem[] = []
  @property({ type: Array }) agents: Agent[] = []
  @property({ type: Object }) currentSprint: Sprint | null = null
  @property({ type: Array }) pastSprints: Sprint[] = []
  @property({ type: Boolean }) offline: boolean = false
  
  @state() private selectedBacklogItems: Set<string> = new Set()
  @state() private sprintCapacity: number = 40
  @state() private sprintGoal: string = ''
  @state() private sprintDuration: number = 14 // days
  @state() private viewMode: 'backlog' | 'sprint' | 'analytics' = 'backlog'
  @state() private sortBy: 'priority' | 'points' | 'value' | 'effort' = 'priority'
  @state() private filterBy: 'all' | 'unassigned' | 'high-value' | 'ready' = 'all'
  @state() private agentCapacities: Map<string, number> = new Map()
  @state() private isCreatingSprint: boolean = false
  
  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .planner-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1.5rem;
      background: white;
      border-bottom: 2px solid #e2e8f0;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .planner-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: #1e293b;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .view-tabs {
      display: flex;
      gap: 0.25rem;
      background: #f1f5f9;
      padding: 0.25rem;
      border-radius: 0.5rem;
    }
    
    .view-tab {
      padding: 0.5rem 1rem;
      font-size: 0.875rem;
      font-weight: 600;
      border: none;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s;
      background: transparent;
      color: #64748b;
    }
    
    .view-tab.active {
      background: white;
      color: #3b82f6;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .view-tab:hover:not(.active) {
      background: #e2e8f0;
      color: #475569;
    }
    
    .planner-controls {
      display: flex;
      gap: 1rem;
      padding: 1rem 1.5rem;
      background: white;
      border-bottom: 1px solid #e2e8f0;
      flex-wrap: wrap;
      align-items: center;
    }
    
    .control-group {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }
    
    .control-label {
      font-size: 0.75rem;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }
    
    .control-select,
    .control-input {
      padding: 0.375rem 0.75rem;
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      background: white;
    }
    
    .control-input[type="number"] {
      width: 80px;
    }
    
    .create-sprint-btn {
      padding: 0.5rem 1rem;
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      color: white;
      border: none;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .create-sprint-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
    }
    
    .create-sprint-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }
    
    .planner-content {
      display: flex;
      height: calc(100% - 140px);
      gap: 1rem;
      padding: 1rem 1.5rem;
    }
    
    .backlog-panel {
      flex: 2;
      background: white;
      border-radius: 0.75rem;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    
    .sprint-panel {
      flex: 1;
      background: white;
      border-radius: 0.75rem;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    
    .panel-header {
      padding: 1rem 1.5rem;
      border-bottom: 1px solid #e2e8f0;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .panel-title {
      font-size: 1.125rem;
      font-weight: 600;
      color: #1e293b;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .panel-subtitle {
      font-size: 0.875rem;
      color: #64748b;
      margin-top: 0.25rem;
    }
    
    .panel-content {
      flex: 1;
      overflow-y: auto;
      padding: 1rem;
    }
    
    .backlog-item {
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 1rem;
      border: 1px solid #e2e8f0;
      border-radius: 0.5rem;
      margin-bottom: 0.75rem;
      cursor: pointer;
      transition: all 0.2s;
      background: white;
    }
    
    .backlog-item:hover {
      border-color: #3b82f6;
      box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
      transform: translateY(-1px);
    }
    
    .backlog-item.selected {
      border-color: #10b981;
      background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
      box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
    }
    
    .item-checkbox {
      width: 18px;
      height: 18px;
      border: 2px solid #d1d5db;
      border-radius: 0.25rem;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .item-checkbox.checked {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      border-color: #10b981;
      color: white;
    }
    
    .item-details {
      flex: 1;
    }
    
    .item-title {
      font-size: 0.875rem;
      font-weight: 600;
      color: #1e293b;
      margin-bottom: 0.25rem;
    }
    
    .item-description {
      font-size: 0.75rem;
      color: #64748b;
      margin-bottom: 0.5rem;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    
    .item-meta {
      display: flex;
      gap: 1rem;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .meta-badge {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.625rem;
      font-weight: 600;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }
    
    .meta-badge.points {
      background: #dbeafe;
      color: #1d4ed8;
    }
    
    .meta-badge.value {
      background: #fef3c7;
      color: #92400e;
    }
    
    .meta-badge.effort {
      background: #f3e8ff;
      color: #7c3aed;
    }
    
    .meta-badge.risk.low {
      background: #d1fae5;
      color: #16a34a;
    }
    
    .meta-badge.risk.medium {
      background: #fed7aa;
      color: #ea580c;
    }
    
    .meta-badge.risk.high {
      background: #fecaca;
      color: #dc2626;
    }
    
    .priority-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      margin-left: auto;
    }
    
    .priority-indicator.high {
      background: #ef4444;
      box-shadow: 0 0 0 2px #fecaca;
    }
    
    .priority-indicator.medium {
      background: #f59e0b;
      box-shadow: 0 0 0 2px #fed7aa;
    }
    
    .priority-indicator.low {
      background: #10b981;
      box-shadow: 0 0 0 2px #d1fae5;
    }
    
    .sprint-summary {
      padding: 1rem;
      border: 2px dashed #d1d5db;
      border-radius: 0.5rem;
      text-align: center;
      margin-bottom: 1rem;
    }
    
    .sprint-summary.has-items {
      border-color: #10b981;
      background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    }
    
    .summary-stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1rem;
      margin-top: 1rem;
    }
    
    .summary-stat {
      text-align: center;
    }
    
    .stat-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #1e293b;
      display: block;
    }
    
    .stat-label {
      font-size: 0.75rem;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }
    
    .capacity-warning {
      background: #fef3c7;
      border: 1px solid #f59e0b;
      color: #92400e;
      padding: 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      margin-top: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .agent-capacity-list {
      margin-top: 1rem;
    }
    
    .agent-capacity-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.5rem;
      border-bottom: 1px solid #f1f5f9;
    }
    
    .agent-name {
      font-size: 0.875rem;
      font-weight: 500;
      color: #1e293b;
    }
    
    .agent-role {
      font-size: 0.75rem;
      color: #64748b;
    }
    
    .capacity-bar {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex: 1;
      margin: 0 1rem;
    }
    
    .capacity-progress {
      flex: 1;
      height: 6px;
      background: #f1f5f9;
      border-radius: 3px;
      overflow: hidden;
    }
    
    .capacity-fill {
      height: 100%;
      background: linear-gradient(90deg, #10b981 0%, #059669 100%);
      transition: width 0.3s;
    }
    
    .capacity-fill.over {
      background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
    }
    
    .capacity-percentage {
      font-size: 0.75rem;
      font-weight: 600;
      min-width: 40px;
      text-align: right;
    }
    
    .empty-state {
      text-align: center;
      padding: 3rem 2rem;
      color: #64748b;
    }
    
    .empty-state-icon {
      font-size: 3rem;
      margin-bottom: 1rem;
      opacity: 0.5;
    }
    
    .empty-state-title {
      font-size: 1.125rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }
    
    .empty-state-description {
      font-size: 0.875rem;
      line-height: 1.5;
    }
    
    @media (max-width: 1024px) {
      .planner-content {
        flex-direction: column;
        height: auto;
      }
      
      .backlog-panel,
      .sprint-panel {
        flex: none;
        height: 400px;
      }
    }
    
    @media (max-width: 768px) {
      .planner-header {
        flex-direction: column;
        gap: 1rem;
        align-items: stretch;
      }
      
      .planner-controls {
        flex-direction: column;
        align-items: stretch;
        gap: 0.5rem;
      }
      
      .control-group {
        justify-content: space-between;
      }
      
      .summary-stats {
        grid-template-columns: 1fr;
      }
    }
  `
  
  connectedCallback() {
    super.connectedCallback()
    this.initializeAgentCapacities()
  }
  
  private initializeAgentCapacities() {
    // Initialize default capacities for agents (40 hours per sprint)
    this.agents.forEach(agent => {
      this.agentCapacities.set(agent.id, 40)
    })
  }
  
  private get filteredBacklog() {
    let filtered = [...this.backlog]
    
    // Apply filters
    switch (this.filterBy) {
      case 'unassigned':
        filtered = filtered.filter(item => !item.agent)
        break
      case 'high-value':
        filtered = filtered.filter(item => item.businessValue >= 8)
        break
      case 'ready':
        filtered = filtered.filter(item => 
          item.status === 'pending' && 
          item.dependencies.length === 0
        )
        break
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      switch (this.sortBy) {
        case 'priority':
          const priorityOrder = { high: 3, medium: 2, low: 1 }
          return priorityOrder[b.priority] - priorityOrder[a.priority]
        case 'points':
          return b.storyPoints - a.storyPoints
        case 'value':
          return b.businessValue - a.businessValue
        case 'effort':
          return a.effort - b.effort
        default:
          return 0
      }
    })
    
    return filtered
  }
  
  private get selectedItems() {
    return this.backlog.filter(item => this.selectedBacklogItems.has(item.id))
  }
  
  private get sprintStats() {
    const selectedItems = this.selectedItems
    const totalPoints = selectedItems.reduce((sum, item) => sum + item.storyPoints, 0)
    const totalValue = selectedItems.reduce((sum, item) => sum + item.businessValue, 0)
    const avgValue = selectedItems.length > 0 ? Math.round(totalValue / selectedItems.length) : 0
    const capacityUtilization = Math.round((totalPoints / this.sprintCapacity) * 100)
    
    return {
      itemCount: selectedItems.length,
      totalPoints,
      avgValue,
      capacityUtilization,
      isOverCapacity: totalPoints > this.sprintCapacity
    }
  }
  
  private toggleBacklogItem(itemId: string) {
    if (this.selectedBacklogItems.has(itemId)) {
      this.selectedBacklogItems.delete(itemId)
    } else {
      this.selectedBacklogItems.add(itemId)
    }
    this.requestUpdate()
  }
  
  private async createSprint() {
    if (this.selectedItems.length === 0 || !this.sprintGoal.trim()) {
      return
    }
    
    this.isCreatingSprint = true
    
    try {
      const sprintStartDate = new Date()
      const sprintEndDate = new Date()
      sprintEndDate.setDate(sprintStartDate.getDate() + this.sprintDuration)
      
      const newSprint: Sprint = {
        id: `sprint-${Date.now()}`,
        name: `Sprint ${(this.pastSprints.length + 1).toString().padStart(2, '0')}`,
        startDate: sprintStartDate.toISOString(),
        endDate: sprintEndDate.toISOString(),
        status: 'planning',
        goal: this.sprintGoal,
        capacity: this.sprintCapacity,
        tasks: this.selectedItems.map(item => ({
          ...item,
          status: 'pending' as TaskStatus
        })),
        metrics: {
          plannedStoryPoints: this.sprintStats.totalPoints,
          completedStoryPoints: 0,
          velocityTarget: this.sprintStats.totalPoints,
          burndownData: [],
          teamUtilization: Object.fromEntries(this.agentCapacities.entries())
        }
      }
      
      const createSprintEvent = new CustomEvent('sprint-created', {
        detail: { sprint: newSprint },
        bubbles: true,
        composed: true
      })
      
      this.dispatchEvent(createSprintEvent)
      
      // Clear selection and reset form
      this.selectedBacklogItems.clear()
      this.sprintGoal = ''
      this.viewMode = 'sprint'
      
    } catch (error) {
      console.error('Failed to create sprint:', error)
    } finally {
      this.isCreatingSprint = false
    }
  }
  
  private handleViewChange(view: 'backlog' | 'sprint' | 'analytics') {
    this.viewMode = view
  }
  
  private handleSortChange(e: Event) {
    this.sortBy = (e.target as HTMLSelectElement).value as any
  }
  
  private handleFilterChange(e: Event) {
    this.filterBy = (e.target as HTMLSelectElement).value as any
  }
  
  private handleCapacityChange(e: Event) {
    this.sprintCapacity = parseInt((e.target as HTMLInputElement).value) || 40
  }
  
  private handleGoalChange(e: Event) {
    this.sprintGoal = (e.target as HTMLTextAreaElement).value
  }
  
  private renderBacklogView() {
    return html`
      <div class="planner-content">
        <div class="backlog-panel">
          <div class="panel-header">
            <div class="panel-title">
              📋 Product Backlog
              <span style="font-weight: 400; color: #64748b;">
                (${this.filteredBacklog.length} items)
              </span>
            </div>
            <div class="panel-subtitle">
              Select items to include in your next sprint
            </div>
          </div>
          <div class="panel-content">
            ${this.filteredBacklog.length === 0 ? html`
              <div class="empty-state">
                <div class="empty-state-icon">📝</div>
                <div class="empty-state-title">No Backlog Items</div>
                <div class="empty-state-description">
                  Create some backlog items to start planning your sprints.
                </div>
              </div>
            ` : ''}
            
            ${this.filteredBacklog.map(item => html`
              <div 
                class="backlog-item ${this.selectedBacklogItems.has(item.id) ? 'selected' : ''}"
                @click=${() => this.toggleBacklogItem(item.id)}
              >
                <div class="item-checkbox ${this.selectedBacklogItems.has(item.id) ? 'checked' : ''}">
                  ${this.selectedBacklogItems.has(item.id) ? '✓' : ''}
                </div>
                
                <div class="item-details">
                  <div class="item-title">${item.title}</div>
                  <div class="item-description">${item.description}</div>
                  <div class="item-meta">
                    <div class="meta-badge points">
                      ${item.storyPoints} pts
                    </div>
                    <div class="meta-badge value">
                      Value: ${item.businessValue}/10
                    </div>
                    <div class="meta-badge effort">
                      Effort: ${item.effort}/10
                    </div>
                    <div class="meta-badge risk ${item.risk}">
                      ${item.risk} risk
                    </div>
                  </div>
                </div>
                
                <div class="priority-indicator ${item.priority}"></div>
              </div>
            `)}
          </div>
        </div>
        
        <div class="sprint-panel">
          <div class="panel-header">
            <div class="panel-title">
              🎯 Sprint Planning
            </div>
            <div class="panel-subtitle">
              Configure your sprint parameters
            </div>
          </div>
          <div class="panel-content">
            <div class="sprint-summary ${this.selectedItems.length > 0 ? 'has-items' : ''}">
              ${this.selectedItems.length === 0 ? html`
                <div style="color: #64748b; font-size: 0.875rem;">
                  Select backlog items to see sprint summary
                </div>
              ` : html`
                <div class="summary-stats">
                  <div class="summary-stat">
                    <span class="stat-value">${this.sprintStats.itemCount}</span>
                    <span class="stat-label">Items</span>
                  </div>
                  <div class="summary-stat">
                    <span class="stat-value">${this.sprintStats.totalPoints}</span>
                    <span class="stat-label">Story Points</span>
                  </div>
                  <div class="summary-stat">
                    <span class="stat-value">${this.sprintStats.avgValue}/10</span>
                    <span class="stat-label">Avg Value</span>
                  </div>
                  <div class="summary-stat">
                    <span class="stat-value" style="color: ${this.sprintStats.isOverCapacity ? '#ef4444' : '#10b981'}">
                      ${this.sprintStats.capacityUtilization}%
                    </span>
                    <span class="stat-label">Capacity</span>
                  </div>
                </div>
                
                ${this.sprintStats.isOverCapacity ? html`
                  <div class="capacity-warning">
                    ⚠️ Sprint is over capacity by ${this.sprintStats.totalPoints - this.sprintCapacity} points
                  </div>
                ` : ''}
              `}
            </div>
            
            <div style="margin-top: 1.5rem;">
              <label class="control-label" style="display: block; margin-bottom: 0.5rem;">
                Sprint Goal
              </label>
              <textarea
                class="control-input"
                style="width: 100%; min-height: 80px; resize: vertical;"
                placeholder="What is the main objective of this sprint?"
                .value=${this.sprintGoal}
                @input=${this.handleGoalChange}
              ></textarea>
            </div>
            
            <div class="agent-capacity-list">
              <div class="control-label" style="margin-bottom: 1rem;">Team Capacity</div>
              ${this.agents.map(agent => {
                const capacity = this.agentCapacities.get(agent.id) || 40
                const utilization = Math.min((capacity / 40) * 100, 100)
                return html`
                  <div class="agent-capacity-item">
                    <div>
                      <div class="agent-name">${agent.name}</div>
                      <div class="agent-role">${agent.role.replace('_', ' ')}</div>
                    </div>
                    <div class="capacity-bar">
                      <div class="capacity-progress">
                        <div 
                          class="capacity-fill ${utilization > 100 ? 'over' : ''}"
                          style="width: ${Math.min(utilization, 100)}%"
                        ></div>
                      </div>
                      <div class="capacity-percentage">${capacity}h</div>
                    </div>
                  </div>
                `
              })}
            </div>
          </div>
        </div>
      </div>
    `
  }
  
  private renderSprintView() {
    if (!this.currentSprint) {
      return html`
        <div class="empty-state">
          <div class="empty-state-icon">🎯</div>
          <div class="empty-state-title">No Active Sprint</div>
          <div class="empty-state-description">
            Create a sprint from the backlog to start tracking progress.
          </div>
        </div>
      `
    }
    
    return html`
      <div style="padding: 2rem;">
        <h2>Sprint View - Coming Soon</h2>
        <p>This will show the active sprint board with burndown charts and team progress.</p>
      </div>
    `
  }
  
  private renderAnalyticsView() {
    return html`
      <div style="padding: 2rem;">
        <h2>Sprint Analytics - Coming Soon</h2>
        <p>This will show velocity trends, team performance metrics, and predictive analytics.</p>
      </div>
    `
  }
  
  render() {
    return html`
      <div class="planner-header">
        <div class="planner-title">
          🚀 Sprint Planning
        </div>
        
        <div class="view-tabs">
          <button 
            class="view-tab ${this.viewMode === 'backlog' ? 'active' : ''}" 
            @click=${() => this.handleViewChange('backlog')}
          >
            Backlog
          </button>
          <button 
            class="view-tab ${this.viewMode === 'sprint' ? 'active' : ''}" 
            @click=${() => this.handleViewChange('sprint')}
          >
            Sprint
          </button>
          <button 
            class="view-tab ${this.viewMode === 'analytics' ? 'active' : ''}" 
            @click=${() => this.handleViewChange('analytics')}
          >
            Analytics
          </button>
        </div>
      </div>
      
      ${this.viewMode === 'backlog' ? html`
        <div class="planner-controls">
          <div class="control-group">
            <span class="control-label">Sort by</span>
            <select class="control-select" .value=${this.sortBy} @change=${this.handleSortChange}>
              <option value="priority">Priority</option>
              <option value="points">Story Points</option>
              <option value="value">Business Value</option>
              <option value="effort">Effort</option>
            </select>
          </div>
          
          <div class="control-group">
            <span class="control-label">Filter</span>
            <select class="control-select" .value=${this.filterBy} @change=${this.handleFilterChange}>
              <option value="all">All Items</option>
              <option value="unassigned">Unassigned</option>
              <option value="high-value">High Value</option>
              <option value="ready">Ready</option>
            </select>
          </div>
          
          <div class="control-group">
            <span class="control-label">Capacity</span>
            <input 
              type="number" 
              class="control-input" 
              min="1" 
              max="200" 
              .value=${this.sprintCapacity.toString()}
              @input=${this.handleCapacityChange}
            />
            <span style="font-size: 0.75rem; color: #64748b;">points</span>
          </div>
          
          <button 
            class="create-sprint-btn"
            ?disabled=${this.selectedItems.length === 0 || !this.sprintGoal.trim() || this.isCreatingSprint}
            @click=${this.createSprint}
          >
            ${this.isCreatingSprint ? html`
              <div style="width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            ` : '🚀'}
            Create Sprint
          </button>
        </div>
      ` : ''}
      
      ${this.viewMode === 'backlog' ? this.renderBacklogView() : ''}
      ${this.viewMode === 'sprint' ? this.renderSprintView() : ''}
      ${this.viewMode === 'analytics' ? this.renderAnalyticsView() : ''}
    `
  }
}