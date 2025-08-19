import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { repeat } from 'lit/directives/repeat.js'
import '../charts/sparkline-chart'

export interface DebtItem {
  id: string
  file_path: string
  debt_type: string
  category: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  debt_score: number
  confidence_score: number
  description: string
  remediation_suggestion: string
  estimated_effort_hours: number
  first_detected_at?: Date
  last_detected_at?: Date
}

export interface DebtMetrics {
  total_debt_score: number
  debt_items_count: number
  category_breakdown: Record<string, number>
  severity_breakdown: Record<string, number>
  file_count: number
  lines_of_code: number
  debt_trend: 'increasing' | 'decreasing' | 'stable'
  debt_velocity: number
  hotspot_files: string[]
}

export interface ProjectDebtStatus {
  project_id: string
  project_name: string
  debt_metrics: DebtMetrics
  debt_items: DebtItem[]
  last_analyzed_at: Date
  analysis_duration: number
  monitoring_enabled: boolean
}

@customElement('technical-debt-panel')
export class TechnicalDebtPanel extends LitElement {
  @property({ type: Array }) declare projects: ProjectDebtStatus[]
  @property({ type: Boolean }) declare compact: boolean
  @property({ type: String }) declare sortBy: 'debt_score' | 'items_count' | 'velocity' | 'last_analyzed'
  @property({ type: String }) declare filterSeverity: string
  @property({ type: Boolean }) declare showRecommendations: boolean
  
  @state() private declare selectedProject: string | null
  @state() private declare isAnalyzing: boolean
  @state() private declare showAdvancedMetrics: boolean
  @state() private declare selectedDebtItem: DebtItem | null
  @state() private declare analysisInProgress: Set<string>
  @state() private declare showHistoricalView: boolean
  @state() private declare activeTab: 'overview' | 'hotspots' | 'trends' | 'recommendations'
  
  constructor() {
    super()
    
    // Initialize reactive properties
    this.projects = []
    this.compact = false
    this.sortBy = 'debt_score'
    this.filterSeverity = 'all'
    this.showRecommendations = true
    this.selectedProject = null
    this.isAnalyzing = false
    this.showAdvancedMetrics = true
    this.selectedDebtItem = null
    this.analysisInProgress = new Set()
    this.showHistoricalView = false
    this.activeTab = 'overview'
  }
  
  static styles = css`
    :host {
      display: block;
      background: white;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }
    
    .debt-panel-header {
      padding: 1rem;
      border-bottom: 1px solid #e5e7eb;
      background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    }
    
    .header-content {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.75rem;
    }
    
    .panel-title {
      font-size: 1rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .debt-icon {
      width: 20px;
      height: 20px;
      color: #f59e0b;
    }
    
    .analyze-button {
      background: #3b82f6;
      border: none;
      color: white;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 0.375rem;
    }
    
    .analyze-button:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }
    
    .analyze-button:disabled {
      background: #9ca3af;
      cursor: not-allowed;
      transform: none;
    }
    
    .analyze-button.analyzing {
      background: #059669;
    }
    
    .tab-navigation {
      display: flex;
      gap: 0.25rem;
      margin-top: 0.75rem;
    }
    
    .tab-button {
      background: none;
      border: none;
      color: #6b7280;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .tab-button:hover {
      background: rgba(59, 130, 246, 0.1);
      color: #3b82f6;
    }
    
    .tab-button.active {
      background: #3b82f6;
      color: white;
    }
    
    .debt-content {
      padding: 1rem;
      max-height: 500px;
      overflow-y: auto;
    }
    
    .debt-overview {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    
    .debt-metric-card {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      text-align: center;
    }
    
    .metric-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin-bottom: 0.25rem;
    }
    
    .metric-value.critical {
      color: #dc2626;
    }
    
    .metric-value.high {
      color: #f59e0b;
    }
    
    .metric-value.medium {
      color: #3b82f6;
    }
    
    .metric-value.low {
      color: #10b981;
    }
    
    .metric-label {
      font-size: 0.875rem;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }
    
    .metric-trend {
      margin-top: 0.5rem;
      font-size: 0.75rem;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.25rem;
    }
    
    .trend-up {
      color: #dc2626;
    }
    
    .trend-down {
      color: #10b981;
    }
    
    .trend-stable {
      color: #6b7280;
    }
    
    .project-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    
    .project-card {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .project-card:hover {
      border-color: #3b82f6;
      box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    
    .project-card.selected {
      border-color: #3b82f6;
      background: #eff6ff;
    }
    
    .project-header {
      display: flex;
      align-items: center;
      justify-content: between;
      margin-bottom: 0.5rem;
    }
    
    .project-name {
      font-weight: 600;
      color: #111827;
      margin: 0;
      flex: 1;
    }
    
    .debt-score-badge {
      background: #dc2626;
      color: white;
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      font-weight: 600;
    }
    
    .debt-score-badge.low {
      background: #10b981;
    }
    
    .debt-score-badge.medium {
      background: #f59e0b;
    }
    
    .debt-score-badge.high {
      background: #dc2626;
    }
    
    .project-stats {
      display: flex;
      gap: 1rem;
      font-size: 0.875rem;
      color: #6b7280;
    }
    
    .stat-item {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
    
    .debt-items-list {
      margin-top: 1rem;
    }
    
    .debt-item {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.375rem;
      padding: 0.75rem;
      margin-bottom: 0.5rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .debt-item:hover {
      border-color: #3b82f6;
      box-shadow: 0 1px 3px rgba(59, 130, 246, 0.1);
    }
    
    .debt-item-header {
      display: flex;
      align-items: center;
      justify-content: between;
      margin-bottom: 0.5rem;
    }
    
    .debt-type {
      font-weight: 600;
      color: #111827;
      font-size: 0.875rem;
    }
    
    .severity-badge {
      background: #dc2626;
      color: white;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      font-weight: 500;
    }
    
    .severity-badge.low {
      background: #10b981;
    }
    
    .severity-badge.medium {
      background: #f59e0b;
    }
    
    .severity-badge.high {
      background: #dc2626;
    }
    
    .severity-badge.critical {
      background: #7c2d12;
    }
    
    .debt-description {
      font-size: 0.875rem;
      color: #6b7280;
      margin-bottom: 0.5rem;
    }
    
    .debt-location {
      font-size: 0.75rem;
      color: #9ca3af;
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
    }
    
    .loading-spinner {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
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
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    
    .empty-state {
      text-align: center;
      padding: 3rem 1rem;
      color: #6b7280;
    }
    
    .empty-state svg {
      width: 48px;
      height: 48px;
      margin-bottom: 1rem;
      color: #d1d5db;
    }
    
    .controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    
    .filter-select {
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      padding: 0.375rem 0.5rem;
      font-size: 0.875rem;
      background: white;
    }
    
    .sort-select {
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      padding: 0.375rem 0.5rem;
      font-size: 0.875rem;
      background: white;
    }
  `
  
  private handleProjectClick(project: ProjectDebtStatus) {
    this.selectedProject = this.selectedProject === project.project_id ? null : project.project_id
  }
  
  private async handleAnalyzeProject(project: ProjectDebtStatus) {
    if (this.analysisInProgress.has(project.project_id)) return
    
    this.analysisInProgress.add(project.project_id)
    this.requestUpdate()
    
    try {
      // Dispatch custom event for parent component to handle
      this.dispatchEvent(new CustomEvent('analyze-project', {
        detail: { projectId: project.project_id },
        bubbles: true
      }))
    } catch (error) {
      console.error('Failed to analyze project:', error)
    } finally {
      this.analysisInProgress.delete(project.project_id)
      this.requestUpdate()
    }
  }
  
  private handleTabClick(tab: 'overview' | 'hotspots' | 'trends' | 'recommendations') {
    this.activeTab = tab
  }
  
  private getDebtScoreClass(score: number): string {
    if (score >= 0.8) return 'critical'
    if (score >= 0.6) return 'high'
    if (score >= 0.4) return 'medium'
    return 'low'
  }
  
  private getTrendIcon(trend: 'increasing' | 'decreasing' | 'stable') {
    switch (trend) {
      case 'increasing':
        return html`<svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 17l9.2-9.2M17 17V7h-10"/>
        </svg>`
      case 'decreasing':
        return html`<svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 7l-9.2 9.2M7 7v10h10"/>
        </svg>`
      default:
        return html`<svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/>
        </svg>`
    }
  }
  
  private renderOverviewTab() {
    if (!this.projects.length) {
      return html`
        <div class="empty-state">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
          </svg>
          <div>No technical debt data available</div>
          <div>Run an analysis to get started</div>
        </div>
      `
    }
    
    const totalProjects = this.projects.length
    const totalDebtScore = this.projects.reduce((sum, p) => sum + p.debt_metrics.total_debt_score, 0)
    const avgDebtScore = totalDebtScore / totalProjects
    const totalDebtItems = this.projects.reduce((sum, p) => sum + p.debt_metrics.debt_items_count, 0)
    const criticalItems = this.projects.reduce((sum, p) => 
      sum + p.debt_items.filter(item => item.severity === 'critical').length, 0)
    
    return html`
      <div class="debt-overview">
        <div class="debt-metric-card">
          <div class="metric-value ${this.getDebtScoreClass(avgDebtScore)}">${avgDebtScore.toFixed(2)}</div>
          <div class="metric-label">Average Debt Score</div>
        </div>
        <div class="debt-metric-card">
          <div class="metric-value">${totalDebtItems}</div>
          <div class="metric-label">Total Debt Items</div>
        </div>
        <div class="debt-metric-card">
          <div class="metric-value critical">${criticalItems}</div>
          <div class="metric-label">Critical Issues</div>
        </div>
        <div class="debt-metric-card">
          <div class="metric-value">${totalProjects}</div>
          <div class="metric-label">Projects Monitored</div>
        </div>
      </div>
      
      <div class="project-list">
        ${repeat(this.projects, project => project.project_id, project => html`
          <div class="project-card ${this.selectedProject === project.project_id ? 'selected' : ''}" 
               @click=${() => this.handleProjectClick(project)}>
            <div class="project-header">
              <h3 class="project-name">${project.project_name}</h3>
              <div class="debt-score-badge ${this.getDebtScoreClass(project.debt_metrics.total_debt_score)}">
                ${project.debt_metrics.total_debt_score.toFixed(2)}
              </div>
            </div>
            <div class="project-stats">
              <div class="stat-item">
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                ${project.debt_metrics.debt_items_count} issues
              </div>
              <div class="stat-item">
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                </svg>
                ${project.debt_metrics.file_count} files
              </div>
              <div class="stat-item">
                <div class="metric-trend trend-${project.debt_metrics.debt_trend}">
                  ${this.getTrendIcon(project.debt_metrics.debt_trend)}
                  ${project.debt_metrics.debt_trend}
                </div>
              </div>
            </div>
            
            ${this.selectedProject === project.project_id ? html`
              <div class="debt-items-list">
                ${repeat(project.debt_items.slice(0, 5), item => item.id, item => html`
                  <div class="debt-item" @click=${(e: Event) => { e.stopPropagation(); this.selectedDebtItem = item; }}>
                    <div class="debt-item-header">
                      <span class="debt-type">${item.debt_type}</span>
                      <span class="severity-badge ${item.severity}">${item.severity}</span>
                    </div>
                    <div class="debt-description">${item.description}</div>
                    <div class="debt-location">${item.file_path}</div>
                  </div>
                `)}
                ${project.debt_items.length > 5 ? html`
                  <div style="text-align: center; margin-top: 0.5rem; font-size: 0.875rem; color: #6b7280;">
                    +${project.debt_items.length - 5} more items
                  </div>
                ` : ''}
              </div>
            ` : ''}
          </div>
        `)}
      </div>
    `
  }
  
  private renderCurrentTab() {
    switch (this.activeTab) {
      case 'overview':
        return this.renderOverviewTab()
      case 'hotspots':
        return html`<div>Debt hotspots visualization (coming soon)</div>`
      case 'trends':
        return html`<div>Historical debt trends (coming soon)</div>`
      case 'recommendations':
        return html`<div>Remediation recommendations (coming soon)</div>`
      default:
        return this.renderOverviewTab()
    }
  }
  
  render() {
    return html`
      <div class="debt-panel-header">
        <div class="header-content">
          <h2 class="panel-title">
            <svg class="debt-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            Technical Debt Analysis
          </h2>
          <div class="controls">
            <select class="filter-select" .value=${this.filterSeverity} @change=${(e: Event) => this.filterSeverity = (e.target as HTMLSelectElement).value}>
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            <button class="analyze-button ${this.isAnalyzing ? 'analyzing' : ''}" 
                    @click=${() => this.dispatchEvent(new CustomEvent('analyze-all-projects', { bubbles: true }))}
                    ?disabled=${this.isAnalyzing}>
              ${this.isAnalyzing ? html`
                <div class="spinner"></div>
                Analyzing...
              ` : html`
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                </svg>
                Analyze All
              `}
            </button>
          </div>
        </div>
        
        <div class="tab-navigation">
          <button class="tab-button ${this.activeTab === 'overview' ? 'active' : ''}" 
                  @click=${() => this.handleTabClick('overview')}>
            Overview
          </button>
          <button class="tab-button ${this.activeTab === 'hotspots' ? 'active' : ''}" 
                  @click=${() => this.handleTabClick('hotspots')}>
            Hotspots
          </button>
          <button class="tab-button ${this.activeTab === 'trends' ? 'active' : ''}" 
                  @click=${() => this.handleTabClick('trends')}>
            Trends
          </button>
          <button class="tab-button ${this.activeTab === 'recommendations' ? 'active' : ''}" 
                  @click=${() => this.handleTabClick('recommendations')}>
            Recommendations
          </button>
        </div>
      </div>
      
      <div class="debt-content">
        ${this.renderCurrentTab()}
      </div>
    `
  }
}