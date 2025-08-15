/**
 * Project Index Dashboard - Main Container Component
 * 
 * Provides the main interface for project analysis, file exploration,
 * dependency visualization, and AI context optimization.
 */

import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { classMap } from 'lit/directives/class-map.js';
import { ProjectIndexStore } from '../../services/project-index-store';
import type {
  ProjectIndex,
  ProjectIndexState,
  AnalysisSession,
  ComponentLoadingState,
  Subscription
} from '../../types/project-index';

// Import child components
import './FileStructureTree';
import './DependencyGraph';
import './AnalysisProgress';
import './ContextOptimizer';
import './IndexMetrics';
import './FileDetails';
import './SearchInterface';

@customElement('project-index-dashboard')
export class ProjectIndexDashboard extends LitElement {
  @property({ type: String }) declare selectedProjectId: string;
  @property({ type: Boolean }) declare compact: boolean;
  @property({ type: String }) declare activeView: string;

  @state() private declare projects: ProjectIndex[];
  @state() private declare selectedProject?: ProjectIndex;
  @state() private declare currentSession?: AnalysisSession;
  @state() private declare loadingStates: Record<string, ComponentLoadingState>;
  @state() private declare showProjectSelector: boolean;
  @state() private declare showCreateProject: boolean;
  @state() private declare sidebarCollapsed: boolean;
  @state() private declare currentTab: string;
  @state() private declare selectedFilePath: string;

  private store: ProjectIndexStore;
  private subscriptions: Subscription[] = [];

  constructor() {
    super();
    
    // Initialize properties
    this.selectedProjectId = '';
    this.compact = false;
    this.activeView = 'overview';
    this.projects = [];
    this.loadingStates = {};
    this.showProjectSelector = false;
    this.showCreateProject = false;
    this.sidebarCollapsed = false;
    this.currentTab = 'files';
    this.selectedFilePath = '';
    
    this.store = ProjectIndexStore.getInstance();
    this.setupStoreSubscriptions();
  }

  connectedCallback() {
    super.connectedCallback();
    this.loadInitialData();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.cleanupSubscriptions();
  }

  static styles = css`
    :host {
      display: block;
      height: 100vh;
      background: var(--surface-color, #ffffff);
      color: var(--text-primary-color, #1f2937);
      font-family: var(--font-family, system-ui, -apple-system, sans-serif);
      --sidebar-width: 320px;
      --header-height: 64px;
      --tab-height: 48px;
    }

    .dashboard-layout {
      display: grid;
      grid-template-areas: 
        "header header"
        "sidebar content";
      grid-template-columns: var(--sidebar-width) 1fr;
      grid-template-rows: var(--header-height) 1fr;
      height: 100vh;
      transition: grid-template-columns 0.3s ease;
    }

    .dashboard-layout.sidebar-collapsed {
      grid-template-columns: 60px 1fr;
    }

    /* Header Styles */
    .dashboard-header {
      grid-area: header;
      background: var(--primary-color, #3b82f6);
      color: white;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 1.5rem;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      z-index: 10;
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .header-title {
      font-size: 1.25rem;
      font-weight: 600;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .project-selector {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 0.5rem;
      padding: 0.5rem 1rem;
      cursor: pointer;
      transition: background 0.2s;
    }

    .project-selector:hover {
      background: rgba(255, 255, 255, 0.2);
    }

    .project-name {
      font-weight: 500;
      max-width: 200px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .header-button {
      background: rgba(255, 255, 255, 0.1);
      border: none;
      color: white;
      padding: 0.5rem;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .header-button:hover {
      background: rgba(255, 255, 255, 0.2);
    }

    .header-button.primary {
      background: rgba(255, 255, 255, 0.2);
      font-weight: 500;
      padding: 0.5rem 1rem;
      gap: 0.5rem;
    }

    /* Sidebar Styles */
    .dashboard-sidebar {
      grid-area: sidebar;
      background: var(--surface-secondary-color, #f8fafc);
      border-right: 1px solid var(--border-color, #e5e7eb);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      transition: all 0.3s ease;
    }

    .sidebar-header {
      padding: 1rem;
      border-bottom: 1px solid var(--border-color, #e5e7eb);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .sidebar-title {
      font-weight: 600;
      font-size: 0.875rem;
      color: var(--text-secondary-color, #6b7280);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .sidebar-toggle {
      background: none;
      border: none;
      color: var(--text-secondary-color, #6b7280);
      cursor: pointer;
      padding: 0.25rem;
      border-radius: 0.25rem;
      transition: all 0.2s;
    }

    .sidebar-toggle:hover {
      background: var(--hover-color, #f3f4f6);
      color: var(--text-primary-color, #1f2937);
    }

    .sidebar-content {
      flex: 1;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .sidebar-tabs {
      display: flex;
      border-bottom: 1px solid var(--border-color, #e5e7eb);
      background: white;
    }

    .sidebar-tab {
      flex: 1;
      background: none;
      border: none;
      padding: 0.75rem 0.5rem;
      cursor: pointer;
      font-size: 0.875rem;
      font-weight: 500;
      color: var(--text-secondary-color, #6b7280);
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
      text-align: center;
    }

    .sidebar-tab:hover {
      color: var(--text-primary-color, #1f2937);
      background: var(--hover-color, #f9fafb);
    }

    .sidebar-tab.active {
      color: var(--primary-color, #3b82f6);
      border-bottom-color: var(--primary-color, #3b82f6);
      background: white;
    }

    .sidebar-panel {
      flex: 1;
      overflow: hidden;
      display: none;
    }

    .sidebar-panel.active {
      display: flex;
      flex-direction: column;
    }

    /* Main Content Styles */
    .dashboard-content {
      grid-area: content;
      background: var(--surface-color, #ffffff);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .content-header {
      padding: 1.5rem;
      border-bottom: 1px solid var(--border-color, #e5e7eb);
      background: white;
    }

    .content-title {
      margin: 0 0 0.5rem 0;
      font-size: 1.5rem;
      font-weight: 600;
      color: var(--text-primary-color, #1f2937);
    }

    .content-subtitle {
      margin: 0;
      color: var(--text-secondary-color, #6b7280);
      font-size: 0.875rem;
    }

    .content-main {
      flex: 1;
      overflow: auto;
      padding: 1.5rem;
    }

    /* Empty State Styles */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      text-align: center;
      color: var(--text-secondary-color, #6b7280);
    }

    .empty-icon {
      width: 64px;
      height: 64px;
      opacity: 0.5;
      margin-bottom: 1rem;
    }

    .empty-title {
      font-size: 1.25rem;
      font-weight: 600;
      margin: 0 0 0.5rem 0;
      color: var(--text-primary-color, #1f2937);
    }

    .empty-description {
      margin: 0 0 1.5rem 0;
      max-width: 400px;
    }

    .empty-action {
      background: var(--primary-color, #3b82f6);
      color: white;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 0.5rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .empty-action:hover {
      background: var(--primary-hover-color, #2563eb);
      transform: translateY(-1px);
    }

    /* Loading Overlay */
    .loading-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(255, 255, 255, 0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 20;
    }

    .loading-spinner {
      width: 32px;
      height: 32px;
      border: 3px solid var(--border-color, #e5e7eb);
      border-top: 3px solid var(--primary-color, #3b82f6);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .dashboard-layout {
        grid-template-areas: 
          "header"
          "content";
        grid-template-columns: 1fr;
        grid-template-rows: var(--header-height) 1fr;
      }

      .dashboard-sidebar {
        position: fixed;
        top: var(--header-height);
        left: 0;
        width: var(--sidebar-width);
        height: calc(100vh - var(--header-height));
        z-index: 30;
        transform: translateX(-100%);
        transition: transform 0.3s ease;
      }

      .dashboard-sidebar.mobile-open {
        transform: translateX(0);
      }

      .project-name {
        max-width: 120px;
      }

      .header-title {
        font-size: 1.125rem;
      }
    }

    @media (max-width: 480px) {
      .dashboard-header {
        padding: 0 1rem;
      }

      .content-header {
        padding: 1rem;
      }

      .content-main {
        padding: 1rem;
      }

      .project-name {
        max-width: 80px;
      }
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `;

  private setupStoreSubscriptions(): void {
    this.subscriptions = [
      this.store.onStateChanged((state: ProjectIndexState) => {
        this.projects = state.projects;
        this.selectedProject = state.selectedProject;
        this.currentSession = state.currentSession;
        this.loadingStates = state.loadingStates;
      }),
      
      this.store.onProjectSelected((project: ProjectIndex) => {
        this.selectedProject = project;
        this.selectedProjectId = project.id;
      }),
      
      this.store.onError((error: Error) => {
        this.showErrorNotification(error.message);
      })
    ];
  }

  private cleanupSubscriptions(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
    this.subscriptions = [];
  }

  private async loadInitialData(): Promise<void> {
    try {
      await this.store.loadProjects();
      
      // Auto-select project if specified
      if (this.selectedProjectId && !this.selectedProject) {
        await this.store.selectProject(this.selectedProjectId);
      }
    } catch (error) {
      console.error('Failed to load initial data:', error);
    }
  }

  private handleProjectSelect(projectId: string): void {
    this.store.selectProject(projectId);
    this.showProjectSelector = false;
  }

  private handleCreateProject(): void {
    this.showCreateProject = true;
  }

  private handleRefresh(): void {
    if (this.selectedProject) {
      this.store.selectProject(this.selectedProject.id);
    }
  }

  private handleSidebarToggle(): void {
    this.sidebarCollapsed = !this.sidebarCollapsed;
  }

  private handleTabChange(tab: string): void {
    this.currentTab = tab;
  }

  private showErrorNotification(message: string): void {
    // Emit error event for notification system
    this.dispatchEvent(new CustomEvent('error-notification', {
      detail: { message },
      bubbles: true,
      composed: true
    }));
  }

  private renderProjectSelector(): any {
    if (!this.selectedProject) {
      return html`
        <div class="project-selector" @click=${() => this.showProjectSelector = true}>
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
          <span>Select Project</span>
          <svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24">
            <path d="M7 10l5 5 5-5z"/>
          </svg>
        </div>
      `;
    }

    return html`
      <div class="project-selector" @click=${() => this.showProjectSelector = true}>
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
          <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
        </svg>
        <span class="project-name">${this.selectedProject.name}</span>
        <svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24">
          <path d="M7 10l5 5 5-5z"/>
        </svg>
      </div>
    `;
  }

  private renderEmptyState(): any {
    return html`
      <div class="empty-state">
        <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" 
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
        <h3 class="empty-title">No Project Selected</h3>
        <p class="empty-description">
          Choose a project to explore its structure, analyze dependencies, and optimize AI context.
        </p>
        <button class="empty-action" @click=${this.handleCreateProject}>
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 4v16m8-8H4"/>
          </svg>
          Create New Project
        </button>
      </div>
    `;
  }

  private renderSidebarContent(): any {
    return html`
      <div class="sidebar-tabs">
        <button class="sidebar-tab ${this.currentTab === 'files' ? 'active' : ''}"
                @click=${() => this.handleTabChange('files')}>
          Files
        </button>
        <button class="sidebar-tab ${this.currentTab === 'search' ? 'active' : ''}"
                @click=${() => this.handleTabChange('search')}>
          Search
        </button>
        <button class="sidebar-tab ${this.currentTab === 'metrics' ? 'active' : ''}"
                @click=${() => this.handleTabChange('metrics')}>
          Metrics
        </button>
      </div>

      <div class="sidebar-panel ${this.currentTab === 'files' ? 'active' : ''}">
        <file-structure-tree
          .projectId=${this.selectedProject?.id}
          .compact=${this.compact}
        ></file-structure-tree>
      </div>

      <div class="sidebar-panel ${this.currentTab === 'search' ? 'active' : ''}">
        <search-interface
          .projectId=${this.selectedProject?.id}
          .compact=${true}
        ></search-interface>
      </div>

      <div class="sidebar-panel ${this.currentTab === 'metrics' ? 'active' : ''}">
        <index-metrics
          .projectId=${this.selectedProject?.id}
          .compact=${true}
        ></index-metrics>
      </div>
    `;
  }

  private renderMainContent(): any {
    if (!this.selectedProject) {
      return this.renderEmptyState();
    }

    switch (this.activeView) {
      case 'dependencies':
        return html`
          <dependency-graph
            .projectId=${this.selectedProject.id}
            .fullScreen=${true}
          ></dependency-graph>
        `;
      
      case 'analysis':
        return html`
          <analysis-progress
            .projectId=${this.selectedProject.id}
            .session=${this.currentSession}
          ></analysis-progress>
        `;
      
      case 'context':
        return html`
          <context-optimizer
            .projectId=${this.selectedProject.id}
          ></context-optimizer>
        `;
      
      case 'file-detail':
        return html`
          <file-details
            .projectId=${this.selectedProject.id}
            .filePath=${this.selectedFilePath}
          ></file-details>
        `;
      
      default:
        return html`
          <div class="overview-grid">
            <dependency-graph
              .projectId=${this.selectedProject.id}
              .compact=${true}
            ></dependency-graph>
            
            <analysis-progress
              .projectId=${this.selectedProject.id}
              .session=${this.currentSession}
              .compact=${true}
            ></analysis-progress>
          </div>
        `;
    }
  }

  render() {
    const dashboardClasses = classMap({
      'dashboard-layout': true,
      'sidebar-collapsed': this.sidebarCollapsed
    });

    const isLoading = this.loadingStates['project-selection']?.isLoading;

    return html`
      <div class=${dashboardClasses}>
        <!-- Header -->
        <header class="dashboard-header">
          <div class="header-left">
            <h1 class="header-title">
              <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
                <path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z"/>
              </svg>
              Project Index
            </h1>
            ${this.renderProjectSelector()}
          </div>
          
          <div class="header-actions">
            ${this.selectedProject ? html`
              <button class="header-button" @click=${this.handleRefresh} title="Refresh">
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                </svg>
              </button>
              
              <button class="header-button primary" @click=${() => this.activeView = 'analysis'}>
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                </svg>
                Analyze
              </button>
            ` : ''}
            
            <button class="header-button" @click=${this.handleCreateProject} title="Create Project">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 4v16m8-8H4"/>
              </svg>
            </button>
          </div>
        </header>

        <!-- Sidebar -->
        <aside class="dashboard-sidebar">
          <div class="sidebar-header">
            <span class="sidebar-title">Explorer</span>
            <button class="sidebar-toggle" @click=${this.handleSidebarToggle}>
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                <path d="M15 19l-7-7 7-7"/>
              </svg>
            </button>
          </div>
          
          <div class="sidebar-content">
            ${this.selectedProject ? this.renderSidebarContent() : ''}
          </div>
        </aside>

        <!-- Main Content -->
        <main class="dashboard-content">
          ${this.selectedProject ? html`
            <div class="content-header">
              <h2 class="content-title">${this.selectedProject.name}</h2>
              <p class="content-subtitle">${this.selectedProject.description || 'Project analysis and insights'}</p>
            </div>
          ` : ''}
          
          <div class="content-main">
            ${this.renderMainContent()}
          </div>
          
          ${isLoading ? html`
            <div class="loading-overlay">
              <div class="loading-spinner"></div>
            </div>
          ` : ''}
        </main>
      </div>
    `;
  }
}