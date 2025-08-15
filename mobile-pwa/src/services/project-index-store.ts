/**
 * Project Index Store
 * 
 * Manages state for project index dashboard components and provides
 * real-time updates via WebSocket integration.
 */

import { EventEmitter } from '../utils/event-emitter';
import { ProjectIndexService } from './project-index';
import { WebSocketService } from './websocket';
import type {
  ProjectIndex,
  ProjectFile,
  DependencyGraph,
  AnalysisSession,
  AnalysisProgress,
  ContextOptimization,
  SearchResults,
  PerformanceMetrics,
  ProjectIndexState,
  FileTreeState,
  DependencyGraphState,
  ComponentLoadingState,
  ProjectIndexEventType,
  PaginationOptions,
  FilterOptions,
  SearchQuery,
  EventCallback,
  Subscription
} from '../types/project-index';

export class ProjectIndexStore extends EventEmitter {
  private static instance: ProjectIndexStore;
  
  private projectService: ProjectIndexService;
  private websocketService: WebSocketService;
  
  // Core state
  private state: ProjectIndexState = {
    projects: [],
    selectedProject: undefined,
    currentSession: undefined,
    searchResults: undefined,
    performanceMetrics: undefined,
    loadingStates: {}
  };

  // Component-specific state
  private fileTreeState: FileTreeState = {
    expandedNodes: new Set(),
    selectedNodes: new Set(),
    filteredFiles: [],
    sortBy: 'name',
    sortOrder: 'asc',
    searchTerm: ''
  };

  private dependencyGraphState: DependencyGraphState = {
    layout: {
      algorithm: 'force',
      width: 800,
      height: 600,
      scale: 1,
      center: { x: 400, y: 300 }
    },
    selectedNodes: new Set(),
    highlightedPaths: [],
    filterOptions: {
      nodeTypes: new Set(['module', 'class', 'function'] as any[]),
      dependencyTypes: new Set(['import', 'inheritance', 'call'] as any[]),
      complexityRange: { min: 0, max: 100 }
    },
    viewMode: 'overview'
  };

  // WebSocket subscriptions
  private subscriptions: Map<string, Subscription> = new Map();
  
  // Polling intervals
  private pollingIntervals: Map<string, () => void> = new Map();

  static getInstance(): ProjectIndexStore {
    if (!ProjectIndexStore.instance) {
      ProjectIndexStore.instance = new ProjectIndexStore();
    }
    return ProjectIndexStore.instance;
  }

  constructor() {
    super();
    this.projectService = ProjectIndexService.getInstance();
    this.websocketService = WebSocketService.getInstance();
    this.setupWebSocketListeners();
  }

  // ===== STATE GETTERS =====

  getState(): ProjectIndexState {
    return { ...this.state };
  }

  getProjects(): ProjectIndex[] {
    return [...this.state.projects];
  }

  getSelectedProject(): ProjectIndex | undefined {
    return this.state.selectedProject;
  }

  getCurrentSession(): AnalysisSession | undefined {
    return this.state.currentSession;
  }

  getSearchResults(): SearchResults | undefined {
    return this.state.searchResults;
  }

  getPerformanceMetrics(): PerformanceMetrics | undefined {
    return this.state.performanceMetrics;
  }

  getLoadingState(key: string): ComponentLoadingState {
    return this.state.loadingStates[key] || { isLoading: false };
  }

  getFileTreeState(): FileTreeState {
    return { ...this.fileTreeState };
  }

  getDependencyGraphState(): DependencyGraphState {
    return {
      ...this.dependencyGraphState,
      selectedNodes: new Set(this.dependencyGraphState.selectedNodes),
      filterOptions: {
        ...this.dependencyGraphState.filterOptions,
        nodeTypes: new Set(this.dependencyGraphState.filterOptions.nodeTypes),
        dependencyTypes: new Set(this.dependencyGraphState.filterOptions.dependencyTypes)
      }
    };
  }

  // ===== PROJECT MANAGEMENT =====

  async loadProjects(options?: PaginationOptions & FilterOptions): Promise<void> {
    this.setLoadingState('projects', { isLoading: true });
    
    try {
      const result = await this.projectService.getProjects(options);
      
      this.updateState({
        projects: result.projects
      });
      
      this.setLoadingState('projects', { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });
      
      this.emit('projects-loaded', result);
    } catch (error) {
      this.setLoadingState('projects', { 
        isLoading: false, 
        error: error instanceof Error ? error.message : 'Failed to load projects' 
      });
      this.emit('error', error);
    }
  }

  async selectProject(projectId: string): Promise<void> {
    this.setLoadingState('project-selection', { isLoading: true });
    
    try {
      const project = await this.projectService.getProject(projectId);
      
      this.updateState({
        selectedProject: project
      });
      
      // Subscribe to project-specific events
      this.subscribeToProjectEvents(projectId);
      
      // Load initial project data
      await Promise.all([
        this.loadProjectFiles(projectId),
        this.loadDependencyGraph(projectId),
        this.loadPerformanceMetrics(projectId)
      ]);
      
      this.setLoadingState('project-selection', { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });
      
      this.emit('project-selected', project);
    } catch (error) {
      this.setLoadingState('project-selection', { 
        isLoading: false, 
        error: error instanceof Error ? error.message : 'Failed to select project' 
      });
      this.emit('error', error);
    }
  }

  async createProject(request: any): Promise<void> {
    this.setLoadingState('project-creation', { isLoading: true });
    
    try {
      const project = await this.projectService.createProject(request);
      
      this.updateState({
        projects: [...this.state.projects, project]
      });
      
      this.setLoadingState('project-creation', { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });
      
      this.emit('project-created', project);
    } catch (error) {
      this.setLoadingState('project-creation', { 
        isLoading: false, 
        error: error instanceof Error ? error.message : 'Failed to create project' 
      });
      this.emit('error', error);
    }
  }

  async deleteProject(projectId: string): Promise<void> {
    this.setLoadingState('project-deletion', { isLoading: true });
    
    try {
      await this.projectService.deleteProject(projectId);
      
      this.updateState({
        projects: this.state.projects.filter(p => p.id !== projectId),
        selectedProject: this.state.selectedProject?.id === projectId ? undefined : this.state.selectedProject
      });
      
      // Unsubscribe from project events
      this.unsubscribeFromProjectEvents(projectId);
      
      this.setLoadingState('project-deletion', { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });
      
      this.emit('project-deleted', projectId);
    } catch (error) {
      this.setLoadingState('project-deletion', { 
        isLoading: false, 
        error: error instanceof Error ? error.message : 'Failed to delete project' 
      });
      this.emit('error', error);
    }
  }

  // ===== FILE MANAGEMENT =====

  async loadProjectFiles(
    projectId: string, 
    options?: PaginationOptions & FilterOptions
  ): Promise<void> {
    this.setLoadingState('project-files', { isLoading: true });
    
    try {
      const result = await this.projectService.getProjectFiles(projectId, options);
      
      this.updateFileTreeState({
        filteredFiles: result.files
      });
      
      this.setLoadingState('project-files', { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });
      
      this.emit('files-loaded', result);
    } catch (error) {
      this.setLoadingState('project-files', { 
        isLoading: false, 
        error: error instanceof Error ? error.message : 'Failed to load files' 
      });
      this.emit('error', error);
    }
  }

  updateFileTreeState(updates: Partial<FileTreeState>): void {
    this.fileTreeState = { ...this.fileTreeState, ...updates };
    this.emit('file-tree-state-changed', this.fileTreeState);
  }

  toggleFileTreeNode(nodeId: string): void {
    const expandedNodes = new Set(this.fileTreeState.expandedNodes);
    
    if (expandedNodes.has(nodeId)) {
      expandedNodes.delete(nodeId);
    } else {
      expandedNodes.add(nodeId);
    }
    
    this.updateFileTreeState({ expandedNodes });
  }

  selectFileTreeNodes(nodeIds: string[], multiSelect = false): void {
    let selectedNodes: Set<string>;
    
    if (multiSelect) {
      selectedNodes = new Set(this.fileTreeState.selectedNodes);
      nodeIds.forEach(id => selectedNodes.add(id));
    } else {
      selectedNodes = new Set(nodeIds);
    }
    
    this.updateFileTreeState({ selectedNodes });
  }

  // ===== DEPENDENCY ANALYSIS =====

  async loadDependencyGraph(projectId: string): Promise<void> {
    this.setLoadingState('dependency-graph', { isLoading: true });
    
    try {
      const graph = await this.projectService.getDependencyGraph(projectId);
      
      this.emit('dependency-graph-loaded', graph);
      
      this.setLoadingState('dependency-graph', { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });
    } catch (error) {
      this.setLoadingState('dependency-graph', { 
        isLoading: false, 
        error: error instanceof Error ? error.message : 'Failed to load dependency graph' 
      });
      this.emit('error', error);
    }
  }

  updateDependencyGraphState(updates: Partial<DependencyGraphState>): void {
    this.dependencyGraphState = { ...this.dependencyGraphState, ...updates };
    this.emit('dependency-graph-state-changed', this.dependencyGraphState);
  }

  selectDependencyNodes(nodeIds: string[], multiSelect = false): void {
    let selectedNodes: Set<string>;
    
    if (multiSelect) {
      selectedNodes = new Set(this.dependencyGraphState.selectedNodes);
      nodeIds.forEach(id => selectedNodes.add(id));
    } else {
      selectedNodes = new Set(nodeIds);
    }
    
    this.updateDependencyGraphState({ selectedNodes });
  }

  highlightDependencyPaths(paths: string[][]): void {
    this.updateDependencyGraphState({ highlightedPaths: paths });
  }

  // ===== ANALYSIS MANAGEMENT =====

  async startAnalysis(projectId: string, config?: any): Promise<void> {
    this.setLoadingState('analysis-start', { isLoading: true });
    
    try {
      const session = await this.projectService.analyzeProject(projectId, config);
      
      this.updateState({ currentSession: session });
      
      // Start monitoring analysis progress
      this.startAnalysisPolling(session.id);
      
      this.setLoadingState('analysis-start', { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });
      
      this.emit('analysis-started', session);
    } catch (error) {
      this.setLoadingState('analysis-start', { 
        isLoading: false, 
        error: error instanceof Error ? error.message : 'Failed to start analysis' 
      });
      this.emit('error', error);
    }
  }

  async cancelAnalysis(sessionId: string): Promise<void> {
    try {
      await this.projectService.cancelAnalysis(sessionId);
      
      // Stop polling
      this.stopAnalysisPolling(sessionId);
      
      this.updateState({ 
        currentSession: this.state.currentSession?.id === sessionId ? undefined : this.state.currentSession 
      });
      
      this.emit('analysis-cancelled', sessionId);
    } catch (error) {
      this.emit('error', error);
    }
  }

  // ===== SEARCH FUNCTIONALITY =====

  async searchProject(query: SearchQuery): Promise<void> {
    this.setLoadingState('search', { isLoading: true });
    
    try {
      const results = await this.projectService.searchProject({ query });
      
      this.updateState({ searchResults: results });
      
      this.setLoadingState('search', { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });
      
      this.emit('search-completed', results);
    } catch (error) {
      this.setLoadingState('search', { 
        isLoading: false, 
        error: error instanceof Error ? error.message : 'Search failed' 
      });
      this.emit('error', error);
    }
  }

  clearSearchResults(): void {
    this.updateState({ searchResults: undefined });
    this.emit('search-cleared');
  }

  // ===== CONTEXT OPTIMIZATION =====

  async optimizeContext(projectId: string, request: any): Promise<void> {
    this.setLoadingState('context-optimization', { isLoading: true });
    
    try {
      const optimization = await this.projectService.optimizeContext(projectId, request);
      
      this.setLoadingState('context-optimization', { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });
      
      this.emit('context-optimized', optimization);
    } catch (error) {
      this.setLoadingState('context-optimization', { 
        isLoading: false, 
        error: error instanceof Error ? error.message : 'Context optimization failed' 
      });
      this.emit('error', error);
    }
  }

  // ===== PERFORMANCE METRICS =====

  async loadPerformanceMetrics(projectId: string): Promise<void> {
    this.setLoadingState('performance-metrics', { isLoading: true });
    
    try {
      const metrics = await this.projectService.getPerformanceMetrics(projectId);
      
      this.updateState({ performanceMetrics: metrics });
      
      this.setLoadingState('performance-metrics', { 
        isLoading: false, 
        lastUpdated: new Date().toISOString() 
      });
      
      this.emit('metrics-loaded', metrics);
    } catch (error) {
      this.setLoadingState('performance-metrics', { 
        isLoading: false, 
        error: error instanceof Error ? error.message : 'Failed to load metrics' 
      });
      this.emit('error', error);
    }
  }

  // ===== WEBSOCKET INTEGRATION =====

  private setupWebSocketListeners(): void {
    // Listen for project index specific events
    this.websocketService.on('project_index_updated', this.handleProjectIndexUpdated.bind(this));
    this.websocketService.on('analysis_progress', this.handleAnalysisProgress.bind(this));
    this.websocketService.on('dependency_changed', this.handleDependencyChanged.bind(this));
    this.websocketService.on('context_optimized', this.handleContextOptimized.bind(this));
    this.websocketService.on('file_change', this.handleFileChange.bind(this));
    this.websocketService.on('project_status_change', this.handleProjectStatusChange.bind(this));
    this.websocketService.on('performance_update', this.handlePerformanceUpdate.bind(this));
  }

  private subscribeToProjectEvents(projectId: string): void {
    const subscriptionKey = `project-${projectId}`;
    
    // Unsubscribe from previous project if any
    if (this.subscriptions.has(subscriptionKey)) {
      this.subscriptions.get(subscriptionKey)?.unsubscribe();
    }
    
    // Subscribe to new project events
    const subscription = this.projectService.subscribeToProjectStatus(
      projectId,
      this.handleProjectStatusChange.bind(this)
    );
    
    this.subscriptions.set(subscriptionKey, subscription);
  }

  private unsubscribeFromProjectEvents(projectId: string): void {
    const subscriptionKey = `project-${projectId}`;
    
    if (this.subscriptions.has(subscriptionKey)) {
      this.subscriptions.get(subscriptionKey)?.unsubscribe();
      this.subscriptions.delete(subscriptionKey);
    }
  }

  // ===== WEBSOCKET EVENT HANDLERS =====

  private handleProjectIndexUpdated(data: any): void {
    // Refresh project data
    if (this.state.selectedProject?.id === data.project_id) {
      this.selectProject(data.project_id);
    }
    this.emit('project-index-updated', data);
  }

  private handleAnalysisProgress(progress: AnalysisProgress): void {
    if (this.state.currentSession?.id === progress.session_id) {
      this.updateState({
        currentSession: {
          ...this.state.currentSession,
          progress
        }
      });
    }
    this.emit('analysis-progress-updated', progress);
  }

  private handleDependencyChanged(data: any): void {
    // Refresh dependency graph if current project
    if (this.state.selectedProject?.id === data.project_id) {
      this.loadDependencyGraph(data.project_id);
    }
    this.emit('dependency-changed', data);
  }

  private handleContextOptimized(data: any): void {
    this.emit('context-optimized', data);
  }

  private handleFileChange(data: any): void {
    // Refresh file list if current project
    if (this.state.selectedProject?.id === data.project_id) {
      this.loadProjectFiles(data.project_id);
    }
    this.emit('file-changed', data);
  }

  private handleProjectStatusChange(data: any): void {
    // Update project status
    if (this.state.selectedProject?.id === data.project_id) {
      this.updateState({
        selectedProject: {
          ...this.state.selectedProject,
          status: data.status
        }
      });
    }
    this.emit('project-status-changed', data);
  }

  private handlePerformanceUpdate(metrics: PerformanceMetrics): void {
    this.updateState({ performanceMetrics: metrics });
    this.emit('performance-updated', metrics);
  }

  // ===== POLLING MANAGEMENT =====

  private startAnalysisPolling(sessionId: string): void {
    const stopPolling = this.projectService.startPollingAnalysisProgress(
      sessionId,
      this.handleAnalysisProgress.bind(this),
      2000
    );
    
    this.pollingIntervals.set(`analysis-${sessionId}`, stopPolling);
  }

  private stopAnalysisPolling(sessionId: string): void {
    const stopPolling = this.pollingIntervals.get(`analysis-${sessionId}`);
    if (stopPolling) {
      stopPolling();
      this.pollingIntervals.delete(`analysis-${sessionId}`);
    }
  }

  // ===== UTILITY METHODS =====

  private updateState(updates: Partial<ProjectIndexState>): void {
    this.state = { ...this.state, ...updates };
    this.emit('state-changed', this.state);
  }

  private setLoadingState(key: string, state: ComponentLoadingState): void {
    this.state.loadingStates[key] = state;
    this.emit('loading-state-changed', { key, state });
  }

  // ===== EVENT SUBSCRIPTION HELPERS =====

  onProjectsLoaded(callback: EventCallback): Subscription {
    return this.subscribe('projects-loaded', callback);
  }

  onProjectSelected(callback: EventCallback): Subscription {
    return this.subscribe('project-selected', callback);
  }

  onAnalysisStarted(callback: EventCallback): Subscription {
    return this.subscribe('analysis-started', callback);
  }

  onAnalysisProgress(callback: EventCallback): Subscription {
    return this.subscribe('analysis-progress-updated', callback);
  }

  onFilesLoaded(callback: EventCallback): Subscription {
    return this.subscribe('files-loaded', callback);
  }

  onDependencyGraphLoaded(callback: EventCallback): Subscription {
    return this.subscribe('dependency-graph-loaded', callback);
  }

  onSearchCompleted(callback: EventCallback): Subscription {
    return this.subscribe('search-completed', callback);
  }

  onStateChanged(callback: EventCallback): Subscription {
    return this.subscribe('state-changed', callback);
  }

  onError(callback: EventCallback): Subscription {
    return this.subscribe('error', callback);
  }

  // ===== CLEANUP =====

  destroy(): void {
    // Stop all polling
    this.pollingIntervals.forEach(stopPolling => stopPolling());
    this.pollingIntervals.clear();
    
    // Unsubscribe from all WebSocket events
    this.subscriptions.forEach(subscription => subscription.unsubscribe());
    this.subscriptions.clear();
    
    // Clear all listeners
    this.removeAllListeners();
  }
}