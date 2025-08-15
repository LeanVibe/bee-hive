/**
 * Project Index Service
 * 
 * Handles all API interactions for project analysis, file management,
 * dependency tracking, and context optimization features.
 */

import { BaseService } from './base-service';
import type { ApiResponse } from '../types/api';
import type {
  ProjectIndex,
  ProjectFile,
  DependencyGraph,
  AnalysisSession,
  AnalysisProgress,
  ContextOptimization,
  SearchQuery,
  SearchResults,
  PerformanceMetrics,
  CreateProjectRequest,
  UpdateProjectRequest,
  AnalyzeProjectRequest,
  OptimizeContextRequest,
  SearchProjectRequest,
  PaginationOptions,
  FilterOptions,
  Subscription,
  EventCallback
} from '../types/project-index';

export class ProjectIndexService extends BaseService {
  private static instance: ProjectIndexService;

  static getInstance(): ProjectIndexService {
    if (!ProjectIndexService.instance) {
      ProjectIndexService.instance = new ProjectIndexService({
        baseUrl: process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8000/api'
      });
    }
    return ProjectIndexService.instance;
  }

  // ===== PROJECT MANAGEMENT =====

  /**
   * Get all projects with optional filtering and pagination
   */
  async getProjects(options: PaginationOptions & FilterOptions = { page: 1, limit: 20 }): Promise<{
    projects: ProjectIndex[];
    total: number;
    page: number;
    limit: number;
  }> {
    const queryParams = this.buildQueryString({
      page: options.page,
      limit: options.limit,
      sort_by: options.sort_by,
      sort_order: options.sort_order,
      language: options.language,
      search_term: options.search_term
    });

    const response = await this.get<ApiResponse<{
      projects: ProjectIndex[];
      total: number;
      page: number;
      limit: number;
    }>>(
      `/project-index${queryParams}`,
      {},
      `projects-${JSON.stringify(options)}`
    );

    return response.data!;
  }

  /**
   * Get specific project by ID
   */
  async getProject(projectId: string): Promise<ProjectIndex> {
    const response = await this.get<ApiResponse<ProjectIndex>>(
      `/project-index/${projectId}`,
      {},
      `project-${projectId}`
    );

    return response.data!;
  }

  /**
   * Create new project index
   */
  async createProject(request: CreateProjectRequest): Promise<ProjectIndex> {
    const response = await this.post<ApiResponse<ProjectIndex>>(
      '/project-index',
      request
    );

    // Clear project list cache
    this.clearCache('projects-');
    
    return response.data!;
  }

  /**
   * Update existing project
   */
  async updateProject(projectId: string, request: UpdateProjectRequest): Promise<ProjectIndex> {
    const response = await this.put<ApiResponse<ProjectIndex>>(
      `/project-index/${projectId}`,
      request
    );

    // Clear related caches
    this.clearCache(`project-${projectId}`);
    this.clearCache('projects-');

    return response.data!;
  }

  /**
   * Delete project
   */
  async deleteProject(projectId: string): Promise<void> {
    await this.delete(`/project-index/${projectId}`);

    // Clear related caches
    this.clearCache(`project-${projectId}`);
    this.clearCache('projects-');
  }

  /**
   * Refresh project analysis
   */
  async refreshProject(projectId: string): Promise<AnalysisSession> {
    const response = await this.put<ApiResponse<AnalysisSession>>(
      `/project-index/${projectId}/refresh`
    );

    // Clear all project-related caches
    this.clearCache(`project-${projectId}`);

    return response.data!;
  }

  // ===== FILE MANAGEMENT =====

  /**
   * Get project files with pagination and filtering
   */
  async getProjectFiles(
    projectId: string, 
    options: PaginationOptions & FilterOptions = { page: 1, limit: 50 }
  ): Promise<{
    files: ProjectFile[];
    total: number;
    page: number;
    limit: number;
  }> {
    const queryParams = this.buildQueryString({
      page: options.page,
      limit: options.limit,
      language: options.language,
      file_type: options.file_type,
      search_term: options.search_term
    });

    const response = await this.get<ApiResponse<{
      files: ProjectFile[];
      total: number;
      page: number;
      limit: number;
    }>>(
      `/project-index/${projectId}/files${queryParams}`,
      {},
      `files-${projectId}-${JSON.stringify(options)}`
    );

    return response.data!;
  }

  /**
   * Get specific file details
   */
  async getFile(projectId: string, filePath: string): Promise<ProjectFile> {
    const encodedPath = encodeURIComponent(filePath);
    const response = await this.get<ApiResponse<ProjectFile>>(
      `/project-index/${projectId}/files/${encodedPath}`,
      {},
      `file-${projectId}-${filePath}`
    );

    return response.data!;
  }

  /**
   * Get file content with syntax highlighting
   */
  async getFileContent(projectId: string, filePath: string): Promise<{
    content: string;
    highlighted: string;
    language: string;
    size: number;
  }> {
    const encodedPath = encodeURIComponent(filePath);
    const response = await this.get<ApiResponse<{
      content: string;
      highlighted: string;
      language: string;
      size: number;
    }>>(
      `/project-index/${projectId}/files/${encodedPath}/content`
    );

    return response.data!;
  }

  // ===== DEPENDENCY ANALYSIS =====

  /**
   * Get project dependency graph
   */
  async getDependencyGraph(
    projectId: string,
    format: 'json' | 'graph' = 'graph'
  ): Promise<DependencyGraph> {
    const response = await this.get<ApiResponse<DependencyGraph>>(
      `/project-index/${projectId}/dependencies?format=${format}`,
      {},
      `dependencies-${projectId}-${format}`
    );

    return response.data!;
  }

  /**
   * Get file dependencies
   */
  async getFileDependencies(
    projectId: string,
    filePath: string
  ): Promise<{
    incoming: string[];
    outgoing: string[];
    depth: number;
  }> {
    const encodedPath = encodeURIComponent(filePath);
    const response = await this.get<ApiResponse<{
      incoming: string[];
      outgoing: string[];
      depth: number;
    }>>(
      `/project-index/${projectId}/files/${encodedPath}/dependencies`
    );

    return response.data!;
  }

  // ===== ANALYSIS MANAGEMENT =====

  /**
   * Start project analysis
   */
  async analyzeProject(
    projectId: string,
    request: AnalyzeProjectRequest = {}
  ): Promise<AnalysisSession> {
    const response = await this.post<ApiResponse<AnalysisSession>>(
      `/project-index/${projectId}/analyze`,
      request
    );

    return response.data!;
  }

  /**
   * Get analysis session status
   */
  async getAnalysisSession(sessionId: string): Promise<AnalysisSession> {
    const response = await this.get<ApiResponse<AnalysisSession>>(
      `/project-index/sessions/${sessionId}`
    );

    return response.data!;
  }

  /**
   * Cancel analysis session
   */
  async cancelAnalysis(sessionId: string): Promise<void> {
    await this.post(`/project-index/sessions/${sessionId}/cancel`);
  }

  /**
   * Get analysis history for project
   */
  async getAnalysisHistory(
    projectId: string,
    options: PaginationOptions = { page: 1, limit: 10 }
  ): Promise<{
    sessions: AnalysisSession[];
    total: number;
    page: number;
    limit: number;
  }> {
    const queryParams = this.buildQueryString(options);
    const response = await this.get<ApiResponse<{
      sessions: AnalysisSession[];
      total: number;
      page: number;
      limit: number;
    }>>(
      `/project-index/${projectId}/analysis/history${queryParams}`
    );

    return response.data!;
  }

  // ===== CONTEXT OPTIMIZATION =====

  /**
   * Optimize context for AI models
   */
  async optimizeContext(
    projectId: string,
    request: OptimizeContextRequest
  ): Promise<ContextOptimization> {
    const response = await this.post<ApiResponse<ContextOptimization>>(
      `/project-index/${projectId}/optimize-context`,
      request
    );

    return response.data!;
  }

  /**
   * Get context optimization results
   */
  async getContextOptimization(optimizationId: string): Promise<ContextOptimization> {
    const response = await this.get<ApiResponse<ContextOptimization>>(
      `/project-index/optimizations/${optimizationId}`
    );

    return response.data!;
  }

  /**
   * Get context optimization history
   */
  async getContextHistory(
    projectId: string,
    options: PaginationOptions = { page: 1, limit: 10 }
  ): Promise<{
    optimizations: ContextOptimization[];
    total: number;
    page: number;
    limit: number;
  }> {
    const queryParams = this.buildQueryString(options);
    const response = await this.get<ApiResponse<{
      optimizations: ContextOptimization[];
      total: number;
      page: number;
      limit: number;
    }>>(
      `/project-index/${projectId}/context/history${queryParams}`
    );

    return response.data!;
  }

  // ===== SEARCH FUNCTIONALITY =====

  /**
   * Search within project
   */
  async searchProject(request: SearchProjectRequest): Promise<SearchResults> {
    const response = await this.post<ApiResponse<SearchResults>>(
      '/project-index/search',
      request
    );

    return response.data!;
  }

  /**
   * Get search suggestions
   */
  async getSearchSuggestions(
    projectId: string,
    query: string
  ): Promise<string[]> {
    const response = await this.get<ApiResponse<string[]>>(
      `/project-index/${projectId}/search/suggestions?q=${encodeURIComponent(query)}`
    );

    return response.data!;
  }

  // ===== PERFORMANCE METRICS =====

  /**
   * Get project performance metrics
   */
  async getPerformanceMetrics(projectId: string): Promise<PerformanceMetrics> {
    const response = await this.get<ApiResponse<PerformanceMetrics>>(
      `/project-index/${projectId}/metrics`,
      {},
      `metrics-${projectId}`
    );

    return response.data!;
  }

  /**
   * Get system-wide performance metrics
   */
  async getSystemMetrics(): Promise<PerformanceMetrics> {
    const response = await this.get<ApiResponse<PerformanceMetrics>>(
      '/project-index/system/metrics',
      {},
      'system-metrics'
    );

    return response.data!;
  }

  // ===== REAL-TIME SUBSCRIPTIONS =====

  /**
   * Subscribe to analysis progress updates
   */
  subscribeToAnalysisProgress(
    sessionId: string,
    callback: EventCallback<AnalysisProgress>
  ): Subscription {
    const eventName = `analysis_progress:${sessionId}`;
    return this.subscribe(eventName, callback);
  }

  /**
   * Subscribe to file change notifications
   */
  subscribeToFileChanges(
    projectId: string,
    callback: EventCallback<{ file_path: string; change_type: string }>
  ): Subscription {
    const eventName = `file_change:${projectId}`;
    return this.subscribe(eventName, callback);
  }

  /**
   * Subscribe to dependency updates
   */
  subscribeToDependencyUpdates(
    projectId: string,
    callback: EventCallback<{ update_type: string; details: any }>
  ): Subscription {
    const eventName = `dependency_update:${projectId}`;
    return this.subscribe(eventName, callback);
  }

  /**
   * Subscribe to project status changes
   */
  subscribeToProjectStatus(
    projectId: string,
    callback: EventCallback<{ status: string; message?: string }>
  ): Subscription {
    const eventName = `project_status_change:${projectId}`;
    return this.subscribe(eventName, callback);
  }

  // ===== UTILITY METHODS =====

  /**
   * Export project data
   */
  async exportProject(
    projectId: string,
    format: 'json' | 'csv' | 'xml' = 'json'
  ): Promise<Blob> {
    const response = await fetch(
      `${this.config.baseUrl}/project-index/${projectId}/export?format=${format}`,
      {
        headers: {
          'Authorization': `Bearer ${this.getAuthToken()}`
        }
      }
    );

    if (!response.ok) {
      throw new Error(`Export failed: ${response.statusText}`);
    }

    return response.blob();
  }

  /**
   * Import project data
   */
  async importProject(file: File): Promise<ProjectIndex> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(
      `${this.config.baseUrl}/project-index/import`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.getAuthToken()}`
        },
        body: formData
      }
    );

    if (!response.ok) {
      throw new Error(`Import failed: ${response.statusText}`);
    }

    const result = await response.json();
    
    // Clear project list cache
    this.clearCache('projects-');
    
    return result.data;
  }

  /**
   * Get project statistics
   */
  async getProjectStats(projectId: string): Promise<{
    file_count_by_language: Record<string, number>;
    complexity_distribution: Record<string, number>;
    size_distribution: Record<string, number>;
    recent_activity: Array<{
      date: string;
      files_analyzed: number;
      dependencies_found: number;
    }>;
  }> {
    const response = await this.get<ApiResponse<{
      file_count_by_language: Record<string, number>;
      complexity_distribution: Record<string, number>;
      size_distribution: Record<string, number>;
      recent_activity: Array<{
        date: string;
        files_analyzed: number;
        dependencies_found: number;
      }>;
    }>>(
      `/project-index/${projectId}/stats`,
      {},
      `stats-${projectId}`
    );

    return response.data!;
  }

  /**
   * Get auth token for API requests
   */
  private getAuthToken(): string {
    try {
      // Try to get token from auth service
      const { AuthService } = require('./auth');
      return AuthService.getInstance().getToken() || '';
    } catch {
      return '';
    }
  }

  // ===== POLLING SUPPORT =====

  /**
   * Start polling for analysis progress
   */
  startPollingAnalysisProgress(
    sessionId: string,
    callback: EventCallback<AnalysisProgress>,
    interval: number = 2000
  ): () => void {
    return this.startPolling(async () => {
      try {
        const session = await this.getAnalysisSession(sessionId);
        if (session.progress) {
          callback(session.progress);
        }
        
        // Stop polling if analysis is complete or failed
        if (['completed', 'failed', 'cancelled'].includes(session.status)) {
          // Return early to stop polling
        }
      } catch (error) {
        this.emit('error', error);
      }
    }, interval);
  }

  /**
   * Start polling for project metrics
   */
  startPollingMetrics(
    projectId: string,
    callback: EventCallback<PerformanceMetrics>,
    interval: number = 10000
  ): () => void {
    return this.startPolling(async () => {
      try {
        const metrics = await this.getPerformanceMetrics(projectId);
        callback(metrics);
      } catch (error) {
        this.emit('error', error);
      }
    }, interval);
  }

  // ===== BATCH OPERATIONS =====

  /**
   * Analyze multiple projects
   */
  async analyzeMultipleProjects(
    projectIds: string[],
    config?: Partial<AnalyzeProjectRequest>
  ): Promise<AnalysisSession[]> {
    const response = await this.post<ApiResponse<AnalysisSession[]>>(
      '/project-index/batch/analyze',
      {
        project_ids: projectIds,
        config
      }
    );

    return response.data!;
  }

  /**
   * Delete multiple projects
   */
  async deleteMultipleProjects(projectIds: string[]): Promise<void> {
    await this.post('/project-index/batch/delete', {
      project_ids: projectIds
    });

    // Clear related caches
    projectIds.forEach(id => {
      this.clearCache(`project-${id}`);
    });
    this.clearCache('projects-');
  }
}