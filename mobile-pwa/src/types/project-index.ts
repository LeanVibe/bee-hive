/**
 * TypeScript interfaces for Project Index Dashboard components
 * 
 * These interfaces define the data structures used for project analysis,
 * file management, dependency tracking, and context optimization.
 */

// ===== BASE PROJECT INDEX TYPES =====

export interface ProjectIndex {
  id: string;
  name: string;
  description?: string;
  path: string;
  language: string;
  framework?: string;
  created_at: string;
  updated_at: string;
  last_analyzed_at?: string;
  status: ProjectStatus;
  metrics: ProjectMetrics;
  analysis_config: AnalysisConfig;
}

export const ProjectStatus = {
  CREATING: 'creating',
  ACTIVE: 'active',
  ANALYZING: 'analyzing',
  COMPLETED: 'completed',
  ERROR: 'error',
  ARCHIVED: 'archived'
} as const;

export type ProjectStatus = typeof ProjectStatus[keyof typeof ProjectStatus];

export interface ProjectMetrics {
  total_files: number;
  analyzed_files: number;
  total_lines: number;
  total_tokens: number;
  dependency_count: number;
  complexity_score: number;
  quality_score: number;
  last_update: string;
}

export interface AnalysisConfig {
  include_patterns: string[];
  exclude_patterns: string[];
  max_file_size: number;
  languages: string[];
  analysis_depth: 'basic' | 'detailed' | 'comprehensive';
  context_strategy: ContextStrategy;
}

export const ContextStrategy = {
  RELEVANCE_BASED: 'relevance_based',
  DEPENDENCY_AWARE: 'dependency_aware',
  TASK_OPTIMIZED: 'task_optimized',
  COMPREHENSIVE: 'comprehensive'
} as const;

export type ContextStrategy = typeof ContextStrategy[keyof typeof ContextStrategy];

// ===== FILE STRUCTURE TYPES =====

export interface ProjectFile {
  id: string;
  project_id: string;
  path: string;
  name: string;
  type: FileType;
  size: number;
  language: string;
  extension: string;
  content_hash: string;
  analyzed: boolean;
  analysis_metadata?: FileAnalysisMetadata;
  created_at: string;
  updated_at: string;
  last_modified: string;
}

export const FileType = {
  FILE: 'file',
  DIRECTORY: 'directory',
  SYMLINK: 'symlink'
} as const;

export type FileType = typeof FileType[keyof typeof FileType];

export interface FileAnalysisMetadata {
  line_count: number;
  token_count: number;
  complexity_score: number;
  imports: string[];
  exports: string[];
  functions: FunctionInfo[];
  classes: ClassInfo[];
  dependencies: DependencyInfo[];
  quality_issues: QualityIssue[];
  semantic_summary: string;
}

export interface FunctionInfo {
  name: string;
  line_start: number;
  line_end: number;
  parameters: string[];
  return_type?: string;
  complexity: number;
  doc_string?: string;
}

export interface ClassInfo {
  name: string;
  line_start: number;
  line_end: number;
  methods: string[];
  properties: string[];
  inheritance: string[];
  doc_string?: string;
}

export interface DependencyInfo {
  name: string;
  type: 'import' | 'require' | 'include' | 'inherit';
  source: string;
  target: string;
  line_number: number;
  is_external: boolean;
}

export interface QualityIssue {
  type: 'warning' | 'error' | 'info';
  category: string;
  message: string;
  line_number?: number;
  severity: number;
  suggestion?: string;
}

// ===== DEPENDENCY GRAPH TYPES =====

export interface DependencyGraph {
  nodes: DependencyNode[];
  edges: DependencyEdge[];
  metrics: GraphMetrics;
  layout: GraphLayout;
}

export interface DependencyNode {
  id: string;
  label: string;
  type: NodeType;
  file_path: string;
  language: string;
  size: number;
  complexity: number;
  centrality: number;
  cluster_id?: string;
  position?: { x: number; y: number };
  metadata: Record<string, any>;
}

export const NodeType = {
  MODULE: 'module',
  CLASS: 'class',
  FUNCTION: 'function',
  VARIABLE: 'variable',
  EXTERNAL: 'external'
} as const;

export type NodeType = typeof NodeType[keyof typeof NodeType];

export interface DependencyEdge {
  id: string;
  source: string;
  target: string;
  type: DependencyType;
  weight: number;
  line_numbers: number[];
  metadata: Record<string, any>;
}

export const DependencyType = {
  IMPORT: 'import',
  INHERITANCE: 'inheritance',
  COMPOSITION: 'composition',
  CALL: 'call',
  REFERENCE: 'reference'
} as const;

export type DependencyType = typeof DependencyType[keyof typeof DependencyType];

export interface GraphMetrics {
  total_nodes: number;
  total_edges: number;
  connected_components: number;
  cycles: number;
  max_depth: number;
  avg_clustering_coefficient: number;
  density: number;
}

export interface GraphLayout {
  algorithm: 'force' | 'hierarchical' | 'circular' | 'tree';
  width: number;
  height: number;
  scale: number;
  center: { x: number; y: number };
}

// ===== ANALYSIS PROGRESS TYPES =====

export interface AnalysisSession {
  id: string;
  project_id: string;
  status: AnalysisStatus;
  progress: AnalysisProgress;
  configuration: AnalysisConfig;
  results?: AnalysisResults;
  error_message?: string;
  started_at: string;
  completed_at?: string;
  estimated_completion?: string;
}

export const AnalysisStatus = {
  QUEUED: 'queued',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled'
} as const;

export type AnalysisStatus = typeof AnalysisStatus[keyof typeof AnalysisStatus];

export interface AnalysisProgress {
  session_id: string;
  project_id: string;
  current_phase: AnalysisPhase;
  total_phases: number;
  current_file: string;
  files_processed: number;
  total_files: number;
  percentage: number;
  estimated_remaining: number;
  speed: AnalysisSpeed;
  errors: AnalysisError[];
}

export const AnalysisPhase = {
  INITIALIZATION: 'initialization',
  FILE_DISCOVERY: 'file_discovery',
  CONTENT_ANALYSIS: 'content_analysis',
  DEPENDENCY_RESOLUTION: 'dependency_resolution',
  QUALITY_ASSESSMENT: 'quality_assessment',
  CONTEXT_OPTIMIZATION: 'context_optimization',
  FINALIZATION: 'finalization'
} as const;

export type AnalysisPhase = typeof AnalysisPhase[keyof typeof AnalysisPhase];

export interface AnalysisSpeed {
  files_per_second: number;
  tokens_per_second: number;
  avg_processing_time: number;
  current_speed: number;
}

export interface AnalysisError {
  type: 'warning' | 'error' | 'critical';
  message: string;
  file_path?: string;
  line_number?: number;
  timestamp: string;
  recoverable: boolean;
}

export interface AnalysisResults {
  summary: AnalysisSummary;
  files: ProjectFile[];
  dependencies: DependencyGraph;
  quality_report: QualityReport;
  optimization_suggestions: OptimizationSuggestion[];
}

export interface AnalysisSummary {
  total_files_analyzed: number;
  total_lines_processed: number;
  total_tokens_processed: number;
  languages_detected: Record<string, number>;
  frameworks_detected: string[];
  complexity_distribution: Record<string, number>;
  quality_score: number;
  maintainability_index: number;
}

export interface QualityReport {
  overall_score: number;
  code_coverage: number;
  test_coverage: number;
  documentation_coverage: number;
  issues_by_severity: Record<string, number>;
  metrics: QualityMetrics;
  recommendations: QualityRecommendation[];
}

export interface QualityMetrics {
  cyclomatic_complexity: number;
  maintainability_index: number;
  technical_debt_ratio: number;
  code_duplication: number;
  test_density: number;
}

export interface QualityRecommendation {
  type: 'refactor' | 'test' | 'document' | 'optimize';
  priority: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affected_files: string[];
  estimated_effort: number;
  impact: string;
}

export interface OptimizationSuggestion {
  type: 'performance' | 'maintainability' | 'security' | 'documentation';
  title: string;
  description: string;
  file_path?: string;
  before_code?: string;
  after_code?: string;
  impact_score: number;
  effort_estimate: number;
}

// ===== CONTEXT OPTIMIZATION TYPES =====

export interface ContextOptimization {
  id: string;
  project_id: string;
  task_description: string;
  strategy: ContextStrategy;
  configuration: ContextConfig;
  results?: ContextResults;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
}

export interface ContextConfig {
  max_files: number;
  max_tokens: number;
  include_tests: boolean;
  include_docs: boolean;
  priority_patterns: string[];
  exclusion_patterns: string[];
  relevance_threshold: number;
  model_constraints: ModelConstraints;
}

export interface ModelConstraints {
  context_window: number;
  token_limit: number;
  model_type: string;
  capabilities: string[];
}

export interface ContextResults {
  selected_files: SelectedFile[];
  total_tokens: number;
  relevance_scores: Record<string, number>;
  optimization_metrics: OptimizationMetrics;
  explanation: string;
  alternatives: ContextAlternative[];
}

export interface SelectedFile {
  file_path: string;
  relevance_score: number;
  token_count: number;
  inclusion_reason: string;
  priority: number;
  sections?: FileSection[];
}

export interface FileSection {
  start_line: number;
  end_line: number;
  type: 'function' | 'class' | 'import' | 'comment';
  relevance: number;
  description: string;
}

export interface OptimizationMetrics {
  total_files_considered: number;
  files_included: number;
  token_utilization: number;
  relevance_coverage: number;
  diversity_score: number;
  completeness_score: number;
}

export interface ContextAlternative {
  id: string;
  name: string;
  description: string;
  file_count: number;
  token_count: number;
  pros: string[];
  cons: string[];
  use_cases: string[];
}

// ===== SEARCH INTERFACE TYPES =====

export interface SearchQuery {
  query: string;
  type: SearchType;
  filters: SearchFilters;
  options: SearchOptions;
}

export const SearchType = {
  TEXT: 'text',
  SEMANTIC: 'semantic',
  STRUCTURAL: 'structural',
  DEPENDENCY: 'dependency'
} as const;

export type SearchType = typeof SearchType[keyof typeof SearchType];

export interface SearchFilters {
  file_types?: string[];
  languages?: string[];
  directories?: string[];
  size_range?: { min: number; max: number };
  date_range?: { start: string; end: string };
  complexity_range?: { min: number; max: number };
}

export interface SearchOptions {
  case_sensitive: boolean;
  regex_enabled: boolean;
  include_content: boolean;
  include_comments: boolean;
  max_results: number;
  sort_by: 'relevance' | 'name' | 'size' | 'date';
  sort_order: 'asc' | 'desc';
}

export interface SearchResults {
  query: SearchQuery;
  total_results: number;
  execution_time: number;
  results: SearchResult[];
  facets: SearchFacets;
  suggestions: string[];
}

export interface SearchResult {
  file_path: string;
  relevance_score: number;
  matches: SearchMatch[];
  context: SearchContext;
  metadata: SearchResultMetadata;
}

export interface SearchMatch {
  line_number: number;
  column_start: number;
  column_end: number;
  matched_text: string;
  context_before: string;
  context_after: string;
  match_type: string;
}

export interface SearchContext {
  function_name?: string;
  class_name?: string;
  section_type: string;
  surrounding_code: string;
}

export interface SearchResultMetadata {
  file_size: number;
  last_modified: string;
  language: string;
  complexity: number;
  quality_score: number;
}

export interface SearchFacets {
  languages: Record<string, number>;
  file_types: Record<string, number>;
  directories: Record<string, number>;
  complexity_ranges: Record<string, number>;
  date_ranges: Record<string, number>;
}

// ===== PERFORMANCE METRICS TYPES =====

export interface PerformanceMetrics {
  analysis_performance: AnalysisPerformanceMetrics;
  system_resources: SystemResourceMetrics;
  api_performance: ApiPerformanceMetrics;
  user_interaction: UserInteractionMetrics;
}

export interface AnalysisPerformanceMetrics {
  avg_analysis_time: number;
  files_per_second: number;
  tokens_per_second: number;
  cache_hit_rate: number;
  error_rate: number;
  memory_usage: number;
  cpu_usage: number;
}

export interface SystemResourceMetrics {
  memory_total: number;
  memory_used: number;
  cpu_cores: number;
  cpu_utilization: number;
  disk_space_total: number;
  disk_space_used: number;
  network_latency: number;
}

export interface ApiPerformanceMetrics {
  avg_response_time: number;
  requests_per_second: number;
  success_rate: number;
  error_distribution: Record<string, number>;
  endpoint_performance: Record<string, EndpointMetrics>;
}

export interface EndpointMetrics {
  avg_response_time: number;
  request_count: number;
  error_count: number;
  p95_response_time: number;
  p99_response_time: number;
}

export interface UserInteractionMetrics {
  session_duration: number;
  pages_per_session: number;
  bounce_rate: number;
  feature_usage: Record<string, number>;
  user_flows: UserFlow[];
}

export interface UserFlow {
  name: string;
  steps: string[];
  completion_rate: number;
  avg_duration: number;
  drop_off_points: Record<string, number>;
}

// ===== WEBSOCKET EVENT TYPES =====

export interface ProjectIndexWebSocketMessage {
  type: ProjectIndexEventType;
  data: any;
  timestamp: string;
  correlation_id?: string;
}

export const ProjectIndexEventType = {
  PROJECT_INDEX_UPDATED: 'project_index_updated',
  ANALYSIS_PROGRESS: 'analysis_progress',
  DEPENDENCY_CHANGED: 'dependency_changed',
  CONTEXT_OPTIMIZED: 'context_optimized',
  FILE_CHANGE: 'file_change',
  DEPENDENCY_UPDATE: 'dependency_update',
  PROJECT_STATUS_CHANGE: 'project_status_change',
  PERFORMANCE_UPDATE: 'performance_update'
} as const;

export type ProjectIndexEventType = typeof ProjectIndexEventType[keyof typeof ProjectIndexEventType];

// ===== API REQUEST/RESPONSE TYPES =====

export interface CreateProjectRequest {
  name: string;
  description?: string;
  path: string;
  language?: string;
  framework?: string;
  analysis_config?: Partial<AnalysisConfig>;
}

export interface UpdateProjectRequest {
  name?: string;
  description?: string;
  analysis_config?: Partial<AnalysisConfig>;
}

export interface AnalyzeProjectRequest {
  analysis_config?: Partial<AnalysisConfig>;
  force_refresh?: boolean;
}

export interface OptimizeContextRequest {
  task_description: string;
  strategy?: ContextStrategy;
  configuration?: Partial<ContextConfig>;
}

export interface SearchProjectRequest {
  query: SearchQuery;
  project_id?: string;
}

// ===== UI COMPONENT STATE TYPES =====

export interface ComponentLoadingState {
  isLoading: boolean;
  error?: string;
  lastUpdated?: string;
}

export interface ProjectIndexState {
  projects: ProjectIndex[];
  selectedProject?: ProjectIndex;
  currentSession?: AnalysisSession;
  searchResults?: SearchResults;
  performanceMetrics?: PerformanceMetrics;
  loadingStates: Record<string, ComponentLoadingState>;
}

export interface FileTreeState {
  expandedNodes: Set<string>;
  selectedNodes: Set<string>;
  filteredFiles: ProjectFile[];
  sortBy: 'name' | 'size' | 'type' | 'modified';
  sortOrder: 'asc' | 'desc';
  searchTerm: string;
}

export interface DependencyGraphState {
  layout: GraphLayout;
  selectedNodes: Set<string>;
  highlightedPaths: string[][];
  filterOptions: {
    nodeTypes: Set<NodeType>;
    dependencyTypes: Set<DependencyType>;
    complexityRange: { min: number; max: number };
  };
  viewMode: 'overview' | 'focused' | 'detailed';
}

// ===== UTILITY TYPES =====

export type EventCallback<T = any> = (data: T) => void;

export interface Subscription {
  unsubscribe: () => void;
}

export interface PaginationOptions {
  page: number;
  limit: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface FilterOptions {
  language?: string;
  file_type?: string;
  complexity_range?: { min: number; max: number };
  size_range?: { min: number; max: number };
  search_term?: string;
}