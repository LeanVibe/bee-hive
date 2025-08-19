/**
 * Technical Debt Service
 * 
 * Handles communication with the technical debt API endpoints,
 * providing data management and real-time updates for debt analysis.
 */

import { EventEmitter } from 'events'
import { DebtItem, DebtMetrics, ProjectDebtStatus } from '../components/dashboard/technical-debt-panel'

export interface DebtAnalysisRequest {
  include_advanced_patterns: boolean
  include_historical_analysis: boolean
  analysis_depth: 'quick' | 'standard' | 'comprehensive'
  file_patterns?: string[]
  exclude_patterns?: string[]
}

export interface DebtAnalysisResponse {
  project_id: string
  analysis_id: string
  total_debt_score: number
  debt_items: DebtItem[]
  category_breakdown: Record<string, number>
  severity_breakdown: Record<string, number>
  file_count: number
  lines_of_code: number
  analysis_duration_seconds: number
  recommendations: string[]
  analysis_timestamp: string
  advanced_patterns_detected?: number
}

export interface HistoricalAnalysisResponse {
  project_id: string
  lookback_days: number
  evolution_timeline: Array<{
    date: string
    total_debt_score: number
    category_scores: Record<string, number>
    files_analyzed: number
    lines_of_code: number
    debt_items_count: number
    debt_delta: number
    commit_hash: string
    commit_message: string
    author: string
  }>
  trend_analysis: {
    trend_direction: 'increasing' | 'decreasing' | 'stable'
    trend_strength: number
    velocity: number
    acceleration: number
    projected_debt_30_days: number
    projected_debt_90_days: number
    confidence_level: number
    seasonal_patterns: any[]
    risk_level: string
  }
  debt_hotspots: Array<{
    file_path: string
    debt_score: number
    debt_velocity: number
    stability_risk: number
    contributor_count: number
    priority: string
    categories_affected: string[]
    recommendations: string[]
  }>
  category_trends: Record<string, any>
  recommendations: string[]
  analysis_timestamp: string
}

export interface RemediationPlanResponse {
  project_id: string
  plan_id: string
  scope: string
  target_path: string
  recommendations_count: number
  execution_phases: string[][]
  total_debt_reduction: number
  total_effort_estimate: number
  total_risk_score: number
  estimated_duration_days: number
  immediate_actions: string[]
  quick_wins: string[]
  long_term_goals: string[]
  success_criteria: string[]
  potential_blockers: string[]
  created_at: string
}

export interface MonitoringStatusResponse {
  enabled: boolean
  active_since?: string
  monitored_projects_count: number
  total_files_monitored: number
  total_debt_events: number
  configuration: Record<string, any>
  projects: Record<string, any>
}

class TechnicalDebtService extends EventEmitter {
  private static instance: TechnicalDebtService
  private baseUrl: string
  private cache: Map<string, any>
  private cacheExpiry: Map<string, number>
  private readonly CACHE_DURATION = 5 * 60 * 1000 // 5 minutes

  constructor() {
    super()
    this.baseUrl = process.env.NODE_ENV === 'production' 
      ? '/api/technical-debt' 
      : 'http://localhost:8000/api/technical-debt'
    this.cache = new Map()
    this.cacheExpiry = new Map()
  }

  static getInstance(): TechnicalDebtService {
    if (!TechnicalDebtService.instance) {
      TechnicalDebtService.instance = new TechnicalDebtService()
    }
    return TechnicalDebtService.instance
  }

  /**
   * Check API health status
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`)
      const data = await response.json()
      return data.success === true
    } catch (error) {
      console.error('Technical debt API health check failed:', error)
      return false
    }
  }

  /**
   * Analyze technical debt for a project
   */
  async analyzeProject(
    projectId: string, 
    request: DebtAnalysisRequest
  ): Promise<DebtAnalysisResponse> {
    const cacheKey = `analysis_${projectId}_${JSON.stringify(request)}`
    const cached = this.getFromCache(cacheKey)
    if (cached) return cached

    const response = await fetch(`${this.baseUrl}/${projectId}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    })

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    if (!data.success) {
      throw new Error(`Analysis failed: ${data.message}`)
    }

    const result = data.data as DebtAnalysisResponse
    this.setCache(cacheKey, result)
    
    // Emit event for real-time updates
    this.emit('debt-analysis-completed', { projectId, result })
    
    return result
  }

  /**
   * Get historical debt analysis
   */
  async getDebtHistory(
    projectId: string,
    lookbackDays: number = 90,
    sampleFrequencyDays: number = 7
  ): Promise<HistoricalAnalysisResponse> {
    const cacheKey = `history_${projectId}_${lookbackDays}_${sampleFrequencyDays}`
    const cached = this.getFromCache(cacheKey)
    if (cached) return cached

    const response = await fetch(
      `${this.baseUrl}/${projectId}/history?lookback_days=${lookbackDays}&sample_frequency_days=${sampleFrequencyDays}`
    )

    if (!response.ok) {
      throw new Error(`History retrieval failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    if (!data.success) {
      throw new Error(`History retrieval failed: ${data.message}`)
    }

    const result = data.data as HistoricalAnalysisResponse
    this.setCache(cacheKey, result)
    
    return result
  }

  /**
   * Generate remediation plan
   */
  async generateRemediationPlan(
    projectId: string,
    scope: 'project' | 'file' | 'directory' = 'project',
    targetPath?: string
  ): Promise<RemediationPlanResponse> {
    const cacheKey = `remediation_${projectId}_${scope}_${targetPath || 'all'}`
    const cached = this.getFromCache(cacheKey)
    if (cached) return cached

    let url = `${this.baseUrl}/${projectId}/remediation-plan?scope=${scope}`
    if (targetPath) {
      url += `&target_path=${encodeURIComponent(targetPath)}`
    }

    const response = await fetch(url, { method: 'POST' })

    if (!response.ok) {
      throw new Error(`Remediation plan generation failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    if (!data.success) {
      throw new Error(`Remediation plan generation failed: ${data.message}`)
    }

    const result = data.data as RemediationPlanResponse
    this.setCache(cacheKey, result)
    
    return result
  }

  /**
   * Get file-specific recommendations
   */
  async getFileRecommendations(projectId: string, filePath: string): Promise<any[]> {
    const cacheKey = `file_recommendations_${projectId}_${filePath}`
    const cached = this.getFromCache(cacheKey)
    if (cached) return cached

    const encodedPath = encodeURIComponent(filePath)
    const response = await fetch(`${this.baseUrl}/${projectId}/recommendations/${encodedPath}`)

    if (!response.ok) {
      throw new Error(`File recommendations failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    if (!data.success) {
      throw new Error(`File recommendations failed: ${data.message}`)
    }

    const result = data.data.recommendations
    this.setCache(cacheKey, result)
    
    return result
  }

  /**
   * Get debt monitoring status
   */
  async getMonitoringStatus(projectId: string): Promise<MonitoringStatusResponse> {
    const cacheKey = `monitoring_${projectId}`
    const cached = this.getFromCache(cacheKey)
    if (cached) return cached

    const response = await fetch(`${this.baseUrl}/${projectId}/monitoring/status`)

    if (!response.ok) {
      throw new Error(`Monitoring status failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    if (!data.success) {
      throw new Error(`Monitoring status failed: ${data.message}`)
    }

    const result = data.data as MonitoringStatusResponse
    this.setCache(cacheKey, result, 30000) // Cache for 30 seconds only
    
    return result
  }

  /**
   * Start debt monitoring for a project
   */
  async startMonitoring(projectId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/${projectId}/monitoring/start`, {
      method: 'POST'
    })

    if (!response.ok) {
      throw new Error(`Start monitoring failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    if (!data.success) {
      throw new Error(`Start monitoring failed: ${data.message}`)
    }

    // Clear monitoring cache to get fresh status
    this.clearCache(`monitoring_${projectId}`)
    
    this.emit('monitoring-started', { projectId })
  }

  /**
   * Stop debt monitoring for a project
   */
  async stopMonitoring(projectId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/${projectId}/monitoring/stop`, {
      method: 'POST'
    })

    if (!response.ok) {
      throw new Error(`Stop monitoring failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    if (!data.success) {
      throw new Error(`Stop monitoring failed: ${data.message}`)
    }

    // Clear monitoring cache to get fresh status
    this.clearCache(`monitoring_${projectId}`)
    
    this.emit('monitoring-stopped', { projectId })
  }

  /**
   * Force immediate debt analysis
   */
  async forceAnalysis(projectId: string, filePaths?: string[]): Promise<any> {
    let url = `${this.baseUrl}/${projectId}/analyze/force`
    if (filePaths && filePaths.length > 0) {
      const params = filePaths.map(path => `file_paths=${encodeURIComponent(path)}`).join('&')
      url += `?${params}`
    }

    const response = await fetch(url, { method: 'POST' })

    if (!response.ok) {
      throw new Error(`Force analysis failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    if (!data.success) {
      throw new Error(`Force analysis failed: ${data.message}`)
    }

    // Clear related caches to get fresh data
    this.clearProjectCaches(projectId)
    
    this.emit('force-analysis-completed', { projectId, result: data.data })
    
    return data.data
  }

  /**
   * Convert API response to ProjectDebtStatus format for UI
   */
  convertToProjectStatus(
    projectId: string,
    projectName: string,
    analysisResult: DebtAnalysisResponse,
    historyResult?: HistoricalAnalysisResponse
  ): ProjectDebtStatus {
    return {
      project_id: projectId,
      project_name: projectName,
      debt_metrics: {
        total_debt_score: analysisResult.total_debt_score,
        debt_items_count: analysisResult.debt_items.length,
        category_breakdown: analysisResult.category_breakdown,
        severity_breakdown: analysisResult.severity_breakdown,
        file_count: analysisResult.file_count,
        lines_of_code: analysisResult.lines_of_code,
        debt_trend: historyResult?.trend_analysis.trend_direction || 'stable',
        debt_velocity: historyResult?.trend_analysis.velocity || 0,
        hotspot_files: historyResult?.debt_hotspots.map(h => h.file_path) || []
      },
      debt_items: analysisResult.debt_items,
      last_analyzed_at: new Date(analysisResult.analysis_timestamp),
      analysis_duration: analysisResult.analysis_duration_seconds,
      monitoring_enabled: false // Will be updated by monitoring status
    }
  }

  /**
   * Cache management methods
   */
  private getFromCache(key: string): any {
    const expiry = this.cacheExpiry.get(key)
    if (expiry && Date.now() > expiry) {
      this.cache.delete(key)
      this.cacheExpiry.delete(key)
      return null
    }
    return this.cache.get(key)
  }

  private setCache(key: string, value: any, duration = this.CACHE_DURATION): void {
    this.cache.set(key, value)
    this.cacheExpiry.set(key, Date.now() + duration)
  }

  private clearCache(key: string): void {
    this.cache.delete(key)
    this.cacheExpiry.delete(key)
  }

  private clearProjectCaches(projectId: string): void {
    for (const key of this.cache.keys()) {
      if (key.includes(projectId)) {
        this.clearCache(key)
      }
    }
  }

  /**
   * Clear all cached data
   */
  clearAllCaches(): void {
    this.cache.clear()
    this.cacheExpiry.clear()
  }
}

export default TechnicalDebtService