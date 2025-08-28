/**
 * Business Analytics Composable for Epic 11
 * 
 * Provides business intelligence data and user-facing insights
 * for enterprise-grade dashboard and decision support.
 */

import { ref, reactive, computed } from 'vue'
import { useApi } from './useApi'
import { useNotifications } from './useNotifications'

// Types
interface BusinessKPI {
  key: string
  label: string
  value: number
  format: 'percentage' | 'currency' | 'number' | 'duration'
  trend: 'up' | 'down' | 'stable'
  change: number
  comparison: string
  icon: string
}

interface PerformanceTrend {
  timestamp: string
  taskThroughput: number
  responseTime: number
  systemUtilization: number
  errorRate: number
  businessValue: number
}

interface BusinessInsight {
  id: string
  title: string
  description: string
  priority: 'high' | 'medium' | 'low'
  impact: string
  category: 'performance' | 'cost' | 'capacity' | 'quality'
}

interface RecommendedAction {
  id: string
  title: string
  description: string
  category: 'optimization' | 'scaling' | 'configuration' | 'maintenance'
  impact: 'high' | 'medium' | 'low'
  icon: string
}

interface BusinessAnalyticsParams {
  timeframe?: string
  includeProjections?: boolean
  includeCostAnalysis?: boolean
}

// State
const businessKPIs = ref<BusinessKPI[]>([])
const performanceTrends = ref<PerformanceTrend[]>([])
const topInsights = ref<BusinessInsight[]>([])
const recommendedActions = ref<RecommendedAction[]>([])
const businessMetrics = reactive({
  totalRevenue: 0,
  costPerTask: 0,
  resourceEfficiency: 0,
  customerSatisfaction: 0,
  systemReliability: 0
})

const loading = ref(false)
const error = ref<string | null>(null)

export function useBusinessAnalytics() {
  const { get, post } = useApi()
  const { addNotification } = useNotifications()

  /**
   * Fetch comprehensive business intelligence data
   */
  const fetchBusinessIntelligence = async (params: BusinessAnalyticsParams = {}) => {
    loading.value = true
    error.value = null

    try {
      // Fetch business KPIs
      const kpiResponse = await get('/api/analytics/kpis/executive-summary', {
        params: {
          timeframe: params.timeframe || '24h',
          include_trends: true
        }
      })

      businessKPIs.value = transformKPIData(kpiResponse.data.kpis)
      Object.assign(businessMetrics, kpiResponse.data.metrics)

      // Fetch performance trends
      const trendsResponse = await get('/api/analytics/performance/business-trends', {
        params: {
          timeframe: params.timeframe || '24h',
          granularity: getGranularity(params.timeframe || '24h')
        }
      })

      performanceTrends.value = trendsResponse.data.trends

      // Fetch business insights
      const insightsResponse = await get('/api/analytics/insights/top-recommendations', {
        params: {
          limit: 5,
          priority: 'high,medium',
          category: 'all'
        }
      })

      topInsights.value = insightsResponse.data.insights
      recommendedActions.value = insightsResponse.data.actions

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch business analytics'
      error.value = errorMessage
      addNotification({
        type: 'error',
        title: 'Business Analytics Error',
        message: errorMessage
      })
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * Fetch cost analysis and optimization data
   */
  const fetchCostAnalysis = async (timeframe: string = '30d') => {
    try {
      const response = await get('/api/analytics/optimization/cost-analysis', {
        params: { timeframe }
      })

      return {
        totalCost: response.data.total_cost,
        costBreakdown: response.data.breakdown,
        optimizationOpportunities: response.data.optimization_opportunities,
        projectedSavings: response.data.projected_savings
      }
    } catch (err) {
      console.error('Failed to fetch cost analysis:', err)
      throw err
    }
  }

  /**
   * Fetch capacity planning predictions
   */
  const fetchCapacityPredictions = async (horizon: string = '30d') => {
    try {
      const response = await get('/api/analytics/predictions/capacity-planning', {
        params: { 
          horizon,
          confidence_level: 0.95 
        }
      })

      return {
        currentCapacity: response.data.current_capacity,
        projectedDemand: response.data.projected_demand,
        scalingRecommendations: response.data.scaling_recommendations,
        riskFactors: response.data.risk_factors
      }
    } catch (err) {
      console.error('Failed to fetch capacity predictions:', err)
      throw err
    }
  }

  /**
   * Execute a recommended business action
   */
  const executeBusinessAction = async (actionId: string, params: Record<string, any> = {}) => {
    try {
      const response = await post(`/api/analytics/actions/${actionId}/execute`, params)
      
      addNotification({
        type: 'success',
        title: 'Action Executed',
        message: response.data.message || 'Business action executed successfully'
      })

      // Refresh insights after action execution
      await fetchBusinessIntelligence()
      
      return response.data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to execute action'
      addNotification({
        type: 'error',
        title: 'Action Failed',
        message: errorMessage
      })
      throw err
    }
  }

  /**
   * Get business ROI calculations
   */
  const calculateROI = async (period: string = '30d') => {
    try {
      const response = await get('/api/analytics/roi/calculate', {
        params: { period }
      })

      return {
        roi: response.data.roi,
        totalInvestment: response.data.total_investment,
        totalReturn: response.data.total_return,
        paybackPeriod: response.data.payback_period,
        breakdown: response.data.breakdown
      }
    } catch (err) {
      console.error('Failed to calculate ROI:', err)
      throw err
    }
  }

  // Helper functions
  const transformKPIData = (rawKPIs: any[]): BusinessKPI[] => {
    return rawKPIs.map(kpi => ({
      key: kpi.key,
      label: kpi.display_name,
      value: kpi.current_value,
      format: kpi.format_type,
      trend: determineTrend(kpi.current_value, kpi.previous_value),
      change: calculateChangePercentage(kpi.current_value, kpi.previous_value),
      comparison: kpi.comparison_period,
      icon: getKPIIcon(kpi.key)
    }))
  }

  const determineTrend = (current: number, previous: number): 'up' | 'down' | 'stable' => {
    const threshold = 0.02 // 2% threshold for considering change significant
    const change = (current - previous) / previous
    
    if (Math.abs(change) < threshold) return 'stable'
    return change > 0 ? 'up' : 'down'
  }

  const calculateChangePercentage = (current: number, previous: number): number => {
    if (previous === 0) return 0
    return (current - previous) / previous
  }

  const getKPIIcon = (key: string): string => {
    const iconMap: Record<string, string> = {
      'task_throughput': 'ChartBarIcon',
      'response_time': 'ClockIcon', 
      'system_utilization': 'CpuChipIcon',
      'cost_efficiency': 'CurrencyDollarIcon',
      'user_satisfaction': 'StarIcon',
      'error_rate': 'ExclamationTriangleIcon'
    }
    return iconMap[key] || 'ChartBarIcon'
  }

  const getGranularity = (timeframe: string): string => {
    const granularityMap: Record<string, string> = {
      '1h': '5m',
      '24h': '1h',
      '7d': '6h',
      '30d': '1d'
    }
    return granularityMap[timeframe] || '1h'
  }

  // Computed properties
  const businessHealth = computed(() => {
    if (businessKPIs.value.length === 0) return 'unknown'
    
    const upTrends = businessKPIs.value.filter(kpi => kpi.trend === 'up').length
    const totalKPIs = businessKPIs.value.length
    const healthRatio = upTrends / totalKPIs
    
    if (healthRatio >= 0.7) return 'excellent'
    if (healthRatio >= 0.5) return 'good'
    if (healthRatio >= 0.3) return 'fair'
    return 'needs-attention'
  })

  const businessHealthColor = computed(() => {
    const colorMap: Record<string, string> = {
      'excellent': 'text-green-600',
      'good': 'text-blue-600',
      'fair': 'text-yellow-600',
      'needs-attention': 'text-red-600',
      'unknown': 'text-gray-500'
    }
    return colorMap[businessHealth.value] || 'text-gray-500'
  })

  const keyMetricsSummary = computed(() => {
    return {
      efficiency: businessMetrics.resourceEfficiency,
      reliability: businessMetrics.systemReliability,
      satisfaction: businessMetrics.customerSatisfaction,
      costOptimization: businessMetrics.costPerTask
    }
  })

  return {
    // State
    businessKPIs: readonly(businessKPIs),
    performanceTrends: readonly(performanceTrends),
    topInsights: readonly(topInsights),
    recommendedActions: readonly(recommendedActions),
    businessMetrics: readonly(businessMetrics),
    loading: readonly(loading),
    error: readonly(error),

    // Computed
    businessHealth,
    businessHealthColor,
    keyMetricsSummary,

    // Methods
    fetchBusinessIntelligence,
    fetchCostAnalysis,
    fetchCapacityPredictions,
    executeBusinessAction,
    calculateROI
  }
}

// Export types for use in components
export type {
  BusinessKPI,
  PerformanceTrend,
  BusinessInsight,
  RecommendedAction,
  BusinessAnalyticsParams
}