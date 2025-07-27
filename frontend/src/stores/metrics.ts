import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiClient } from '@/services/api'

export interface SystemMetric {
  name: string
  value: number
  unit: string
  timestamp: string
  labels?: Record<string, string>
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy'
  components: Record<string, {
    status: string
    message?: string
    error?: string
    [key: string]: any
  }>
  timestamp: string
}

export interface MetricSeries {
  name: string
  data: Array<{
    timestamp: string
    value: number
  }>
  unit: string
}

export const useMetricsStore = defineStore('metrics', () => {
  // State
  const metrics = ref<SystemMetric[]>([])
  const healthStatus = ref<HealthStatus | null>(null)
  const metricSeries = ref<Record<string, MetricSeries>>({})
  const loading = ref(false)
  const error = ref<string | null>(null)
  const lastUpdated = ref<Date | null>(null)
  
  // Computed
  const overallHealth = computed(() => {
    if (!healthStatus.value) return 'unknown'
    return healthStatus.value.status
  })
  
  const componentHealth = computed(() => {
    if (!healthStatus.value) return {}
    return healthStatus.value.components
  })
  
  const healthyComponents = computed(() => {
    const components = componentHealth.value
    return Object.values(components).filter(c => c.status === 'healthy').length
  })
  
  const totalComponents = computed(() => {
    return Object.keys(componentHealth.value).length
  })
  
  const healthPercentage = computed(() => {
    if (totalComponents.value === 0) return 0
    return Math.round((healthyComponents.value / totalComponents.value) * 100)
  })
  
  const latestMetrics = computed(() => {
    const latest: Record<string, SystemMetric> = {}
    
    metrics.value.forEach(metric => {
      const key = `${metric.name}_${JSON.stringify(metric.labels || {})}`
      if (!latest[key] || new Date(metric.timestamp) > new Date(latest[key].timestamp)) {
        latest[key] = metric
      }
    })
    
    return Object.values(latest)
  })
  
  const performanceMetrics = computed(() => {
    return latestMetrics.value.filter(m => 
      m.name.includes('latency') || 
      m.name.includes('duration') || 
      m.name.includes('rate') ||
      m.name.includes('throughput')
    )
  })
  
  const systemMetrics = computed(() => {
    return latestMetrics.value.filter(m => 
      m.name.includes('cpu') || 
      m.name.includes('memory') || 
      m.name.includes('disk') ||
      m.name.includes('network')
    )
  })
  
  const eventMetrics = computed(() => {
    return latestMetrics.value.filter(m => 
      m.name.includes('event') || 
      m.name.includes('agent') ||
      m.name.includes('session')
    )
  })
  
  // Actions
  const fetchMetrics = async () => {
    loading.value = true
    error.value = null
    
    try {
      const response = await apiClient.get('/observability/metrics')
      
      // Parse Prometheus format metrics
      const parsed = parsePrometheusMetrics(response)
      metrics.value = parsed
      lastUpdated.value = new Date()
      
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch metrics'
      console.error('Failed to fetch metrics:', err)
    } finally {
      loading.value = false
    }
  }
  
  const fetchHealthStatus = async () => {
    try {
      const response = await apiClient.get('/observability/health')
      healthStatus.value = response
      
    } catch (err) {
      console.error('Failed to fetch health status:', err)
      healthStatus.value = {
        status: 'unhealthy',
        components: { error: { status: 'error', message: 'Failed to fetch health' } },
        timestamp: new Date().toISOString(),
      }
    }
  }
  
  const fetchSystemStatus = async () => {
    try {
      const response = await apiClient.get('/status')
      
      // Convert system status to health format
      const components: Record<string, any> = {}
      
      Object.entries(response.components).forEach(([name, component]: [string, any]) => {
        components[name] = {
          status: component.connected ? 'healthy' : 'unhealthy',
          ...component,
        }
      })
      
      healthStatus.value = {
        status: Object.values(components).every((c: any) => c.status === 'healthy') 
          ? 'healthy' 
          : 'degraded',
        components,
        timestamp: response.timestamp,
      }
      
    } catch (err) {
      console.error('Failed to fetch system status:', err)
    }
  }
  
  const fetchMetricSeries = async (metricName: string, _timeRange = '1h') => {
    try {
      // In a real implementation, this would fetch time series data
      // For now, we'll simulate it
      const now = new Date()
      const data = []
      
      for (let i = 0; i < 60; i++) {
        const timestamp = new Date(now.getTime() - i * 60 * 1000)
        data.push({
          timestamp: timestamp.toISOString(),
          value: Math.random() * 100 + Math.sin(i / 10) * 20,
        })
      }
      
      metricSeries.value[metricName] = {
        name: metricName,
        data: data.reverse(),
        unit: getMetricUnit(metricName),
      }
      
    } catch (err) {
      console.error(`Failed to fetch metric series for ${metricName}:`, err)
    }
  }
  
  const updateRealtimeMetric = (metric: SystemMetric) => {
    // Add or update metric
    const existingIndex = metrics.value.findIndex(m => 
      m.name === metric.name && 
      JSON.stringify(m.labels) === JSON.stringify(metric.labels)
    )
    
    if (existingIndex > -1) {
      metrics.value[existingIndex] = metric
    } else {
      metrics.value.push(metric)
    }
    
    // Update time series if available
    if (metricSeries.value[metric.name]) {
      metricSeries.value[metric.name].data.push({
        timestamp: metric.timestamp,
        value: metric.value,
      })
      
      // Keep only last 100 points
      if (metricSeries.value[metric.name].data.length > 100) {
        metricSeries.value[metric.name].data = metricSeries.value[metric.name].data.slice(-100)
      }
    }
    
    lastUpdated.value = new Date()
  }
  
  const refreshAll = async () => {
    await Promise.all([
      fetchMetrics(),
      fetchHealthStatus(),
      fetchSystemStatus(),
    ])
  }
  
  const getMetric = (name: string, labels?: Record<string, string>) => {
    return metrics.value.find(m => 
      m.name === name && 
      JSON.stringify(m.labels || {}) === JSON.stringify(labels || {})
    )
  }
  
  const getMetricValue = (name: string, labels?: Record<string, string>, defaultValue = 0) => {
    const metric = getMetric(name, labels)
    return metric ? metric.value : defaultValue
  }
  
  return {
    // State
    metrics,
    healthStatus,
    metricSeries,
    loading,
    error,
    lastUpdated,
    
    // Computed
    overallHealth,
    componentHealth,
    healthyComponents,
    totalComponents,
    healthPercentage,
    latestMetrics,
    performanceMetrics,
    systemMetrics,
    eventMetrics,
    
    // Actions
    fetchMetrics,
    fetchHealthStatus,
    fetchSystemStatus,
    fetchMetricSeries,
    updateRealtimeMetric,
    refreshAll,
    getMetric,
    getMetricValue,
  }
})

// Helper functions
function parsePrometheusMetrics(text: string): SystemMetric[] {
  const metrics: SystemMetric[] = []
  const lines = text.split('\n')
  
  for (const line of lines) {
    if (line.startsWith('#') || !line.trim()) continue
    
    const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*?)(\{[^}]*\})?\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)/)
    if (match) {
      const [, name, labelsStr, valueStr] = match
      
      let labels: Record<string, string> = {}
      if (labelsStr) {
        const labelPairs = labelsStr.slice(1, -1).split(',')
        for (const pair of labelPairs) {
          const [key, value] = pair.split('=')
          if (key && value) {
            labels[key.trim()] = value.trim().replace(/"/g, '')
          }
        }
      }
      
      metrics.push({
        name,
        value: parseFloat(valueStr),
        unit: getMetricUnit(name),
        timestamp: new Date().toISOString(),
        labels: Object.keys(labels).length > 0 ? labels : undefined,
      })
    }
  }
  
  return metrics
}

function getMetricUnit(metricName: string): string {
  if (metricName.includes('seconds') || metricName.includes('duration')) {
    return 's'
  }
  if (metricName.includes('bytes') || metricName.includes('memory')) {
    return 'bytes'
  }
  if (metricName.includes('rate') || metricName.includes('per_second')) {
    return '/s'
  }
  if (metricName.includes('percentage') || metricName.includes('ratio')) {
    return '%'
  }
  if (metricName.includes('total') || metricName.includes('count')) {
    return 'count'
  }
  return ''
}