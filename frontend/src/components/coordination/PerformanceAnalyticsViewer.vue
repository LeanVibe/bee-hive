<template>
  <div class="performance-analytics-viewer">
    <!-- Header with Controls -->
    <div class="analytics-header mb-6">
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 class="text-2xl font-bold text-slate-900 dark:text-white">
            Performance Analytics
          </h2>
          <p class="text-slate-600 dark:text-slate-400 mt-1">
            Real-time insights into agent performance and system efficiency
          </p>
        </div>
        <div class="flex items-center space-x-3 mt-4 sm:mt-0">
          <!-- Time Range Selector -->
          <select
            v-model="selectedTimeRange"
            @change="refreshAnalytics"
            class="input-field text-sm"
          >
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          
          <!-- Refresh Button -->
          <button
            @click="refreshAnalytics"
            :disabled="loading"
            class="btn-secondary flex items-center"
          >
            <ArrowPathIcon 
              class="w-4 h-4 mr-2" 
              :class="{ 'animate-spin': loading }"
            />
            Refresh
          </button>
          
          <!-- Auto-refresh Toggle -->
          <button
            @click="toggleAutoRefresh"
            :class="autoRefresh ? 'btn-primary' : 'btn-secondary'"
            class="flex items-center text-sm"
          >
            <BoltIcon class="w-4 h-4 mr-1" />
            Live
            <div v-if="autoRefresh" class="ml-2 w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          </button>
        </div>
      </div>
    </div>

    <!-- Key Metrics Overview -->
    <div class="metrics-overview grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <MetricCard
        title="System Efficiency"
        :value="Math.round(metrics.systemEfficiency * 100)"
        unit="%"
        :trend="efficiencyTrend"
        :icon="ChartBarIcon"
        :color="getEfficiencyColor(metrics.systemEfficiency)"
      />
      <MetricCard
        title="Agent Utilization"
        :value="Math.round(metrics.averageUtilization * 100)"
        unit="%"
        :trend="utilizationTrend"
        :icon="UsersIcon"
        :color="getUtilizationColor(metrics.averageUtilization)"
      />
      <MetricCard
        title="Task Throughput"
        :value="metrics.taskThroughput"
        unit="/hour"
        :trend="throughputTrend"
        :icon="RocketLaunchIcon"
        color="primary"
      />
      <MetricCard
        title="Error Rate"
        :value="(metrics.errorRate * 100).toFixed(1)"
        unit="%"
        :trend="errorTrend"
        :icon="ExclamationTriangleIcon"
        :color="getErrorColor(metrics.errorRate)"
      />
    </div>

    <!-- Charts Grid -->
    <div class="charts-grid grid grid-cols-1 xl:grid-cols-2 gap-6 mb-8">
      
      <!-- Task Completion Timeline -->
      <div class="chart-container glass-card rounded-xl p-6">
        <div class="chart-header flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
            Task Completion Timeline
          </h3>
          <div class="flex items-center space-x-2">
            <select
              v-model="completionChartType"
              class="input-field text-sm"
            >
              <option value="line">Line Chart</option>
              <option value="bar">Bar Chart</option>
              <option value="area">Area Chart</option>
            </select>
          </div>
        </div>
        <div class="chart-wrapper h-64">
          <canvas ref="completionChart"></canvas>
        </div>
      </div>

      <!-- Agent Performance Comparison -->
      <div class="chart-container glass-card rounded-xl p-6">
        <div class="chart-header flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
            Agent Performance Comparison
          </h3>
          <div class="flex items-center space-x-2">
            <select
              v-model="performanceMetric"
              class="input-field text-sm"
            >
              <option value="efficiency">Efficiency</option>
              <option value="speed">Response Time</option>
              <option value="quality">Quality Score</option>
              <option value="workload">Workload</option>
            </select>
          </div>
        </div>
        <div class="chart-wrapper h-64">
          <canvas ref="performanceChart"></canvas>
        </div>
      </div>

      <!-- Resource Utilization Heatmap -->
      <div class="chart-container glass-card rounded-xl p-6">
        <div class="chart-header flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
            Resource Utilization Heatmap
          </h3>
          <div class="flex items-center space-x-2">
            <button
              @click="toggleHeatmapView"
              class="btn-secondary text-sm"
            >
              {{ heatmapView === 'hourly' ? 'Daily View' : 'Hourly View' }}
            </button>
          </div>
        </div>
        <div class="chart-wrapper h-64">
          <canvas ref="heatmapChart"></canvas>
        </div>
      </div>

      <!-- Bottleneck Analysis -->
      <div class="chart-container glass-card rounded-xl p-6">
        <div class="chart-header flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
            System Bottlenecks
          </h3>
          <div class="flex items-center space-x-2">
            <span 
              class="px-2 py-1 text-xs rounded-full"
              :class="bottlenecks.length > 0 
                ? 'bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-400' 
                : 'bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400'"
            >
              {{ bottlenecks.length }} detected
            </span>
          </div>
        </div>
        
        <!-- Bottleneck List -->
        <div class="bottleneck-list space-y-3 max-h-52 overflow-y-auto">
          <div
            v-for="bottleneck in bottlenecks"
            :key="bottleneck.id"
            class="bottleneck-item p-3 bg-slate-50 dark:bg-slate-800 rounded-lg"
          >
            <div class="flex items-start justify-between mb-2">
              <div>
                <h4 class="font-medium text-sm text-slate-900 dark:text-white">
                  {{ bottleneck.title }}
                </h4>
                <p class="text-xs text-slate-600 dark:text-slate-400 mt-1">
                  {{ bottleneck.description }}
                </p>
              </div>
              <span 
                class="px-2 py-1 text-xs rounded-full"
                :class="getSeverityClass(bottleneck.severity)"
              >
                {{ bottleneck.severity }}
              </span>
            </div>
            
            <!-- Impact Visualization -->
            <div class="impact-bar w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2 mb-2">
              <div 
                class="impact-fill h-full rounded-full transition-all duration-500"
                :class="getImpactColor(bottleneck.impact)"
                :style="{ width: `${bottleneck.impact * 100}%` }"
              ></div>
            </div>
            
            <div class="flex items-center justify-between text-xs">
              <span class="text-slate-500 dark:text-slate-400">
                Impact: {{ Math.round(bottleneck.impact * 100) }}%
              </span>
              <button
                @click="resolveBottleneck(bottleneck)"
                class="text-primary-600 dark:text-primary-400 hover:underline"
              >
                Investigate →
              </button>
            </div>
          </div>
          
          <!-- No Bottlenecks -->
          <div 
            v-if="!bottlenecks.length"
            class="text-center py-8 text-slate-500 dark:text-slate-400"
          >
            <CheckCircleIcon class="w-12 h-12 mx-auto mb-3 text-green-500" />
            <p>No bottlenecks detected</p>
            <p class="text-sm mt-1">System is running optimally</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Detailed Analytics Table -->
    <div class="detailed-analytics glass-card rounded-xl p-6">
      <div class="table-header flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          Agent Performance Details
        </h3>
        <div class="flex items-center space-x-3">
          <!-- Search/Filter -->
          <div class="relative">
            <MagnifyingGlassIcon class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              v-model="searchQuery"
              placeholder="Search agents..."
              class="input-field pl-10 text-sm w-64"
            />
          </div>
          
          <!-- Sort Options -->
          <select
            v-model="sortBy"
            class="input-field text-sm"
          >
            <option value="performance">Performance</option>
            <option value="efficiency">Efficiency</option>
            <option value="workload">Workload</option>
            <option value="tasks_completed">Tasks Completed</option>
            <option value="response_time">Response Time</option>
          </select>
          
          <!-- Export Button -->
          <button
            @click="exportAnalytics"
            class="btn-secondary text-sm flex items-center"
          >
            <ArrowDownTrayIcon class="w-4 h-4 mr-2" />
            Export
          </button>
        </div>
      </div>

      <!-- Performance Table -->
      <div class="performance-table overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="bg-slate-50 dark:bg-slate-800">
            <tr>
              <th class="table-header-cell text-left">Agent</th>
              <th class="table-header-cell text-center">Status</th>
              <th class="table-header-cell text-center">Performance</th>
              <th class="table-header-cell text-center">Efficiency</th>
              <th class="table-header-cell text-center">Workload</th>
              <th class="table-header-cell text-center">Tasks</th>
              <th class="table-header-cell text-center">Avg Response</th>
              <th class="table-header-cell text-center">Last Active</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="agent in filteredAgentMetrics"
              :key="agent.id"
              class="table-row border-b border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800"
            >
              <td class="table-cell">
                <div class="flex items-center space-x-3">
                  <div 
                    class="w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-medium"
                    :style="{ backgroundColor: getAgentColor(agent.id) }"
                  >
                    {{ getAgentInitials(agent.name) }}
                  </div>
                  <div>
                    <div class="font-medium text-slate-900 dark:text-white">
                      {{ agent.name }}
                    </div>
                    <div class="text-xs text-slate-500 dark:text-slate-400">
                      {{ agent.type }}
                    </div>
                  </div>
                </div>
              </td>
              <td class="table-cell text-center">
                <span 
                  class="inline-flex items-center px-2 py-1 text-xs rounded-full"
                  :class="getStatusClass(agent.status)"
                >
                  <div 
                    class="w-2 h-2 rounded-full mr-1"
                    :class="getStatusDotClass(agent.status)"
                  ></div>
                  {{ agent.status }}
                </span>
              </td>
              <td class="table-cell text-center">
                <div class="flex items-center justify-center space-x-2">
                  <div class="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                    <div 
                      class="h-full rounded-full transition-all duration-500"
                      :class="getPerformanceColor(agent.performance)"
                      :style="{ width: `${agent.performance * 100}%` }"
                    ></div>
                  </div>
                  <span class="text-xs font-medium w-8">
                    {{ Math.round(agent.performance * 100) }}%
                  </span>
                </div>
              </td>
              <td class="table-cell text-center">
                <span 
                  class="font-medium"
                  :class="getEfficiencyTextColor(agent.efficiency)"
                >
                  {{ Math.round(agent.efficiency * 100) }}%
                </span>
              </td>
              <td class="table-cell text-center">
                <div class="flex items-center justify-center space-x-2">
                  <div class="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                    <div 
                      class="h-full rounded-full transition-all duration-500"
                      :class="getWorkloadColor(agent.workload)"
                      :style="{ width: `${agent.workload * 100}%` }"
                    ></div>
                  </div>
                  <span class="text-xs font-medium w-8">
                    {{ Math.round(agent.workload * 100) }}%
                  </span>
                </div>
              </td>
              <td class="table-cell text-center font-medium">
                {{ agent.tasksCompleted }}
              </td>
              <td class="table-cell text-center">
                <span 
                  class="font-mono"
                  :class="getResponseTimeColor(agent.avgResponseTime)"
                >
                  {{ agent.avgResponseTime }}ms
                </span>
              </td>
              <td class="table-cell text-center text-xs text-slate-500 dark:text-slate-400">
                {{ formatTime(agent.lastActive) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { Chart, registerables } from 'chart.js'
import { formatDistanceToNow } from 'date-fns'

// Register Chart.js components
Chart.register(...registerables)

// Icons
import {
  ArrowPathIcon,
  BoltIcon,
  ChartBarIcon,
  UsersIcon,
  RocketLaunchIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  MagnifyingGlassIcon,
  ArrowDownTrayIcon
} from '@heroicons/vue/24/outline'

// Components
import MetricCard from '@/components/dashboard/MetricCard.vue'

// Services and composables
import { useCoordinationService } from '@/services/coordinationService'
import { useSessionColors } from '@/utils/SessionColorManager'
import { api } from '@/services/api'

// Types
interface PerformanceMetrics {
  systemEfficiency: number
  averageUtilization: number
  taskThroughput: number
  errorRate: number
}

interface AgentMetrics {
  id: string
  name: string
  type: string
  status: string
  performance: number
  efficiency: number
  workload: number
  tasksCompleted: number
  avgResponseTime: number
  lastActive: string
}

interface Bottleneck {
  id: string
  title: string
  description: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  impact: number
  category: string
}

const coordinationService = useCoordinationService()
const { getAgentColor } = useSessionColors()

// State
const loading = ref(false)
const autoRefresh = ref(false)
const selectedTimeRange = ref('24h')
const completionChartType = ref('line')
const performanceMetric = ref('efficiency')
const heatmapView = ref('hourly')
const searchQuery = ref('')
const sortBy = ref('performance')

// Data
const metrics = ref<PerformanceMetrics>({
  systemEfficiency: 0.85,
  averageUtilization: 0.72,
  taskThroughput: 45,
  errorRate: 0.03
})

const agentMetrics = ref<AgentMetrics[]>([])
const bottlenecks = ref<Bottleneck[]>([])
const chartData = ref({
  completion: [],
  performance: [],
  utilization: []
})

// Chart refs
const completionChart = ref<HTMLCanvasElement>()
const performanceChart = ref<HTMLCanvasElement>()
const heatmapChart = ref<HTMLCanvasElement>()

// Chart instances
let completionChartInstance: Chart | null = null
let performanceChartInstance: Chart | null = null
let heatmapChartInstance: Chart | null = null

// Auto-refresh interval
let refreshInterval: number | null = null

// Computed
const efficiencyTrend = computed(() => {
  // Mock trend calculation
  return Math.random() > 0.5 ? 'up' : 'down'
})

const utilizationTrend = computed(() => {
  return Math.random() > 0.5 ? 'up' : 'down'
})

const throughputTrend = computed(() => {
  return Math.random() > 0.5 ? 'up' : 'down'
})

const errorTrend = computed(() => {
  return Math.random() > 0.5 ? 'down' : 'up'
})

const filteredAgentMetrics = computed(() => {
  let filtered = agentMetrics.value

  // Apply search filter
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(agent => 
      agent.name.toLowerCase().includes(query) ||
      agent.type.toLowerCase().includes(query)
    )
  }

  // Apply sorting
  filtered.sort((a, b) => {
    switch (sortBy.value) {
      case 'performance':
        return b.performance - a.performance
      case 'efficiency':
        return b.efficiency - a.efficiency
      case 'workload':
        return b.workload - a.workload
      case 'tasks_completed':
        return b.tasksCompleted - a.tasksCompleted
      case 'response_time':
        return a.avgResponseTime - b.avgResponseTime
      default:
        return 0
    }
  })

  return filtered
})

// Methods
const refreshAnalytics = async () => {
  loading.value = true
  
  try {
    await Promise.all([
      loadPerformanceMetrics(),
      loadAgentMetrics(),
      loadBottlenecks(),
      loadChartData()
    ])
    
    await nextTick()
    updateCharts()
  } catch (error) {
    console.error('Failed to refresh analytics:', error)
  } finally {
    loading.value = false
  }
}

const loadPerformanceMetrics = async () => {
  try {
    const params = new URLSearchParams({ 
      time_range_hours: getTimeRangeHours(selectedTimeRange.value).toString()
    })
    const response = await api.get('/team-coordination/metrics', params)
    
    metrics.value = {
      systemEfficiency: response.data.system_efficiency_score || 0.85,
      averageUtilization: response.data.agent_utilization_percentage / 100 || 0.72,
      taskThroughput: calculateTaskThroughput(response.data),
      errorRate: calculateErrorRate(response.data)
    }
  } catch (error) {
    console.error('Failed to load performance metrics:', error)
  }
}

const loadAgentMetrics = async () => {
  try {
    const response = await api.get('/team-coordination/agents')
    
    agentMetrics.value = response.data.map((agent: any) => ({
      id: agent.agent_id,
      name: agent.name,
      type: agent.type,
      status: agent.status,
      performance: agent.performance_score || Math.random(),
      efficiency: calculateEfficiency(agent),
      workload: agent.current_workload || Math.random(),
      tasksCompleted: agent.completed_today || Math.floor(Math.random() * 20),
      avgResponseTime: Math.floor(agent.average_response_time_ms || Math.random() * 1000),
      lastActive: agent.last_heartbeat || new Date().toISOString()
    }))
  } catch (error) {
    console.error('Failed to load agent metrics:', error)
  }
}

const loadBottlenecks = async () => {
  try {
    // Mock bottleneck detection - in real implementation, this would call an API
    const mockBottlenecks = [
      {
        id: 'bt-1',
        title: 'High Memory Usage',
        description: 'Agent-001 using 95% of available memory',
        severity: 'high' as const,
        impact: 0.8,
        category: 'resource'
      },
      {
        id: 'bt-2', 
        title: 'Slow Response Times',
        description: 'Average response time increased by 40%',
        severity: 'medium' as const,
        impact: 0.6,
        category: 'performance'
      }
    ]
    
    bottlenecks.value = Math.random() > 0.3 ? mockBottlenecks : []
  } catch (error) {
    console.error('Failed to load bottlenecks:', error)
  }
}

const loadChartData = async () => {
  try {
    // Mock chart data - in real implementation, this would fetch time series data
    chartData.value = {
      completion: generateMockTimeSeriesData(24),
      performance: generateMockPerformanceData(),
      utilization: generateMockUtilizationData()
    }
  } catch (error) {
    console.error('Failed to load chart data:', error)
  }
}

const updateCharts = () => {
  updateCompletionChart()
  updatePerformanceChart()
  updateHeatmapChart()
}

const updateCompletionChart = () => {
  if (!completionChart.value) return

  const ctx = completionChart.value.getContext('2d')
  if (!ctx) return

  if (completionChartInstance) {
    completionChartInstance.destroy()
  }

  const data = chartData.value.completion
  
  completionChartInstance = new Chart(ctx, {
    type: completionChartType.value as any,
    data: {
      labels: data.map((_, i) => `${i}:00`),
      datasets: [{
        label: 'Tasks Completed',
        data: data,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: completionChartType.value === 'area' 
          ? 'rgba(59, 130, 246, 0.1)' 
          : 'rgba(59, 130, 246, 0.8)',
        tension: 0.4,
        fill: completionChartType.value === 'area'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: 'rgba(148, 163, 184, 0.1)'
          }
        },
        x: {
          grid: {
            display: false
          }
        }
      }
    }
  })
}

const updatePerformanceChart = () => {
  if (!performanceChart.value) return

  const ctx = performanceChart.value.getContext('2d')
  if (!ctx) return

  if (performanceChartInstance) {
    performanceChartInstance.destroy()
  }

  const data = agentMetrics.value.slice(0, 10) // Top 10 agents
  const metricKey = performanceMetric.value
  
  performanceChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(agent => agent.name),
      datasets: [{
        label: getMetricLabel(metricKey),
        data: data.map(agent => getMetricValue(agent, metricKey)),
        backgroundColor: data.map(agent => getAgentColor(agent.id) + '80'),
        borderColor: data.map(agent => getAgentColor(agent.id)),
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: metricKey === 'response_time' ? undefined : 1,
          grid: {
            color: 'rgba(148, 163, 184, 0.1)'
          }
        },
        x: {
          grid: {
            display: false
          }
        }
      }
    }
  })
}

const updateHeatmapChart = () => {
  if (!heatmapChart.value) return

  const ctx = heatmapChart.value.getContext('2d')
  if (!ctx) return

  if (heatmapChartInstance) {
    heatmapChartInstance.destroy()
  }

  // Create a mock heatmap using a line chart for simplicity
  // In a real implementation, you'd use a proper heatmap library
  const data = chartData.value.utilization
  
  heatmapChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map((_, i) => `${i}:00`),
      datasets: [{
        label: 'System Utilization',
        data: data,
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          grid: {
            color: 'rgba(148, 163, 184, 0.1)'
          }
        },
        x: {
          grid: {
            display: false
          }
        }
      }
    }
  })
}

const toggleAutoRefresh = () => {
  autoRefresh.value = !autoRefresh.value
  
  if (autoRefresh.value) {
    refreshInterval = setInterval(refreshAnalytics, 30000) // Refresh every 30 seconds
  } else {
    if (refreshInterval) {
      clearInterval(refreshInterval)
      refreshInterval = null
    }
  }
}

const toggleHeatmapView = () => {
  heatmapView.value = heatmapView.value === 'hourly' ? 'daily' : 'hourly'
  updateHeatmapChart()
}

const resolveBottleneck = (bottleneck: Bottleneck) => {
  // Navigate to detailed bottleneck analysis
  console.log('Investigating bottleneck:', bottleneck)
}

const exportAnalytics = () => {
  // Export analytics data as CSV or JSON
  const data = {
    metrics: metrics.value,
    agentMetrics: agentMetrics.value,
    bottlenecks: bottlenecks.value,
    timestamp: new Date().toISOString()
  }
  
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `performance-analytics-${Date.now()}.json`
  link.click()
  URL.revokeObjectURL(url)
}

// Utility functions
const getTimeRangeHours = (range: string): number => {
  const ranges = {
    '1h': 1,
    '6h': 6,
    '24h': 24,
    '7d': 168,
    '30d': 720
  }
  return ranges[range] || 24
}

const calculateTaskThroughput = (data: any): number => {
  return data.tasks_completed_today || Math.floor(Math.random() * 100)
}

const calculateErrorRate = (data: any): number => {
  return (data.error_count || 0) / Math.max(data.total_tasks || 100, 1)
}

const calculateEfficiency = (agent: any): number => {
  return agent.efficiency || Math.random()
}

const generateMockTimeSeriesData = (points: number): number[] => {
  return Array.from({ length: points }, () => Math.floor(Math.random() * 20))
}

const generateMockPerformanceData = (): number[] => {
  return Array.from({ length: 10 }, () => Math.random())
}

const generateMockUtilizationData = (): number[] => {
  return Array.from({ length: 24 }, () => Math.random() * 0.8 + 0.2)
}

const getMetricLabel = (metric: string): string => {
  const labels = {
    efficiency: 'Efficiency',
    speed: 'Response Time (ms)',
    quality: 'Quality Score',
    workload: 'Workload'
  }
  return labels[metric] || metric
}

const getMetricValue = (agent: AgentMetrics, metric: string): number => {
  switch (metric) {
    case 'efficiency':
      return agent.efficiency
    case 'speed':
      return agent.avgResponseTime
    case 'quality':
      return agent.performance
    case 'workload':
      return agent.workload
    default:
      return agent.performance
  }
}

const getAgentInitials = (name: string): string => {
  return name
    .split(' ')
    .map(n => n.charAt(0))
    .join('')
    .toUpperCase()
    .slice(0, 2)
}

const formatTime = (dateString: string): string => {
  return formatDistanceToNow(new Date(dateString), { addSuffix: true })
}

// Color utility functions
const getEfficiencyColor = (efficiency: number): string => {
  if (efficiency >= 0.8) return 'success'
  if (efficiency >= 0.6) return 'warning'
  return 'error'
}

const getUtilizationColor = (utilization: number): string => {
  if (utilization >= 0.9) return 'error'
  if (utilization >= 0.7) return 'warning'
  if (utilization >= 0.3) return 'success'
  return 'info'
}

const getErrorColor = (errorRate: number): string => {
  if (errorRate >= 0.1) return 'error'
  if (errorRate >= 0.05) return 'warning'
  return 'success'
}

const getSeverityClass = (severity: string): string => {
  const classes = {
    low: 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400',
    medium: 'bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400',
    high: 'bg-orange-100 dark:bg-orange-900 text-orange-600 dark:text-orange-400',
    critical: 'bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-400'
  }
  return classes[severity] || classes.low
}

const getImpactColor = (impact: number): string => {
  if (impact >= 0.8) return 'bg-red-500'
  if (impact >= 0.6) return 'bg-orange-500'
  if (impact >= 0.4) return 'bg-yellow-500'
  return 'bg-green-500'
}

const getStatusClass = (status: string): string => {
  const classes = {
    active: 'bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400',
    idle: 'bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400',
    busy: 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400',
    sleeping: 'bg-gray-100 dark:bg-gray-900 text-gray-600 dark:text-gray-400',
    error: 'bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-400'
  }
  return classes[status] || classes.active
}

const getStatusDotClass = (status: string): string => {
  const classes = {
    active: 'bg-green-500',
    idle: 'bg-yellow-500',
    busy: 'bg-blue-500',
    sleeping: 'bg-gray-500',
    error: 'bg-red-500'
  }
  return classes[status] || classes.active
}

const getPerformanceColor = (performance: number): string => {
  if (performance >= 0.8) return 'bg-green-500'
  if (performance >= 0.6) return 'bg-yellow-500'
  return 'bg-red-500'
}

const getEfficiencyTextColor = (efficiency: number): string => {
  if (efficiency >= 0.8) return 'text-green-600 dark:text-green-400'
  if (efficiency >= 0.6) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

const getWorkloadColor = (workload: number): string => {
  if (workload >= 0.9) return 'bg-red-500'
  if (workload >= 0.7) return 'bg-orange-500'
  if (workload >= 0.5) return 'bg-yellow-500'
  return 'bg-green-500'
}

const getResponseTimeColor = (responseTime: number): string => {
  if (responseTime >= 1000) return 'text-red-600 dark:text-red-400'
  if (responseTime >= 500) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-green-600 dark:text-green-400'
}

// Watchers
watch(() => selectedTimeRange.value, refreshAnalytics)
watch(() => completionChartType.value, updateCompletionChart)
watch(() => performanceMetric.value, updatePerformanceChart)

// Lifecycle
onMounted(async () => {
  await refreshAnalytics()
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
  
  if (completionChartInstance) {
    completionChartInstance.destroy()
  }
  if (performanceChartInstance) {
    performanceChartInstance.destroy()
  }
  if (heatmapChartInstance) {
    heatmapChartInstance.destroy()
  }
})
</script>

<style scoped>
.performance-analytics-viewer {
  @apply min-h-screen bg-slate-50 dark:bg-slate-900;
}

.glass-card {
  @apply bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50;
}

.input-field {
  @apply bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent;
}

.btn-primary {
  @apply bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

.btn-secondary {
  @apply bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 px-4 py-2 rounded-md font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

.chart-wrapper {
  @apply relative;
}

.chart-wrapper canvas {
  @apply max-h-full;
}

.bottleneck-list::-webkit-scrollbar {
  @apply w-2;
}

.bottleneck-list::-webkit-scrollbar-track {
  @apply bg-slate-100 dark:bg-slate-800 rounded;
}

.bottleneck-list::-webkit-scrollbar-thumb {
  @apply bg-slate-300 dark:bg-slate-600 rounded hover:bg-slate-400 dark:hover:bg-slate-500;
}

.performance-table::-webkit-scrollbar {
  @apply h-2;
}

.performance-table::-webkit-scrollbar-track {
  @apply bg-slate-100 dark:bg-slate-800 rounded;
}

.performance-table::-webkit-scrollbar-thumb {
  @apply bg-slate-300 dark:bg-slate-600 rounded hover:bg-slate-400 dark:hover:bg-slate-500;
}

.table-header-cell {
  @apply px-6 py-3 text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider;
}

.table-cell {
  @apply px-6 py-4;
}

.table-row {
  @apply transition-colors duration-150;
}

.impact-bar {
  @apply relative overflow-hidden;
}

.impact-fill {
  @apply relative;
}

.impact-fill::after {
  content: '';
  @apply absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent;
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.bottleneck-item {
  animation: slideIn 0.3s ease-out;
}
</style>