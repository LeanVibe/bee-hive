<template>
  <div class="bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
    <div class="p-6 border-b border-slate-200 dark:border-slate-700">
      <div class="flex items-center justify-between">
        <div>
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
            Business Intelligence
          </h3>
          <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Real-time business performance insights
          </p>
        </div>
        
        <div class="flex items-center space-x-2">
          <button
            @click="refreshData"
            :disabled="loading"
            class="inline-flex items-center px-3 py-2 border border-slate-300 dark:border-slate-600 shadow-sm text-sm leading-4 font-medium rounded-md text-slate-700 dark:text-slate-300 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
          >
            <ArrowPathIcon 
              class="w-4 h-4 mr-2" 
              :class="{ 'animate-spin': loading }"
            />
            Refresh
          </button>
          
          <select
            v-model="selectedTimeframe"
            @change="onTimeframeChange"
            class="rounded-md border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white text-sm focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last Week</option>
            <option value="30d">Last Month</option>
          </select>
        </div>
      </div>
    </div>

    <div class="p-6">
      <!-- Loading State -->
      <div v-if="loading" class="flex justify-center py-8">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
      </div>

      <!-- Error State -->
      <div v-else-if="error" class="text-center py-8">
        <div class="text-red-500 dark:text-red-400 mb-2">
          <ExclamationTriangleIcon class="w-8 h-8 mx-auto mb-2" />
          Failed to load business intelligence data
        </div>
        <p class="text-sm text-slate-500 dark:text-slate-400">{{ error }}</p>
      </div>

      <!-- Business KPIs Grid -->
      <div v-else class="space-y-6">
        <!-- Key Performance Indicators -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div 
            v-for="kpi in businessKPIs" 
            :key="kpi.key"
            class="bg-slate-50 dark:bg-slate-700 rounded-lg p-4"
          >
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm font-medium text-slate-600 dark:text-slate-300">
                  {{ kpi.label }}
                </p>
                <p class="text-2xl font-bold text-slate-900 dark:text-white">
                  {{ formatValue(kpi.value, kpi.format) }}
                </p>
              </div>
              <div 
                class="p-3 rounded-full"
                :class="getKPIIconClass(kpi.trend)"
              >
                <component :is="kpi.icon" class="w-5 h-5" />
              </div>
            </div>
            
            <div class="mt-2 flex items-center">
              <component 
                :is="getTrendIcon(kpi.trend)" 
                class="w-4 h-4 mr-1"
                :class="getTrendColor(kpi.trend)"
              />
              <span 
                class="text-sm font-medium"
                :class="getTrendColor(kpi.trend)"
              >
                {{ formatTrendChange(kpi.change) }}
              </span>
              <span class="text-sm text-slate-500 dark:text-slate-400 ml-1">
                vs {{ kpi.comparison }}
              </span>
            </div>
          </div>
        </div>

        <!-- Performance Trends Chart -->
        <div class="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
          <h4 class="text-md font-medium text-slate-900 dark:text-white mb-4">
            Performance Trends
          </h4>
          <PerformanceTrendsChart
            :data="performanceTrends"
            :timeframe="selectedTimeframe"
            :height="300"
          />
        </div>

        <!-- Business Insights -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Top Insights -->
          <div class="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
            <h4 class="text-md font-medium text-slate-900 dark:text-white mb-4">
              Key Insights
            </h4>
            <div class="space-y-3">
              <div 
                v-for="insight in topInsights" 
                :key="insight.id"
                class="flex items-start space-x-3"
              >
                <div 
                  class="flex-shrink-0 w-2 h-2 rounded-full mt-2"
                  :class="getInsightPriorityColor(insight.priority)"
                ></div>
                <div class="flex-1">
                  <p class="text-sm font-medium text-slate-900 dark:text-white">
                    {{ insight.title }}
                  </p>
                  <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    {{ insight.description }}
                  </p>
                  <p class="text-xs text-slate-400 dark:text-slate-500 mt-1">
                    Impact: {{ insight.impact }}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <!-- Quick Actions -->
          <div class="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
            <h4 class="text-md font-medium text-slate-900 dark:text-white mb-4">
              Recommended Actions
            </h4>
            <div class="space-y-2">
              <button
                v-for="action in recommendedActions"
                :key="action.id"
                @click="executeAction(action)"
                class="w-full text-left px-3 py-2 rounded-md text-sm bg-white dark:bg-slate-600 hover:bg-slate-100 dark:hover:bg-slate-500 transition-colors duration-150"
              >
                <div class="flex items-center justify-between">
                  <span class="font-medium text-slate-900 dark:text-white">
                    {{ action.title }}
                  </span>
                  <component :is="action.icon" class="w-4 h-4 text-slate-400" />
                </div>
                <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  {{ action.description }}
                </p>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import {
  ArrowPathIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  MinusIcon,
  CpuChipIcon,
  ChartBarIcon,
  ClockIcon,
  CurrencyDollarIcon,
  PlayIcon,
  AdjustmentsHorizontalIcon
} from '@heroicons/vue/24/outline'
import PerformanceTrendsChart from '../charts/PerformanceTrendsChart.vue'
import { useBusinessAnalytics } from '@/composables/useBusinessAnalytics'

// Props
interface Props {
  refreshInterval?: number
}

const props = withDefaults(defineProps<Props>(), {
  refreshInterval: 30000 // 30 seconds
})

// State
const loading = ref(false)
const error = ref<string | null>(null)
const selectedTimeframe = ref('24h')

// Composable
const { 
  businessKPIs, 
  performanceTrends, 
  topInsights, 
  recommendedActions,
  fetchBusinessIntelligence 
} = useBusinessAnalytics()

// Computed
const formatValue = (value: number, format: string): string => {
  switch (format) {
    case 'percentage':
      return `${(value * 100).toFixed(1)}%`
    case 'currency':
      return `$${value.toLocaleString()}`
    case 'number':
      return value.toLocaleString()
    case 'duration':
      return `${value.toFixed(1)}s`
    default:
      return value.toString()
  }
}

const formatTrendChange = (change: number): string => {
  const sign = change >= 0 ? '+' : ''
  return `${sign}${(change * 100).toFixed(1)}%`
}

const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
  switch (trend) {
    case 'up':
      return ArrowTrendingUpIcon
    case 'down':
      return ArrowTrendingDownIcon
    default:
      return MinusIcon
  }
}

const getTrendColor = (trend: 'up' | 'down' | 'stable'): string => {
  switch (trend) {
    case 'up':
      return 'text-green-600 dark:text-green-400'
    case 'down':
      return 'text-red-600 dark:text-red-400'
    default:
      return 'text-slate-500 dark:text-slate-400'
  }
}

const getKPIIconClass = (trend: 'up' | 'down' | 'stable'): string => {
  switch (trend) {
    case 'up':
      return 'bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400'
    case 'down':
      return 'bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400'
    default:
      return 'bg-slate-100 dark:bg-slate-600 text-slate-500 dark:text-slate-400'
  }
}

const getInsightPriorityColor = (priority: 'high' | 'medium' | 'low'): string => {
  switch (priority) {
    case 'high':
      return 'bg-red-400'
    case 'medium':
      return 'bg-yellow-400'
    default:
      return 'bg-green-400'
  }
}

// Methods
const refreshData = async () => {
  if (loading.value) return
  
  loading.value = true
  error.value = null
  
  try {
    await fetchBusinessIntelligence({
      timeframe: selectedTimeframe.value
    })
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Unknown error occurred'
  } finally {
    loading.value = false
  }
}

const onTimeframeChange = () => {
  refreshData()
}

const executeAction = (action: any) => {
  // Emit action to parent or handle directly
  console.log('Executing action:', action)
  // Implementation depends on specific action type
}

// Lifecycle
onMounted(() => {
  refreshData()
  
  // Set up auto-refresh
  if (props.refreshInterval > 0) {
    setInterval(refreshData, props.refreshInterval)
  }
})

// Watch for external changes
watch(() => props.refreshInterval, (newInterval) => {
  if (newInterval > 0) {
    setInterval(refreshData, newInterval)
  }
})
</script>

<style scoped>
/* Custom animations for smooth transitions */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>