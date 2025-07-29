<template>
  <div class="intelligence-kpi-dashboard">
    <!-- Header -->
    <div class="kpi-header flex items-center justify-between mb-6">
      <div>
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          Intelligence KPI Dashboard
        </h3>
        <p class="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Real-time metrics for intelligence improvement and coordination amplification
        </p>
      </div>
      
      <div class="flex items-center space-x-3">
        <!-- Time Range Selector -->
        <select
          v-model="timeRangeHours"
          @change="refreshData"
          class="text-xs bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded px-3 py-1"
        >
          <option :value="1">1 Hour</option>
          <option :value="6">6 Hours</option>
          <option :value="24">24 Hours</option>
          <option :value="168">1 Week</option>
        </select>
        
        <!-- Aggregation Interval -->
        <select
          v-model="aggregationInterval"
          @change="refreshData"
          class="text-xs bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded px-3 py-1"
        >
          <option value="1m">1min</option>
          <option value="5m">5min</option>
          <option value="15m">15min</option>
          <option value="1h">1hour</option>
        </select>
        
        <!-- Forecast Toggle -->
        <button
          @click="toggleForecasts"
          :class="showForecasts 
            ? 'bg-blue-600 text-white' 
            : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300'"
          class="text-xs px-3 py-1 rounded transition-colors"
        >
          <ChartBarIcon class="w-3 h-3 mr-1 inline" />
          Forecast
        </button>
        
        <!-- Refresh Button -->
        <button
          @click="refreshData"
          :disabled="loading"
          class="text-xs px-3 py-1 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
        >
          <ArrowPathIcon class="w-3 h-3 mr-1 inline" :class="{ 'animate-spin': loading }" />
          Refresh
        </button>
      </div>
    </div>

    <!-- KPI Grid -->
    <div class="kpi-grid grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
      <div
        v-for="kpi in kpiMetrics"
        :key="kpi.name"
        class="kpi-card bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6 hover:shadow-lg transition-shadow"
      >
        <!-- KPI Header -->
        <div class="flex items-center justify-between mb-4">
          <div>
            <h4 class="text-sm font-medium text-slate-900 dark:text-white">
              {{ formatKPIName(kpi.name) }}
            </h4>
            <p class="text-xs text-slate-600 dark:text-slate-400 mt-1">
              {{ kpi.description }}
            </p>
          </div>
          
          <div class="flex items-center space-x-2">
            <!-- Threshold Status Indicator -->
            <div 
              class="w-3 h-3 rounded-full"
              :class="getThresholdStatusClass(kpi.threshold_status)"
              :title="kpi.threshold_status"
            ></div>
            
            <!-- Trend Direction -->
            <div class="flex items-center text-xs" :class="getTrendColorClass(kpi.trend_direction)">
              <ArrowTrendingUpIcon 
                v-if="kpi.trend_direction === 'up'" 
                class="w-3 h-3 mr-1" 
              />
              <ArrowTrendingDownIcon 
                v-else-if="kpi.trend_direction === 'down'" 
                class="w-3 h-3 mr-1" 
              />
              <MinusIcon 
                v-else 
                class="w-3 h-3 mr-1" 
              />
              <span>{{ (kpi.trend_strength * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
        
        <!-- Current Value -->
        <div class="mb-4">
          <div class="flex items-baseline space-x-2">
            <span class="text-2xl font-bold text-slate-900 dark:text-white">
              {{ formatKPIValue(kpi.current_value, kpi.unit) }}
            </span>
            <span class="text-sm text-slate-500 dark:text-slate-400">
              {{ kpi.unit }}
            </span>
          </div>
        </div>
        
        <!-- Mini Chart -->
        <div class="kpi-chart-container h-24 mb-4">
          <svg
            :ref="`chart-${kpi.name}`"
            class="w-full h-full"
            @click="selectKPI(kpi)"
          >
            <!-- Rendered by D3 -->
          </svg>
        </div>
        
        <!-- KPI Actions -->
        <div class="flex items-center justify-between">
          <div class="text-xs text-slate-500 dark:text-slate-400">
            {{ kpi.historical_data.length }} data points
          </div>
          
          <div class="flex items-center space-x-2">
            <button
              @click="viewKPIDetails(kpi)"
              class="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
              title="View details"
            >
              <EyeIcon class="w-4 h-4" />
            </button>
            
            <button
              @click="exportKPIData(kpi)"
              class="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
              title="Export data"
            >
              <ArrowDownTrayIcon class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Detailed Chart View -->
    <div 
      v-if="selectedKPI"
      class="detailed-chart-section bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6 mb-6"
    >
      <div class="flex items-center justify-between mb-4">
        <div>
          <h4 class="text-lg font-medium text-slate-900 dark:text-white">
            {{ formatKPIName(selectedKPI.name) }} - Detailed View
          </h4>
          <p class="text-sm text-slate-600 dark:text-slate-400">
            {{ selectedKPI.description }}
          </p>
        </div>
        
        <button
          @click="selectedKPI = null"
          class="p-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
        >
          <XMarkIcon class="w-5 h-5" />
        </button>
      </div>
      
      <!-- Detailed Chart Container -->
      <div class="detailed-chart-container h-80 mb-4">
        <svg
          ref="detailedChart"
          class="w-full h-full"
        >
          <!-- Rendered by D3 -->
        </svg>
      </div>
      
      <!-- Chart Legend and Statistics -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="chart-legend">
          <h5 class="text-sm font-medium text-slate-900 dark:text-white mb-2">
            Legend
          </h5>
          <div class="space-y-1">
            <div class="flex items-center space-x-2 text-xs">
              <div class="w-3 h-0.5 bg-blue-500"></div>
              <span class="text-slate-600 dark:text-slate-400">Historical Data</span>
            </div>
            <div v-if="selectedKPI.forecast_data" class="flex items-center space-x-2 text-xs">
              <div class="w-3 h-0.5 bg-orange-500 opacity-60"></div>
              <span class="text-slate-600 dark:text-slate-400">Forecast</span>
            </div>
            <div class="flex items-center space-x-2 text-xs">
              <div class="w-3 h-0.5 bg-green-500"></div>
              <span class="text-slate-600 dark:text-slate-400">Normal Range</span>
            </div>
            <div class="flex items-center space-x-2 text-xs">
              <div class="w-3 h-0.5 bg-red-500"></div>
              <span class="text-slate-600 dark:text-slate-400">Critical Range</span>
            </div>
          </div>
        </div>
        
        <div class="chart-statistics">
          <h5 class="text-sm font-medium text-slate-900 dark:text-white mb-2">
            Statistics
          </h5>
          <div class="space-y-2 text-xs">
            <div class="flex justify-between">
              <span class="text-slate-600 dark:text-slate-400">Current:</span>
              <span class="font-medium text-slate-900 dark:text-white">
                {{ formatKPIValue(selectedKPI.current_value, selectedKPI.unit) }}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-600 dark:text-slate-400">Average:</span>
              <span class="font-medium text-slate-900 dark:text-white">
                {{ formatKPIValue(getKPIAverage(selectedKPI), selectedKPI.unit) }}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-600 dark:text-slate-400">Min:</span>
              <span class="font-medium text-slate-900 dark:text-white">
                {{ formatKPIValue(getKPIMin(selectedKPI), selectedKPI.unit) }}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-600 dark:text-slate-400">Max:</span>
              <span class="font-medium text-slate-900 dark:text-white">
                {{ formatKPIValue(getKPIMax(selectedKPI), selectedKPI.unit) }}
              </span>
            </div>
          </div>
        </div>
        
        <div class="chart-insights">
          <h5 class="text-sm font-medium text-slate-900 dark:text-white mb-2">
            Insights
          </h5>
          <div class="space-y-2">
            <div 
              v-for="insight in getKPIInsights(selectedKPI)"
              :key="insight.text"
              class="flex items-start space-x-2 text-xs"
            >
              <div 
                class="w-2 h-2 rounded-full mt-1 flex-shrink-0"
                :class="insight.severity === 'info' ? 'bg-blue-500' : insight.severity === 'warning' ? 'bg-yellow-500' : 'bg-red-500'"
              ></div>
              <span class="text-slate-600 dark:text-slate-400">{{ insight.text }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Summary Cards -->
    <div class="summary-cards grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="summary-card bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
        <div class="flex items-center space-x-3">
          <div class="p-2 bg-green-100 dark:bg-green-900/40 rounded-lg">
            <CheckCircleIcon class="w-5 h-5 text-green-600 dark:text-green-400" />
          </div>
          <div>
            <div class="text-sm font-medium text-green-900 dark:text-green-100">
              Normal KPIs
            </div>
            <div class="text-lg font-bold text-green-700 dark:text-green-300">
              {{ normalKPICount }}
            </div>
          </div>
        </div>
      </div>
      
      <div class="summary-card bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <div class="flex items-center space-x-3">
          <div class="p-2 bg-yellow-100 dark:bg-yellow-900/40 rounded-lg">
            <ExclamationTriangleIcon class="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
          </div>
          <div>
            <div class="text-sm font-medium text-yellow-900 dark:text-yellow-100">
              Warning KPIs
            </div>
            <div class="text-lg font-bold text-yellow-700 dark:text-yellow-300">
              {{ warningKPICount }}
            </div>
          </div>
        </div>
      </div>
      
      <div class="summary-card bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <div class="flex items-center space-x-3">
          <div class="p-2 bg-red-100 dark:bg-red-900/40 rounded-lg">
            <XCircleIcon class="w-5 h-5 text-red-600 dark:text-red-400" />
          </div>
          <div>
            <div class="text-sm font-medium text-red-900 dark:text-red-100">
              Critical KPIs
            </div>
            <div class="text-lg font-bold text-red-700 dark:text-red-300">
              {{ criticalKPICount }}
            </div>
          </div>
        </div>
      </div>
      
      <div class="summary-card bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <div class="flex items-center space-x-3">
          <div class="p-2 bg-blue-100 dark:bg-blue-900/40 rounded-lg">
            <ChartBarIcon class="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <div class="text-sm font-medium text-blue-900 dark:text-blue-100">
              Avg Performance
            </div>
            <div class="text-lg font-bold text-blue-700 dark:text-blue-300">
              {{ averagePerformance.toFixed(1) }}%
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { formatDistanceToNow } from 'date-fns'
import * as d3 from 'd3'

// Icons
import {
  ChartBarIcon,
  ArrowPathIcon,
  EyeIcon,
  ArrowDownTrayIcon,
  XMarkIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  MinusIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon
} from '@heroicons/vue/24/outline'

// Services
import { useObservabilityEvents, DashboardEventType } from '@/services/observabilityEventService'
import type { 
  IntelligenceKPI,
  ObservabilityEvent 
} from '@/services/observabilityEventService'
import { DashboardComponent } from '@/types/coordination'

interface Props {
  kpiNames?: string[]
  autoRefresh?: boolean
  refreshInterval?: number
}

const props = withDefaults(defineProps<Props>(), {
  kpiNames: () => [],
  autoRefresh: true,
  refreshInterval: 30000 // 30 seconds
})

const emit = defineEmits<{
  kpiSelected: [kpi: IntelligenceKPI]
  alertTriggered: [kpi: IntelligenceKPI, severity: string]
}>()

// Services
const observabilityEvents = useObservabilityEvents()

// Component state
const loading = ref(false)
const timeRangeHours = ref(24)
const aggregationInterval = ref('1h')
const showForecasts = ref(false)
const selectedKPI = ref<IntelligenceKPI | null>(null)

// KPI data
const kpiMetrics = ref<IntelligenceKPI[]>([])

// Real-time subscription
let subscriptionId: string | null = null
let refreshTimer: number | null = null

// Computed properties
const normalKPICount = computed(() => 
  kpiMetrics.value.filter(kpi => kpi.threshold_status === 'normal').length
)

const warningKPICount = computed(() => 
  kpiMetrics.value.filter(kpi => kpi.threshold_status === 'warning').length
)

const criticalKPICount = computed(() => 
  kpiMetrics.value.filter(kpi => kpi.threshold_status === 'critical').length
)

const averagePerformance = computed(() => {
  if (kpiMetrics.value.length === 0) return 0
  
  const performanceScores = kpiMetrics.value.map(kpi => {
    switch (kpi.threshold_status) {
      case 'normal': return 100
      case 'warning': return 60
      case 'critical': return 20
      default: return 50
    }
  })
  
  return performanceScores.reduce((sum, score) => sum + score, 0) / performanceScores.length
})

/**
 * Initialize component
 */
onMounted(async () => {
  await nextTick()
  await loadKPIData()
  setupRealTimeUpdates()
  
  if (props.autoRefresh) {
    startAutoRefresh()
  }
})

/**
 * Cleanup on unmount
 */
onUnmounted(() => {
  if (subscriptionId) {
    observabilityEvents.unsubscribe(subscriptionId)
  }
  
  if (refreshTimer) {
    clearInterval(refreshTimer)
  }
})

/**
 * Load KPI data
 */
async function loadKPIData() {
  loading.value = true
  
  try {
    const params = {
      kpi_names: props.kpiNames.length > 0 ? props.kpiNames : undefined,
      time_range_hours: timeRangeHours.value,
      aggregation_interval: aggregationInterval.value,
      include_trends: true,
      include_forecasts: showForecasts.value
    }
    
    kpiMetrics.value = await observabilityEvents.getIntelligenceKPIs(params)
    
    // Render charts after data loads
    await nextTick()
    renderAllCharts()
    
    console.log(`✅ Loaded ${kpiMetrics.value.length} KPI metrics`)
    
  } catch (error) {
    console.error('Failed to load KPI data:', error)
    kpiMetrics.value = []
  } finally {
    loading.value = false
  }
}

/**
 * Setup real-time updates
 */
function setupRealTimeUpdates() {
  subscriptionId = observabilityEvents.subscribe(
    DashboardComponent.DASHBOARD,
    [DashboardEventType.INTELLIGENCE_KPI, DashboardEventType.PERFORMANCE_METRIC],
    handleRealTimeKPIUpdate,
    {},
    9 // High priority
  )
}

/**
 * Handle real-time KPI updates
 */
async function handleRealTimeKPIUpdate(event: ObservabilityEvent) {
  if (event.type === DashboardEventType.INTELLIGENCE_KPI) {
    const kpiUpdate = event.data
    
    // Find and update existing KPI
    const existingKPIIndex = kpiMetrics.value.findIndex(kpi => kpi.name === kpiUpdate.kpi_name)
    
    if (existingKPIIndex >= 0) {
      const existingKPI = kpiMetrics.value[existingKPIIndex]
      
      // Update current value
      existingKPI.current_value = kpiUpdate.current_value
      
      // Add new data point to historical data
      existingKPI.historical_data.push({
        timestamp: new Date().toISOString(),
        value: kpiUpdate.current_value,
        metadata: kpiUpdate.metadata
      })
      
      // Keep only recent data points
      if (existingKPI.historical_data.length > 100) {
        existingKPI.historical_data.shift()
      }
      
      // Update threshold status if changed
      if (kpiUpdate.threshold_status) {
        const oldStatus = existingKPI.threshold_status
        existingKPI.threshold_status = kpiUpdate.threshold_status
        
        // Emit alert if status worsened
        if (oldStatus === 'normal' && kpiUpdate.threshold_status !== 'normal') {
          emit('alertTriggered', existingKPI, kpiUpdate.threshold_status)
        }
      }
      
      // Re-render chart for updated KPI
      await nextTick()
      renderKPIChart(existingKPI)
      
      if (selectedKPI.value?.name === existingKPI.name) {
        renderDetailedChart(existingKPI)
      }
    }
  }
}

/**
 * Render all KPI charts
 */
function renderAllCharts() {
  kpiMetrics.value.forEach(kpi => {
    renderKPIChart(kpi)
  })
}

/**
 * Render individual KPI chart
 */
function renderKPIChart(kpi: IntelligenceKPI) {
  const chartRef = `chart-${kpi.name}`
  const svgElement = document.querySelector(`[ref="${chartRef}"]`) as SVGElement
  
  if (!svgElement || !kpi.historical_data.length) return

  const svg = d3.select(svgElement)
  svg.selectAll('*').remove()

  const margin = { top: 5, right: 5, bottom: 5, left: 5 }
  const width = svgElement.clientWidth - margin.left - margin.right
  const height = svgElement.clientHeight - margin.top - margin.bottom

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`)

  // Scales
  const xScale = d3.scaleTime()
    .domain(d3.extent(kpi.historical_data, d => new Date(d.timestamp)) as [Date, Date])
    .range([0, width])

  const yScale = d3.scaleLinear()
    .domain(d3.extent(kpi.historical_data, d => d.value) as [number, number])
    .nice()
    .range([height, 0])

  // Line generator
  const line = d3.line<any>()
    .x(d => xScale(new Date(d.timestamp)))
    .y(d => yScale(d.value))
    .curve(d3.curveMonotoneX)

  // Area generator for fill
  const area = d3.area<any>()
    .x(d => xScale(new Date(d.timestamp)))
    .y0(height)
    .y1(d => yScale(d.value))
    .curve(d3.curveMonotoneX)

  // Add area
  g.append('path')
    .datum(kpi.historical_data)
    .attr('fill', getKPIColor(kpi.name))
    .attr('fill-opacity', 0.1)
    .attr('d', area)

  // Add line
  g.append('path')
    .datum(kpi.historical_data)
    .attr('fill', 'none')
    .attr('stroke', getKPIColor(kpi.name))
    .attr('stroke-width', 2)
    .attr('d', line)

  // Add current value dot
  const lastDataPoint = kpi.historical_data[kpi.historical_data.length - 1]
  if (lastDataPoint) {
    g.append('circle')
      .attr('cx', xScale(new Date(lastDataPoint.timestamp)))
      .attr('cy', yScale(lastDataPoint.value))
      .attr('r', 3)
      .attr('fill', getKPIColor(kpi.name))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
  }

  // Add forecast if available
  if (showForecasts.value && kpi.forecast_data?.length) {
    const forecastLine = d3.line<any>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX)

    g.append('path')
      .datum(kpi.forecast_data)
      .attr('fill', 'none')
      .attr('stroke', getKPIColor(kpi.name))
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,4')
      .attr('stroke-opacity', 0.6)
      .attr('d', forecastLine)
  }
}

/**
 * Render detailed chart
 */
function renderDetailedChart(kpi: IntelligenceKPI) {
  const svgElement = document.querySelector('[ref="detailedChart"]') as SVGElement
  
  if (!svgElement || !kpi.historical_data.length) return

  const svg = d3.select(svgElement)
  svg.selectAll('*').remove()

  const margin = { top: 20, right: 30, bottom: 40, left: 50 }
  const width = svgElement.clientWidth - margin.left - margin.right
  const height = svgElement.clientHeight - margin.top - margin.bottom

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`)

  // Scales
  const xScale = d3.scaleTime()
    .domain(d3.extent(kpi.historical_data, d => new Date(d.timestamp)) as [Date, Date])
    .range([0, width])

  const yScale = d3.scaleLinear()
    .domain(d3.extent(kpi.historical_data, d => d.value) as [number, number])
    .nice()
    .range([height, 0])

  // Axes
  g.append('g')
    .attr('transform', `translate(0,${height})`)
    .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M')))

  g.append('g')
    .call(d3.axisLeft(yScale))

  // Grid lines
  g.append('g')
    .attr('class', 'grid')
    .attr('transform', `translate(0,${height})`)
    .call(d3.axisBottom(xScale)
      .tickSize(-height)
      .tickFormat(() => '')
    )
    .style('stroke-dasharray', '2,2')
    .style('opacity', 0.3)

  g.append('g')
    .attr('class', 'grid')
    .call(d3.axisLeft(yScale)
      .tickSize(-width)
      .tickFormat(() => '')
    )
    .style('stroke-dasharray', '2,2')
    .style('opacity', 0.3)

  // Line and area
  const line = d3.line<any>()
    .x(d => xScale(new Date(d.timestamp)))
    .y(d => yScale(d.value))
    .curve(d3.curveMonotoneX)

  const area = d3.area<any>()
    .x(d => xScale(new Date(d.timestamp)))
    .y0(height)
    .y1(d => yScale(d.value))
    .curve(d3.curveMonotoneX)

  g.append('path')
    .datum(kpi.historical_data)
    .attr('fill', getKPIColor(kpi.name))
    .attr('fill-opacity', 0.1)
    .attr('d', area)

  g.append('path')
    .datum(kpi.historical_data)
    .attr('fill', 'none')
    .attr('stroke', getKPIColor(kpi.name))
    .attr('stroke-width', 2)
    .attr('d', line)

  // Add forecast
  if (showForecasts.value && kpi.forecast_data?.length) {
    const forecastLine = d3.line<any>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX)

    g.append('path')
      .datum(kpi.forecast_data)
      .attr('fill', 'none')
      .attr('stroke', '#F59E0B')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4,4')
      .attr('stroke-opacity', 0.8)
      .attr('d', forecastLine)
  }

  // Add data points
  g.selectAll('.data-point')
    .data(kpi.historical_data)
    .join('circle')
    .attr('class', 'data-point')
    .attr('cx', d => xScale(new Date(d.timestamp)))
    .attr('cy', d => yScale(d.value))
    .attr('r', 3)
    .attr('fill', getKPIColor(kpi.name))
    .attr('stroke', '#fff')
    .attr('stroke-width', 2)
    .on('mouseover', function(event, d) {
      // Add tooltip
      const tooltip = d3.select('body').append('div')
        .attr('class', 'kpi-tooltip')
        .style('position', 'absolute')
        .style('background', 'rgba(0,0,0,0.8)')
        .style('color', 'white')
        .style('padding', '8px')
        .style('border-radius', '4px')
        .style('font-size', '12px')
        .style('z-index', '1000')
        .html(`
          <div>Value: ${formatKPIValue(d.value, kpi.unit)}</div>
          <div>Time: ${new Date(d.timestamp).toLocaleString()}</div>
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px')
      
      setTimeout(() => tooltip.remove(), 3000)
    })
}

/**
 * Control functions
 */
async function refreshData() {
  await loadKPIData()
}

function toggleForecasts() {
  showForecasts.value = !showForecasts.value
  refreshData()
}

function selectKPI(kpi: IntelligenceKPI) {
  selectedKPI.value = selectedKPI.value?.name === kpi.name ? null : kpi
  
  if (selectedKPI.value) {
    nextTick(() => {
      renderDetailedChart(selectedKPI.value!)
    })
  }
  
  emit('kpiSelected', kpi)
}

function viewKPIDetails(kpi: IntelligenceKPI) {
  selectKPI(kpi)
}

function exportKPIData(kpi: IntelligenceKPI) {
  try {
    const exportData = {
      kpi: kpi.name,
      description: kpi.description,
      current_value: kpi.current_value,
      unit: kpi.unit,
      threshold_status: kpi.threshold_status,
      trend_direction: kpi.trend_direction,
      trend_strength: kpi.trend_strength,
      historical_data: kpi.historical_data,
      forecast_data: kpi.forecast_data,
      exported_at: new Date().toISOString()
    }
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    })
    
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `kpi-${kpi.name}-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
  } catch (error) {
    console.error('Failed to export KPI data:', error)
  }
}

/**
 * Auto-refresh functionality
 */
function startAutoRefresh() {
  refreshTimer = setInterval(() => {
    if (!loading.value) {
      loadKPIData()
    }
  }, props.refreshInterval)
}

/**
 * Utility functions
 */
function formatKPIName(name: string): string {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

function formatKPIValue(value: number, unit: string): string {
  if (unit === 'ms') {
    return value.toFixed(0)
  } else if (unit === 'ratio' || unit === 'score') {
    return value.toFixed(3)
  } else if (unit === 'multiplier') {
    return value.toFixed(2)
  } else if (unit === 'concepts/hour') {
    return Math.round(value).toString()
  }
  return value.toFixed(2)
}

function getKPIColor(kpiName: string): string {
  const colorMap: { [key: string]: string } = {
    'workflow_quality': '#10B981',
    'coordination_amplification': '#3B82F6',
    'semantic_search_latency': '#F59E0B',
    'context_compression_efficiency': '#8B5CF6',
    'agent_knowledge_acquisition': '#EF4444'
  }
  return colorMap[kpiName] || '#64748B'
}

function getThresholdStatusClass(status: string): string {
  const classMap: { [key: string]: string } = {
    'normal': 'bg-green-500',
    'warning': 'bg-yellow-500',
    'critical': 'bg-red-500'
  }
  return classMap[status] || 'bg-slate-500'
}

function getTrendColorClass(direction: string): string {
  const classMap: { [key: string]: string } = {
    'up': 'text-green-600 dark:text-green-400',
    'down': 'text-red-600 dark:text-red-400',
    'stable': 'text-slate-600 dark:text-slate-400'
  }
  return classMap[direction] || 'text-slate-600 dark:text-slate-400'
}

function getKPIAverage(kpi: IntelligenceKPI): number {
  if (!kpi.historical_data.length) return 0
  return kpi.historical_data.reduce((sum, point) => sum + point.value, 0) / kpi.historical_data.length
}

function getKPIMin(kpi: IntelligenceKPI): number {
  if (!kpi.historical_data.length) return 0
  return Math.min(...kpi.historical_data.map(point => point.value))
}

function getKPIMax(kpi: IntelligenceKPI): number {
  if (!kpi.historical_data.length) return 0
  return Math.max(...kpi.historical_data.map(point => point.value))
}

function getKPIInsights(kpi: IntelligenceKPI): Array<{ text: string; severity: 'info' | 'warning' | 'critical' }> {
  const insights = []
  
  if (kpi.trend_direction === 'down' && kpi.trend_strength > 0.5) {
    insights.push({
      text: `Strong downward trend detected (${(kpi.trend_strength * 100).toFixed(1)}%)`,
      severity: 'warning' as const
    })
  }
  
  if (kpi.threshold_status === 'critical') {
    insights.push({
      text: 'KPI is in critical range - immediate attention required',
      severity: 'critical' as const
    })
  }
  
  if (kpi.historical_data.length > 10) {
    const recent = kpi.historical_data.slice(-5)
    const older = kpi.historical_data.slice(-10, -5)
    const recentAvg = recent.reduce((sum, p) => sum + p.value, 0) / recent.length
    const olderAvg = older.reduce((sum, p) => sum + p.value, 0) / older.length
    
    if (recentAvg > olderAvg * 1.1) {
      insights.push({
        text: 'Recent improvement detected in performance',
        severity: 'info' as const
      })
    }
  }
  
  return insights
}

// Watch for prop changes
watch(() => props.kpiNames, () => {
  loadKPIData()
})
</script>

<style scoped>
.intelligence-kpi-dashboard {
  @apply w-full;
}

.kpi-card {
  transition: all 0.3s ease;
}

.kpi-card:hover {
  transform: translateY(-2px);
}

.kpi-chart-container {
  cursor: pointer;
}

.detailed-chart-container {
  position: relative;
}

/* Chart styling */
:deep(.grid line) {
  stroke: rgba(148, 163, 184, 0.3);
}

:deep(.grid path) {
  stroke-width: 0;
}

/* Loading animation */
.loading-spinner {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Summary cards */
.summary-card {
  transition: all 0.2s ease;
}

.summary-card:hover {
  transform: scale(1.02);
}

/* Tooltip styles */
:deep(.kpi-tooltip) {
  pointer-events: none !important;
}

/* Dark mode adjustments */
.dark :deep(.grid line) {
  stroke: rgba(148, 163, 184, 0.2);
}

.dark .kpi-chart-container svg {
  background: transparent;
}
</style>