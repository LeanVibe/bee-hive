<template>
  <div class="agent-graph-dashboard">
    <!-- Header -->
    <div class="dashboard-header flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
      <div>
        <h1 class="text-3xl font-bold text-slate-900 dark:text-white">
          Agent Graph Visualization
        </h1>
        <p class="mt-2 text-slate-600 dark:text-slate-400">
          Real-time multi-agent coordination and performance visualization
        </p>
      </div>
      
      <div class="mt-4 sm:mt-0 flex items-center space-x-3">
        <!-- View Toggle -->
        <div class="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
          <button
            @click="currentView = 'graph'"
            class="px-3 py-2 text-sm font-medium rounded-md transition-colors"
            :class="currentView === 'graph' 
              ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm' 
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'"
          >
            Graph View
          </button>
          <button
            @click="currentView = 'heatmap'"
            class="px-3 py-2 text-sm font-medium rounded-md transition-colors"
            :class="currentView === 'heatmap' 
              ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm' 
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'"
          >
            Heatmap
          </button>
          <button
            @click="currentView = 'combined'"
            class="px-3 py-2 text-sm font-medium rounded-md transition-colors"
            :class="currentView === 'combined' 
              ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm' 
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'"
          >
            Combined
          </button>
        </div>
        
        <!-- Connection Status -->
        <div class="flex items-center space-x-2">
          <div
            class="w-2 h-2 rounded-full"
            :class="connectionStatus.connected ? 'bg-green-500' : 'bg-red-500'"
          ></div>
          <span class="text-sm text-gray-600 dark:text-gray-400">
            {{ connectionStatus.connected ? 'Connected' : 'Disconnected' }}
          </span>
        </div>
        
        <!-- Real-time Toggle -->
        <label class="flex items-center">
          <input
            type="checkbox"
            v-model="realtimeEnabled"
            @change="toggleRealtime"
            class="rounded border-gray-300 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
          />
          <span class="ml-2 text-sm text-gray-600 dark:text-gray-400">Real-time</span>
        </label>
        
        <!-- Refresh Button -->
        <button
          @click="refreshData"
          :disabled="isRefreshing"
          class="btn-secondary"
          :class="{ 'opacity-50 cursor-not-allowed': isRefreshing }"
        >
          <ArrowPathIcon 
            class="w-4 h-4 mr-2" 
            :class="{ 'animate-spin': isRefreshing }"
          />
          Refresh
        </button>
      </div>
    </div>
    
    <!-- System Metrics Overview -->
    <div class="metrics-overview grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <div class="metric-card bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Active Agents</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ metrics.activeAgents }}</p>
          </div>
          <div class="p-3 bg-blue-100 dark:bg-blue-900 rounded-full">
            <UsersIcon class="w-6 h-6 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        <div class="mt-2 flex items-center text-sm">
          <component 
            :is="getTrendIcon(metrics.agentTrend)" 
            class="w-4 h-4 mr-1"
            :class="getTrendColorClass(metrics.agentTrend)"
          />
          <span class="text-gray-600 dark:text-gray-400">vs last hour</span>
        </div>
      </div>
      
      <div class="metric-card bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Active Sessions</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ metrics.activeSessions }}</p>
          </div>
          <div class="p-3 bg-green-100 dark:bg-green-900 rounded-full">
            <PlayIcon class="w-6 h-6 text-green-600 dark:text-green-400" />
          </div>
        </div>
        <div class="mt-2 flex items-center text-sm">
          <component 
            :is="getTrendIcon(metrics.sessionTrend)" 
            class="w-4 h-4 mr-1"
            :class="getTrendColorClass(metrics.sessionTrend)"
          />
          <span class="text-gray-600 dark:text-gray-400">vs last hour</span>
        </div>
      </div>
      
      <div class="metric-card bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Performance</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ metrics.avgPerformance }}%</p>
          </div>
          <div class="p-3 bg-yellow-100 dark:bg-yellow-900 rounded-full">
            <ChartBarIcon class="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
          </div>
        </div>
        <div class="mt-2 flex items-center text-sm">
          <component 
            :is="getTrendIcon(metrics.performanceTrend)" 
            class="w-4 h-4 mr-1"
            :class="getTrendColorClass(metrics.performanceTrend)"
          />
          <span class="text-gray-600 dark:text-gray-400">vs last hour</span>
        </div>
      </div>
      
      <div class="metric-card bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Events/Min</p>
            <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ metrics.eventRate }}</p>
          </div>
          <div class="p-3 bg-purple-100 dark:bg-purple-900 rounded-full">
            <BoltIcon class="w-6 h-6 text-purple-600 dark:text-purple-400" />
          </div>
        </div>
        <div class="mt-2 flex items-center text-sm">
          <component 
            :is="getTrendIcon(metrics.eventTrend)" 
            class="w-4 h-4 mr-1"
            :class="getTrendColorClass(metrics.eventTrend)"
          />
          <span class="text-gray-600 dark:text-gray-400">vs last hour</span>
        </div>
      </div>
    </div>
    
    <!-- Main Visualization Area -->
    <div class="visualization-area relative">
      <!-- Graph View -->
      <div 
        v-show="currentView === 'graph' || currentView === 'combined'"
        class="graph-section"
        :class="{ 'mb-6': currentView === 'combined' }"
      >
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div class="relative" :style="{ height: currentView === 'combined' ? '400px' : '600px' }">
            <!-- Graph Controls -->
            <AgentGraphControls
              :agents="realtimeGraph.getNodes()"
              :sessions="availableSessions"
              :zoom-level="graphState.zoomLevel"
              :node-count="realtimeGraph.metrics.totalNodes"
              :link-count="realtimeGraph.metrics.totalLinks"
              :performance-stats="realtimeGraph.performanceStats"
              @filters-changed="handleFiltersChanged"
              @layout-changed="handleLayoutChanged"
              @zoom-in="handleZoomIn"
              @zoom-out="handleZoomOut"
              @zoom-to="handleZoomTo"
              @reset-zoom="handleResetZoom"
              @focus-on="handleFocusOn"
              @performance-settings-changed="handlePerformanceSettingsChanged"
            />
            
            <!-- Main Graph Visualization -->
            <AgentGraphVisualization
              ref="graphVisualization"
              :width="graphDimensions.width"
              :height="graphDimensions.height"
              :auto-layout="graphSettings.autoLayout"
              :show-controls="false"
              :initial-zoom="graphState.zoomLevel"
              @node-selected="handleNodeSelected"
              @graph-updated="handleGraphUpdated"
            />
            
            <!-- Performance Overlay -->
            <div 
              v-if="showPerformanceOverlay"
              class="performance-overlay absolute bottom-4 right-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 border border-gray-200 dark:border-gray-700"
            >
              <div class="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                <div class="flex justify-between">
                  <span>FPS:</span>
                  <span class="font-mono">{{ Math.round(realtimeGraph.performanceStats.lastFrameTime || 0) }}</span>
                </div>
                <div class="flex justify-between">
                  <span>Updates/sec:</span>
                  <span class="font-mono">{{ Math.round(realtimeGraph.metrics.eventRate) }}</span>
                </div>
                <div class="flex justify-between">
                  <span>Latency:</span>
                  <span class="font-mono">{{ Math.round(realtimeGraph.metrics.averageLatency) }}ms</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Heatmap View -->
      <div 
        v-show="currentView === 'heatmap' || currentView === 'combined'"
        class="heatmap-section"
      >
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <AgentPerformanceHeatmap
            :agents="availableAgents"
            :sessions="availableSessions"
            :width="heatmapDimensions.width"
            :height="heatmapDimensions.height"
            :show-trend-lines="heatmapSettings.showTrendLines"
            @agent-selected="handleAgentSelectedFromHeatmap"
            @cell-selected="handleHeatmapCellSelected"
            @anomaly-detected="handleAnomalyDetected"
          />
        </div>
      </div>
    </div>
    
    <!-- Side Panel for Details -->
    <div 
      v-if="selectedAgent || selectedEvent"
      class="details-panel fixed right-0 top-0 h-full w-96 bg-white dark:bg-gray-800 shadow-xl border-l border-gray-200 dark:border-gray-700 z-50 transform transition-transform duration-300"
      :class="{ 'translate-x-0': showDetailsPanel, 'translate-x-full': !showDetailsPanel }"
    >
      <div class="p-6 h-full overflow-y-auto">
        <!-- Panel Header -->
        <div class="flex items-center justify-between mb-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
            {{ selectedAgent ? 'Agent Details' : 'Event Details' }}
          </h3>
          <button
            @click="closeDetailsPanel"
            class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <XMarkIcon class="w-5 h-5" />
          </button>
        </div>
        
        <!-- Agent Details -->
        <div v-if="selectedAgent" class="space-y-6">
          <div>
            <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Agent Information
            </h4>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">ID:</span>
                <span class="font-mono text-gray-900 dark:text-white">{{ selectedAgent.agent_id }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">Status:</span>
                <span 
                  class="px-2 py-1 rounded text-xs font-medium"
                  :class="getStatusClass(selectedAgent.status)"
                >
                  {{ selectedAgent.status }}
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">Sessions:</span>
                <span class="text-gray-900 dark:text-white">{{ selectedAgent.session_ids.length }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">Events:</span>
                <span class="text-gray-900 dark:text-white">{{ selectedAgent.event_count }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">Errors:</span>
                <span class="text-gray-900 dark:text-white">{{ selectedAgent.error_count }}</span>
              </div>
            </div>
          </div>
          
          <!-- Performance Metrics -->
          <div>
            <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Performance Metrics
            </h4>
            <div class="space-y-3">
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span class="text-gray-600 dark:text-gray-400">Success Rate</span>
                  <span class="text-gray-900 dark:text-white">{{ calculateSuccessRate(selectedAgent) }}%</span>
                </div>
                <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div 
                    class="h-2 rounded-full bg-green-500"
                    :style="{ width: `${calculateSuccessRate(selectedAgent)}%` }"
                  ></div>
                </div>
              </div>
              
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span class="text-gray-600 dark:text-gray-400">Activity Level</span>
                  <span class="text-gray-900 dark:text-white">{{ calculateActivityLevel(selectedAgent) }}%</span>
                </div>
                <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div 
                    class="h-2 rounded-full bg-blue-500"
                    :style="{ width: `${calculateActivityLevel(selectedAgent)}%` }"
                  ></div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Recent Events -->
          <div>
            <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Recent Events
            </h4>
            <div class="space-y-2 max-h-64 overflow-y-auto">
              <div
                v-for="event in getAgentRecentEvents(selectedAgent)"
                :key="event.event_id"
                class="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                @click="selectEvent(event)"
              >
                <div class="flex items-center justify-between mb-1">
                  <span class="text-xs font-medium text-gray-900 dark:text-white">
                    {{ event.hook_type }}
                  </span>
                  <span class="text-xs text-gray-500 dark:text-gray-400">
                    {{ formatEventTime(event.timestamp) }}
                  </span>
                </div>
                <div class="text-xs text-gray-600 dark:text-gray-400 truncate">
                  {{ JSON.stringify(event.payload).substring(0, 50) }}...
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Event Details -->
        <div v-if="selectedEvent" class="space-y-6">
          <div>
            <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Event Information
            </h4>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">Type:</span>
                <span class="font-mono text-gray-900 dark:text-white">{{ selectedEvent.hook_type }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">Agent:</span>
                <span class="font-mono text-gray-900 dark:text-white">{{ selectedEvent.agent_id }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">Session:</span>
                <span class="font-mono text-gray-900 dark:text-white">{{ selectedEvent.session_id }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">Priority:</span>
                <span class="text-gray-900 dark:text-white">{{ selectedEvent.priority }}</span>
              </div>
            </div>
          </div>
          
          <!-- Event Payload -->
          <div>
            <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Payload
            </h4>
            <pre class="text-xs bg-gray-100 dark:bg-gray-700 rounded-lg p-3 overflow-x-auto">{{ JSON.stringify(selectedEvent.payload, null, 2) }}</pre>
          </div>
          
          <!-- Event Metadata -->
          <div v-if="selectedEvent.metadata && Object.keys(selectedEvent.metadata).length > 0">
            <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Metadata
            </h4>
            <pre class="text-xs bg-gray-100 dark:bg-gray-700 rounded-lg p-3 overflow-x-auto">{{ JSON.stringify(selectedEvent.metadata, null, 2) }}</pre>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Alerts and Notifications -->
    <div 
      v-if="alerts.length > 0"
      class="alerts-container fixed bottom-4 right-4 space-y-2 z-40"
    >
      <div
        v-for="alert in alerts"
        :key="alert.id"
        class="alert-card bg-white dark:bg-gray-800 rounded-lg shadow-lg border-l-4 p-4 max-w-sm"
        :class="getAlertBorderClass(alert.type)"
      >
        <div class="flex items-start">
          <div class="flex-shrink-0">
            <component 
              :is="getAlertIcon(alert.type)"
              class="w-5 h-5"
              :class="getAlertIconClass(alert.type)"
            />
          </div>
          <div class="ml-3 flex-1">
            <h4 class="text-sm font-medium text-gray-900 dark:text-white">
              {{ alert.title }}
            </h4>
            <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {{ alert.message }}
            </p>
            <div class="mt-2 flex space-x-2">
              <button
                @click="dismissAlert(alert.id)"
                class="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
              >
                Dismiss
              </button>
              <button
                v-if="alert.actionLabel"
                @click="handleAlertAction(alert)"
                class="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200"
              >
                {{ alert.actionLabel }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { 
  ArrowPathIcon,
  UsersIcon,
  PlayIcon,
  ChartBarIcon,
  BoltIcon,
  XMarkIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  MinusIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/vue/24/outline'

// Components
import AgentGraphVisualization from '@/components/visualization/AgentGraphVisualization.vue'
import AgentGraphControls from '@/components/visualization/AgentGraphControls.vue'
import AgentPerformanceHeatmap from '@/components/visualization/AgentPerformanceHeatmap.vue'

// Composables and utilities
import { useAgentGraphRealtime } from '@/composables/useAgentGraphRealtime'
import { useEventsStore } from '@/stores/events'
import type { AgentInfo, SessionInfo, HookEvent } from '@/types/hooks'

// Reactive state
const currentView = ref<'graph' | 'heatmap' | 'combined'>('graph')
const realtimeEnabled = ref(true)
const isRefreshing = ref(false)
const showDetailsPanel = ref(false)
const showPerformanceOverlay = ref(true)
const selectedAgent = ref<AgentInfo | null>(null)
const selectedEvent = ref<HookEvent | null>(null)

// Graph visualization ref
const graphVisualization = ref()

// Stores and composables
const eventsStore = useEventsStore()
const realtimeGraph = useAgentGraphRealtime()

// Graph state
const graphState = reactive({
  zoomLevel: 1,
  selectedNodes: [] as string[],
  focusedNode: null as string | null,
  layout: 'force' as 'force' | 'circle' | 'grid'
})

// Graph settings
const graphSettings = reactive({
  autoLayout: true,
  showWeakLinks: false,
  showLabels: true,
  animationSpeed: 1
})

// Heatmap settings
const heatmapSettings = reactive({
  showTrendLines: false,
  metric: 'performance' as 'performance' | 'activity' | 'errors',
  timeRange: '6h' as '1h' | '6h' | '24h' | '7d'
})

// Dimensions
const graphDimensions = reactive({
  width: 800,
  height: 600
})

const heatmapDimensions = reactive({
  width: 800,
  height: 400
})

// Connection status
const connectionStatus = reactive({
  connected: false,
  lastUpdate: null as Date | null,
  reconnectAttempts: 0
})

// System metrics
const metrics = reactive({
  activeAgents: 0,
  activeSessions: 0,
  avgPerformance: 0,
  eventRate: 0,
  agentTrend: 'stable' as 'up' | 'down' | 'stable',
  sessionTrend: 'stable' as 'up' | 'down' | 'stable',
  performanceTrend: 'stable' as 'up' | 'down' | 'stable',
  eventTrend: 'stable' as 'up' | 'down' | 'stable'
})

// Alerts
const alerts = ref<Array<{
  id: string
  type: 'info' | 'warning' | 'error' | 'success'
  title: string
  message: string
  timestamp: Date
  actionLabel?: string
  action?: () => void
}>>([])

// Computed properties
const availableAgents = computed(() => eventsStore.agents)
const availableSessions = computed(() => eventsStore.sessions)

// Auto-refresh timer
let refreshTimer: number | null = null
let metricsTimer: number | null = null

// Methods
const refreshData = async () => {
  isRefreshing.value = true
  
  try {
    // Refresh events store data
    await eventsStore.refreshEvents()
    
    // Reload graph data
    await realtimeGraph.loadInitialData()
    
    // Update metrics
    updateMetrics()
    
    connectionStatus.lastUpdate = new Date()
    
    showAlert({
      type: 'success',
      title: 'Data Updated',
      message: 'Graph data has been refreshed successfully'
    })
    
  } catch (error) {
    console.error('Failed to refresh data:', error)
    showAlert({
      type: 'error',
      title: 'Refresh Failed',
      message: 'Failed to refresh graph data. Please try again.'
    })
  } finally {
    isRefreshing.value = false
  }
}

const toggleRealtime = () => {
  if (realtimeEnabled.value) {
    eventsStore.connectWebSocket()
    startAutoRefresh()
    showAlert({
      type: 'info',
      title: 'Real-time Enabled',
      message: 'Graph will update automatically with new events'
    })
  } else {
    eventsStore.disconnectWebSocket()
    stopAutoRefresh()
    showAlert({
      type: 'info',
      title: 'Real-time Disabled',
      message: 'Graph updates paused. Use refresh button to update manually.'
    })
  }
}

const startAutoRefresh = () => {
  if (refreshTimer) clearInterval(refreshTimer)
  
  refreshTimer = window.setInterval(async () => {
    if (realtimeEnabled.value && !isRefreshing.value) {
      await refreshData()
    }
  }, 30000) // Refresh every 30 seconds
}

const stopAutoRefresh = () => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
    refreshTimer = null
  }
}

const updateMetrics = () => {
  // Update active agents count
  const previousActiveAgents = metrics.activeAgents
  metrics.activeAgents = availableAgents.value.filter(a => a.status === 'active').length
  metrics.agentTrend = getNumberTrend(metrics.activeAgents, previousActiveAgents)
  
  // Update active sessions count
  const previousActiveSessions = metrics.activeSessions
  metrics.activeSessions = availableSessions.value.filter(s => s.status === 'active').length
  metrics.sessionTrend = getNumberTrend(metrics.activeSessions, previousActiveSessions)
  
  // Update average performance
  const previousAvgPerformance = metrics.avgPerformance
  if (availableAgents.value.length > 0) {
    const totalPerformance = availableAgents.value.reduce((sum, agent) => {
      return sum + calculateAgentPerformance(agent)
    }, 0)
    metrics.avgPerformance = Math.round(totalPerformance / availableAgents.value.length)
  } else {
    metrics.avgPerformance = 0
  }
  metrics.performanceTrend = getNumberTrend(metrics.avgPerformance, previousAvgPerformance)
  
  // Update event rate
  const previousEventRate = metrics.eventRate
  metrics.eventRate = Math.round(realtimeGraph.metrics.eventRate * 60) // Convert to events per minute
  metrics.eventTrend = getNumberTrend(metrics.eventRate, previousEventRate)
}

const getNumberTrend = (current: number, previous: number): 'up' | 'down' | 'stable' => {
  if (current > previous) return 'up'
  if (current < previous) return 'down'
  return 'stable'
}

const calculateAgentPerformance = (agent: AgentInfo): number => {
  if (agent.event_count === 0) return 50
  const errorRate = agent.error_count / agent.event_count
  return Math.max(0, Math.round(100 - (errorRate * 100)))
}

const calculateSuccessRate = (agent: AgentInfo): number => {
  if (agent.event_count === 0) return 0
  const successCount = agent.event_count - agent.error_count
  return Math.round((successCount / agent.event_count) * 100)
}

const calculateActivityLevel = (agent: AgentInfo): number => {
  // Calculate activity based on event count relative to other agents
  const maxEvents = Math.max(...availableAgents.value.map(a => a.event_count), 1)
  return Math.round((agent.event_count / maxEvents) * 100)
}

const getAgentRecentEvents = (agent: AgentInfo): HookEvent[] => {
  return eventsStore.filteredHookEvents
    .filter(event => event.agent_id === agent.agent_id)
    .slice(0, 10)
}

const formatEventTime = (timestamp: string): string => {
  const date = new Date(timestamp)
  return date.toLocaleTimeString()
}

// Event handlers
const handleFiltersChanged = (filters: any) => {
  console.log('Filters changed:', filters)
  // Apply filters to the graph visualization
}

const handleLayoutChanged = (settings: any) => {
  console.log('Layout changed:', settings)
  Object.assign(graphSettings, settings)
}

const handleZoomIn = () => {
  graphState.zoomLevel = Math.min(3, graphState.zoomLevel * 1.2)
}

const handleZoomOut = () => {
  graphState.zoomLevel = Math.max(0.1, graphState.zoomLevel / 1.2)
}

const handleZoomTo = (level: number) => {
  graphState.zoomLevel = level
}

const handleResetZoom = () => {
  graphState.zoomLevel = 1
  graphState.focusedNode = null
}

const handleFocusOn = (target: string) => {
  console.log('Focus on:', target)
  if (target.startsWith('agent-')) {
    const agentId = target.replace('agent-', '')
    const agent = availableAgents.value.find(a => a.agent_id === agentId)
    if (agent) {
      selectedAgent.value = agent
      showDetailsPanel.value = true
    }
  }
}

const handlePerformanceSettingsChanged = (settings: any) => {
  console.log('Performance settings changed:', settings)
  // Apply performance settings
}

const handleNodeSelected = (node: any) => {
  const agent = availableAgents.value.find(a => a.agent_id === node.agent_id)
  if (agent) {
    selectedAgent.value = agent
    selectedEvent.value = null
    showDetailsPanel.value = true
  }
}

const handleGraphUpdated = (data: any) => {
  console.log('Graph updated:', data)
  // Handle graph updates
}

const handleAgentSelectedFromHeatmap = (agent: AgentInfo) => {
  selectedAgent.value = agent
  selectedEvent.value = null
  showDetailsPanel.value = true
}

const handleHeatmapCellSelected = (data: any) => {
  console.log('Heatmap cell selected:', data)
  selectedAgent.value = data.agent
  showDetailsPanel.value = true
}

const handleAnomalyDetected = (anomaly: any) => {
  console.log('Anomaly detected:', anomaly)
  showAlert({
    type: 'warning',
    title: 'Performance Anomaly Detected',
    message: `Agent ${anomaly.agent.agent_id} showing unusual performance pattern`,
    actionLabel: 'View Details',
    action: () => {
      selectedAgent.value = anomaly.agent
      showDetailsPanel.value = true
    }
  })
}

const selectEvent = (event: HookEvent) => {
  selectedEvent.value = event
  selectedAgent.value = null
}

const closeDetailsPanel = () => {
  showDetailsPanel.value = false
  selectedAgent.value = null
  selectedEvent.value = null
}

// Alert methods
const showAlert = (alertData: Omit<typeof alerts.value[0], 'id' | 'timestamp'>) => {
  const alert = {
    id: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date(),
    ...alertData
  }
  
  alerts.value.unshift(alert)
  
  // Auto-dismiss after 5 seconds unless it's an error
  if (alert.type !== 'error') {
    setTimeout(() => {
      dismissAlert(alert.id)
    }, 5000)
  }
}

const dismissAlert = (id: string) => {
  const index = alerts.value.findIndex(alert => alert.id === id)
  if (index > -1) {
    alerts.value.splice(index, 1)
  }
}

const handleAlertAction = (alert: any) => {
  if (alert.action) {
    alert.action()
  }
  dismissAlert(alert.id)
}

// Utility methods
const getTrendIcon = (trend: string) => {
  switch (trend) {
    case 'up': return ArrowTrendingUpIcon
    case 'down': return ArrowTrendingDownIcon
    default: return MinusIcon
  }
}

const getTrendColorClass = (trend: string): string => {
  switch (trend) {
    case 'up': return 'text-green-600 dark:text-green-400'
    case 'down': return 'text-red-600 dark:text-red-400'
    default: return 'text-gray-600 dark:text-gray-400'
  }
}

const getStatusClass = (status: string): string => {
  const classes = {
    'active': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'idle': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'error': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    'blocked': 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
  }
  return classes[status as keyof typeof classes] || 'bg-gray-100 text-gray-800'
}

const getAlertIcon = (type: string) => {
  switch (type) {
    case 'error': return XCircleIcon
    case 'warning': return ExclamationTriangleIcon
    case 'success': return CheckCircleIcon
    default: return InformationCircleIcon
  }
}

const getAlertIconClass = (type: string): string => {
  switch (type) {
    case 'error': return 'text-red-500'
    case 'warning': return 'text-yellow-500'
    case 'success': return 'text-green-500'
    default: return 'text-blue-500'
  }
}

const getAlertBorderClass = (type: string): string => {
  switch (type) {
    case 'error': return 'border-red-400'
    case 'warning': return 'border-yellow-400'
    case 'success': return 'border-green-400'
    default: return 'border-blue-400'
  }
}

// Resize handler
const handleResize = () => {
  const container = document.querySelector('.graph-section')
  if (container) {
    const rect = container.getBoundingClientRect()
    graphDimensions.width = rect.width - 40 // Account for padding
    graphDimensions.height = currentView.value === 'combined' ? 400 : 600
  }
  
  const heatmapContainer = document.querySelector('.heatmap-section')
  if (heatmapContainer) {
    const rect = heatmapContainer.getBoundingClientRect()
    heatmapDimensions.width = rect.width - 40
    heatmapDimensions.height = 400
  }
}

// Lifecycle
onMounted(async () => {
  // Initialize connection status
  connectionStatus.connected = eventsStore.wsConnected
  
  // Load initial data
  await refreshData()
  
  // Start auto-refresh if real-time is enabled
  if (realtimeEnabled.value) {
    startAutoRefresh()
  }
  
  // Start metrics update timer
  metricsTimer = window.setInterval(updateMetrics, 5000)
  
  // Set up resize listener
  window.addEventListener('resize', handleResize)
  nextTick(handleResize)
  
  // Show welcome message
  showAlert({
    type: 'info',
    title: 'Agent Graph Dashboard',
    message: 'Welcome to the real-time agent coordination visualization'
  })
})

onUnmounted(() => {
  stopAutoRefresh()
  
  if (metricsTimer) {
    clearInterval(metricsTimer)
  }
  
  window.removeEventListener('resize', handleResize)
})

// Watchers
watch(() => eventsStore.wsConnected, (connected) => {
  connectionStatus.connected = connected
  
  if (connected) {
    connectionStatus.reconnectAttempts = 0
    showAlert({
      type: 'success',
      title: 'Connection Restored',
      message: 'Real-time updates are now active'
    })
  } else {
    connectionStatus.reconnectAttempts++
    showAlert({
      type: 'warning',
      title: 'Connection Lost',
      message: 'Attempting to reconnect to real-time updates'
    })
  }
})

watch(() => currentView.value, () => {
  nextTick(() => {
    handleResize()
  })
})
</script>

<style scoped>
.agent-graph-dashboard {
  @apply min-h-screen bg-gray-50 dark:bg-gray-900 p-6;
}

.btn-secondary {
  @apply inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500;
}

.dark .btn-secondary {
  @apply border-gray-600 text-gray-300 bg-gray-800 hover:bg-gray-700 focus:ring-offset-gray-800;
}

.metric-card {
  transition: all 0.2s ease;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.visualization-area {
  min-height: 600px;
}

.details-panel {
  backdrop-filter: blur(10px);
}

.alert-card {
  backdrop-filter: blur(10px);
  animation: slideInRight 0.3s ease-out;
}

@keyframes slideInRight {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.performance-overlay {
  backdrop-filter: blur(10px);
  background-color: rgba(255, 255, 255, 0.95);
}

.dark .performance-overlay {
  background-color: rgba(31, 41, 55, 0.95);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .agent-graph-dashboard {
    padding: 1rem;
  }
  
  .details-panel {
    width: 100%;
  }
  
  .metrics-overview {
    grid-template-columns: 1fr;
  }
  
  .dashboard-header {
    flex-direction: column;
    align-items: stretch;
  }
  
  .dashboard-header > div:last-child {
    margin-top: 1rem;
    justify-content: center;
  }
}
</style>