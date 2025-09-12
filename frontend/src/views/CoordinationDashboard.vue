<template>
  <div class="coordination-dashboard">
    <!-- Header -->
    <div class="dashboard-header flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8">
      <div>
        <h1 class="text-3xl font-bold text-slate-900 dark:text-white">
          Multi-Agent Coordination Dashboard
        </h1>
        
        <!-- Breadcrumb Navigation -->
        <nav class="mt-2 flex items-center space-x-1 text-sm text-slate-600 dark:text-slate-400" aria-label="Breadcrumb">
          <template v-for="(breadcrumb, index) in navigation.breadcrumbs.value" :key="index">
            <button
              v-if="breadcrumb.onClick"
              @click="breadcrumb.onClick"
              class="hover:text-slate-900 dark:hover:text-white transition-colors"
            >
              {{ breadcrumb.label }}
            </button>
            <span v-else class="text-slate-500 dark:text-slate-400">
              {{ breadcrumb.label }}
            </span>
            
            <ChevronRightIcon 
              v-if="index < navigation.breadcrumbs.value.length - 1" 
              class="w-4 h-4 text-slate-400 dark:text-slate-500" 
            />
          </template>
        </nav>
        
        <p class="mt-1 text-slate-600 dark:text-slate-400">
          Real-time agent coordination, communication analysis, and system monitoring
        </p>
      </div>
      
      <div class="mt-4 sm:mt-0 flex items-center space-x-3">
        <!-- Session Selector -->
        <select
          v-model="selectedSession"
          @change="onSessionChange"
          class="input-field text-sm min-w-32"
        >
          <option value="all">All Sessions</option>
          <option 
            v-for="session in availableSessions" 
            :key="session.id" 
            :value="session.id"
          >
            {{ session.label }}
          </option>
        </select>
        
        <!-- Refresh Button -->
        <button
          @click="refreshDashboard"
          :disabled="loading"
          class="btn-secondary"
          :class="{ 'opacity-50 cursor-not-allowed': loading }"
        >
          <ArrowPathIcon 
            class="w-4 h-4 mr-2" 
            :class="{ 'animate-spin': loading }"
          />
          Refresh
        </button>
        
        <!-- Last Updated -->
        <div class="text-sm text-slate-500 dark:text-slate-400">
          Updated: {{ formatTime(lastUpdated) }}
        </div>
        
        <!-- Connection Status -->
        <div class="flex items-center space-x-2">
          <div 
            class="w-2 h-2 rounded-full"
            :class="connectionStatusClass"
          ></div>
          <span class="text-xs text-slate-500 dark:text-slate-400">
            {{ connectionStatus }}
          </span>
        </div>
      </div>
    </div>

    <!-- Main Content Container -->
    <div class="dashboard-content space-y-6">
      <!-- Quick Stats Row -->
      <div class="stats-grid grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Active Agents"
          :value="dashboardStats.activeAgents"
          :previous="dashboardStats.previousActiveAgents"
          icon="users"
          color="primary"
        />
        <MetricCard
          title="Total Messages"
          :value="dashboardStats.totalMessages"
          :unit="'today'"
          icon="chat-bubble-left"
          color="success"
        />
        <MetricCard
          title="Avg Response"
          :value="dashboardStats.avgResponseTime"
          unit="ms"
          icon="clock"
          color="warning"
        />
        <MetricCard
          title="Sessions"
          :value="dashboardStats.activeSessions"
          icon="squares-plus"
          color="info"
        />
      </div>

      <!-- Main Dashboard Layout -->
      <div class="dashboard-layout">
        <!-- Navigation Tabs -->
        <div class="dashboard-tabs flex space-x-1 bg-slate-100 dark:bg-slate-800 rounded-lg p-1 mb-6">
          <button
            v-for="tab in dashboardTabs"
            :key="tab.id"
            @click="activeTab = tab.id"
            class="tab-button flex-1 px-4 py-2 text-sm font-medium rounded-md transition-colors"
            :class="activeTab === tab.id 
              ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-sm' 
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'"
          >
            <component :is="tab.icon" class="w-4 h-4 inline mr-2" />
            {{ tab.label }}
            <span 
              v-if="tab.badge" 
              class="ml-2 px-2 py-0.5 text-xs bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400 rounded-full"
            >
              {{ tab.badge }}
            </span>
          </button>
        </div>

        <!-- Tab Content -->
        <div class="tab-content">
          <!-- Agent Graph View -->
          <div v-show="activeTab === 'graph'" class="graph-view">
            <ErrorBoundary
              boundary-id="graph-component"
              :component="DashboardComponent.GRAPH"
              fallback-component="GraphFallback"
              @error="(error) => errorHandler.reportError(error)"
            >
              <div class="glass-card rounded-xl p-6 h-[600px] relative">
                <div class="flex items-center justify-between mb-4">
                  <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
                    Agent Interaction Graph
                  </h3>
                  <div class="flex items-center space-x-3">
                    <AgentGraphControls 
                      v-model:layout="graphLayout"
                      v-model:visualization-mode="graphVisualizationMode"
                      @reset-view="resetGraphView"
                    />
                  </div>
                </div>
                
                <AgentGraphVisualization
                  ref="agentGraphRef"
                  :width="graphDimensions.width"
                  :height="graphDimensions.height"
                  :auto-layout="true"
                  :show-controls="false"
                  @node-selected="onGraphNodeSelected"
                  @node-double-click="onGraphNodeDoubleClick"
                />
              </div>
            </ErrorBoundary>
          </div>

          <!-- Chat Transcript View -->
          <div v-show="activeTab === 'transcript'" class="transcript-view">
            <div class="glass-card rounded-xl p-6">
              <div class="flex flex-col lg:flex-row lg:space-x-6 space-y-6 lg:space-y-0">
                <!-- Transcript Panel -->
                <div class="flex-1">
                  <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
                      Communication Transcript
                    </h3>
                    <div class="flex items-center space-x-3">
                      <!-- Transcript Controls -->
                      <select 
                        v-model="transcriptTimeRange"
                        @change="refreshTranscript"
                        class="input-field text-sm"
                      >
                        <option value="1h">Last Hour</option>
                        <option value="6h">Last 6 Hours</option>
                        <option value="24h">Last 24 Hours</option>
                        <option value="all">All Time</option>
                      </select>
                      
                      <button
                        @click="toggleTranscriptAutoRefresh"
                        :class="transcriptAutoRefresh 
                          ? 'btn-primary text-xs' 
                          : 'btn-secondary text-xs'"
                      >
                        <BoltIcon class="w-3 h-3 mr-1" />
                        Live
                      </button>
                    </div>
                  </div>
                  
                  <!-- Transcript Events -->
                  <div class="transcript-container max-h-96 overflow-y-auto space-y-3">
                    <div
                      v-for="event in transcriptEvents"
                      :key="event.id"
                      class="transcript-event p-4 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors cursor-pointer"
                      @click="selectTranscriptEvent(event)"
                      :class="selectedTranscriptEvent?.id === event.id ? 'ring-2 ring-primary-500' : ''"
                    >
                      <div class="flex items-start justify-between">
                        <div class="flex items-center space-x-3">
                          <div 
                            class="w-3 h-3 rounded-full"
                            :style="{ backgroundColor: getAgentColor(event.source_agent) }"
                          ></div>
                          <div>
                            <div class="font-medium text-sm text-slate-900 dark:text-white">
                              {{ event.source_agent }} → {{ event.target_agent || 'Broadcast' }}
                            </div>
                            <div class="text-xs text-slate-500 dark:text-slate-400">
                              {{ formatEventTime(event.timestamp) }}
                            </div>
                          </div>
                        </div>
                        <div class="flex items-center space-x-2">
                          <span 
                            class="px-2 py-1 text-xs rounded-full"
                            :class="getEventTypeClass(event.message_type)"
                          >
                            {{ event.message_type }}
                          </span>
                          <span 
                            v-if="event.tool_calls?.length"
                            class="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full"
                          >
                            {{ event.tool_calls.length }} tools
                          </span>
                        </div>
                      </div>
                      
                      <div class="mt-2 text-sm text-slate-700 dark:text-slate-300 line-clamp-2">
                        {{ event.content }}
                      </div>
                    </div>
                    
                    <!-- Empty State -->
                    <div 
                      v-if="!transcriptEvents.length && !loadingTranscript"
                      class="text-center py-8 text-slate-500 dark:text-slate-400"
                    >
                      <ChatBubbleLeftIcon class="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p>No communication events found</p>
                      <p class="text-sm">Select a different time range or session</p>
                    </div>
                    
                    <!-- Loading State -->
                    <div 
                      v-if="loadingTranscript"
                      class="text-center py-8"
                    >
                      <div class="animate-spin w-6 h-6 border-2 border-primary-600 border-t-transparent rounded-full mx-auto mb-3"></div>
                      <p class="text-slate-500 dark:text-slate-400">Loading transcript...</p>
                    </div>
                  </div>
                </div>

                <!-- Transcript Details Panel -->
                <div class="w-full lg:w-80">
                  <div class="sticky top-0">
                    <h4 class="text-sm font-semibold text-slate-900 dark:text-white mb-3">
                      Event Details
                    </h4>
                    
                    <div 
                      v-if="selectedTranscriptEvent"
                      class="space-y-4 bg-slate-50 dark:bg-slate-800 rounded-lg p-4"
                    >
                      <!-- Event Metadata -->
                      <div class="space-y-2">
                        <div class="flex justify-between text-sm">
                          <span class="text-slate-500 dark:text-slate-400">Event ID:</span>
                          <span class="font-mono text-xs">{{ selectedTranscriptEvent.id.substring(0, 8) }}...</span>
                        </div>
                        <div class="flex justify-between text-sm">
                          <span class="text-slate-500 dark:text-slate-400">Session:</span>
                          <span class="font-mono text-xs">{{ selectedTranscriptEvent.session_id.substring(0, 8) }}...</span>
                        </div>
                        <div class="flex justify-between text-sm">
                          <span class="text-slate-500 dark:text-slate-400">Timestamp:</span>
                          <span class="text-xs">{{ formatDetailedTime(selectedTranscriptEvent.timestamp) }}</span>
                        </div>
                        <div class="flex justify-between text-sm">
                          <span class="text-slate-500 dark:text-slate-400">Response Time:</span>
                          <span class="text-xs">{{ selectedTranscriptEvent.response_time_ms || 'N/A' }}ms</span>
                        </div>
                      </div>
                      
                      <!-- Event Content -->
                      <div>
                        <h5 class="text-sm font-medium text-slate-900 dark:text-white mb-2">Content</h5>
                        <div class="text-sm text-slate-700 dark:text-slate-300 bg-white dark:bg-slate-900 rounded p-3 max-h-32 overflow-y-auto">
                          {{ selectedTranscriptEvent.content }}
                        </div>
                      </div>
                      
                      <!-- Tool Calls -->
                      <div v-if="selectedTranscriptEvent.tool_calls?.length">
                        <h5 class="text-sm font-medium text-slate-900 dark:text-white mb-2">Tool Calls</h5>
                        <div class="space-y-2">
                          <div 
                            v-for="(tool, index) in selectedTranscriptEvent.tool_calls"
                            :key="index"
                            class="text-xs bg-blue-50 dark:bg-blue-900/20 rounded p-2"
                          >
                            <div class="font-medium text-blue-700 dark:text-blue-300">{{ tool.name }}</div>
                            <div class="text-blue-600 dark:text-blue-400 mt-1">{{ tool.description || 'No description' }}</div>
                          </div>
                        </div>
                      </div>
                      
                      <!-- Context References -->
                      <div v-if="selectedTranscriptEvent.context_references?.length">
                        <h5 class="text-sm font-medium text-slate-900 dark:text-white mb-2">Context Shared</h5>
                        <div class="space-y-1">
                          <div 
                            v-for="context in selectedTranscriptEvent.context_references"
                            :key="context"
                            class="text-xs bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded px-2 py-1"
                          >
                            {{ context }}
                          </div>
                        </div>
                      </div>
                      
                      <!-- Action Buttons -->
                      <div class="flex space-x-2 pt-3 border-t border-slate-200 dark:border-slate-600">
                        <button
                          @click="jumpToAgentInGraph(selectedTranscriptEvent.source_agent)"
                          class="btn-secondary text-xs flex-1"
                        >
                          <ShareIcon class="w-3 h-3 mr-1" />
                          Show in Graph
                        </button>
                        <button
                          @click="analyzeEventPattern(selectedTranscriptEvent)"
                          class="btn-secondary text-xs flex-1"
                        >
                          <ChartBarIcon class="w-3 h-3 mr-1" />
                          Analyze
                        </button>
                      </div>
                    </div>
                    
                    <!-- No Event Selected -->
                    <div 
                      v-else
                      class="text-center py-8 text-slate-500 dark:text-slate-400"
                    >
                      <DocumentTextIcon class="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p class="text-sm">Select an event to view details</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Analysis & Debugging View -->
          <div v-show="activeTab === 'analysis'" class="analysis-view">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <!-- Pattern Analysis -->
              <div class="glass-card rounded-xl p-6">
                <h3 class="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                  Communication Patterns
                </h3>
                
                <div class="space-y-4">
                  <div 
                    v-for="pattern in detectedPatterns"
                    :key="pattern.id"
                    class="pattern-item p-4 rounded-lg border"
                    :class="getPatternSeverityClass(pattern.severity)"
                  >
                    <div class="flex items-start justify-between">
                      <div>
                        <h4 class="font-medium text-sm">{{ pattern.name }}</h4>
                        <p class="text-xs text-slate-600 dark:text-slate-400 mt-1">
                          {{ pattern.description }}
                        </p>
                      </div>
                      <span 
                        class="px-2 py-1 text-xs rounded-full"
                        :class="getSeverityBadgeClass(pattern.severity)"
                      >
                        {{ pattern.severity }}
                      </span>
                    </div>
                    
                    <div class="flex items-center justify-between mt-3">
                      <span class="text-xs text-slate-500 dark:text-slate-400">
                        Detected {{ pattern.occurrences }} times
                      </span>
                      <button
                        @click="investigatePattern(pattern)"
                        class="text-xs text-primary-600 dark:text-primary-400 hover:underline"
                      >
                        Investigate →
                      </button>
                    </div>
                  </div>
                  
                  <!-- No Patterns -->
                  <div 
                    v-if="!detectedPatterns.length"
                    class="text-center py-8 text-slate-500 dark:text-slate-400"
                  >
                    <CheckCircleIcon class="w-8 h-8 mx-auto mb-2 text-green-500" />
                    <p class="text-sm">No concerning patterns detected</p>
                  </div>
                </div>
              </div>

              <!-- Performance Metrics -->
              <div class="glass-card rounded-xl p-6">
                <h3 class="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                  Performance Analysis
                </h3>
                
                <div class="space-y-4">
                  <!-- Response Time Distribution -->
                  <div>
                    <h4 class="text-sm font-medium text-slate-900 dark:text-white mb-2">
                      Response Time Distribution
                    </h4>
                    <div class="bg-slate-100 dark:bg-slate-800 rounded-lg p-3">
                      <div class="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                        <div class="flex justify-between">
                          <span>Average:</span>
                          <span class="font-mono">{{ performanceMetrics.avgResponseTime }}ms</span>
                        </div>
                        <div class="flex justify-between">
                          <span>95th percentile:</span>
                          <span class="font-mono">{{ performanceMetrics.p95ResponseTime }}ms</span>
                        </div>
                        <div class="flex justify-between">
                          <span>Max:</span>
                          <span class="font-mono">{{ performanceMetrics.maxResponseTime }}ms</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <!-- Error Rates -->
                  <div>
                    <h4 class="text-sm font-medium text-slate-900 dark:text-white mb-2">
                      Error Analysis
                    </h4>
                    <div class="space-y-2">
                      <div class="flex justify-between text-sm">
                        <span>Error Rate:</span>
                        <span 
                          class="font-mono"
                          :class="performanceMetrics.errorRate > 0.05 ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'"
                        >
                          {{ (performanceMetrics.errorRate * 100).toFixed(2) }}%
                        </span>
                      </div>
                      <div class="flex justify-between text-sm">
                        <span>Total Errors:</span>  
                        <span class="font-mono">{{ performanceMetrics.totalErrors }}</span>
                      </div>
                    </div>
                  </div>
                  
                  <!-- Agent Performance -->
                  <div>
                    <h4 class="text-sm font-medium text-slate-900 dark:text-white mb-2">
                      Agent Performance Ranking
                    </h4>
                    <div class="space-y-2">
                      <div 
                        v-for="agent in topPerformingAgents"
                        :key="agent.id"
                        class="flex items-center justify-between text-xs"
                      >
                        <div class="flex items-center space-x-2">
                          <div 
                            class="w-2 h-2 rounded-full"
                            :style="{ backgroundColor: getAgentColor(agent.id) }"
                          ></div>
                          <span>{{ agent.name }}</span>
                        </div>
                        <div class="flex items-center space-x-2">
                          <span class="font-mono">{{ agent.avgResponseTime }}ms</span>
                          <div 
                            class="w-2 h-2 rounded-full"
                            :class="agent.performance > 0.9 ? 'bg-green-500' : agent.performance > 0.7 ? 'bg-yellow-500' : 'bg-red-500'"
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- System Monitoring View -->
          <div v-show="activeTab === 'monitoring'" class="monitoring-view">
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <!-- System Health -->
              <div class="glass-card rounded-xl p-6">
                <h3 class="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                  System Health
                </h3>
                <SystemHealthCard />
              </div>
              
              <!-- Recent Events -->
              <div class="glass-card rounded-xl p-6">
                <h3 class="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                  Recent Events
                </h3>
                <RecentEventsCard />
              </div>
              
              <!-- Hook Performance -->
              <div class="glass-card rounded-xl p-6">
                <h3 class="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                  Hook Performance
                </h3>
                <HookPerformanceCard />
              </div>
            </div>
            
            <!-- Agent Status Grid -->
            <div class="glass-card rounded-xl p-6 mt-6">
              <h3 class="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Agent Status Overview
              </h3>
              <AgentStatusGrid />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Event Analysis Modal -->
    <EventAnalysisModal
      v-if="showEventAnalysisModal"
      :event="selectedEventForAnalysis"
      @close="showEventAnalysisModal = false"
    />
    
    <!-- Pattern Investigation Modal -->
    <PatternInvestigationModal
      v-if="showPatternModal"
      :pattern="selectedPattern"
      @close="showPatternModal = false"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { formatDistanceToNow, format } from 'date-fns'

// Icons
import {
  ArrowPathIcon,
  ShareIcon,
  ChartBarIcon,
  DocumentTextIcon,
  ChatBubbleLeftIcon,
  BoltIcon,
  CheckCircleIcon,
  ChevronRightIcon,
  ShareIcon as ShareIconOutline
} from '@heroicons/vue/24/outline'

import {
  ChartBarIcon as ChartBarIconSolid,
  DocumentMagnifyingGlassIcon,
  Cog6ToothIcon,
  WrenchScrewdriverIcon
} from '@heroicons/vue/24/solid'

// Components
import MetricCard from '@/components/dashboard/MetricCard.vue'
import AgentGraphVisualization from '@/components/visualization/AgentGraphVisualization.vue'
import AgentGraphControls from '@/components/visualization/AgentGraphControls.vue'
import SystemHealthCard from '@/components/dashboard/SystemHealthCard.vue'
import RecentEventsCard from '@/components/dashboard/RecentEventsCard.vue'
import HookPerformanceCard from '@/components/dashboard/HookPerformanceCard.vue'
import AgentStatusGrid from '@/components/dashboard/AgentStatusGrid.vue'
import ErrorBoundary from '@/components/errors/ErrorBoundary.vue'

// Stores and services
import { useEventsStore } from '@/stores/events'
import { useConnectionStore } from '@/stores/connection'
import { useSessionColors } from '@/utils/SessionColorManager'
import { useCoordinationService } from '@/services/coordinationService'
import { useUnifiedWebSocket } from '@/services/unifiedWebSocketManager'
import { api } from '@/services/api'

// Modals (to be created)
// import EventAnalysisModal from '@/components/modals/EventAnalysisModal.vue'
// import PatternInvestigationModal from '@/components/modals/PatternInvestigationModal.vue'

const router = useRouter()
const eventsStore = useEventsStore()
const connectionStore = useConnectionStore()
const { getAgentColor, getSessionColor } = useSessionColors()

// Coordination services
const coordinationService = useCoordinationService()
const webSocketManager = useUnifiedWebSocket()

// Navigation system
import { useDashboardNavigation, NavigationIntent } from '@/composables/useDashboardNavigation'
const navigation = useDashboardNavigation()

// Error handling system
import { useDashboardErrorHandling } from '@/composables/useDashboardErrorHandling'
import { DashboardComponent } from '@/types/coordination'
const errorHandler = useDashboardErrorHandling()

// Refs
const agentGraphRef = ref()
const loading = ref(false)
const lastUpdated = ref<Date | null>(null)

// Dashboard state
const activeTab = computed({
  get: () => navigation.activeTab.value,
  set: (value: string) => navigation.navigateToTab(value)
})
const selectedSession = ref('all')
const selectedTranscriptEvent = ref(null)
const selectedEventForAnalysis = ref(null)
const selectedPattern = ref(null)

// Modal states
const showEventAnalysisModal = ref(false)
const showPatternModal = ref(false)

// Graph settings
const graphLayout = ref('force')
const graphVisualizationMode = ref('session')
const graphDimensions = ref({ width: 800, height: 500 })

// Transcript settings
const transcriptTimeRange = ref('1h')
const transcriptAutoRefresh = ref(true)
const loadingTranscript = ref(false)

// Data
const availableSessions = ref([])
const transcriptEvents = ref([])
const detectedPatterns = ref([])
const performanceMetrics = ref({
  avgResponseTime: 0,
  p95ResponseTime: 0,
  maxResponseTime: 0,
  errorRate: 0,
  totalErrors: 0
})
const topPerformingAgents = ref([])

// Dashboard tabs configuration
const dashboardTabs = ref([
  {
    id: 'graph',
    label: 'Agent Graph',
    icon: ChartBarIconSolid,
    badge: null
  },
  {
    id: 'transcript',
    label: 'Communications',
    icon: ChatBubbleLeftIcon,
    badge: computed(() => transcriptEvents.value.length || null)
  },
  {
    id: 'analysis',
    label: 'Analysis',
    icon: DocumentMagnifyingGlassIcon,
    badge: computed(() => detectedPatterns.value.filter(p => p.severity === 'HIGH').length || null)
  },
  {
    id: 'monitoring',
    label: 'Monitoring',
    icon: Cog6ToothIcon,
    badge: null
  }
])

// Computed properties
const connectionStatus = computed(() => {
  return connectionStore.isConnected ? 'Connected' : 'Disconnected'
})

const connectionStatusClass = computed(() => {
  return connectionStore.isConnected 
    ? 'bg-green-500 animate-pulse' 
    : 'bg-red-500'
})

const dashboardStats = computed(() => {
  const agents = eventsStore.agents || []
  const sessions = eventsStore.sessions || []
  
  return {
    activeAgents: agents.filter(a => a.status === 'active').length,
    previousActiveAgents: agents.length, // Mock previous value
    totalMessages: transcriptEvents.value.length,
    avgResponseTime: performanceMetrics.value.avgResponseTime,
    activeSessions: sessions.length
  }
})

// Methods
const formatTime = (date: Date | null) => {
  if (!date) return 'Never'
  return formatDistanceToNow(date, { addSuffix: true })
}

const formatEventTime = (timestamp: string) => {
  return format(new Date(timestamp), 'HH:mm:ss')
}

const formatDetailedTime = (timestamp: string) => {
  return format(new Date(timestamp), 'MMM dd, HH:mm:ss')
}

const refreshDashboard = async () => {
  loading.value = true
  
  try {
    await Promise.all([
      eventsStore.refreshEvents(),
      loadAvailableSessions(),
      refreshTranscript(),
      loadPerformanceMetrics(),
      detectPatterns()
    ])
    
    lastUpdated.value = new Date()
  } catch (error) {
    console.error('Failed to refresh dashboard:', error)
  } finally {
    loading.value = false
  }
}

const loadAvailableSessions = async () => {
  try {
    const response = await api.get('/coordination/sessions')
    availableSessions.value = response.data.active_sessions.map(sessionId => ({
      id: sessionId,
      label: `Session ${sessionId.substring(0, 8)}...`
    }))
  } catch (error) {
    console.error('Failed to load sessions:', error)
  }
}

const refreshTranscript = async () => {
  if (!selectedSession.value) return
  
  loadingTranscript.value = true
  
  try {
    const sessionId = selectedSession.value === 'all' ? 'all' : selectedSession.value
    const params = new URLSearchParams({
      limit: '100',
      start_time: getTimeRangeStart(transcriptTimeRange.value)
    })
    const response = await api.get(`/coordination/transcript/${sessionId}`, params)
    
    transcriptEvents.value = response.data.events || []
  } catch (error) {
    console.error('Failed to load transcript:', error)
    transcriptEvents.value = []
  } finally {
    loadingTranscript.value = false
  }
}

const loadPerformanceMetrics = async () => {
  try {
    // Mock performance data - replace with actual API call
    performanceMetrics.value = {
      avgResponseTime: Math.floor(Math.random() * 500) + 100,
      p95ResponseTime: Math.floor(Math.random() * 1000) + 500,
      maxResponseTime: Math.floor(Math.random() * 2000) + 1000,
      errorRate: Math.random() * 0.1,
      totalErrors: Math.floor(Math.random() * 50)
    }
    
    // Mock top performing agents
    topPerformingAgents.value = eventsStore.agents.slice(0, 5).map(agent => ({
      id: agent.agent_id,
      name: `Agent-${agent.agent_id.substring(0, 8)}`,
      avgResponseTime: Math.floor(Math.random() * 300) + 50,
      performance: Math.random()
    }))
  } catch (error) {
    console.error('Failed to load performance metrics:', error)
  }
}

const detectPatterns = async () => {
  try {
    // Mock pattern detection - replace with actual API call
    const mockPatterns = [
      {
        id: 'loop-1',
        name: 'Potential Infinite Loop',
        description: 'Agents exchanging similar messages repeatedly',
        severity: 'HIGH',
        occurrences: 5
      },
      {
        id: 'bottleneck-1',
        name: 'Communication Bottleneck',
        description: 'Single agent handling excessive message load',
        severity: 'MEDIUM',
        occurrences: 3
      }
    ]
    
    detectedPatterns.value = Math.random() > 0.7 ? mockPatterns : []
  } catch (error) {
    console.error('Failed to detect patterns:', error)
  }
}

const getTimeRangeStart = (range: string) => {
  const now = new Date()
  switch (range) {
    case '1h':
      return new Date(now.getTime() - 60 * 60 * 1000).toISOString()
    case '6h':
      return new Date(now.getTime() - 6 * 60 * 60 * 1000).toISOString()
    case '24h':
      return new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString()
    default:
      return null
  }
}

// Event handlers
const onSessionChange = () => {
  refreshTranscript()
}

const onGraphNodeSelected = (node: any) => {
  // Use navigation system to switch to transcript with context
  navigation.navigateFromGraphNode(node, NavigationIntent.INVESTIGATE)
  
  // Update coordination service filters
  coordinationService.setFilters({
    agentIds: [node.id],
    sessionIds: node.metadata.session_id ? [node.metadata.session_id] : []
  })
}

const onGraphNodeDoubleClick = (node: any) => {
  // Navigate to analysis for detailed debugging
  navigation.navigateFromGraphNode(node, NavigationIntent.DEBUG)
  selectedEventForAnalysis.value = node
}

const selectTranscriptEvent = (event: any) => {
  selectedTranscriptEvent.value = event
}

const jumpToAgentInGraph = (agentId: string) => {
  // Use navigation system to jump to graph with agent context
  navigation.navigateToTab('graph', {
    correlationData: { agentId },
    intent: NavigationIntent.CORRELATE,
    preserveFilters: false
  })
  
  // Focus on specific agent in graph
  if (agentGraphRef.value) {
    // Implementation to focus on agent node
    setTimeout(() => {
      // Focus graph on the specific agent node
      const graphComponent = agentGraphRef.value
      if (graphComponent && graphComponent.focusOnNode) {
        graphComponent.focusOnNode(agentId)
      }
    }, 100)
  }
}

const analyzeEventPattern = (event: any) => {
  // Navigate to analysis with event context
  navigation.navigateFromTranscriptEvent(event, NavigationIntent.ANALYZE)
  selectedEventForAnalysis.value = event
  showEventAnalysisModal.value = true
}

const investigatePattern = (pattern: any) => {
  // Navigate with pattern investigation context
  navigation.navigateFromPattern(pattern, NavigationIntent.INVESTIGATE)
  selectedPattern.value = pattern
  showPatternModal.value = true
}

const toggleTranscriptAutoRefresh = () => {
  transcriptAutoRefresh.value = !transcriptAutoRefresh.value
}

const resetGraphView = () => {
  if (agentGraphRef.value) {
    // Implementation to reset graph view
  }
}

// Utility functions
const getEventTypeClass = (eventType: string) => {
  const classes = {
    'message': 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400',
    'tool_call': 'bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400',
    'error': 'bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-400',
    'status': 'bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400'
  }
  return classes[eventType] || 'bg-gray-100 dark:bg-gray-900 text-gray-600 dark:text-gray-400'
}

const getPatternSeverityClass = (severity: string) => {
  const classes = {
    'HIGH': 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/20',
    'MEDIUM': 'border-yellow-300 dark:border-yellow-700 bg-yellow-50 dark:bg-yellow-900/20',
    'LOW': 'border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/20'
  }
  return classes[severity] || 'border-gray-300 dark:border-gray-700'
}

const getSeverityBadgeClass = (severity: string) => {
  const classes = {
    'HIGH': 'bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-400',
    'MEDIUM': 'bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400',
    'LOW': 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400'
  }
  return classes[severity] || 'bg-gray-100 dark:bg-gray-900 text-gray-600 dark:text-gray-400'
}

// Auto-refresh functionality
let refreshInterval: number | null = null

const startAutoRefresh = () => {
  refreshInterval = setInterval(() => {
    if (!loading.value && transcriptAutoRefresh.value) {
      refreshTranscript()
    }
  }, 10000) // Refresh every 10 seconds
}

const stopAutoRefresh = () => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

// Watchers
watch(() => transcriptTimeRange.value, () => {
  refreshTranscript()
})

watch(() => selectedSession.value, () => {
  refreshTranscript()
})

// Lifecycle
onMounted(async () => {
  await refreshDashboard()
  startAutoRefresh()
  
  // Setup WebSocket connections
  connectionStore.connect()
})

onUnmounted(() => {
  stopAutoRefresh()
  connectionStore.disconnect()
})
</script>

<style scoped>
.coordination-dashboard {
  @apply min-h-screen bg-slate-50 dark:bg-slate-900 p-6;
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

.line-clamp-2 {
  overflow: hidden;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.transcript-event:hover {
  transform: translateX(2px);
}

.tab-button {
  @apply transition-all duration-200;
}

.tab-button:hover {
  @apply transform scale-105;
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

.transcript-event {
  animation: slideIn 0.3s ease-out;
}
</style>