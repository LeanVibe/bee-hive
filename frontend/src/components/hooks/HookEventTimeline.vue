<template>
  <div class="hook-event-timeline bg-white dark:bg-gray-800 rounded-lg shadow-sm">
    <!-- Header -->
    <div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
      <div>
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
          Hook Event Timeline
        </h3>
        <p class="text-sm text-gray-500 dark:text-gray-400">
          Real-time agent lifecycle events
        </p>
      </div>
      
      <div class="flex items-center space-x-3">
        <!-- Connection status -->
        <div class="flex items-center space-x-2">
          <div 
            :class="[
              'w-2 h-2 rounded-full',
              wsConnected ? 'bg-green-500' : 'bg-red-500'
            ]"
          ></div>
          <span class="text-sm text-gray-600 dark:text-gray-300">
            {{ wsConnected ? 'Live' : 'Disconnected' }}
          </span>
        </div>

        <!-- Auto-scroll toggle -->
        <button
          @click="autoScroll = !autoScroll"
          :class="[
            'px-3 py-1 text-xs rounded-full transition-colors',
            autoScroll
              ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
              : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
          ]"
        >
          Auto-scroll {{ autoScroll ? 'ON' : 'OFF' }}
        </button>

        <!-- Clear events -->
        <button
          @click="clearEvents"
          class="px-3 py-1 text-xs bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200 rounded-full hover:bg-red-200 dark:hover:bg-red-800 transition-colors"
        >
          Clear
        </button>
      </div>
    </div>

    <!-- Timeline container -->
    <div 
      ref="timelineContainer"
      class="timeline-container"
      :style="{ height: `${height}px` }"
    >
      <div class="timeline-content">
        <!-- Timeline events -->
        <div
          v-for="(event, index) in displayedEvents"
          :key="event.event_id || `${event.agent_id}-${index}`"
          class="timeline-event group"
          :class="getEventClasses(event)"
          @click="$emit('eventClick', event)"
        >
          <!-- Timeline marker -->
          <div class="timeline-marker">
            <div 
              :class="[
                'timeline-dot',
                getEventTypeColor(event.hook_type)
              ]"
            ></div>
            <div 
              v-if="index < displayedEvents.length - 1"
              class="timeline-line"
            ></div>
          </div>

          <!-- Event content -->
          <div class="timeline-content-wrapper">
            <div class="event-header">
              <div class="event-title">
                <span class="event-type-badge" :class="getEventTypeBadgeClass(event.hook_type)">
                  {{ formatEventType(event.hook_type) }}
                </span>
                <span class="agent-id">
                  Agent {{ event.agent_id.substring(0, 8) }}
                </span>
                <span v-if="event.session_id" class="session-id">
                  Session {{ event.session_id.substring(0, 8) }}
                </span>
              </div>
              
              <div class="event-meta">
                <span class="timestamp">
                  {{ formatTimestamp(event.timestamp) }}
                </span>
                <span v-if="event.priority <= 3" class="priority-indicator high-priority">
                  High Priority
                </span>
                <span v-if="event.metadata?.security_blocked" class="security-blocked">
                  Blocked
                </span>
              </div>
            </div>

            <!-- Event details -->
            <div class="event-details">
              <div v-if="event.hook_type === HookType.PRE_TOOL_USE" class="tool-usage">
                <strong>Tool:</strong> {{ event.payload.tool_name }}
                <div v-if="Object.keys(event.payload.parameters || {}).length > 0" class="parameters">
                  <span class="text-xs text-gray-500">{{ Object.keys(event.payload.parameters).length }} parameters</span>
                </div>
              </div>

              <div v-else-if="event.hook_type === HookType.POST_TOOL_USE" class="tool-result">
                <div class="tool-result-header">
                  <strong>Tool:</strong> {{ event.payload.tool_name }}
                  <span 
                    :class="[
                      'result-status',
                      event.payload.success ? 'success' : 'error'
                    ]"
                  >
                    {{ event.payload.success ? 'Success' : 'Failed' }}
                  </span>
                  <span v-if="event.payload.execution_time_ms" class="execution-time">
                    {{ event.payload.execution_time_ms }}ms
                  </span>
                </div>
                <div v-if="event.payload.error" class="error-message">
                  {{ event.payload.error }}
                </div>
              </div>

              <div v-else-if="event.hook_type === HookType.NOTIFICATION" class="notification">
                <span :class="getNotificationLevelClass(event.payload.level)">
                  {{ event.payload.level?.toUpperCase() }}
                </span>
                {{ event.payload.message }}
              </div>

              <div v-else-if="event.hook_type === HookType.STOP" class="stop-event">
                <strong>Reason:</strong> {{ event.payload.reason }}
                <div v-if="event.payload.details" class="stop-details">
                  {{ Object.keys(event.payload.details).length }} details
                </div>
              </div>

              <div v-else-if="event.hook_type === HookType.ERROR" class="error-event">
                <div class="error-header">
                  <strong>Error:</strong> {{ event.payload.error_type || 'Unknown Error' }}
                </div>
                <div class="error-message">
                  {{ event.payload.error_message }}
                </div>
              </div>

              <div v-else class="generic-event">
                <div class="payload-summary">
                  {{ getPayloadSummary(event.payload) }}
                </div>
              </div>
            </div>

            <!-- Security indicators -->
            <div v-if="event.metadata?.security_decision" class="security-indicator">
              <span 
                :class="[
                  'security-badge',
                  getSecurityDecisionClass(event.metadata.security_decision)
                ]"
              >
                {{ event.metadata.security_decision }}
              </span>
              <span v-if="event.metadata.blocked_reason" class="blocked-reason">
                {{ event.metadata.blocked_reason }}
              </span>
            </div>
          </div>
        </div>

        <!-- Empty state -->
        <div 
          v-if="displayedEvents.length === 0"
          class="empty-state"
        >
          <div class="empty-icon">
            <ClockIcon class="w-12 h-12 text-gray-400" />
          </div>
          <h4 class="text-lg font-medium text-gray-900 dark:text-white">
            No events yet
          </h4>
          <p class="text-gray-500 dark:text-gray-400">
            Hook events will appear here in real-time
          </p>
        </div>

        <!-- Loading indicator -->
        <div 
          v-if="loading" 
          class="loading-indicator"
        >
          <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span class="text-sm text-gray-600 dark:text-gray-300">Loading events...</span>
        </div>
      </div>
    </div>

    <!-- Footer with event count and stats -->
    <div class="timeline-footer">
      <div class="event-stats">
        <span class="stat">
          <strong>{{ displayedEvents.length }}</strong> events
        </span>
        <span class="stat">
          <strong>{{ uniqueAgentsCount }}</strong> agents
        </span>
        <span class="stat">
          <strong>{{ uniqueSessionsCount }}</strong> sessions
        </span>
        <span v-if="errorCount > 0" class="stat error">
          <strong>{{ errorCount }}</strong> errors
        </span>
        <span v-if="blockedCount > 0" class="stat blocked">
          <strong>{{ blockedCount }}</strong> blocked
        </span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useEventsStore } from '@/stores/events'
import { 
  HookType, 
  SecurityRisk, 
  ControlDecision,
  type HookEvent as TypedHookEvent,
  type HookEventTimelineProps
} from '@/types/hooks'
import { ClockIcon } from '@heroicons/vue/24/outline'
import { format } from 'date-fns'

// Props
interface Props extends HookEventTimelineProps {}

const props = withDefaults(defineProps<Props>(), {
  height: 600,
  maxEvents: 200,
  autoScroll: true,
  showFilters: true,
  initialFilters: () => ({})
})

// Emits
const emit = defineEmits<{
  eventClick: [event: TypedHookEvent]
}>()

// Store
const eventsStore = useEventsStore()

// Local state
const timelineContainer = ref<HTMLElement>()
const autoScroll = ref(props.autoScroll)
const loading = ref(false)

// Computed
const { 
  hookEvents, 
  filteredHookEvents, 
  wsConnected,
  hookFilters
} = eventsStore

const displayedEvents = computed(() => {
  let events = props.initialFilters && Object.keys(props.initialFilters).length > 0 
    ? hookEvents.filter(event => {
        // Apply initial filters if provided
        if (props.initialFilters!.agent_ids?.length && 
            !props.initialFilters!.agent_ids.includes(event.agent_id)) {
          return false
        }
        if (props.initialFilters!.hook_types?.length && 
            !props.initialFilters!.hook_types.includes(event.hook_type)) {
          return false
        }
        if (props.initialFilters!.session_ids?.length && event.session_id &&
            !props.initialFilters!.session_ids.includes(event.session_id)) {
          return false
        }
        return true
      })
    : filteredHookEvents

  return events.slice(0, props.maxEvents)
})

const uniqueAgentsCount = computed(() => {
  const agents = new Set(displayedEvents.value.map(e => e.agent_id))
  return agents.size
})

const uniqueSessionsCount = computed(() => {
  const sessions = new Set(
    displayedEvents.value
      .map(e => e.session_id)
      .filter(Boolean)
  )
  return sessions.size
})

const errorCount = computed(() => {
  return displayedEvents.value.filter(e => 
    e.hook_type === HookType.ERROR || 
    e.payload?.error || 
    e.metadata?.error
  ).length
})

const blockedCount = computed(() => {
  return displayedEvents.value.filter(e => 
    e.metadata?.security_blocked
  ).length
})

// Methods
const formatTimestamp = (timestamp: string) => {
  return format(new Date(timestamp), 'HH:mm:ss.SSS')
}

const formatEventType = (hookType: HookType) => {
  const typeMap = {
    [HookType.PRE_TOOL_USE]: 'Pre Tool',
    [HookType.POST_TOOL_USE]: 'Post Tool',
    [HookType.STOP]: 'Stop',
    [HookType.NOTIFICATION]: 'Notification',
    [HookType.AGENT_START]: 'Agent Start',
    [HookType.AGENT_STOP]: 'Agent Stop',
    [HookType.ERROR]: 'Error',
    [HookType.SUBAGENT_STOP]: 'Subagent Stop'
  }
  return typeMap[hookType] || hookType
}

const getEventClasses = (event: TypedHookEvent) => {
  const classes = ['timeline-event']
  
  if (event.hook_type === HookType.ERROR) {
    classes.push('error-event')
  }
  
  if (event.metadata?.security_blocked) {
    classes.push('blocked-event')
  }
  
  if (event.priority <= 3) {
    classes.push('high-priority-event')
  }
  
  return classes.join(' ')
}

const getEventTypeColor = (hookType: HookType) => {
  const colorMap = {
    [HookType.PRE_TOOL_USE]: 'bg-blue-500',
    [HookType.POST_TOOL_USE]: 'bg-green-500',
    [HookType.STOP]: 'bg-red-500',
    [HookType.NOTIFICATION]: 'bg-yellow-500',
    [HookType.AGENT_START]: 'bg-purple-500',
    [HookType.AGENT_STOP]: 'bg-gray-500',
    [HookType.ERROR]: 'bg-red-600',
    [HookType.SUBAGENT_STOP]: 'bg-orange-500'
  }
  return colorMap[hookType] || 'bg-gray-400'
}

const getEventTypeBadgeClass = (hookType: HookType) => {
  const classMap = {
    [HookType.PRE_TOOL_USE]: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    [HookType.POST_TOOL_USE]: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    [HookType.STOP]: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    [HookType.NOTIFICATION]: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    [HookType.AGENT_START]: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    [HookType.AGENT_STOP]: 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200',
    [HookType.ERROR]: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    [HookType.SUBAGENT_STOP]: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
  }
  return classMap[hookType] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const getNotificationLevelClass = (level: string) => {
  const classMap = {
    info: 'text-blue-600 dark:text-blue-400',
    warning: 'text-yellow-600 dark:text-yellow-400',
    error: 'text-red-600 dark:text-red-400',
    critical: 'text-red-800 dark:text-red-300'
  }
  return classMap[level as keyof typeof classMap] || 'text-gray-600 dark:text-gray-400'
}

const getSecurityDecisionClass = (decision: ControlDecision) => {
  const classMap = {
    [ControlDecision.ALLOW]: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    [ControlDecision.DENY]: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    [ControlDecision.REQUIRE_APPROVAL]: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    [ControlDecision.ESCALATE]: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
  }
  return classMap[decision] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const getPayloadSummary = (payload: Record<string, any>) => {
  const keys = Object.keys(payload)
  return keys.length > 0 
    ? `${keys.length} ${keys.length === 1 ? 'property' : 'properties'}`
    : 'No payload data'
}

const clearEvents = () => {
  eventsStore.clearHookEvents()
}

const scrollToBottom = async () => {
  if (!autoScroll.value || !timelineContainer.value) return
  
  await nextTick()
  timelineContainer.value.scrollTop = timelineContainer.value.scrollHeight
}

// Watch for new events and auto-scroll
watch(
  () => displayedEvents.value.length,
  () => {
    if (autoScroll.value) {
      scrollToBottom()
    }
  }
)

// Lifecycle
onMounted(() => {
  // Connect to WebSocket for real-time events
  eventsStore.connectWebSocket()
  
  // Scroll to bottom initially
  if (autoScroll.value) {
    scrollToBottom()
  }
})

onUnmounted(() => {
  // Cleanup handled by store
})
</script>

<style scoped>
.hook-event-timeline {
  display: flex;
  flex-direction: column;
  min-height: 400px;
}

.timeline-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background: linear-gradient(to bottom, 
    transparent 0%, 
    rgba(0, 0, 0, 0.02) 50%, 
    transparent 100%
  );
}

.dark .timeline-container {
  background: linear-gradient(to bottom, 
    transparent 0%, 
    rgba(255, 255, 255, 0.02) 50%, 
    transparent 100%
  );
}

.timeline-content {
  position: relative;
}

.timeline-event {
  display: flex;
  margin-bottom: 1.5rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.timeline-event:hover {
  transform: translateX(4px);
}

.timeline-event.error-event {
  border-left: 3px solid #ef4444;
  padding-left: 0.75rem;
  margin-left: -0.75rem;
}

.timeline-event.blocked-event {
  border-left: 3px solid #f59e0b;
  padding-left: 0.75rem;
  margin-left: -0.75rem;
}

.timeline-event.high-priority-event {
  background: rgba(59, 130, 246, 0.05);
  border-radius: 0.5rem;
  padding: 0.5rem;
  margin: -0.5rem -0.5rem 1rem -0.5rem;
}

.timeline-marker {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-right: 1rem;
  flex-shrink: 0;
}

.timeline-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 2px solid white;
  box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.1);
}

.dark .timeline-dot {
  border-color: #374151;
}

.timeline-line {
  width: 2px;
  height: 2rem;
  background: linear-gradient(to bottom, 
    rgba(156, 163, 175, 0.5) 0%, 
    transparent 100%
  );
  margin-top: 0.5rem;
}

.timeline-content-wrapper {
  flex: 1;
  min-width: 0;
}

.event-header {
  display: flex;
  justify-content: between;
  align-items: flex-start;
  margin-bottom: 0.5rem;
  gap: 1rem;
}

.event-title {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
}

.event-type-badge {
  padding: 0.125rem 0.5rem;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.025em;
}

.agent-id, .session-id {
  font-size: 0.75rem;
  color: #6b7280;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
}

.dark .agent-id, .dark .session-id {
  color: #9ca3af;
}

.event-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.25rem;
  flex-shrink: 0;
}

.timestamp {
  font-size: 0.75rem;
  color: #6b7280;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
}

.dark .timestamp {
  color: #9ca3af;
}

.priority-indicator {
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  font-size: 0.625rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.high-priority {
  background: rgba(239, 68, 68, 0.1);
  color: #dc2626;
}

.dark .high-priority {
  background: rgba(239, 68, 68, 0.2);
  color: #fca5a5;
}

.security-blocked {
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  font-size: 0.625rem;
  font-weight: 600;
  text-transform: uppercase;
  background: rgba(245, 158, 11, 0.1);
  color: #d97706;
}

.dark .security-blocked {
  background: rgba(245, 158, 11, 0.2);
  color: #fbbf24;
}

.event-details {
  font-size: 0.875rem;
  color: #374151;
  margin-bottom: 0.5rem;
}

.dark .event-details {
  color: #d1d5db;
}

.tool-usage, .tool-result {
  line-height: 1.5;
}

.tool-result-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
}

.result-status {
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 600;
}

.result-status.success {
  background: rgba(34, 197, 94, 0.1);
  color: #16a34a;
}

.result-status.error {
  background: rgba(239, 68, 68, 0.1);
  color: #dc2626;
}

.execution-time {
  font-size: 0.75rem;
  color: #6b7280;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
}

.error-message {
  color: #dc2626;
  font-size: 0.8125rem;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  margin-top: 0.25rem;
}

.dark .error-message {
  color: #fca5a5;
}

.parameters, .stop-details {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: #6b7280;
}

.dark .parameters, .dark .stop-details {
  color: #9ca3af;
}

.security-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.security-badge {
  padding: 0.125rem 0.5rem;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.025em;
}

.blocked-reason {
  font-size: 0.75rem;
  color: #6b7280;
}

.dark .blocked-reason {
  color: #9ca3af;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem 1rem;
  text-align: center;
}

.empty-icon {
  margin-bottom: 1rem;
}

.loading-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  padding: 2rem;
}

.timeline-footer {
  border-top: 1px solid #e5e7eb;
  padding: 0.75rem 1rem;
  background: rgba(249, 250, 251, 0.5);
}

.dark .timeline-footer {
  border-top-color: #374151;
  background: rgba(17, 24, 39, 0.5);
}

.event-stats {
  display: flex;
  gap: 1.5rem;
  align-items: center;
  font-size: 0.875rem;
}

.stat {
  color: #6b7280;
}

.stat.error {
  color: #dc2626;
}

.stat.blocked {
  color: #d97706;
}

.dark .stat {
  color: #9ca3af;
}

.dark .stat.error {
  color: #fca5a5;
}

.dark .stat.blocked {
  color: #fbbf24;
}

/* Responsive design */
@media (max-width: 768px) {
  .event-header {
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .event-meta {
    align-items: flex-start;
    flex-direction: row;
    gap: 1rem;
  }
  
  .event-stats {
    flex-wrap: wrap;
    gap: 1rem;
  }
  
  .timeline-event:hover {
    transform: none;
  }
}
</style>