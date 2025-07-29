<template>
  <div class="semantic-query-explorer">
    <!-- Header -->
    <div class="explorer-header flex items-center justify-between mb-6">
      <div>
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          Semantic Query Explorer
        </h3>
        <p class="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Ask questions about your observability data in natural language
        </p>
      </div>
      
      <div class="flex items-center space-x-2">
        <div 
          class="w-2 h-2 rounded-full"
          :class="isProcessing ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'"
        ></div>
        <span class="text-xs text-slate-500 dark:text-slate-400">
          {{ isProcessing ? 'Processing...' : 'Ready' }}
        </span>
      </div>
    </div>

    <!-- Query Input -->
    <div class="query-input-section mb-6">
      <div class="relative">
        <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <MagnifyingGlassIcon class="h-5 w-5 text-slate-400" />
        </div>
        
        <input
          ref="queryInput"
          v-model="currentQuery"
          type="text"
          placeholder="Ask anything... e.g., 'Show me slow responses from the last hour' or 'Which agents had errors yesterday?'"
          class="w-full pl-10 pr-20 py-3 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg text-sm placeholder-slate-500 dark:placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          @keydown="handleKeydown"
          @input="handleQueryInput"
        />
        
        <div class="absolute inset-y-0 right-0 flex items-center space-x-1 pr-2">
          <!-- Query suggestions toggle -->
          <button
            v-if="querySuggestions.length > 0 && !showSuggestions"
            @click="showSuggestions = true"
            class="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
            title="Show suggestions"
          >
            <LightBulbIcon class="h-4 w-4" />
          </button>
          
          <!-- Submit button -->
          <button
            @click="executeQuery"
            :disabled="!currentQuery.trim() || isProcessing"
            class="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white text-xs rounded-md transition-colors"
          >
            {{ isProcessing ? 'Searching...' : 'Search' }}
          </button>
        </div>
      </div>
      
      <!-- Query Suggestions Dropdown -->
      <div 
        v-if="showSuggestions && querySuggestions.length > 0"
        class="absolute z-10 mt-1 w-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg max-h-60 overflow-y-auto"
      >
        <div class="p-2">
          <div class="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">
            Suggested Queries
          </div>
          <div
            v-for="(suggestion, index) in querySuggestions"
            :key="index"
            @click="selectSuggestion(suggestion)"
            class="px-2 py-1 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 rounded cursor-pointer"
          >
            {{ suggestion }}
          </div>
        </div>
      </div>
    </div>

    <!-- Query Configuration -->
    <div class="query-config-section mb-6">
      <div class="flex items-center space-x-4">
        <div class="flex items-center space-x-2">
          <label class="text-xs text-slate-600 dark:text-slate-400">Time Window:</label>
          <select
            v-model="queryConfig.context_window_hours"
            class="text-xs bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded px-2 py-1"
          >
            <option :value="1">1 hour</option>
            <option :value="6">6 hours</option>
            <option :value="24">24 hours</option>
            <option :value="168">1 week</option>
          </select>
        </div>
        
        <div class="flex items-center space-x-2">
          <label class="text-xs text-slate-600 dark:text-slate-400">Max Results:</label>
          <select
            v-model="queryConfig.max_results"
            class="text-xs bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded px-2 py-1"
          >
            <option :value="10">10</option>
            <option :value="25">25</option>
            <option :value="50">50</option>
            <option :value="100">100</option>
          </select>
        </div>
        
        <div class="flex items-center space-x-2">
          <label class="text-xs text-slate-600 dark:text-slate-400">Similarity:</label>
          <input
            v-model.number="queryConfig.similarity_threshold"
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            class="w-20"
          />
          <span class="text-xs text-slate-500 dark:text-slate-400 min-w-8">
            {{ queryConfig.similarity_threshold.toFixed(1) }}
          </span>
        </div>
        
        <div class="flex items-center space-x-3">
          <label class="flex items-center space-x-1">
            <input
              v-model="queryConfig.include_context"
              type="checkbox"
              class="w-3 h-3 text-blue-600 rounded focus:ring-blue-500"
            />
            <span class="text-xs text-slate-600 dark:text-slate-400">Context</span>
          </label>
          
          <label class="flex items-center space-x-1">
            <input
              v-model="queryConfig.include_performance"
              type="checkbox"
              class="w-3 h-3 text-blue-600 rounded focus:ring-blue-500"
            />
            <span class="text-xs text-slate-600 dark:text-slate-400">Performance</span>
          </label>
        </div>
      </div>
    </div>

    <!-- Search Results -->
    <div class="search-results-section">
      <!-- Loading State -->
      <div 
        v-if="isProcessing"
        class="flex items-center justify-center py-12"
      >
        <div class="flex items-center space-x-3 text-slate-600 dark:text-slate-400">
          <div class="animate-spin w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full"></div>
          <span>Processing semantic query...</span>
        </div>
      </div>
      
      <!-- Results -->
      <div v-else-if="searchResults.length > 0">
        <!-- Results Header -->
        <div class="flex items-center justify-between mb-4">
          <div class="text-sm text-slate-600 dark:text-slate-400">
            Found {{ searchResults.length }} result{{ searchResults.length !== 1 ? 's' : '' }}
            {{ lastQuery ? `for "${lastQuery}"` : '' }}
          </div>
          
          <div class="flex items-center space-x-2">
            <!-- Sort options -->
            <select
              v-model="sortBy"
              @change="sortResults"
              class="text-xs bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded px-2 py-1"
            >
              <option value="relevance">Relevance</option>
              <option value="timestamp">Time</option>
              <option value="event_type">Type</option>
            </select>
            
            <!-- Export button -->
            <button
              @click="exportResults"
              class="text-xs px-2 py-1 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
            >
              <ArrowDownTrayIcon class="w-3 h-3 mr-1 inline" />
              Export
            </button>
          </div>
        </div>
        
        <!-- Results List -->
        <div class="space-y-3">
          <div
            v-for="(result, index) in sortedResults"
            :key="result.id"
            class="result-card bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
            @click="selectResult(result)"
            :class="selectedResult?.id === result.id ? 'ring-2 ring-blue-500' : ''"
          >
            <div class="flex items-start justify-between">
              <div class="flex-1">
                <!-- Result Header -->
                <div class="flex items-center space-x-2 mb-2">
                  <span 
                    class="px-2 py-1 text-xs rounded-full"
                    :class="getEventTypeClass(result.event_type)"
                  >
                    {{ result.event_type }}
                  </span>
                  
                  <div class="flex items-center space-x-1">
                    <div 
                      class="w-2 h-2 rounded-full"
                      :style="{ backgroundColor: getRelevanceColor(result.relevance_score) }"
                    ></div>
                    <span class="text-xs text-slate-500 dark:text-slate-400">
                      {{ Math.round(result.relevance_score * 100) }}% match
                    </span>
                  </div>
                  
                  <span class="text-xs text-slate-500 dark:text-slate-400">
                    {{ formatTime(result.timestamp) }}
                  </span>
                </div>
                
                <!-- Content Summary -->
                <div class="text-sm text-slate-700 dark:text-slate-300 mb-2">
                  <div class="font-medium mb-1">{{ result.content_summary }}</div>
                </div>
                
                <!-- Metadata -->
                <div class="flex items-center space-x-4 text-xs text-slate-500 dark:text-slate-400">
                  <div v-if="result.agent_id" class="flex items-center space-x-1">
                    <CpuChipIcon class="w-3 h-3" />
                    <span>{{ result.agent_id.substring(0, 8) }}...</span>
                  </div>
                  
                  <div v-if="result.session_id" class="flex items-center space-x-1">
                    <RectangleGroupIcon class="w-3 h-3" />
                    <span>{{ result.session_id.substring(0, 8) }}...</span>
                  </div>
                  
                  <div v-if="result.performance_metrics?.execution_time_ms" class="flex items-center space-x-1">
                    <ClockIcon class="w-3 h-3" />
                    <span>{{ result.performance_metrics.execution_time_ms }}ms</span>
                  </div>
                </div>
                
                <!-- Semantic Concepts -->
                <div v-if="result.semantic_concepts.length > 0" class="mt-2">
                  <div class="flex flex-wrap gap-1">
                    <span
                      v-for="concept in result.semantic_concepts.slice(0, 5)"
                      :key="concept"
                      class="px-2 py-0.5 text-xs bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-full"
                    >
                      {{ concept }}
                    </span>
                    <span
                      v-if="result.semantic_concepts.length > 5"
                      class="px-2 py-0.5 text-xs bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 rounded-full"
                    >
                      +{{ result.semantic_concepts.length - 5 }} more
                    </span>
                  </div>
                </div>
              </div>
              
              <!-- Actions -->
              <div class="flex items-center space-x-1 ml-4">
                <button
                  @click.stop="viewResultDetails(result)"
                  class="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                  title="View details"
                >
                  <EyeIcon class="w-4 h-4" />
                </button>
                
                <button
                  @click.stop="addToWorkspace(result)"
                  class="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                  title="Add to workspace"
                >
                  <PlusIcon class="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Load More -->
        <div 
          v-if="hasMoreResults"
          class="text-center mt-6"
        >
          <button
            @click="loadMoreResults"
            class="px-4 py-2 text-sm bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          >
            Load More Results
          </button>
        </div>
      </div>
      
      <!-- Empty State -->
      <div 
        v-else-if="!isProcessing && hasSearched"
        class="text-center py-12"
      >
        <MagnifyingGlassIcon class="w-12 h-12 mx-auto text-slate-400 mb-4" />
        <h4 class="text-lg font-medium text-slate-900 dark:text-white mb-2">
          No results found
        </h4>
        <p class="text-slate-600 dark:text-slate-400 mb-4">
          Try adjusting your query or expanding the time window
        </p>
        <div class="flex justify-center space-x-2">
          <button
            @click="showQueryHelp = true"
            class="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            Query Examples
          </button>
          <span class="text-slate-400">•</span>
          <button
            @click="resetQuery"
            class="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            Clear Query
          </button>
        </div>
      </div>
      
      <!-- Welcome State -->
      <div 
        v-else-if="!hasSearched"
        class="text-center py-12"
      >
        <SparklesIcon class="w-12 h-12 mx-auto text-blue-500 mb-4" />
        <h4 class="text-lg font-medium text-slate-900 dark:text-white mb-2">
          Ask Questions About Your Data
        </h4>
        <p class="text-slate-600 dark:text-slate-400 mb-6">
          Use natural language to explore your observability data
        </p>
        
        <!-- Example Queries -->
        <div class="max-w-2xl mx-auto">
          <div class="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
            Try these example queries:
          </div>
          <div class="space-y-2">
            <button
              v-for="example in exampleQueries"
              :key="example"
              @click="selectSuggestion(example)"
              class="block w-full text-left px-4 py-2 text-sm bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
            >
              "{{ example }}"
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Result Details Modal -->
    <ResultDetailsModal
      v-if="showResultDetails"
      :result="selectedResult"
      @close="showResultDetails = false"
      @navigate-to-context="handleNavigateToContext"
    />
    
    <!-- Query Help Modal -->
    <QueryHelpModal
      v-if="showQueryHelp"
      @close="showQueryHelp = false"
      @select-example="selectSuggestion"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue'
import { formatDistanceToNow } from 'date-fns'

// Icons
import {
  MagnifyingGlassIcon,
  LightBulbIcon,
  ArrowDownTrayIcon,
  EyeIcon,
  PlusIcon,
  CpuChipIcon,
  RectangleGroupIcon,
  ClockIcon,
  SparklesIcon
} from '@heroicons/vue/24/outline'

// Services
import { useObservabilityEvents, DashboardEventType } from '@/services/observabilityEventService'
import type { 
  SemanticQueryRequest, 
  SemanticSearchResult,
  ObservabilityEvent 
} from '@/services/observabilityEventService'
import { DashboardComponent } from '@/types/coordination'

// Components (to be created)
// import ResultDetailsModal from './ResultDetailsModal.vue'
// import QueryHelpModal from './QueryHelpModal.vue'

interface Props {
  autoSuggest?: boolean
  persistQuery?: boolean
  maxResults?: number
}

const props = withDefaults(defineProps<Props>(), {
  autoSuggest: true,
  persistQuery: true,
  maxResults: 50
})

const emit = defineEmits<{
  resultSelected: [result: SemanticSearchResult]
  queryExecuted: [query: string, results: SemanticSearchResult[]]
  navigateToContext: [contextId: string, type: string]
}>()

// Refs
const queryInput = ref<HTMLInputElement>()

// Services
const observabilityEvents = useObservabilityEvents()

// Component state
const currentQuery = ref('')
const lastQuery = ref('')
const isProcessing = ref(false)
const hasSearched = ref(false)
const showSuggestions = ref(false)
const showResultDetails = ref(false)
const showQueryHelp = ref(false)

// Query configuration
const queryConfig = reactive<SemanticQueryRequest>({
  query: '',
  context_window_hours: 24,
  max_results: props.maxResults,
  similarity_threshold: 0.7,
  include_context: true,
  include_performance: true
})

// Results state
const searchResults = ref<SemanticSearchResult[]>([])
const selectedResult = ref<SemanticSearchResult | null>(null)
const sortBy = ref('relevance')
const hasMoreResults = ref(false)

// Query suggestions
const querySuggestions = ref<string[]>([])
const recentQueries = ref<string[]>([])

// Example queries for onboarding
const exampleQueries = [
  "Show me slow responses from the last hour",
  "Which agents had errors yesterday?",
  "Find all context sharing events this week",
  "What semantic concepts were used most frequently?",
  "Show me high latency events for agent-123",
  "Find failed tool executions in the last 6 hours",
  "Which sessions had the most activity today?",
  "Show me performance metrics for semantic search"
]

// Computed properties
const sortedResults = computed(() => {
  const results = [...searchResults.value]
  
  switch (sortBy.value) {
    case 'timestamp':
      return results.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
    case 'event_type':
      return results.sort((a, b) => a.event_type.localeCompare(b.event_type))
    case 'relevance':
    default:
      return results.sort((a, b) => b.relevance_score - a.relevance_score)
  }
})

/**
 * Initialize component
 */
onMounted(() => {
  loadRecentQueries()
  generateQuerySuggestions()
  
  // Focus query input
  if (queryInput.value) {
    queryInput.value.focus()
  }
})

/**
 * Handle query input events
 */
function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    executeQuery()
  } else if (event.key === 'Escape') {
    showSuggestions.value = false
    selectedResult.value = null
  }
}

function handleQueryInput() {
  if (props.autoSuggest && currentQuery.value.length > 2) {
    generateQuerySuggestions()
    showSuggestions.value = true
  } else {
    showSuggestions.value = false
  }
}

/**
 * Execute semantic query
 */
async function executeQuery() {
  if (!currentQuery.value.trim() || isProcessing.value) return

  isProcessing.value = true
  hasSearched.value = true
  lastQuery.value = currentQuery.value
  showSuggestions.value = false

  try {
    // Update query config
    queryConfig.query = currentQuery.value.trim()
    
    // Execute semantic search
    const results = await observabilityEvents.performSemanticSearch(queryConfig)
    
    searchResults.value = results
    selectedResult.value = null
    
    // Save to recent queries
    addToRecentQueries(currentQuery.value)
    
    // Check if there are more results available
    hasMoreResults.value = results.length === queryConfig.max_results
    
    emit('queryExecuted', currentQuery.value, results)
    
    console.log(`✅ Semantic query executed: "${currentQuery.value}" - ${results.length} results`)
    
  } catch (error) {
    console.error('Semantic query failed:', error)
    searchResults.value = []
    // Show error notification
  } finally {
    isProcessing.value = false
  }
}

/**
 * Load more results
 */
async function loadMoreResults() {
  if (isProcessing.value || !hasMoreResults.value) return

  try {
    const currentCount = searchResults.value.length
    const moreResultsConfig = {
      ...queryConfig,
      max_results: currentCount + 25
    }

    const results = await observabilityEvents.performSemanticSearch(moreResultsConfig)
    
    // Add new results
    const newResults = results.slice(currentCount)
    searchResults.value.push(...newResults)
    
    hasMoreResults.value = results.length === moreResultsConfig.max_results
    
  } catch (error) {
    console.error('Failed to load more results:', error)
  }
}

/**
 * Query suggestion management
 */
function generateQuerySuggestions() {
  const input = currentQuery.value.toLowerCase()
  
  // Smart suggestions based on input
  const suggestions: string[] = []
  
  // Intent-based suggestions
  if (input.includes('slow') || input.includes('latency')) {
    suggestions.push(
      'Show me slow responses from the last hour',
      'Find high latency events for semantic search',
      'Which agents have performance issues?'
    )
  } else if (input.includes('error') || input.includes('fail')) {
    suggestions.push(
      'Show me all errors from yesterday',
      'Find failed tool executions in the last 6 hours',
      'Which sessions had the most errors?'
    )
  } else if (input.includes('agent')) {
    suggestions.push(
      'Which agents were most active today?',
      'Show me agent performance metrics',
      'Find communication between specific agents'
    )
  } else if (input.includes('context') || input.includes('semantic')) {
    suggestions.push(
      'Show me context sharing events',
      'What semantic concepts were used most?',
      'Find semantic intelligence updates'
    )
  }
  
  // Add recent queries
  suggestions.push(...recentQueries.value.filter(q => 
    q.toLowerCase().includes(input) && q !== input
  ))
  
  // Add example queries that match
  suggestions.push(...exampleQueries.filter(q => 
    q.toLowerCase().includes(input) && !suggestions.includes(q)
  ))
  
  querySuggestions.value = [...new Set(suggestions)].slice(0, 8)
}

function selectSuggestion(suggestion: string) {
  currentQuery.value = suggestion
  showSuggestions.value = false
  
  // Auto-execute if it's a complete query
  if (suggestion.includes('?') || suggestion.includes('show me') || suggestion.includes('find')) {
    executeQuery()
  }
}

/**
 * Recent queries management
 */
function loadRecentQueries() {
  if (props.persistQuery) {
    const stored = localStorage.getItem('semantic-query-recent')
    if (stored) {
      try {
        recentQueries.value = JSON.parse(stored)
      } catch (error) {
        console.error('Failed to load recent queries:', error)
      }
    }
  }
}

function addToRecentQueries(query: string) {
  if (!props.persistQuery) return
  
  const trimmed = query.trim()
  if (!trimmed) return
  
  // Remove if already exists
  const index = recentQueries.value.indexOf(trimmed)
  if (index > -1) {
    recentQueries.value.splice(index, 1)
  }
  
  // Add to beginning
  recentQueries.value.unshift(trimmed)
  
  // Keep only last 10
  recentQueries.value = recentQueries.value.slice(0, 10)
  
  // Save to localStorage
  localStorage.setItem('semantic-query-recent', JSON.stringify(recentQueries.value))
}

/**
 * Result management
 */
function selectResult(result: SemanticSearchResult) {
  selectedResult.value = selectedResult.value?.id === result.id ? null : result
  emit('resultSelected', result)
}

function viewResultDetails(result: SemanticSearchResult) {
  selectedResult.value = result
  showResultDetails.value = true
}

function addToWorkspace(result: SemanticSearchResult) {
  // Implementation would depend on workspace functionality
  console.log('Adding to workspace:', result)
}

function sortResults() {
  // Results are computed reactively
}

function exportResults() {
  try {
    const exportData = {
      query: lastQuery.value,
      timestamp: new Date().toISOString(),
      results: searchResults.value,
      config: queryConfig
    }
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    })
    
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `semantic-query-results-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
  } catch (error) {
    console.error('Failed to export results:', error)
  }
}

/**
 * Navigation handlers
 */
function handleNavigateToContext(contextId: string, type: string) {
  emit('navigateToContext', contextId, type)
}

/**
 * Utility functions
 */
function resetQuery() {
  currentQuery.value = ''
  searchResults.value = []
  selectedResult.value = null
  hasSearched.value = false
  lastQuery.value = ''
  showSuggestions.value = false
}

function formatTime(timestamp: string): string {
  return formatDistanceToNow(new Date(timestamp), { addSuffix: true })
}

function getEventTypeClass(eventType: string): string {
  const classes: { [key: string]: string } = {
    'hook_event': 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300',
    'workflow_update': 'bg-purple-100 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300',
    'semantic_intelligence': 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300',
    'performance_metric': 'bg-orange-100 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300',
    'agent_status': 'bg-indigo-100 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-300',
    'system_alert': 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300'
  }
  return classes[eventType] || 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
}

function getRelevanceColor(score: number): string {
  if (score >= 0.8) return '#10B981' // Green
  if (score >= 0.6) return '#F59E0B' // Amber
  return '#EF4444' // Red
}

// Watch for external query changes
watch(() => props.maxResults, (newMax) => {
  queryConfig.max_results = newMax
})

// Click outside to close suggestions
onMounted(() => {
  document.addEventListener('click', (event) => {
    if (!event.target || !(event.target as Element).closest('.semantic-query-explorer')) {
      showSuggestions.value = false
    }
  })
})
</script>

<style scoped>
.semantic-query-explorer {
  @apply w-full;
}

.result-card {
  transition: all 0.2s ease;
}

.result-card:hover {
  transform: translateY(-1px);
}

/* Query input enhancements */
.query-input-section input:focus {
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* Suggestions dropdown animation */
.suggestions-dropdown {
  animation: slideDown 0.2s ease-out;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Loading animation */
.loading-dots::after {
  content: '';
  animation: dots 1.5s steps(5, end) infinite;
}

@keyframes dots {
  0%, 20% {
    color: rgba(0,0,0,0);
    text-shadow:
      .25em 0 0 rgba(0,0,0,0),
      .5em 0 0 rgba(0,0,0,0);
  }
  40% {
    color: currentColor;
    text-shadow:
      .25em 0 0 rgba(0,0,0,0),
      .5em 0 0 rgba(0,0,0,0);
  }
  60% {
    text-shadow:
      .25em 0 0 currentColor,
      .5em 0 0 rgba(0,0,0,0);
  }
  80%, 100% {
    text-shadow:
      .25em 0 0 currentColor,
      .5em 0 0 currentColor;
  }
}

/* Semantic concepts tags */
.semantic-concept-tag {
  animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: scale(0.8);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

/* Dark mode specific adjustments */
.dark .query-input-section input:focus {
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}
</style>