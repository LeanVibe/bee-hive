<template>
  <div class="context-trajectory-view">
    <!-- Header -->
    <div class="trajectory-header flex items-center justify-between mb-6">
      <div>
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          Context Trajectory View
        </h3>
        <p class="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Trace semantic context lineage and knowledge flow across agents
        </p>
      </div>
      
      <div class="flex items-center space-x-3">
        <!-- Analysis Mode Toggle -->
        <div class="flex items-center space-x-2">
          <button
            v-for="mode in analysisModes"
            :key="mode.value"
            @click="analysisMode = mode.value"
            :class="analysisMode === mode.value 
              ? 'bg-blue-600 text-white' 
              : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300'"
            class="px-3 py-1 text-xs rounded transition-colors"
          >
            <component :is="mode.icon" class="w-3 h-3 mr-1 inline" />
            {{ mode.label }}
          </button>
        </div>
        
        <!-- Refresh Button -->
        <button
          @click="refreshTrajectory"
          :disabled="loading"
          class="px-3 py-1 text-xs bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
        >
          <ArrowPathIcon class="w-3 h-3 mr-1 inline" :class="{ 'animate-spin': loading }" />
          Refresh
        </button>
      </div>
    </div>

    <!-- Search and Filters -->
    <div class="trajectory-controls mb-6">
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <!-- Context/Concept Search -->
        <div class="col-span-2">
          <div class="relative">
            <MagnifyingGlassIcon class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Search context ID, concept, or agent..."
              class="w-full pl-10 pr-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              @input="debouncedSearch"
            />
          </div>
        </div>
        
        <!-- Time Range -->
        <div>
          <select
            v-model="timeRangeHours"
            @change="updateTimeRange"
            class="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg text-sm"
          >
            <option :value="1">Last Hour</option>
            <option :value="6">Last 6 Hours</option>
            <option :value="24">Last 24 Hours</option>
            <option :value="168">Last Week</option>
          </select>
        </div>
        
        <!-- Max Depth -->
        <div>
          <select
            v-model="maxDepth"
            @change="updateDepth"
            class="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg text-sm"
          >
            <option :value="3">Depth: 3</option>
            <option :value="5">Depth: 5</option>
            <option :value="10">Depth: 10</option>
            <option :value="20">Depth: 20</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <div class="trajectory-content">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Trajectory Visualization -->
        <div class="lg:col-span-2">
          <div class="trajectory-canvas bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4">
            <!-- Canvas Header -->
            <div class="flex items-center justify-between mb-4">
              <h4 class="text-sm font-medium text-slate-900 dark:text-white">
                Context Flow Visualization
              </h4>
              
              <div class="flex items-center space-x-2">
                <!-- View Controls -->
                <button
                  @click="fitToView"
                  class="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                  title="Fit to view"
                >
                  <ArrowsPointingOutIcon class="w-4 h-4" />
                </button>
                
                <button
                  @click="toggleAnimation"
                  :class="animationEnabled ? 'text-blue-600' : 'text-slate-400'"
                  class="p-1 hover:text-blue-700"
                  title="Toggle animation"
                >
                  <PlayIcon v-if="!animationEnabled" class="w-4 h-4" />
                  <PauseIcon v-else class="w-4 h-4" />
                </button>
              </div>
            </div>
            
            <!-- SVG Canvas -->
            <div 
              ref="canvasContainer"
              class="relative w-full bg-slate-50 dark:bg-slate-900 rounded border border-slate-200 dark:border-slate-700"
              :style="{ height: `${canvasHeight}px` }"
            >
              <svg
                ref="svgCanvas"
                :width="canvasWidth"
                :height="canvasHeight"
                class="absolute inset-0 w-full h-full"
              >
                <!-- Definitions -->
                <defs>
                  <!-- Gradients for different node types -->
                  <linearGradient id="contextGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#3B82F6;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#1E40AF;stop-opacity:0.8" />
                  </linearGradient>
                  
                  <linearGradient id="eventGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#10B981;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#059669;stop-opacity:0.8" />
                  </linearGradient>
                  
                  <linearGradient id="agentGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#8B5CF6;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#7C3AED;stop-opacity:0.8" />
                  </linearGradient>
                  
                  <linearGradient id="conceptGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#F59E0B;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#D97706;stop-opacity:0.8" />
                  </linearGradient>
                  
                  <!-- Arrow markers -->
                  <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                          refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#64748B" />
                  </marker>
                  
                  <!-- Flow animation -->
                  <circle id="flowParticle" r="2" fill="#F59E0B" opacity="0.8" />
                </defs>
                
                <!-- Rendered by D3 -->
                <g class="trajectory-layer"></g>
              </svg>
              
              <!-- Loading Overlay -->
              <div 
                v-if="loading"
                class="absolute inset-0 flex items-center justify-center bg-white/50 dark:bg-slate-900/50"
              >
                <div class="flex items-center space-x-2 text-slate-600 dark:text-slate-400">
                  <div class="animate-spin w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                  <span class="text-sm">Tracing context flow...</span>
                </div>
              </div>
              
              <!-- Node Tooltip -->
              <div
                v-if="hoveredNode"
                :style="{ 
                  left: `${tooltipPosition.x}px`, 
                  top: `${tooltipPosition.y}px`,
                  transform: 'translate(-50%, -100%)'
                }"
                class="absolute pointer-events-none bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg p-3 max-w-64 z-10"
              >
                <div class="text-sm font-medium text-slate-900 dark:text-white mb-1">
                  {{ hoveredNode.label }}
                </div>
                <div class="text-xs text-slate-500 dark:text-slate-400 space-y-1">
                  <div>Type: {{ hoveredNode.type }}</div>
                  <div v-if="hoveredNode.timestamp">
                    Time: {{ formatTime(hoveredNode.timestamp) }}
                  </div>
                  <div v-if="hoveredNode.connections.length">
                    Connections: {{ hoveredNode.connections.length }}
                  </div>
                  <div v-if="hoveredNode.metadata.similarity_score">
                    Similarity: {{ (hoveredNode.metadata.similarity_score * 100).toFixed(1) }}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Trajectory Details Panel -->
        <div class="trajectory-details">
          <!-- Path Information -->
          <div class="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4 mb-4">
            <h4 class="text-sm font-medium text-slate-900 dark:text-white mb-3">
              Trajectory Paths
            </h4>
            
            <div v-if="trajectoryPaths.length > 0" class="space-y-3">
              <div
                v-for="(path, index) in trajectoryPaths.slice(0, 5)"
                :key="index"
                class="path-card p-3 bg-slate-50 dark:bg-slate-900 rounded border cursor-pointer transition-colors hover:bg-slate-100 dark:hover:bg-slate-800"
                @click="selectPath(path)"
                :class="selectedPath?.id === path.id ? 'ring-2 ring-blue-500' : ''"
              >
                <div class="flex items-center justify-between mb-2">
                  <div class="flex items-center space-x-2">
                    <div 
                      class="w-3 h-3 rounded-full"
                      :style="{ backgroundColor: getPathStrengthColor(path.path_strength) }"
                    ></div>
                    <span class="text-xs font-medium text-slate-700 dark:text-slate-300">
                      Path {{ index + 1 }}
                    </span>
                  </div>
                  <span class="text-xs text-slate-500 dark:text-slate-400">
                    {{ path.nodes.length }} nodes
                  </span>
                </div>
                
                <div class="text-xs text-slate-600 dark:text-slate-400 mb-2">
                  Strength: {{ (path.path_strength * 100).toFixed(1) }}%
                </div>
                
                <div class="flex items-center space-x-1">
                  <div
                    v-for="node in path.nodes.slice(0, 4)"
                    :key="node.id"
                    class="w-2 h-2 rounded-full"
                    :style="{ backgroundColor: getNodeTypeColor(node.type) }"
                    :title="node.label"
                  ></div>
                  <span
                    v-if="path.nodes.length > 4"
                    class="text-xs text-slate-400"
                  >
                    +{{ path.nodes.length - 4 }}
                  </span>
                </div>
              </div>
              
              <div v-if="trajectoryPaths.length > 5" class="text-center">
                <button
                  @click="showAllPaths = !showAllPaths"
                  class="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                >
                  {{ showAllPaths ? 'Show Less' : `Show ${trajectoryPaths.length - 5} More Paths` }}
                </button>
              </div>
            </div>
            
            <div v-else-if="!loading" class="text-center py-4">
              <div class="text-sm text-slate-500 dark:text-slate-400">
                No trajectory paths found
              </div>
              <div class="text-xs text-slate-400 mt-1">
                Try adjusting your search parameters
              </div>
            </div>
          </div>
          
          <!-- Selected Path Details -->
          <div 
            v-if="selectedPath"
            class="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4 mb-4"
          >
            <h4 class="text-sm font-medium text-slate-900 dark:text-white mb-3">
              Path Details
            </h4>
            
            <div class="space-y-3">
              <!-- Path Metrics -->
              <div class="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <div class="text-slate-500 dark:text-slate-400">Semantic Similarity</div>
                  <div class="font-medium text-slate-900 dark:text-white">
                    {{ (selectedPath.semantic_similarity * 100).toFixed(1) }}%
                  </div>
                </div>
                <div>
                  <div class="text-slate-500 dark:text-slate-400">Path Strength</div>
                  <div class="font-medium text-slate-900 dark:text-white">
                    {{ (selectedPath.path_strength * 100).toFixed(1) }}%
                  </div>
                </div>
              </div>
              
              <!-- Temporal Flow -->
              <div v-if="selectedPath.temporal_flow.length > 0">
                <div class="text-xs text-slate-500 dark:text-slate-400 mb-2">Temporal Flow</div>
                <div class="space-y-1">
                  <div
                    v-for="(timestamp, index) in selectedPath.temporal_flow.slice(0, 3)"
                    :key="index"
                    class="text-xs text-slate-600 dark:text-slate-400"
                  >
                    {{ formatTime(timestamp) }}
                  </div>
                  <div v-if="selectedPath.temporal_flow.length > 3" class="text-xs text-slate-400">
                    +{{ selectedPath.temporal_flow.length - 3 }} more events
                  </div>
                </div>
              </div>
              
              <!-- Path Nodes -->
              <div>
                <div class="text-xs text-slate-500 dark:text-slate-400 mb-2">Path Nodes</div>
                <div class="space-y-1">
                  <div
                    v-for="node in selectedPath.nodes"
                    :key="node.id"
                    class="flex items-center space-x-2 text-xs cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700 rounded p-1"
                    @click="highlightNode(node)"
                  >
                    <div 
                      class="w-2 h-2 rounded-full"
                      :style="{ backgroundColor: getNodeTypeColor(node.type) }"
                    ></div>
                    <span class="text-slate-700 dark:text-slate-300">{{ node.label }}</span>
                    <span class="text-slate-400 text-xs">({{ node.type }})</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Context Statistics -->
          <div class="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4">
            <h4 class="text-sm font-medium text-slate-900 dark:text-white mb-3">
              Context Statistics
            </h4>
            
            <div class="space-y-3">
              <div class="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <div class="text-slate-500 dark:text-slate-400">Total Contexts</div>
                  <div class="font-medium text-blue-600 dark:text-blue-400">
                    {{ stats.totalContexts }}
                  </div>
                </div>
                <div>
                  <div class="text-slate-500 dark:text-slate-400">Active Paths</div>
                  <div class="font-medium text-green-600 dark:text-green-400">
                    {{ stats.activePaths }}
                  </div>
                </div>
                <div>
                  <div class="text-slate-500 dark:text-slate-400">Avg Depth</div>
                  <div class="font-medium text-purple-600 dark:text-purple-400">
                    {{ stats.averageDepth.toFixed(1) }}
                  </div>
                </div>
                <div>
                  <div class="text-slate-500 dark:text-slate-400">Max Similarity</div>
                  <div class="font-medium text-orange-600 dark:text-orange-400">
                    {{ (stats.maxSimilarity * 100).toFixed(1) }}%
                  </div>
                </div>
              </div>
              
              <!-- Top Concepts -->
              <div v-if="stats.topConcepts.length > 0">
                <div class="text-xs text-slate-500 dark:text-slate-400 mb-2">Top Concepts</div>
                <div class="space-y-1">
                  <div
                    v-for="concept in stats.topConcepts.slice(0, 3)"
                    :key="concept.name"
                    class="flex items-center justify-between text-xs"
                  >
                    <span class="text-slate-700 dark:text-slate-300">{{ concept.name }}</span>
                    <span class="text-slate-500 dark:text-slate-400">{{ concept.count }}</span>
                  </div>
                </div>
              </div>
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
import { debounce } from 'lodash-es'

// Icons
import {
  MagnifyingGlassIcon,
  ArrowPathIcon,
  ArrowsPointingOutIcon,
  PlayIcon,
  PauseIcon,
  ShareIcon,
  DocumentTextIcon,
  CpuChipIcon,
  SparklesIcon
} from '@heroicons/vue/24/outline'

// Services
import { useObservabilityEvents } from '@/services/observabilityEventService'
import type { 
  ContextTrajectoryNode, 
  ContextTrajectoryPath,
  ObservabilityEvent 
} from '@/services/observabilityEventService'
import { DashboardComponent } from '@/types/coordination'

interface Props {
  contextId?: string
  agentId?: string
  sessionId?: string
  concept?: string
  autoRefresh?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  autoRefresh: true
})

const emit = defineEmits<{
  nodeSelected: [node: ContextTrajectoryNode]
  pathSelected: [path: ContextTrajectoryPath]
  conceptHighlighted: [concept: string]
}>()

// Refs
const canvasContainer = ref<HTMLDivElement>()
const svgCanvas = ref<SVGSVGElement>()

// Services
const observabilityEvents = useObservabilityEvents()

// Component state
const loading = ref(false)
const searchQuery = ref('')
const timeRangeHours = ref(24)
const maxDepth = ref(10)
const analysisMode = ref('flow')
const animationEnabled = ref(true)
const showAllPaths = ref(false)

// Canvas dimensions
const canvasWidth = ref(800)
const canvasHeight = ref(500)

// Analysis modes
const analysisModes = [
  { value: 'flow', label: 'Flow', icon: ShareIcon },
  { value: 'hierarchy', label: 'Hierarchy', icon: DocumentTextIcon },
  { value: 'semantic', label: 'Semantic', icon: SparklesIcon }
]

// Trajectory data
const trajectoryPaths = ref<ContextTrajectoryPath[]>([])
const selectedPath = ref<ContextTrajectoryPath | null>(null)
const hoveredNode = ref<ContextTrajectoryNode | null>(null)
const tooltipPosition = ref({ x: 0, y: 0 })

// D3 elements
let svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | null = null
let trajectoryGroup: d3.Selection<SVGGElement, unknown, null, undefined> | null = null
let simulation: d3.Simulation<ContextTrajectoryNode, any> | null = null

// Statistics
const stats = computed(() => {
  if (trajectoryPaths.value.length === 0) {
    return {
      totalContexts: 0,
      activePaths: 0,
      averageDepth: 0,
      maxSimilarity: 0,
      topConcepts: []
    }
  }

  const allNodes = trajectoryPaths.value.flatMap(path => path.nodes)
  const contextNodes = allNodes.filter(node => node.type === 'context')
  const conceptNodes = allNodes.filter(node => node.type === 'concept')
  
  // Count concept occurrences
  const conceptCounts: { [key: string]: number } = {}
  conceptNodes.forEach(node => {
    conceptCounts[node.label] = (conceptCounts[node.label] || 0) + 1
  })
  
  const topConcepts = Object.entries(conceptCounts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)

  return {
    totalContexts: contextNodes.length,
    activePaths: trajectoryPaths.value.length,
    averageDepth: trajectoryPaths.value.reduce((sum, path) => sum + path.nodes.length, 0) / trajectoryPaths.value.length,
    maxSimilarity: Math.max(...trajectoryPaths.value.map(path => path.semantic_similarity)),
    topConcepts
  }
})

/**
 * Initialize component
 */
onMounted(async () => {
  await nextTick()
  initializeVisualization()
  await loadTrajectoryData()
  
  // Set up canvas resize handling
  const resizeObserver = new ResizeObserver(handleResize)
  if (canvasContainer.value) {
    resizeObserver.observe(canvasContainer.value)
  }
})

/**
 * Initialize D3 visualization
 */
function initializeVisualization() {
  if (!svgCanvas.value) return

  svg = d3.select(svgCanvas.value)
  
  // Create main group
  trajectoryGroup = svg.select('.trajectory-layer')
  
  // Set up zoom behavior
  const zoom = d3.zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.1, 4])
    .on('zoom', (event) => {
      trajectoryGroup?.attr('transform', event.transform)
    })
  
  svg.call(zoom)
  
  // Initialize force simulation
  simulation = d3.forceSimulation<ContextTrajectoryNode>()
    .force('link', d3.forceLink<ContextTrajectoryNode, any>()
      .id((d: any) => d.id)
      .distance(100)
      .strength(0.1))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(canvasWidth.value / 2, canvasHeight.value / 2))
    .force('collision', d3.forceCollide().radius(20))
}

/**
 * Load trajectory data
 */
async function loadTrajectoryData() {
  loading.value = true
  
  try {
    const params = {
      context_id: props.contextId,
      concept: props.concept || (searchQuery.value.length > 2 ? searchQuery.value : undefined),
      agent_id: props.agentId,
      session_id: props.sessionId,
      time_range_hours: timeRangeHours.value,
      max_depth: maxDepth.value
    }
    
    trajectoryPaths.value = await observabilityEvents.getContextTrajectory(params)
    
    if (trajectoryPaths.value.length > 0) {
      await renderTrajectoryVisualization()
    }
    
  } catch (error) {
    console.error('Failed to load trajectory data:', error)
    trajectoryPaths.value = []
  } finally {
    loading.value = false
  }
}

/**
 * Render trajectory visualization
 */
async function renderTrajectoryVisualization() {
  if (!trajectoryGroup || !simulation) return

  // Clear existing visualization
  trajectoryGroup.selectAll('*').remove()

  // Combine all nodes and edges from paths
  const allNodes: ContextTrajectoryNode[] = []
  const allEdges: any[] = []
  const nodeMap = new Map<string, ContextTrajectoryNode>()

  trajectoryPaths.value.forEach((path, pathIndex) => {
    path.nodes.forEach(node => {
      if (!nodeMap.has(node.id)) {
        const enhancedNode = {
          ...node,
          pathIndex,
          x: canvasWidth.value / 2 + (Math.random() - 0.5) * 200,
          y: canvasHeight.value / 2 + (Math.random() - 0.5) * 200
        }
        nodeMap.set(node.id, enhancedNode)
        allNodes.push(enhancedNode)
      }
    })
    
    path.edges.forEach(edge => {
      allEdges.push({
        ...edge,
        source: edge.source_id,
        target: edge.target_id,
        pathIndex
      })
    })
  })

  // Update simulation
  simulation.nodes(allNodes)
  
  const linkForce = simulation.force('link') as d3.ForceLink<ContextTrajectoryNode, any>
  linkForce.links(allEdges)

  // Render edges
  const edgeSelection = trajectoryGroup
    .selectAll<SVGLineElement, any>('line')
    .data(allEdges)
    .join('line')
    .attr('class', 'trajectory-edge')
    .style('stroke', (d: any) => getEdgeColor(d.relationship_type))
    .style('stroke-width', 2)
    .style('stroke-opacity', 0.6)
    .style('marker-end', 'url(#arrowhead)')

  // Render nodes
  const nodeSelection = trajectoryGroup
    .selectAll<SVGGElement, ContextTrajectoryNode>('g')
    .data(allNodes)
    .join('g')
    .attr('class', 'trajectory-node')
    .style('cursor', 'pointer')
    .call(d3.drag<SVGGElement, ContextTrajectoryNode>()
      .on('start', dragStarted)
      .on('drag', dragging)
      .on('end', dragEnded))
    .on('click', handleNodeClick)
    .on('mouseover', handleNodeHover)
    .on('mouseout', handleNodeLeave)

  // Add node circles
  nodeSelection
    .append('circle')
    .attr('r', (d: ContextTrajectoryNode) => getNodeSize(d.type))
    .style('fill', (d: ContextTrajectoryNode) => getNodeTypeColor(d.type))
    .style('stroke', '#fff')
    .style('stroke-width', 2)

  // Add node labels
  nodeSelection
    .append('text')
    .attr('dy', (d: ContextTrajectoryNode) => getNodeSize(d.type) + 15)
    .attr('text-anchor', 'middle')
    .style('font-size', '10px')
    .style('font-weight', 'bold')
    .style('fill', '#334155')
    .text((d: ContextTrajectoryNode) => truncateLabel(d.label, 12))

  // Update positions on simulation tick
  simulation.on('tick', () => {
    edgeSelection
      .attr('x1', (d: any) => d.source.x)
      .attr('y1', (d: any) => d.source.y)
      .attr('x2', (d: any) => d.target.x)
      .attr('y2', (d: any) => d.target.y)

    nodeSelection
      .attr('transform', (d: any) => `translate(${d.x},${d.y})`)
  })

  // Start animation if enabled
  if (animationEnabled.value) {
    startFlowAnimation()
  }

  // Restart simulation
  simulation.alpha(1).restart()
}

/**
 * Start flow animation
 */
function startFlowAnimation() {
  if (!trajectoryGroup || !animationEnabled.value) return

  const animateFlow = () => {
    trajectoryPaths.value.forEach((path, pathIndex) => {
      if (path.nodes.length < 2) return

      const firstNode = path.nodes[0]
      const lastNode = path.nodes[path.nodes.length - 1]
      
      // Create particle
      const particle = trajectoryGroup!
        .append('circle')
        .classed('flow-particle', true)
        .attr('r', 3)
        .style('fill', '#F59E0B')
        .style('opacity', 0.8)
        .attr('cx', firstNode.x || 0)
        .attr('cy', firstNode.y || 0)

      // Animate along path
      particle
        .transition()
        .duration(3000 + Math.random() * 2000)
        .ease(d3.easeLinear)
        .attr('cx', lastNode.x || 0)
        .attr('cy', lastNode.y || 0)
        .style('opacity', 0)
        .on('end', function() {
          d3.select(this).remove()
        })
    })
  }

  // Start animation and repeat
  animateFlow()
  const animationInterval = setInterval(() => {
    if (animationEnabled.value) {
      animateFlow()
    } else {
      clearInterval(animationInterval)
    }
  }, 5000)
}

/**
 * Node interaction handlers
 */
function handleNodeClick(event: MouseEvent, node: ContextTrajectoryNode) {
  selectedPath.value = null
  emit('nodeSelected', node)
}

function handleNodeHover(event: MouseEvent, node: ContextTrajectoryNode) {
  hoveredNode.value = node
  tooltipPosition.value = {
    x: event.offsetX,
    y: event.offsetY
  }
}

function handleNodeLeave() {
  hoveredNode.value = null
}

function highlightNode(node: ContextTrajectoryNode) {
  if (!trajectoryGroup) return
  
  // Highlight the node
  trajectoryGroup.selectAll('.trajectory-node')
    .style('opacity', (d: any) => d.id === node.id ? 1 : 0.3)
}

/**
 * Drag handlers
 */
function dragStarted(event: any, d: ContextTrajectoryNode) {
  if (!event.active && simulation) simulation.alphaTarget(0.3).restart()
  d.fx = d.x
  d.fy = d.y
}

function dragging(event: any, d: ContextTrajectoryNode) {
  d.fx = event.x
  d.fy = event.y
}

function dragEnded(event: any, d: ContextTrajectoryNode) {
  if (!event.active && simulation) simulation.alphaTarget(0)
  d.fx = null
  d.fy = null
}

/**
 * Path management
 */
function selectPath(path: ContextTrajectoryPath) {
  selectedPath.value = selectedPath.value?.nodes === path.nodes ? null : path
  
  if (selectedPath.value && trajectoryGroup) {
    // Highlight path nodes
    const pathNodeIds = new Set(path.nodes.map(n => n.id))
    
    trajectoryGroup.selectAll('.trajectory-node')
      .style('opacity', (d: any) => pathNodeIds.has(d.id) ? 1 : 0.3)
    
    trajectoryGroup.selectAll('.trajectory-edge')
      .style('opacity', (d: any) => {
        const sourceId = typeof d.source === 'object' ? d.source.id : d.source
        const targetId = typeof d.target === 'object' ? d.target.id : d.target
        return pathNodeIds.has(sourceId) && pathNodeIds.has(targetId) ? 1 : 0.2
      })
  } else if (trajectoryGroup) {
    // Reset highlighting
    trajectoryGroup.selectAll('.trajectory-node').style('opacity', 1)
    trajectoryGroup.selectAll('.trajectory-edge').style('opacity', 0.6)
  }
  
  emit('pathSelected', path)
}

/**
 * Control handlers
 */
async function refreshTrajectory() {
  await loadTrajectoryData()
}

function fitToView() {
  if (!svg) return
  
  svg.transition().duration(750).call(
    d3.zoom<SVGSVGElement, unknown>().transform,
    d3.zoomIdentity
  )
}

function toggleAnimation() {
  animationEnabled.value = !animationEnabled.value
  
  if (animationEnabled.value) {
    startFlowAnimation()
  } else {
    // Remove existing particles
    trajectoryGroup?.selectAll('.flow-particle').remove()
  }
}

function updateTimeRange() {
  loadTrajectoryData()
}

function updateDepth() {
  loadTrajectoryData()
}

// Debounced search
const debouncedSearch = debounce(() => {
  if (searchQuery.value.length > 2 || searchQuery.value.length === 0) {
    loadTrajectoryData()
  }
}, 500)

/**
 * Utility functions
 */
function getNodeTypeColor(type: string): string {
  const colorMap: { [key: string]: string } = {
    'context': '#3B82F6',
    'event': '#10B981',
    'agent': '#8B5CF6',
    'concept': '#F59E0B'
  }
  return colorMap[type] || '#64748B'
}

function getNodeSize(type: string): number {
  const sizeMap: { [key: string]: number } = {
    'context': 12,
    'event': 8,
    'agent': 10,
    'concept': 6
  }
  return sizeMap[type] || 8
}

function getEdgeColor(relationshipType: string): string {
  const colorMap: { [key: string]: string } = {
    'connection': '#64748B',
    'reference': '#3B82F6',
    'derivation': '#10B981',
    'similarity': '#F59E0B'
  }
  return colorMap[relationshipType] || '#64748B'
}

function getPathStrengthColor(strength: number): string {
  if (strength >= 0.8) return '#10B981' // Green
  if (strength >= 0.6) return '#F59E0B' // Amber
  if (strength >= 0.4) return '#EF4444' // Red
  return '#64748B' // Gray
}

function truncateLabel(text: string, maxLength: number): string {
  return text.length > maxLength ? text.substring(0, maxLength) + '...' : text
}

function formatTime(timestamp: string): string {
  return formatDistanceToNow(new Date(timestamp), { addSuffix: true })
}

function handleResize() {
  if (!canvasContainer.value) return
  
  const rect = canvasContainer.value.getBoundingClientRect()
  canvasWidth.value = rect.width
  canvasHeight.value = rect.height
  
  // Update simulation center
  if (simulation) {
    simulation.force('center', d3.forceCenter(canvasWidth.value / 2, canvasHeight.value / 2))
    simulation.alpha(0.3).restart()
  }
}

// Watch for prop changes
watch(() => [props.contextId, props.agentId, props.sessionId, props.concept], () => {
  loadTrajectoryData()
})

watch(() => analysisMode.value, () => {
  // Update visualization layout based on analysis mode
  if (simulation) {
    switch (analysisMode.value) {
      case 'hierarchy':
        simulation.force('y', d3.forceY((d: any) => {
          const depthMap = { 'context': 100, 'event': 200, 'agent': 300, 'concept': 400 }
          return depthMap[d.type] || 250
        }))
        break
      case 'semantic':
        simulation.force('radial', d3.forceRadial(150, canvasWidth.value / 2, canvasHeight.value / 2))
        break
      default: // flow
        simulation.force('y', null)
        simulation.force('radial', null)
    }
    simulation.alpha(0.3).restart()
  }
})
</script>

<style scoped>
.context-trajectory-view {
  @apply w-full;
}

.trajectory-canvas {
  position: relative;
}

.path-card {
  transition: all 0.2s ease;
}

.path-card:hover {
  transform: translateX(2px);
}

/* Node and edge styling */
:deep(.trajectory-node) {
  transition: opacity 0.3s ease;
}

:deep(.trajectory-edge) {
  transition: opacity 0.3s ease;
}

/* Flow animation */
:deep(.flow-particle) {
  filter: drop-shadow(0 0 4px rgba(245, 158, 11, 0.5));
}

/* Tooltip */
.trajectory-tooltip {
  animation: fadeIn 0.2s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translate(-50%, -100%) scale(0.8);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -100%) scale(1);
  }
}

/* Loading overlay */
.loading-overlay {
  backdrop-filter: blur(2px);
}

/* Dark mode adjustments */
.dark :deep(.trajectory-node text) {
  fill: #E2E8F0;
}

.dark :deep(.trajectory-edge) {
  stroke-opacity: 0.8;
}
</style>