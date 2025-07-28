<template>
  <div class="agent-graph-container">
    <!-- Graph Controls -->
    <div class="absolute top-4 left-4 z-10 space-y-2">
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 space-y-2">
        <!-- Zoom Controls -->
        <div class="flex items-center space-x-2">
          <button
            @click="zoomIn"
            class="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Zoom In"
          >
            <PlusIcon class="w-4 h-4" />
          </button>
          <button
            @click="zoomOut"
            class="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Zoom Out"
          >
            <MinusIcon class="w-4 h-4" />
          </button>
          <button
            @click="resetZoom"
            class="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Reset View"
          >
            <ArrowsPointingOutIcon class="w-4 h-4" />
          </button>
        </div>
        
        <!-- Layout Controls -->
        <div class="border-t pt-2">
          <select
            v-model="layoutType"
            @change="changeLayout"
            class="text-xs bg-transparent border border-gray-300 dark:border-gray-600 rounded px-2 py-1"
          >
            <option value="force">Force Layout</option>
            <option value="circle">Circle Layout</option>
            <option value="grid">Grid Layout</option>
          </select>
        </div>
        
        <!-- Visualization Mode -->
        <div class="border-t pt-2">
          <select
            v-model="visualizationMode"
            @change="updateVisualization"
            class="text-xs bg-transparent border border-gray-300 dark:border-gray-600 rounded px-2 py-1"
          >
            <option value="session">By Session</option>
            <option value="performance">Performance</option>
            <option value="security">Security Risk</option>
            <option value="activity">Activity Level</option>
          </select>
        </div>
      </div>
      
      <!-- Graph Statistics -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3">
        <div class="text-xs text-gray-600 dark:text-gray-400 space-y-1">
          <div class="flex justify-between">
            <span>Agents:</span>
            <span class="font-mono">{{ nodeCount }}</span>
          </div>
          <div class="flex justify-between">
            <span>Connections:</span>
            <span class="font-mono">{{ linkCount }}</span>
          </div>
          <div class="flex justify-between">
            <span>Sessions:</span>
            <span class="font-mono">{{ sessionCount }}</span>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Legend -->
    <div class="absolute top-4 right-4 z-10">
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 max-w-xs">
        <h4 class="text-sm font-semibold text-gray-900 dark:text-white mb-2">
          {{ getLegendTitle() }}
        </h4>
        <div class="space-y-1">
          <div
            v-for="item in legendItems"
            :key="item.id"
            class="flex items-center text-xs text-gray-600 dark:text-gray-400"
          >
            <div
              class="w-3 h-3 rounded-full mr-2"
              :style="{ backgroundColor: item.color }"
            ></div>
            <span>{{ item.label }}</span>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Loading Overlay -->
    <div
      v-if="loading"
      class="absolute inset-0 bg-white dark:bg-gray-900 bg-opacity-75 flex items-center justify-center z-20"
    >
      <div class="text-center">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p class="text-gray-600 dark:text-gray-400">Loading agent graph...</p>
      </div>
    </div>
    
    <!-- SVG Container -->
    <svg
      ref="svgRef"
      class="w-full h-full"
      :class="{ 'cursor-grab': !isDragging, 'cursor-grabbing': isDragging }"
    >
      <!-- Background -->
      <rect
        width="100%"
        height="100%"
        :fill="themeColors.background"
        @click="deselectAll"
      />
      
      <!-- Defs for gradients and patterns -->
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge> 
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
        
        <filter id="shadow">
          <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>
        </filter>
      </defs>
      
      <!-- Graph Container Group -->
      <g ref="graphRef" class="graph-container">
        <!-- Links -->
        <g class="links">
          <line
            v-for="link in links"
            :key="`${link.source.id}-${link.target.id}`"
            :x1="link.source.x"
            :y1="link.source.y"
            :x2="link.target.x"
            :y2="link.target.y"
            :stroke="getLinkColor(link)"
            :stroke-width="getLinkWidth(link)"
            :stroke-opacity="getLinkOpacity(link)"
            stroke-linecap="round"
          />
        </g>
        
        <!-- Nodes -->
        <g class="nodes">
          <g
            v-for="node in nodes"
            :key="node.id"
            :transform="`translate(${node.x || 0}, ${node.y || 0})`"
            class="node-group cursor-pointer"
            @click="selectNode(node)"
            @mouseenter="highlightNode(node)"
            @mouseleave="unhighlightNode(node)"
          >
            <!-- Node Background Circle -->
            <circle
              :r="getNodeRadius(node)"
              :fill="getNodeColor(node)"
              :stroke="getNodeBorderColor(node)"
              :stroke-width="isNodeSelected(node) ? 3 : 1"
              :filter="isNodeHighlighted(node) ? 'url(#glow)' : 'url(#shadow)'"
              :opacity="getNodeOpacity(node)"
            />
            
            <!-- Performance Ring (if in performance mode) -->
            <circle
              v-if="visualizationMode === 'performance'"
              :r="getNodeRadius(node) + 3"
              fill="none"
              :stroke="getPerformanceRingColor(node)"
              :stroke-width="2"
              :stroke-dasharray="getPerformanceDashArray(node)"
              :opacity="0.7"
            />
            
            <!-- Agent Icon or Text -->
            <text
              :font-size="getNodeRadius(node) * 0.4"
              text-anchor="middle"
              dominant-baseline="central"
              :fill="getNodeTextColor(node)"
              class="node-text pointer-events-none select-none"
            >
              {{ getNodeLabel(node) }}
            </text>
            
            <!-- Activity Indicator -->
            <circle
              v-if="node.isActive"
              :r="getNodeRadius(node) * 0.3"
              :cx="getNodeRadius(node) * 0.7"
              :cy="-getNodeRadius(node) * 0.7"
              fill="#10B981"
              class="activity-pulse"
            />
          </g>
        </g>
      </g>
    </svg>
    
    <!-- Node Details Panel -->
    <div
      v-if="selectedNode"
      class="absolute bottom-4 left-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 max-w-sm z-10"
    >
      <div class="flex items-center justify-between mb-3">
        <h4 class="text-sm font-semibold text-gray-900 dark:text-white">
          Agent {{ selectedNode.name }}
        </h4>
        <button
          @click="deselectAll"
          class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
        >
          <XMarkIcon class="w-4 h-4" />
        </button>
      </div>
      
      <div class="space-y-2 text-xs text-gray-600 dark:text-gray-400">
        <div class="flex justify-between">
          <span>Session:</span>
          <span class="font-mono">{{ selectedNode.sessionId?.substring(0, 8) }}...</span>
        </div>
        <div class="flex justify-between">
          <span>Status:</span>
          <span
            class="px-2 py-1 rounded text-xs font-medium"
            :class="getStatusClass(selectedNode.status)"
          >
            {{ selectedNode.status }}
          </span>
        </div>
        <div class="flex justify-between">
          <span>Performance:</span>
          <span :style="{ color: getPerformanceColor(selectedNode.performance) }">
            {{ selectedNode.performance }}%
          </span>
        </div>
        <div class="flex justify-between">
          <span>Connections:</span>
          <span>{{ getNodeConnections(selectedNode).length }}</span>
        </div>
        <div class="flex justify-between">
          <span>Memory:</span>
          <span>{{ selectedNode.memoryUsage }}MB</span>
        </div>
        
        <div v-if="selectedNode.currentActivity" class="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
          <p class="text-xs text-gray-500 dark:text-gray-400 mb-1">Current Activity:</p>
          <p class="text-xs font-medium text-gray-700 dark:text-gray-300">
            {{ selectedNode.currentActivity }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { select, Selection } from 'd3-selection'
import { forceSimulation, forceLink, forceManyBody, forceCenter, forceX, forceY } from 'd3-force'
import { zoom, zoomTransform, ZoomBehavior } from 'd3-zoom'
import { drag } from 'd3-drag'
import { scaleLinear } from 'd3-scale'
import {
  PlusIcon,
  MinusIcon,
  ArrowsPointingOutIcon,
  XMarkIcon
} from '@heroicons/vue/24/outline'

import { useSessionColors } from '@/utils/SessionColorManager'
import { useEventsStore } from '@/stores/events'
import type { AgentInfo, SessionInfo } from '@/types/hooks'

// Props
interface Props {
  width?: number
  height?: number
  autoLayout?: boolean
  showControls?: boolean
  initialZoom?: number
}

const props = withDefaults(defineProps<Props>(), {
  width: 800,
  height: 600,
  autoLayout: true,
  showControls: true,
  initialZoom: 1
})

// Graph Node Interface
interface GraphNode extends AgentInfo {
  x?: number
  y?: number
  vx?: number
  vy?: number
  fx?: number | null
  fy?: number | null
  name: string
  performance: number
  memoryUsage: number
  isActive: boolean
  currentActivity?: string
  connections: string[]
}

// Graph Link Interface
interface GraphLink {
  source: GraphNode
  target: GraphNode
  strength: number
  type: 'communication' | 'collaboration' | 'dependency'
  timestamp: string
}

// Reactive state
const svgRef = ref<SVGElement>()
const graphRef = ref<SVGGElement>()
const loading = ref(true)
const isDragging = ref(false)
const selectedNode = ref<GraphNode | null>(null)
const highlightedNode = ref<GraphNode | null>(null)
const layoutType = ref<'force' | 'circle' | 'grid'>('force')
const visualizationMode = ref<'session' | 'performance' | 'security' | 'activity'>('session')

// D3 instances
let simulation: any = null
let zoomBehavior: ZoomBehavior<SVGElement, unknown> | null = null
let svgSelection: Selection<SVGElement, unknown, null, undefined> | null = null

// Stores and utilities
const eventsStore = useEventsStore()
const { 
  getSessionColor, 
  getAgentColor, 
  getPerformanceColor, 
  getSecurityRiskColor,
  getThemeColors 
} = useSessionColors()

// Theme colors
const themeColors = computed(() => getThemeColors('light'))

// Graph data
const nodes = ref<GraphNode[]>([])
const links = ref<GraphLink[]>([])

// Computed properties
const nodeCount = computed(() => nodes.value.length)
const linkCount = computed(() => links.value.length)
const sessionCount = computed(() => {
  const sessionIds = new Set(nodes.value.map(n => n.session_ids).flat())
  return sessionIds.size
})

const legendItems = computed(() => {
  switch (visualizationMode.value) {
    case 'session':
      return eventsStore.sessions.map(session => ({
        id: session.session_id,
        label: `Session ${session.session_id.substring(0, 8)}`,
        color: getSessionColor(session.session_id).primary
      }))
    case 'performance':
      return [
        { id: 'excellent', label: 'Excellent (90-100%)', color: '#10B981' },
        { id: 'good', label: 'Good (70-89%)', color: '#3B82F6' },
        { id: 'average', label: 'Average (50-69%)', color: '#F59E0B' },
        { id: 'poor', label: 'Poor (30-49%)', color: '#EF4444' },
        { id: 'critical', label: 'Critical (0-29%)', color: '#7F1D1D' }
      ]
    case 'security':
      return [
        { id: 'safe', label: 'Safe', color: '#10B981' },
        { id: 'low', label: 'Low Risk', color: '#84CC16' },
        { id: 'medium', label: 'Medium Risk', color: '#F59E0B' },
        { id: 'high', label: 'High Risk', color: '#EF4444' },
        { id: 'critical', label: 'Critical Risk', color: '#7F1D1D' }
      ]
    case 'activity':
      return [
        { id: 'high', label: 'High Activity', color: '#10B981' },
        { id: 'medium', label: 'Medium Activity', color: '#F59E0B' },
        { id: 'low', label: 'Low Activity', color: '#EF4444' },
        { id: 'idle', label: 'Idle', color: '#6B7280' }
      ]
    default:
      return []
  }
})

// Node and link styling functions
const getNodeRadius = (node: GraphNode): number => {
  const baseRadius = 20
  const performanceMultiplier = (node.performance || 50) / 100
  return baseRadius + (performanceMultiplier * 10)
}

const getNodeColor = (node: GraphNode): string => {
  switch (visualizationMode.value) {
    case 'session':
      const sessionId = node.session_ids[0] || 'default'
      return getAgentColor(node.agent_id, sessionId)
    case 'performance':
      return getPerformanceColor(node.performance || 50)
    case 'security':
      // Mock security risk calculation based on blocked count
      const riskLevel = node.blocked_count > 5 ? 'CRITICAL' : 
                       node.blocked_count > 2 ? 'HIGH' :
                       node.blocked_count > 0 ? 'MEDIUM' : 'SAFE'
      return getSecurityRiskColor(riskLevel as any)
    case 'activity':
      const activityLevel = node.event_count > 100 ? '#10B981' :
                           node.event_count > 50 ? '#F59E0B' :
                           node.event_count > 0 ? '#EF4444' : '#6B7280'
      return activityLevel
    default:
      return '#3B82F6'
  }
}

const getNodeBorderColor = (node: GraphNode): string => {
  if (isNodeSelected(node)) return '#1F2937'
  if (isNodeHighlighted(node)) return '#FFFFFF'
  return '#E5E7EB'
}

const getNodeTextColor = (node: GraphNode): string => {
  return '#FFFFFF'
}

const getNodeOpacity = (node: GraphNode): number => {
  if (selectedNode.value && selectedNode.value !== node) return 0.6
  return 1
}

const getNodeLabel = (node: GraphNode): string => {
  return node.name?.substring(0, 2) || node.agent_id.substring(0, 2).toUpperCase()
}

const getLinkColor = (link: GraphLink): string => {
  const sessionId = link.source.session_ids[0] || link.target.session_ids[0]
  if (sessionId) {
    return getSessionColor(sessionId).primary
  }
  return '#9CA3AF'
}

const getLinkWidth = (link: GraphLink): number => {
  return Math.max(1, link.strength * 3)
}

const getLinkOpacity = (link: GraphLink): number => {
  if (selectedNode.value) {
    if (link.source === selectedNode.value || link.target === selectedNode.value) {
      return 0.8
    }
    return 0.2
  }
  return 0.6
}

const getPerformanceRingColor = (node: GraphNode): string => {
  return getPerformanceColor(node.performance || 50)
}

const getPerformanceDashArray = (node: GraphNode): string => {
  const performance = node.performance || 50
  const circumference = 2 * Math.PI * (getNodeRadius(node) + 3)
  const dashLength = (performance / 100) * circumference
  return `${dashLength} ${circumference - dashLength}`
}

const isNodeSelected = (node: GraphNode): boolean => {
  return selectedNode.value === node
}

const isNodeHighlighted = (node: GraphNode): boolean => {
  return highlightedNode.value === node
}

const getNodeConnections = (node: GraphNode) => {
  return links.value.filter(link => 
    link.source === node || link.target === node
  )
}

const getStatusClass = (status: string) => {
  const classes = {
    'active': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'idle': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'error': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    'blocked': 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
  }
  return classes[status as keyof typeof classes] || 'bg-gray-100 text-gray-800'
}

const getLegendTitle = (): string => {
  switch (visualizationMode.value) {
    case 'session': return 'Sessions'
    case 'performance': return 'Performance Levels'
    case 'security': return 'Security Risk Levels'
    case 'activity': return 'Activity Levels'
    default: return 'Legend'
  }
}

// Node interaction functions
const selectNode = (node: GraphNode) => {
  selectedNode.value = selectedNode.value === node ? null : node
  updateVisualization()
}

const highlightNode = (node: GraphNode) => {
  highlightedNode.value = node
}

const unhighlightNode = (node: GraphNode) => {
  if (highlightedNode.value === node) {
    highlightedNode.value = null
  }
}

const deselectAll = () => {
  selectedNode.value = null
  highlightedNode.value = null
  updateVisualization()
}

// Zoom and pan functions
const zoomIn = () => {
  if (svgSelection && zoomBehavior) {
    svgSelection.transition().duration(300).call(
      zoomBehavior.scaleBy, 1.5
    )
  }
}

const zoomOut = () => {
  if (svgSelection && zoomBehavior) {
    svgSelection.transition().duration(300).call(
      zoomBehavior.scaleBy, 1 / 1.5
    )
  }
}

const resetZoom = () => {
  if (svgSelection && zoomBehavior) {
    svgSelection.transition().duration(500).call(
      zoomBehavior.transform,
      zoomTransform(svgSelection.node()!)
        .scale(1)
        .translate(0, 0)
    )
  }
}

// Layout functions
const changeLayout = () => {
  if (simulation) {
    simulation.stop()
  }
  initializeSimulation()
  startSimulation()
}

const updateVisualization = () => {
  // Force reactivity update
  nextTick(() => {
    if (simulation) {
      simulation.alpha(0.3).restart()
    }
  })
}

// Simulation setup
const initializeSimulation = () => {
  if (!nodes.value.length) return

  const width = props.width
  const height = props.height

  simulation = forceSimulation(nodes.value)
    .force('link', forceLink(links.value)
      .id((d: any) => d.agent_id)
      .distance(80)
      .strength(0.1)
    )
    .force('charge', forceManyBody().strength(-300))
    .force('center', forceCenter(width / 2, height / 2))
    .force('x', forceX(width / 2).strength(0.1))
    .force('y', forceY(height / 2).strength(0.1))

  // Apply different layouts
  switch (layoutType.value) {
    case 'circle':
      applyCircleLayout()
      break
    case 'grid':
      applyGridLayout()
      break
    case 'force':
    default:
      // Default force layout is already set up
      break
  }
}

const applyCircleLayout = () => {
  const centerX = props.width / 2
  const centerY = props.height / 2
  const radius = Math.min(props.width, props.height) / 3

  nodes.value.forEach((node, i) => {
    const angle = (i / nodes.value.length) * 2 * Math.PI
    node.fx = centerX + radius * Math.cos(angle)
    node.fy = centerY + radius * Math.sin(angle)
  })
  
  simulation?.force('center', null)
  simulation?.force('x', null)
  simulation?.force('y', null)
}

const applyGridLayout = () => {
  const cols = Math.ceil(Math.sqrt(nodes.value.length))
  const cellWidth = props.width / cols
  const cellHeight = props.height / Math.ceil(nodes.value.length / cols)

  nodes.value.forEach((node, i) => {
    const col = i % cols
    const row = Math.floor(i / cols)
    node.fx = col * cellWidth + cellWidth / 2
    node.fy = row * cellHeight + cellHeight / 2
  })
  
  simulation?.force('center', null)
  simulation?.force('x', null)
  simulation?.force('y', null)
}

const startSimulation = () => {
  if (!simulation) return

  simulation.on('tick', () => {
    // The template will reactively update as node positions change
  })

  simulation.alpha(1).restart()
}

// Data loading and processing
const loadGraphData = async () => {
  loading.value = true
  
  try {
    // Convert agents to graph nodes
    nodes.value = eventsStore.agents.map((agent, index) => ({
      ...agent,
      name: `Agent-${index + 1}`,
      performance: Math.floor(Math.random() * 40) + 60, // Mock performance
      memoryUsage: Math.floor(Math.random() * 200) + 50, // Mock memory usage
      isActive: agent.status === 'active',
      currentActivity: agent.status === 'active' 
        ? ['Processing tasks', 'Analyzing data', 'Executing commands'][Math.floor(Math.random() * 3)]
        : undefined,
      connections: [] // Will be populated based on links
    }))
    
    // Generate mock links between agents
    links.value = []
    for (let i = 0; i < nodes.value.length; i++) {
      for (let j = i + 1; j < nodes.value.length; j++) {
        const source = nodes.value[i]
        const target = nodes.value[j]
        
        // Create links based on shared sessions
        const sharedSessions = source.session_ids.filter(sid => 
          target.session_ids.includes(sid)
        )
        
        if (sharedSessions.length > 0 || Math.random() < 0.3) {
          links.value.push({
            source,
            target,
            strength: Math.random() * 0.8 + 0.2,
            type: ['communication', 'collaboration', 'dependency'][Math.floor(Math.random() * 3)] as any,
            timestamp: new Date().toISOString()
          })
        }
      }
    }
    
    // Update node connections
    nodes.value.forEach(node => {
      node.connections = links.value
        .filter(link => link.source === node || link.target === node)
        .map(link => link.source === node ? link.target.agent_id : link.source.agent_id)
    })
    
  } catch (error) {
    console.error('Failed to load graph data:', error)
  } finally {
    loading.value = false
  }
}

// Setup D3 behaviors
const setupZoomAndPan = () => {
  if (!svgRef.value || !graphRef.value) return

  svgSelection = select(svgRef.value)
  const graphSelection = select(graphRef.value)

  zoomBehavior = zoom<SVGElement, unknown>()
    .scaleExtent([0.1, 5])
    .on('zoom', (event) => {
      graphSelection.attr('transform', event.transform)
    })

  svgSelection.call(zoomBehavior as any)
}

const setupNodeDragging = () => {
  // Node dragging is handled by the simulation itself
  // This would be expanded for custom drag behaviors if needed
}

// Watchers
watch(() => eventsStore.agents, () => {
  loadGraphData()
}, { deep: true })

watch(() => eventsStore.sessions, () => {
  loadGraphData()
}, { deep: true })

watch(() => nodes.value, () => {
  if (nodes.value.length > 0) {
    nextTick(() => {
      initializeSimulation()
      startSimulation()
    })
  }
}, { deep: true })

// Lifecycle
onMounted(async () => {
  await loadGraphData()
  
  nextTick(() => {
    setupZoomAndPan()
    setupNodeDragging()
    
    if (nodes.value.length > 0) {
      initializeSimulation()
      startSimulation()
    }
  })
})

onUnmounted(() => {
  if (simulation) {
    simulation.stop()
  }
})
</script>

<style scoped>
.agent-graph-container {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: linear-gradient(45deg, #f8fafc 0%, #f1f5f9 100%);
  border-radius: 0.75rem;
}

.dark .agent-graph-container {
  background: linear-gradient(45deg, #1e293b 0%, #0f172a 100%);
}

.node-group {
  transition: all 0.2s ease;
}

.node-group:hover {
  transform: scale(1.05);
}

.activity-pulse {
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.links line {
  transition: all 0.2s ease;
}

.btn-secondary {
  @apply inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500;
}

.dark .btn-secondary {
  @apply border-gray-600 text-gray-300 bg-gray-800 hover:bg-gray-700 focus:ring-offset-gray-800;
}
</style>