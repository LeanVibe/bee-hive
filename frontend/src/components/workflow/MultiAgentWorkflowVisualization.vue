<template>
  <div class="workflow-visualization" ref="containerRef">
    <!-- Visualization Controls -->
    <div class="visualization-controls absolute top-4 left-4 z-10 flex items-center space-x-2">
      <div class="controls-group bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-lg px-3 py-2 shadow-lg">
        <!-- Layout Controls -->
        <div class="flex items-center space-x-2">
          <label class="text-xs font-medium text-slate-600 dark:text-slate-400">Layout:</label>
          <select
            v-model="selectedLayout"
            @change="changeLayout"
            class="text-xs border border-slate-300 dark:border-slate-600 rounded px-2 py-1 bg-white dark:bg-slate-700"
          >
            <option value="force">Force</option>
            <option value="hierarchical">Hierarchical</option>
            <option value="circular">Circular</option>
            <option value="dagre">Dagre</option>
          </select>
        </div>
        
        <!-- View Controls -->
        <div class="flex items-center space-x-1 ml-4">
          <button
            @click="resetView"
            class="p-1.5 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white rounded transition-colors"
            title="Reset View"
          >
            <ViewfinderCircleIcon class="w-4 h-4" />
          </button>
          <button
            @click="centerView"
            class="p-1.5 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white rounded transition-colors"
            title="Center View"
          >
            <MagnifyingGlassIcon class="w-4 h-4" />
          </button>
          <button
            @click="fitToScreen"
            class="p-1.5 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white rounded transition-colors"
            title="Fit to Screen"
          >
            <ArrowsPointingOutIcon class="w-4 h-4" />
          </button>
        </div>
      </div>
      
      <!-- Real-time Status -->
      <div class="status-indicator bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-lg px-3 py-2 shadow-lg">
        <div class="flex items-center space-x-2">
          <div 
            class="w-2 h-2 rounded-full"
            :class="connectionStatusClass"
          ></div>
          <span class="text-xs font-medium">
            {{ isConnected ? 'Live' : 'Offline' }}
          </span>
          <span class="text-xs text-slate-500 dark:text-slate-400">
            {{ nodeCount }} nodes, {{ edgeCount }} edges
          </span>
        </div>
      </div>
    </div>

    <!-- Legend -->
    <div class="workflow-legend absolute top-4 right-4 z-10 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-lg p-3 shadow-lg">
      <h4 class="text-xs font-semibold text-slate-900 dark:text-white mb-2">Node Status</h4>
      <div class="space-y-1">
        <div class="flex items-center space-x-2">
          <div class="w-3 h-3 rounded-full bg-gray-400"></div>
          <span class="text-xs text-slate-600 dark:text-slate-400">Pending</span>
        </div>
        <div class="flex items-center space-x-2">
          <div class="w-3 h-3 rounded-full bg-blue-500"></div>
          <span class="text-xs text-slate-600 dark:text-slate-400">Running</span>
        </div>
        <div class="flex items-center space-x-2">
          <div class="w-3 h-3 rounded-full bg-green-500"></div>
          <span class="text-xs text-slate-600 dark:text-slate-400">Completed</span>
        </div>
        <div class="flex items-center space-x-2">
          <div class="w-3 h-3 rounded-full bg-red-500"></div>
          <span class="text-xs text-slate-600 dark:text-slate-400">Failed</span>
        </div>
      </div>
    </div>

    <!-- Main SVG Canvas -->
    <svg
      ref="svgRef"
      class="workflow-canvas w-full h-full"
      :viewBox="`0 0 ${dimensions.width} ${dimensions.height}`"
    ></svg>

    <!-- Loading Overlay -->
    <div
      v-if="isLoading"
      class="absolute inset-0 bg-white/50 dark:bg-slate-900/50 flex items-center justify-center z-20"
    >
      <div class="text-center">
        <div class="animate-spin w-8 h-8 border-2 border-primary-600 border-t-transparent rounded-full mx-auto mb-2"></div>
        <p class="text-sm text-slate-600 dark:text-slate-400">Loading workflow...</p>
      </div>
    </div>

    <!-- Node Details Tooltip -->
    <div
      v-if="tooltip.visible"
      ref="tooltipRef"
      class="tooltip absolute z-30 bg-slate-900 text-white text-xs rounded-lg px-3 py-2 shadow-xl pointer-events-none"
      :style="{ left: tooltip.x + 'px', top: tooltip.y + 'px' }"
    >
      <div class="font-semibold">{{ tooltip.data.name }}</div>
      <div class="text-slate-300 mt-1">{{ tooltip.data.type }}</div>
      <div v-if="tooltip.data.status" class="text-slate-300">
        Status: <span :class="getStatusColor(tooltip.data.status)">{{ tooltip.data.status }}</span>
      </div>
      <div v-if="tooltip.data.agent" class="text-slate-300">
        Agent: {{ tooltip.data.agent }}
      </div>
      <div v-if="tooltip.data.progress !== undefined" class="text-slate-300">
        Progress: {{ Math.round(tooltip.data.progress) }}%
      </div>
      <div v-if="tooltip.data.duration" class="text-slate-300">
        Duration: {{ formatDuration(tooltip.data.duration) }}
      </div>
    </div>

    <!-- Context Menu -->
    <div
      v-if="contextMenu.visible"
      ref="contextMenuRef"
      class="context-menu absolute z-40 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-xl py-1 min-w-[160px]"
      :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }"
    >
      <button
        v-for="action in contextMenu.actions"
        :key="action.id"
        @click="executeContextAction(action)"
        class="w-full px-3 py-2 text-left text-sm hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
        :class="action.disabled ? 'text-slate-400 cursor-not-allowed' : 'text-slate-700 dark:text-slate-300'"
        :disabled="action.disabled"
      >
        <component :is="action.icon" class="w-4 h-4 inline mr-2" />
        {{ action.label }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as d3 from 'd3'
import {
  ViewfinderCircleIcon,
  MagnifyingGlassIcon,
  ArrowsPointingOutIcon,
  PlayIcon,
  PauseIcon,
  StopIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/vue/24/outline'

import type {
  WorkflowGraphData,
  WorkflowGraphNode,
  WorkflowGraphEdge,
  NodeStatus,
  LayoutType
} from '@/types/workflows'
import { useWorkflowStore } from '@/stores/workflows'
import { useUnifiedWebSocket } from '@/services/unifiedWebSocketManager'

// Props
interface Props {
  width?: number
  height?: number
  autoLayout?: boolean
  showControls?: boolean
  workflowId?: string | null
  executionId?: string | null
}

const props = withDefaults(defineProps<Props>(), {
  width: 800,
  height: 600,
  autoLayout: true,
  showControls: true,
  workflowId: null,
  executionId: null
})

// Emits
const emit = defineEmits<{
  nodeSelected: [node: WorkflowGraphNode]
  nodeDoubleClick: [node: WorkflowGraphNode]
  edgeSelected: [edge: WorkflowGraphEdge]
  layoutChanged: [layout: LayoutType]
}>()

// Refs
const containerRef = ref<HTMLDivElement>()
const svgRef = ref<SVGSVGElement>()
const tooltipRef = ref<HTMLDivElement>()
const contextMenuRef = ref<HTMLDivElement>()

// Store
const workflowStore = useWorkflowStore()
const webSocket = useUnifiedWebSocket()

// State
const isLoading = ref(false)
const isConnected = ref(false)
const selectedLayout = ref<LayoutType>('force')
const dimensions = ref({ width: props.width, height: props.height })

// D3 References
let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>
let g: d3.Selection<SVGGElement, unknown, null, undefined>
let zoom: d3.ZoomBehavior<SVGSVGElement, unknown>
let simulation: d3.Simulation<WorkflowGraphNode, WorkflowGraphEdge>

// Graph Data
const graphData = ref<WorkflowGraphData>({
  nodes: [],
  edges: [],
  layout: { type: 'force', options: {} },
  metadata: {
    title: 'Workflow Visualization',
    description: '',
    version: '1.0.0',
    author: '',
    createdAt: new Date(),
    updatedAt: new Date(),
    tags: [],
    complexity: {
      nodeCount: 0,
      edgeCount: 0,
      depth: 0,
      parallelism: 0,
      cyclomaticComplexity: 0
    }
  }
})

// Tooltip State
const tooltip = ref({
  visible: false,
  x: 0,
  y: 0,
  data: {} as any
})

// Context Menu State
const contextMenu = ref({
  visible: false,
  x: 0,
  y: 0,
  node: null as WorkflowGraphNode | null,
  actions: [] as any[]
})

// Computed Properties
const nodeCount = computed(() => graphData.value.nodes.length)
const edgeCount = computed(() => graphData.value.edges.length)

const connectionStatusClass = computed(() => 
  isConnected.value ? 'bg-green-500 animate-pulse' : 'bg-red-500'
)

// Initialization
onMounted(async () => {
  await initializeVisualization()
  setupWebSocketSubscriptions()
  setupEventListeners()
  
  if (props.workflowId || props.executionId) {
    await loadWorkflowData()
  }
})

onUnmounted(() => {
  cleanup()
})

// Watch for prop changes
watch(() => props.workflowId, async (newId) => {
  if (newId) {
    await loadWorkflowData()
  }
})

watch(() => props.executionId, async (newId) => {
  if (newId) {
    await loadExecutionData()
  }
})

// Methods
const initializeVisualization = async (): Promise<void> => {
  if (!svgRef.value || !containerRef.value) return

  // Get container dimensions
  const rect = containerRef.value.getBoundingClientRect()
  dimensions.value = { width: rect.width, height: rect.height }

  // Initialize D3 SVG
  svg = d3.select(svgRef.value)
    .attr('width', dimensions.value.width)
    .attr('height', dimensions.value.height)

  // Create main group for zoom/pan
  g = svg.append('g').attr('class', 'workflow-group')

  // Setup zoom behavior
  zoom = d3.zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.1, 4])
    .on('zoom', (event) => {
      g.attr('transform', event.transform)
    })

  svg.call(zoom)

  // Create marker definitions for arrows
  svg.append('defs').selectAll('marker')
    .data(['end-arrow', 'end-arrow-selected'])
    .enter().append('marker')
    .attr('id', d => d)
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 15)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', d => d === 'end-arrow-selected' ? '#3b82f6' : '#64748b')

  // Initialize force simulation
  initializeSimulation()
}

const initializeSimulation = (): void => {
  simulation = d3.forceSimulation<WorkflowGraphNode>()
    .force('link', d3.forceLink<WorkflowGraphNode, WorkflowGraphEdge>()
      .id(d => d.id)
      .distance(100)
      .strength(0.1)
    )
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(dimensions.value.width / 2, dimensions.value.height / 2))
    .force('collision', d3.forceCollide().radius(40))
    .alphaDecay(0.02)
    .velocityDecay(0.3)

  simulation.on('tick', updatePositions)
}

const loadWorkflowData = async (): Promise<void> => {
  if (!props.workflowId) return

  try {
    isLoading.value = true
    
    // Load workflow from store
    const workflow = workflowStore.state.activeWorkflows.get(props.workflowId)
    if (workflow) {
      // Convert workflow to graph data
      graphData.value = convertWorkflowToGraph(workflow)
      await nextTick()
      updateVisualization()
    }
  } catch (error) {
    console.error('Failed to load workflow data:', error)
  } finally {
    isLoading.value = false
  }
}

const loadExecutionData = async (): Promise<void> => {
  if (!props.executionId) return

  try {
    isLoading.value = true
    
    // Load execution from store
    const execution = workflowStore.state.executions.get(props.executionId)
    if (execution) {
      // Convert execution to graph data with progress
      graphData.value = convertExecutionToGraph(execution)
      await nextTick()
      updateVisualization()
    }
  } catch (error) {
    console.error('Failed to load execution data:', error)
  } finally {
    isLoading.value = false
  }
}

const convertWorkflowToGraph = (workflow: any): WorkflowGraphData => {
  // This would convert workflow data to graph format
  // For now, creating sample data
  const nodes: WorkflowGraphNode[] = [
    {
      id: 'start',
      label: 'Start',
      type: 'start',
      status: 'completed',
      position: { x: 100, y: 300 },
      size: { width: 60, height: 40 },
      style: { fill: '#10b981', stroke: '#059669', strokeWidth: 2, shape: 'circle' },
      data: { label: 'Start', description: 'Workflow start', capabilities: [], parameters: {}, metadata: {} }
    },
    {
      id: 'task1',
      label: 'Data Processing',
      type: 'task',
      status: 'running',
      position: { x: 200, y: 200 },
      size: { width: 120, height: 60 },
      style: { fill: '#3b82f6', stroke: '#2563eb', strokeWidth: 2, shape: 'rectangle' },
      data: { label: 'Data Processing', description: 'Process input data', capabilities: ['data-processing'], parameters: {}, metadata: {} },
      agentId: 'agent-1',
      progress: 75
    },
    {
      id: 'task2',
      label: 'Analysis',
      type: 'task',
      status: 'pending',
      position: { x: 200, y: 400 },
      size: { width: 120, height: 60 },
      style: { fill: '#6b7280', stroke: '#4b5563', strokeWidth: 2, shape: 'rectangle' },
      data: { label: 'Analysis', description: 'Analyze processed data', capabilities: ['analysis'], parameters: {}, metadata: {} }
    },
    {
      id: 'gateway1',
      label: 'Decision',
      type: 'condition',
      status: 'pending',
      position: { x: 350, y: 300 },
      size: { width: 80, height: 80 },
      style: { fill: '#f59e0b', stroke: '#d97706', strokeWidth: 2, shape: 'diamond' },
      data: { label: 'Decision', description: 'Route based on results', capabilities: [], parameters: {}, metadata: {} }
    },
    {
      id: 'end',
      label: 'End',
      type: 'end',
      status: 'pending',
      position: { x: 500, y: 300 },
      size: { width: 60, height: 40 },
      style: { fill: '#ef4444', stroke: '#dc2626', strokeWidth: 2, shape: 'circle' },
      data: { label: 'End', description: 'Workflow end', capabilities: [], parameters: {}, metadata: {} }
    }
  ]

  const edges: WorkflowGraphEdge[] = [
    {
      id: 'start-task1',
      source: 'start',
      target: 'task1',
      type: 'sequence',
      style: { stroke: '#64748b', strokeWidth: 2 }
    },
    {
      id: 'start-task2',
      source: 'start',
      target: 'task2',
      type: 'parallel',
      style: { stroke: '#64748b', strokeWidth: 2 }
    },
    {
      id: 'task1-gateway1',
      source: 'task1',
      target: 'gateway1',
      type: 'sequence',
      style: { stroke: '#64748b', strokeWidth: 2 }
    },
    {
      id: 'task2-gateway1',
      source: 'task2',
      target: 'gateway1',
      type: 'sequence',
      style: { stroke: '#64748b', strokeWidth: 2 }
    },
    {
      id: 'gateway1-end',
      source: 'gateway1',
      target: 'end',
      type: 'sequence',
      style: { stroke: '#64748b', strokeWidth: 2 }
    }
  ]

  return {
    nodes,
    edges,
    layout: { type: selectedLayout.value, options: {} },
    metadata: {
      title: workflow.name || 'Workflow',
      description: workflow.description || '',
      version: '1.0.0',
      author: '',
      createdAt: workflow.createdAt || new Date(),
      updatedAt: new Date(),
      tags: [],
      complexity: {
        nodeCount: nodes.length,
        edgeCount: edges.length,
        depth: 3,
        parallelism: 2,
        cyclomaticComplexity: 2
      }
    }
  }
}

const convertExecutionToGraph = (execution: any): WorkflowGraphData => {
  // This would convert execution data to graph format with real-time status
  return convertWorkflowToGraph(execution) // Simplified for now
}

const updateVisualization = (): void => {
  if (!g || !graphData.value.nodes.length) return

  // Clear existing elements
  g.selectAll('*').remove()

  // Create links
  const links = g.selectAll('.link')
    .data(graphData.value.edges)
    .enter().append('line')
    .attr('class', 'link')
    .attr('stroke', d => d.style.stroke || '#64748b')
    .attr('stroke-width', d => d.style.strokeWidth || 2)
    .attr('stroke-dasharray', d => d.style.strokeDasharray || null)
    .attr('marker-end', 'url(#end-arrow)')

  // Create nodes
  const nodes = g.selectAll('.node')
    .data(graphData.value.nodes)
    .enter().append('g')
    .attr('class', 'node')
    .style('cursor', 'pointer')
    .call(d3.drag<SVGGElement, WorkflowGraphNode>()
      .on('start', dragStarted)
      .on('drag', dragged)
      .on('end', dragEnded)
    )
    .on('click', handleNodeClick)
    .on('dblclick', handleNodeDoubleClick)
    .on('contextmenu', handleNodeContextMenu)
    .on('mouseenter', handleNodeMouseEnter)
    .on('mouseleave', handleNodeMouseLeave)

  // Add shapes based on node type
  nodes.each(function(d) {
    const node = d3.select(this)
    
    if (d.style.shape === 'circle') {
      node.append('circle')
        .attr('r', Math.max(d.size.width, d.size.height) / 2)
        .attr('fill', d.style.fill)
        .attr('stroke', d.style.stroke)
        .attr('stroke-width', d.style.strokeWidth || 2)
    } else if (d.style.shape === 'diamond') {
      const size = Math.max(d.size.width, d.size.height) / 2
      node.append('polygon')
        .attr('points', `0,${-size} ${size},0 0,${size} ${-size},0`)
        .attr('fill', d.style.fill)
        .attr('stroke', d.style.stroke)
        .attr('stroke-width', d.style.strokeWidth || 2)
    } else {
      node.append('rect')
        .attr('width', d.size.width)
        .attr('height', d.size.height)
        .attr('x', -d.size.width / 2)
        .attr('y', -d.size.height / 2)
        .attr('rx', 8)
        .attr('fill', d.style.fill)
        .attr('stroke', d.style.stroke)
        .attr('stroke-width', d.style.strokeWidth || 2)
    }
  })

  // Add progress indicators
  nodes.filter(d => d.progress !== undefined)
    .append('rect')
    .attr('class', 'progress-bar')
    .attr('x', -50)
    .attr('y', 25)
    .attr('width', 100)
    .attr('height', 4)
    .attr('fill', '#e5e7eb')
    .attr('rx', 2)

  nodes.filter(d => d.progress !== undefined)
    .append('rect')
    .attr('class', 'progress-fill')
    .attr('x', -50)
    .attr('y', 25)
    .attr('width', d => (d.progress || 0))
    .attr('height', 4)
    .attr('fill', '#10b981')
    .attr('rx', 2)

  // Add labels
  nodes.append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', '.35em')
    .attr('font-size', '12px')
    .attr('font-weight', '500')
    .attr('fill', '#1f2937')
    .text(d => d.label)

  // Add agent labels
  nodes.filter(d => d.agentId)
    .append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', '1.5em')
    .attr('font-size', '10px')
    .attr('fill', '#6b7280')
    .text(d => `Agent: ${d.agentId}`)

  // Update simulation
  simulation.nodes(graphData.value.nodes)
  ;(simulation.force('link') as d3.ForceLink<WorkflowGraphNode, WorkflowGraphEdge>)?.links(graphData.value.edges)
  simulation.alpha(1).restart()
}

const updatePositions = (): void => {
  if (!g) return

  g.selectAll('.link')
    .attr('x1', (d: any) => d.source.x)
    .attr('y1', (d: any) => d.source.y)
    .attr('x2', (d: any) => d.target.x)
    .attr('y2', (d: any) => d.target.y)

  g.selectAll('.node')
    .attr('transform', (d: any) => `translate(${d.x},${d.y})`)
}

// Event Handlers
const handleNodeClick = (event: MouseEvent, d: WorkflowGraphNode): void => {
  event.stopPropagation()
  emit('nodeSelected', d)
}

const handleNodeDoubleClick = (event: MouseEvent, d: WorkflowGraphNode): void => {
  event.stopPropagation()
  emit('nodeDoubleClick', d)
}

const handleNodeContextMenu = (event: MouseEvent, d: WorkflowGraphNode): void => {
  event.preventDefault()
  showContextMenu(event, d)
}

const handleNodeMouseEnter = (event: MouseEvent, d: WorkflowGraphNode): void => {
  showTooltip(event, d)
}

const handleNodeMouseLeave = (): void => {
  hideTooltip()
}

// Tooltip Methods
const showTooltip = (event: MouseEvent, node: WorkflowGraphNode): void => {
  tooltip.value = {
    visible: true,
    x: event.clientX + 10,
    y: event.clientY - 10,
    data: {
      name: node.label,
      type: node.type,
      status: node.status,
      agent: node.agentId,
      progress: node.progress,
      duration: node.metrics?.executionTime
    }
  }
}

const hideTooltip = (): void => {
  tooltip.value.visible = false
}

// Context Menu Methods
const showContextMenu = (event: MouseEvent, node: WorkflowGraphNode): void => {
  const actions = [
    {
      id: 'info',
      label: 'View Details',
      icon: InformationCircleIcon,
      disabled: false
    },
    {
      id: 'start',
      label: 'Start Task',
      icon: PlayIcon,
      disabled: node.status !== 'pending'
    },
    {
      id: 'pause',
      label: 'Pause Task',
      icon: PauseIcon,
      disabled: node.status !== 'running'
    },
    {
      id: 'stop',
      label: 'Stop Task',
      icon: StopIcon,
      disabled: !['running', 'pending'].includes(node.status)
    }
  ]

  contextMenu.value = {
    visible: true,
    x: event.clientX,
    y: event.clientY,
    node,
    actions
  }
}

const hideContextMenu = (): void => {
  contextMenu.value.visible = false
}

const executeContextAction = (action: any): void => {
  if (action.disabled || !contextMenu.value.node) return

  const node = contextMenu.value.node
  
  switch (action.id) {
    case 'info':
      emit('nodeSelected', node)
      break
    case 'start':
      // Implement start logic
      console.log('Starting task:', node.id)
      break
    case 'pause':
      // Implement pause logic
      console.log('Pausing task:', node.id)
      break
    case 'stop':
      // Implement stop logic
      console.log('Stopping task:', node.id)
      break
  }

  hideContextMenu()
}

// Layout Methods
const changeLayout = (): void => {
  emit('layoutChanged', selectedLayout.value)
  applyLayout(selectedLayout.value)
}

const applyLayout = (layoutType: LayoutType): void => {
  if (!simulation) return

  simulation.stop()

  switch (layoutType) {
    case 'hierarchical':
      applyHierarchicalLayout()
      break
    case 'circular':
      applyCircularLayout()
      break
    case 'dagre':
      applyDagreLayout()
      break
    default:
      applyForceLayout()
  }

  simulation.alpha(1).restart()
}

const applyForceLayout = (): void => {
  simulation
    .force('link', d3.forceLink<WorkflowGraphNode, WorkflowGraphEdge>()
      .id(d => d.id)
      .distance(100)
      .strength(0.1)
    )
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(dimensions.value.width / 2, dimensions.value.height / 2))
    .force('collision', d3.forceCollide().radius(40))
}

const applyHierarchicalLayout = (): void => {
  // Simplified hierarchical layout
  const levels = new Map<string, number>()
  const queue = graphData.value.nodes.filter(n => n.type === 'start')
  
  queue.forEach(node => levels.set(node.id, 0))
  
  let level = 0
  while (queue.length > 0) {
    const node = queue.shift()!
    const currentLevel = levels.get(node.id)!
    
    const outgoing = graphData.value.edges.filter(e => e.source === node.id)
    outgoing.forEach(edge => {
      const target = graphData.value.nodes.find(n => n.id === edge.target)
      if (target && !levels.has(target.id)) {
        levels.set(target.id, currentLevel + 1)
        queue.push(target)
        level = Math.max(level, currentLevel + 1)
      }
    })
  }

  // Position nodes based on levels
  const levelNodes = new Map<number, WorkflowGraphNode[]>()
  levels.forEach((level, nodeId) => {
    const node = graphData.value.nodes.find(n => n.id === nodeId)
    if (node) {
      if (!levelNodes.has(level)) levelNodes.set(level, [])
      levelNodes.get(level)!.push(node)
    }
  })

  levelNodes.forEach((nodes, level) => {
    const y = (level + 1) * (dimensions.value.height / (levelNodes.size + 1))
    nodes.forEach((node, index) => {
      node.position.x = (index + 1) * (dimensions.value.width / (nodes.length + 1))
      node.position.y = y
      node.x = node.position.x
      node.y = node.position.y
    })
  })

  simulation.force('link', null).force('charge', null).force('center', null)
}

const applyCircularLayout = (): void => {
  const radius = Math.min(dimensions.value.width, dimensions.value.height) / 3
  const centerX = dimensions.value.width / 2
  const centerY = dimensions.value.height / 2

  graphData.value.nodes.forEach((node, index) => {
    const angle = (index / graphData.value.nodes.length) * 2 * Math.PI
    node.position.x = centerX + radius * Math.cos(angle)
    node.position.y = centerY + radius * Math.sin(angle)
    node.x = node.position.x
    node.y = node.position.y
  })

  simulation.force('link', null).force('charge', null).force('center', null)
}

const applyDagreLayout = (): void => {
  // Would implement Dagre layout here
  applyHierarchicalLayout() // Fallback to hierarchical
}

// View Control Methods
const resetView = (): void => {
  if (!svg || !zoom) return
  
  svg.transition().duration(750).call(
    zoom.transform,
    d3.zoomIdentity
  )
}

const centerView = (): void => {
  if (!svg || !zoom || !graphData.value.nodes.length) return

  const bounds = g.node()?.getBBox()
  if (!bounds) return

  const centerX = dimensions.value.width / 2
  const centerY = dimensions.value.height / 2
  const translateX = centerX - (bounds.x + bounds.width / 2)
  const translateY = centerY - (bounds.y + bounds.height / 2)

  svg.transition().duration(750).call(
    zoom.transform,
    d3.zoomIdentity.translate(translateX, translateY)
  )
}

const fitToScreen = (): void => {
  if (!svg || !zoom || !graphData.value.nodes.length) return

  const bounds = g.node()?.getBBox()
  if (!bounds) return

  const padding = 50
  const scale = Math.min(
    (dimensions.value.width - padding) / bounds.width,
    (dimensions.value.height - padding) / bounds.height
  )

  const centerX = dimensions.value.width / 2
  const centerY = dimensions.value.height / 2
  const translateX = centerX - scale * (bounds.x + bounds.width / 2)
  const translateY = centerY - scale * (bounds.y + bounds.height / 2)

  svg.transition().duration(750).call(
    zoom.transform,
    d3.zoomIdentity.translate(translateX, translateY).scale(scale)
  )
}

// Drag Handlers
const dragStarted = (event: d3.D3DragEvent<SVGGElement, WorkflowGraphNode, WorkflowGraphNode>): void => {
  if (!event.active) simulation.alphaTarget(0.3).restart()
  event.subject.fx = event.subject.x
  event.subject.fy = event.subject.y
}

const dragged = (event: d3.D3DragEvent<SVGGElement, WorkflowGraphNode, WorkflowGraphNode>): void => {
  event.subject.fx = event.x
  event.subject.fy = event.y
}

const dragEnded = (event: d3.D3DragEvent<SVGGElement, WorkflowGraphNode, WorkflowGraphNode>): void => {
  if (!event.active) simulation.alphaTarget(0)
  event.subject.fx = null
  event.subject.fy = null
}

// WebSocket Integration
const setupWebSocketSubscriptions = (): void => {
  // Subscribe to workflow execution updates
  webSocket.onMessage('workflow_execution_update', (message) => {
    updateNodeStatus(message.data)
  })

  // Subscribe to node status updates
  webSocket.onMessage('node_status_update', (message) => {
    updateNodeStatus(message.data)
  })
}

const updateNodeStatus = (update: any): void => {
  const node = graphData.value.nodes.find(n => n.id === update.node_id)
  if (node) {
    node.status = update.status
    node.progress = update.progress
    if (update.agent_id) {
      node.agentId = update.agent_id
    }
    
    // Update visual representation
    updateVisualization()
  }
}

// Event Listeners
const setupEventListeners = (): void => {
  document.addEventListener('click', handleDocumentClick)
}

const handleDocumentClick = (event: MouseEvent): void => {
  if (contextMenu.value.visible && contextMenuRef.value && 
      !contextMenuRef.value.contains(event.target as Node)) {
    hideContextMenu()
  }
}

// Utility Methods
const getStatusColor = (status: NodeStatus): string => {
  const colors = {
    pending: 'text-gray-400',
    ready: 'text-blue-400',
    running: 'text-blue-500',
    completed: 'text-green-500',
    failed: 'text-red-500',
    cancelled: 'text-orange-500',
    blocked: 'text-yellow-500',
    skipped: 'text-gray-500'
  }
  return colors[status] || 'text-gray-400'
}

const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

// Cleanup
const cleanup = (): void => {
  if (simulation) {
    simulation.stop()
  }
  document.removeEventListener('click', handleDocumentClick)
}

// Expose methods for parent components
defineExpose({
  resetView,
  centerView,
  fitToScreen,
  focusOnNode: (nodeId: string) => {
    const node = graphData.value.nodes.find(n => n.id === nodeId)
    if (node && svg && zoom) {
      const scale = 1.5
      const translateX = dimensions.value.width / 2 - scale * node.position.x
      const translateY = dimensions.value.height / 2 - scale * node.position.y

      svg.transition().duration(750).call(
        zoom.transform,
        d3.zoomIdentity.translate(translateX, translateY).scale(scale)
      )
    }
  },
  updateGraphData: (newData: WorkflowGraphData) => {
    graphData.value = newData
    updateVisualization()
  }
})
</script>

<style scoped>
.workflow-visualization {
  @apply relative w-full h-full bg-slate-50 dark:bg-slate-900 rounded-lg overflow-hidden;
}

.workflow-canvas {
  @apply cursor-grab;
}

.workflow-canvas:active {
  @apply cursor-grabbing;
}

.tooltip {
  @apply max-w-xs;
}

.context-menu {
  @apply min-w-[160px];
}

/* D3 specific styles */
:deep(.node) {
  @apply transition-all duration-200;
}

:deep(.node:hover) {
  @apply drop-shadow-lg;
}

:deep(.link) {
  @apply transition-all duration-200;
}

:deep(.progress-bar) {
  @apply opacity-80;
}

:deep(.progress-fill) {
  @apply transition-all duration-300;
}

/* Animation classes */
@keyframes pulse-node {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}

:deep(.node.running) {
  animation: pulse-node 2s ease-in-out infinite;
}

@keyframes flow {
  0% { stroke-dashoffset: 10; }
  100% { stroke-dashoffset: -10; }
}

:deep(.link.active) {
  stroke-dasharray: 5,5;
  animation: flow 1s linear infinite;
}
</style>