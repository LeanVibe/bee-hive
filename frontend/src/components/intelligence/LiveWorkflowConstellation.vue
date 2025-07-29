<template>
  <div class="live-workflow-constellation">
    <!-- Controls Header -->
    <div class="constellation-controls flex items-center justify-between mb-4">
      <div class="flex items-center space-x-4">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          Live Workflow Constellation
        </h3>
        <div class="flex items-center space-x-2">
          <div 
            class="w-2 h-2 rounded-full"
            :class="isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'"
          ></div>
          <span class="text-xs text-slate-500 dark:text-slate-400">
            {{ isConnected ? 'Live' : 'Disconnected' }}
          </span>
        </div>
      </div>
      
      <div class="flex items-center space-x-3">
        <!-- Layout Controls -->
        <select
          v-model="layoutType"
          @change="updateLayout"
          class="text-xs bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded px-2 py-1"
        >
          <option value="force">Force Layout</option>
          <option value="circular">Circular Layout</option>
          <option value="hierarchical">Hierarchical Layout</option>
        </select>
        
        <!-- Semantic Flow Toggle -->
        <button
          @click="toggleSemanticFlow"
          :class="showSemanticFlow 
            ? 'bg-blue-600 text-white' 
            : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300'"
          class="text-xs px-3 py-1 rounded transition-colors"
        >
          <BeakerIcon class="w-3 h-3 mr-1 inline" />
          Semantic Flow
        </button>
        
        <!-- Reset View -->
        <button
          @click="resetView"
          class="text-xs bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-3 py-1 rounded hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
        >
          <ArrowPathIcon class="w-3 h-3 mr-1 inline" />
          Reset
        </button>
        
        <!-- Time Range -->
        <select
          v-model="timeRangeHours"
          @change="updateTimeRange"
          class="text-xs bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded px-2 py-1"
        >
          <option :value="0.5">30min</option>
          <option :value="1">1h</option>
          <option :value="6">6h</option>
          <option :value="24">24h</option>
        </select>
      </div>
    </div>

    <!-- Constellation Visualization -->
    <div 
      ref="constellationContainer"
      class="constellation-container relative bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700"
      :style="{ height: `${height}px` }"
    >
      <!-- SVG Canvas -->
      <svg
        ref="svgCanvas"
        :width="width"
        :height="height"
        class="absolute inset-0 w-full h-full"
      >
        <!-- Definitions for gradients, patterns, etc. -->
        <defs>
          <radialGradient id="agentGradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%" style="stop-color:#3B82F6;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#1E40AF;stop-opacity:0.8" />
          </radialGradient>
          
          <radialGradient id="conceptGradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%" style="stop-color:#10B981;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#059669;stop-opacity:0.8" />
          </radialGradient>
          
          <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                  refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#64748B" />
          </marker>
          
          <!-- Semantic flow particle animation -->
          <circle id="semanticParticle" r="2" fill="#F59E0B" opacity="0.8" />
        </defs>
        
        <!-- Background grid (optional) -->
        <g v-if="showGrid" class="grid-lines">
          <!-- Grid implementation would go here -->
        </g>
        
        <!-- Edges layer -->
        <g class="edges-layer">
          <!-- Rendered by D3 -->
        </g>
        
        <!-- Semantic flow layer -->
        <g v-if="showSemanticFlow" class="semantic-flow-layer">
          <!-- Rendered by D3 -->
        </g>
        
        <!-- Nodes layer -->
        <g class="nodes-layer">
          <!-- Rendered by D3 -->
        </g>
        
        <!-- Labels layer -->
        <g class="labels-layer">
          <!-- Rendered by D3 -->
        </g>
      </svg>
      
      <!-- Overlay UI Elements -->
      <div class="absolute inset-0 pointer-events-none">
        <!-- Loading Indicator -->
        <div 
          v-if="loading"
          class="absolute inset-0 flex items-center justify-center bg-white/50 dark:bg-slate-900/50"
        >
          <div class="flex items-center space-x-2 text-slate-600 dark:text-slate-400">
            <div class="animate-spin w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
            <span class="text-sm">Loading constellation...</span>
          </div>
        </div>
        
        <!-- Node Info Tooltip -->
        <div
          v-if="selectedNode"
          :style="{ 
            left: `${tooltipPosition.x}px`, 
            top: `${tooltipPosition.y}px`,
            transform: 'translate(-50%, -100%)'
          }"
          class="absolute pointer-events-auto bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg p-3 max-w-64 z-10"
        >
          <div class="text-sm font-medium text-slate-900 dark:text-white mb-1">
            {{ selectedNode.label }}
          </div>
          <div class="text-xs text-slate-500 dark:text-slate-400 space-y-1">
            <div>Type: {{ selectedNode.type }}</div>
            <div v-if="selectedNode.metadata.event_count">
              Events: {{ selectedNode.metadata.event_count }}
            </div>
            <div v-if="selectedNode.metadata.usage_count">
              Usage: {{ selectedNode.metadata.usage_count }}
            </div>
            <div v-if="selectedNode.metadata.last_seen">
              Last seen: {{ formatTime(selectedNode.metadata.last_seen) }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Statistics Panel -->
    <div class="constellation-stats grid grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
      <div class="stat-card bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
        <div class="text-xs text-slate-500 dark:text-slate-400">Active Agents</div>
        <div class="text-lg font-semibold text-blue-600 dark:text-blue-400">
          {{ stats.activeAgents }}
        </div>
      </div>
      
      <div class="stat-card bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
        <div class="text-xs text-slate-500 dark:text-slate-400">Semantic Concepts</div>
        <div class="text-lg font-semibold text-green-600 dark:text-green-400">
          {{ stats.semanticConcepts }}
        </div>
      </div>
      
      <div class="stat-card bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
        <div class="text-xs text-slate-500 dark:text-slate-400">Interactions</div>
        <div class="text-lg font-semibold text-purple-600 dark:text-purple-400">
          {{ stats.totalInteractions }}
        </div>
      </div>
      
      <div class="stat-card bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
        <div class="text-xs text-slate-500 dark:text-slate-400">Avg Latency</div>
        <div class="text-lg font-semibold text-orange-600 dark:text-orange-400">
          {{ stats.avgLatency }}ms
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
  BeakerIcon, 
  ArrowPathIcon,
  CpuChipIcon,
  SparklesIcon
} from '@heroicons/vue/24/outline'

// Services
import { useObservabilityEvents, DashboardEventType } from '@/services/observabilityEventService'
import type { 
  WorkflowConstellation, 
  WorkflowConstellationNode, 
  WorkflowConstellationEdge,
  ObservabilityEvent 
} from '@/services/observabilityEventService'
import { DashboardComponent } from '@/types/coordination'

interface Props {
  width?: number
  height?: number
  autoRefresh?: boolean
  sessionIds?: string[]
  agentIds?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  width: 800,
  height: 600,
  autoRefresh: true,
  sessionIds: () => [],
  agentIds: () => []
})

const emit = defineEmits<{
  nodeSelected: [node: WorkflowConstellationNode]
  edgeSelected: [edge: WorkflowConstellationEdge]
  layoutChanged: [layout: string]
}>()

// Refs
const constellationContainer = ref<HTMLDivElement>()
const svgCanvas = ref<SVGSVGElement>()

// Services
const observabilityEvents = useObservabilityEvents()

// Component state
const loading = ref(false)
const isConnected = computed(() => observabilityEvents.isConnected.value)
const layoutType = ref('force')
const showSemanticFlow = ref(true)
const showGrid = ref(false)
const timeRangeHours = ref(1)

// Constellation data
const constellationData = ref<WorkflowConstellation | null>(null)
const selectedNode = ref<WorkflowConstellationNode | null>(null)
const tooltipPosition = ref({ x: 0, y: 0 })

// D3 simulation and elements
let simulation: d3.Simulation<WorkflowConstellationNode, WorkflowConstellationEdge> | null = null
let svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | null = null
let nodesGroup: d3.Selection<SVGGElement, unknown, null, undefined> | null = null
let edgesGroup: d3.Selection<SVGGElement, unknown, null, undefined> | null = null
let semanticFlowGroup: d3.Selection<SVGGElement, unknown, null, undefined> | null = null

// Statistics
const stats = computed(() => {
  if (!constellationData.value) {
    return {
      activeAgents: 0,
      semanticConcepts: 0,
      totalInteractions: 0,
      avgLatency: 0
    }
  }

  const data = constellationData.value
  return {
    activeAgents: data.nodes.filter(n => n.type === 'agent').length,
    semanticConcepts: data.nodes.filter(n => n.type === 'concept').length,
    totalInteractions: data.edges.reduce((sum, edge) => sum + edge.frequency, 0),
    avgLatency: Math.round(
      data.edges
        .filter(e => e.latency_ms)
        .reduce((sum, e) => sum + (e.latency_ms || 0), 0) / 
      Math.max(1, data.edges.filter(e => e.latency_ms).length)
    )
  }
})

// Real-time update subscription
let subscriptionId: string | null = null

/**
 * Initialize the constellation visualization
 */
onMounted(async () => {
  await nextTick()
  initializeD3()
  await loadConstellationData()
  setupRealTimeSubscription()
  
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
  
  if (simulation) {
    simulation.stop()
  }
  
  stopAutoRefresh()
})

/**
 * Initialize D3 elements and simulation
 */
function initializeD3() {
  if (!svgCanvas.value) return

  svg = d3.select(svgCanvas.value)
  
  // Clear existing content
  svg.selectAll('*').remove()
  
  // Create layer groups
  edgesGroup = svg.append('g').attr('class', 'edges-layer')
  semanticFlowGroup = svg.append('g').attr('class', 'semantic-flow-layer')
  nodesGroup = svg.append('g').attr('class', 'nodes-layer')
  
  // Create zoom behavior
  const zoom = d3.zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.1, 4])
    .on('zoom', (event) => {
      const transform = event.transform
      nodesGroup?.attr('transform', transform)
      edgesGroup?.attr('transform', transform)
      semanticFlowGroup?.attr('transform', transform)
    })
  
  svg.call(zoom)
  
  // Initialize force simulation
  simulation = d3.forceSimulation<WorkflowConstellationNode>()
    .force('link', d3.forceLink<WorkflowConstellationNode, WorkflowConstellationEdge>()
      .id((d: any) => d.id)
      .distance(100)
      .strength(0.1))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(props.width / 2, props.height / 2))
    .force('collision', d3.forceCollide().radius((d: any) => d.size * 10 + 5))
}

/**
 * Load constellation data from backend
 */
async function loadConstellationData() {
  loading.value = true
  
  try {
    const params = {
      session_ids: props.sessionIds.length ? props.sessionIds : undefined,
      agent_ids: props.agentIds.length ? props.agentIds : undefined,
      time_range_hours: timeRangeHours.value,
      include_semantic_flow: showSemanticFlow.value,
      min_interaction_count: 1
    }
    
    constellationData.value = await observabilityEvents.getWorkflowConstellation(params)
    
    if (constellationData.value) {
      await renderConstellation(constellationData.value)
    }
    
  } catch (error) {
    console.error('Failed to load constellation data:', error)
  } finally {
    loading.value = false
  }
}

/**
 * Render the constellation visualization
 */
async function renderConstellation(data: WorkflowConstellation) {
  if (!simulation || !nodesGroup || !edgesGroup) return

  // Update simulation with new data
  simulation.nodes(data.nodes)
  
  const linkForce = simulation.force('link') as d3.ForceLink<WorkflowConstellationNode, WorkflowConstellationEdge>
  linkForce.links(data.edges)

  // Render edges
  renderEdges(data.edges)
  
  // Render nodes
  renderNodes(data.nodes)
  
  // Render semantic flows if enabled
  if (showSemanticFlow.value && semanticFlowGroup) {
    renderSemanticFlows(data.semantic_flows)
  }
  
  // Restart simulation
  simulation.alpha(1).restart()
}

/**
 * Render constellation edges
 */
function renderEdges(edges: WorkflowConstellationEdge[]) {
  if (!edgesGroup) return

  const edgeSelection = edgesGroup
    .selectAll<SVGLineElement, WorkflowConstellationEdge>('line')
    .data(edges, (d: WorkflowConstellationEdge) => `${d.source}-${d.target}`)

  // Remove old edges
  edgeSelection.exit().remove()

  // Add new edges
  const edgeEnter = edgeSelection
    .enter()
    .append('line')
    .attr('class', 'constellation-edge')
    .style('stroke', (d) => getEdgeColor(d.type))
    .style('stroke-width', (d) => Math.max(1, d.strength * 4))
    .style('stroke-opacity', 0.6)
    .style('marker-end', 'url(#arrowhead)')

  // Merge and update
  const edgeMerge = edgeEnter.merge(edgeSelection)
  
  edgeMerge
    .style('stroke-width', (d) => Math.max(1, d.strength * 4))
    .style('stroke-opacity', (d) => Math.max(0.3, d.strength))

  // Position edges on simulation tick
  simulation?.on('tick', () => {
    edgeMerge
      .attr('x1', (d: any) => d.source.x)
      .attr('y1', (d: any) => d.source.y)
      .attr('x2', (d: any) => d.target.x)
      .attr('y2', (d: any) => d.target.y)
  })
}

/**
 * Render constellation nodes
 */
function renderNodes(nodes: WorkflowConstellationNode[]) {
  if (!nodesGroup) return

  const nodeSelection = nodesGroup
    .selectAll<SVGGElement, WorkflowConstellationNode>('g')
    .data(nodes, (d: WorkflowConstellationNode) => d.id)

  // Remove old nodes
  nodeSelection.exit().remove()

  // Add new nodes
  const nodeEnter = nodeSelection
    .enter()
    .append('g')
    .attr('class', 'constellation-node')
    .style('cursor', 'pointer')
    .call(d3.drag<SVGGElement, WorkflowConstellationNode>()
      .on('start', dragStarted)
      .on('drag', dragging)
      .on('end', dragEnded))
    .on('click', handleNodeClick)
    .on('mouseover', handleNodeHover)
    .on('mouseout', handleNodeLeave)

  // Add node circles
  nodeEnter
    .append('circle')
    .attr('class', 'node-circle')
    .attr('r', (d) => d.size * 8)
    .style('fill', (d) => getNodeColor(d.type))
    .style('stroke', '#fff')
    .style('stroke-width', 2)

  // Add node labels
  nodeEnter
    .append('text')
    .attr('class', 'node-label')
    .attr('dy', (d) => d.size * 8 + 15)
    .attr('text-anchor', 'middle')
    .style('font-size', '11px')
    .style('font-weight', 'bold')
    .style('fill', '#334155')
    .text((d) => d.label)

  // Add pulse animation for active nodes
  nodeEnter
    .filter((d) => d.type === 'agent')
    .append('circle')
    .attr('class', 'node-pulse')
    .attr('r', (d) => d.size * 8)
    .style('fill', 'none')
    .style('stroke', (d) => getNodeColor(d.type))
    .style('stroke-width', 2)
    .style('opacity', 0)

  // Merge and update
  const nodeMerge = nodeEnter.merge(nodeSelection)
  
  nodeMerge.select('.node-circle')
    .attr('r', (d) => d.size * 8)
    .style('fill', (d) => getNodeColor(d.type))

  nodeMerge.select('.node-label')
    .attr('dy', (d) => d.size * 8 + 15)
    .text((d) => d.label)

  // Animate pulse for active agents
  nodeMerge
    .selectAll('.node-pulse')
    .transition()
    .duration(2000)
    .ease(d3.easeLinear)
    .attr('r', (d: any) => d.size * 12)
    .style('opacity', 0)
    .on('end', function() {
      d3.select(this).attr('r', (d: any) => d.size * 8).style('opacity', 0)
    })

  // Position nodes on simulation tick
  simulation?.on('tick', () => {
    nodeMerge.attr('transform', (d: any) => `translate(${d.x},${d.y})`)
  })
}

/**
 * Render semantic flow animations
 */
function renderSemanticFlows(semanticFlows: Array<Record<string, any>>) {
  if (!semanticFlowGroup || !constellationData.value) return

  // Create particles for semantic concept flows
  semanticFlows.forEach((flow, index) => {
    if (flow.agents && flow.agents.length > 1) {
      animateSemanticFlow(flow, index)
    }
  })
}

/**
 * Animate semantic flow between agents
 */
function animateSemanticFlow(flow: Record<string, any>, index: number) {
  if (!semanticFlowGroup || !constellationData.value) return

  const sourceAgent = constellationData.value.nodes.find(n => n.id === flow.agents[0])
  const targetAgent = constellationData.value.nodes.find(n => n.id === flow.agents[1])
  
  if (!sourceAgent || !targetAgent) return

  // Create particle
  const particle = semanticFlowGroup
    .append('circle')
    .attr('r', 3)
    .style('fill', '#F59E0B')
    .style('opacity', 0.8)
    .attr('cx', sourceAgent.position.x)
    .attr('cy', sourceAgent.position.y)

  // Animate particle movement
  particle
    .transition()
    .duration(2000 + Math.random() * 1000)
    .ease(d3.easeQuadInOut)
    .attr('cx', targetAgent.position.x)
    .attr('cy', targetAgent.position.y)
    .style('opacity', 0)
    .on('end', function() {
      d3.select(this).remove()
    })
}

/**
 * Get color for node based on type
 */
function getNodeColor(type: string): string {
  const colorMap: Record<string, string> = {
    'agent': '#3B82F6',
    'concept': '#10B981',
    'session': '#8B5CF6'
  }
  return colorMap[type] || '#64748B'
}

/**
 * Get color for edge based on type
 */
function getEdgeColor(type: string): string {
  const colorMap: Record<string, string> = {
    'communication': '#3B82F6',
    'semantic_flow': '#10B981',
    'context_sharing': '#F59E0B'
  }
  return colorMap[type] || '#64748B'
}

/**
 * Handle node drag events
 */
function dragStarted(event: any, d: WorkflowConstellationNode) {
  if (!event.active && simulation) simulation.alphaTarget(0.3).restart()
  d.fx = d.position.x
  d.fy = d.position.y
}

function dragging(event: any, d: WorkflowConstellationNode) {
  d.fx = event.x
  d.fy = event.y
}

function dragEnded(event: any, d: WorkflowConstellationNode) {
  if (!event.active && simulation) simulation.alphaTarget(0)
  d.fx = null
  d.fy = null
}

/**
 * Handle node interactions
 */
function handleNodeClick(event: MouseEvent, node: WorkflowConstellationNode) {
  selectedNode.value = selectedNode.value?.id === node.id ? null : node
  emit('nodeSelected', node)
}

function handleNodeHover(event: MouseEvent, node: WorkflowConstellationNode) {
  selectedNode.value = node
  tooltipPosition.value = {
    x: event.offsetX,
    y: event.offsetY
  }
}

function handleNodeLeave() {
  // Keep tooltip visible for a moment to allow interaction
  setTimeout(() => {
    if (!selectedNode.value) return
    selectedNode.value = null
  }, 100)
}

/**
 * Setup real-time event subscription
 */
function setupRealTimeSubscription() {
  subscriptionId = observabilityEvents.subscribe(
    DashboardComponent.GRAPH,
    [
      DashboardEventType.WORKFLOW_UPDATE,
      DashboardEventType.AGENT_STATUS,
      DashboardEventType.SEMANTIC_INTELLIGENCE
    ],
    handleRealTimeEvent,
    {
      agent_ids: props.agentIds.length ? props.agentIds : undefined,
      session_ids: props.sessionIds.length ? props.sessionIds : undefined
    },
    8 // High priority
  )
}

/**
 * Handle real-time events
 */
async function handleRealTimeEvent(event: ObservabilityEvent) {
  if (event.type === DashboardEventType.WORKFLOW_UPDATE) {
    // Update constellation with new agent interactions
    await updateConstellationFromEvent(event)
  } else if (event.type === DashboardEventType.SEMANTIC_INTELLIGENCE) {
    // Add semantic flow animation
    if (showSemanticFlow.value) {
      animateSemanticEvent(event)
    }
  }
}

/**
 * Update constellation from real-time event
 */
async function updateConstellationFromEvent(event: ObservabilityEvent) {
  if (!constellationData.value || !simulation) return

  const agentUpdates = event.data.agent_updates || []
  let needsRerender = false

  for (const update of agentUpdates) {
    const existingNode = constellationData.value.nodes.find(n => n.id === update.agent_id)
    
    if (existingNode) {
      // Update existing node
      existingNode.metadata = { ...existingNode.metadata, ...update.metadata }
      existingNode.size = Math.min(3.0, 1.0 + (update.activity_level || 0))
    } else if (update.agent_id) {
      // Add new node
      const newNode: WorkflowConstellationNode = {
        id: update.agent_id,
        type: 'agent',
        label: `Agent ${update.agent_id.substring(0, 8)}`,
        position: { x: props.width / 2, y: props.height / 2 },
        size: 1.0,
        color: getNodeColor('agent'),
        metadata: update.metadata || {}
      }
      
      constellationData.value.nodes.push(newNode)
      needsRerender = true
    }
  }

  if (needsRerender) {
    await renderConstellation(constellationData.value)
  } else {
    // Just update the visual representation
    nodesGroup?.selectAll('.node-circle')
      .data(constellationData.value.nodes, (d: any) => d.id)
      .attr('r', (d: any) => d.size * 8)
  }
}

/**
 * Animate semantic event
 */
function animateSemanticEvent(event: ObservabilityEvent) {
  if (!semanticFlowGroup || !event.semantic_concepts?.length) return

  // Create sparkle effect for semantic concepts
  const concept = event.semantic_concepts[0]
  const conceptNode = constellationData.value?.nodes.find(n => 
    n.type === 'concept' && n.label === concept
  )

  if (conceptNode) {
    const sparkle = semanticFlowGroup
      .append('circle')
      .attr('r', 0)
      .style('fill', '#F59E0B')
      .style('opacity', 1)
      .attr('cx', conceptNode.position.x)
      .attr('cy', conceptNode.position.y)

    sparkle
      .transition()
      .duration(500)
      .attr('r', 15)
      .style('opacity', 0)
      .on('end', function() {
        d3.select(this).remove()
      })
  }
}

/**
 * Control methods
 */
function updateLayout() {
  if (!simulation) return

  // Update force simulation based on layout type
  switch (layoutType.value) {
    case 'circular':
      simulation.force('center', null)
      simulation.force('radial', d3.forceRadial(150, props.width / 2, props.height / 2))
      break
    case 'hierarchical':
      simulation.force('center', null)
      simulation.force('y', d3.forceY().y((d: any) => d.type === 'agent' ? props.height / 3 : 2 * props.height / 3))
      break
    default: // force
      simulation.force('radial', null)
      simulation.force('y', null)
      simulation.force('center', d3.forceCenter(props.width / 2, props.height / 2))
  }

  simulation.alpha(1).restart()
  emit('layoutChanged', layoutType.value)
}

function toggleSemanticFlow() {
  showSemanticFlow.value = !showSemanticFlow.value
  
  if (showSemanticFlow.value) {
    loadConstellationData() // Reload with semantic flow data
  } else {
    semanticFlowGroup?.selectAll('*').remove()
  }
}

function resetView() {
  if (svg) {
    svg.transition().duration(750).call(
      d3.zoom<SVGSVGElement, unknown>().transform,
      d3.zoomIdentity
    )
  }
  
  if (simulation) {
    simulation.alpha(1).restart()
  }
}

function updateTimeRange() {
  loadConstellationData()
}

/**
 * Auto-refresh functionality
 */
let refreshInterval: number | null = null

function startAutoRefresh() {
  refreshInterval = setInterval(() => {
    if (!loading.value) {
      loadConstellationData()
    }
  }, 30000) // Refresh every 30 seconds
}

function stopAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

/**
 * Utility methods
 */
function formatTime(timestamp: string): string {
  return formatDistanceToNow(new Date(timestamp), { addSuffix: true })
}

// Watch for prop changes
watch(() => props.sessionIds, () => {
  if (subscriptionId) {
    observabilityEvents.updateSubscriptionFilters(subscriptionId, {
      session_ids: props.sessionIds.length ? props.sessionIds : undefined
    })
  }
  loadConstellationData()
})

watch(() => props.agentIds, () => {
  if (subscriptionId) {
    observabilityEvents.updateSubscriptionFilters(subscriptionId, {
      agent_ids: props.agentIds.length ? props.agentIds : undefined
    })
  }
  loadConstellationData()
})
</script>

<style scoped>
.live-workflow-constellation {
  @apply w-full;
}

.constellation-container {
  position: relative;
  overflow: hidden;
}

.constellation-container svg {
  background: radial-gradient(circle at center, rgba(59, 130, 246, 0.05) 0%, transparent 70%);
}

/* Dark mode adjustments */
.dark .constellation-container svg {
  background: radial-gradient(circle at center, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
}

/* Node animations */
.constellation-node {
  transition: all 0.3s ease;
}

.constellation-node:hover {
  transform: scale(1.1);
}

.constellation-edge {
  transition: all 0.3s ease;
}

/* Pulse animation for active nodes */
@keyframes pulse {
  0% {
    transform: scale(1);
    opacity: 1;
  }
  100% {
    transform: scale(1.5);
    opacity: 0;
  }
}

.node-pulse {
  animation: pulse 2s infinite;
}

/* Semantic flow particles */
.semantic-flow-layer circle {
  filter: drop-shadow(0 0 4px rgba(245, 158, 11, 0.5));
}

/* Grid lines */
.grid-lines line {
  stroke: rgba(148, 163, 184, 0.1);
  stroke-width: 1;
}

.dark .grid-lines line {
  stroke: rgba(148, 163, 184, 0.05);
}
</style>