/**
 * Real-time Agent Graph Updates Composable
 * 
 * Handles efficient real-time updates to the agent graph visualization
 * with optimized rendering and event processing.
 */

import { ref, reactive, onMounted, onUnmounted, nextTick } from 'vue'
import { useEventsStore } from '@/stores/events'
import type { HookEvent, AgentInfo, SessionInfo } from '@/types/hooks'

export interface GraphNode extends AgentInfo {
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
  lastUpdate: string
  recentEvents: HookEvent[]
  networkLoad: number
  securityLevel: 'safe' | 'warning' | 'danger'
}

export interface GraphLink {
  source: GraphNode
  target: GraphNode
  strength: number
  type: 'communication' | 'collaboration' | 'dependency' | 'data_flow'
  timestamp: string
  eventCount: number
  latency: number
  status: 'active' | 'idle' | 'error'
}

export interface GraphMetrics {
  totalNodes: number
  totalLinks: number
  activeSessions: number
  eventRate: number
  averageLatency: number
  errorRate: number
  networkDensity: number
  clusteringCoefficient: number
}

export interface UpdateEvent {
  type: 'node_added' | 'node_removed' | 'node_updated' | 'link_added' | 'link_removed' | 'link_updated'
  timestamp: string
  data: any
  priority: 'low' | 'medium' | 'high' | 'critical'
}

export function useAgentGraphRealtime() {
  const eventsStore = useEventsStore()
  
  // Reactive state
  const nodes = ref<Map<string, GraphNode>>(new Map())
  const links = ref<Map<string, GraphLink>>(new Map())
  const metrics = reactive<GraphMetrics>({
    totalNodes: 0,
    totalLinks: 0,
    activeSessions: 0,
    eventRate: 0,
    averageLatency: 0,
    errorRate: 0,
    networkDensity: 0,
    clusteringCoefficient: 0
  })
  
  const updateQueue = ref<UpdateEvent[]>([])
  const isProcessingUpdates = ref(false)
  const performanceStats = reactive({
    updateCount: 0,
    averageUpdateTime: 0,
    droppedUpdates: 0,
    renderFrames: 0,
    lastFrameTime: 0
  })
  
  // Configuration
  const config = reactive({
    maxUpdateQueueSize: 1000,
    batchUpdateInterval: 50, // ms
    maxBatchSize: 10,
    performanceMonitoringInterval: 1000,
    enableAdaptiveThrottling: true,
    lowPerformanceThreshold: 30, // fps
    adaptiveUpdateInterval: 100
  })
  
  // Performance monitoring
  let updateTimer: number | null = null
  let performanceTimer: number | null = null
  let lastUpdateTime = Date.now()
  let frameCount = 0
  let lastFrameCheck = Date.now()
  
  /**
   * Initialize the real-time graph system
   */
  const initialize = () => {
    console.log('Initializing real-time agent graph system')
    
    // Load initial data
    loadInitialData()
    
    // Set up event listeners
    setupEventListeners()
    
    // Start update processing
    startUpdateProcessing()
    
    // Start performance monitoring
    startPerformanceMonitoring()
  }
  
  /**
   * Load initial graph data from the events store
   */
  const loadInitialData = () => {
    // Clear existing data
    nodes.value.clear()
    links.value.clear()
    
    // Load agents as nodes
    eventsStore.agents.forEach(agent => {
      const node = createGraphNode(agent)
      nodes.value.set(agent.agent_id, node)
    })
    
    // Generate links based on agent interactions
    generateInitialLinks()
    
    // Update metrics
    updateMetrics()
    
    console.log(`Loaded ${nodes.value.size} nodes and ${links.value.size} links`)
  }
  
  /**
   * Create a graph node from agent info
   */
  const createGraphNode = (agent: AgentInfo): GraphNode => {
    const recentEvents = eventsStore.filteredHookEvents
      .filter(event => event.agent_id === agent.agent_id)
      .slice(0, 10)
    
    return {
      ...agent,
      name: `Agent-${agent.agent_id.slice(-4)}`,
      performance: calculateAgentPerformance(agent, recentEvents),
      memoryUsage: Math.floor(Math.random() * 200) + 50, // Mock for now
      isActive: agent.status === 'active',
      currentActivity: agent.status === 'active' 
        ? inferCurrentActivity(recentEvents)
        : undefined,
      connections: [],
      lastUpdate: new Date().toISOString(),
      recentEvents,
      networkLoad: calculateNetworkLoad(agent, recentEvents),
      securityLevel: calculateSecurityLevel(agent)
    }
  }
  
  /**
   * Generate initial links between nodes
   */
  const generateInitialLinks = () => {
    const nodeArray = Array.from(nodes.value.values())
    
    for (let i = 0; i < nodeArray.length; i++) {
      for (let j = i + 1; j < nodeArray.length; j++) {
        const source = nodeArray[i]
        const target = nodeArray[j]
        
        const linkStrength = calculateLinkStrength(source, target)
        if (linkStrength > 0.1) {
          const linkId = `${source.agent_id}-${target.agent_id}`
          const link = createGraphLink(source, target, linkStrength)
          links.value.set(linkId, link)
          
          // Update node connections
          source.connections.push(target.agent_id)
          target.connections.push(source.agent_id)
        }
      }
    }
  }
  
  /**
   * Create a graph link between two nodes
   */
  const createGraphLink = (source: GraphNode, target: GraphNode, strength: number): GraphLink => {
    const sharedEvents = getSharedEvents(source, target)
    const latency = calculateAverageLatency(sharedEvents)
    
    return {
      source,
      target,
      strength,
      type: inferLinkType(source, target, sharedEvents),
      timestamp: new Date().toISOString(),
      eventCount: sharedEvents.length,
      latency,
      status: latency < 100 ? 'active' : latency < 500 ? 'idle' : 'error'
    }
  }
  
  /**
   * Set up WebSocket event listeners
   */
  const setupEventListeners = () => {
    // Listen for new hook events
    eventsStore.onEvent((event: HookEvent) => {
      queueUpdate({
        type: 'node_updated',
        timestamp: event.timestamp,
        data: { event, agentId: event.agent_id },
        priority: 'medium'
      })
    })
    
    // Listen for agent status changes
    // This would be implemented based on the actual event structure
    // For now, we'll simulate it by watching the agents array
  }
  
  /**
   * Queue an update for batch processing
   */
  const queueUpdate = (update: UpdateEvent) => {
    if (updateQueue.value.length >= config.maxUpdateQueueSize) {
      console.warn('Update queue full, dropping oldest updates')
      updateQueue.value.shift()
      performanceStats.droppedUpdates++
    }
    
    updateQueue.value.push(update)
    
    // Process immediately for critical updates
    if (update.priority === 'critical') {
      processUpdateQueue()
    }
  }
  
  /**
   * Start the update processing loop
   */
  const startUpdateProcessing = () => {
    updateTimer = window.setInterval(() => {
      if (updateQueue.value.length > 0 && !isProcessingUpdates.value) {
        processUpdateQueue()
      }
    }, config.batchUpdateInterval)
  }
  
  /**
   * Process queued updates in batches
   */
  const processUpdateQueue = async () => {
    if (isProcessingUpdates.value || updateQueue.value.length === 0) {
      return
    }
    
    isProcessingUpdates.value = true
    const startTime = performance.now()
    
    try {
      // Process updates in batches
      const batchSize = Math.min(config.maxBatchSize, updateQueue.value.length)
      const batch = updateQueue.value.splice(0, batchSize)
      
      for (const update of batch) {
        await processUpdate(update)
      }
      
      // Update metrics after processing batch
      updateMetrics()
      
      // Update performance stats
      const processingTime = performance.now() - startTime
      performanceStats.updateCount += batch.length
      performanceStats.averageUpdateTime = 
        (performanceStats.averageUpdateTime + processingTime) / 2
      
    } catch (error) {
      console.error('Error processing update queue:', error)
    } finally {
      isProcessingUpdates.value = false
    }
  }
  
  /**
   * Process a single update
   */
  const processUpdate = async (update: UpdateEvent) => {
    switch (update.type) {
      case 'node_updated':
        await updateNode(update.data)
        break
      case 'node_added':
        await addNode(update.data)
        break
      case 'node_removed':
        await removeNode(update.data)
        break
      case 'link_updated':
        await updateLink(update.data)
        break
      case 'link_added':
        await addLink(update.data)
        break
      case 'link_removed':
        await removeLink(update.data)
        break
    }
  }
  
  /**
   * Update an existing node
   */
  const updateNode = async (data: { event: HookEvent; agentId: string }) => {
    const node = nodes.value.get(data.agentId)
    if (!node) return
    
    // Update node properties based on event
    node.recentEvents.unshift(data.event)
    if (node.recentEvents.length > 10) {
      node.recentEvents = node.recentEvents.slice(0, 10)
    }
    
    node.performance = calculateAgentPerformance(node, node.recentEvents)
    node.networkLoad = calculateNetworkLoad(node, node.recentEvents)
    node.securityLevel = calculateSecurityLevel(node)
    node.isActive = node.status === 'active'
    node.currentActivity = node.isActive ? inferCurrentActivity(node.recentEvents) : undefined
    node.lastUpdate = data.event.timestamp
    
    // Update related links
    await updateNodeLinks(node)
  }
  
  /**
   * Update links related to a node
   */
  const updateNodeLinks = async (node: GraphNode) => {
    const nodeLinks = Array.from(links.value.values()).filter(link =>
      link.source.agent_id === node.agent_id || link.target.agent_id === node.agent_id
    )
    
    for (const link of nodeLinks) {
      const sharedEvents = getSharedEvents(link.source, link.target)
      link.eventCount = sharedEvents.length
      link.latency = calculateAverageLatency(sharedEvents)
      link.status = link.latency < 100 ? 'active' : link.latency < 500 ? 'idle' : 'error'
      link.timestamp = new Date().toISOString()
    }
  }
  
  /**
   * Add a new node to the graph
   */
  const addNode = async (agentInfo: AgentInfo) => {
    const node = createGraphNode(agentInfo)
    nodes.value.set(agentInfo.agent_id, node)
    
    // Generate links to existing nodes
    await generateLinksForNewNode(node)
    
    console.log(`Added new node: ${node.agent_id}`)
  }
  
  /**
   * Remove a node from the graph
   */
  const removeNode = async (agentId: string) => {
    const node = nodes.value.get(agentId)
    if (!node) return
    
    // Remove all links involving this node
    const linksToRemove = Array.from(links.value.entries()).filter(([_, link]) =>
      link.source.agent_id === agentId || link.target.agent_id === agentId
    )
    
    for (const [linkId] of linksToRemove) {
      links.value.delete(linkId)
    }
    
    // Remove from other nodes' connections
    nodes.value.forEach(otherNode => {
      otherNode.connections = otherNode.connections.filter(id => id !== agentId)
    })
    
    nodes.value.delete(agentId)
    console.log(`Removed node: ${agentId}`)
  }
  
  /**
   * Generate links for a new node
   */
  const generateLinksForNewNode = async (newNode: GraphNode) => {
    for (const [_, existingNode] of nodes.value) {
      if (existingNode.agent_id === newNode.agent_id) continue
      
      const linkStrength = calculateLinkStrength(newNode, existingNode)
      if (linkStrength > 0.1) {
        const linkId = `${newNode.agent_id}-${existingNode.agent_id}`
        const link = createGraphLink(newNode, existingNode, linkStrength)
        links.value.set(linkId, link)
        
        newNode.connections.push(existingNode.agent_id)
        existingNode.connections.push(newNode.agent_id)
      }
    }
  }
  
  /**
   * Update graph metrics
   */
  const updateMetrics = () => {
    metrics.totalNodes = nodes.value.size
    metrics.totalLinks = links.value.size
    
    // Calculate active sessions
    const sessions = new Set()
    nodes.value.forEach(node => {
      node.session_ids.forEach(sessionId => sessions.add(sessionId))
    })
    metrics.activeSessions = sessions.size
    
    // Calculate event rate (events per second in last minute)
    const now = Date.now()
    const oneMinuteAgo = now - 60000
    let recentEventCount = 0
    
    nodes.value.forEach(node => {
      recentEventCount += node.recentEvents.filter(event =>
        new Date(event.timestamp).getTime() > oneMinuteAgo
      ).length
    })
    
    metrics.eventRate = recentEventCount / 60
    
    // Calculate average latency
    const activeLinks = Array.from(links.value.values()).filter(link => link.status !== 'error')
    metrics.averageLatency = activeLinks.length > 0
      ? activeLinks.reduce((sum, link) => sum + link.latency, 0) / activeLinks.length
      : 0
    
    // Calculate error rate
    const errorLinks = Array.from(links.value.values()).filter(link => link.status === 'error')
    metrics.errorRate = links.value.size > 0 ? errorLinks.length / links.value.size : 0
    
    // Calculate network density
    const maxPossibleLinks = (nodes.value.size * (nodes.value.size - 1)) / 2
    metrics.networkDensity = maxPossibleLinks > 0 ? links.value.size / maxPossibleLinks : 0
    
    // Calculate clustering coefficient (simplified)
    metrics.clusteringCoefficient = calculateClusteringCoefficient()
  }
  
  /**
   * Start performance monitoring
   */
  const startPerformanceMonitoring = () => {
    performanceTimer = window.setInterval(() => {
      const now = Date.now()
      const timeDelta = now - lastFrameCheck
      
      if (timeDelta >= 1000) {
        const fps = (frameCount * 1000) / timeDelta
        performanceStats.lastFrameTime = fps
        
        // Adaptive throttling based on performance
        if (config.enableAdaptiveThrottling) {
          if (fps < config.lowPerformanceThreshold) {
            config.batchUpdateInterval = Math.min(config.batchUpdateInterval * 1.2, 200)
            config.maxBatchSize = Math.max(config.maxBatchSize - 1, 1)
          } else if (fps > config.lowPerformanceThreshold * 1.5) {
            config.batchUpdateInterval = Math.max(config.batchUpdateInterval * 0.9, 16)
            config.maxBatchSize = Math.min(config.maxBatchSize + 1, 20)
          }
        }
        
        frameCount = 0
        lastFrameCheck = now
      }
      
      frameCount++
      performanceStats.renderFrames++
    }, config.performanceMonitoringInterval)
  }
  
  // Utility functions
  const calculateAgentPerformance = (agent: AgentInfo, events: HookEvent[]): number => {
    if (events.length === 0) return 50
    
    const errorEvents = events.filter(e => e.hook_type === 'Error').length
    const successfulEvents = events.length - errorEvents
    const errorRate = errorEvents / events.length
    
    // Base performance on error rate and activity
    let performance = Math.max(0, 100 - (errorRate * 100))
    
    // Adjust based on recent activity
    const recentActivityBonus = Math.min(events.length * 2, 20)
    performance = Math.min(100, performance + recentActivityBonus)
    
    return Math.round(performance)
  }
  
  const calculateNetworkLoad = (agent: AgentInfo, events: HookEvent[]): number => {
    const recentEvents = events.filter(e => 
      Date.now() - new Date(e.timestamp).getTime() < 60000 // Last minute
    )
    return Math.min(100, recentEvents.length * 2)
  }
  
  const calculateSecurityLevel = (agent: AgentInfo): 'safe' | 'warning' | 'danger' => {
    if (agent.blocked_count > 5) return 'danger'
    if (agent.blocked_count > 0) return 'warning'
    return 'safe'
  }
  
  const inferCurrentActivity = (events: HookEvent[]): string | undefined => {
    if (events.length === 0) return undefined
    
    const recentEvent = events[0]
    if (recentEvent.hook_type === 'PreToolUse') {
      return `Using ${recentEvent.payload.tool_name || 'tool'}`
    }
    if (recentEvent.hook_type === 'PostToolUse') {
      return `Completed ${recentEvent.payload.tool_name || 'tool'}`
    }
    
    return 'Processing tasks'
  }
  
  const calculateLinkStrength = (source: GraphNode, target: GraphNode): number => {
    // Calculate based on shared sessions
    const sharedSessions = source.session_ids.filter(sid => 
      target.session_ids.includes(sid)
    ).length
    
    if (sharedSessions > 0) {
      return Math.min(1, sharedSessions * 0.3 + 0.2)
    }
    
    // Random weak connection for visualization
    return Math.random() < 0.1 ? Math.random() * 0.3 : 0
  }
  
  const getSharedEvents = (source: GraphNode, target: GraphNode): HookEvent[] => {
    const sharedSessions = source.session_ids.filter(sid => 
      target.session_ids.includes(sid)
    )
    
    return eventsStore.filteredHookEvents.filter(event =>
      sharedSessions.includes(event.session_id || '')
    )
  }
  
  const calculateAverageLatency = (events: HookEvent[]): number => {
    if (events.length === 0) return 0
    
    // Mock latency calculation based on event processing
    const latencies = events
      .filter(e => e.payload.execution_time_ms)
      .map(e => e.payload.execution_time_ms)
    
    if (latencies.length === 0) return Math.random() * 200 + 50 // Mock latency
    
    return latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length
  }
  
  const inferLinkType = (source: GraphNode, target: GraphNode, events: HookEvent[]): GraphLink['type'] => {
    const toolEvents = events.filter(e => e.hook_type === 'PreToolUse' || e.hook_type === 'PostToolUse')
    
    if (toolEvents.length > 5) return 'data_flow'
    if (toolEvents.length > 2) return 'collaboration'
    if (events.length > 0) return 'communication'
    return 'dependency'
  }
  
  const calculateClusteringCoefficient = (): number => {
    // Simplified clustering coefficient calculation
    let totalCoefficient = 0
    let nodeCount = 0
    
    nodes.value.forEach(node => {
      const neighbors = node.connections
      if (neighbors.length < 2) return
      
      let actualConnections = 0
      const maxPossibleConnections = (neighbors.length * (neighbors.length - 1)) / 2
      
      for (let i = 0; i < neighbors.length; i++) {
        for (let j = i + 1; j < neighbors.length; j++) {
          const linkId1 = `${neighbors[i]}-${neighbors[j]}`
          const linkId2 = `${neighbors[j]}-${neighbors[i]}`
          if (links.value.has(linkId1) || links.value.has(linkId2)) {
            actualConnections++
          }
        }
      }
      
      if (maxPossibleConnections > 0) {
        totalCoefficient += actualConnections / maxPossibleConnections
        nodeCount++
      }
    })
    
    return nodeCount > 0 ? totalCoefficient / nodeCount : 0
  }
  
  // Cleanup
  const cleanup = () => {
    if (updateTimer) {
      clearInterval(updateTimer)
      updateTimer = null
    }
    
    if (performanceTimer) {
      clearInterval(performanceTimer)
      performanceTimer = null
    }
    
    // Remove event listeners
    // This would be implemented based on the actual events store API
  }
  
  // Lifecycle
  onMounted(() => {
    initialize()
  })
  
  onUnmounted(() => {
    cleanup()
  })
  
  return {
    // Reactive data
    nodes: nodes,
    links: links,
    metrics,
    performanceStats,
    updateQueue,
    isProcessingUpdates,
    
    // Configuration
    config,
    
    // Methods
    initialize,
    loadInitialData,
    queueUpdate,
    updateMetrics,
    cleanup,
    
    // Computed getters
    getNodes: () => Array.from(nodes.value.values()),
    getLinks: () => Array.from(links.value.values()),
    getNodeById: (id: string) => nodes.value.get(id),
    getLinkById: (id: string) => links.value.get(id),
    
    // Utility methods
    calculateAgentPerformance,
    calculateNetworkLoad,
    calculateSecurityLevel
  }
}