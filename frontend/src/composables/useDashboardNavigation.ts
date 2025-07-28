/**
 * Dashboard Navigation Composable
 * 
 * Manages cross-component navigation, persistent state, and context-aware
 * transitions between different dashboard views with intelligent data correlation.
 */

import { ref, reactive, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import type { 
  DashboardComponent,
  GraphNode,
  ConversationEvent,
  DetectedPattern,
  AgentInfo
} from '@/types/coordination'

export interface NavigationContext {
  sourceComponent: DashboardComponent
  targetComponent: DashboardComponent
  correlationData: {
    agentId?: string
    sessionId?: string
    eventId?: string
    patternId?: string
    timestamp?: string
    [key: string]: any
  }
  intent: NavigationIntent
  preserveFilters: boolean
}

export enum NavigationIntent {
  EXPLORE = 'explore',
  INVESTIGATE = 'investigate',  
  DEBUG = 'debug',
  ANALYZE = 'analyze',
  MONITOR = 'monitor',
  CORRELATE = 'correlate'
}

export interface TabState {
  id: string
  component: DashboardComponent
  isActive: boolean
  hasData: boolean
  lastUpdated: Date | null
  context: Record<string, any>
  scrollPosition?: number
  filterState?: Record<string, any>
}

export interface NavigationBreadcrumb {
  label: string
  component: DashboardComponent
  context?: Record<string, any>
  onClick?: () => void
}

export interface NavigationHistory {
  component: DashboardComponent
  context: NavigationContext
  timestamp: Date
  description: string
}

class DashboardNavigationManager {
  // Navigation state
  private state = reactive({
    activeTab: 'graph' as string,
    previousTab: null as string | null,
    navigationHistory: [] as NavigationHistory[],
    breadcrumbs: [] as NavigationBreadcrumb[],
    tabs: new Map<string, TabState>(),
    crossComponentContext: {} as Record<string, any>,
    pendingNavigation: null as NavigationContext | null
  })

  // Tab definitions
  private tabDefinitions = new Map([
    ['graph', { 
      id: 'graph', 
      component: DashboardComponent.GRAPH, 
      label: 'Agent Graph',
      description: 'Visual representation of agent interactions'
    }],
    ['transcript', { 
      id: 'transcript', 
      component: DashboardComponent.TRANSCRIPT, 
      label: 'Communications',
      description: 'Agent-to-agent communication transcript'
    }],
    ['analysis', { 
      id: 'analysis', 
      component: DashboardComponent.ANALYSIS, 
      label: 'Analysis',
      description: 'Pattern detection and debugging tools'
    }],
    ['monitoring', { 
      id: 'monitoring', 
      component: DashboardComponent.MONITORING, 
      label: 'Monitoring',
      description: 'System health and performance metrics'
    }]
  ])

  // Event listeners for cross-component communication
  private navigationListeners = new Map<string, Array<(context: NavigationContext) => void>>()

  constructor() {
    this.initializeTabs()
    this.initializeRouteWatching()
  }

  // Public reactive state
  public readonly activeTab = computed(() => this.state.activeTab)
  public readonly previousTab = computed(() => this.state.previousTab)
  public readonly navigationHistory = computed(() => this.state.navigationHistory)
  public readonly breadcrumbs = computed(() => this.state.breadcrumbs)
  public readonly tabs = computed(() => Array.from(this.state.tabs.values()))
  public readonly crossComponentContext = computed(() => this.state.crossComponentContext)

  /**
   * Navigate to a specific tab with context
   */
  public navigateToTab(
    tabId: string, 
    context?: Partial<NavigationContext>,
    updateUrl = true
  ): void {
    const targetTab = this.tabDefinitions.get(tabId)
    if (!targetTab) {
      console.error(`Unknown tab: ${tabId}`)
      return
    }

    const navigationContext: NavigationContext = {
      sourceComponent: this.getComponentForTab(this.state.activeTab),
      targetComponent: targetTab.component,
      correlationData: context?.correlationData || {},
      intent: context?.intent || NavigationIntent.EXPLORE,
      preserveFilters: context?.preserveFilters ?? true
    }

    // Store current tab state before switching
    this.preserveTabState(this.state.activeTab)

    // Update active tab
    this.state.previousTab = this.state.activeTab
    this.state.activeTab = tabId

    // Update tab state
    const tabState = this.state.tabs.get(tabId)
    if (tabState) {
      tabState.isActive = true
      tabState.context = { ...tabState.context, ...navigationContext.correlationData }
    }

    // Deactivate previous tab
    if (this.state.previousTab) {
      const prevTabState = this.state.tabs.get(this.state.previousTab)
      if (prevTabState) {
        prevTabState.isActive = false
      }
    }

    // Update cross-component context
    this.updateCrossComponentContext(navigationContext)

    // Add to navigation history
    this.addToNavigationHistory(navigationContext)

    // Update breadcrumbs
    this.updateBreadcrumbs(navigationContext)

    // Emit navigation event
    this.emitNavigationEvent('tab_changed', navigationContext)

    // Update URL if requested
    if (updateUrl) {
      this.updateUrlForNavigation(tabId, navigationContext)
    }

    console.log(`Navigated to ${tabId} with context:`, navigationContext)
  }

  /**
   * Navigate with specific intent and data correlation
   */
  public navigateWithIntent(
    intent: NavigationIntent,
    data: any,
    targetComponent?: DashboardComponent
  ): void {
    const targetTab = this.determineTargetTabForIntent(intent, data, targetComponent)
    if (!targetTab) {
      console.error(`Could not determine target tab for intent: ${intent}`)
      return
    }

    const context: Partial<NavigationContext> = {
      intent,
      correlationData: this.extractCorrelationData(data),
      preserveFilters: intent === NavigationIntent.INVESTIGATE || intent === NavigationIntent.DEBUG
    }

    this.navigateToTab(targetTab, context)
  }

  /**
   * Navigate from graph node selection
   */
  public navigateFromGraphNode(node: GraphNode, intent: NavigationIntent = NavigationIntent.EXPLORE): void {
    const correlationData = {
      agentId: node.id,
      sessionId: node.metadata.session_id,
      nodeType: node.type,
      timestamp: node.last_updated
    }

    switch (intent) {
      case NavigationIntent.INVESTIGATE:
        // Go to transcript to see agent communications
        this.navigateToTab('transcript', {
          intent,
          correlationData,
          preserveFilters: false
        })
        break

      case NavigationIntent.DEBUG:
        // Go to analysis for debugging tools
        this.navigateToTab('analysis', {
          intent,
          correlationData,
          preserveFilters: true
        })
        break

      case NavigationIntent.MONITOR:
        // Go to monitoring for performance data
        this.navigateToTab('monitoring', {
          intent,
          correlationData,
          preserveFilters: false
        })
        break

      default:
        // Default exploration behavior
        this.navigateToTab('transcript', {
          intent: NavigationIntent.EXPLORE,
          correlationData,
          preserveFilters: false
        })
    }
  }

  /**
   * Navigate from transcript event selection  
   */
  public navigateFromTranscriptEvent(
    event: ConversationEvent, 
    intent: NavigationIntent = NavigationIntent.EXPLORE
  ): void {
    const correlationData = {
      eventId: event.id,
      agentId: event.source_agent,
      sessionId: event.session_id,
      eventType: event.event_type,
      timestamp: event.timestamp
    }

    switch (intent) {
      case NavigationIntent.INVESTIGATE:
        // Go to graph to see agent relationships
        this.navigateToTab('graph', {
          intent,
          correlationData,
          preserveFilters: false
        })
        break

      case NavigationIntent.DEBUG:
        // Go to analysis for pattern investigation
        this.navigateToTab('analysis', {
          intent,
          correlationData,
          preserveFilters: true
        })
        break

      case NavigationIntent.ANALYZE:
        // Stay in analysis but update context
        this.navigateToTab('analysis', {
          intent,
          correlationData,
          preserveFilters: true
        })
        break

      default:
        // Default exploration
        this.navigateToTab('graph', {
          intent: NavigationIntent.EXPLORE,
          correlationData,
          preserveFilters: false
        })
    }
  }

  /**
   * Navigate from detected pattern
   */
  public navigateFromPattern(
    pattern: DetectedPattern,
    intent: NavigationIntent = NavigationIntent.INVESTIGATE
  ): void {
    const correlationData = {
      patternId: pattern.id,
      patternType: pattern.pattern_type,
      affectedAgents: pattern.affectedAgents,
      severity: pattern.severity
    }

    switch (intent) {
      case NavigationIntent.INVESTIGATE:
        // Go to transcript to see related communications
        this.navigateToTab('transcript', {
          intent,
          correlationData: {
            ...correlationData,
            agentIds: pattern.affectedAgents
          },
          preserveFilters: false
        })
        break

      case NavigationIntent.DEBUG:
        // Stay in analysis for debugging
        this.navigateToTab('analysis', {
          intent,
          correlationData,
          preserveFilters: true
        })
        break

      default:
        // Show in graph visualization
        this.navigateToTab('graph', {
          intent: NavigationIntent.EXPLORE,
          correlationData,
          preserveFilters: false
        })
    }
  }

  /**
   * Get navigation context for current tab
   */
  public getCurrentContext(): Record<string, any> {
    const tabState = this.state.tabs.get(this.state.activeTab)
    return tabState?.context || {}
  }

  /**
   * Update context for current tab
   */
  public updateCurrentContext(context: Record<string, any>): void {
    const tabState = this.state.tabs.get(this.state.activeTab)
    if (tabState) {
      tabState.context = { ...tabState.context, ...context }
      tabState.lastUpdated = new Date()
    }

    // Update cross-component context
    this.state.crossComponentContext = { ...this.state.crossComponentContext, ...context }
  }

  /**
   * Go back to previous tab
   */
  public goBack(): void {
    if (this.state.previousTab) {
      this.navigateToTab(this.state.previousTab, undefined, false)
    } else if (this.state.navigationHistory.length > 1) {
      const previousNavigation = this.state.navigationHistory[this.state.navigationHistory.length - 2]
      const tabId = this.getTabForComponent(previousNavigation.component)
      if (tabId) {
        this.navigateToTab(tabId, undefined, false)
      }
    }
  }

  /**
   * Clear navigation history
   */
  public clearHistory(): void {
    this.state.navigationHistory = []
    this.state.breadcrumbs = []
  }

  /**
   * Register navigation event listener
   */
  public onNavigation(
    eventType: string,
    handler: (context: NavigationContext) => void
  ): () => void {
    if (!this.navigationListeners.has(eventType)) {
      this.navigationListeners.set(eventType, [])
    }

    const listeners = this.navigationListeners.get(eventType)!
    listeners.push(handler)

    // Return unsubscribe function
    return () => {
      const index = listeners.indexOf(handler)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }

  /**
   * Get tab state
   */
  public getTabState(tabId: string): TabState | undefined {
    return this.state.tabs.get(tabId)
  }

  /**
   * Update tab data status
   */
  public updateTabDataStatus(tabId: string, hasData: boolean): void {
    const tabState = this.state.tabs.get(tabId)
    if (tabState) {
      tabState.hasData = hasData
      tabState.lastUpdated = new Date()
    }
  }

  /**
   * Set scroll position for tab
   */
  public setTabScrollPosition(tabId: string, position: number): void {
    const tabState = this.state.tabs.get(tabId)
    if (tabState) {
      tabState.scrollPosition = position
    }
  }

  /**
   * Get scroll position for tab
   */
  public getTabScrollPosition(tabId: string): number {
    const tabState = this.state.tabs.get(tabId)
    return tabState?.scrollPosition || 0
  }

  // Private methods

  private initializeTabs(): void {
    for (const [tabId, definition] of this.tabDefinitions) {
      this.state.tabs.set(tabId, {
        id: tabId,
        component: definition.component,
        isActive: tabId === this.state.activeTab,
        hasData: false,
        lastUpdated: null,
        context: {},
        scrollPosition: 0
      })
    }
  }

  private initializeRouteWatching(): void {
    // Watch for URL changes and update active tab
    // This would be implemented with Vue Router in a real application
  }

  private preserveTabState(tabId: string): void {
    const tabState = this.state.tabs.get(tabId)
    if (tabState) {
      // Store any transient state that should be preserved
      // This could include scroll position, form data, etc.
    }
  }

  private updateCrossComponentContext(context: NavigationContext): void {
    // Merge correlation data into shared context
    this.state.crossComponentContext = {
      ...this.state.crossComponentContext,
      ...context.correlationData,
      lastNavigation: {
        from: context.sourceComponent,
        to: context.targetComponent,
        intent: context.intent,
        timestamp: new Date().toISOString()
      }
    }
  }

  private addToNavigationHistory(context: NavigationContext): void {
    const historyEntry: NavigationHistory = {
      component: context.targetComponent,
      context,
      timestamp: new Date(),
      description: this.generateNavigationDescription(context)
    }

    this.state.navigationHistory.push(historyEntry)

    // Keep only last 50 entries
    if (this.state.navigationHistory.length > 50) {
      this.state.navigationHistory.shift()
    }
  }

  private updateBreadcrumbs(context: NavigationContext): void {
    const targetDefinition = Array.from(this.tabDefinitions.values())
      .find(def => def.component === context.targetComponent)

    if (!targetDefinition) return

    // Build breadcrumb trail
    const breadcrumbs: NavigationBreadcrumb[] = [
      {
        label: 'Dashboard',
        component: DashboardComponent.SERVICE,
        onClick: () => this.navigateToTab('graph')
      }
    ]

    // Add context-specific breadcrumbs
    if (context.correlationData.sessionId) {
      breadcrumbs.push({
        label: `Session ${context.correlationData.sessionId.substring(0, 8)}`,
        component: DashboardComponent.SERVICE,
        context: { sessionId: context.correlationData.sessionId }
      })
    }

    if (context.correlationData.agentId) {
      breadcrumbs.push({
        label: `Agent ${context.correlationData.agentId.substring(0, 8)}`,
        component: DashboardComponent.GRAPH,
        context: { agentId: context.correlationData.agentId },
        onClick: () => this.navigateToTab('graph', {
          correlationData: { agentId: context.correlationData.agentId }
        })
      })
    }

    // Add current tab
    breadcrumbs.push({
      label: targetDefinition.label,
      component: context.targetComponent
    })

    this.state.breadcrumbs = breadcrumbs
  }

  private emitNavigationEvent(eventType: string, context: NavigationContext): void {
    const listeners = this.navigationListeners.get(eventType) || []
    const allListeners = this.navigationListeners.get('*') || []
    const combinedListeners = [...listeners, ...allListeners]

    combinedListeners.forEach(listener => {
      try {
        listener(context)
      } catch (listenerError) {
        console.error('Error in navigation event listener:', listenerError)
      }
    })
  }

  private updateUrlForNavigation(tabId: string, context: NavigationContext): void {
    // Update browser URL to reflect navigation state
    // This would be implemented with Vue Router in a real application
    const params = new URLSearchParams()
    
    if (context.correlationData.sessionId) {
      params.set('session', context.correlationData.sessionId)
    }
    
    if (context.correlationData.agentId) {
      params.set('agent', context.correlationData.agentId)
    }
    
    const url = `/coordination/${tabId}${params.toString() ? '?' + params.toString() : ''}`
    
    // Update URL without triggering navigation
    if (typeof window !== 'undefined') {
      window.history.replaceState({}, '', url)
    }
  }

  private determineTargetTabForIntent(
    intent: NavigationIntent,
    data: any,
    targetComponent?: DashboardComponent
  ): string | null {
    if (targetComponent) {
      return this.getTabForComponent(targetComponent)
    }

    switch (intent) {
      case NavigationIntent.INVESTIGATE:
        return data.eventId ? 'analysis' : 'transcript'
      case NavigationIntent.DEBUG:
        return 'analysis'
      case NavigationIntent.MONITOR:
        return 'monitoring'
      case NavigationIntent.ANALYZE:
        return 'analysis'
      case NavigationIntent.CORRELATE:
        return data.agentId ? 'graph' : 'transcript'
      default:
        return 'graph'
    }
  }

  private extractCorrelationData(data: any): Record<string, any> {
    const correlationData: Record<string, any> = {}

    if (data.agent_id || data.agentId) {
      correlationData.agentId = data.agent_id || data.agentId
    }

    if (data.session_id || data.sessionId) {
      correlationData.sessionId = data.session_id || data.sessionId
    }

    if (data.id) {
      correlationData.eventId = data.id
    }

    if (data.timestamp) {
      correlationData.timestamp = data.timestamp
    }

    return correlationData
  }

  private getComponentForTab(tabId: string): DashboardComponent {
    const definition = this.tabDefinitions.get(tabId)
    return definition?.component || DashboardComponent.SERVICE
  }

  private getTabForComponent(component: DashboardComponent): string | null {
    for (const [tabId, definition] of this.tabDefinitions) {
      if (definition.component === component) {
        return tabId
      }
    }
    return null
  }

  private generateNavigationDescription(context: NavigationContext): string {
    const sourceLabel = this.getComponentLabel(context.sourceComponent)
    const targetLabel = this.getComponentLabel(context.targetComponent)
    
    let description = `${sourceLabel} → ${targetLabel}`
    
    if (context.correlationData.agentId) {
      description += ` (Agent: ${context.correlationData.agentId.substring(0, 8)})`
    }
    
    if (context.intent !== NavigationIntent.EXPLORE) {
      description += ` [${context.intent}]`
    }
    
    return description
  }

  private getComponentLabel(component: DashboardComponent): string {
    const labelsMap: Record<DashboardComponent, string> = {}
    labelsMap[DashboardComponent.GRAPH] = 'Graph'
    labelsMap[DashboardComponent.TRANSCRIPT] = 'Transcript'
    labelsMap[DashboardComponent.ANALYSIS] = 'Analysis'
    labelsMap[DashboardComponent.MONITORING] = 'Monitoring'
    labelsMap[DashboardComponent.SERVICE] = 'Service'
    
    return labelsMap[component] || 'Unknown'
  }
}

// Create singleton instance
const navigationManager = new DashboardNavigationManager()

// Vue composable
export function useDashboardNavigation() {
  return {
    // State
    activeTab: navigationManager.activeTab,
    previousTab: navigationManager.previousTab,
    navigationHistory: navigationManager.navigationHistory,
    breadcrumbs: navigationManager.breadcrumbs,
    tabs: navigationManager.tabs,
    crossComponentContext: navigationManager.crossComponentContext,

    // Navigation methods
    navigateToTab: navigationManager.navigateToTab.bind(navigationManager),
    navigateWithIntent: navigationManager.navigateWithIntent.bind(navigationManager),
    navigateFromGraphNode: navigationManager.navigateFromGraphNode.bind(navigationManager),
    navigateFromTranscriptEvent: navigationManager.navigateFromTranscriptEvent.bind(navigationManager),
    navigateFromPattern: navigationManager.navigateFromPattern.bind(navigationManager),
    goBack: navigationManager.goBack.bind(navigationManager),

    // Context management
    getCurrentContext: navigationManager.getCurrentContext.bind(navigationManager),
    updateCurrentContext: navigationManager.updateCurrentContext.bind(navigationManager),

    // State management
    getTabState: navigationManager.getTabState.bind(navigationManager),
    updateTabDataStatus: navigationManager.updateTabDataStatus.bind(navigationManager),
    setTabScrollPosition: navigationManager.setTabScrollPosition.bind(navigationManager),
    getTabScrollPosition: navigationManager.getTabScrollPosition.bind(navigationManager),

    // Events
    onNavigation: navigationManager.onNavigation.bind(navigationManager),

    // Utility
    clearHistory: navigationManager.clearHistory.bind(navigationManager),

    // Navigation intents enum
    NavigationIntent
  }
}

export default navigationManager