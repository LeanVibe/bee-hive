import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../services/websocket'
import { getAgentService } from '../services/agent'

interface PriorityItem {
  id: string
  type: 'task' | 'alert' | 'decision' | 'milestone' | 'performance' | 'error'
  priority: 'critical' | 'high' | 'medium' | 'low' | 'info'
  title: string
  description: string
  timestamp: string
  agentId?: string
  agentName?: string
  agentRole?: string
  data: any
  actions: PriorityAction[]
  tags: string[]
  estimatedTime?: number
  confidence?: number
  impact?: 'high' | 'medium' | 'low'
  urgency?: 'high' | 'medium' | 'low'
  category?: string
  relatedItems?: string[]
}

interface PriorityAction {
  id: string
  label: string
  command: string
  icon: string
  style: 'primary' | 'secondary' | 'danger' | 'success'
  requiresConfirmation?: boolean
  estimatedTime?: number
}

interface FilterOptions {
  types: string[]
  priorities: string[]
  agents: string[]
  timeRange: 'last-hour' | 'last-4-hours' | 'last-day' | 'all'
  showCompleted: boolean
  sortBy: 'priority' | 'timestamp' | 'impact' | 'urgency'
}

interface ContextualInsight {
  id: string
  type: 'pattern' | 'recommendation' | 'prediction' | 'optimization'
  title: string
  description: string
  confidence: number
  impact: 'high' | 'medium' | 'low'
  data: any
  actions: PriorityAction[]
}

@customElement('priority-information-hub')
export class PriorityInformationHub extends LitElement {
  @property({ type: Boolean }) declare mobile: boolean
  @property({ type: String }) declare focusMode: string

  @state() private declare priorityItems: PriorityItem[]
  @state() private declare contextualInsights: ContextualInsight[]
  @state() private declare filterOptions: FilterOptions
  @state() private declare activeFilter: string
  @state() private declare searchQuery: string
  @state() private declare selectedItem: PriorityItem | null
  @state() private declare viewMode: 'list' | 'kanban' | 'timeline' | 'matrix'
  @state() private declare autoRefresh: boolean
  @state() private declare showInsights: boolean
  @state() private declare compactMode: boolean
  @state() private declare focusedPriority: string | null
  @state() private declare loadingItems: Set<string>
  @state() private declare completedItems: Set<string>
  @state() private declare expandedItems: Set<string>

  private websocketService: WebSocketService
  private agentService = getAgentService()
  private refreshInterval: number | null = null
  private itemCache: Map<string, PriorityItem> = new Map()
  private insightEngine: any = null

  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      overflow: hidden;
    }

    .hub-container {
      height: 100%;
      display: flex;
      flex-direction: column;
    }

    .hub-header {
      background: white;
      border-bottom: 1px solid #e5e7eb;
      padding: 1rem;
      flex-shrink: 0;
    }

    .header-controls {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .hub-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: #111827;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .priority-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 2s infinite;
    }

    .priority-indicator.critical {
      background: #ef4444;
      animation: pulse 0.5s infinite;
    }

    .priority-indicator.high {
      background: #f59e0b;
      animation: pulse 1s infinite;
    }

    .view-mode-selector {
      display: flex;
      background: #f3f4f6;
      border-radius: 8px;
      padding: 0.25rem;
      gap: 0.25rem;
    }

    .view-mode-button {
      padding: 0.5rem 1rem;
      border: none;
      background: none;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.875rem;
      font-weight: 500;
      color: #6b7280;
      transition: all 0.2s;
    }

    .view-mode-button.active {
      background: white;
      color: #3b82f6;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    .filter-bar {
      display: flex;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
    }

    .search-input {
      flex: 1;
      min-width: 200px;
      padding: 0.75rem 1rem;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      background: #f9fafb;
      outline: none;
      font-size: 0.875rem;
      transition: all 0.2s;
    }

    .search-input:focus {
      border-color: #3b82f6;
      background: white;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .filter-chips {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .filter-chip {
      padding: 0.5rem 1rem;
      border-radius: 20px;
      border: 1px solid #e5e7eb;
      background: white;
      color: #6b7280;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }

    .filter-chip:hover {
      border-color: #d1d5db;
      background: #f9fafb;
    }

    .filter-chip.active {
      background: #3b82f6;
      color: white;
      border-color: #3b82f6;
    }

    .filter-chip.critical {
      background: #ef4444;
      color: white;
      border-color: #ef4444;
    }

    .filter-chip.high {
      background: #f59e0b;
      color: white;
      border-color: #f59e0b;
    }

    .filter-chip.medium {
      background: #3b82f6;
      color: white;
      border-color: #3b82f6;
    }

    .main-content {
      flex: 1;
      overflow: hidden;
      display: flex;
    }

    .priority-list {
      flex: 1;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      padding: 1rem;
    }

    .insights-panel {
      width: 300px;
      background: white;
      border-left: 1px solid #e5e7eb;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      transform: translateX(100%);
      transition: transform 0.3s ease-out;
    }

    .insights-panel.visible {
      transform: translateX(0);
    }

    .priority-item {
      background: white;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1rem;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      border-left: 4px solid #e5e7eb;
      transition: all 0.2s;
      position: relative;
    }

    .priority-item:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .priority-item.critical {
      border-left-color: #ef4444;
      background: linear-gradient(135deg, #fefefe 0%, #fef2f2 100%);
    }

    .priority-item.high {
      border-left-color: #f59e0b;
      background: linear-gradient(135deg, #fefefe 0%, #fef3c7 100%);
    }

    .priority-item.medium {
      border-left-color: #3b82f6;
      background: linear-gradient(135deg, #fefefe 0%, #eff6ff 100%);
    }

    .priority-item.low {
      border-left-color: #10b981;
      background: linear-gradient(135deg, #fefefe 0%, #ecfdf5 100%);
    }

    .priority-item.info {
      border-left-color: #6b7280;
    }

    .priority-item.loading {
      opacity: 0.6;
      pointer-events: none;
    }

    .priority-item.completed {
      opacity: 0.7;
      background: #f9fafb;
    }

    .priority-item.expanded {
      margin-bottom: 2rem;
      box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    .item-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .item-meta {
      flex: 1;
      min-width: 0;
    }

    .item-title {
      font-weight: 700;
      color: #111827;
      font-size: 1.1rem;
      margin-bottom: 0.5rem;
      line-height: 1.3;
    }

    .item-description {
      color: #6b7280;
      font-size: 0.875rem;
      line-height: 1.4;
      margin-bottom: 1rem;
    }

    .item-indicators {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 0.5rem;
    }

    .priority-badge {
      padding: 0.25rem 0.75rem;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .priority-badge.critical {
      background: #fef2f2;
      color: #dc2626;
      border: 1px solid #fecaca;
    }

    .priority-badge.high {
      background: #fef3c7;
      color: #d97706;
      border: 1px solid #fde68a;
    }

    .priority-badge.medium {
      background: #eff6ff;
      color: #2563eb;
      border: 1px solid #dbeafe;
    }

    .priority-badge.low {
      background: #ecfdf5;
      color: #059669;
      border: 1px solid #a7f3d0;
    }

    .priority-badge.info {
      background: #f3f4f6;
      color: #374151;
      border: 1px solid #e5e7eb;
    }

    .type-badge {
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 500;
      text-transform: uppercase;
      background: #f3f4f6;
      color: #6b7280;
    }

    .type-badge.task {
      background: #ecfdf5;
      color: #047857;
    }

    .type-badge.alert {
      background: #fef2f2;
      color: #dc2626;
    }

    .type-badge.decision {
      background: #fef3c7;
      color: #d97706;
    }

    .type-badge.milestone {
      background: #f3e8ff;
      color: #7c3aed;
    }

    .item-details {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
      padding: 1rem;
      background: rgba(0, 0, 0, 0.02);
      border-radius: 8px;
    }

    .detail-item {
      text-align: center;
    }

    .detail-value {
      font-weight: 700;
      color: #111827;
      font-size: 0.9rem;
      margin-bottom: 0.25rem;
    }

    .detail-label {
      font-size: 0.75rem;
      color: #6b7280;
      text-transform: uppercase;
      font-weight: 500;
    }

    .agent-info {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 1rem;
      padding: 0.75rem;
      background: rgba(59, 130, 246, 0.05);
      border-radius: 8px;
      border: 1px solid rgba(59, 130, 246, 0.1);
    }

    .agent-avatar {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: #3b82f6;
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.875rem;
      font-weight: 600;
    }

    .agent-meta {
      flex: 1;
    }

    .agent-name {
      font-weight: 600;
      color: #111827;
      font-size: 0.875rem;
    }

    .agent-role {
      color: #6b7280;
      font-size: 0.75rem;
      text-transform: capitalize;
    }

    .item-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }

    .tag {
      padding: 0.25rem 0.5rem;
      background: #f3f4f6;
      color: #374151;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 500;
    }

    .item-actions {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .action-button {
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      border: none;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      font-size: 0.875rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .action-button.primary {
      background: #3b82f6;
      color: white;
    }

    .action-button.primary:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }

    .action-button.secondary {
      background: #f3f4f6;
      color: #374151;
    }

    .action-button.secondary:hover {
      background: #e5e7eb;
    }

    .action-button.danger {
      background: #ef4444;
      color: white;
    }

    .action-button.danger:hover {
      background: #dc2626;
    }

    .action-button.success {
      background: #10b981;
      color: white;
    }

    .action-button.success:hover {
      background: #059669;
    }

    .expand-button {
      position: absolute;
      top: 1rem;
      right: 1rem;
      background: none;
      border: none;
      color: #6b7280;
      cursor: pointer;
      padding: 0.5rem;
      border-radius: 6px;
      transition: all 0.2s;
    }

    .expand-button:hover {
      background: rgba(0, 0, 0, 0.05);
      color: #374151;
    }

    .progress-bar {
      width: 100%;
      height: 4px;
      background: #e5e7eb;
      border-radius: 2px;
      overflow: hidden;
      margin: 1rem 0;
    }

    .progress-fill {
      height: 100%;
      background: #3b82f6;
      border-radius: 2px;
      transition: width 0.3s ease-out;
    }

    .progress-fill.critical {
      background: #ef4444;
    }

    .progress-fill.high {
      background: #f59e0b;
    }

    .insights-header {
      padding: 1rem;
      border-bottom: 1px solid #e5e7eb;
    }

    .insights-title {
      font-weight: 700;
      color: #111827;
      margin-bottom: 0.5rem;
    }

    .insights-subtitle {
      color: #6b7280;
      font-size: 0.875rem;
    }

    .insight-card {
      padding: 1rem;
      border-bottom: 1px solid #f3f4f6;
      transition: all 0.2s;
    }

    .insight-card:hover {
      background: #f9fafb;
    }

    .insight-header {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 0.5rem;
    }

    .insight-icon {
      width: 24px;
      height: 24px;
      border-radius: 6px;
      background: #eff6ff;
      color: #3b82f6;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
    }

    .insight-meta {
      flex: 1;
    }

    .insight-title {
      font-weight: 600;
      color: #111827;
      font-size: 0.875rem;
    }

    .insight-confidence {
      color: #6b7280;
      font-size: 0.75rem;
    }

    .insight-description {
      color: #6b7280;
      font-size: 0.8rem;
      line-height: 1.4;
      margin-bottom: 0.75rem;
    }

    .insight-actions {
      display: flex;
      gap: 0.5rem;
    }

    .insight-action {
      padding: 0.5rem 0.75rem;
      border-radius: 6px;
      border: 1px solid #e5e7eb;
      background: white;
      color: #6b7280;
      font-size: 0.75rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }

    .insight-action:hover {
      background: #f9fafb;
      border-color: #d1d5db;
    }

    .empty-state {
      text-align: center;
      padding: 3rem 1rem;
      color: #6b7280;
    }

    .empty-icon {
      font-size: 3rem;
      margin-bottom: 1rem;
      opacity: 0.5;
    }

    .empty-message {
      font-size: 1.1rem;
      font-weight: 500;
      margin-bottom: 0.5rem;
    }

    .empty-description {
      font-size: 0.875rem;
      line-height: 1.4;
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
      .insights-panel {
        position: fixed;
        top: 0;
        right: 0;
        height: 100%;
        z-index: 200;
      }
      
      .filter-bar {
        flex-direction: column;
        align-items: stretch;
        gap: 0.75rem;
      }
      
      .search-input {
        min-width: unset;
      }
      
      .item-details {
        grid-template-columns: repeat(2, 1fr);
      }
      
      .item-actions {
        flex-direction: column;
      }
    }

    /* Touch-friendly improvements */
    @media (hover: none) and (pointer: coarse) {
      .action-button,
      .filter-chip,
      .view-mode-button {
        min-height: 44px;
        padding: 0.75rem 1rem;
      }
      
      .priority-item {
        padding: 2rem 1.5rem;
      }
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      :host {
        background: #111827;
      }
      
      .hub-header,
      .priority-item,
      .insights-panel,
      .insight-card {
        background: #1f2937;
        border-color: #374151;
        color: #f9fafb;
      }
      
      .hub-title,
      .item-title,
      .insight-title {
        color: #f9fafb;
      }
    }

    /* Animations */
    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.5;
      }
    }

    /* Accessibility */
    @media (prefers-reduced-motion: reduce) {
      * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }
  `

  constructor() {
    super()
    this.mobile = true
    this.focusMode = 'all'
    this.priorityItems = []
    this.contextualInsights = []
    this.filterOptions = {
      types: [],
      priorities: [],
      agents: [],
      timeRange: 'all',
      showCompleted: false,
      sortBy: 'priority'
    }
    this.activeFilter = 'all'
    this.searchQuery = ''
    this.selectedItem = null
    this.viewMode = 'list'
    this.autoRefresh = true
    this.showInsights = false
    this.compactMode = true
    this.focusedPriority = null
    this.loadingItems = new Set()
    this.completedItems = new Set()
    this.expandedItems = new Set()
    this.websocketService = WebSocketService.getInstance()
  }

  connectedCallback() {
    super.connectedCallback()
    this.setupWebSocketListeners()
    this.loadInitialData()
    this.startAutoRefresh()
    this.generateMockInsights()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.cleanup()
  }

  private setupWebSocketListeners() {
    this.websocketService.on('priority-update', (data) => {
      this.handlePriorityUpdate(data)
    })

    this.websocketService.on('agent-event', (data) => {
      this.handleAgentEvent(data)
    })

    this.websocketService.on('system-insight', (insight) => {
      this.addContextualInsight(insight)
    })
  }

  private async loadInitialData() {
    // Generate mock priority items
    this.generateMockPriorityItems()
    
    // Load real data from services
    try {
      const agents = this.agentService.getAgents()
      const agentSummary = this.agentService.getAgentSummary()
      
      // Generate priority items from agent status
      agents.forEach(agent => {
        if (agent.status === 'ERROR') {
          this.addPriorityItem({
            id: `agent-error-${agent.id}`,
            type: 'error',
            priority: 'critical',
            title: `Agent Error: ${agent.name}`,
            description: agent.error_message || 'Agent encountered an error and needs attention',
            timestamp: new Date().toISOString(),
            agentId: agent.id,
            agentName: agent.name,
            agentRole: agent.role,
            data: { agent },
            actions: [
              {
                id: 'restart',
                label: 'Restart Agent',
                command: `/hive:agent-restart --id=${agent.id}`,
                icon: '🔄',
                style: 'primary'
              },
              {
                id: 'diagnose',
                label: 'Diagnose',
                command: `/hive:agent-diagnose --id=${agent.id}`,
                icon: '🔍',
                style: 'secondary'
              }
            ],
            tags: ['error', 'agent', agent.role],
            impact: 'high',
            urgency: 'high'
          })
        }
      })
      
    } catch (error) {
      console.warn('Failed to load real agent data:', error)
    }
  }

  private generateMockPriorityItems() {
    const mockItems: PriorityItem[] = [
      {
        id: 'mobile-pwa-performance',
        type: 'task',
        priority: 'critical',
        title: 'Mobile PWA Performance Critical',
        description: 'Dashboard load time exceeds 10s target. Memory usage at 200MB, needs optimization to <50MB.',
        timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(), // 5 minutes ago
        agentId: 'frontend-dev-001',
        agentName: 'Frontend Developer',
        agentRole: 'frontend-developer',
        data: {
          currentLoadTime: '10.2s',
          targetLoadTime: '3s',
          memoryUsage: '200MB',
          targetMemory: '50MB',
          batteryImpact: '5%/hour'
        },
        actions: [
          {
            id: 'optimize-now',
            label: 'Optimize Now',
            command: '/hive:optimize-mobile-performance',
            icon: '⚡',
            style: 'primary',
            estimatedTime: 30
          },
          {
            id: 'analyze',
            label: 'Deep Analysis',
            command: '/hive:analyze-performance-bottlenecks',
            icon: '🔍',
            style: 'secondary',
            estimatedTime: 15
          }
        ],
        tags: ['performance', 'mobile', 'critical', 'pwa'],
        estimatedTime: 45,
        confidence: 95,
        impact: 'high',
        urgency: 'high',
        category: 'performance'
      },
      {
        id: 'agent-coordination-decision',
        type: 'decision',
        priority: 'high',
        title: 'Multi-Agent Task Assignment Strategy',
        description: 'Architecture agent requesting guidance on task distribution for mobile gesture interface implementation.',
        timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(), // 15 minutes ago
        agentId: 'architect-001',
        agentName: 'System Architect',
        agentRole: 'architect',
        data: {
          taskComplexity: 'High',
          estimatedDuration: '2-4 hours',
          requiredSpecializations: ['mobile', 'gestures', 'webapi'],
          availableAgents: 3
        },
        actions: [
          {
            id: 'approve-parallel',
            label: 'Approve Parallel Work',
            command: '/hive:approve-parallel-task-assignment',
            icon: '✅',
            style: 'success',
            estimatedTime: 5
          },
          {
            id: 'sequential',
            label: 'Sequential Assignment',
            command: '/hive:sequential-task-assignment',
            icon: '📋',
            style: 'secondary',
            estimatedTime: 10
          },
          {
            id: 'human-review',
            label: 'Human Review',
            command: '/hive:escalate-to-human',
            icon: '👤',
            style: 'danger',
            requiresConfirmation: true
          }
        ],
        tags: ['decision', 'architecture', 'coordination'],
        estimatedTime: 20,
        confidence: 87,
        impact: 'medium',
        urgency: 'high',
        category: 'coordination'
      },
      {
        id: 'websocket-integration-milestone',
        type: 'milestone',
        priority: 'medium',
        title: 'Real-time WebSocket Integration Complete',
        description: 'WebSocket service successfully integrated with <50ms update latency achieved. Ready for dashboard connection.',
        timestamp: new Date(Date.now() - 2 * 60 * 1000).toISOString(), // 2 minutes ago
        agentId: 'backend-dev-001',
        agentName: 'Backend Developer',
        agentRole: 'backend-developer',
        data: {
          latency: '42ms',
          target: '<50ms',
          connectionStability: '99.8%',
          throughput: '1000+ RPS'
        },
        actions: [
          {
            id: 'connect-dashboard',
            label: 'Connect Dashboard',
            command: '/hive:connect-websocket-dashboard',
            icon: '🔌',
            style: 'primary',
            estimatedTime: 10
          },
          {
            id: 'stress-test',
            label: 'Stress Test',
            command: '/hive:websocket-stress-test',
            icon: '🧪',
            style: 'secondary',
            estimatedTime: 15
          }
        ],
        tags: ['milestone', 'websocket', 'performance', 'complete'],
        estimatedTime: 10,
        confidence: 98,
        impact: 'medium',
        urgency: 'medium',
        category: 'integration'
      },
      {
        id: 'gesture-interface-alert',
        type: 'alert',
        priority: 'high',
        title: 'Mobile Gesture Interface Testing Required',
        description: 'New gesture commands implemented but require device testing before deployment. Touch targets need COPPA validation.',
        timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(), // 30 minutes ago
        agentId: 'qa-engineer-001',
        agentName: 'QA Engineer',
        agentRole: 'qa-engineer',
        data: {
          gesturesImplemented: 4,
          testingRequired: 'Device Testing',
          complianceCheck: 'COPPA',
          riskLevel: 'Medium'
        },
        actions: [
          {
            id: 'schedule-testing',
            label: 'Schedule Device Testing',
            command: '/hive:schedule-device-testing',
            icon: '📱',
            style: 'primary',
            estimatedTime: 60
          },
          {
            id: 'coppa-validation',
            label: 'COPPA Validation',
            command: '/hive:validate-coppa-compliance',
            icon: '🛡️',
            style: 'secondary',
            estimatedTime: 30
          }
        ],
        tags: ['testing', 'gestures', 'compliance', 'mobile'],
        estimatedTime: 90,
        confidence: 92,
        impact: 'high',
        urgency: 'medium',
        category: 'testing'
      }
    ]

    this.priorityItems = mockItems
  }

  private generateMockInsights() {
    const insights: ContextualInsight[] = [
      {
        id: 'performance-pattern',
        type: 'pattern',
        title: 'Performance Bottleneck Pattern Detected',
        description: 'Mobile dashboard consistently shows high memory usage during agent coordination tasks. Pattern suggests optimization opportunity.',
        confidence: 89,
        impact: 'high',
        data: {
          pattern: 'memory-spike-during-coordination',
          frequency: '73% of coordination tasks',
          avgImpact: '40MB memory increase'
        },
        actions: [
          {
            id: 'implement-memory-optimization',
            label: 'Implement Memory Pool',
            command: '/hive:implement-memory-optimization',
            icon: '🚀',
            style: 'primary'
          }
        ]
      },
      {
        id: 'workload-prediction',
        type: 'prediction',
        title: 'High Agent Workload Predicted',
        description: 'Based on current patterns, expect 150% increase in agent tasks over next 2 hours. Recommend proactive scaling.',
        confidence: 76,
        impact: 'medium',
        data: {
          currentLoad: '34 tasks/hour',
          predictedLoad: '85 tasks/hour',
          timeframe: '2 hours',
          confidence: '76%'
        },
        actions: [
          {
            id: 'scale-agents',
            label: 'Scale Agent Team',
            command: '/hive:scale-agent-team',
            icon: '📈',
            style: 'primary'
          }
        ]
      },
      {
        id: 'optimization-recommendation',
        type: 'optimization',
        title: 'Gesture Response Time Optimization',
        description: 'Gesture interface could benefit from predictive caching. Estimated 40% response time improvement.',
        confidence: 84,
        impact: 'medium',
        data: {
          currentResponseTime: '120ms',
          predictedImprovement: '40%',
          newResponseTime: '72ms',
          implementationEffort: 'Medium'
        },
        actions: [
          {
            id: 'implement-caching',
            label: 'Implement Predictive Cache',
            command: '/hive:implement-gesture-caching',
            icon: '⚡',
            style: 'primary'
          }
        ]
      }
    ]

    this.contextualInsights = insights
  }

  private addPriorityItem(item: PriorityItem) {
    // Check for duplicates
    const existingIndex = this.priorityItems.findIndex(existing => existing.id === item.id)
    if (existingIndex >= 0) {
      this.priorityItems[existingIndex] = item
    } else {
      this.priorityItems.unshift(item)
    }
    
    // Sort by priority and timestamp
    this.sortPriorityItems()
    this.requestUpdate()
  }

  private addContextualInsight(insight: ContextualInsight) {
    this.contextualInsights.unshift(insight)
    if (this.contextualInsights.length > 10) {
      this.contextualInsights = this.contextualInsights.slice(0, 10)
    }
    this.requestUpdate()
  }

  private handlePriorityUpdate(data: any) {
    if (data.type === 'item-update') {
      this.addPriorityItem(data.item)
    } else if (data.type === 'item-complete') {
      this.completedItems.add(data.itemId)
      this.requestUpdate()
    }
  }

  private handleAgentEvent(data: any) {
    // Convert agent events to priority items
    if (data.type === 'task-start') {
      this.addPriorityItem({
        id: `task-${data.taskId}`,
        type: 'task',
        priority: 'medium',
        title: `Task Started: ${data.taskTitle}`,
        description: `Agent ${data.agentName} started working on ${data.taskTitle}`,
        timestamp: new Date().toISOString(),
        agentId: data.agentId,
        agentName: data.agentName,
        agentRole: data.agentRole,
        data: data,
        actions: [
          {
            id: 'monitor',
            label: 'Monitor Progress',
            command: `/hive:monitor-task --id=${data.taskId}`,
            icon: '👁️',
            style: 'secondary'
          }
        ],
        tags: ['task', 'active', data.agentRole],
        impact: 'medium',
        urgency: 'low'
      })
    }
  }

  private sortPriorityItems() {
    const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1, info: 0 }
    
    this.priorityItems.sort((a, b) => {
      // First by priority
      const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority]
      if (priorityDiff !== 0) return priorityDiff
      
      // Then by timestamp (newest first)
      return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    })
  }

  private startAutoRefresh() {
    if (this.autoRefresh) {
      this.refreshInterval = window.setInterval(() => {
        this.loadInitialData()
      }, 30000) // Refresh every 30 seconds
    }
  }

  private cleanup() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval)
    }
  }

  private getFilteredItems(): PriorityItem[] {
    let filtered = this.priorityItems

    // Filter by search query
    if (this.searchQuery) {
      const query = this.searchQuery.toLowerCase()
      filtered = filtered.filter(item => 
        item.title.toLowerCase().includes(query) ||
        item.description.toLowerCase().includes(query) ||
        item.tags.some(tag => tag.toLowerCase().includes(query)) ||
        item.agentName?.toLowerCase().includes(query)
      )
    }

    // Filter by active filter
    if (this.activeFilter !== 'all') {
      filtered = filtered.filter(item => 
        item.priority === this.activeFilter ||
        item.type === this.activeFilter ||
        item.agentRole === this.activeFilter
      )
    }

    // Filter by completed status
    if (!this.filterOptions.showCompleted) {
      filtered = filtered.filter(item => !this.completedItems.has(item.id))
    }

    return filtered
  }

  private toggleItemExpansion(itemId: string) {
    if (this.expandedItems.has(itemId)) {
      this.expandedItems.delete(itemId)
    } else {
      this.expandedItems.add(itemId)
    }
    this.requestUpdate()
  }

  private async executeAction(item: PriorityItem, action: PriorityAction) {
    if (action.requiresConfirmation) {
      const confirmed = confirm(`Execute action: ${action.label}?`)
      if (!confirmed) return
    }

    this.loadingItems.add(item.id)
    this.requestUpdate()
    
    try {
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: action.command,
          context: { 
            itemId: item.id, 
            agentId: item.agentId,
            data: item.data 
          }
        })
      })
      
      if (response.ok) {
        const result = await response.json()
        if (result.success) {
          // Mark as completed or update status
          this.completedItems.add(item.id)
          
          // Show success feedback
          this.dispatchEvent(new CustomEvent('action-completed', {
            detail: { item, action, result },
            bubbles: true,
            composed: true
          }))
        } else {
          throw new Error(result.error || 'Action failed')
        }
      } else {
        throw new Error('Network error')
      }
    } catch (error) {
      console.error('Action execution failed:', error)
      this.dispatchEvent(new CustomEvent('action-failed', {
        detail: { item, action, error },
        bubbles: true,
        composed: true
      }))
    } finally {
      this.loadingItems.delete(item.id)
      this.requestUpdate()
    }
  }

  private formatTimestamp(timestamp: string): string {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    
    if (diff < 60000) return 'Just now'
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
    return date.toLocaleDateString()
  }

  private getTypeIcon(type: PriorityItem['type']): string {
    const icons = {
      task: '📋',
      alert: '🚨',
      decision: '🤔',
      milestone: '🎯',
      performance: '📊',
      error: '❌'
    }
    return icons[type] || '📎'
  }

  render() {
    const filteredItems = this.getFilteredItems()
    const criticalCount = this.priorityItems.filter(item => item.priority === 'critical').length
    const highCount = this.priorityItems.filter(item => item.priority === 'high').length

    return html`
      <div class="hub-container">
        ${this.renderHeader(criticalCount, highCount)}
        ${this.renderMainContent(filteredItems)}
        ${this.renderInsightsPanel()}
      </div>
    `
  }

  private renderHeader(criticalCount: number, highCount: number) {
    return html`
      <div class="hub-header">
        <div class="header-controls">
          <div class="hub-title">
            <div class="priority-indicator ${criticalCount > 0 ? 'critical' : highCount > 0 ? 'high' : ''}"></div>
            🎯 Priority Hub
          </div>
          
          <div class="view-mode-selector">
            <button class="view-mode-button ${this.viewMode === 'list' ? 'active' : ''}" 
                    @click=${() => { this.viewMode = 'list' }}>
              List
            </button>
            <button class="view-mode-button ${this.viewMode === 'kanban' ? 'active' : ''}" 
                    @click=${() => { this.viewMode = 'kanban' }}>
              Board
            </button>
            <button class="view-mode-button" 
                    @click=${() => { this.showInsights = !this.showInsights }}>
              Insights
            </button>
          </div>
        </div>
        
        <div class="filter-bar">
          <input type="text" 
                 class="search-input" 
                 placeholder="Search priority items..."
                 .value=${this.searchQuery}
                 @input=${(e: InputEvent) => { 
                   this.searchQuery = (e.target as HTMLInputElement).value 
                 }}>
          
          <div class="filter-chips">
            <button class="filter-chip ${this.activeFilter === 'all' ? 'active' : ''}" 
                    @click=${() => { this.activeFilter = 'all' }}>
              All
            </button>
            <button class="filter-chip critical ${this.activeFilter === 'critical' ? 'active' : ''}" 
                    @click=${() => { this.activeFilter = 'critical' }}>
              Critical (${criticalCount})
            </button>
            <button class="filter-chip high ${this.activeFilter === 'high' ? 'active' : ''}" 
                    @click=${() => { this.activeFilter = 'high' }}>
              High (${highCount})
            </button>
            <button class="filter-chip ${this.activeFilter === 'task' ? 'active' : ''}" 
                    @click=${() => { this.activeFilter = 'task' }}>
              Tasks
            </button>
            <button class="filter-chip ${this.activeFilter === 'alert' ? 'active' : ''}" 
                    @click=${() => { this.activeFilter = 'alert' }}>
              Alerts
            </button>
          </div>
        </div>
      </div>
    `
  }

  private renderMainContent(filteredItems: PriorityItem[]) {
    return html`
      <div class="main-content">
        <div class="priority-list">
          ${filteredItems.length > 0 ? 
            filteredItems.map(item => this.renderPriorityItem(item)) :
            this.renderEmptyState()
          }
        </div>
      </div>
    `
  }

  private renderPriorityItem(item: PriorityItem) {
    const isLoading = this.loadingItems.has(item.id)
    const isCompleted = this.completedItems.has(item.id)
    const isExpanded = this.expandedItems.has(item.id)

    return html`
      <div class="priority-item ${item.priority} ${isLoading ? 'loading' : ''} ${isCompleted ? 'completed' : ''} ${isExpanded ? 'expanded' : ''}">
        <button class="expand-button" @click=${() => this.toggleItemExpansion(item.id)}>
          ${isExpanded ? '▼' : '▶'}
        </button>
        
        <div class="item-header">
          <div class="item-meta">
            <div class="item-title">${item.title}</div>
            <div class="item-description">${item.description}</div>
          </div>
          <div class="item-indicators">
            <div class="priority-badge ${item.priority}">${item.priority}</div>
            <div class="type-badge ${item.type}">${item.type}</div>
          </div>
        </div>

        ${item.agentId ? html`
          <div class="agent-info">
            <div class="agent-avatar">${item.agentName?.charAt(0) || 'A'}</div>
            <div class="agent-meta">
              <div class="agent-name">${item.agentName}</div>
              <div class="agent-role">${item.agentRole?.replace('_', ' ')}</div>
            </div>
          </div>
        ` : ''}

        ${isExpanded ? html`
          ${Object.keys(item.data).length > 0 ? html`
            <div class="item-details">
              ${Object.entries(item.data).slice(0, 4).map(([key, value]) => html`
                <div class="detail-item">
                  <div class="detail-value">${value}</div>
                  <div class="detail-label">${key.replace(/([A-Z])/g, ' $1').toLowerCase()}</div>
                </div>
              `)}
            </div>
          ` : ''}
          
          ${item.estimatedTime ? html`
            <div class="progress-bar">
              <div class="progress-fill ${item.priority}" style="width: ${isCompleted ? 100 : Math.random() * 60 + 20}%"></div>
            </div>
          ` : ''}
        ` : ''}

        ${item.tags.length > 0 ? html`
          <div class="item-tags">
            ${item.tags.slice(0, 4).map(tag => html`
              <span class="tag">${tag}</span>
            `)}
          </div>
        ` : ''}

        <div class="item-actions">
          ${item.actions.map(action => html`
            <button class="action-button ${action.style}" 
                    @click=${() => this.executeAction(item, action)}
                    ?disabled=${isLoading}>
              <span>${action.icon}</span>
              <span>${action.label}</span>
              ${action.estimatedTime ? html`<span>(${action.estimatedTime}m)</span>` : ''}
            </button>
          `)}
        </div>
      </div>
    `
  }

  private renderInsightsPanel() {
    return html`
      <div class="insights-panel ${this.showInsights ? 'visible' : ''}">
        <div class="insights-header">
          <div class="insights-title">🧠 Contextual Insights</div>
          <div class="insights-subtitle">AI-powered recommendations</div>
        </div>
        
        ${this.contextualInsights.map(insight => html`
          <div class="insight-card">
            <div class="insight-header">
              <div class="insight-icon">${this.getInsightIcon(insight.type)}</div>
              <div class="insight-meta">
                <div class="insight-title">${insight.title}</div>
                <div class="insight-confidence">${insight.confidence}% confidence</div>
              </div>
            </div>
            <div class="insight-description">${insight.description}</div>
            <div class="insight-actions">
              ${insight.actions.map(action => html`
                <button class="insight-action" @click=${() => this.executeInsightAction(insight, action)}>
                  ${action.icon} ${action.label}
                </button>
              `)}
            </div>
          </div>
        `)}
      </div>
    `
  }

  private renderEmptyState() {
    return html`
      <div class="empty-state">
        <div class="empty-icon">🎯</div>
        <div class="empty-message">No priority items found</div>
        <div class="empty-description">
          All caught up! Priority items will appear here when they need attention.
        </div>
      </div>
    `
  }

  private getInsightIcon(type: ContextualInsight['type']): string {
    const icons = {
      pattern: '🔍',
      recommendation: '💡',
      prediction: '🔮',
      optimization: '⚡'
    }
    return icons[type] || '🤖'
  }

  private async executeInsightAction(insight: ContextualInsight, action: PriorityAction) {
    // Similar to executeAction but for insights
    try {
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: action.command,
          context: { 
            insightId: insight.id,
            data: insight.data 
          }
        })
      })
      
      if (response.ok) {
        const result = await response.json()
        // Handle insight action result
        console.log('Insight action executed:', result)
      }
    } catch (error) {
      console.error('Insight action failed:', error)
    }
  }
}