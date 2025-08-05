import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../services/websocket'
import { getAgentService, AgentService, AgentSummary, TeamComposition } from '../services/agent'
import { Agent, AgentRole, AgentStatus, AgentPerformanceMetrics } from '../types/api'

interface PriorityAlert {
  id: string
  type: 'critical' | 'high' | 'medium' | 'info'
  title: string
  message: string
  agentId?: string
  timestamp: string
  actions: AlertAction[]
}

interface AlertAction {
  id: string
  label: string
  command: string
  style: 'primary' | 'secondary' | 'danger'
}

interface GestureCommand {
  gesture: 'swipe-right' | 'swipe-left' | 'swipe-up' | 'long-press'
  action: string
  agentId?: string
  confirmation?: boolean
}

@customElement('enhanced-agent-management-dashboard')
export class EnhancedAgentManagementDashboard extends LitElement {
  @property({ type: Boolean }) declare mobile: boolean
  @property({ type: String }) declare initialView: string

  @state() private declare agents: Agent[]
  @state() private declare agentSummary: AgentSummary | null
  @state() private declare teamComposition: TeamComposition
  @state() private declare priorityAlerts: PriorityAlert[]
  @state() private declare activeView: 'overview' | 'agents' | 'tasks' | 'alerts' | 'performance'
  @state() private declare selectedAgent: Agent | null
  @state() private declare systemReady: boolean
  @state() private declare loading: boolean
  @state() private declare error: string | null
  @state() private declare lastUpdate: Date | null
  @state() private declare connectionStatus: 'connected' | 'disconnected' | 'reconnecting'
  @state() private declare gestureEnabled: boolean
  @state() private declare filterPriority: 'all' | 'critical' | 'high' | 'medium'
  @state() private declare showQuickActions: boolean
  @state() private declare performanceMetrics: Map<string, AgentPerformanceMetrics>
  @state() private declare bulkSelectionMode: boolean
  @state() private declare selectedAgents: Set<string>

  private agentService: AgentService
  private websocketService: WebSocketService
  private refreshInterval: number | null = null
  private touchStartX = 0
  private touchStartY = 0
  private gestureThreshold = 80
  private performanceUpdateInterval: number | null = null

  static styles = css`
    :host {
      display: block;
      height: 100vh;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      overflow: hidden;
      position: relative;
    }

    .dashboard-container {
      height: 100%;
      display: flex;
      flex-direction: column;
    }

    .dashboard-header {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(0, 0, 0, 0.1);
      padding: 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 100;
      flex-shrink: 0;
    }

    .header-title {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .system-status-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 2s infinite;
      position: relative;
    }

    .system-status-indicator.warning {
      background: #f59e0b;
    }

    .system-status-indicator.error {
      background: #ef4444;
    }

    .system-status-indicator.disconnected {
      background: #6b7280;
    }

    .header-metrics {
      display: flex;
      align-items: center;
      gap: 1rem;
      font-size: 0.875rem;
      color: #6b7280;
    }

    .metric-item {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }

    .metric-value {
      font-weight: 600;
      color: #111827;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .header-button {
      background: none;
      border: none;
      padding: 0.75rem;
      border-radius: 12px;
      cursor: pointer;
      color: #6b7280;
      transition: all 0.2s;
      font-size: 1.25rem;
      position: relative;
    }

    .header-button:hover {
      background: rgba(0, 0, 0, 0.05);
      color: #374151;
    }

    .header-button.active {
      background: #eff6ff;
      color: #3b82f6;
    }

    .alert-badge {
      position: absolute;
      top: 0.25rem;
      right: 0.25rem;
      background: #ef4444;
      color: white;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 50%;
      width: 18px;
      height: 18px;
      display: flex;
      align-items: center;
      justify-content: center;
      line-height: 1;
    }

    .priority-filter-bar {
      background: white;
      border-bottom: 1px solid #e5e7eb;
      padding: 0.75rem 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      flex-shrink: 0;
    }

    .filter-chip {
      padding: 0.5rem 1rem;
      border-radius: 20px;
      border: 1px solid #e5e7eb;
      background: white;
      color: #6b7280;
      font-size: 0.875rem;
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

    .main-content {
      flex: 1;
      overflow: hidden;
      position: relative;
    }

    .content-area {
      height: 100%;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      padding: 1rem;
    }

    .system-overview-card {
      background: white;
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 1rem;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      position: relative;
    }

    .overview-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.5rem;
    }

    .overview-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: #111827;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .quick-actions-button {
      background: #3b82f6;
      color: white;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      font-size: 0.875rem;
    }

    .quick-actions-button:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .metric-card {
      text-align: center;
      padding: 1rem;
      background: #f8fafc;
      border-radius: 12px;
      border: 1px solid #e2e8f0;
      transition: all 0.2s;
    }

    .metric-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .metric-value {
      font-size: 1.75rem;
      font-weight: 700;
      color: #111827;
      margin-bottom: 0.25rem;
    }

    .metric-value.success {
      color: #10b981;
    }

    .metric-value.warning {
      color: #f59e0b;
    }

    .metric-value.error {
      color: #ef4444;
    }

    .metric-label {
      font-size: 0.8rem;
      color: #6b7280;
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.05em;
    }

    .agent-grid {
      display: grid;
      gap: 1rem;
      grid-template-columns: 1fr;
    }

    .agent-card {
      background: white;
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      transition: all 0.2s;
      position: relative;
    }

    .agent-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    .agent-card.selected {
      border: 2px solid #3b82f6;
      background: #eff6ff;
    }

    .agent-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .agent-info {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .agent-avatar {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.5rem;
      background: #eff6ff;
      color: #3b82f6;
    }

    .agent-avatar.product-manager {
      background: #fef3c7;
      color: #f59e0b;
    }

    .agent-avatar.architect {
      background: #f3e8ff;
      color: #8b5cf6;
    }

    .agent-avatar.backend-developer {
      background: #dcfce7;
      color: #16a34a;
    }

    .agent-avatar.frontend-developer {
      background: #fef2f2;
      color: #dc2626;
    }

    .agent-avatar.qa-engineer {
      background: #e0f2fe;
      color: #0284c7;
    }

    .agent-meta {
      flex: 1;
    }

    .agent-name {
      font-weight: 700;
      color: #111827;
      font-size: 1.1rem;
      margin-bottom: 0.25rem;
    }

    .agent-role {
      color: #6b7280;
      font-size: 0.875rem;
      text-transform: capitalize;
    }

    .agent-status {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .status-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
    }

    .status-indicator.idle {
      background: #f59e0b;
    }

    .status-indicator.busy {
      background: #3b82f6;
      animation: pulse 1.5s infinite;
    }

    .status-indicator.error {
      background: #ef4444;
    }

    .status-indicator.offline {
      background: #6b7280;
    }

    .status-text {
      font-size: 0.8rem;
      font-weight: 600;
      color: #6b7280;
      text-transform: uppercase;
    }

    .agent-performance {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1rem;
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid #f3f4f6;
    }

    .performance-metric {
      text-align: center;
    }

    .performance-value {
      font-weight: 700;
      color: #111827;
      font-size: 1.1rem;
    }

    .performance-label {
      font-size: 0.75rem;
      color: #6b7280;
      margin-top: 0.25rem;
    }

    .agent-actions {
      display: flex;
      gap: 0.5rem;
      margin-top: 1rem;
    }

    .agent-action-button {
      padding: 0.5rem 1rem;
      border-radius: 8px;
      border: 1px solid #e5e7eb;
      background: white;
      color: #6b7280;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }

    .agent-action-button:hover {
      background: #f9fafb;
      border-color: #d1d5db;
    }

    .agent-action-button.primary {
      background: #3b82f6;
      color: white;
      border-color: #3b82f6;
    }

    .agent-action-button.primary:hover {
      background: #2563eb;
    }

    .agent-action-button.danger {
      background: #ef4444;
      color: white;
      border-color: #ef4444;
    }

    .agent-action-button.danger:hover {
      background: #dc2626;
    }

    .priority-alerts-container {
      margin-bottom: 1rem;
    }

    .alert-card {
      background: white;
      border-radius: 12px;
      padding: 1rem;
      margin-bottom: 0.75rem;
      border-left: 4px solid #3b82f6;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      transition: all 0.2s;
    }

    .alert-card:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .alert-card.critical {
      border-left-color: #ef4444;
      background: #fefefe;
    }

    .alert-card.high {
      border-left-color: #f59e0b;
    }

    .alert-card.medium {
      border-left-color: #3b82f6;
    }

    .alert-card.info {
      border-left-color: #10b981;
    }

    .alert-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }

    .alert-title {
      font-weight: 600;
      color: #111827;
      font-size: 0.95rem;
    }

    .alert-priority {
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }

    .alert-priority.critical {
      background: #fef2f2;
      color: #dc2626;
    }

    .alert-priority.high {
      background: #fef3c7;
      color: #d97706;
    }

    .alert-priority.medium {
      background: #eff6ff;
      color: #2563eb;
    }

    .alert-priority.info {
      background: #ecfdf5;
      color: #059669;
    }

    .alert-message {
      color: #6b7280;
      font-size: 0.875rem;
      line-height: 1.4;
      margin-bottom: 1rem;
    }

    .alert-actions {
      display: flex;
      gap: 0.5rem;
    }

    .alert-action-button {
      padding: 0.5rem 1rem;
      border-radius: 6px;
      border: none;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }

    .alert-action-button.primary {
      background: #3b82f6;
      color: white;
    }

    .alert-action-button.secondary {
      background: #f3f4f6;
      color: #374151;
    }

    .alert-action-button.danger {
      background: #ef4444;
      color: white;
    }

    .quick-actions-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.5);
      z-index: 200;
      display: flex;
      align-items: center;
      justify-content: center;
      backdrop-filter: blur(5px);
    }

    .quick-actions-panel {
      background: white;
      border-radius: 20px;
      padding: 2rem;
      max-width: 90%;
      width: 400px;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }

    .quick-actions-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.5rem;
    }

    .quick-actions-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
    }

    .close-button {
      background: none;
      border: none;
      color: #6b7280;
      cursor: pointer;
      padding: 0.5rem;
      border-radius: 8px;
      transition: all 0.2s;
      font-size: 1.25rem;
    }

    .close-button:hover {
      background: #f3f4f6;
      color: #374151;
    }

    .quick-actions-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1rem;
    }

    .quick-action-item {
      padding: 1.5rem;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
      background: white;
      cursor: pointer;
      transition: all 0.2s;
      text-align: center;
    }

    .quick-action-item:hover {
      background: #f9fafb;
      border-color: #3b82f6;
      transform: translateY(-2px);
    }

    .quick-action-icon {
      font-size: 2rem;
      margin-bottom: 0.75rem;
    }

    .quick-action-title {
      font-weight: 600;
      color: #111827;
      margin-bottom: 0.5rem;
    }

    .quick-action-description {
      font-size: 0.8rem;
      color: #6b7280;
      line-height: 1.3;
    }

    .bulk-selection-bar {
      background: #3b82f6;
      color: white;
      padding: 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-radius: 12px;
      margin-bottom: 1rem;
    }

    .bulk-selection-info {
      font-weight: 600;
    }

    .bulk-actions {
      display: flex;
      gap: 0.5rem;
    }

    .bulk-action-button {
      background: rgba(255, 255, 255, 0.2);
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }

    .bulk-action-button:hover {
      background: rgba(255, 255, 255, 0.3);
    }

    .gesture-hint {
      position: fixed;
      bottom: 50%;
      left: 50%;
      transform: translate(-50%, 50%);
      background: rgba(59, 130, 246, 0.9);
      color: white;
      padding: 0.75rem 1.5rem;
      border-radius: 12px;
      font-size: 0.9rem;
      font-weight: 500;
      z-index: 150;
      animation: fadeInOut 2s ease-in-out;
      pointer-events: none;
    }

    .loading-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(255, 255, 255, 0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 100;
      backdrop-filter: blur(2px);
    }

    .loading-spinner {
      width: 40px;
      height: 40px;
      border: 3px solid #e5e7eb;
      border-top: 3px solid #3b82f6;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    .error-banner {
      background: #fef2f2;
      color: #dc2626;
      padding: 1rem;
      border-radius: 8px;
      margin-bottom: 1rem;
      border: 1px solid #fecaca;
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

    @keyframes spin {
      0% {
        transform: rotate(0deg);
      }
      100% {
        transform: rotate(360deg);
      }
    }

    @keyframes fadeInOut {
      0%, 100% {
        opacity: 0;
        transform: translate(-50%, 50%) scale(0.8);
      }
      20%, 80% {
        opacity: 1;
        transform: translate(-50%, 50%) scale(1);
      }
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
      .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
      }
      
      .agent-performance {
        grid-template-columns: repeat(2, 1fr);
      }
      
      .quick-actions-grid {
        grid-template-columns: 1fr;
      }
      
      .header-metrics {
        display: none;
      }
    }

    /* Touch-friendly improvements */
    @media (hover: none) and (pointer: coarse) {
      .header-button,
      .agent-action-button,
      .alert-action-button {
        min-height: 44px;
        min-width: 44px;
      }
      
      .agent-card {
        padding: 2rem 1.5rem;
      }
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      :host {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
      }
      
      .dashboard-header,
      .priority-filter-bar,
      .system-overview-card,
      .agent-card,
      .alert-card,
      .quick-actions-panel {
        background: #1f2937;
        border-color: #374151;
        color: #f9fafb;
      }
      
      .overview-title,
      .agent-name {
        color: #f9fafb;
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
    this.initialView = 'overview'
    this.agents = []
    this.agentSummary = null
    this.teamComposition = {}
    this.priorityAlerts = []
    this.activeView = 'overview'
    this.selectedAgent = null
    this.systemReady = false
    this.loading = false
    this.error = null
    this.lastUpdate = null
    this.connectionStatus = 'disconnected'
    this.gestureEnabled = true
    this.filterPriority = 'all'
    this.showQuickActions = false
    this.performanceMetrics = new Map()
    this.bulkSelectionMode = false
    this.selectedAgents = new Set()
    
    this.agentService = getAgentService()
    this.websocketService = WebSocketService.getInstance()
  }

  connectedCallback() {
    super.connectedCallback()
    this.setupEventListeners()
    this.setupWebSocketListeners()
    this.setupGestureHandlers()
    this.loadInitialData()
    this.startRealTimeUpdates()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.cleanup()
  }

  private setupEventListeners() {
    // Agent service events
    this.agentService.onAgentStatusChanged((status) => {
      this.updateSystemStatus(status)
    })

    this.agentService.onAgentSpawned((result) => {
      this.showTemporaryMessage(`Agent ${result.role} spawned successfully`, 'success')
      this.refreshData()
    })

    this.agentService.onAgentDeactivated((result) => {
      this.showTemporaryMessage(`Agent deactivated: ${result.message}`, 'info')
      this.refreshData()
    })

    this.agentService.onTeamActivated((result) => {
      this.showTemporaryMessage(`Team activated: ${result.agents.length} agents`, 'success')
      this.refreshData()
    })
  }

  private setupWebSocketListeners() {
    this.websocketService.on('agent-status-update', (data) => {
      this.handleAgentStatusUpdate(data)
    })

    this.websocketService.on('system-alert', (alert) => {
      this.addPriorityAlert(alert)
    })

    this.websocketService.on('performance-update', (data) => {
      this.handlePerformanceUpdate(data)
    })

    this.websocketService.on('connection-status', (status) => {
      this.connectionStatus = status
    })
  }

  private setupGestureHandlers() {
    if (!this.gestureEnabled) return

    this.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: true })
    this.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: true })
  }

  private handleTouchStart(event: TouchEvent) {
    if (event.touches.length === 1) {
      this.touchStartX = event.touches[0].clientX
      this.touchStartY = event.touches[0].clientY
    }
  }

  private handleTouchEnd(event: TouchEvent) {
    if (event.changedTouches.length === 1) {
      const deltaX = event.changedTouches[0].clientX - this.touchStartX
      const deltaY = event.changedTouches[0].clientY - this.touchStartY
      
      if (Math.abs(deltaX) > this.gestureThreshold && Math.abs(deltaX) > Math.abs(deltaY)) {
        if (deltaX > 0) {
          this.handleSwipeRight()
        } else {
          this.handleSwipeLeft()
        }
      } else if (Math.abs(deltaY) > this.gestureThreshold && Math.abs(deltaY) > Math.abs(deltaX)) {
        if (deltaY < 0) {
          this.handleSwipeUp()
        }
      }
    }
  }

  private handleSwipeRight() {
    // Approve/activate selected agents
    if (this.selectedAgents.size > 0) {
      this.executeGestureCommand({
        gesture: 'swipe-right',
        action: 'approve',
        confirmation: false
      })
      this.showGestureHint('✅ Agents approved')
    }
  }

  private handleSwipeLeft() {
    // Pause selected agents for review
    if (this.selectedAgents.size > 0) {
      this.executeGestureCommand({
        gesture: 'swipe-left',
        action: 'pause',
        confirmation: false
      })
      this.showGestureHint('⏸️ Agents paused for review')
    }
  }

  private handleSwipeUp() {
    // Escalate to human intervention
    this.executeGestureCommand({
      gesture: 'swipe-up',
      action: 'escalate',
      confirmation: true
    })
    this.showGestureHint('🚨 Escalated to human intervention')
  }

  private async executeGestureCommand(command: GestureCommand) {
    try {
      if (command.confirmation) {
        const confirmed = confirm(`Confirm action: ${command.action}?`)
        if (!confirmed) return
      }

      // Execute the gesture command
      await this.performBulkAction(command.action)
      
    } catch (error) {
      console.error('Gesture command failed:', error)
      this.showTemporaryMessage('Gesture command failed', 'error')
    }
  }

  private showGestureHint(message: string) {
    const hint = document.createElement('div')
    hint.className = 'gesture-hint'
    hint.textContent = message
    this.shadowRoot?.appendChild(hint)
    
    setTimeout(() => {
      hint.remove()
    }, 2000)
  }

  private async loadInitialData() {
    this.loading = true
    this.error = null
    
    try {
      // Load agent system status
      const status = await this.agentService.getAgentSystemStatus(false)
      this.updateSystemStatus(status)
      
      // Load agents
      this.agents = this.agentService.getAgents()
      this.agentSummary = this.agentService.getAgentSummary()
      this.teamComposition = this.agentService.getTeamComposition()
      
      // Check system readiness
      const readiness = this.agentService.isSystemReady()
      this.systemReady = readiness.ready
      
      // Generate initial alerts if system not ready
      if (!readiness.ready) {
        this.generateReadinessAlerts(readiness)
      }
      
      this.lastUpdate = new Date()
      
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to load agent data'
      console.error('Failed to load initial data:', error)
    } finally {
      this.loading = false
    }
  }

  private startRealTimeUpdates() {
    // Start agent monitoring
    this.agentService.startMonitoring()
    
    // Refresh data every 10 seconds
    this.refreshInterval = window.setInterval(() => {
      this.refreshData()
    }, 10000)
    
    // Update performance metrics every 5 seconds
    this.performanceUpdateInterval = window.setInterval(() => {
      this.updatePerformanceMetrics()
    }, 5000)
  }

  private async refreshData() {
    try {
      await this.agentService.getAgentSystemStatus(false)
      this.agents = this.agentService.getAgents()
      this.agentSummary = this.agentService.getAgentSummary()
      this.teamComposition = this.agentService.getTeamComposition()
      this.lastUpdate = new Date()
      this.connectionStatus = 'connected'
    } catch (error) {
      this.connectionStatus = 'disconnected'
      console.error('Failed to refresh data:', error)
    }
  }

  private updateSystemStatus(status: any) {
    this.systemReady = status.system_ready
    this.connectionStatus = status.active ? 'connected' : 'disconnected'
  }

  private handleAgentStatusUpdate(data: any) {
    const agent = this.agents.find(a => a.id === data.agentId)
    if (agent) {
      agent.status = data.status
      agent.last_activity = data.last_activity
      this.requestUpdate()
    }
  }

  private handlePerformanceUpdate(data: any) {
    this.performanceMetrics.set(data.agentId, data.metrics)
    this.requestUpdate()
  }

  private addPriorityAlert(alert: PriorityAlert) {
    this.priorityAlerts.unshift(alert)
    // Keep only last 20 alerts
    if (this.priorityAlerts.length > 20) {
      this.priorityAlerts = this.priorityAlerts.slice(0, 20)
    }
    this.requestUpdate()
  }

  private generateReadinessAlerts(readiness: any) {
    readiness.recommendations.forEach((recommendation: string, index: number) => {
      this.addPriorityAlert({
        id: `readiness-${index}`,
        type: 'high',
        title: 'System Readiness Issue',
        message: recommendation,
        timestamp: new Date().toISOString(),
        actions: [
          {
            id: 'fix',
            label: 'Fix Now',
            command: '/hive:fix-readiness',
            style: 'primary'
          }
        ]
      })
    })
  }

  private updatePerformanceMetrics() {
    this.agents.forEach(agent => {
      // Simulate performance metrics updates
      const metrics = {
        cpuUsage: Math.random() * 100,
        memoryUsage: Math.random() * 100,
        tasksCompleted: agent.performance_metrics.tasks_completed + Math.floor(Math.random() * 3),
        averageTaskTime: 2000 + Math.random() * 3000,
        successRate: 0.85 + Math.random() * 0.15,
        lastUpdated: new Date().toISOString()
      }
      this.performanceMetrics.set(agent.id, metrics)
    })
    this.requestUpdate()
  }

  private showTemporaryMessage(message: string, type: 'success' | 'error' | 'info' = 'info') {
    const alertType = type === 'success' ? 'info' : type === 'error' ? 'critical' : 'medium'
    this.addPriorityAlert({
      id: `temp-${Date.now()}`,
      type: alertType,
      title: 'System Update',
      message,
      timestamp: new Date().toISOString(),
      actions: []
    })
  }

  private async performBulkAction(action: string) {
    if (this.selectedAgents.size === 0) return
    
    const agentIds = Array.from(this.selectedAgents)
    
    try {
      await this.agentService.performBulkOperation(agentIds, action as any)
      this.selectedAgents.clear()
      this.bulkSelectionMode = false
      this.showTemporaryMessage(`Bulk ${action} completed successfully`, 'success')
    } catch (error) {
      this.showTemporaryMessage(`Bulk ${action} failed`, 'error')
    }
  }

  private toggleAgentSelection(agentId: string) {
    if (this.selectedAgents.has(agentId)) {
      this.selectedAgents.delete(agentId)
    } else {
      this.selectedAgents.add(agentId)
    }
    
    if (this.selectedAgents.size === 0) {
      this.bulkSelectionMode = false
    }
    
    this.requestUpdate()
  }

  private async executeQuickAction(action: string) {
    try {
      switch (action) {
        case 'activate-team':
          await this.agentService.activateAgentTeam()
          break
        case 'spawn-architect':
          await this.agentService.spawnAgent(AgentRole.ARCHITECT)
          break
        case 'spawn-frontend':
          await this.agentService.spawnAgent(AgentRole.FRONTEND_DEVELOPER)
          break
        case 'system-health':
          const readiness = this.agentService.isSystemReady()
          this.showTemporaryMessage(`System ready: ${readiness.ready}`, readiness.ready ? 'success' : 'error')
          break
      }
      this.showQuickActions = false
    } catch (error) {
      this.showTemporaryMessage('Quick action failed', 'error')
    }
  }

  private async executeAlertAction(alert: PriorityAlert, action: AlertAction) {
    try {
      // Execute alert action command
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: action.command,
          context: { alertId: alert.id, agentId: alert.agentId }
        })
      })
      
      if (response.ok) {
        // Remove the alert
        this.priorityAlerts = this.priorityAlerts.filter(a => a.id !== alert.id)
        this.showTemporaryMessage(`Action ${action.label} completed`, 'success')
      } else {
        throw new Error('Action failed')
      }
    } catch (error) {
      this.showTemporaryMessage(`Action ${action.label} failed`, 'error')
    }
  }

  private getAgentRoleIcon(role: AgentRole): string {
    const icons = {
      [AgentRole.PRODUCT_MANAGER]: '📋',
      [AgentRole.ARCHITECT]: '🏗️',
      [AgentRole.BACKEND_DEVELOPER]: '⚙️',
      [AgentRole.FRONTEND_DEVELOPER]: '🎨',
      [AgentRole.QA_ENGINEER]: '🔍'
    }
    return icons[role] || '🤖'
  }

  private getStatusColor(status: AgentStatus): string {
    const colors = {
      [AgentStatus.ACTIVE]: 'success',
      [AgentStatus.IDLE]: 'warning',
      [AgentStatus.BUSY]: 'info',
      [AgentStatus.ERROR]: 'error',
      [AgentStatus.OFFLINE]: 'error'
    }
    return colors[status] || 'info'
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

  private cleanup() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval)
    }
    if (this.performanceUpdateInterval) {
      clearInterval(this.performanceUpdateInterval)
    }
    this.agentService.stopMonitoring()
  }

  render() {
    return html`
      <div class="dashboard-container">
        ${this.renderHeader()}
        ${this.renderPriorityFilter()}
        ${this.renderMainContent()}
        ${this.renderQuickActionsOverlay()}
        ${this.loading ? this.renderLoadingOverlay() : ''}
      </div>
    `
  }

  private renderHeader() {
    const criticalAlerts = this.priorityAlerts.filter(a => a.type === 'critical').length
    const highAlerts = this.priorityAlerts.filter(a => a.type === 'high').length
    
    return html`
      <div class="dashboard-header">
        <div class="header-title">
          <div class="system-status-indicator ${this.systemReady ? '' : 'warning'} ${this.connectionStatus === 'disconnected' ? 'disconnected' : ''}"></div>
          <span>🤖 Agent Management</span>
        </div>
        
        <div class="header-metrics">
          <div class="metric-item">
            <span>Agents:</span>
            <span class="metric-value">${this.agents.length}</span>
          </div>
          <div class="metric-item">
            <span>Active:</span>
            <span class="metric-value success">${this.agentSummary?.active || 0}</span>
          </div>
          <div class="metric-item">
            <span>Ready:</span>
            <span class="metric-value ${this.systemReady ? 'success' : 'warning'}">${this.systemReady ? 'Yes' : 'No'}</span>
          </div>
        </div>
        
        <div class="header-actions">
          <button class="header-button ${this.filterPriority !== 'all' ? 'active' : ''}" 
                  @click=${() => this.showQuickActions = true}
                  title="Quick Actions">
            ⚡
          </button>
          <button class="header-button ${this.bulkSelectionMode ? 'active' : ''}" 
                  @click=${() => { this.bulkSelectionMode = !this.bulkSelectionMode }}
                  title="Bulk Selection">
            ☑️
          </button>
          <button class="header-button" 
                  @click=${this.refreshData}
                  title="Refresh">
            🔄
            ${criticalAlerts + highAlerts > 0 ? html`
              <div class="alert-badge">${criticalAlerts + highAlerts}</div>
            ` : ''}
          </button>
        </div>
      </div>
    `
  }

  private renderPriorityFilter() {
    return html`
      <div class="priority-filter-bar">
        <button class="filter-chip ${this.filterPriority === 'all' ? 'active' : ''}" 
                @click=${() => { this.filterPriority = 'all' }}>
          All
        </button>
        <button class="filter-chip critical ${this.filterPriority === 'critical' ? 'active' : ''}" 
                @click=${() => { this.filterPriority = 'critical' }}>
          Critical
        </button>
        <button class="filter-chip high ${this.filterPriority === 'high' ? 'active' : ''}" 
                @click=${() => { this.filterPriority = 'high' }}>
          High
        </button>
        <button class="filter-chip ${this.filterPriority === 'medium' ? 'active' : ''}" 
                @click=${() => { this.filterPriority = 'medium' }}>
          Medium
        </button>
      </div>
    `
  }

  private renderMainContent() {
    return html`
      <div class="main-content">
        <div class="content-area">
          ${this.error ? html`
            <div class="error-banner">
              ${this.error}
            </div>
          ` : ''}
          
          ${this.renderSystemOverview()}
          ${this.renderPriorityAlerts()}
          ${this.renderBulkSelectionBar()}
          ${this.renderAgentGrid()}
        </div>
      </div>
    `
  }

  private renderSystemOverview() {
    return html`
      <div class="system-overview-card">
        <div class="overview-header">
          <div class="overview-title">
            📊 System Overview
          </div>
          <button class="quick-actions-button" @click=${() => { this.showQuickActions = true }}>
            Quick Actions
          </button>
        </div>
        
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-value success">${this.agentSummary?.active || 0}</div>
            <div class="metric-label">Active</div>
          </div>
          <div class="metric-card">
            <div class="metric-value warning">${this.agentSummary?.idle || 0}</div>
            <div class="metric-label">Idle</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${this.agentSummary?.busy || 0}</div>
            <div class="metric-label">Busy</div>
          </div>
          <div class="metric-card">
            <div class="metric-value error">${this.agentSummary?.error || 0}</div>
            <div class="metric-label">Error</div>
          </div>
        </div>
      </div>
    `
  }

  private renderPriorityAlerts() {
    const filteredAlerts = this.filterPriority === 'all' 
      ? this.priorityAlerts 
      : this.priorityAlerts.filter(alert => alert.type === this.filterPriority)
    
    if (filteredAlerts.length === 0) return html``
    
    return html`
      <div class="priority-alerts-container">
        ${filteredAlerts.slice(0, 5).map(alert => html`
          <div class="alert-card ${alert.type}">
            <div class="alert-header">
              <div class="alert-title">${alert.title}</div>
              <div class="alert-priority ${alert.type}">${alert.type}</div>
            </div>
            <div class="alert-message">${alert.message}</div>
            ${alert.actions.length > 0 ? html`
              <div class="alert-actions">
                ${alert.actions.map(action => html`
                  <button class="alert-action-button ${action.style}" 
                          @click=${() => this.executeAlertAction(alert, action)}>
                    ${action.label}
                  </button>
                `)}
              </div>
            ` : ''}
          </div>
        `)}
      </div>
    `
  }

  private renderBulkSelectionBar() {
    if (!this.bulkSelectionMode || this.selectedAgents.size === 0) return html``
    
    return html`
      <div class="bulk-selection-bar">
        <div class="bulk-selection-info">
          ${this.selectedAgents.size} agents selected
        </div>
        <div class="bulk-actions">
          <button class="bulk-action-button" @click=${() => this.performBulkAction('restart')}>
            Restart
          </button>
          <button class="bulk-action-button" @click=${() => this.performBulkAction('pause')}>
            Pause
          </button>
          <button class="bulk-action-button" @click=${() => this.performBulkAction('resume')}>
            Resume
          </button>
          <button class="bulk-action-button" @click=${() => { this.selectedAgents.clear(); this.requestUpdate() }}>
            Clear
          </button>
        </div>
      </div>
    `
  }

  private renderAgentGrid() {
    return html`
      <div class="agent-grid">
        ${this.agents.map(agent => this.renderAgentCard(agent))}
      </div>
    `
  }

  private renderAgentCard(agent: Agent) {
    const isSelected = this.selectedAgents.has(agent.id)
    const metrics = this.performanceMetrics.get(agent.id)
    
    return html`
      <div class="agent-card ${isSelected ? 'selected' : ''}" 
           @click=${() => this.bulkSelectionMode ? this.toggleAgentSelection(agent.id) : null}>
        <div class="agent-header">
          <div class="agent-info">
            <div class="agent-avatar ${agent.role}">
              ${this.getAgentRoleIcon(agent.role)}
            </div>
            <div class="agent-meta">
              <div class="agent-name">${agent.name}</div>
              <div class="agent-role">${agent.role.replace('_', ' ')}</div>
            </div>
          </div>
          <div class="agent-status">
            <div class="status-indicator ${agent.status}"></div>
            <div class="status-text">${agent.status}</div>
          </div>
        </div>
        
        ${metrics ? html`
          <div class="agent-performance">
            <div class="performance-metric">
              <div class="performance-value">${Math.round(metrics.cpuUsage)}%</div>
              <div class="performance-label">CPU</div>
            </div>
            <div class="performance-metric">
              <div class="performance-value">${Math.round(metrics.memoryUsage)}%</div>
              <div class="performance-label">Memory</div>
            </div>
            <div class="performance-metric">
              <div class="performance-value">${Math.round(metrics.successRate * 100)}%</div>
              <div class="performance-label">Success</div>
            </div>
          </div>
        ` : ''}
        
        <div class="agent-actions">
          <button class="agent-action-button primary" 
                  @click=${(e: Event) => { e.stopPropagation(); this.selectedAgent = agent }}>
            Configure
          </button>
          <button class="agent-action-button" 
                  @click=${(e: Event) => { e.stopPropagation(); /* View details */ }}>
            Details
          </button>
          <button class="agent-action-button danger" 
                  @click=${(e: Event) => { e.stopPropagation(); this.agentService.deactivateAgent(agent.id) }}>
            Stop
          </button>
        </div>
      </div>
    `
  }

  private renderQuickActionsOverlay() {
    if (!this.showQuickActions) return html``
    
    return html`
      <div class="quick-actions-overlay" @click=${() => { this.showQuickActions = false }}>
        <div class="quick-actions-panel" @click=${(e: Event) => e.stopPropagation()}>
          <div class="quick-actions-header">
            <div class="quick-actions-title">⚡ Quick Actions</div>
            <button class="close-button" @click=${() => { this.showQuickActions = false }}>
              ✕
            </button>
          </div>
          
          <div class="quick-actions-grid">
            <div class="quick-action-item" @click=${() => this.executeQuickAction('activate-team')}>
              <div class="quick-action-icon">🚀</div>
              <div class="quick-action-title">Activate Team</div>
              <div class="quick-action-description">Spawn 5-agent development team</div>
            </div>
            <div class="quick-action-item" @click=${() => this.executeQuickAction('spawn-architect')}>
              <div class="quick-action-icon">🏗️</div>
              <div class="quick-action-title">Add Architect</div>
              <div class="quick-action-description">Spawn system architect agent</div>
            </div>
            <div class="quick-action-item" @click=${() => this.executeQuickAction('spawn-frontend')}>
              <div class="quick-action-icon">🎨</div>
              <div class="quick-action-title">Add Frontend</div>
              <div class="quick-action-description">Spawn frontend developer agent</div>
            </div>
            <div class="quick-action-item" @click=${() => this.executeQuickAction('system-health')}>
              <div class="quick-action-icon">🏥</div>
              <div class="quick-action-title">System Health</div>
              <div class="quick-action-description">Check system readiness</div>
            </div>
          </div>
        </div>
      </div>
    `
  }

  private renderLoadingOverlay() {
    return html`
      <div class="loading-overlay">
        <div class="loading-spinner"></div>
      </div>
    `
  }
}