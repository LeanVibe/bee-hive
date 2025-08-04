import { LitElement, html, css } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { AgentStatus } from '../components/dashboard/agent-health-panel'
import { getAgentService } from '../services'
import type { Agent, AgentActivationOptions } from '../services'
import { AgentRole } from '../types/api'
import '../components/dashboard/agent-health-panel'
import '../components/common/loading-spinner'
import '../components/modals/agent-config-modal'

@customElement('agents-view')
export class AgentsView extends LitElement {
  @state() private agents: AgentStatus[] = []
  @state() private isLoading: boolean = true
  @state() private error: string = ''
  @state() private selectedAgent: AgentStatus | null = null
  @state() private selectedAgents: Set<string> = new Set()
  @state() private agentService = getAgentService()
  @state() private monitoringActive: boolean = false
  @state() private showAgentConfigModal: boolean = false
  @state() private configModalMode: 'create' | 'edit' = 'create'
  @state() private configModalAgent?: Agent
  @state() private bulkActionMode: boolean = false
  @state() private viewMode: 'grid' | 'list' = 'grid'

  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: #f9fafb;
    }

    .agents-container {
      display: flex;
      flex-direction: column;
      height: 100%;
      max-width: 1400px;
      margin: 0 auto;
    }

    .agents-header {
      background: white;
      border-bottom: 1px solid #e5e7eb;
      padding: 2rem;
      position: sticky;
      top: 0;
      z-index: 10;
    }

    .header-content {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .header-actions {
      display: flex;
      gap: 1rem;
      align-items: center;
      flex-wrap: wrap;
    }

    .team-controls {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }

    .action-button {
      background: #3b82f6;
      color: white;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .action-button:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }

    .action-button.secondary {
      background: white;
      color: #374151;
      border: 1px solid #d1d5db;
    }

    .action-button.secondary:hover {
      background: #f9fafb;
      border-color: #9ca3af;
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

    .action-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }

    .bulk-action-bar {
      background: #f0f9ff;
      border: 1px solid #bae6fd;
      border-radius: 0.75rem;
      padding: 1rem 1.5rem;
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
    }

    .bulk-info {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      color: #0369a1;
      font-weight: 500;
    }

    .bulk-actions {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }

    .view-toggle {
      display: flex;
      background: #f3f4f6;
      border-radius: 0.5rem;
      padding: 0.25rem;
    }

    .view-button {
      background: none;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      font-weight: 500;
      color: #6b7280;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .view-button.active {
      background: white;
      color: #3b82f6;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    .page-title {
      font-size: 1.875rem;
      font-weight: 700;
      color: #111827;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .title-icon {
      width: 32px;
      height: 32px;
      color: #3b82f6;
    }

    .agents-summary {
      display: flex;
      gap: 2rem;
      align-items: center;
      flex-wrap: wrap;
    }

    .summary-item {
      text-align: center;
    }

    .summary-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin: 0;
    }

    .summary-label {
      font-size: 0.875rem;
      color: #6b7280;
      margin: 0;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }

    .summary-item.active .summary-value {
      color: #10b981;
    }

    .summary-item.idle .summary-value {
      color: #f59e0b;
    }

    .summary-item.error .summary-value {
      color: #ef4444;
    }

    .agents-content {
      flex: 1;
      padding: 2rem;
      overflow-y: auto;
    }

    .agents-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
      gap: 1.5rem;
      margin-bottom: 2rem;
    }

    .agent-card {
      background: white;
      border-radius: 0.75rem;
      border: 1px solid #e5e7eb;
      padding: 1.5rem;
      transition: all 0.2s ease;
      cursor: pointer;
      position: relative;
      overflow: hidden;
    }

    .agent-card:hover {
      border-color: #3b82f6;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      transform: translateY(-1px);
    }

    .agent-card.selected {
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .agent-card.bulk-selected {
      border-color: #10b981;
      box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
      background: #f0fdf4;
    }

    .agent-checkbox {
      position: absolute;
      top: 1rem;
      left: 1rem;
      width: 18px;
      height: 18px;
      cursor: pointer;
      z-index: 2;
    }

    .agent-list {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .agent-list-item {
      background: white;
      border-radius: 0.75rem;
      border: 1px solid #e5e7eb;
      padding: 1.5rem;
      transition: all 0.2s ease;
      cursor: pointer;
      position: relative;
    }

    .agent-list-item:hover {
      border-color: #3b82f6;
      box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    }

    .agent-list-item.bulk-selected {
      border-color: #10b981;
      box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
      background: #f0fdf4;
    }

    .agent-list-content {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 2rem;
      align-items: center;
    }

    .agent-list-info {
      display: grid;
      grid-template-columns: 200px 1fr 150px;
      gap: 2rem;
      align-items: center;
    }

    .agent-basic-info {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .agent-list-name {
      font-size: 1.125rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
    }

    .agent-list-role {
      font-size: 0.875rem;
      color: #6b7280;
      margin: 0;
      text-transform: capitalize;
    }

    .agent-list-metrics {
      display: flex;
      align-items: center;
      gap: 2rem;
      font-size: 0.875rem;
      color: #6b7280;
    }

    .agent-list-status {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.5rem;
    }

    .status-badge {
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 500;
      text-transform: capitalize;
    }

    .status-badge.active {
      background: #d1fae5;
      color: #065f46;
    }

    .status-badge.idle {
      background: #fef3c7;
      color: #92400e;
    }

    .status-badge.error {
      background: #fee2e2;
      color: #991b1b;
    }

    .status-badge.offline {
      background: #f3f4f6;
      color: #374151;
    }

    .agent-list-actions {
      display: flex;
      gap: 0.5rem;
    }

    .agent-status-indicator {
      position: absolute;
      top: 1rem;
      right: 1rem;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      border: 2px solid white;
    }

    .agent-status-indicator.active {
      background: #10b981;
      animation: pulse 2s infinite;
    }

    .agent-status-indicator.idle {
      background: #f59e0b;
    }

    .agent-status-indicator.error {
      background: #ef4444;
    }

    .agent-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .agent-name {
      font-size: 1.125rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
    }

    .agent-type {
      font-size: 0.75rem;
      color: #6b7280;
      background: #f3f4f6;
      padding: 0.25rem 0.5rem;
      border-radius: 0.375rem;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }

    .agent-metrics {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .metric-item {
      text-align: center;
    }

    .metric-value {
      font-size: 1.25rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
    }

    .metric-label {
      font-size: 0.75rem;
      color: #6b7280;
      margin: 0;
    }

    .agent-current-task {
      background: #f8fafc;
      border-radius: 0.5rem;
      padding: 0.75rem;
      margin-top: 1rem;
    }

    .current-task-label {
      font-size: 0.75rem;
      color: #6b7280;
      margin: 0 0 0.25rem 0;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }

    .current-task-text {
      font-size: 0.875rem;
      color: #374151;
      margin: 0;
      font-weight: 500;
    }

    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 60vh;
      text-align: center;
      color: #6b7280;
    }

    .empty-icon {
      width: 64px;
      height: 64px;
      color: #d1d5db;
      margin-bottom: 1rem;
    }

    .empty-title {
      font-size: 1.25rem;
      font-weight: 600;
      color: #374151;
      margin: 0 0 0.5rem 0;
    }

    .empty-description {
      font-size: 0.875rem;
      margin: 0 0 1.5rem 0;
      max-width: 400px;
    }

    .refresh-button {
      background: #3b82f6;
      color: white;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .refresh-button:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }
    
    .agent-controls {
      display: flex;
      gap: 0.5rem;
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid #f3f4f6;
    }
    
    .control-button {
      flex: 1;
      padding: 0.5rem 1rem;
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      background: white;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.25rem;
    }
    
    .control-button:hover {
      border-color: #9ca3af;
      transform: translateY(-1px);
    }
    
    .control-button.activate {
      color: #059669;
      border-color: #059669;
    }
    
    .control-button.activate:hover {
      background: #ecfdf5;
      border-color: #047857;
    }
    
    .control-button.deactivate {
      color: #dc2626;
      border-color: #dc2626;
    }
    
    .control-button.deactivate:hover {
      background: #fef2f2;
      border-color: #b91c1c;
    }
    
    .control-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }
    
    .control-button:disabled:hover {
      border-color: #d1d5db;
      background: white;
      transform: none;
    }

    .loading-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 60vh;
      gap: 1rem;
    }

    .error-state {
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #dc2626;
      padding: 1rem;
      border-radius: 0.5rem;
      margin: 1rem;
      text-align: center;
    }

    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.5;
      }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .agents-header {
        padding: 1rem;
      }

      .page-title {
        font-size: 1.5rem;
      }

      .agents-content {
        padding: 1rem;
      }

      .agents-grid {
        grid-template-columns: 1fr;
        gap: 1rem;
      }

      .agents-summary {
        gap: 1rem;
      }

      .summary-item {
        min-width: 80px;
      }
    }

    @media (max-width: 640px) {
      .header-content {
        flex-direction: column;
        align-items: flex-start;
      }

      .agents-summary {
        width: 100%;
        justify-content: space-around;
      }
    }
  `

  async connectedCallback() {
    super.connectedCallback()
    await this.initializeAgentService()
    await this.loadAgents()
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    this.stopMonitoring()
  }
  
  /**
   * Initialize agent service with real-time monitoring
   */
  private async initializeAgentService() {
    try {
      // Set up event listeners for real-time updates
      this.agentService.addEventListener('agentsChanged', this.handleAgentsChanged.bind(this))
      this.agentService.addEventListener('agentActivated', this.handleAgentActivated.bind(this))
      this.agentService.addEventListener('agentDeactivated', this.handleAgentDeactivated.bind(this))
      
      // Start monitoring for real-time updates
      this.startMonitoring()
      
      console.log('Agent service initialized successfully')
      
    } catch (error) {
      console.error('Failed to initialize agent service:', error)
      this.error = 'Failed to initialize agent service'
    }
  }
  
  /**
   * Start real-time monitoring
   */
  private startMonitoring() {
    if (this.monitoringActive) return
    
    try {
      this.agentService.startMonitoring()
      this.monitoringActive = true
      console.log('Agent monitoring started')
      
    } catch (error) {
      console.error('Failed to start agent monitoring:', error)
    }
  }
  
  /**
   * Stop monitoring
   */
  private stopMonitoring() {
    if (!this.monitoringActive) return
    
    try {
      this.agentService.stopMonitoring()
      this.monitoringActive = false
      console.log('Agent monitoring stopped')
      
    } catch (error) {
      console.error('Failed to stop agent monitoring:', error)
    }
  }
  
  /**
   * Real-time event handlers
   */
  private handleAgentsChanged = (event: CustomEvent) => {
    this.agents = event.detail.agents.map(this.transformAgentToUIFormat)
    console.log('Agents updated via real-time:', event.detail.agents.length, 'agents')
  }
  
  private handleAgentActivated = (event: CustomEvent) => {
    const activatedAgent = this.transformAgentToUIFormat(event.detail.agent)
    const index = this.agents.findIndex(a => a.id === activatedAgent.id)
    if (index >= 0) {
      this.agents[index] = activatedAgent
    } else {
      this.agents = [...this.agents, activatedAgent]
    }
    console.log('Agent activated:', activatedAgent.name)
  }
  
  private handleAgentDeactivated = (event: CustomEvent) => {
    const agentId = event.detail.agentId
    this.agents = this.agents.filter(a => a.id !== agentId)
    if (this.selectedAgent?.id === agentId) {
      this.selectedAgent = null
    }
    console.log('Agent deactivated:', agentId)
  }
  
  /**
   * Transform Agent API data to UI AgentStatus format
   */
  private transformAgentToUIFormat = (agent: Agent): AgentStatus => {
    return {
      id: agent.id,
      name: agent.name || `Agent ${agent.id}`,
      status: agent.status === 'active' ? 'active' : 
             agent.status === 'idle' ? 'idle' : 
             agent.status === 'error' ? 'error' : 'offline',
      uptime: agent.performance_metrics?.uptime || 0,
      lastSeen: agent.last_seen || new Date().toISOString(),
      currentTask: agent.current_task_id || undefined,
      metrics: {
        cpuUsage: agent.performance_metrics?.cpu_usage || [],
        memoryUsage: agent.performance_metrics?.memory_usage || [],
        tokenUsage: agent.performance_metrics?.token_usage || [],
        tasksCompleted: agent.performance_metrics?.tasks_completed || [],
        errorRate: agent.performance_metrics?.error_rate || [],
        responseTime: agent.performance_metrics?.response_time || [],
        timestamps: agent.performance_metrics?.timestamps || []
      },
      performance: {
        score: agent.performance_metrics?.overall_score || 85,
        trend: agent.performance_metrics?.trend || 'stable'
      }
    }
  }

  private async loadAgents() {
    this.isLoading = true
    this.error = ''

    try {
      // Load agents using integrated service
      const agentData = await this.agentService.getAgents()
      
      // Transform to UI format
      this.agents = agentData.map(this.transformAgentToUIFormat)
      
      console.log('Loaded', this.agents.length, 'agents from service')

    } catch (error) {
      console.error('Failed to load agents:', error)
      this.error = error instanceof Error ? error.message : 'Failed to load agents'
      
      // Fall back to mock data for demonstration
      this.agents = [
        {
          id: 'agent-001',
          name: 'Architect Agent',
          status: 'active',
          uptime: 86400000, // 24 hours in ms
          lastSeen: new Date(Date.now() - 5000).toISOString(),
          currentTask: 'Designing microservice architecture for payment system',
          metrics: {
            cpuUsage: [45, 52, 48, 55, 49],
            memoryUsage: [67, 72, 69, 74, 71],
            tokenUsage: [1250, 1380, 1420, 1560, 1490],
            tasksCompleted: [5, 8, 12, 15, 18],
            errorRate: [0, 0, 1, 0, 0],
            responseTime: [250, 280, 245, 290, 265],
            timestamps: ['09:00', '10:00', '11:00', '12:00', '13:00']
          },
          performance: {
            score: 92,
            trend: 'up'
          }
        },
        {
          id: 'agent-002',
          name: 'Developer Agent',
          status: 'active',
          uptime: 72000000, // 20 hours in ms
          lastSeen: new Date(Date.now() - 30000).toISOString(),
          currentTask: 'Implementing REST API endpoints',
          metrics: {
            cpuUsage: [38, 42, 45, 41, 39],
            memoryUsage: [55, 58, 61, 57, 54],
            tokenUsage: [980, 1120, 1050, 1180, 1090],
            tasksCompleted: [8, 12, 15, 18, 22],
            errorRate: [1, 0, 0, 0, 1],
            responseTime: [180, 195, 210, 185, 175],
            timestamps: ['09:00', '10:00', '11:00', '12:00', '13:00']
          },
          performance: {
            score: 88,
            trend: 'stable'
          }
        },
        {
          id: 'agent-003',
          name: 'Tester Agent',
          status: 'idle',
          uptime: 43200000, // 12 hours in ms
          lastSeen: new Date(Date.now() - 300000).toISOString(),
          currentTask: undefined,
          metrics: {
            cpuUsage: [15, 18, 12, 20, 16],
            memoryUsage: [32, 35, 28, 38, 31],
            tokenUsage: [450, 520, 480, 560, 510],
            tasksCompleted: [3, 5, 7, 9, 11],
            errorRate: [0, 1, 0, 0, 0],
            responseTime: [120, 135, 140, 125, 130],
            timestamps: ['09:00', '10:00', '11:00', '12:00', '13:00']
          },
          performance: {
            score: 75,
            trend: 'stable'
          }
        }
      ]
    } finally {
      this.isLoading = false
    }
  }

  private handleAgentSelect(agent: AgentStatus) {
    this.selectedAgent = this.selectedAgent?.id === agent.id ? null : agent
  }

  private async handleRefresh() {
    console.log('Manual agent refresh triggered')
    await this.loadAgents()
  }

  /**
   * Toggle bulk action mode
   */
  private handleToggleBulkMode() {
    this.bulkActionMode = !this.bulkActionMode
    if (!this.bulkActionMode) {
      this.selectedAgents.clear()
    }
    this.requestUpdate()
  }

  /**
   * Toggle agent selection for bulk operations
   */
  private handleAgentSelection(agentId: string, event?: Event) {
    if (event) {
      event.stopPropagation()
    }

    if (this.selectedAgents.has(agentId)) {
      this.selectedAgents.delete(agentId)
    } else {
      this.selectedAgents.add(agentId)
    }
    this.requestUpdate()
  }

  /**
   * Select all agents
   */
  private handleSelectAll() {
    if (this.selectedAgents.size === this.agents.length) {
      this.selectedAgents.clear()
    } else {
      this.selectedAgents.clear()
      this.agents.forEach(agent => this.selectedAgents.add(agent.id))
    }
    this.requestUpdate()
  }

  /**
   * Activate multiple agents
   */
  private async handleBulkActivate() {
    if (this.selectedAgents.size === 0) return

    this.isLoading = true
    this.error = ''

    try {
      const promises = Array.from(this.selectedAgents).map(agentId => 
        this.handleActivateAgent(agentId)
      )
      
      await Promise.all(promises)
      
      console.log(`Bulk activated ${this.selectedAgents.size} agents`)
      this.selectedAgents.clear()
      this.bulkActionMode = false

    } catch (error) {
      console.error('Failed to bulk activate agents:', error)
      this.error = error instanceof Error ? error.message : 'Failed to bulk activate agents'
    } finally {
      this.isLoading = false
    }
  }

  /**
   * Deactivate multiple agents
   */
  private async handleBulkDeactivate() {
    if (this.selectedAgents.size === 0) return

    this.isLoading = true
    this.error = ''

    try {
      const promises = Array.from(this.selectedAgents).map(agentId => 
        this.handleDeactivateAgent(agentId)
      )
      
      await Promise.all(promises)
      
      console.log(`Bulk deactivated ${this.selectedAgents.size} agents`)
      this.selectedAgents.clear()
      this.bulkActionMode = false

    } catch (error) {
      console.error('Failed to bulk deactivate agents:', error)
      this.error = error instanceof Error ? error.message : 'Failed to bulk deactivate agents'
    } finally {
      this.isLoading = false
    }
  }

  /**
   * Activate agent system with team configuration
   */
  private async handleActivateTeam() {
    this.isLoading = true
    this.error = ''

    try {
      const options: AgentActivationOptions = {
        teamSize: 5,
        roles: [AgentRole.PRODUCT_MANAGER, AgentRole.BACKEND_DEVELOPER, AgentRole.FRONTEND_DEVELOPER, AgentRole.QA_ENGINEER, AgentRole.DEVOPS_ENGINEER],
        autoStartTasks: true
      }
      
      await this.agentService.activateAgentSystem(options)
      
      console.log('Agent team activated successfully')
      await this.loadAgents()

    } catch (error) {
      console.error('Failed to activate agent team:', error)
      this.error = error instanceof Error ? error.message : 'Failed to activate agent team'
    } finally {
      this.isLoading = false
    }
  }

  /**
   * Deactivate entire agent system
   */
  private async handleDeactivateTeam() {
    this.isLoading = true
    this.error = ''

    try {
      await this.agentService.deactivateAgentSystem()
      
      console.log('Agent team deactivated successfully')
      await this.loadAgents()

    } catch (error) {
      console.error('Failed to deactivate agent team:', error)
      this.error = error instanceof Error ? error.message : 'Failed to deactivate agent team'
    } finally {
      this.isLoading = false
    }
  }

  /**
   * Show agent configuration modal
   */
  private handleShowAgentConfig(mode: 'create' | 'edit', agent?: Agent) {
    this.configModalMode = mode
    this.configModalAgent = agent
    this.showAgentConfigModal = true
  }

  /**
   * Handle agent configuration save
   */
  private async handleAgentConfigSave(event: CustomEvent) {
    const { mode, config } = event.detail

    this.isLoading = true
    this.error = ''

    try {
      if (mode === 'create') {
        // Create new agent - this would need backend support
        console.log('Creating new agent with config:', config)
        // await this.agentService.createAgent(config)
      } else {
        // Update existing agent - this would need backend support
        console.log('Updating agent with config:', config)
        // await this.agentService.updateAgent(this.configModalAgent.id, config)
      }

      await this.loadAgents()

    } catch (error) {
      console.error('Failed to save agent configuration:', error)
      this.error = error instanceof Error ? error.message : 'Failed to save agent configuration'
    } finally {
      this.isLoading = false
    }
  }

  /**
   * Change view mode
   */
  private handleViewModeChange(mode: 'grid' | 'list') {
    this.viewMode = mode
  }
  
  /**
   * Activate an agent
   */
  private async handleActivateAgent(agentId: string) {
    try {
      const options: AgentActivationOptions = {
        role: 'developer', // Default role, could be determined by agent type
        priority: 'normal'
      }
      
      await this.agentService.activateAgent(agentId, options)
      console.log('Agent activation requested:', agentId)
      
      // Update local state optimistically
      const agentIndex = this.agents.findIndex(a => a.id === agentId)
      if (agentIndex >= 0) {
        const updatedAgents = [...this.agents]
        updatedAgents[agentIndex] = {
          ...updatedAgents[agentIndex],
          status: 'active'
        }
        this.agents = updatedAgents
      }
      
    } catch (error) {
      console.error('Failed to activate agent:', error)
      this.error = error instanceof Error ? error.message : 'Failed to activate agent'
    }
  }
  
  /**
   * Deactivate an agent
   */
  private async handleDeactivateAgent(agentId: string) {
    try {
      await this.agentService.deactivateAgent(agentId)
      console.log('Agent deactivation requested:', agentId)
      
      // Update local state optimistically
      const agentIndex = this.agents.findIndex(a => a.id === agentId)
      if (agentIndex >= 0) {
        const updatedAgents = [...this.agents]
        updatedAgents[agentIndex] = {
          ...updatedAgents[agentIndex],
          status: 'idle'
        }
        this.agents = updatedAgents
      }
      
    } catch (error) {
      console.error('Failed to deactivate agent:', error)
      this.error = error instanceof Error ? error.message : 'Failed to deactivate agent'
    }
  }

  private get agentsSummary() {
    const active = this.agents.filter(a => a.status === 'active').length
    const idle = this.agents.filter(a => a.status === 'idle').length
    const error = this.agents.filter(a => a.status === 'error').length
    
    return { active, idle, error, total: this.agents.length }
  }

  private formatUptime(uptime: number): string {
    const hours = Math.floor(uptime / (1000 * 60 * 60))
    return `${hours}h`
  }

  private formatLastSeen(lastSeen: string): string {
    const date = new Date(lastSeen)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / (1000 * 60))
    
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    
    const diffHours = Math.floor(diffMins / 60)
    return `${diffHours}h ago`
  }

  /**
   * Render grid view of agents
   */
  private renderGridView() {
    return html`
      <div class="agents-grid">
        ${this.agents.map(agent => html`
          <div 
            class="agent-card ${this.selectedAgent?.id === agent.id ? 'selected' : ''} ${this.selectedAgents.has(agent.id) ? 'bulk-selected' : ''}"
            @click=${() => this.bulkActionMode ? this.handleAgentSelection(agent.id) : this.handleAgentSelect(agent)}
          >
            ${this.bulkActionMode ? html`
              <input
                type="checkbox"
                class="agent-checkbox"
                .checked=${this.selectedAgents.has(agent.id)}
                @click=${(e: Event) => this.handleAgentSelection(agent.id, e)}
              />
            ` : ''}
            
            <div class="agent-status-indicator ${agent.status}"></div>
            
            <div class="agent-header">
              <h3 class="agent-name">${agent.name}</h3>
              <span class="agent-type">${agent.status}</span>
            </div>

            <div class="agent-metrics">
              <div class="metric-item">
                <p class="metric-value">${this.formatUptime(agent.uptime)}</p>
                <p class="metric-label">Uptime</p>
              </div>
              <div class="metric-item">
                <p class="metric-value">${agent.performance.score}%</p>
                <p class="metric-label">Performance</p>
              </div>
              <div class="metric-item">
                <p class="metric-value">${agent.metrics.tasksCompleted[agent.metrics.tasksCompleted.length - 1]}</p>
                <p class="metric-label">Tasks Done</p>
              </div>
              <div class="metric-item">
                <p class="metric-value">${this.formatLastSeen(agent.lastSeen)}</p>
                <p class="metric-label">Last Seen</p>
              </div>
            </div>

            ${agent.currentTask ? html`
              <div class="agent-current-task">
                <p class="current-task-label">Current Task</p>
                <p class="current-task-text">${agent.currentTask}</p>
              </div>
            ` : ''}
            
            ${!this.bulkActionMode ? html`
              <div class="agent-controls">
                ${agent.status === 'active' ? html`
                  <button 
                    class="control-button deactivate"
                    @click=${(e: Event) => {
                      e.stopPropagation()
                      this.handleDeactivateAgent(agent.id)
                    }}
                    ?disabled=${this.isLoading}
                  >
                    <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    Deactivate
                  </button>
                ` : html`
                  <button 
                    class="control-button activate"
                    @click=${(e: Event) => {
                      e.stopPropagation()
                      this.handleActivateAgent(agent.id)
                    }}
                    ?disabled=${this.isLoading}
                  >
                    <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1.586a1 1 0 01.707.293l2.414 2.414a1 1 0 00.707.293H15"/>
                    </svg>
                    Activate
                  </button>
                `}
                
                <button 
                  class="control-button"
                  @click=${(e: Event) => {
                    e.stopPropagation()
                    this.handleShowAgentConfig('edit', agent as any)
                  }}
                  ?disabled=${this.isLoading}
                >
                  <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                  </svg>
                  Configure
                </button>
              </div>
            ` : ''}
          </div>
        `)}
      </div>
    `
  }

  /**
   * Render list view of agents
   */
  private renderListView() {
    return html`
      <div class="agent-list">
        ${this.agents.map(agent => html`
          <div 
            class="agent-list-item ${this.selectedAgents.has(agent.id) ? 'bulk-selected' : ''}"
            @click=${() => this.bulkActionMode ? this.handleAgentSelection(agent.id) : this.handleAgentSelect(agent)}
          >
            ${this.bulkActionMode ? html`
              <input
                type="checkbox"
                class="agent-checkbox"
                .checked=${this.selectedAgents.has(agent.id)}
                @click=${(e: Event) => this.handleAgentSelection(agent.id, e)}
              />
            ` : ''}

            <div class="agent-list-content">
              <div class="agent-list-info">
                <div class="agent-basic-info">
                  <h3 class="agent-list-name">${agent.name}</h3>
                  <p class="agent-list-role">${agent.status}</p>
                </div>

                <div class="agent-list-metrics">
                  <span>Uptime: ${this.formatUptime(agent.uptime)}</span>
                  <span>Performance: ${agent.performance.score}%</span>
                  <span>Tasks: ${agent.metrics.tasksCompleted[agent.metrics.tasksCompleted.length - 1]}</span>
                  <span>Last seen: ${this.formatLastSeen(agent.lastSeen)}</span>
                </div>

                <div class="agent-list-status">
                  <span class="status-badge ${agent.status}">${agent.status}</span>
                  ${agent.currentTask ? html`
                    <span style="font-size: 0.75rem; color: #6b7280;">
                      ${agent.currentTask.length > 30 ? agent.currentTask.substring(0, 30) + '...' : agent.currentTask}
                    </span>
                  ` : ''}
                </div>
              </div>

              ${!this.bulkActionMode ? html`
                <div class="agent-list-actions">
                  ${agent.status === 'active' ? html`
                    <button 
                      class="control-button deactivate"
                      @click=${(e: Event) => {
                        e.stopPropagation()
                        this.handleDeactivateAgent(agent.id)
                      }}
                      ?disabled=${this.isLoading}
                    >
                      <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
                      </svg>
                      Deactivate
                    </button>
                  ` : html`
                    <button 
                      class="control-button activate"
                      @click=${(e: Event) => {
                        e.stopPropagation()
                        this.handleActivateAgent(agent.id)
                      }}
                      ?disabled=${this.isLoading}
                    >
                      <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1.586a1 1 0 01.707.293l2.414 2.414a1 1 0 00.707.293H15"/>
                      </svg>
                      Activate
                    </button>
                  `}
                  
                  <button 
                    class="control-button"
                    @click=${(e: Event) => {
                      e.stopPropagation()
                      this.handleShowAgentConfig('edit', agent as any)
                    }}
                    ?disabled=${this.isLoading}
                  >
                    <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                    </svg>
                    Configure
                  </button>
                </div>
              ` : ''}
            </div>
          </div>
        `)}
      </div>
    `
  }

  render() {
    if (this.isLoading && this.agents.length === 0) {
      return html`
        <div class="loading-state">
          <loading-spinner size="large"></loading-spinner>
          <p>Loading agents...</p>
        </div>
      `
    }

    if (this.error) {
      return html`
        <div class="error-state">
          <p><strong>Error:</strong> ${this.error}</p>
          <button class="refresh-button" @click=${this.handleRefresh}>
            Try Again
          </button>
        </div>
      `
    }

    if (this.agents.length === 0) {
      return html`
        <div class="empty-state">
          <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
          </svg>
          <h2 class="empty-title">No Agents Available</h2>
          <p class="empty-description">
            There are currently no agents deployed in your hive. Agents will appear here once they are activated and connected.
          </p>
          <button class="refresh-button" @click=${this.handleRefresh}>
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
            </svg>
            Refresh
          </button>
        </div>
      `
    }

    const summary = this.agentsSummary

    return html`
      <div class="agents-container">
        <div class="agents-header">
          <div class="header-content">
            <h1 class="page-title">
              <svg class="title-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
              </svg>
              Agent Management
            </h1>
            
            <div class="header-actions">
              <div class="view-toggle">
                <button 
                  class="view-button ${this.viewMode === 'grid' ? 'active' : ''}"
                  @click=${() => this.handleViewModeChange('grid')}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/>
                  </svg>
                  Grid
                </button>
                <button 
                  class="view-button ${this.viewMode === 'list' ? 'active' : ''}"
                  @click=${() => this.handleViewModeChange('list')}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 10h16M4 14h16M4 18h16"/>
                  </svg>
                  List
                </button>
              </div>

              <div class="team-controls">
                <button 
                  class="action-button success"
                  @click=${this.handleActivateTeam}
                  ?disabled=${this.isLoading}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"/>
                  </svg>
                  Activate Team
                </button>

                <button 
                  class="action-button danger"
                  @click=${this.handleDeactivateTeam}
                  ?disabled=${this.isLoading || this.agents.length === 0}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728L5.636 5.636m12.728 12.728L18.364 5.636M5.636 18.364l12.728-12.728"/>
                  </svg>
                  Deactivate Team
                </button>

                <button 
                  class="action-button"
                  @click=${() => this.handleShowAgentConfig('create')}
                  ?disabled=${this.isLoading}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
                  </svg>
                  Add Agent
                </button>

                <button 
                  class="action-button secondary"
                  @click=${this.handleToggleBulkMode}
                  ?disabled=${this.isLoading || this.agents.length === 0}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                  ${this.bulkActionMode ? 'Exit Bulk Mode' : 'Bulk Actions'}
                </button>
              </div>
            </div>
            
            <div class="agents-summary">
              <div class="summary-item active">
                <p class="summary-value">${summary.active}</p>
                <p class="summary-label">Active</p>
              </div>
              <div class="summary-item idle">
                <p class="summary-value">${summary.idle}</p>
                <p class="summary-label">Idle</p>
              </div>
              <div class="summary-item error">
                <p class="summary-value">${summary.error}</p>
                <p class="summary-label">Error</p>
              </div>
              <div class="summary-item">
                <p class="summary-value">${summary.total}</p>
                <p class="summary-label">Total</p>
              </div>
            </div>
          </div>

          ${this.bulkActionMode && this.agents.length > 0 ? html`
            <div class="bulk-action-bar">
              <div class="bulk-info">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                <span>${this.selectedAgents.size} of ${this.agents.length} agents selected</span>
              </div>
              <div class="bulk-actions">
                <button 
                  class="action-button secondary"
                  @click=${this.handleSelectAll}
                  ?disabled=${this.isLoading}
                >
                  ${this.selectedAgents.size === this.agents.length ? 'Deselect All' : 'Select All'}
                </button>
                <button 
                  class="action-button success"
                  @click=${this.handleBulkActivate}
                  ?disabled=${this.isLoading || this.selectedAgents.size === 0}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1.586a1 1 0 01.707.293l2.414 2.414a1 1 0 00.707.293H15"/>
                  </svg>
                  Activate Selected
                </button>
                <button 
                  class="action-button danger"
                  @click=${this.handleBulkDeactivate}
                  ?disabled=${this.isLoading || this.selectedAgents.size === 0}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                  Deactivate Selected
                </button>
              </div>
            </div>
          ` : ''}
        </div>

        <div class="agents-content">
          ${this.viewMode === 'grid' ? this.renderGridView() : this.renderListView()}
        </div>

        <!-- Agent Configuration Modal -->
        <agent-config-modal
          .open=${this.showAgentConfigModal}
          .mode=${this.configModalMode}
          .agent=${this.configModalAgent}
          @close=${() => this.showAgentConfigModal = false}
          @save=${this.handleAgentConfigSave}
        ></agent-config-modal>
      </div>
    `
  }
}