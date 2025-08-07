import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

interface RecoveryAction {
  id: string
  name: string
  description: string
  type: 'emergency' | 'diagnostic' | 'maintenance'
  impact: 'low' | 'medium' | 'high' | 'critical'
  confirmationRequired: boolean
  endpoint: string
  method: string
  parameters?: any
}

interface SystemDiagnostic {
  name: string
  status: 'healthy' | 'warning' | 'critical' | 'unknown'
  message: string
  last_check: string
  metric_value?: number
  threshold?: number
}

@customElement('recovery-controls-panel')
export class RecoveryControlsPanel extends LitElement {
  @property({ type: Array }) declare agents: any[]
  @property({ type: Object }) declare systemHealth: any
  @property({ type: Boolean }) declare emergencyMode: boolean

  @state() private declare isExecuting: boolean
  @state() private declare executingAction: string | null
  @state() private declare lastResults: Map<string, any>
  @state() private declare confirmationDialog: RecoveryAction | null
  @state() private declare diagnostics: SystemDiagnostic[]
  @state() private declare selectedAgents: Set<string>
  @state() private declare showAdvancedActions: boolean

  private readonly recoveryActions: RecoveryAction[] = [
    {
      id: 'reset-coordination',
      name: '🚨 Reset Coordination System',
      description: 'Emergency reset of the entire multi-agent coordination system. Clears all workflows and monitoring state.',
      type: 'emergency',
      impact: 'critical',
      confirmationRequired: true,
      endpoint: '/api/v1/coordination-monitoring/recovery-actions/reset-coordination',
      method: 'POST'
    },
    {
      id: 'restart-all-agents',
      name: '🔄 Restart All Agents',
      description: 'Gracefully restart all active agents to recover from coordination failures.',
      type: 'emergency',
      impact: 'high',
      confirmationRequired: true,
      endpoint: '/api/v1/agents/restart-all',
      method: 'POST'
    },
    {
      id: 'clear-failed-tasks',
      name: '🧹 Clear Failed Tasks',
      description: 'Remove all failed task assignments and reset task queue.',
      type: 'maintenance',
      impact: 'medium',
      confirmationRequired: false,
      endpoint: '/api/v1/coordination-monitoring/recovery-actions/clear-failed-tasks',
      method: 'POST'
    },
    {
      id: 'force-redis-reconnect',
      name: '🔌 Force Redis Reconnection',
      description: 'Force reconnection to Redis message bus to resolve communication issues.',
      type: 'maintenance',
      impact: 'low',
      confirmationRequired: false,
      endpoint: '/api/v1/coordination-monitoring/recovery-actions/reconnect-redis',
      method: 'POST'
    },
    {
      id: 'generate-test-data',
      name: '🧪 Generate Test Data',
      description: 'Generate test coordination data for validation and troubleshooting.',
      type: 'diagnostic',
      impact: 'low',
      confirmationRequired: false,
      endpoint: '/api/v1/coordination-monitoring/test/generate-coordination-data',
      method: 'POST'
    },
    {
      id: 'run-system-diagnostics',
      name: '🔍 Run System Diagnostics',
      description: 'Comprehensive system health check and diagnostic report.',
      type: 'diagnostic',
      impact: 'low',
      confirmationRequired: false,
      endpoint: '/api/v1/coordination-monitoring/diagnostics/full-check',
      method: 'POST'
    },
    {
      id: 'emergency-stop',
      name: '⛔ Emergency Stop All Operations',
      description: 'Immediately halt all agent operations and coordination workflows.',
      type: 'emergency',
      impact: 'critical',
      confirmationRequired: true,
      endpoint: '/api/v1/coordination-monitoring/recovery-actions/emergency-stop',
      method: 'POST'
    }
  ]

  static styles = css`
    :host {
      display: block;
      background: white;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
      overflow: hidden;
    }

    .panel-header {
      background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
      color: white;
      padding: 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .panel-header.emergency {
      animation: pulse-emergency 2s infinite;
    }

    @keyframes pulse-emergency {
      0%, 100% { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); }
      50% { background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%); }
    }

    .panel-title {
      font-size: 1.125rem;
      font-weight: 600;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .emergency-badge {
      background: rgba(255, 255, 255, 0.2);
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.025em;
      animation: blink 1s infinite;
    }

    @keyframes blink {
      0%, 50% { opacity: 1; }
      51%, 100% { opacity: 0.5; }
    }

    .system-status {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(255, 255, 255, 0.1);
      padding: 0.5rem 1rem;
      border-radius: 8px;
    }

    .panel-content {
      padding: 1rem;
    }

    .section {
      margin-bottom: 2rem;
    }

    .section-title {
      font-size: 1rem;
      font-weight: 600;
      color: #111827;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid #e5e7eb;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .toggle-advanced {
      background: none;
      border: 1px solid #d1d5db;
      color: #6b7280;
      padding: 0.25rem 0.75rem;
      border-radius: 4px;
      font-size: 0.75rem;
      cursor: pointer;
      transition: all 0.2s;
    }

    .toggle-advanced:hover {
      border-color: #9ca3af;
    }

    .toggle-advanced.active {
      background: #3b82f6;
      color: white;
      border-color: #3b82f6;
    }

    .actions-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1rem;
    }

    .action-card {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 1rem;
      transition: all 0.2s;
      position: relative;
    }

    .action-card:hover {
      border-color: #3b82f6;
      transform: translateY(-1px);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .action-card.emergency {
      border-color: #ef4444;
      background: #fef2f2;
    }

    .action-card.executing {
      opacity: 0.6;
      pointer-events: none;
    }

    .action-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 0.75rem;
    }

    .action-name {
      font-weight: 600;
      color: #111827;
      margin: 0 0 0.25rem 0;
    }

    .action-impact {
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }

    .impact-low {
      background: #dcfce7;
      color: #166534;
    }

    .impact-medium {
      background: #fef3c7;
      color: #92400e;
    }

    .impact-high {
      background: #fed7aa;
      color: #ea580c;
    }

    .impact-critical {
      background: #fee2e2;
      color: #991b1b;
    }

    .action-description {
      font-size: 0.875rem;
      color: #6b7280;
      margin-bottom: 1rem;
      line-height: 1.4;
    }

    .action-button {
      width: 100%;
      padding: 0.75rem;
      border: 1px solid #d1d5db;
      background: white;
      color: #374151;
      border-radius: 6px;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }

    .action-button:hover {
      background: #f9fafb;
      border-color: #9ca3af;
    }

    .action-button.emergency {
      background: #ef4444;
      color: white;
      border-color: #dc2626;
    }

    .action-button.emergency:hover {
      background: #dc2626;
    }

    .action-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .last-result {
      margin-top: 0.75rem;
      padding: 0.75rem;
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-radius: 6px;
      font-size: 0.875rem;
    }

    .last-result.error {
      background: #fef2f2;
      border-color: #fecaca;
      color: #dc2626;
    }

    .last-result-time {
      font-size: 0.75rem;
      color: #6b7280;
      margin-top: 0.25rem;
    }

    .diagnostics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }

    .diagnostic-item {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 1rem;
      text-align: center;
    }

    .diagnostic-item.warning {
      border-color: #f59e0b;
      background: #fffbeb;
    }

    .diagnostic-item.critical {
      border-color: #ef4444;
      background: #fef2f2;
    }

    .diagnostic-name {
      font-weight: 600;
      color: #111827;
      margin-bottom: 0.5rem;
    }

    .diagnostic-status {
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      margin-bottom: 0.5rem;
    }

    .status-healthy {
      background: #dcfce7;
      color: #166534;
    }

    .status-warning {
      background: #fef3c7;
      color: #92400e;
    }

    .status-critical {
      background: #fee2e2;
      color: #991b1b;
    }

    .status-unknown {
      background: #f3f4f6;
      color: #374151;
    }

    .diagnostic-message {
      font-size: 0.875rem;
      color: #6b7280;
      margin-bottom: 0.5rem;
    }

    .diagnostic-metric {
      font-size: 0.75rem;
      color: #6b7280;
    }

    .agent-selection {
      margin-bottom: 1rem;
      padding: 1rem;
      background: #f9fafb;
      border-radius: 8px;
    }

    .agent-checkboxes {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.5rem;
      margin-top: 0.75rem;
    }

    .agent-checkbox {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .confirmation-dialog {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }

    .dialog-content {
      background: white;
      border-radius: 12px;
      padding: 2rem;
      max-width: 400px;
      width: 90%;
      text-align: center;
    }

    .dialog-title {
      font-size: 1.25rem;
      font-weight: 600;
      color: #111827;
      margin-bottom: 1rem;
    }

    .dialog-message {
      color: #6b7280;
      margin-bottom: 2rem;
      line-height: 1.5;
    }

    .dialog-actions {
      display: flex;
      gap: 1rem;
      justify-content: center;
    }

    .dialog-btn {
      padding: 0.75rem 1.5rem;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }

    .dialog-btn.cancel {
      background: white;
      color: #374151;
    }

    .dialog-btn.cancel:hover {
      background: #f9fafb;
    }

    .dialog-btn.confirm {
      background: #ef4444;
      color: white;
      border-color: #dc2626;
    }

    .dialog-btn.confirm:hover {
      background: #dc2626;
    }

    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid #e5e7eb;
      border-top: 2px solid currentColor;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @media (max-width: 768px) {
      .actions-grid {
        grid-template-columns: 1fr;
      }

      .diagnostics-grid {
        grid-template-columns: repeat(2, 1fr);
      }

      .dialog-content {
        margin: 1rem;
        padding: 1.5rem;
      }

      .dialog-actions {
        flex-direction: column;
      }
    }
  `

  constructor() {
    super()
    this.agents = []
    this.systemHealth = null
    this.emergencyMode = false
    this.isExecuting = false
    this.executingAction = null
    this.lastResults = new Map()
    this.confirmationDialog = null
    this.diagnostics = []
    this.selectedAgents = new Set()
    this.showAdvancedActions = false
  }

  connectedCallback() {
    super.connectedCallback()
    this.loadDiagnostics()
  }

  private async loadDiagnostics() {
    try {
      // Generate mock diagnostic data based on system health
      this.diagnostics = [
        {
          name: 'Coordination Success Rate',
          status: this.systemHealth?.success_rate > 90 ? 'healthy' : 
                  this.systemHealth?.success_rate > 50 ? 'warning' : 'critical',
          message: `Current success rate: ${this.systemHealth?.success_rate || 0}%`,
          last_check: new Date().toISOString(),
          metric_value: this.systemHealth?.success_rate || 0,
          threshold: 95
        },
        {
          name: 'Agent Connectivity',
          status: this.agents?.length > 0 ? 'healthy' : 'warning',
          message: `${this.agents?.length || 0} agents connected`,
          last_check: new Date().toISOString(),
          metric_value: this.agents?.length || 0
        },
        {
          name: 'Redis Communication',
          status: 'healthy', // Would check actual Redis status
          message: 'Message bus operational',
          last_check: new Date().toISOString()
        },
        {
          name: 'Task Queue Health',
          status: 'healthy',
          message: 'Task distribution functioning normally',
          last_check: new Date().toISOString()
        }
      ]

    } catch (error) {
      console.warn('Failed to load diagnostics:', error)
    }
  }

  private async executeAction(action: RecoveryAction) {
    if (action.confirmationRequired) {
      this.confirmationDialog = action
      return
    }

    await this.performAction(action)
  }

  private async performAction(action: RecoveryAction) {
    this.isExecuting = true
    this.executingAction = action.id

    try {
      const body = action.parameters ? JSON.stringify(action.parameters) : undefined
      const headers = body ? { 'Content-Type': 'application/json' } : undefined

      const response = await fetch(action.endpoint, {
        method: action.method,
        headers,
        body
      })

      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.detail || `Action failed: ${response.statusText}`)
      }

      this.lastResults.set(action.id, {
        success: true,
        message: result.message || 'Action completed successfully',
        timestamp: new Date(),
        data: result
      })

      // Dispatch success event
      this.dispatchEvent(new CustomEvent('recovery-action-completed', {
        detail: { action: action.id, result, success: true },
        bubbles: true,
        composed: true
      }))

      // Reload diagnostics after action
      setTimeout(() => this.loadDiagnostics(), 2000)

    } catch (error) {
      console.error('Recovery action failed:', error)
      
      this.lastResults.set(action.id, {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        timestamp: new Date(),
        data: null
      })

      // Dispatch error event
      this.dispatchEvent(new CustomEvent('recovery-action-failed', {
        detail: { action: action.id, error: error?.toString(), success: false },
        bubbles: true,
        composed: true
      }))

    } finally {
      this.isExecuting = false
      this.executingAction = null
      this.confirmationDialog = null
      this.requestUpdate()
    }
  }

  private handleAgentSelection(agentId: string, selected: boolean) {
    if (selected) {
      this.selectedAgents.add(agentId)
    } else {
      this.selectedAgents.delete(agentId)
    }
    this.requestUpdate()
  }

  private getVisibleActions() {
    return this.recoveryActions.filter(action => 
      this.showAdvancedActions || action.type !== 'diagnostic'
    )
  }

  private renderAction(action: RecoveryAction) {
    const lastResult = this.lastResults.get(action.id)
    const isExecuting = this.executingAction === action.id

    return html`
      <div class="action-card ${action.type} ${isExecuting ? 'executing' : ''}">
        <div class="action-header">
          <div>
            <h3 class="action-name">${action.name}</h3>
          </div>
          <div class="action-impact impact-${action.impact}">
            ${action.impact}
          </div>
        </div>

        <p class="action-description">${action.description}</p>

        <button 
          class="action-button ${action.type}"
          @click=${() => this.executeAction(action)}
          ?disabled=${this.isExecuting}
        >
          ${isExecuting ? html`
            <div class="spinner"></div>
            Executing...
          ` : html`
            ${action.confirmationRequired ? '⚠️' : '▶️'} Execute Action
          `}
        </button>

        ${lastResult ? html`
          <div class="last-result ${lastResult.success ? '' : 'error'}">
            <div>${lastResult.message}</div>
            <div class="last-result-time">
              ${lastResult.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ` : ''}
      </div>
    `
  }

  private renderDiagnostic(diagnostic: SystemDiagnostic) {
    return html`
      <div class="diagnostic-item ${diagnostic.status}">
        <div class="diagnostic-name">${diagnostic.name}</div>
        <div class="diagnostic-status status-${diagnostic.status}">
          ${diagnostic.status}
        </div>
        <div class="diagnostic-message">${diagnostic.message}</div>
        ${diagnostic.metric_value !== undefined ? html`
          <div class="diagnostic-metric">
            Value: ${diagnostic.metric_value}${diagnostic.threshold ? ` / ${diagnostic.threshold}` : ''}
          </div>
        ` : ''}
      </div>
    `
  }

  render() {
    const visibleActions = this.getVisibleActions()

    return html`
      <div class="panel-header ${this.emergencyMode ? 'emergency' : ''}">
        <div class="panel-title">
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          Recovery Action Controls
          ${this.emergencyMode ? html`
            <span class="emergency-badge">EMERGENCY MODE</span>
          ` : ''}
        </div>

        <div class="system-status">
          <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          System Status
        </div>
      </div>

      <div class="panel-content">
        <!-- System Diagnostics -->
        <div class="section">
          <div class="section-title">
            🔍 System Diagnostics
            <button 
              class="toggle-advanced ${this.showAdvancedActions ? 'active' : ''}"
              @click=${() => {
                this.showAdvancedActions = !this.showAdvancedActions
                this.requestUpdate()
              }}
            >
              ${this.showAdvancedActions ? 'Hide Advanced' : 'Show Advanced'}
            </button>
          </div>

          <div class="diagnostics-grid">
            ${this.diagnostics.map(diagnostic => this.renderDiagnostic(diagnostic))}
          </div>
        </div>

        <!-- Agent Selection for bulk operations -->
        ${this.agents.length > 0 ? html`
          <div class="section">
            <div class="section-title">🎯 Target Agent Selection</div>
            <div class="agent-selection">
              <div>Select agents for bulk recovery actions:</div>
              <div class="agent-checkboxes">
                ${this.agents.map(agent => html`
                  <label class="agent-checkbox">
                    <input 
                      type="checkbox" 
                      ?checked=${this.selectedAgents.has(agent.id)}
                      @change=${(e: Event) => {
                        const checked = (e.target as HTMLInputElement).checked
                        this.handleAgentSelection(agent.id, checked)
                      }}
                    >
                    ${agent.name || agent.id}
                  </label>
                `)}
              </div>
            </div>
          </div>
        ` : ''}

        <!-- Recovery Actions -->
        <div class="section">
          <div class="section-title">🚨 Recovery Actions</div>
          <div class="actions-grid">
            ${visibleActions.map(action => this.renderAction(action))}
          </div>
        </div>
      </div>

      <!-- Confirmation Dialog -->
      ${this.confirmationDialog ? html`
        <div class="confirmation-dialog" @click=${(e: Event) => {
          if (e.target === e.currentTarget) {
            this.confirmationDialog = null
          }
        }}>
          <div class="dialog-content">
            <div class="dialog-title">
              ⚠️ Confirm Recovery Action
            </div>
            <div class="dialog-message">
              <strong>${this.confirmationDialog.name}</strong><br><br>
              ${this.confirmationDialog.description}<br><br>
              Impact Level: <strong>${this.confirmationDialog.impact.toUpperCase()}</strong><br><br>
              Are you sure you want to proceed?
            </div>
            <div class="dialog-actions">
              <button 
                class="dialog-btn cancel"
                @click=${() => this.confirmationDialog = null}
              >
                Cancel
              </button>
              <button 
                class="dialog-btn confirm"
                @click=${() => {
                  const action = this.confirmationDialog!
                  this.confirmationDialog = null
                  this.performAction(action)
                }}
              >
                Execute Action
              </button>
            </div>
          </div>
        </div>
      ` : ''}
    `
  }
}