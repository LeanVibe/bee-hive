import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../../services/websocket'

export interface EmergencyAction {
  id: string
  type: 'stop' | 'pause' | 'restart' | 'intervention'
  title: string
  description: string
  confirmRequired: boolean
  severity: 'low' | 'medium' | 'high' | 'critical'
  estimatedTime?: string
  icon: string
}

@customElement('emergency-controls')
export class EmergencyControls extends LitElement {
  @property({ type: Boolean }) declare expanded: boolean
  @property({ type: Boolean }) declare disabled: boolean
  
  @state() private declare showConfirmDialog: boolean
  @state() private declare selectedAction: EmergencyAction | null
  @state() private declare isExecuting: boolean
  @state() private declare lastActionTime: Date | null

  private websocketService: WebSocketService
  private emergencyActions: EmergencyAction[]

  static styles = css`
    :host {
      display: block;
      position: fixed;
      bottom: 80px; /* Above bottom navigation */
      right: 1rem;
      z-index: 100;
    }

    .emergency-fab {
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
      color: white;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.5rem;
      box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4);
      transition: all 0.3s ease;
      position: relative;
    }

    .emergency-fab:hover {
      transform: scale(1.05);
      box-shadow: 0 6px 20px rgba(220, 38, 38, 0.6);
    }

    .emergency-fab:active {
      transform: scale(0.95);
    }

    .emergency-fab.disabled {
      background: #9ca3af;
      cursor: not-allowed;
      transform: none;
    }

    .emergency-fab.executing {
      animation: pulse 1s infinite;
    }

    .pulse-ring {
      position: absolute;
      top: -4px;
      left: -4px;
      right: -4px;
      bottom: -4px;
      border: 2px solid #dc2626;
      border-radius: 50%;
      opacity: 0;
      animation: emergencyPulse 2s infinite;
    }

    .controls-panel {
      position: absolute;
      bottom: 70px;
      right: 0;
      background: white;
      border-radius: 12px;
      padding: 1rem;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
      min-width: 280px;
      transform: translateY(20px);
      opacity: 0;
      pointer-events: none;
      transition: all 0.3s ease;
    }

    .controls-panel.expanded {
      transform: translateY(0);
      opacity: 1;
      pointer-events: all;
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid #e5e7eb;
    }

    .panel-title {
      font-weight: 600;
      color: #111827;
      font-size: 1rem;
    }

    .close-button {
      background: none;
      border: none;
      color: #6b7280;
      cursor: pointer;
      padding: 0.25rem;
      border-radius: 4px;
      transition: background 0.2s;
    }

    .close-button:hover {
      background: #f3f4f6;
    }

    .action-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .action-button {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.75rem;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: white;
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: left;
    }

    .action-button:hover {
      border-color: #d1d5db;
      background: #f9fafb;
    }

    .action-button.critical {
      border-color: #fecaca;
      background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
    }

    .action-button.critical:hover {
      border-color: #dc2626;
      background: #fef2f2;
    }

    .action-button.high {
      border-color: #fed7aa;
      background: linear-gradient(135deg, #ffffff 0%, #fffbeb 100%);
    }

    .action-icon {
      font-size: 1.25rem;
      width: 1.5rem;
      text-align: center;
    }

    .action-content {
      flex: 1;
    }

    .action-title {
      font-weight: 600;
      color: #111827;
      font-size: 0.875rem;
      margin-bottom: 0.25rem;
    }

    .action-description {
      font-size: 0.75rem;
      color: #6b7280;
      line-height: 1.4;
    }

    .action-time {
      font-size: 0.75rem;
      color: #9ca3af;
      margin-top: 0.25rem;
    }

    /* Confirm Dialog */
    .confirm-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.5);
      backdrop-filter: blur(4px);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 200;
      padding: 1rem;
    }

    .confirm-dialog {
      background: white;
      border-radius: 12px;
      padding: 1.5rem;
      max-width: 400px;
      width: 100%;
      animation: dialogSlideIn 0.3s ease;
    }

    .confirm-header {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }

    .confirm-icon {
      font-size: 2rem;
    }

    .confirm-title {
      font-weight: 600;
      color: #111827;
      font-size: 1.125rem;
    }

    .confirm-message {
      color: #4b5563;
      margin-bottom: 1.5rem;
      line-height: 1.5;
    }

    .confirm-actions {
      display: flex;
      gap: 0.75rem;
      justify-content: flex-end;
    }

    .confirm-button {
      padding: 0.5rem 1rem;
      border-radius: 6px;
      border: 1px solid;
      cursor: pointer;
      font-weight: 500;
      font-size: 0.875rem;
      transition: all 0.2s;
    }

    .confirm-button.cancel {
      background: white;
      border-color: #d1d5db;
      color: #374151;
    }

    .confirm-button.cancel:hover {
      background: #f9fafb;
    }

    .confirm-button.execute {
      background: #dc2626;
      border-color: #dc2626;
      color: white;
    }

    .confirm-button.execute:hover {
      background: #b91c1c;
    }

    .confirm-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    /* Status Indicator */
    .status-indicator {
      position: absolute;
      top: -2px;
      right: -2px;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #10b981;
      border: 2px solid white;
      animation: statusPulse 2s infinite;
    }

    .status-indicator.executing {
      background: #f59e0b;
      animation: spin 1s linear infinite;
    }

    .status-indicator.error {
      background: #ef4444;
      animation: errorBlink 1s infinite;
    }

    /* Animations */
    @keyframes emergencyPulse {
      0% {
        transform: scale(0.95);
        opacity: 1;
      }
      70% {
        transform: scale(1.2);
        opacity: 0;
      }
      100% {
        transform: scale(0.95);
        opacity: 0;
      }
    }

    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.05); }
    }

    @keyframes statusPulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.6; }
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @keyframes errorBlink {
      0%, 50% { opacity: 1; }
      25%, 75% { opacity: 0.3; }
    }

    @keyframes dialogSlideIn {
      from {
        transform: translateY(20px);
        opacity: 0;
      }
      to {
        transform: translateY(0);
        opacity: 1;
      }
    }

    /* Accessibility */
    @media (prefers-reduced-motion: reduce) {
      * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
      }
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
      .controls-panel,
      .confirm-dialog {
        background: #1f2937;
        color: #f9fafb;
        border: 1px solid #374151;
      }

      .panel-title,
      .confirm-title {
        color: #f9fafb;
      }

      .action-button {
        background: #374151;
        border-color: #4b5563;
        color: #f9fafb;
      }

      .action-button:hover {
        background: #4b5563;
      }
    }
  `

  constructor() {
    super()
    this.expanded = false
    this.disabled = false
    this.showConfirmDialog = false
    this.selectedAction = null
    this.isExecuting = false
    this.lastActionTime = null

    this.websocketService = WebSocketService.getInstance()
    
    this.emergencyActions = [
      {
        id: 'emergency_stop',
        type: 'stop',
        title: 'Emergency Stop',
        description: 'Immediately stop all agent activities',
        confirmRequired: true,
        severity: 'critical',
        estimatedTime: 'Immediate',
        icon: '🛑'
      },
      {
        id: 'pause_agents',
        type: 'pause',
        title: 'Pause All Agents',
        description: 'Temporarily pause agent operations',
        confirmRequired: true,
        severity: 'high',
        estimatedTime: '< 5 sec',
        icon: '⏸️'
      },
      {
        id: 'restart_failed',
        type: 'restart',
        title: 'Restart Failed Agents',
        description: 'Restart agents with errors or failures',
        confirmRequired: false,
        severity: 'medium',
        estimatedTime: '1-2 min',
        icon: '🔄'
      },
      {
        id: 'human_takeover',
        type: 'intervention',
        title: 'Request Intervention',
        description: 'Signal need for human oversight',
        confirmRequired: false,
        severity: 'high',
        estimatedTime: 'Manual review',
        icon: '✋'
      }
    ]
  }

  connectedCallback() {
    super.connectedCallback()
    this.setupEventListeners()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    document.removeEventListener('click', this.handleOutsideClick.bind(this))
  }

  private setupEventListeners() {
    // Close panel when clicking outside
    document.addEventListener('click', this.handleOutsideClick.bind(this))

    // Listen for WebSocket connection status
    this.websocketService.on('connected', () => {
      this.disabled = false
    })

    this.websocketService.on('disconnected', () => {
      this.disabled = true
    })

    // Listen for emergency responses
    this.websocketService.on('emergency-response', (data) => {
      this.handleEmergencyResponse(data)
    })
  }

  private handleOutsideClick(event: Event) {
    const target = event.target as HTMLElement
    if (!this.contains(target) && this.expanded && !this.showConfirmDialog) {
      this.togglePanel()
    }
  }

  private togglePanel() {
    this.expanded = !this.expanded
    
    // Emit event for haptic feedback
    this.dispatchEvent(new CustomEvent('panel-toggle', {
      detail: { expanded: this.expanded },
      bubbles: true
    }))
  }

  private handleActionClick(action: EmergencyAction) {
    if (this.disabled || this.isExecuting) return

    this.selectedAction = action

    if (action.confirmRequired) {
      this.showConfirmDialog = true
    } else {
      this.executeAction(action)
    }
  }

  private async executeAction(action: EmergencyAction) {
    if (!action) return

    this.isExecuting = true
    this.expanded = false
    this.showConfirmDialog = false

    try {
      // Send emergency command via WebSocket
      switch (action.type) {
        case 'stop':
          this.websocketService.sendEmergencyStop(`Mobile emergency stop: ${action.title}`)
          break
        case 'pause':
          this.websocketService.sendEmergencyPause()
          break
        case 'restart':
          this.websocketService.sendMessage({
            type: 'restart-failed-agents',
            data: { reason: 'Mobile dashboard restart request' }
          })
          break
        case 'intervention':
          this.websocketService.sendMessage({
            type: 'request-human-intervention',
            data: { 
              reason: 'Mobile dashboard intervention request',
              priority: action.severity
            }
          })
          break
      }

      this.lastActionTime = new Date()

      // Emit event for parent components
      this.dispatchEvent(new CustomEvent('emergency-action', {
        detail: {
          action: action.id,
          type: action.type,
          timestamp: this.lastActionTime
        },
        bubbles: true,
        composed: true
      }))

    } catch (error) {
      console.error('Emergency action failed:', error)
      
      this.dispatchEvent(new CustomEvent('emergency-error', {
        detail: { action: action.id, error: error.message },
        bubbles: true,
        composed: true
      }))
    } finally {
      setTimeout(() => {
        this.isExecuting = false
      }, 2000) // Show executing state for 2 seconds
    }
  }

  private handleEmergencyResponse(data: any) {
    console.log('Emergency response received:', data)
    
    this.dispatchEvent(new CustomEvent('emergency-response', {
      detail: data,
      bubbles: true,
      composed: true
    }))
  }

  private confirmAction() {
    if (this.selectedAction) {
      this.executeAction(this.selectedAction)
    }
  }

  private cancelAction() {
    this.showConfirmDialog = false
    this.selectedAction = null
  }

  private renderConfirmDialog() {
    if (!this.showConfirmDialog || !this.selectedAction) return ''

    return html`
      <div class="confirm-overlay" @click="${(e: Event) => e.target === e.currentTarget && this.cancelAction()}">
        <div class="confirm-dialog">
          <div class="confirm-header">
            <span class="confirm-icon">${this.selectedAction.icon}</span>
            <h3 class="confirm-title">Confirm ${this.selectedAction.title}</h3>
          </div>
          
          <div class="confirm-message">
            <p><strong>Action:</strong> ${this.selectedAction.description}</p>
            <p><strong>Estimated Time:</strong> ${this.selectedAction.estimatedTime}</p>
            ${this.selectedAction.severity === 'critical' ? html`
              <p style="color: #dc2626; font-weight: 600; margin-top: 0.75rem;">
                ⚠️ This action will immediately affect all running agents
              </p>
            ` : ''}
          </div>
          
          <div class="confirm-actions">
            <button 
              class="confirm-button cancel"
              @click="${this.cancelAction}"
              ?disabled="${this.isExecuting}"
            >
              Cancel
            </button>
            <button 
              class="confirm-button execute"
              @click="${this.confirmAction}"
              ?disabled="${this.isExecuting}"
            >
              ${this.isExecuting ? 'Executing...' : `Confirm ${this.selectedAction.title}`}
            </button>
          </div>
        </div>
      </div>
    `
  }

  render() {
    return html`
      <button 
        class="emergency-fab ${this.disabled ? 'disabled' : ''} ${this.isExecuting ? 'executing' : ''}"
        @click="${this.togglePanel}"
        ?disabled="${this.disabled}"
        aria-label="Emergency controls"
        title="Emergency controls"
      >
        ${this.isExecuting ? '⏳' : '🚨'}
        <div class="pulse-ring"></div>
        <div class="status-indicator ${this.isExecuting ? 'executing' : ''}"></div>
      </button>

      <div class="controls-panel ${this.expanded ? 'expanded' : ''}">
        <div class="panel-header">
          <h3 class="panel-title">Emergency Controls</h3>
          <button class="close-button" @click="${this.togglePanel}">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div class="action-list">
          ${this.emergencyActions.map(action => html`
            <button 
              class="action-button ${action.severity}"
              @click="${() => this.handleActionClick(action)}"
              ?disabled="${this.disabled || this.isExecuting}"
            >
              <span class="action-icon">${action.icon}</span>
              <div class="action-content">
                <div class="action-title">${action.title}</div>
                <div class="action-description">${action.description}</div>
                ${action.estimatedTime ? html`
                  <div class="action-time">Est. ${action.estimatedTime}</div>
                ` : ''}
              </div>
            </button>
          `)}
        </div>
      </div>

      ${this.renderConfirmDialog()}
    `
  }
}