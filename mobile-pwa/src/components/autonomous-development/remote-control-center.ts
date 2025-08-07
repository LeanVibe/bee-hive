import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../../services/websocket'
import { Agent, AgentRole, AgentStatus } from '../../types/api'

interface RemoteCommand {
  id: string
  name: string
  description: string
  category: 'control' | 'coordination' | 'emergency' | 'development'
  icon: string
  requiresConfirmation: boolean
  parameters?: CommandParameter[]
}

interface CommandParameter {
  name: string
  type: 'text' | 'number' | 'select' | 'boolean'
  required: boolean
  options?: string[]
  placeholder?: string
  min?: number
  max?: number
}

interface CommandExecution {
  id: string
  command: string
  timestamp: Date
  status: 'pending' | 'executing' | 'completed' | 'failed'
  result?: any
  error?: string
  agentIds: string[]
}

interface SystemCommand {
  command: string
  description: string
  icon: string
  shortcut?: string
}

@customElement('remote-control-center')
export class RemoteControlCenter extends LitElement {
  @property({ type: Boolean }) declare expanded: boolean
  @property({ type: Array }) declare selectedAgents: string[]
  @property({ type: Boolean }) declare emergencyMode: boolean

  @state() private declare availableCommands: RemoteCommand[]
  @state() private declare commandHistory: CommandExecution[]
  @state() private declare activeExecutions: Map<string, CommandExecution>
  @state() private declare selectedCommand: RemoteCommand | null
  @state() private declare commandParameters: Record<string, any>
  @state() private declare quickCommands: SystemCommand[]
  @state() private declare bulkOperationMode: boolean
  @state() private declare connectionStatus: 'connected' | 'reconnecting' | 'disconnected'
  @state() private declare voiceControlEnabled: boolean
  @state() private declare shortcuts: Record<string, string>

  private websocketService: WebSocketService
  private commandExecutionTimeout: number = 30000 // 30 seconds
  private recognition?: SpeechRecognition

  static styles = css`
    :host {
      display: block;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      height: 100%;
      overflow: hidden;
    }

    .control-center-container {
      height: 100%;
      display: flex;
      flex-direction: column;
      position: relative;
    }

    .header {
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(148, 163, 184, 0.2);
      padding: 1rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .header-title {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .emergency-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #10b981;
      position: relative;
    }

    .emergency-indicator.emergency {
      background: #ef4444;
      animation: pulse 1s infinite;
    }

    .connection-status {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.875rem;
      color: #94a3b8;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
    }

    .status-dot.reconnecting {
      background: #f59e0b;
      animation: pulse 1s infinite;
    }

    .status-dot.disconnected {
      background: #ef4444;
    }

    .control-panel {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      padding: 1.5rem;
      overflow: hidden;
    }

    .command-section {
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 16px;
      padding: 1.5rem;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .section-title {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      color: #f8fafc;
    }

    .quick-commands {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .quick-command-btn {
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      border: none;
      color: white;
      padding: 1.25rem;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.3s;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.875rem;
      font-weight: 600;
      position: relative;
      overflow: hidden;
    }

    .quick-command-btn::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
      transition: left 0.5s;
    }

    .quick-command-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
    }

    .quick-command-btn:hover::before {
      left: 100%;
    }

    .quick-command-btn.emergency {
      background: linear-gradient(135deg, #ef4444, #dc2626);
    }

    .quick-command-btn.emergency:hover {
      box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
    }

    .command-icon {
      font-size: 1.5rem;
    }

    .keyboard-shortcut {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      background: rgba(0, 0, 0, 0.3);
      color: rgba(255, 255, 255, 0.7);
      font-size: 0.75rem;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      font-family: monospace;
    }

    .advanced-commands {
      flex: 1;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
    }

    .command-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .command-item {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(148, 163, 184, 0.1);
      border-radius: 8px;
      padding: 1rem;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .command-item:hover {
      background: rgba(59, 130, 246, 0.1);
      border-color: rgba(59, 130, 246, 0.3);
    }

    .command-item.selected {
      background: rgba(59, 130, 246, 0.2);
      border-color: #3b82f6;
    }

    .command-meta {
      flex: 1;
    }

    .command-name {
      font-weight: 600;
      color: #f8fafc;
      margin-bottom: 0.25rem;
    }

    .command-description {
      font-size: 0.875rem;
      color: #94a3b8;
      line-height: 1.4;
    }

    .command-category {
      padding: 0.25rem 0.75rem;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }

    .command-category.control {
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
    }

    .command-category.coordination {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
    }

    .command-category.emergency {
      background: rgba(239, 68, 68, 0.2);
      color: #f87171;
    }

    .command-category.development {
      background: rgba(168, 85, 247, 0.2);
      color: #c084fc;
    }

    .execution-panel {
      max-height: 400px;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
    }

    .execution-history {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .execution-item {
      background: rgba(15, 23, 42, 0.4);
      border-radius: 8px;
      padding: 1rem;
      border-left: 4px solid #3b82f6;
    }

    .execution-item.completed {
      border-left-color: #10b981;
    }

    .execution-item.failed {
      border-left-color: #ef4444;
    }

    .execution-item.executing {
      border-left-color: #f59e0b;
      position: relative;
      overflow: hidden;
    }

    .execution-item.executing::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 4px;
      background: linear-gradient(90deg, #f59e0b, #d97706);
      animation: loading 2s infinite;
    }

    .execution-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }

    .execution-command {
      font-weight: 600;
      color: #f8fafc;
    }

    .execution-timestamp {
      font-size: 0.75rem;
      color: #64748b;
    }

    .execution-status {
      padding: 0.25rem 0.75rem;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }

    .execution-status.pending {
      background: rgba(100, 116, 139, 0.2);
      color: #94a3b8;
    }

    .execution-status.executing {
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
    }

    .execution-status.completed {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
    }

    .execution-status.failed {
      background: rgba(239, 68, 68, 0.2);
      color: #f87171;
    }

    .execution-result {
      font-size: 0.875rem;
      color: #94a3b8;
      margin-top: 0.5rem;
      font-family: monospace;
      background: rgba(0, 0, 0, 0.3);
      padding: 0.75rem;
      border-radius: 6px;
      white-space: pre-wrap;
    }

    .command-form {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(15, 23, 42, 0.98);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(148, 163, 184, 0.3);
      border-radius: 16px;
      padding: 2rem;
      width: 90%;
      max-width: 500px;
      max-height: 80vh;
      overflow-y: auto;
      z-index: 1000;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    }

    .form-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.5rem;
    }

    .form-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: #f8fafc;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .close-btn {
      background: none;
      border: none;
      color: #94a3b8;
      cursor: pointer;
      padding: 0.5rem;
      border-radius: 6px;
      transition: all 0.2s;
      font-size: 1.25rem;
    }

    .close-btn:hover {
      background: rgba(148, 163, 184, 0.1);
      color: #f8fafc;
    }

    .form-group {
      margin-bottom: 1.5rem;
    }

    .form-label {
      display: block;
      font-weight: 600;
      color: #f8fafc;
      margin-bottom: 0.5rem;
      font-size: 0.875rem;
    }

    .form-input, .form-select, .form-textarea {
      width: 100%;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.3);
      border-radius: 8px;
      padding: 0.75rem;
      color: #f8fafc;
      font-size: 0.875rem;
      transition: all 0.2s;
    }

    .form-input:focus, .form-select:focus, .form-textarea:focus {
      outline: none;
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .form-checkbox {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      cursor: pointer;
    }

    .checkbox-input {
      width: 18px;
      height: 18px;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.3);
      border-radius: 4px;
      position: relative;
      cursor: pointer;
    }

    .checkbox-input:checked {
      background: #3b82f6;
      border-color: #3b82f6;
    }

    .checkbox-input:checked::after {
      content: '✓';
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: white;
      font-size: 12px;
      font-weight: bold;
    }

    .form-actions {
      display: flex;
      gap: 1rem;
      justify-content: flex-end;
      margin-top: 2rem;
    }

    .btn {
      padding: 0.75rem 1.5rem;
      border: none;
      border-radius: 8px;
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .btn.primary {
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      color: white;
    }

    .btn.primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    }

    .btn.secondary {
      background: rgba(148, 163, 184, 0.1);
      color: #94a3b8;
      border: 1px solid rgba(148, 163, 184, 0.3);
    }

    .btn.secondary:hover {
      background: rgba(148, 163, 184, 0.2);
      color: #f8fafc;
    }

    .voice-control {
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      width: 64px;
      height: 64px;
      border-radius: 50%;
      background: linear-gradient(135deg, #16a34a, #15803d);
      border: none;
      color: white;
      font-size: 1.5rem;
      cursor: pointer;
      transition: all 0.3s;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 8px 32px rgba(22, 163, 74, 0.3);
      z-index: 100;
    }

    .voice-control:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 48px rgba(22, 163, 74, 0.4);
    }

    .voice-control.active {
      background: linear-gradient(135deg, #ef4444, #dc2626);
      animation: pulse 1s infinite;
    }

    .bulk-selection-bar {
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      color: white;
      padding: 1rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-weight: 600;
    }

    .bulk-actions {
      display: flex;
      gap: 1rem;
    }

    .bulk-action-btn {
      background: rgba(255, 255, 255, 0.2);
      border: 1px solid rgba(255, 255, 255, 0.3);
      color: white;
      padding: 0.5rem 1rem;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
      font-size: 0.875rem;
    }

    .bulk-action-btn:hover {
      background: rgba(255, 255, 255, 0.3);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .control-panel {
        grid-template-columns: 1fr;
        padding: 1rem;
      }

      .quick-commands {
        grid-template-columns: repeat(2, 1fr);
      }

      .command-form {
        width: 95%;
        padding: 1.5rem;
      }
    }

    /* Animations */
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; }
    }

    @keyframes loading {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }

    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .command-item, .execution-item {
      animation: slideIn 0.3s ease-out;
    }
  `

  constructor() {
    super()
    this.expanded = false
    this.selectedAgents = []
    this.emergencyMode = false
    
    this.availableCommands = []
    this.commandHistory = []
    this.activeExecutions = new Map()
    this.selectedCommand = null
    this.commandParameters = {}
    this.quickCommands = []
    this.bulkOperationMode = false
    this.connectionStatus = 'disconnected'
    this.voiceControlEnabled = false
    this.shortcuts = {}
    
    this.websocketService = WebSocketService.getInstance()
    
    this.initializeCommands()
    this.initializeShortcuts()
  }

  connectedCallback() {
    super.connectedCallback()
    this.setupWebSocketListeners()
    this.setupKeyboardShortcuts()
    this.initializeVoiceControl()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.cleanup()
  }

  private initializeCommands() {
    // Initialize available remote commands
    this.availableCommands = [
      {
        id: 'spawn-agent',
        name: 'Spawn New Agent',
        description: 'Create and activate a new AI agent with specified role',
        category: 'control',
        icon: '🤖',
        requiresConfirmation: false,
        parameters: [
          {
            name: 'role',
            type: 'select',
            required: true,
            options: ['product_manager', 'architect', 'backend_developer', 'frontend_developer', 'qa_engineer']
          },
          {
            name: 'name',
            type: 'text',
            required: false,
            placeholder: 'Custom agent name (optional)'
          }
        ]
      },
      {
        id: 'coordinate-agents',
        name: 'Coordinate Team',
        description: 'Initiate team coordination sequence for collaborative task execution',
        category: 'coordination',
        icon: '🤝',
        requiresConfirmation: false,
        parameters: [
          {
            name: 'task',
            type: 'text',
            required: true,
            placeholder: 'Task description for team coordination'
          },
          {
            name: 'priority',
            type: 'select',
            required: true,
            options: ['low', 'medium', 'high', 'critical']
          }
        ]
      },
      {
        id: 'emergency-shutdown',
        name: 'Emergency Shutdown',
        description: 'Immediately halt all agent operations and enter safe mode',
        category: 'emergency',
        icon: '🛑',
        requiresConfirmation: true
      },
      {
        id: 'deploy-code',
        name: 'Deploy Code',
        description: 'Trigger automated deployment pipeline with specified parameters',
        category: 'development',
        icon: '🚀',
        requiresConfirmation: true,
        parameters: [
          {
            name: 'environment',
            type: 'select',
            required: true,
            options: ['development', 'staging', 'production']
          },
          {
            name: 'runTests',
            type: 'boolean',
            required: false
          }
        ]
      },
      {
        id: 'system-health-check',
        name: 'System Health Check',
        description: 'Perform comprehensive system health assessment',
        category: 'control',
        icon: '🏥',
        requiresConfirmation: false
      }
    ]

    // Initialize quick commands for immediate execution
    this.quickCommands = [
      { command: 'activate-team', description: 'Activate 5-Agent Team', icon: '🚀', shortcut: 'Ctrl+T' },
      { command: 'pause-all', description: 'Pause All Agents', icon: '⏸️', shortcut: 'Ctrl+P' },
      { command: 'resume-all', description: 'Resume All Agents', icon: '▶️', shortcut: 'Ctrl+R' },
      { command: 'system-status', description: 'System Status', icon: '📊', shortcut: 'Ctrl+S' },
      { command: 'emergency-stop', description: 'Emergency Stop', icon: '🛑', shortcut: 'Ctrl+E' }
    ]
  }

  private initializeShortcuts() {
    this.shortcuts = {
      'ctrl+t': 'activate-team',
      'ctrl+p': 'pause-all',
      'ctrl+r': 'resume-all',
      'ctrl+s': 'system-status',
      'ctrl+e': 'emergency-stop',
      'ctrl+v': 'toggle-voice'
    }
  }

  private setupWebSocketListeners() {
    // Connection status monitoring
    this.websocketService.on('connected', () => {
      this.connectionStatus = 'connected'
    })

    this.websocketService.on('disconnected', () => {
      this.connectionStatus = 'disconnected'
    })

    this.websocketService.on('reconnecting', () => {
      this.connectionStatus = 'reconnecting'
    })

    // Command execution responses
    this.websocketService.on('command-response', (data) => {
      this.handleCommandResponse(data)
    })

    this.websocketService.on('command-error', (data) => {
      this.handleCommandError(data)
    })
  }

  private setupKeyboardShortcuts() {
    document.addEventListener('keydown', (event) => {
      const key = `${event.ctrlKey ? 'ctrl+' : ''}${event.key.toLowerCase()}`
      const command = this.shortcuts[key]
      
      if (command && !event.target || (event.target as HTMLElement).tagName !== 'INPUT') {
        event.preventDefault()
        this.executeQuickCommand(command)
      }
    })
  }

  private initializeVoiceControl() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      // @ts-ignore - Browser API may not be in TypeScript definitions
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
      this.recognition = new SpeechRecognition()
      
      this.recognition.continuous = false
      this.recognition.interimResults = false
      this.recognition.lang = 'en-US'

      this.recognition.onresult = (event) => {
        const command = event.results[0][0].transcript.toLowerCase()
        this.processVoiceCommand(command)
      }

      this.recognition.onerror = (event) => {
        console.error('Voice recognition error:', event.error)
        this.voiceControlEnabled = false
      }

      this.recognition.onend = () => {
        this.voiceControlEnabled = false
      }
    }
  }

  private processVoiceCommand(command: string) {
    const commandMap: Record<string, string> = {
      'activate team': 'activate-team',
      'pause all': 'pause-all',
      'resume all': 'resume-all',
      'emergency stop': 'emergency-stop',
      'system status': 'system-status',
      'spawn agent': 'spawn-agent'
    }

    const matchedCommand = Object.entries(commandMap).find(([voice]) => 
      command.includes(voice)
    )?.[1]

    if (matchedCommand) {
      this.executeQuickCommand(matchedCommand)
    }
  }

  private async executeQuickCommand(command: string) {
    const executionId = this.generateExecutionId()
    const execution: CommandExecution = {
      id: executionId,
      command,
      timestamp: new Date(),
      status: 'pending',
      agentIds: this.selectedAgents
    }

    this.activeExecutions.set(executionId, execution)
    this.commandHistory.unshift(execution)
    this.requestUpdate()

    try {
      execution.status = 'executing'
      this.requestUpdate()

      // Execute the command via WebSocket
      if (this.selectedAgents.length > 0) {
        this.websocketService.sendBulkAgentCommand(this.selectedAgents, command)
      } else {
        this.websocketService.sendMessage({
          type: 'system-command',
          data: { command, parameters: {} }
        })
      }

      // Set timeout for command execution
      setTimeout(() => {
        if (execution.status === 'executing') {
          execution.status = 'failed'
          execution.error = 'Command execution timeout'
          this.requestUpdate()
        }
      }, this.commandExecutionTimeout)

    } catch (error) {
      execution.status = 'failed'
      execution.error = error instanceof Error ? error.message : 'Unknown error'
      this.requestUpdate()
    }
  }

  private async executeAdvancedCommand(command: RemoteCommand, parameters: Record<string, any>) {
    if (command.requiresConfirmation) {
      const confirmed = confirm(`Are you sure you want to execute "${command.name}"?`)
      if (!confirmed) return
    }

    const executionId = this.generateExecutionId()
    const execution: CommandExecution = {
      id: executionId,
      command: command.id,
      timestamp: new Date(),
      status: 'executing',
      agentIds: this.selectedAgents
    }

    this.activeExecutions.set(executionId, execution)
    this.commandHistory.unshift(execution)
    this.selectedCommand = null
    this.requestUpdate()

    try {
      // Send command with parameters
      this.websocketService.sendMessage({
        type: 'advanced-command',
        data: {
          commandId: command.id,
          parameters,
          agentIds: this.selectedAgents
        }
      })

      // Set timeout for command execution
      setTimeout(() => {
        if (execution.status === 'executing') {
          execution.status = 'failed'
          execution.error = 'Command execution timeout'
          this.requestUpdate()
        }
      }, this.commandExecutionTimeout)

    } catch (error) {
      execution.status = 'failed'
      execution.error = error instanceof Error ? error.message : 'Unknown error'
      this.requestUpdate()
    }
  }

  private handleCommandResponse(data: any) {
    const execution = Array.from(this.activeExecutions.values())
      .find(e => e.id === data.executionId || e.command === data.command)

    if (execution) {
      execution.status = 'completed'
      execution.result = data.result
      this.activeExecutions.delete(execution.id)
      this.requestUpdate()
    }
  }

  private handleCommandError(data: any) {
    const execution = Array.from(this.activeExecutions.values())
      .find(e => e.id === data.executionId || e.command === data.command)

    if (execution) {
      execution.status = 'failed'
      execution.error = data.error
      this.activeExecutions.delete(execution.id)
      this.requestUpdate()
    }
  }

  private generateExecutionId(): string {
    return `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private toggleVoiceControl() {
    if (!this.recognition) return

    if (this.voiceControlEnabled) {
      this.recognition.stop()
      this.voiceControlEnabled = false
    } else {
      this.recognition.start()
      this.voiceControlEnabled = true
    }
  }

  private openCommandForm(command: RemoteCommand) {
    this.selectedCommand = command
    this.commandParameters = {}
  }

  private closeCommandForm() {
    this.selectedCommand = null
    this.commandParameters = {}
  }

  private submitCommand() {
    if (this.selectedCommand) {
      this.executeAdvancedCommand(this.selectedCommand, this.commandParameters)
    }
  }

  private cleanup() {
    if (this.recognition) {
      this.recognition.stop()
    }
  }

  private formatTimestamp(date: Date): string {
    return date.toLocaleTimeString()
  }

  render() {
    return html`
      <div class="control-center-container">
        <div class="header">
          <div class="header-title">
            <div class="emergency-indicator ${this.emergencyMode ? 'emergency' : ''}"></div>
            <h1>🎛️ Remote Control Center</h1>
          </div>
          
          <div class="connection-status">
            <div class="status-dot ${this.connectionStatus}"></div>
            <span>${this.connectionStatus}</span>
          </div>
        </div>

        ${this.selectedAgents.length > 0 ? html`
          <div class="bulk-selection-bar">
            <div>${this.selectedAgents.length} agents selected</div>
            <div class="bulk-actions">
              <button class="bulk-action-btn" @click=${() => this.executeQuickCommand('pause')}>
                Pause Selected
              </button>
              <button class="bulk-action-btn" @click=${() => this.executeQuickCommand('restart')}>
                Restart Selected
              </button>
              <button class="bulk-action-btn" @click=${() => this.executeQuickCommand('terminate')}>
                Terminate Selected
              </button>
            </div>
          </div>
        ` : ''}

        <div class="control-panel">
          <div class="command-section">
            <div class="section-title">
              ⚡ Quick Commands
            </div>
            
            <div class="quick-commands">
              ${this.quickCommands.map(cmd => html`
                <button class="quick-command-btn ${cmd.command.includes('emergency') ? 'emergency' : ''}"
                        @click=${() => this.executeQuickCommand(cmd.command)}>
                  <div class="keyboard-shortcut">${cmd.shortcut}</div>
                  <div class="command-icon">${cmd.icon}</div>
                  <div>${cmd.description}</div>
                </button>
              `)}
            </div>

            <div class="section-title">
              🛠️ Advanced Commands
            </div>
            
            <div class="advanced-commands">
              <div class="command-list">
                ${this.availableCommands.map(command => html`
                  <div class="command-item ${this.selectedCommand?.id === command.id ? 'selected' : ''}"
                       @click=${() => this.openCommandForm(command)}>
                    <div class="command-icon">${command.icon}</div>
                    <div class="command-meta">
                      <div class="command-name">${command.name}</div>
                      <div class="command-description">${command.description}</div>
                    </div>
                    <div class="command-category ${command.category}">${command.category}</div>
                  </div>
                `)}
              </div>
            </div>
          </div>

          <div class="command-section">
            <div class="section-title">
              📋 Execution History
            </div>
            
            <div class="execution-panel">
              <div class="execution-history">
                ${this.commandHistory.slice(0, 10).map(execution => html`
                  <div class="execution-item ${execution.status}">
                    <div class="execution-header">
                      <div class="execution-command">${execution.command}</div>
                      <div class="execution-timestamp">${this.formatTimestamp(execution.timestamp)}</div>
                    </div>
                    <div class="execution-status ${execution.status}">${execution.status}</div>
                    ${execution.result ? html`
                      <div class="execution-result">${JSON.stringify(execution.result, null, 2)}</div>
                    ` : ''}
                    ${execution.error ? html`
                      <div class="execution-result" style="color: #f87171;">${execution.error}</div>
                    ` : ''}
                  </div>
                `)}
              </div>
            </div>
          </div>
        </div>

        ${this.selectedCommand ? html`
          <div class="command-form">
            <div class="form-header">
              <div class="form-title">
                ${this.selectedCommand.icon} ${this.selectedCommand.name}
              </div>
              <button class="close-btn" @click=${this.closeCommandForm}>✕</button>
            </div>

            <div class="command-description">${this.selectedCommand.description}</div>

            ${this.selectedCommand.parameters?.map(param => html`
              <div class="form-group">
                <label class="form-label">${param.name} ${param.required ? '*' : ''}</label>
                
                ${param.type === 'text' ? html`
                  <input type="text" class="form-input" 
                         placeholder="${param.placeholder || ''}"
                         .value=${this.commandParameters[param.name] || ''}
                         @input=${(e: Event) => {
                           this.commandParameters[param.name] = (e.target as HTMLInputElement).value
                         }}>
                ` : ''}

                ${param.type === 'number' ? html`
                  <input type="number" class="form-input"
                         min="${param.min || 0}"
                         max="${param.max || 100}"
                         .value=${this.commandParameters[param.name] || ''}
                         @input=${(e: Event) => {
                           this.commandParameters[param.name] = Number((e.target as HTMLInputElement).value)
                         }}>
                ` : ''}

                ${param.type === 'select' ? html`
                  <select class="form-select"
                          .value=${this.commandParameters[param.name] || ''}
                          @change=${(e: Event) => {
                            this.commandParameters[param.name] = (e.target as HTMLSelectElement).value
                          }}>
                    <option value="">Select ${param.name}</option>
                    ${param.options?.map(option => html`
                      <option value="${option}">${option}</option>
                    `)}
                  </select>
                ` : ''}

                ${param.type === 'boolean' ? html`
                  <label class="form-checkbox">
                    <input type="checkbox" class="checkbox-input"
                           .checked=${this.commandParameters[param.name] || false}
                           @change=${(e: Event) => {
                             this.commandParameters[param.name] = (e.target as HTMLInputElement).checked
                           }}>
                    <span>Enable ${param.name}</span>
                  </label>
                ` : ''}
              </div>
            `)}

            <div class="form-actions">
              <button class="btn secondary" @click=${this.closeCommandForm}>Cancel</button>
              <button class="btn primary" @click=${this.submitCommand}>
                ${this.selectedCommand.icon} Execute Command
              </button>
            </div>
          </div>
        ` : ''}

        ${this.recognition ? html`
          <button class="voice-control ${this.voiceControlEnabled ? 'active' : ''}"
                  @click=${this.toggleVoiceControl}
                  title="${this.voiceControlEnabled ? 'Stop' : 'Start'} Voice Control">
            ${this.voiceControlEnabled ? '🛑' : '🎙️'}
          </button>
        ` : ''}
      </div>
    `
  }
}