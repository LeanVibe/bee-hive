import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { AgentRole, AgentStatus } from '../../types/api'
import type { Agent } from '../../types/api'
import '../common/loading-spinner'

export interface AgentConfigData {
  name: string
  role: AgentRole
  capabilities: string[]
  priority: 'low' | 'normal' | 'high' | 'critical'
  maxConcurrentTasks: number
  autoAssign: boolean
  specialization?: string
  teamIntegration: boolean
}

@customElement('agent-config-modal')
export class AgentConfigModal extends LitElement {
  @property({ type: Boolean }) open = false
  @property({ type: Object }) agent?: Agent
  @property({ type: String }) mode: 'create' | 'edit' = 'create'
  
  @state() private isLoading = false
  @state() private error = ''
  @state() private formData: AgentConfigData = {
    name: '',
    role: AgentRole.BACKEND_DEVELOPER,
    capabilities: [],
    priority: 'normal',
    maxConcurrentTasks: 5,
    autoAssign: true,
    teamIntegration: true
  }
  @state() private availableCapabilities = [
    'code-generation',
    'testing',
    'debugging',
    'documentation',
    'code-review',
    'architecture-design',
    'deployment',
    'monitoring',
    'security-analysis',
    'performance-optimization',
    'database-design',
    'api-development'
  ]

  static styles = css`
    :host {
      display: block;
    }

    .modal-overlay {
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
      padding: 1rem;
      opacity: 0;
      visibility: hidden;
      transition: all 0.3s ease;
    }

    .modal-overlay.open {
      opacity: 1;
      visibility: visible;
    }

    .modal {
      background: white;
      border-radius: 1rem;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
      width: 100%;
      max-width: 600px;
      max-height: 90vh;
      overflow-y: auto;
      transform: scale(0.95) translateY(20px);
      transition: all 0.3s ease;
    }

    .modal-overlay.open .modal {
      transform: scale(1) translateY(0);
    }

    .modal-header {
      padding: 2rem 2rem 1rem 2rem;
      border-bottom: 1px solid #e5e7eb;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .modal-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .title-icon {
      width: 24px;
      height: 24px;
      color: #3b82f6;
    }

    .close-button {
      background: none;
      border: none;
      color: #6b7280;
      cursor: pointer;
      padding: 0.5rem;
      border-radius: 0.5rem;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .close-button:hover {
      background: #f3f4f6;
      color: #374151;
    }

    .modal-content {
      padding: 2rem;
    }

    .form-section {
      margin-bottom: 2rem;
    }

    .section-title {
      font-size: 1.125rem;
      font-weight: 600;
      color: #111827;
      margin: 0 0 1rem 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .section-icon {
      width: 18px;
      height: 18px;
      color: #6b7280;
    }

    .form-group {
      margin-bottom: 1.5rem;
    }

    .form-label {
      display: block;
      font-weight: 500;
      color: #374151;
      margin-bottom: 0.5rem;
      font-size: 0.875rem;
    }

    .form-input {
      width: 100%;
      padding: 0.75rem;
      border: 1px solid #d1d5db;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      transition: all 0.2s ease;
      background: white;
    }

    .form-input:focus {
      outline: none;
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .form-select {
      width: 100%;
      padding: 0.75rem;
      border: 1px solid #d1d5db;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      background: white;
      cursor: pointer;
    }

    .form-select:focus {
      outline: none;
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .form-textarea {
      width: 100%;
      padding: 0.75rem;
      border: 1px solid #d1d5db;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      resize: vertical;
      min-height: 80px;
    }

    .form-textarea:focus {
      outline: none;
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .capabilities-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.75rem;
      margin-top: 0.5rem;
    }

    .capability-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem;
      background: #f9fafb;
      border-radius: 0.375rem;
      transition: all 0.2s ease;
    }

    .capability-item:hover {
      background: #f3f4f6;
    }

    .capability-checkbox {
      cursor: pointer;
    }

    .capability-label {
      font-size: 0.875rem;
      color: #374151;
      cursor: pointer;
      text-transform: capitalize;
      flex: 1;
    }

    .form-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }

    .checkbox-group {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-top: 0.5rem;
    }

    .checkbox-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .checkbox-label {
      font-size: 0.875rem;
      color: #374151;
      cursor: pointer;
    }

    .priority-selector {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.5rem;
      margin-top: 0.5rem;
    }

    .priority-option {
      background: #f9fafb;
      border: 2px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 0.75rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      font-size: 0.875rem;
      font-weight: 500;
    }

    .priority-option:hover {
      border-color: #d1d5db;
      background: #f3f4f6;
    }

    .priority-option.selected {
      background: #dbeafe;
      border-color: #3b82f6;
      color: #1d4ed8;
    }

    .priority-option.low { --priority-color: #10b981; }
    .priority-option.normal { --priority-color: #3b82f6; }
    .priority-option.high { --priority-color: #f59e0b; }
    .priority-option.critical { --priority-color: #ef4444; }

    .priority-option.selected.low {
      background: #d1fae5;
      border-color: #10b981;
      color: #065f46;
    }

    .priority-option.selected.normal {
      background: #dbeafe;
      border-color: #3b82f6;
      color: #1d4ed8;
    }

    .priority-option.selected.high {
      background: #fef3c7;
      border-color: #f59e0b;
      color: #92400e;
    }

    .priority-option.selected.critical {
      background: #fee2e2;
      border-color: #ef4444;
      color: #991b1b;
    }

    .modal-footer {
      padding: 1.5rem 2rem 2rem 2rem;
      display: flex;
      gap: 1rem;
      justify-content: flex-end;
      border-top: 1px solid #e5e7eb;
    }

    .button {
      padding: 0.75rem 1.5rem;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      border: none;
    }

    .button-secondary {
      background: white;
      color: #374151;
      border: 1px solid #d1d5db;
    }

    .button-secondary:hover {
      background: #f9fafb;
      border-color: #9ca3af;
    }

    .button-primary {
      background: #3b82f6;
      color: white;
    }

    .button-primary:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }

    .button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }

    .error-message {
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #dc2626;
      padding: 0.75rem;
      border-radius: 0.5rem;
      margin-bottom: 1rem;
      font-size: 0.875rem;
    }

    .help-text {
      font-size: 0.75rem;
      color: #6b7280;
      margin-top: 0.25rem;
    }

    .range-input {
      width: 100%;
      margin: 0.5rem 0;
    }

    .range-value {
      display: inline-block;
      background: #f3f4f6;
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.875rem;
      font-weight: 500;
      color: #374151;
      margin-left: 0.5rem;
    }

    /* Responsive Design */
    @media (max-width: 640px) {
      .modal {
        margin: 0;
        border-radius: 0;
        height: 100vh;
        max-height: 100vh;
      }

      .modal-header,
      .modal-content,
      .modal-footer {
        padding-left: 1rem;
        padding-right: 1rem;
      }

      .form-row {
        grid-template-columns: 1fr;
      }

      .capabilities-grid {
        grid-template-columns: 1fr;
      }

      .priority-selector {
        grid-template-columns: repeat(2, 1fr);
      }
    }
  `

  connectedCallback() {
    super.connectedCallback()
    if (this.agent && this.mode === 'edit') {
      this.loadAgentData()
    }
  }

  private loadAgentData() {
    if (!this.agent) return

    this.formData = {
      name: this.agent.name,
      role: this.agent.role,
      capabilities: [...this.agent.capabilities],
      priority: 'normal', // Default, could be extracted from agent metadata
      maxConcurrentTasks: 5, // Default, could be extracted from agent metadata
      autoAssign: true, // Default, could be extracted from agent metadata
      specialization: '', // Could be extracted from agent metadata
      teamIntegration: true // Default, could be extracted from agent metadata
    }
  }

  private handleClose() {
    if (this.isLoading) return
    this.open = false
    this.error = ''
    this.dispatchEvent(new CustomEvent('close', { bubbles: true, composed: true }))
  }

  private handleInputChange(field: keyof AgentConfigData, value: any) {
    this.formData = {
      ...this.formData,
      [field]: value
    }
  }

  private handleCapabilityToggle(capability: string) {
    const capabilities = [...this.formData.capabilities]
    const index = capabilities.indexOf(capability)
    
    if (index >= 0) {
      capabilities.splice(index, 1)
    } else {
      capabilities.push(capability)
    }
    
    this.handleInputChange('capabilities', capabilities)
  }

  private handlePrioritySelect(priority: AgentConfigData['priority']) {
    this.handleInputChange('priority', priority)
  }

  private async handleSubmit() {
    if (this.isLoading) return

    this.error = ''
    this.isLoading = true

    try {
      // Validate form data
      if (!this.formData.name.trim()) {
        throw new Error('Agent name is required')
      }

      if (this.formData.capabilities.length === 0) {
        throw new Error('At least one capability must be selected')
      }

      // Emit save event with form data
      this.dispatchEvent(new CustomEvent('save', {
        detail: {
          mode: this.mode,
          agent: this.agent,
          config: this.formData
        },
        bubbles: true,
        composed: true
      }))

      // Close modal on successful save
      this.handleClose()

    } catch (error) {
      this.error = error instanceof Error ? error.message : 'An error occurred'
    } finally {
      this.isLoading = false
    }
  }

  render() {
    return html`
      <div class="modal-overlay ${this.open ? 'open' : ''}" @click=${this.handleClose}>
        <div class="modal" @click=${(e: Event) => e.stopPropagation()}>
          <div class="modal-header">
            <h2 class="modal-title">
              <svg class="title-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
              </svg>
              ${this.mode === 'edit' ? 'Configure Agent' : 'Create New Agent'}
            </h2>
            <button class="close-button" @click=${this.handleClose} ?disabled=${this.isLoading}>
              <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
              </svg>
            </button>
          </div>

          <div class="modal-content">
            ${this.error ? html`
              <div class="error-message">${this.error}</div>
            ` : ''}

            <!-- Basic Information -->
            <div class="form-section">
              <h3 class="section-title">
                <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                Basic Information
              </h3>

              <div class="form-group">
                <label class="form-label">Agent Name</label>
                <input
                  type="text"
                  class="form-input"
                  placeholder="Enter agent name"
                  .value=${this.formData.name}
                  @input=${(e: Event) => this.handleInputChange('name', (e.target as HTMLInputElement).value)}
                  ?disabled=${this.isLoading}
                />
                <div class="help-text">Choose a descriptive name for your agent</div>
              </div>

              <div class="form-group">
                <label class="form-label">Role</label>
                <select
                  class="form-select"
                  .value=${this.formData.role}
                  @change=${(e: Event) => this.handleInputChange('role', (e.target as HTMLSelectElement).value)}
                  ?disabled=${this.isLoading}
                >
                  ${Object.values(AgentRole).map(role => html`
                    <option value=${role}>${role.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase())}</option>
                  `)}
                </select>
                <div class="help-text">Select the primary role for this agent</div>
              </div>

              <div class="form-group">
                <label class="form-label">Specialization (Optional)</label>
                <textarea
                  class="form-textarea"
                  placeholder="Describe any specific specializations or focus areas"
                  .value=${this.formData.specialization || ''}
                  @input=${(e: Event) => this.handleInputChange('specialization', (e.target as HTMLTextAreaElement).value)}
                  ?disabled=${this.isLoading}
                ></textarea>
              </div>
            </div>

            <!-- Capabilities -->
            <div class="form-section">
              <h3 class="section-title">
                <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                Capabilities
              </h3>

              <div class="capabilities-grid">
                ${this.availableCapabilities.map(capability => html`
                  <div class="capability-item">
                    <input
                      type="checkbox"
                      class="capability-checkbox"
                      id="cap-${capability}"
                      .checked=${this.formData.capabilities.includes(capability)}
                      @change=${() => this.handleCapabilityToggle(capability)}
                      ?disabled=${this.isLoading}
                    />
                    <label class="capability-label" for="cap-${capability}">
                      ${capability.replace(/-/g, ' ')}
                    </label>
                  </div>
                `)}
              </div>
              <div class="help-text">Select all capabilities this agent should have</div>
            </div>

            <!-- Priority & Performance -->
            <div class="form-section">
              <h3 class="section-title">
                <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                </svg>
                Priority & Performance
              </h3>

              <div class="form-group">
                <label class="form-label">Priority Level</label>
                <div class="priority-selector">
                  ${(['low', 'normal', 'high', 'critical'] as const).map(priority => html`
                    <div
                      class="priority-option ${priority} ${this.formData.priority === priority ? 'selected' : ''}"
                      @click=${() => this.handlePrioritySelect(priority)}
                    >
                      ${priority.toUpperCase()}
                    </div>
                  `)}
                </div>
                <div class="help-text">Higher priority agents get preference for task assignment</div>
              </div>

              <div class="form-group">
                <label class="form-label">Max Concurrent Tasks</label>
                <input
                  type="range"
                  class="range-input"
                  min="1"
                  max="20"
                  .value=${this.formData.maxConcurrentTasks.toString()}
                  @input=${(e: Event) => this.handleInputChange('maxConcurrentTasks', parseInt((e.target as HTMLInputElement).value))}
                  ?disabled=${this.isLoading}
                />
                <span class="range-value">${this.formData.maxConcurrentTasks}</span>
                <div class="help-text">Maximum number of tasks this agent can handle simultaneously</div>
              </div>
            </div>

            <!-- Team Integration -->
            <div class="form-section">
              <h3 class="section-title">
                <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"/>
                </svg>
                Team Integration
              </h3>

              <div class="checkbox-group">
                <div class="checkbox-item">
                  <input
                    type="checkbox"
                    id="auto-assign"
                    .checked=${this.formData.autoAssign}
                    @change=${(e: Event) => this.handleInputChange('autoAssign', (e.target as HTMLInputElement).checked)}
                    ?disabled=${this.isLoading}
                  />
                  <label class="checkbox-label" for="auto-assign">Enable Auto-Assignment</label>
                </div>
                <div class="checkbox-item">
                  <input
                    type="checkbox"
                    id="team-integration"
                    .checked=${this.formData.teamIntegration}
                    @change=${(e: Event) => this.handleInputChange('teamIntegration', (e.target as HTMLInputElement).checked)}
                    ?disabled=${this.isLoading}
                  />
                  <label class="checkbox-label" for="team-integration">Enable Team Collaboration</label>
                </div>
              </div>
              <div class="help-text">Configure how this agent integrates with the team</div>
            </div>
          </div>

          <div class="modal-footer">
            <button class="button button-secondary" @click=${this.handleClose} ?disabled=${this.isLoading}>
              Cancel
            </button>
            <button class="button button-primary" @click=${this.handleSubmit} ?disabled=${this.isLoading}>
              ${this.isLoading ? html`<loading-spinner size="small"></loading-spinner>` : ''}
              ${this.mode === 'edit' ? 'Update Agent' : 'Create Agent'}
            </button>
          </div>
        </div>
      </div>
    `
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'agent-config-modal': AgentConfigModal
  }
}