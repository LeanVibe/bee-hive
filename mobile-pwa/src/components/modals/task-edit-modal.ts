import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { Task, TaskStatus, TaskPriority, TaskType } from '../../types/task'
import { AgentRole } from '../../types/api'
import { getAgentService, getTaskService } from '../../services'
import '../common/loading-spinner'

export interface TaskFormData {
  title: string
  description: string
  status: TaskStatus
  priority: TaskPriority
  type: TaskType
  assignee?: string
  tags: string[]
  estimatedHours?: number
  dependencies: string[]
  acceptanceCriteria: string[]
  dueDate?: string
}

@customElement('task-edit-modal')
export class TaskEditModal extends LitElement {
  @property({ type: Boolean }) open = false
  @property({ type: Object }) task?: Task
  @property({ type: String }) mode: 'create' | 'edit' = 'create'
  
  @state() private isLoading = false
  @state() private error = ''
  @state() private availableAgents: Array<{id: string, name: string, role: AgentRole}> = []
  @state() private availableTasks: Array<{id: string, title: string}> = []
  @state() private formData: TaskFormData = {
    title: '',
    description: '',
    status: 'todo',
    priority: 'medium',
    type: 'feature',
    tags: [],
    dependencies: [],
    acceptanceCriteria: ['']
  }
  @state() private tagInput = ''
  @state() private criteriaInput = ''

  private agentService = getAgentService()
  private taskService = getTaskService()

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
      max-width: 700px;
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
      min-height: 100px;
    }

    .form-textarea:focus {
      outline: none;
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .form-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }

    .form-row-3 {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 1rem;
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

    .priority-option.low.selected {
      background: #d1fae5;
      border-color: #10b981;
      color: #065f46;
    }

    .priority-option.medium.selected {
      background: #dbeafe;
      border-color: #3b82f6;
      color: #1d4ed8;
    }

    .priority-option.high.selected {
      background: #fef3c7;
      border-color: #f59e0b;
      color: #92400e;
    }

    .priority-option.critical.selected {
      background: #fee2e2;
      border-color: #ef4444;
      color: #991b1b;
    }

    .tags-container {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }

    .tag {
      background: #e0e7ff;
      color: #3730a3;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .tag-remove {
      background: none;
      border: none;
      color: currentColor;
      cursor: pointer;
      padding: 0;
      display: flex;
      align-items: center;
      font-size: 14px;
    }

    .tag-input-container {
      display: flex;
      gap: 0.5rem;
    }

    .tag-input {
      flex: 1;
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      font-size: 0.875rem;
    }

    .tag-add-button {
      background: #3b82f6;
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .tag-add-button:hover {
      background: #2563eb;
    }

    .tag-add-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .dependencies-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      margin-top: 0.5rem;
    }

    .dependency-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #f9fafb;
      padding: 0.75rem;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
    }

    .dependency-info {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex: 1;
    }

    .dependency-title {
      font-size: 0.875rem;
      color: #374151;
      font-weight: 500;
    }

    .dependency-remove {
      background: none;
      border: none;
      color: #ef4444;
      cursor: pointer;
      padding: 0.25rem;
      border-radius: 0.25rem;
      transition: all 0.2s ease;
    }

    .dependency-remove:hover {
      background: #fef2f2;
    }

    .criteria-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }

    .criteria-item {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      background: #f9fafb;
      padding: 0.75rem;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
    }

    .criteria-checkbox {
      cursor: pointer;
    }

    .criteria-text {
      flex: 1;
      font-size: 0.875rem;
      color: #374151;
    }

    .criteria-remove {
      background: none;
      border: none;
      color: #ef4444;
      cursor: pointer;
      padding: 0.25rem;
      border-radius: 0.25rem;
      transition: all 0.2s ease;
    }

    .criteria-remove:hover {
      background: #fef2f2;
    }

    .criteria-input-container {
      display: flex;
      gap: 0.5rem;
    }

    .criteria-input {
      flex: 1;
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      font-size: 0.875rem;
    }

    .criteria-add-button {
      background: #10b981;
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .criteria-add-button:hover {
      background: #059669;
    }

    .criteria-add-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
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

    .multi-select {
      border: 1px solid #d1d5db;
      border-radius: 0.5rem;
      max-height: 150px;
      overflow-y: auto;
    }

    .multi-select-option {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 0.75rem;
      cursor: pointer;
      transition: background-color 0.2s ease;
    }

    .multi-select-option:hover {
      background: #f9fafb;
    }

    .multi-select-option.selected {
      background: #e0e7ff;
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

      .form-row,
      .form-row-3 {
        grid-template-columns: 1fr;
      }

      .priority-selector {
        grid-template-columns: repeat(2, 1fr);
      }
    }
  `

  async connectedCallback() {
    super.connectedCallback()
    await this.loadFormData()
    if (this.task && this.mode === 'edit') {
      this.loadTaskData()
    }
  }

  private async loadFormData() {
    try {
      // Load available agents
      const agents = this.agentService.getAgents()
      this.availableAgents = agents.map(agent => ({
        id: agent.id,
        name: agent.name,
        role: agent.role
      }))

      // Load available tasks for dependencies
      const tasks = await this.taskService.getTasks()
      this.availableTasks = tasks.map(task => ({
        id: task.id,
        title: task.title
      }))
    } catch (error) {
      console.error('Failed to load form data:', error)
    }
  }

  private loadTaskData() {
    if (!this.task) return

    this.formData = {
      title: this.task.title,
      description: this.task.description || '',
      status: this.task.status,
      priority: this.task.priority,
      type: this.task.type,
      assignee: this.task.assignee,
      tags: [...(this.task.tags || [])],
      estimatedHours: this.task.estimatedHours,
      dependencies: this.task.dependencies || [],
      acceptanceCriteria: this.task.acceptanceCriteria || [''],
      dueDate: this.task.dueDate
    }
  }

  private handleClose() {
    if (this.isLoading) return
    this.open = false
    this.error = ''
    this.dispatchEvent(new CustomEvent('close', { bubbles: true, composed: true }))
  }

  private handleInputChange(field: keyof TaskFormData, value: any) {
    this.formData = {
      ...this.formData,
      [field]: value
    }
  }

  private handlePrioritySelect(priority: TaskPriority) {
    this.handleInputChange('priority', priority)
  }

  private handleAddTag() {
    const tag = this.tagInput.trim()
    if (tag && !this.formData.tags.includes(tag)) {
      this.handleInputChange('tags', [...this.formData.tags, tag])
      this.tagInput = ''
    }
  }

  private handleRemoveTag(tagToRemove: string) {
    this.handleInputChange('tags', this.formData.tags.filter(tag => tag !== tagToRemove))
  }

  private handleAddDependency(taskId: string) {
    if (!this.formData.dependencies.includes(taskId)) {
      this.handleInputChange('dependencies', [...this.formData.dependencies, taskId])
    }
  }

  private handleRemoveDependency(taskId: string) {
    this.handleInputChange('dependencies', this.formData.dependencies.filter(id => id !== taskId))
  }

  private handleAddCriteria() {
    const criteria = this.criteriaInput.trim()
    if (criteria) {
      // Replace the last empty criteria or add new one
      const newCriteria = [...this.formData.acceptanceCriteria]
      if (newCriteria[newCriteria.length - 1] === '') {
        newCriteria[newCriteria.length - 1] = criteria
      } else {
        newCriteria.push(criteria)
      }
      this.handleInputChange('acceptanceCriteria', newCriteria)
      this.criteriaInput = ''
    }
  }

  private handleRemoveCriteria(index: number) {
    const newCriteria = this.formData.acceptanceCriteria.filter((_, i) => i !== index)
    // Ensure at least one empty criteria remains
    if (newCriteria.length === 0 || newCriteria[newCriteria.length - 1] !== '') {
      newCriteria.push('')
    }
    this.handleInputChange('acceptanceCriteria', newCriteria)
  }

  private async handleSubmit() {
    if (this.isLoading) return

    this.error = ''
    this.isLoading = true

    try {
      // Validate form data
      if (!this.formData.title.trim()) {
        throw new Error('Task title is required')
      }

      if (!this.formData.description.trim()) {
        throw new Error('Task description is required')
      }

      // Clean up acceptance criteria
      const cleanCriteria = this.formData.acceptanceCriteria.filter(c => c.trim())
      
      const taskData = {
        ...this.formData,
        acceptanceCriteria: cleanCriteria
      }

      // Emit save event with form data
      this.dispatchEvent(new CustomEvent('save', {
        detail: {
          mode: this.mode,
          task: this.task,
          data: taskData
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
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/>
              </svg>
              ${this.mode === 'edit' ? 'Edit Task' : 'Create New Task'}
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
                <label class="form-label">Task Title</label>
                <input
                  type="text"
                  class="form-input"
                  placeholder="Enter task title"
                  .value=${this.formData.title}
                  @input=${(e: Event) => this.handleInputChange('title', (e.target as HTMLInputElement).value)}
                  ?disabled=${this.isLoading}
                />
              </div>

              <div class="form-group">
                <label class="form-label">Description</label>
                <textarea
                  class="form-textarea"
                  placeholder="Describe the task in detail"
                  .value=${this.formData.description}
                  @input=${(e: Event) => this.handleInputChange('description', (e.target as HTMLTextAreaElement).value)}
                  ?disabled=${this.isLoading}
                ></textarea>
              </div>

              <div class="form-row-3">
                <div class="form-group">
                  <label class="form-label">Status</label>
                  <select
                    class="form-select"
                    .value=${this.formData.status}
                    @change=${(e: Event) => this.handleInputChange('status', (e.target as HTMLSelectElement).value)}
                    ?disabled=${this.isLoading}
                  >
                    <option value="todo">To Do</option>
                    <option value="in-progress">In Progress</option>
                    <option value="review">Review</option>
                    <option value="done">Done</option>
                  </select>
                </div>

                <div class="form-group">
                  <label class="form-label">Type</label>
                  <select
                    class="form-select"
                    .value=${this.formData.type}
                    @change=${(e: Event) => this.handleInputChange('type', (e.target as HTMLSelectElement).value)}
                    ?disabled=${this.isLoading}
                  >
                    <option value="feature">Feature</option>
                    <option value="bug">Bug</option>
                    <option value="enhancement">Enhancement</option>
                    <option value="documentation">Documentation</option>
                    <option value="testing">Testing</option>
                    <option value="refactoring">Refactoring</option>
                  </select>
                </div>

                <div class="form-group">
                  <label class="form-label">Assignee</label>
                  <select
                    class="form-select"
                    .value=${this.formData.assignee || ''}
                    @change=${(e: Event) => this.handleInputChange('assignee', (e.target as HTMLSelectElement).value || undefined)}
                    ?disabled=${this.isLoading}
                  >
                    <option value="">Unassigned</option>
                    ${this.availableAgents.map(agent => html`
                      <option value=${agent.id}>${agent.name} (${agent.role})</option>
                    `)}
                  </select>
                </div>
              </div>
            </div>

            <!-- Priority -->
            <div class="form-section">
              <h3 class="section-title">
                <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                </svg>
                Priority & Planning
              </h3>

              <div class="form-group">
                <label class="form-label">Priority Level</label>
                <div class="priority-selector">
                  ${(['low', 'medium', 'high', 'critical'] as const).map(priority => html`
                    <div
                      class="priority-option ${priority} ${this.formData.priority === priority ? 'selected' : ''}"
                      @click=${() => this.handlePrioritySelect(priority)}
                    >
                      ${priority.toUpperCase()}
                    </div>
                  `)}
                </div>
              </div>

              <div class="form-row">
                <div class="form-group">
                  <label class="form-label">Estimated Hours</label>
                  <input
                    type="number"
                    class="form-input"
                    placeholder="0"
                    min="0"
                    step="0.5"
                    .value=${this.formData.estimatedHours?.toString() || ''}
                    @input=${(e: Event) => this.handleInputChange('estimatedHours', parseFloat((e.target as HTMLInputElement).value) || undefined)}
                    ?disabled=${this.isLoading}
                  />
                </div>

                <div class="form-group">
                  <label class="form-label">Due Date</label>
                  <input
                    type="date"
                    class="form-input"
                    .value=${this.formData.dueDate || ''}
                    @input=${(e: Event) => this.handleInputChange('dueDate', (e.target as HTMLInputElement).value || undefined)}
                    ?disabled=${this.isLoading}
                  />
                </div>
              </div>
            </div>

            <!-- Tags -->
            <div class="form-section">
              <h3 class="section-title">
                <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"/>
                </svg>
                Tags & Labels
              </h3>

              <div class="form-group">
                <label class="form-label">Tags</label>
                <div class="tags-container">
                  ${this.formData.tags.map(tag => html`
                    <div class="tag">
                      ${tag}
                      <button
                        class="tag-remove"
                        @click=${() => this.handleRemoveTag(tag)}
                        ?disabled=${this.isLoading}
                      >
                        ×
                      </button>
                    </div>
                  `)}
                </div>
                <div class="tag-input-container">
                  <input
                    type="text"
                    class="tag-input"
                    placeholder="Add tag"
                    .value=${this.tagInput}
                    @input=${(e: Event) => this.tagInput = (e.target as HTMLInputElement).value}
                    @keypress=${(e: KeyboardEvent) => e.key === 'Enter' && this.handleAddTag()}
                    ?disabled=${this.isLoading}
                  />
                  <button
                    class="tag-add-button"
                    @click=${this.handleAddTag}
                    ?disabled=${this.isLoading || !this.tagInput.trim()}
                  >
                    Add
                  </button>
                </div>
              </div>
            </div>

            <!-- Dependencies -->
            <div class="form-section">
              <h3 class="section-title">
                <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/>
                </svg>
                Dependencies
              </h3>

              <div class="form-group">
                <label class="form-label">Task Dependencies</label>
                <div class="dependencies-list">
                  ${this.formData.dependencies.map(depId => {
                    const task = this.availableTasks.find(t => t.id === depId)
                    return html`
                      <div class="dependency-item">
                        <div class="dependency-info">
                          <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
                          </svg>
                          <span class="dependency-title">${task?.title || depId}</span>
                        </div>
                        <button
                          class="dependency-remove"
                          @click=${() => this.handleRemoveDependency(depId)}
                          ?disabled=${this.isLoading}
                        >
                          <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                          </svg>
                        </button>
                      </div>
                    `
                  })}
                </div>
                <div class="multi-select">
                  ${this.availableTasks
                    .filter(task => task.id !== this.task?.id) // Don't show current task
                    .map(task => html`
                      <div
                        class="multi-select-option ${this.formData.dependencies.includes(task.id) ? 'selected' : ''}"
                        @click=${() => this.formData.dependencies.includes(task.id) 
                          ? this.handleRemoveDependency(task.id)
                          : this.handleAddDependency(task.id)}
                      >
                        <input
                          type="checkbox"
                          .checked=${this.formData.dependencies.includes(task.id)}
                          @change=${() => {}}
                        />
                        <span>${task.title}</span>
                      </div>
                    `)}
                </div>
              </div>
            </div>

            <!-- Acceptance Criteria -->
            <div class="form-section">
              <h3 class="section-title">
                <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                Acceptance Criteria
              </h3>

              <div class="form-group">
                <label class="form-label">Completion Criteria</label>
                <div class="criteria-list">
                  ${this.formData.acceptanceCriteria.map((criteria, index) => html`
                    <div class="criteria-item">
                      <input type="checkbox" class="criteria-checkbox" disabled />
                      <span class="criteria-text">${criteria || 'Empty criteria'}</span>
                      <button
                        class="criteria-remove"
                        @click=${() => this.handleRemoveCriteria(index)}
                        ?disabled=${this.isLoading}
                      >
                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                        </svg>
                      </button>
                    </div>
                  `)}
                </div>
                <div class="criteria-input-container">
                  <input
                    type="text"
                    class="criteria-input"
                    placeholder="Add acceptance criteria"
                    .value=${this.criteriaInput}
                    @input=${(e: Event) => this.criteriaInput = (e.target as HTMLInputElement).value}
                    @keypress=${(e: KeyboardEvent) => e.key === 'Enter' && this.handleAddCriteria()}
                    ?disabled=${this.isLoading}
                  />
                  <button
                    class="criteria-add-button"
                    @click=${this.handleAddCriteria}
                    ?disabled=${this.isLoading || !this.criteriaInput.trim()}
                  >
                    Add
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div class="modal-footer">
            <button class="button button-secondary" @click=${this.handleClose} ?disabled=${this.isLoading}>
              Cancel
            </button>
            <button class="button button-primary" @click=${this.handleSubmit} ?disabled=${this.isLoading}>
              ${this.isLoading ? html`<loading-spinner size="small"></loading-spinner>` : ''}
              ${this.mode === 'edit' ? 'Update Task' : 'Create Task'}
            </button>
          </div>
        </div>
      </div>
    `
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'task-edit-modal': TaskEditModal
  }
}