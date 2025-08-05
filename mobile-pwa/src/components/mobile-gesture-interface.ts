import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../services/websocket'

interface GestureAction {
  id: string
  gesture: 'swipe-right' | 'swipe-left' | 'swipe-up' | 'swipe-down' | 'long-press' | 'double-tap'
  action: string
  command?: string
  description: string
  priority: 'critical' | 'high' | 'medium' | 'low'
  icon: string
  feedback: string
}

interface TouchGesture {
  startX: number
  startY: number
  currentX: number
  currentY: number
  startTime: number
  element: HTMLElement
}

@customElement('mobile-gesture-interface')
export class MobileGestureInterface extends LitElement {
  @property({ type: Boolean }) declare enabled: boolean
  @property({ type: Boolean }) declare trainingMode: boolean
  @property({ type: Number }) declare gestureThreshold: number
  @property({ type: Number }) declare longPressTimeout: number

  @state() private declare currentGesture: TouchGesture | null
  @state() private declare gestureInProgress: boolean
  @state() private declare gesturePreview: string | null
  @state() private declare feedbackMessage: string | null
  @state() private declare gestureActions: GestureAction[]
  @state() private declare showTraining: boolean

  private websocketService: WebSocketService
  private longPressTimer: number | null = null
  private doubleTapTimer: number | null = null
  private lastTapTime = 0
  private hapticFeedback = true

  static styles = css`
    :host {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1000;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .gesture-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: all;
      background: transparent;
    }

    .gesture-preview {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(0, 0, 0, 0.9);
      color: white;
      padding: 1.5rem 2rem;
      border-radius: 16px;
      font-size: 1.25rem;
      font-weight: 600;
      z-index: 1001;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      animation: gesturePreview 0.3s ease-out;
      display: flex;
      align-items: center;
      gap: 1rem;
      min-width: 280px;
      text-align: center;
    }

    .gesture-preview-icon {
      font-size: 2rem;
      opacity: 0.9;
    }

    .gesture-preview-text {
      flex: 1;
    }

    .gesture-preview-priority {
      padding: 0.25rem 0.75rem;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
    }

    .gesture-preview-priority.critical {
      background: #ef4444;
    }

    .gesture-preview-priority.high {
      background: #f59e0b;
    }

    .gesture-preview-priority.medium {
      background: #10b981;
    }

    .gesture-preview-priority.low {
      background: #6b7280;
    }

    .feedback-toast {
      position: fixed;
      bottom: 2rem;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(59, 130, 246, 0.95);
      color: white;
      padding: 1rem 1.5rem;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 500;
      z-index: 1002;
      box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
      backdrop-filter: blur(10px);
      animation: slideUp 0.3s ease-out;
      max-width: 320px;
      text-align: center;
    }

    .training-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      z-index: 1100;
      display: flex;
      align-items: center;
      justify-content: center;
      backdrop-filter: blur(5px);
    }

    .training-panel {
      background: white;
      border-radius: 20px;
      padding: 2rem;
      max-width: 90%;
      width: 400px;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }

    .training-header {
      text-align: center;
      margin-bottom: 2rem;
    }

    .training-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin-bottom: 0.5rem;
    }

    .training-subtitle {
      color: #6b7280;
      font-size: 1rem;
    }

    .gesture-list {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .gesture-item {
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 1rem;
      background: #f9fafb;
      border-radius: 12px;
      border: 2px solid transparent;
      transition: all 0.2s;
    }

    .gesture-item:hover {
      background: #f3f4f6;
      border-color: #e5e7eb;
    }

    .gesture-item-icon {
      font-size: 1.5rem;
      width: 40px;
      text-align: center;
    }

    .gesture-item-content {
      flex: 1;
    }

    .gesture-item-title {
      font-weight: 600;
      color: #111827;
      margin-bottom: 0.25rem;
    }

    .gesture-item-description {
      font-size: 0.875rem;
      color: #6b7280;
      margin-bottom: 0.5rem;
    }

    .gesture-item-action {
      font-size: 0.8rem;
      color: #3b82f6;
      font-weight: 500;
    }

    .training-actions {
      display: flex;
      gap: 1rem;
      justify-content: center;
    }

    .training-button {
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      border: none;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }

    .training-button.primary {
      background: #3b82f6;
      color: white;
    }

    .training-button.primary:hover {
      background: #2563eb;
    }

    .training-button.secondary {
      background: #f3f4f6;
      color: #374151;
    }

    .training-button.secondary:hover {
      background: #e5e7eb;
    }

    .visual-feedback {
      position: fixed;
      pointer-events: none;
      z-index: 999;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(59, 130, 246, 0.3) 0%, transparent 70%);
      animation: ripple 0.6s ease-out;
    }

    /* Gesture direction indicators */
    .gesture-arrow {
      position: fixed;
      font-size: 3rem;
      color: rgba(59, 130, 246, 0.8);
      pointer-events: none;
      z-index: 998;
      animation: gestureArrow 0.4s ease-out;
    }

    @keyframes gesturePreview {
      0% {
        opacity: 0;
        transform: translate(-50%, -50%) scale(0.8);
      }
      100% {
        opacity: 1;
        transform: translate(-50%, -50%) scale(1);
      }
    }

    @keyframes slideUp {
      0% {
        opacity: 0;
        transform: translateX(-50%) translateY(20px);
      }
      100% {
        opacity: 1;
        transform: translateX(-50%) translateY(0);
      }
    }

    @keyframes ripple {
      0% {
        transform: scale(0);
        opacity: 1;
      }
      100% {
        transform: scale(4);
        opacity: 0;
      }
    }

    @keyframes gestureArrow {
      0% {
        opacity: 0;
        transform: scale(0.5);
      }
      50% {
        opacity: 1;
        transform: scale(1.2);
      }
      100% {
        opacity: 0;
        transform: scale(1);
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
    this.enabled = true
    this.trainingMode = false
    this.gestureThreshold = 50
    this.longPressTimeout = 700
    this.currentGesture = null
    this.gestureInProgress = false
    this.gesturePreview = null
    this.feedbackMessage = null
    this.showTraining = false
    this.gestureActions = this.getDefaultGestureActions()
    this.websocketService = WebSocketService.getInstance()
  }

  connectedCallback() {
    super.connectedCallback()
    this.setupGestureListeners()
    this.loadGesturePreferences()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.cleanupGestureListeners()
  }

  private getDefaultGestureActions(): GestureAction[] {
    return [
      {
        id: 'approve-continue',
        gesture: 'swipe-right',
        action: 'Approve/Continue Agent Task',
        command: '/hive:approve-task',
        description: 'Approve current agent activity and allow continuation',
        priority: 'high',
        icon: '✅',
        feedback: 'Task approved - agents continuing'
      },
      {
        id: 'pause-review',
        gesture: 'swipe-left',
        action: 'Pause/Review Agent Activity',
        command: '/hive:pause-for-review',
        description: 'Pause current activity for human review',
        priority: 'medium',
        icon: '⏸️',
        feedback: 'Activity paused for review'
      },
      {
        id: 'escalate',
        gesture: 'swipe-up',
        action: 'Escalate to Human Review',
        command: '/hive:escalate-human',
        description: 'Escalate current decision to human oversight',
        priority: 'critical',
        icon: '🚨',
        feedback: 'Escalated to human review'
      },
      {
        id: 'dismiss-archive',
        gesture: 'swipe-down',
        action: 'Dismiss/Archive Alert',
        command: '/hive:dismiss-alert',
        description: 'Dismiss current alert or archive completed task',
        priority: 'low',
        icon: '📁',
        feedback: 'Alert dismissed'
      },
      {
        id: 'detailed-view',
        gesture: 'long-press',
        action: 'Open Detailed Context',
        command: '/hive:detailed-context',
        description: 'Open detailed view of current context or task',
        priority: 'medium',
        icon: '🔍',
        feedback: 'Opening detailed view'
      },
      {
        id: 'quick-status',
        gesture: 'double-tap',
        action: 'Quick Status Check',
        command: '/hive:quick-status',
        description: 'Show quick system status and agent health',
        priority: 'medium',
        icon: '📊',
        feedback: 'Showing quick status'
      }
    ]
  }

  private setupGestureListeners() {
    if (!this.enabled) return

    document.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false })
    document.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false })
    document.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false })
    document.addEventListener('touchcancel', this.handleTouchCancel.bind(this), { passive: false })
  }

  private cleanupGestureListeners() {
    document.removeEventListener('touchstart', this.handleTouchStart.bind(this))
    document.removeEventListener('touchmove', this.handleTouchMove.bind(this))
    document.removeEventListener('touchend', this.handleTouchEnd.bind(this))
    document.removeEventListener('touchcancel', this.handleTouchCancel.bind(this))
  }

  private handleTouchStart(event: TouchEvent) {
    if (event.touches.length > 1) return // Multi-touch not supported

    const touch = event.touches[0]
    const element = event.target as HTMLElement

    // Skip if touching interactive elements
    if (this.isInteractiveElement(element)) return

    this.currentGesture = {
      startX: touch.clientX,
      startY: touch.clientY,
      currentX: touch.clientX,
      currentY: touch.clientY,
      startTime: Date.now(),
      element
    }

    this.gestureInProgress = true

    // Start long press timer
    this.longPressTimer = window.setTimeout(() => {
      if (this.currentGesture && this.gestureInProgress) {
        this.handleLongPress()
      }
    }, this.longPressTimeout)

    // Add visual feedback
    this.addTouchFeedback(touch.clientX, touch.clientY)
  }

  private handleTouchMove(event: TouchEvent) {
    if (!this.currentGesture || !this.gestureInProgress) return

    const touch = event.touches[0]
    this.currentGesture.currentX = touch.clientX
    this.currentGesture.currentY = touch.clientY

    const deltaX = this.currentGesture.currentX - this.currentGesture.startX
    const deltaY = this.currentGesture.currentY - this.currentGesture.startY
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY)

    // Clear long press if moved too much
    if (distance > 20 && this.longPressTimer) {
      clearTimeout(this.longPressTimer)
      this.longPressTimer = null
    }

    // Show gesture preview if moved enough
    if (distance > this.gestureThreshold) {
      const gesture = this.detectGestureDirection(deltaX, deltaY)
      this.showGesturePreview(gesture)
      
      // Show directional arrow
      this.showDirectionalFeedback(gesture, touch.clientX, touch.clientY)
    }
  }

  private handleTouchEnd(event: TouchEvent) {
    if (!this.currentGesture || !this.gestureInProgress) return

    const deltaX = this.currentGesture.currentX - this.currentGesture.startX
    const deltaY = this.currentGesture.currentY - this.currentGesture.startY
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY)
    const duration = Date.now() - this.currentGesture.startTime

    // Clear timers
    if (this.longPressTimer) {
      clearTimeout(this.longPressTimer)
      this.longPressTimer = null
    }

    // Detect gesture type
    if (distance < 10 && duration < 300) {
      // Potential tap or double tap
      this.handleTap()
    } else if (distance > this.gestureThreshold) {
      // Swipe gesture
      const gestureType = this.detectGestureDirection(deltaX, deltaY)
      this.executeGesture(gestureType)
    }

    this.cleanup()
  }

  private handleTouchCancel() {
    this.cleanup()
  }

  private handleTap() {
    const now = Date.now()
    if (now - this.lastTapTime < 300) {
      // Double tap
      this.executeGesture('double-tap')
      this.lastTapTime = 0
    } else {
      this.lastTapTime = now
      // Wait to see if there's a second tap
      setTimeout(() => {
        if (this.lastTapTime === now) {
          // Single tap - no action for now
          this.lastTapTime = 0
        }
      }, 300)
    }
  }

  private handleLongPress() {
    this.executeGesture('long-press')
    this.triggerHapticFeedback(100) // Heavy feedback
  }

  private detectGestureDirection(deltaX: number, deltaY: number): GestureAction['gesture'] {
    const absX = Math.abs(deltaX)
    const absY = Math.abs(deltaY)

    if (absX > absY) {
      return deltaX > 0 ? 'swipe-right' : 'swipe-left'
    } else {
      return deltaY > 0 ? 'swipe-down' : 'swipe-up'
    }
  }

  private showGesturePreview(gestureType: GestureAction['gesture']) {
    const action = this.gestureActions.find(a => a.gesture === gestureType)
    if (action) {
      this.gesturePreview = action.id
      setTimeout(() => {
        this.gesturePreview = null
      }, 2000)
    }
  }

  private showDirectionalFeedback(gestureType: GestureAction['gesture'], x: number, y: number) {
    const arrows = {
      'swipe-right': '→',
      'swipe-left': '←',
      'swipe-up': '↑',
      'swipe-down': '↓'
    }

    const arrow = arrows[gestureType as keyof typeof arrows]
    if (!arrow) return

    const arrowElement = document.createElement('div')
    arrowElement.className = 'gesture-arrow'
    arrowElement.textContent = arrow
    arrowElement.style.left = `${x - 25}px`
    arrowElement.style.top = `${y - 25}px`

    document.body.appendChild(arrowElement)
    setTimeout(() => arrowElement.remove(), 400)
  }

  private async executeGesture(gestureType: GestureAction['gesture']) {
    const action = this.gestureActions.find(a => a.gesture === gestureType)
    if (!action) return

    // Show feedback
    this.showFeedback(action.feedback)
    this.triggerHapticFeedback(gestureType === 'swipe-up' ? 100 : 50)

    // Execute command if available
    if (action.command) {
      try {
        const response = await fetch('/api/hive/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ command: action.command })
        })

        const result = await response.json()
        if (!result.success) {
          this.showFeedback(`Failed: ${action.action}`, 'error')
        }
      } catch (error) {
        this.showFeedback('Command execution failed', 'error')
      }
    }

    // Dispatch event for other components
    this.dispatchEvent(new CustomEvent('gesture-executed', {
      detail: { action, gestureType },
      bubbles: true,
      composed: true
    }))
  }

  private showFeedback(message: string, type: 'success' | 'error' = 'success') {
    this.feedbackMessage = message
    setTimeout(() => {
      this.feedbackMessage = null
    }, 3000)
  }

  private addTouchFeedback(x: number, y: number) {
    const feedback = document.createElement('div')
    feedback.className = 'visual-feedback'
    feedback.style.left = `${x - 25}px`
    feedback.style.top = `${y - 25}px`
    feedback.style.width = '50px'
    feedback.style.height = '50px'

    document.body.appendChild(feedback)
    setTimeout(() => feedback.remove(), 600)
  }

  private triggerHapticFeedback(intensity: number = 50) {
    if (!this.hapticFeedback) return

    if ('vibrate' in navigator) {
      navigator.vibrate(intensity)
    }
  }

  private isInteractiveElement(element: HTMLElement): boolean {
    const interactiveTags = ['button', 'input', 'textarea', 'select', 'a']
    const interactiveRoles = ['button', 'link', 'textbox']
    
    return interactiveTags.includes(element.tagName.toLowerCase()) ||
           interactiveRoles.includes(element.getAttribute('role') || '') ||
           element.hasAttribute('contenteditable') ||
           element.closest('[data-interactive="true"]') !== null
  }

  private cleanup() {
    this.currentGesture = null
    this.gestureInProgress = false
    this.gesturePreview = null
    
    if (this.longPressTimer) {
      clearTimeout(this.longPressTimer)
      this.longPressTimer = null
    }
  }

  private async loadGesturePreferences() {
    // Load from localStorage or API
    const saved = localStorage.getItem('mobile-gesture-preferences')
    if (saved) {
      try {
        const preferences = JSON.parse(saved)
        this.hapticFeedback = preferences.hapticFeedback !== false
        this.gestureThreshold = preferences.gestureThreshold || 50
        this.longPressTimeout = preferences.longPressTimeout || 700
      } catch (error) {
        console.warn('Failed to load gesture preferences:', error)
      }
    }
  }

  private saveGesturePreferences() {
    const preferences = {
      hapticFeedback: this.hapticFeedback,
      gestureThreshold: this.gestureThreshold,
      longPressTimeout: this.longPressTimeout
    }
    localStorage.setItem('mobile-gesture-preferences', JSON.stringify(preferences))
  }

  showTrainingModal() {
    this.showTraining = true
  }

  hideTrainingModal() {
    this.showTraining = false
  }

  render() {
    if (!this.enabled) return html``

    return html`
      <div class="gesture-overlay"></div>
      
      ${this.gesturePreview ? this.renderGesturePreview() : ''}
      ${this.feedbackMessage ? this.renderFeedbackToast() : ''}
      ${this.showTraining ? this.renderTrainingModal() : ''}
    `
  }

  private renderGesturePreview() {
    const action = this.gestureActions.find(a => a.id === this.gesturePreview)
    if (!action) return html``

    return html`
      <div class="gesture-preview">
        <div class="gesture-preview-icon">${action.icon}</div>
        <div class="gesture-preview-text">
          <div>${action.action}</div>
          <div class="gesture-preview-priority ${action.priority}">
            ${action.priority}
          </div>
        </div>
      </div>
    `
  }

  private renderFeedbackToast() {
    return html`
      <div class="feedback-toast">
        ${this.feedbackMessage}
      </div>
    `
  }

  private renderTrainingModal() {
    return html`
      <div class="training-overlay" @click=${this.hideTrainingModal}>
        <div class="training-panel" @click=${(e: Event) => e.stopPropagation()}>
          <div class="training-header">
            <div class="training-title">📱 Mobile Gestures</div>
            <div class="training-subtitle">Swipe and tap to control your agents</div>
          </div>
          
          <div class="gesture-list">
            ${this.gestureActions.map(action => html`
              <div class="gesture-item">
                <div class="gesture-item-icon">${action.icon}</div>
                <div class="gesture-item-content">
                  <div class="gesture-item-title">
                    ${this.getGestureDisplayName(action.gesture)}
                  </div>
                  <div class="gesture-item-description">
                    ${action.description}
                  </div>
                  <div class="gesture-item-action">
                    ${action.action}
                  </div>
                </div>
              </div>
            `)}
          </div>
          
          <div class="training-actions">
            <button class="training-button secondary" @click=${this.hideTrainingModal}>
              Got it!
            </button>
            <button class="training-button primary" @click=${() => this.startPracticeMode()}>
              Practice Mode
            </button>
          </div>
        </div>
      </div>
    `
  }

  private getGestureDisplayName(gesture: GestureAction['gesture']): string {
    const names = {
      'swipe-right': 'Swipe Right →',
      'swipe-left': 'Swipe Left ←',
      'swipe-up': 'Swipe Up ↑',
      'swipe-down': 'Swipe Down ↓',
      'long-press': 'Long Press & Hold',
      'double-tap': 'Double Tap'
    }
    return names[gesture] || gesture
  }

  private startPracticeMode() {
    this.trainingMode = true
    this.hideTrainingModal()
    this.showFeedback('Practice mode activated - try gestures now!')
    
    // Auto-disable practice mode after 30 seconds
    setTimeout(() => {
      this.trainingMode = false
      this.showFeedback('Practice mode completed')
    }, 30000)
  }
}
