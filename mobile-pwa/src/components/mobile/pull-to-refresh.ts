import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

@customElement('pull-to-refresh')
export class PullToRefresh extends LitElement {
  @property({ type: Boolean }) enabled: boolean = true
  @property({ type: Number }) threshold: number = 80
  @property({ type: Number }) resistance: number = 2.5
  
  @state() private pullDistance: number = 0
  @state() private isRefreshing: boolean = false
  @state() private isTriggered: boolean = false
  
  private startY: number = 0
  private currentY: number = 0
  private isDragging: boolean = false
  private element?: HTMLElement
  private content?: HTMLElement
  
  static styles = css`
    :host {
      display: block;
      position: relative;
      overflow: hidden;
    }
    
    .pull-refresh-container {
      position: relative;
      transition: transform 0.2s ease-out;
      will-change: transform;
    }
    
    .pull-refresh-container.refreshing {
      transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    .pull-indicator {
      position: absolute;
      top: -60px;
      left: 50%;
      transform: translateX(-50%);
      width: 40px;
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: white;
      border-radius: 50%;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      z-index: 10;
      transition: opacity 0.2s ease-out;
    }
    
    .pull-indicator.visible {
      opacity: 1;
    }
    
    .pull-indicator.hidden {
      opacity: 0;
    }
    
    .refresh-icon {
      width: 20px;
      height: 20px;
      color: #3b82f6;
      transition: transform 0.2s ease-out;
    }
    
    .refresh-icon.triggered {
      transform: rotate(180deg);
    }
    
    .refresh-icon.spinning {
      animation: spin 1s linear infinite;
    }
    
    .pull-text {
      position: absolute;
      top: -30px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 0.875rem;
      color: #6b7280;
      white-space: nowrap;
      z-index: 11;
      transition: opacity 0.2s ease-out;
    }
    
    .pull-text.visible {
      opacity: 1;
    }
    
    .pull-text.hidden {
      opacity: 0;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    /* Prevent overscroll bounce on iOS */
    .content {
      overscroll-behavior-y: contain;
      -webkit-overflow-scrolling: touch;
    }
    
    @media (hover: hover) {
      :host {
        /* Disable on desktop/hover devices */
        pointer-events: none;
      }
      
      ::slotted(*) {
        pointer-events: auto;
      }
    }
  `
  
  firstUpdated() {
    this.element = this.shadowRoot?.querySelector('.pull-refresh-container') as HTMLElement
    this.content = this.shadowRoot?.querySelector('.content') as HTMLElement
    
    if (this.enabled && 'ontouchstart' in window) {
      this.setupTouchListeners()
    }
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    this.removeTouchListeners()
  }
  
  private setupTouchListeners() {
    if (!this.element) return
    
    this.element.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false })
    this.element.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false })
    this.element.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false })
  }
  
  private removeTouchListeners() {
    if (!this.element) return
    
    this.element.removeEventListener('touchstart', this.handleTouchStart.bind(this))
    this.element.removeEventListener('touchmove', this.handleTouchMove.bind(this))
    this.element.removeEventListener('touchend', this.handleTouchEnd.bind(this))
  }
  
  private handleTouchStart(e: TouchEvent) {
    if (!this.enabled || this.isRefreshing) return
    
    // Only handle if at top of scrollable content
    const scrollTop = this.getScrollTop()
    if (scrollTop > 0) return
    
    this.startY = e.touches[0].clientY
    this.currentY = this.startY
    this.isDragging = false
  }
  
  private handleTouchMove(e: TouchEvent) {
    if (!this.enabled || this.isRefreshing) return
    
    this.currentY = e.touches[0].clientY
    const deltaY = this.currentY - this.startY
    
    // Only handle downward pulls at the top
    const scrollTop = this.getScrollTop()
    if (deltaY <= 0 || scrollTop > 0) {
      this.resetPull()
      return
    }
    
    // Prevent default scrolling when pulling down
    e.preventDefault()
    this.isDragging = true
    
    // Apply resistance to the pull
    this.pullDistance = Math.min(deltaY / this.resistance, this.threshold * 1.5)
    
    // Check if threshold is reached
    this.isTriggered = this.pullDistance >= this.threshold
    
    // Update UI
    this.updateTransform()
    this.requestUpdate()
  }
  
  private handleTouchEnd() {
    if (!this.enabled || this.isRefreshing || !this.isDragging) return
    
    this.isDragging = false
    
    if (this.isTriggered) {
      this.triggerRefresh()
    } else {
      this.resetPull()
    }
  }
  
  private getScrollTop(): number {
    // Check various scroll containers
    const containers = [
      this.content,
      this.closest('.dashboard-content'),
      this.closest('.view-container'),
      document.documentElement,
      document.body
    ].filter(Boolean) as Element[]
    
    for (const container of containers) {
      if (container.scrollTop > 0) {
        return container.scrollTop
      }
    }
    
    return 0
  }
  
  private updateTransform() {
    if (!this.element) return
    
    this.element.style.transform = `translateY(${this.pullDistance}px)`
  }
  
  private async triggerRefresh() {
    this.isRefreshing = true
    this.pullDistance = this.threshold
    this.updateTransform()
    this.requestUpdate()
    
    // Dispatch refresh event
    const refreshEvent = new CustomEvent('refresh', {
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(refreshEvent)
    
    // Auto-reset after 2 seconds if not manually reset
    setTimeout(() => {
      if (this.isRefreshing) {
        this.finishRefresh()
      }
    }, 2000)
  }
  
  private resetPull() {
    this.pullDistance = 0
    this.isTriggered = false
    this.updateTransform()
    this.requestUpdate()
  }
  
  // Public method to finish refresh
  finishRefresh() {
    this.isRefreshing = false
    this.resetPull()
    
    if (this.element) {
      this.element.classList.add('refreshing')
      
      // Remove class after animation
      setTimeout(() => {
        this.element?.classList.remove('refreshing')
      }, 300)
    }
  }
  
  private get pullText(): string {
    if (this.isRefreshing) {
      return 'Refreshing...'
    } else if (this.isTriggered) {
      return 'Release to refresh'
    } else {
      return 'Pull to refresh'
    }
  }
  
  private get showIndicator(): boolean {
    return this.pullDistance > 10 || this.isRefreshing
  }
  
  render() {
    return html`
      <div class="pull-refresh-container">
        <div class="pull-indicator ${this.showIndicator ? 'visible' : 'hidden'}">
          <svg 
            class="refresh-icon ${this.isTriggered ? 'triggered' : ''} ${this.isRefreshing ? 'spinning' : ''}"
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </div>
        
        <div class="pull-text ${this.showIndicator ? 'visible' : 'hidden'}">
          ${this.pullText}
        </div>
        
        <div class="content">
          <slot></slot>
        </div>
      </div>
    `
  }
}