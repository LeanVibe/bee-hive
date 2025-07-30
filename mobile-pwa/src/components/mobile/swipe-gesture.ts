import { LitElement, html, css } from 'lit'
import { customElement, property } from 'lit/decorators.js'

export interface SwipeEvent {
  direction: 'left' | 'right' | 'up' | 'down'
  distance: number
  velocity: number
  duration: number
}

@customElement('swipe-gesture')
export class SwipeGesture extends LitElement {
  @property({ type: Number }) minDistance: number = 50
  @property({ type: Number }) maxTime: number = 300
  @property({ type: Boolean }) horizontal: boolean = true
  @property({ type: Boolean }) vertical: boolean = true
  
  private startX: number = 0
  private startY: number = 0
  private startTime: number = 0
  private element?: HTMLElement
  
  static styles = css`
    :host {
      display: block;
      touch-action: pan-x pan-y;
    }
    
    :host([horizontal-only]) {
      touch-action: pan-y;
    }
    
    :host([vertical-only]) {
      touch-action: pan-x;
    }
    
    .swipe-container {
      width: 100%;
      height: 100%;
      position: relative;
    }
  `
  
  firstUpdated() {
    this.element = this.shadowRoot?.querySelector('.swipe-container') as HTMLElement
    
    if ('ontouchstart' in window) {
      this.setupTouchListeners()
    }
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    this.removeTouchListeners()
  }
  
  private setupTouchListeners() {
    if (!this.element) return
    
    this.element.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: true })
    this.element.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: true })
  }
  
  private removeTouchListeners() {
    if (!this.element) return
    
    this.element.removeEventListener('touchstart', this.handleTouchStart.bind(this))
    this.element.removeEventListener('touchend', this.handleTouchEnd.bind(this))
  }
  
  private handleTouchStart(e: TouchEvent) {
    const touch = e.touches[0]
    this.startX = touch.clientX
    this.startY = touch.clientY
    this.startTime = Date.now()
  }
  
  private handleTouchEnd(e: TouchEvent) {
    const touch = e.changedTouches[0]
    const endX = touch.clientX
    const endY = touch.clientY
    const endTime = Date.now()
    
    const deltaX = endX - this.startX
    const deltaY = endY - this.startY
    const duration = endTime - this.startTime
    
    // Check if the swipe is fast enough
    if (duration > this.maxTime) return
    
    const distanceX = Math.abs(deltaX)
    const distanceY = Math.abs(deltaY)
    
    // Determine primary direction
    let direction: SwipeEvent['direction']
    let distance: number
    
    if (distanceX > distanceY) {
      // Horizontal swipe
      if (!this.horizontal) return
      
      direction = deltaX > 0 ? 'right' : 'left'
      distance = distanceX
    } else {
      // Vertical swipe
      if (!this.vertical) return
      
      direction = deltaY > 0 ? 'down' : 'up'
      distance = distanceY
    }
    
    // Check minimum distance
    if (distance < this.minDistance) return
    
    // Calculate velocity (pixels per millisecond)
    const velocity = distance / duration
    
    // Dispatch swipe event
    const swipeEvent = new CustomEvent('swipe', {
      detail: {
        direction,
        distance,
        velocity,
        duration
      } as SwipeEvent,
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(swipeEvent)
    
    // Dispatch specific direction event
    const directionEvent = new CustomEvent(`swipe-${direction}`, {
      detail: {
        direction,
        distance,
        velocity,
        duration
      } as SwipeEvent,
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(directionEvent)
  }
  
  render() {
    return html`
      <div class="swipe-container">
        <slot></slot>
      </div>
    `
  }
}