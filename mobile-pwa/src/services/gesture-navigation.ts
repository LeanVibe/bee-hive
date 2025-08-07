import { EventEmitter } from '../utils/event-emitter'

export interface GestureEvent {
  type: 'swipe' | 'pinch' | 'tap' | 'long-press' | 'double-tap'
  direction?: 'up' | 'down' | 'left' | 'right'
  deltaX?: number
  deltaY?: number
  scale?: number
  velocity?: number
  center?: { x: number, y: number }
  target?: Element
  timestamp: number
}

export interface SwipeConfig {
  threshold: number // Minimum distance in pixels
  velocity: number // Minimum velocity in pixels/ms
  maxTime: number // Maximum time for swipe in ms
}

export interface PinchConfig {
  threshold: number // Minimum scale change
  minScale: number
  maxScale: number
}

export interface GestureAction {
  gesture: GestureEvent
  action: string
  handler: Function
  enabled: boolean
}

export class GestureNavigationService extends EventEmitter {
  private static instance: GestureNavigationService
  private element: HTMLElement | null = null
  private isEnabled: boolean = false
  private gestureActions: Map<string, GestureAction> = new Map()
  
  // Touch tracking
  private touches: TouchList | null = null
  private startTouches: TouchList | null = null
  private startTime: number = 0
  private isGesturing: boolean = false
  
  // Configuration
  private swipeConfig: SwipeConfig = {
    threshold: 50, // 50px minimum distance
    velocity: 0.3, // 0.3 pixels/ms minimum velocity  
    maxTime: 500 // 500ms maximum duration
  }
  
  private pinchConfig: PinchConfig = {
    threshold: 0.1, // 10% scale change minimum
    minScale: 0.5,
    maxScale: 3.0
  }
  
  // Haptic feedback support
  private supportsHaptics: boolean = 'vibrate' in navigator
  private hapticEnabled: boolean = true
  
  // Long press tracking
  private longPressTimer: number | null = null
  private longPressDelay: number = 500 // 500ms for long press
  
  // Double tap tracking
  private lastTapTime: number = 0
  private doubleTapDelay: number = 300 // 300ms window for double tap
  
  static getInstance(): GestureNavigationService {
    if (!GestureNavigationService.instance) {
      GestureNavigationService.instance = new GestureNavigationService()
    }
    return GestureNavigationService.instance
  }

  constructor() {
    super()
    this.checkHapticSupport()
    this.initializeDefaultGestures()
  }

  private checkHapticSupport(): void {
    this.supportsHaptics = 'vibrate' in navigator
    console.log(`🤲 Haptic feedback ${this.supportsHaptics ? 'supported' : 'not supported'}`)
  }

  initialize(element: HTMLElement): void {
    this.element = element
    this.attachEventListeners()
    this.isEnabled = true
    
    console.log('🤲 Gesture navigation initialized')
    this.emit('initialized')
  }

  private attachEventListeners(): void {
    if (!this.element) return

    // Touch events
    this.element.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false })
    this.element.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false })
    this.element.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false })
    this.element.addEventListener('touchcancel', this.handleTouchCancel.bind(this), { passive: false })

    // Mouse events for desktop testing
    this.element.addEventListener('mousedown', this.handleMouseDown.bind(this))
    this.element.addEventListener('mousemove', this.handleMouseMove.bind(this))
    this.element.addEventListener('mouseup', this.handleMouseUp.bind(this))

    // Pointer events (modern alternative)
    if ('PointerEvent' in window) {
      this.element.addEventListener('pointerdown', this.handlePointerDown.bind(this))
      this.element.addEventListener('pointermove', this.handlePointerMove.bind(this))
      this.element.addEventListener('pointerup', this.handlePointerUp.bind(this))
    }

    // Prevent default behaviors for better gesture handling
    this.element.addEventListener('contextmenu', (e) => e.preventDefault())
    this.element.addEventListener('selectstart', (e) => e.preventDefault())
  }

  private handleTouchStart(event: TouchEvent): void {
    if (!this.isEnabled) return

    this.startTouches = event.touches
    this.touches = event.touches
    this.startTime = Date.now()
    this.isGesturing = true

    // Start long press timer for single touch
    if (event.touches.length === 1) {
      this.startLongPressTimer(event.touches[0])
    }

    // Prevent scrolling on gesture start
    if (event.touches.length > 1) {
      event.preventDefault()
    }

    this.emit('gesture-start', { 
      type: 'touch-start', 
      touches: event.touches.length,
      timestamp: Date.now() 
    })
  }

  private handleTouchMove(event: TouchEvent): void {
    if (!this.isEnabled || !this.isGesturing) return

    this.touches = event.touches
    
    // Cancel long press on movement
    this.cancelLongPressTimer()

    // Handle multi-touch gestures
    if (event.touches.length === 2 && this.startTouches?.length === 2) {
      this.handlePinchGesture(event)
      event.preventDefault() // Prevent zoom
    }

    // Handle single touch movement (potential swipe)
    if (event.touches.length === 1 && this.startTouches?.length === 1) {
      this.handleSwipeProgress(event)
    }
  }

  private handleTouchEnd(event: TouchEvent): void {
    if (!this.isEnabled || !this.isGesturing) return

    const endTime = Date.now()
    const duration = endTime - this.startTime

    this.cancelLongPressTimer()

    // Handle single touch gestures
    if (this.startTouches?.length === 1 && event.changedTouches.length === 1) {
      const startTouch = this.startTouches[0]
      const endTouch = event.changedTouches[0]
      
      const deltaX = endTouch.clientX - startTouch.clientX
      const deltaY = endTouch.clientY - startTouch.clientY
      const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY)
      const velocity = distance / duration

      if (distance >= this.swipeConfig.threshold && velocity >= this.swipeConfig.velocity && duration <= this.swipeConfig.maxTime) {
        this.handleSwipeGesture(deltaX, deltaY, velocity, endTouch)
      } else if (distance < 10) {
        // Small movement, likely a tap
        this.handleTapGesture(endTouch, endTime)
      }
    }

    this.cleanupGesture()
  }

  private handleTouchCancel(event: TouchEvent): void {
    this.cancelLongPressTimer()
    this.cleanupGesture()
  }

  private handleSwipeGesture(deltaX: number, deltaY: number, velocity: number, touch: Touch): void {
    const absX = Math.abs(deltaX)
    const absY = Math.abs(deltaY)
    
    let direction: 'up' | 'down' | 'left' | 'right'
    
    if (absX > absY) {
      direction = deltaX > 0 ? 'right' : 'left'
    } else {
      direction = deltaY > 0 ? 'down' : 'up'
    }

    const gestureEvent: GestureEvent = {
      type: 'swipe',
      direction,
      deltaX,
      deltaY,
      velocity,
      center: { x: touch.clientX, y: touch.clientY },
      target: touch.target as Element,
      timestamp: Date.now()
    }

    console.log(`🤲 Swipe ${direction} detected (velocity: ${velocity.toFixed(2)})`)
    
    this.triggerHapticFeedback('light')
    this.executeGestureAction('swipe', gestureEvent)
    this.emit('swipe', gestureEvent)
  }

  private handlePinchGesture(event: TouchEvent): void {
    if (!this.startTouches || this.startTouches.length !== 2) return

    const touch1 = event.touches[0]
    const touch2 = event.touches[1]
    const startTouch1 = this.startTouches[0]
    const startTouch2 = this.startTouches[1]

    // Calculate current and initial distances
    const currentDistance = Math.sqrt(
      Math.pow(touch2.clientX - touch1.clientX, 2) + 
      Math.pow(touch2.clientY - touch1.clientY, 2)
    )
    
    const initialDistance = Math.sqrt(
      Math.pow(startTouch2.clientX - startTouch1.clientX, 2) + 
      Math.pow(startTouch2.clientY - startTouch1.clientY, 2)
    )

    const scale = currentDistance / initialDistance
    
    // Only trigger if scale change is significant
    if (Math.abs(scale - 1.0) >= this.pinchConfig.threshold) {
      const center = {
        x: (touch1.clientX + touch2.clientX) / 2,
        y: (touch1.clientY + touch2.clientY) / 2
      }

      const gestureEvent: GestureEvent = {
        type: 'pinch',
        scale,
        center,
        timestamp: Date.now()
      }

      console.log(`🤲 Pinch gesture detected (scale: ${scale.toFixed(2)})`)
      
      this.triggerHapticFeedback('medium')
      this.executeGestureAction('pinch', gestureEvent)
      this.emit('pinch', gestureEvent)
    }
  }

  private handleTapGesture(touch: Touch, time: number): void {
    const timeSinceLastTap = time - this.lastTapTime
    
    if (timeSinceLastTap <= this.doubleTapDelay) {
      // Double tap detected
      const gestureEvent: GestureEvent = {
        type: 'double-tap',
        center: { x: touch.clientX, y: touch.clientY },
        target: touch.target as Element,
        timestamp: time
      }

      console.log('🤲 Double tap detected')
      
      this.triggerHapticFeedback('heavy')
      this.executeGestureAction('double-tap', gestureEvent)
      this.emit('double-tap', gestureEvent)
      
      this.lastTapTime = 0 // Reset to prevent triple tap
    } else {
      // Single tap
      const gestureEvent: GestureEvent = {
        type: 'tap',
        center: { x: touch.clientX, y: touch.clientY },
        target: touch.target as Element,
        timestamp: time
      }

      console.log('🤲 Tap detected')
      
      this.triggerHapticFeedback('light')
      this.executeGestureAction('tap', gestureEvent)
      this.emit('tap', gestureEvent)
      
      this.lastTapTime = time
    }
  }

  private startLongPressTimer(touch: Touch): void {
    this.longPressTimer = window.setTimeout(() => {
      const gestureEvent: GestureEvent = {
        type: 'long-press',
        center: { x: touch.clientX, y: touch.clientY },
        target: touch.target as Element,
        timestamp: Date.now()
      }

      console.log('🤲 Long press detected')
      
      this.triggerHapticFeedback('heavy')
      this.executeGestureAction('long-press', gestureEvent)
      this.emit('long-press', gestureEvent)
      
    }, this.longPressDelay)
  }

  private cancelLongPressTimer(): void {
    if (this.longPressTimer) {
      clearTimeout(this.longPressTimer)
      this.longPressTimer = null
    }
  }

  private handleSwipeProgress(event: TouchEvent): void {
    if (!this.startTouches) return
    
    const startTouch = this.startTouches[0]
    const currentTouch = event.touches[0]
    
    const deltaX = currentTouch.clientX - startTouch.clientX
    const deltaY = currentTouch.clientY - startTouch.clientY
    
    this.emit('swipe-progress', { deltaX, deltaY, timestamp: Date.now() })
  }

  private cleanupGesture(): void {
    this.startTouches = null
    this.touches = null
    this.isGesturing = false
    this.startTime = 0
  }

  // Mouse event handlers for desktop testing
  private handleMouseDown(event: MouseEvent): void {
    // Convert mouse event to touch-like event for testing
    if (!this.isEnabled) return
    // Implementation would mirror touch handling
  }

  private handleMouseMove(event: MouseEvent): void {
    // Mouse move handling for desktop testing
  }

  private handleMouseUp(event: MouseEvent): void {
    // Mouse up handling for desktop testing  
  }

  // Pointer event handlers (modern alternative)
  private handlePointerDown(event: PointerEvent): void {
    // Pointer down handling
  }

  private handlePointerMove(event: PointerEvent): void {
    // Pointer move handling
  }

  private handlePointerUp(event: PointerEvent): void {
    // Pointer up handling
  }

  triggerHapticFeedback(intensity: 'light' | 'medium' | 'heavy' = 'light'): void {
    if (!this.hapticEnabled || !this.supportsHaptics) return

    try {
      let duration: number
      switch (intensity) {
        case 'light':
          duration = 10
          break
        case 'medium':
          duration = 25
          break
        case 'heavy':
          duration = 50
          break
      }

      navigator.vibrate(duration)
    } catch (error) {
      console.warn('Haptic feedback failed:', error)
    }
  }

  private executeGestureAction(gestureType: string, gesture: GestureEvent): void {
    const actionKey = this.getGestureActionKey(gestureType, gesture)
    const action = this.gestureActions.get(actionKey)
    
    if (action && action.enabled) {
      try {
        action.handler(gesture)
      } catch (error) {
        console.error('Gesture action execution failed:', error)
        this.emit('action-error', { gesture, error })
      }
    }
  }

  private getGestureActionKey(gestureType: string, gesture: GestureEvent): string {
    let key = gestureType
    if (gesture.direction) {
      key += `-${gesture.direction}`
    }
    return key
  }

  private initializeDefaultGestures(): void {
    // Default navigation gestures
    this.registerGestureAction('swipe-right', (gesture: GestureEvent) => {
      this.emit('navigate', { direction: 'back', gesture })
    })

    this.registerGestureAction('swipe-left', (gesture: GestureEvent) => {
      this.emit('navigate', { direction: 'forward', gesture })
    })

    this.registerGestureAction('swipe-up', (gesture: GestureEvent) => {
      this.emit('refresh', { gesture })
    })

    this.registerGestureAction('swipe-down', (gesture: GestureEvent) => {
      this.emit('menu-toggle', { gesture })
    })

    this.registerGestureAction('pinch', (gesture: GestureEvent) => {
      if (gesture.scale && gesture.scale > 1.1) {
        this.emit('zoom-in', { gesture })
      } else if (gesture.scale && gesture.scale < 0.9) {
        this.emit('zoom-out', { gesture })
      }
    })

    this.registerGestureAction('double-tap', (gesture: GestureEvent) => {
      this.emit('quick-action', { gesture })
    })

    this.registerGestureAction('long-press', (gesture: GestureEvent) => {
      this.emit('context-menu', { gesture })
    })
  }

  // Public API
  registerGestureAction(gestureKey: string, handler: Function, enabled: boolean = true): void {
    this.gestureActions.set(gestureKey, {
      gesture: { type: 'tap', timestamp: Date.now() }, // Placeholder
      action: gestureKey,
      handler,
      enabled
    })
  }

  unregisterGestureAction(gestureKey: string): void {
    this.gestureActions.delete(gestureKey)
  }

  enableGestureAction(gestureKey: string): void {
    const action = this.gestureActions.get(gestureKey)
    if (action) {
      action.enabled = true
    }
  }

  disableGestureAction(gestureKey: string): void {
    const action = this.gestureActions.get(gestureKey)
    if (action) {
      action.enabled = false
    }
  }

  setSwipeConfig(config: Partial<SwipeConfig>): void {
    this.swipeConfig = { ...this.swipeConfig, ...config }
  }

  setPinchConfig(config: Partial<PinchConfig>): void {
    this.pinchConfig = { ...this.pinchConfig, ...config }
  }

  enableHapticFeedback(): void {
    this.hapticEnabled = true
  }

  disableHapticFeedback(): void {
    this.hapticEnabled = false
  }

  enable(): void {
    this.isEnabled = true
    this.emit('enabled')
  }

  disable(): void {
    this.isEnabled = false
    this.cleanupGesture()
    this.emit('disabled')
  }

  getStatus(): {
    enabled: boolean
    hapticSupport: boolean
    hapticEnabled: boolean
    gestureCount: number
    activeGestures: number
  } {
    return {
      enabled: this.isEnabled,
      hapticSupport: this.supportsHaptics,
      hapticEnabled: this.hapticEnabled,
      gestureCount: this.gestureActions.size,
      activeGestures: Array.from(this.gestureActions.values()).filter(a => a.enabled).length
    }
  }

  destroy(): void {
    this.disable()
    this.gestureActions.clear()
    this.removeAllListeners()
    
    if (this.element) {
      // Remove all event listeners
      this.element.removeEventListener('touchstart', this.handleTouchStart.bind(this))
      this.element.removeEventListener('touchmove', this.handleTouchMove.bind(this))
      this.element.removeEventListener('touchend', this.handleTouchEnd.bind(this))
      this.element.removeEventListener('touchcancel', this.handleTouchCancel.bind(this))
    }
  }
}