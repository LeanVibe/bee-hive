export interface EventListener {
  (data?: any): void
}

export class EventEmitter {
  private events: Map<string, Set<EventListener>> = new Map()
  
  on(eventName: string, listener: EventListener): void {
    if (!this.events.has(eventName)) {
      this.events.set(eventName, new Set())
    }
    this.events.get(eventName)!.add(listener)
  }
  
  off(eventName: string, listener: EventListener): void {
    const listeners = this.events.get(eventName)
    if (listeners) {
      listeners.delete(listener)
      if (listeners.size === 0) {
        this.events.delete(eventName)
      }
    }
  }
  
  emit(eventName: string, data?: any): void {
    const listeners = this.events.get(eventName)
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(data)
        } catch (error) {
          console.error(`Error in event listener for ${eventName}:`, error)
        }
      })
    }
  }
  
  removeAllListeners(eventName?: string): void {
    if (eventName) {
      this.events.delete(eventName)
    } else {
      this.events.clear()
    }
  }
  
  listenerCount(eventName: string): number {
    return this.events.get(eventName)?.size || 0
  }
  
  getEventNames(): string[] {
    return Array.from(this.events.keys())
  }
}