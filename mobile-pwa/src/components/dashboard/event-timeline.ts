import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { repeat } from 'lit/directives/repeat.js'
import { classMap } from 'lit/directives/class-map.js'

export interface TimelineEvent {
  id: string
  type: 'task_created' | 'task_updated' | 'task_completed' | 'agent_started' | 'agent_stopped' | 'error' | 'build_success' | 'build_failed' | 'deployment'
  title: string
  description?: string
  agent: string
  timestamp: string
  metadata?: Record<string, any>
  severity: 'info' | 'success' | 'warning' | 'error'
}

@customElement('event-timeline')
export class EventTimeline extends LitElement {
  @property({ type: Array }) declare events: TimelineEvent[]
  @property({ type: Number }) declare maxEvents: number
  @property({ type: Boolean }) declare realtime: boolean
  @property({ type: String }) declare filterAgent: string
  @property({ type: String }) declare filterType: string
  @property({ type: String }) declare filterSeverity: string
  @property({ type: Boolean }) declare compact: boolean
  
  @state() declare private isAutoScrollEnabled: boolean
  @state() declare private isPaused: boolean
  @state() declare private hasNewEvents: boolean
  @state() declare private lastEventCount: number
  
  constructor() {
    super()
    
    // Initialize properties
    this.events = []
    this.maxEvents = 50
    this.realtime = true
    this.filterAgent = ''
    this.filterType = ''
    this.filterSeverity = ''
    this.compact = false
    
    // Initialize state properties
    this.isAutoScrollEnabled = true
    this.isPaused = false
    this.hasNewEvents = false
    this.lastEventCount = 0
  }
  
  private timelineContainer?: HTMLElement
  private eventCountBadge?: HTMLElement
  
  static styles = css`
    :host {
      display: block;
      background: white;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
      overflow: hidden;
      height: 100%;
      display: flex;
      flex-direction: column;
    }
    
    .timeline-header {
      padding: 1rem;
      border-bottom: 1px solid #e5e7eb;
      background: #f9fafb;
      flex-shrink: 0;
    }
    
    .header-content {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.75rem;
    }
    
    .timeline-title {
      font-size: 1rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .timeline-controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .control-button {
      background: none;
      border: 1px solid #d1d5db;
      color: #374151;
      padding: 0.375rem;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    
    .control-button:hover {
      background: #f3f4f6;
      border-color: #9ca3af;
    }
    
    .control-button.active {
      background: #3b82f6;
      border-color: #3b82f6;
      color: white;
    }
    
    .new-events-badge {
      position: absolute;
      top: -6px;
      right: -6px;
      background: #ef4444;
      color: white;
      font-size: 0.6875rem;
      font-weight: 600;
      padding: 0.125rem 0.375rem;
      border-radius: 0.75rem;
      min-width: 18px;
      text-align: center;
      animation: pulse 2s infinite;
    }
    
    .timeline-filters {
      display: flex;
      gap: 0.5rem;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .filter-select {
      padding: 0.375rem 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 0.25rem;
      font-size: 0.875rem;
      background: white;
      min-width: 100px;
    }
    
    .timeline-content {
      flex: 1;
      overflow-y: auto;
      overflow-x: hidden;
      scroll-behavior: smooth;
      -webkit-overflow-scrolling: touch;
    }
    
    .timeline-list {
      padding: 1rem;
      position: relative;
    }
    
    .timeline-list.compact {
      padding: 0.75rem;
    }
    
    .timeline-line {
      position: absolute;
      left: 2rem;
      top: 0;
      bottom: 0;
      width: 2px;
      background: linear-gradient(to bottom, #e5e7eb 0%, #f3f4f6 100%);
    }
    
    .event-item {
      position: relative;
      padding-left: 3rem;
      margin-bottom: 1.5rem;
      animation: slideInLeft 0.3s ease-out;
    }
    
    .event-item.compact {
      margin-bottom: 1rem;
      padding-left: 2.5rem;
    }
    
    .event-item:last-child {
      margin-bottom: 0;
    }
    
    .event-dot {
      position: absolute;
      left: -3rem;
      top: 0.25rem;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      border: 2px solid white;
      box-shadow: 0 0 0 2px currentColor;
      z-index: 1;
    }
    
    .event-dot.info { color: #3b82f6; }
    .event-dot.success { color: #10b981; }
    .event-dot.warning { color: #f59e0b; }
    .event-dot.error { color: #ef4444; animation: pulse 2s infinite; }
    
    .event-card {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.375rem;
      padding: 0.75rem;
      transition: all 0.2s;
    }
    
    .event-card:hover {
      border-color: #d1d5db;
      box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    .event-card.compact {
      padding: 0.5rem;
    }
    
    .event-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 0.5rem;
      gap: 0.5rem;
    }
    
    .event-title {
      font-weight: 600;
      font-size: 0.875rem;
      color: #111827;
      margin: 0;
      flex: 1;
      line-height: 1.25;
    }
    
    .event-timestamp {
      font-size: 0.75rem;
      color: #9ca3af;
      white-space: nowrap;
      flex-shrink: 0;
    }
    
    .event-meta {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }
    
    .agent-badge {
      background: #f3f4f6;
      color: #374151;
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      font-weight: 500;
    }
    
    .event-type-badge {
      padding: 0.125rem 0.375rem;
      border-radius: 0.25rem;
      font-size: 0.6875rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.025em;
    }
    
    .type-task_created,
    .type-task_updated { background: #dbeafe; color: #1e40af; }
    .type-task_completed { background: #d1fae5; color: #065f46; }
    .type-agent_started,
    .type-agent_stopped { background: #f3e8ff; color: #7c3aed; }
    .type-error { background: #fee2e2; color: #dc2626; }
    .type-build_success { background: #d1fae5; color: #065f46; }
    .type-build_failed { background: #fee2e2; color: #dc2626; }
    .type-deployment { background: #fef3c7; color: #92400e; }
    
    .event-description {
      font-size: 0.8125rem;
      color: #6b7280;
      line-height: 1.4;
      margin: 0;
    }
    
    .event-metadata {
      margin-top: 0.5rem;
      padding-top: 0.5rem;
      border-top: 1px solid #f3f4f6;
      font-size: 0.75rem;
      color: #9ca3af;
    }
    
    .metadata-item {
      display: inline-block;
      margin-right: 1rem;
    }
    
    .empty-state {
      padding: 3rem 1rem;
      text-align: center;
      color: #9ca3af;
    }
    
    .empty-icon {
      width: 48px;
      height: 48px;
      margin: 0 auto 1rem;
      opacity: 0.5;
    }
    
    .scroll-to-bottom {
      position: absolute;
      bottom: 1rem;
      right: 1rem;
      background: #3b82f6;
      color: white;
      border: none;
      border-radius: 50%;
      width: 48px;
      height: 48px;
      cursor: pointer;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      transition: all 0.2s;
      z-index: 10;
    }
    
    .scroll-to-bottom:hover {
      background: #2563eb;
      transform: translateY(-2px);
      box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.15);
    }
    
    .scroll-to-bottom.hidden {
      opacity: 0;
      pointer-events: none;
      transform: translateY(20px);
    }
    
    @keyframes slideInLeft {
      from {
        opacity: 0;
        transform: translateX(-20px);
      }
      to {
        opacity: 1;
        transform: translateX(0);
      }
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    @media (max-width: 768px) {
      .timeline-header {
        padding: 0.75rem;
      }
      
      .timeline-filters {
        gap: 0.25rem;
      }
      
      .filter-select {
        min-width: 80px;
        font-size: 0.8125rem;
      }
      
      .timeline-list {
        padding: 0.75rem;
      }
      
      .event-item {
        padding-left: 2.5rem;
        margin-bottom: 1rem;
      }
      
      .event-dot {
        left: -2.5rem;
        width: 10px;
        height: 10px;
      }
      
      .timeline-line {
        left: 1.75rem;
      }
      
      .scroll-to-bottom {
        width: 40px;
        height: 40px;
        bottom: 0.75rem;
        right: 0.75rem;
      }
    }
  `
  
  private get filteredEvents() {
    let filtered = this.events.slice(-this.maxEvents)
    
    if (this.filterAgent) {
      filtered = filtered.filter(event => event.agent === this.filterAgent)
    }
    
    if (this.filterType) {
      filtered = filtered.filter(event => event.type === this.filterType)
    }
    
    if (this.filterSeverity) {
      filtered = filtered.filter(event => event.severity === this.filterSeverity)
    }
    
    // Sort by timestamp (newest first)
    return filtered.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
  }
  
  private get uniqueAgents() {
    const agents = new Set(this.events.map(event => event.agent))
    return Array.from(agents).sort()
  }
  
  private get uniqueTypes() {
    const types = new Set(this.events.map(event => event.type))
    return Array.from(types).sort()
  }
  
  updated(changedProperties: Map<string | number | symbol, unknown>) {
    super.updated(changedProperties)
    
    if (changedProperties.has('events')) {
      const newEventCount = this.events.length
      
      // Check for new events
      if (newEventCount > this.lastEventCount && this.lastEventCount > 0) {
        this.hasNewEvents = true
        
        // Auto-scroll to bottom if enabled
        if (this.isAutoScrollEnabled && !this.isPaused) {
          this.scrollToBottom()
        }
      }
      
      this.lastEventCount = newEventCount
    }
  }
  
  firstUpdated() {
    this.timelineContainer = this.shadowRoot?.querySelector('.timeline-content') as HTMLElement
    this.eventCountBadge = this.shadowRoot?.querySelector('.new-events-badge') as HTMLElement
    
    // Monitor scroll to disable auto-scroll when user scrolls up
    this.timelineContainer?.addEventListener('scroll', this.handleScroll.bind(this))
  }
  
  private handleScroll() {
    if (!this.timelineContainer) return
    
    const { scrollTop, scrollHeight, clientHeight } = this.timelineContainer
    const isNearBottom = scrollTop + clientHeight >= scrollHeight - 100
    
    this.isAutoScrollEnabled = isNearBottom
    
    if (isNearBottom) {
      this.hasNewEvents = false
    }
  }
  
  private scrollToBottom() {
    if (this.timelineContainer) {
      this.timelineContainer.scrollTop = this.timelineContainer.scrollHeight
      this.hasNewEvents = false
    }
  }
  
  private togglePause() {
    this.isPaused = !this.isPaused
    
    const pauseEvent = new CustomEvent('timeline-pause-toggle', {
      detail: { paused: this.isPaused },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(pauseEvent)
  }
  
  private clearEvents() {
    const clearEvent = new CustomEvent('timeline-clear', {
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(clearEvent)
  }
  
  private formatTimestamp(timestamp: string): string {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffSecs = Math.floor(diffMs / 1000)
    const diffMins = Math.floor(diffSecs / 60)
    const diffHours = Math.floor(diffMins / 60)
    
    if (diffSecs < 60) {
      return `${diffSecs}s ago`
    } else if (diffMins < 60) {
      return `${diffMins}m ago`
    } else if (diffHours < 24) {
      return `${diffHours}h ago`
    } else {
      return date.toLocaleDateString()
    }
  }
  
  private handleEventClick(event: TimelineEvent) {
    const clickEvent = new CustomEvent('event-selected', {
      detail: { event },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(clickEvent)
  }
  
  render() {
    const filtered = this.filteredEvents
    
    return html`
      <div class="timeline-header">
        <div class="header-content">
          <h3 class="timeline-title">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Event Timeline
            ${this.realtime ? html`
              <span style="color: #10b981; font-size: 0.75rem; font-weight: 500;">LIVE</span>
            ` : ''}
          </h3>
          
          <div class="timeline-controls">
            <button
              class="control-button ${this.isPaused ? 'active' : ''}"
              @click=${this.togglePause}
              title="${this.isPaused ? 'Resume' : 'Pause'} updates"
            >
              ${this.isPaused ? html`
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h8a2 2 0 002-2V6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2z" />
                </svg>
              ` : html`
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6l4-3-4-3z" />
                </svg>
              `}
            </button>
            
            <button
              class="control-button"
              @click=${this.clearEvents}
              title="Clear all events"
            >
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
            
            <button
              class="control-button"
              @click=${this.scrollToBottom}
              title="Scroll to latest"
              style="position: relative;"
            >
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
              ${this.hasNewEvents ? html`
                <div class="new-events-badge">${Math.min(this.events.length - this.lastEventCount + 1, 9)}+</div>
              ` : ''}
            </button>
          </div>
        </div>
        
        <div class="timeline-filters">
          <select
            class="filter-select"
            .value=${this.filterAgent}
            @change=${(e: Event) => {
              this.filterAgent = (e.target as HTMLSelectElement).value
            }}
          >
            <option value="">All Agents</option>
            ${this.uniqueAgents.map(agent => html`
              <option value=${agent}>${agent}</option>
            `)}
          </select>
          
          <select
            class="filter-select"
            .value=${this.filterType}
            @change=${(e: Event) => {
              this.filterType = (e.target as HTMLSelectElement).value
            }}
          >
            <option value="">All Types</option>
            ${this.uniqueTypes.map(type => html`
              <option value=${type}>${type.replace(/_/g, ' ')}</option>
            `)}
          </select>
          
          <select
            class="filter-select"
            .value=${this.filterSeverity}
            @change=${(e: Event) => {
              this.filterSeverity = (e.target as HTMLSelectElement).value
            }}
          >
            <option value="">All Levels</option>
            <option value="info">Info</option>
            <option value="success">Success</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
          </select>
        </div>
      </div>
      
      <div class="timeline-content">
        ${filtered.length === 0 ? html`
          <div class="empty-state">
            <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p>No events found matching the current filters.</p>
          </div>
        ` : html`
          <div class="timeline-list ${this.compact ? 'compact' : ''}">
            <div class="timeline-line"></div>
            
            ${repeat(
              filtered,
              event => event.id,
              event => html`
                <div class="event-item ${this.compact ? 'compact' : ''}">
                  <div class="event-dot ${event.severity}"></div>
                  
                  <div class="event-card ${this.compact ? 'compact' : ''}" @click=${() => this.handleEventClick(event)}>
                    <div class="event-header">
                      <h4 class="event-title">${event.title}</h4>
                      <span class="event-timestamp">${this.formatTimestamp(event.timestamp)}</span>
                    </div>
                    
                    <div class="event-meta">
                      <span class="agent-badge">${event.agent}</span>
                      <span class="event-type-badge type-${event.type}">
                        ${event.type.replace(/_/g, ' ')}
                      </span>
                    </div>
                    
                    ${event.description ? html`
                      <p class="event-description">${event.description}</p>
                    ` : ''}
                    
                    ${event.metadata && Object.keys(event.metadata).length > 0 ? html`
                      <div class="event-metadata">
                        ${Object.entries(event.metadata).map(([key, value]) => html`
                          <span class="metadata-item">
                            <strong>${key}:</strong> ${value}
                          </span>
                        `)}
                      </div>
                    ` : ''}
                  </div>
                </div>
              `
            )}
          </div>
        `}
      </div>
    `
  }
}