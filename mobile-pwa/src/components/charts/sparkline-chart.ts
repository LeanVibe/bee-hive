import { LitElement, html, css, svg } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

export interface SparklineDataPoint {
  value: number
  label?: string
  timestamp?: string
}

@customElement('sparkline-chart')
export class SparklineChart extends LitElement {
  @property({ type: Array }) declare data: SparklineDataPoint[]
  @property({ type: Number }) declare width: number
  @property({ type: Number }) declare height: number
  @property({ type: String }) declare color: string
  @property({ type: String }) declare fillColor: string
  @property({ type: Boolean }) declare showArea: boolean
  @property({ type: Boolean }) declare showDots: boolean
  @property({ type: Number }) declare strokeWidth: number
  @property({ type: String }) declare label: string
  @property({ type: Boolean }) declare interactive: boolean
  @property({ type: Boolean }) declare realtime: boolean
  @property({ type: String }) declare animationType: 'none' | 'fade' | 'slide' | 'pulse'
  @property({ type: Number }) declare updateInterval: number
  
  @state() private declare hoveredPoint: number | null
  @state() private declare tooltipVisible: boolean
  @state() private declare tooltipPosition: { x: number; y: number }
  @state() private declare isAnimating: boolean
  
  constructor() {
    super()
    
    // Initialize properties
    this.data = []
    this.width = 120
    this.height = 40
    this.color = '#3b82f6'
    this.fillColor = 'rgba(59, 130, 246, 0.1)'
    this.showArea = true
    this.showDots = false
    this.strokeWidth = 2
    this.label = ''
    this.interactive = true
    this.realtime = false
    this.animationType = 'slide'
    this.updateInterval = 1000
    this.hoveredPoint = null
    this.tooltipVisible = false
    this.tooltipPosition = { x: 0, y: 0 }
    this.isAnimating = false
  }
  
  static styles = css`
    :host {
      display: inline-block;
    }
    
    .sparkline-container {
      position: relative;
      width: 100%;
      height: 100%;
    }
    
    .sparkline-svg {
      width: 100%;
      height: 100%;
      overflow: visible;
    }
    
    .sparkline-label {
      position: absolute;
      top: -20px;
      left: 0;
      font-size: 0.75rem;
      font-weight: 500;
      color: #6b7280;
    }
    
    .sparkline-value {
      position: absolute;
      top: -20px;
      right: 0;
      font-size: 0.75rem;
      font-weight: 600;
      color: #111827;
    }
    
    .sparkline-path {
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      transition: all 0.3s ease;
    }
    
    .sparkline-area {
      opacity: 0.6;
      transition: all 0.3s ease;
    }
    
    .sparkline-dot {
      transition: all 0.2s ease;
    }
    
    :host(:hover) .sparkline-path {
      stroke-width: 3;
    }
    
    :host(:hover) .sparkline-area {
      opacity: 0.8;
    }
    
    :host(:hover) .sparkline-dot {
      r: 3;
    }
    
    /* Interactive enhancements */
    .sparkline-interactive {
      cursor: crosshair;
    }
    
    .sparkline-tooltip {
      position: absolute;
      background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
      color: white;
      padding: 0.5rem 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 500;
      white-space: nowrap;
      pointer-events: none;
      z-index: 1000;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.1);
      transform: translate(-50%, -100%);
      margin-top: -8px;
      opacity: 0;
      transition: opacity 0.2s ease, transform 0.2s ease;
    }
    
    .sparkline-tooltip::after {
      content: '';
      position: absolute;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      border: 4px solid transparent;
      border-top-color: #1f2937;
    }
    
    .sparkline-tooltip.visible {
      opacity: 1;
      transform: translate(-50%, -100%) translateY(-2px);
    }
    
    .sparkline-hover-line {
      stroke: rgba(59, 130, 246, 0.3);
      stroke-width: 1;
      stroke-dasharray: 2, 2;
      opacity: 0;
      transition: opacity 0.2s ease;
    }
    
    .sparkline-hover-line.visible {
      opacity: 1;
    }
    
    .sparkline-hover-dot {
      fill: #3b82f6;
      stroke: white;
      stroke-width: 2;
      opacity: 0;
      transition: opacity 0.2s ease, r 0.2s ease;
      r: 4;
    }
    
    .sparkline-hover-dot.visible {
      opacity: 1;
    }
    
    /* Animation styles */
    .sparkline-animate-slide .sparkline-path {
      stroke-dasharray: 1000;
      stroke-dashoffset: 1000;
      animation: drawLine 1s ease-out forwards;
    }
    
    .sparkline-animate-fade {
      animation: fadeIn 0.5s ease-out;
    }
    
    .sparkline-animate-pulse .sparkline-path {
      animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes drawLine {
      to {
        stroke-dashoffset: 0;
      }
    }
    
    @keyframes fadeIn {
      from {
        opacity: 0;
      }
      to {
        opacity: 1;
      }
    }
    
    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.7;
      }
    }
    
    /* Real-time update effects */
    .sparkline-realtime .sparkline-path {
      transition: d 0.3s ease-out;
    }
    
    .sparkline-realtime .sparkline-area {
      transition: d 0.3s ease-out;
    }
    
    .sparkline-new-data-point {
      animation: newPointGlow 0.6s ease-out;
    }
    
    @keyframes newPointGlow {
      0% {
        opacity: 0;
        r: 8;
        fill: #10b981;
      }
      50% {
        opacity: 1;
        r: 6;
        fill: #10b981;
      }
      100% {
        opacity: 1;
        r: 2.5;
        fill: currentColor;
      }
    }
    
    /* Trend indicators */
    .trend-indicator {
      position: absolute;
      top: -16px;
      right: 20px;
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.6875rem;
      font-weight: 600;
    }
    
    .trend-indicator.up {
      color: #10b981;
    }
    
    .trend-indicator.down {
      color: #ef4444;
    }
    
    .trend-indicator.stable {
      color: #6b7280;
    }
    
    .trend-arrow {
      width: 10px;
      height: 10px;
    }
    
    @media (max-width: 768px) {
      .sparkline-label,
      .sparkline-value,
      .trend-indicator {
        font-size: 0.6875rem;
      }
      
      .sparkline-tooltip {
        font-size: 0.6875rem;
        padding: 0.375rem 0.5rem;
      }
      
      .sparkline-interactive {
        cursor: pointer;
      }
    }
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {
      .sparkline-tooltip {
        background: #000000;
        border: 2px solid #ffffff;
      }
      
      .sparkline-hover-line {
        stroke: #ffffff;
        stroke-width: 2;
      }
    }
    
    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {
      .sparkline-path,
      .sparkline-area,
      .sparkline-dot,
      .sparkline-tooltip,
      .sparkline-hover-line,
      .sparkline-hover-dot {
        animation: none;
        transition: none;
      }
    }
  `
  
  private get currentValue(): number {
    return this.data.length > 0 ? this.data[this.data.length - 1].value : 0
  }
  
  private get minValue(): number {
    return this.data.length > 0 ? Math.min(...this.data.map(d => d.value)) : 0
  }
  
  private get maxValue(): number {
    return this.data.length > 0 ? Math.max(...this.data.map(d => d.value)) : 0
  }
  
  private get normalizedData(): { x: number; y: number }[] {
    if (this.data.length === 0) return []
    
    const minValue = this.minValue
    const maxValue = this.maxValue
    const range = maxValue - minValue || 1
    
    return this.data.map((point, index) => ({
      x: (index / Math.max(this.data.length - 1, 1)) * this.width,
      y: this.height - ((point.value - minValue) / range) * this.height
    }))
  }
  
  private get pathData(): string {
    const points = this.normalizedData
    if (points.length === 0) return ''
    
    let path = `M ${points[0].x} ${points[0].y}`
    
    for (let i = 1; i < points.length; i++) {
      const curr = points[i]
      const prev = points[i - 1]
      
      // Create smooth curves using quadratic bezier
      const cpx = prev.x + (curr.x - prev.x) / 2
      path += ` Q ${cpx} ${prev.y} ${curr.x} ${curr.y}`
    }
    
    return path
  }
  
  private get areaPathData(): string {
    const linePath = this.pathData
    if (!linePath) return ''
    
    const points = this.normalizedData
    if (points.length === 0) return ''
    
    const lastPoint = points[points.length - 1]
    const firstPoint = points[0]
    
    return `${linePath} L ${lastPoint.x} ${this.height} L ${firstPoint.x} ${this.height} Z`
  }
  
  private formatValue(value: number): string {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`
    } else if (value % 1 === 0) {
      return value.toString()
    } else {
      return value.toFixed(1)
    }
  }
  
  private get trend(): 'up' | 'down' | 'stable' {
    if (this.data.length < 2) return 'stable'
    
    const current = this.data[this.data.length - 1].value
    const previous = this.data[this.data.length - 2].value
    const change = ((current - previous) / previous) * 100
    
    if (Math.abs(change) < 1) return 'stable'
    return change > 0 ? 'up' : 'down'
  }
  
  private get trendPercentage(): string {
    if (this.data.length < 2) return '0%'
    
    const current = this.data[this.data.length - 1].value
    const previous = this.data[this.data.length - 2].value
    const change = ((current - previous) / previous) * 100
    
    return `${change > 0 ? '+' : ''}${change.toFixed(1)}%`
  }
  
  // Interactive methods
  private handleMouseMove(event: MouseEvent) {
    if (!this.interactive || this.data.length === 0) return
    
    const rect = (event.currentTarget as SVGElement).getBoundingClientRect()
    const x = event.clientX - rect.left
    const dataIndex = Math.round((x / this.width) * (this.data.length - 1))
    
    if (dataIndex >= 0 && dataIndex < this.data.length && dataIndex !== this.hoveredPoint) {
      this.hoveredPoint = dataIndex
      this.tooltipPosition = { x: event.clientX, y: event.clientY }
      this.tooltipVisible = true
    }
  }
  
  private handleMouseLeave() {
    this.hoveredPoint = null
    this.tooltipVisible = false
  }
  
  private handleClick(event: MouseEvent) {
    if (!this.interactive || this.hoveredPoint === null) return
    
    const dataPoint = this.data[this.hoveredPoint]
    
    const clickEvent = new CustomEvent('sparkline-point-click', {
      detail: {
        point: dataPoint,
        index: this.hoveredPoint,
        value: dataPoint.value,
        timestamp: dataPoint.timestamp
      },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(clickEvent)
  }
  
  private getTooltipContent(): string {
    if (this.hoveredPoint === null) return ''
    
    const point = this.data[this.hoveredPoint]
    const formattedValue = this.formatValue(point.value)
    
    if (point.timestamp) {
      const date = new Date(point.timestamp)
      const timeString = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      return `${formattedValue} at ${timeString}`
    }
    
    return point.label ? `${point.label}: ${formattedValue}` : formattedValue
  }
  
  private getTrendIcon() {
    const trend = this.trend
    switch (trend) {
      case 'up':
        return html`
          <svg class="trend-arrow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 17l9.2-9.2M17 17V7H7"/>
          </svg>
        `
      case 'down':
        return html`
          <svg class="trend-arrow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 7l-9.2 9.2M7 7v10h10"/>
          </svg>
        `
      case 'stable':
      default:
        return html`
          <svg class="trend-arrow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14"/>
          </svg>
        `
    }
  }
  
  private getAnimationClass(): string {
    if (this.animationType === 'none') return ''
    return `sparkline-animate-${this.animationType}`
  }
  
  private getContainerClass(): string {
    const classes = ['sparkline-container']
    
    if (this.interactive) classes.push('sparkline-interactive')
    if (this.realtime) classes.push('sparkline-realtime')
    if (this.animationType !== 'none') classes.push(this.getAnimationClass())
    
    return classes.join(' ')
  }
  
  // Lifecycle methods for real-time updates
  connectedCallback() {
    super.connectedCallback()
    
    if (this.realtime && this.updateInterval > 0) {
      this.startRealtimeUpdates()
    }
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    this.stopRealtimeUpdates()
  }
  
  private realtimeTimer: number | null = null
  
  private startRealtimeUpdates() {
    this.stopRealtimeUpdates()
    
    this.realtimeTimer = window.setInterval(() => {
      this.dispatchUpdateRequest()
    }, this.updateInterval)
  }
  
  private stopRealtimeUpdates() {
    if (this.realtimeTimer) {
      clearInterval(this.realtimeTimer)
      this.realtimeTimer = null
    }
  }
  
  private dispatchUpdateRequest() {
    const updateEvent = new CustomEvent('sparkline-update-request', {
      detail: {
        chartId: this.id || 'sparkline',
        lastDataPoint: this.data[this.data.length - 1],
        dataLength: this.data.length
      },
      bubbles: true,
      composed: true
    })
    
    this.dispatchEvent(updateEvent)
  }
  
  render() {
    if (this.data.length === 0) {
      return html`
        <div class="${this.getContainerClass()}" style="width: ${this.width}px; height: ${this.height}px;">
          ${this.label ? html`<div class="sparkline-label">${this.label}</div>` : ''}
          <svg class="sparkline-svg" viewBox="0 0 ${this.width} ${this.height}">
            <line 
              x1="0" 
              y1="${this.height / 2}" 
              x2="${this.width}" 
              y2="${this.height / 2}" 
              stroke="#e5e7eb" 
              stroke-width="1"
              stroke-dasharray="2,2"
            />
          </svg>
          <div class="sparkline-value">--</div>
        </div>
      `
    }
    
    const points = this.normalizedData
    const hoveredPoint = this.hoveredPoint !== null ? points[this.hoveredPoint] : null
    
    return html`
      <div class="${this.getContainerClass()}" style="width: ${this.width}px; height: ${this.height}px;">
        ${this.label ? html`<div class="sparkline-label">${this.label}</div>` : ''}
        
        <!-- Trend indicator -->
        ${this.data.length >= 2 ? html`
          <div class="trend-indicator ${this.trend}">
            ${this.getTrendIcon()}
            ${this.trendPercentage}
          </div>
        ` : ''}
        
        <svg 
          class="sparkline-svg" 
          viewBox="0 0 ${this.width} ${this.height}"
          @mousemove=${this.handleMouseMove}
          @mouseleave=${this.handleMouseLeave}
          @click=${this.handleClick}
        >
          <!-- Area fill -->
          ${this.showArea ? svg`
            <path
              class="sparkline-area"
              d="${this.areaPathData}"
              fill="${this.fillColor}"
            />
          ` : ''}
          
          <!-- Main line -->
          <path
            class="sparkline-path"
            d="${this.pathData}"
            stroke="${this.color}"
            stroke-width="${this.strokeWidth}"
          />
          
          <!-- Hover line -->
          ${this.interactive && hoveredPoint ? svg`
            <line
              class="sparkline-hover-line visible"
              x1="${hoveredPoint.x}"
              y1="0"
              x2="${hoveredPoint.x}"
              y2="${this.height}"
            />
          ` : ''}
          
          <!-- Data points -->
          ${this.showDots ? points.map((point, index) => svg`
            <circle
              class="sparkline-dot ${index === this.hoveredPoint ? 'visible' : ''}"
              cx="${point.x}"
              cy="${point.y}"
              r="2"
              fill="${this.color}"
            />
          `) : ''}
          
          <!-- Hovered point -->
          ${this.interactive && hoveredPoint ? svg`
            <circle
              class="sparkline-hover-dot visible"
              cx="${hoveredPoint.x}"
              cy="${hoveredPoint.y}"
            />
          ` : ''}
          
          <!-- Last point (always visible with potential animation) -->
          ${points.length > 0 ? svg`
            <circle
              class="${this.realtime ? 'sparkline-new-data-point' : ''}"
              cx="${points[points.length - 1].x}"
              cy="${points[points.length - 1].y}"
              r="2.5"
              fill="${this.color}"
              stroke="white"
              stroke-width="1"
            />
          ` : ''}
        </svg>
        
        <!-- Interactive tooltip -->
        ${this.interactive && this.tooltipVisible ? html`
          <div 
            class="sparkline-tooltip visible"
            style="left: ${this.tooltipPosition.x}px; top: ${this.tooltipPosition.y}px;"
          >
            ${this.getTooltipContent()}
          </div>
        ` : ''}
        
        <div class="sparkline-value">${this.formatValue(this.currentValue)}</div>
      </div>
    `
  }
}