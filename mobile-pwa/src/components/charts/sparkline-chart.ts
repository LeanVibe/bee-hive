import { LitElement, html, css, svg } from 'lit'
import { customElement, property } from 'lit/decorators.js'

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
    
    @media (max-width: 768px) {
      .sparkline-label,
      .sparkline-value {
        font-size: 0.6875rem;
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
  
  render() {
    if (this.data.length === 0) {
      return html`
        <div class="sparkline-container" style="width: ${this.width}px; height: ${this.height}px;">
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
    
    return html`
      <div class="sparkline-container" style="width: ${this.width}px; height: ${this.height}px;">
        ${this.label ? html`<div class="sparkline-label">${this.label}</div>` : ''}
        
        <svg class="sparkline-svg" viewBox="0 0 ${this.width} ${this.height}">
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
          
          <!-- Data points -->
          ${this.showDots ? points.map((point, index) => svg`
            <circle
              class="sparkline-dot"
              cx="${point.x}"
              cy="${point.y}"
              r="2"
              fill="${this.color}"
            />
          `) : ''}
          
          <!-- Last point (always visible) -->
          ${points.length > 0 ? svg`
            <circle
              cx="${points[points.length - 1].x}"
              cy="${points[points.length - 1].y}"
              r="2.5"
              fill="${this.color}"
              stroke="white"
              stroke-width="1"
            />
          ` : ''}
        </svg>
        
        <div class="sparkline-value">${this.formatValue(this.currentValue)}</div>
      </div>
    `
  }
}