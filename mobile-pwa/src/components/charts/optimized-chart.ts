/**
 * Performance-Optimized Chart Components
 * High-performance chart components with mobile optimization
 */

import { LitElement, html, css, PropertyValues } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { Chart, ChartConfiguration, ChartData, ChartOptions } from 'chart.js';
import { OptimizedChartConfig } from '../utils/optimized-chart-config';
import { dataVirtualizer, DataPoint } from '../utils/data-virtualizer';

@customElement('optimized-chart')
export class OptimizedChart extends LitElement {
  @property({ type: String }) chartType: 'line' | 'bar' | 'doughnut' | 'scatter' = 'line';
  @property({ type: Object }) data: ChartData = { datasets: [] };
  @property({ type: Object }) options: ChartOptions = {};
  @property({ type: Boolean }) realtime = false;
  @property({ type: Boolean }) virtualized = false;
  @property({ type: Number }) maxDataPoints = 1000;
  @property({ type: Number }) refreshRate = 1000;

  @state() private chart: Chart | null = null;
  @state() private canvas: HTMLCanvasElement | null = null;
  @state() private performanceMetrics = {
    renderTime: 0,
    dataPoints: 0,
    updateCount: 0
  };

  private resizeObserver: ResizeObserver | null = null;
  private updateTimer: number | null = null;
  private rawData: Map<string, DataPoint[]> = new Map();

  static styles = css`
    :host {
      display: block;
      position: relative;
      width: 100%;
      height: 100%;
    }

    .chart-container {
      position: relative;
      width: 100%;
      height: 100%;
      max-height: 400px;
    }

    canvas {
      max-width: 100%;
      max-height: 100%;
    }

    .performance-indicator {
      position: absolute;
      top: 4px;
      right: 4px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 10px;
      display: none;
    }

    :host([debug]) .performance-indicator {
      display: block;
    }

    .loading {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 200px;
      color: #666;
    }

    @media (max-width: 768px) {
      .chart-container {
        max-height: 300px;
      }
    }
  `;

  protected firstUpdated() {
    this.initializeChart();
    this.setupResizeObserver();
    
    if (this.realtime) {
      this.startRealTimeUpdates();
    }
  }

  protected updated(changedProperties: PropertyValues) {
    if (changedProperties.has('data') && this.chart) {
      this.updateChartData();
    }
    
    if (changedProperties.has('chartType') || changedProperties.has('options')) {
      this.reinitializeChart();
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.cleanup();
  }

  private initializeChart() {
    const canvas = this.shadowRoot?.querySelector('canvas') as HTMLCanvasElement;
    if (!canvas) return;

    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const startTime = performance.now();

    // Get optimized configuration
    const config = OptimizedChartConfig.getOptimizedConfig(this.chartType, {
      realtime: this.realtime,
      mobile: true,
      virtualized: this.virtualized,
      dataPoints: this.maxDataPoints
    });

    // Merge with custom options
    const mergedOptions = {
      ...config.options,
      ...this.options,
      onResize: (chart: Chart) => {
        this.handleResize(chart);
      }
    };

    this.chart = new Chart(ctx, {
      type: this.chartType,
      data: this.processedData,
      options: mergedOptions
    });

    const renderTime = performance.now() - startTime;
    this.updatePerformanceMetrics(renderTime);
    
    console.log(`Chart initialized in ${renderTime.toFixed(2)}ms`);
  }

  private reinitializeChart() {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
    this.initializeChart();
  }

  private get processedData(): ChartData {
    if (!this.virtualized) {
      return this.data;
    }

    // Apply data virtualization
    const processedData = { ...this.data };
    processedData.datasets = processedData.datasets.map(dataset => {
      if (Array.isArray(dataset.data) && dataset.data.length > this.maxDataPoints) {
        const virtualizedData = dataVirtualizer.virtualize(
          dataset.data.map((point: any, index: number) => ({
            x: typeof point === 'object' ? point.x : index,
            y: typeof point === 'object' ? point.y : point,
            timestamp: typeof point === 'object' ? point.timestamp : Date.now()
          }))
        );
        
        return {
          ...dataset,
          data: virtualizedData.map(point => ({ x: point.x, y: point.y }))
        };
      }
      return dataset;
    });

    return processedData;
  }

  private updateChartData() {
    if (!this.chart) return;

    const startTime = performance.now();
    
    // Update data with virtualization if needed
    this.chart.data = this.processedData;
    
    // Use optimized update mode
    const updateMode = this.realtime ? 'none' : 'resize';
    this.chart.update(updateMode);

    const renderTime = performance.now() - startTime;
    this.updatePerformanceMetrics(renderTime);
    
    // Adjust virtualization based on performance
    if (this.virtualized && renderTime > 16) {
      dataVirtualizer.adjustForPerformance(renderTime);
    }
  }

  private setupResizeObserver() {
    if ('ResizeObserver' in window) {
      this.resizeObserver = new ResizeObserver(entries => {
        if (this.chart) {
          // Debounce resize events
          clearTimeout(this.updateTimer);
          this.updateTimer = setTimeout(() => {
            this.chart?.resize();
          }, 100);
        }
      });
      
      this.resizeObserver.observe(this);
    }
  }

  private handleResize(chart: Chart) {
    // Adjust data points based on container size
    const container = this.shadowRoot?.querySelector('.chart-container') as HTMLElement;
    if (container && this.virtualized) {
      const containerWidth = container.offsetWidth;
      const optimalPoints = Math.min(containerWidth, this.maxDataPoints);
      
      if (optimalPoints !== this.maxDataPoints) {
        this.maxDataPoints = optimalPoints;
        this.updateChartData();
      }
    }
  }

  private startRealTimeUpdates() {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
    }

    this.updateTimer = setInterval(() => {
      if (this.chart && this.realtime) {
        this.updateRealTimeData();
      }
    }, this.refreshRate);
  }

  private updateRealTimeData() {
    // This method should be overridden by consumers to provide real data
    // For now, it just triggers an update if there's new data
    this.dispatchEvent(new CustomEvent('request-data', {
      detail: { chartId: this.id, timestamp: Date.now() }
    }));
  }

  private updatePerformanceMetrics(renderTime: number) {
    this.performanceMetrics = {
      renderTime: Math.round(renderTime * 100) / 100,
      dataPoints: this.getTotalDataPoints(),
      updateCount: this.performanceMetrics.updateCount + 1
    };

    this.requestUpdate();
  }

  private getTotalDataPoints(): number {
    return this.data.datasets.reduce((total, dataset) => {
      return total + (Array.isArray(dataset.data) ? dataset.data.length : 0);
    }, 0);
  }

  private cleanup() {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }

    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }

    if (this.updateTimer) {
      clearInterval(this.updateTimer);
      this.updateTimer = null;
    }
  }

  // Public API
  public updateData(newData: ChartData) {
    this.data = newData;
  }

  public addDataPoint(datasetIndex: number, dataPoint: { x: number; y: number }) {
    if (this.chart && this.chart.data.datasets[datasetIndex]) {
      const dataset = this.chart.data.datasets[datasetIndex];
      if (Array.isArray(dataset.data)) {
        dataset.data.push(dataPoint);
        
        // Maintain data window for real-time charts
        if (this.realtime && dataset.data.length > this.maxDataPoints) {
          dataset.data.shift();
        }
        
        this.updateChartData();
      }
    }
  }

  public getPerformanceMetrics() {
    return { ...this.performanceMetrics };
  }

  render() {
    return html`
      <div class="chart-container">
        <canvas></canvas>
        <div class="performance-indicator">
          ${this.performanceMetrics.renderTime}ms | ${this.performanceMetrics.dataPoints} pts
        </div>
      </div>
    `;
  }
}