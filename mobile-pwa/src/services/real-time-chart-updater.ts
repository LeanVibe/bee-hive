/**
 * Real-Time Chart Update System
 * Efficient real-time data updates with WebSocket integration
 */

export interface RealTimeDataPoint {
  timestamp: number;
  value: number;
  metric: string;
  source?: string;
}

export class RealTimeChartUpdater {
  private charts: Map<string, Chart> = new Map();
  private dataStreams: Map<string, RealTimeDataPoint[]> = new Map();
  private updateTimers: Map<string, number> = new Map();
  private websocket: WebSocket | null = null;
  private maxDataPoints: number;
  private updateInterval: number;

  constructor(maxDataPoints: number = 1000, updateInterval: number = 1000) {
    this.maxDataPoints = maxDataPoints;
    this.updateInterval = updateInterval;
    this.initializeWebSocket();
  }

  private initializeWebSocket() {
    // Connect to the backend WebSocket for real-time data
    const wsUrl = this.getWebSocketUrl();
    
    try {
      this.websocket = new WebSocket(wsUrl);
      
      this.websocket.onopen = () => {
        console.log('Real-time data WebSocket connected');
      };
      
      this.websocket.onmessage = (event) => {
        this.handleWebSocketMessage(event);
      };
      
      this.websocket.onclose = () => {
        console.log('WebSocket disconnected, attempting reconnect...');
        setTimeout(() => this.initializeWebSocket(), 5000);
      };
      
      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  }

  private getWebSocketUrl(): string {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = location.host;
    return `${protocol}//${host}/dashboard/simple-ws`;
  }

  private handleWebSocketMessage(event: MessageEvent) {
    try {
      const data = JSON.parse(event.data);
      
      if (data.type === 'performance-metrics' || data.type === 'live-data') {
        this.processRealTimeData(data);
      }
    } catch (error) {
      console.error('Failed to process WebSocket message:', error);
    }
  }

  private processRealTimeData(data: any) {
    const timestamp = Date.now();
    
    // Process different types of metrics
    if (data.cpu_usage !== undefined) {
      this.addDataPoint('cpu-usage', {
        timestamp,
        value: data.cpu_usage,
        metric: 'CPU Usage',
        source: data.source
      });
    }
    
    if (data.memory_usage !== undefined) {
      this.addDataPoint('memory-usage', {
        timestamp,
        value: data.memory_usage,
        metric: 'Memory Usage',
        source: data.source
      });
    }
    
    if (data.response_time !== undefined) {
      this.addDataPoint('response-time', {
        timestamp,
        value: data.response_time,
        metric: 'Response Time',
        source: data.source
      });
    }
    
    // Process task metrics
    if (data.tasks) {
      this.processTaskMetrics(data.tasks, timestamp);
    }
    
    // Process agent metrics
    if (data.agents) {
      this.processAgentMetrics(data.agents, timestamp);
    }
  }

  private processTaskMetrics(tasks: any[], timestamp: number) {
    const statusCounts = {
      pending: 0,
      'in-progress': 0,
      completed: 0,
      failed: 0
    };
    
    tasks.forEach(task => {
      statusCounts[task.status as keyof typeof statusCounts]++;
    });
    
    Object.entries(statusCounts).forEach(([status, count]) => {
      this.addDataPoint(`tasks-${status}`, {
        timestamp,
        value: count,
        metric: `Tasks ${status}`,
        source: 'task-service'
      });
    });
  }

  private processAgentMetrics(agents: any[], timestamp: number) {
    const activeCounts = {
      active: agents.filter(a => a.status === 'active').length,
      idle: agents.filter(a => a.status === 'idle').length,
      busy: agents.filter(a => a.status === 'busy').length,
      error: agents.filter(a => a.status === 'error').length
    };
    
    Object.entries(activeCounts).forEach(([status, count]) => {
      this.addDataPoint(`agents-${status}`, {
        timestamp,
        value: count,
        metric: `Agents ${status}`,
        source: 'agent-service'
      });
    });
  }

  private addDataPoint(streamId: string, dataPoint: RealTimeDataPoint) {
    if (!this.dataStreams.has(streamId)) {
      this.dataStreams.set(streamId, []);
    }
    
    const stream = this.dataStreams.get(streamId)!;
    stream.push(dataPoint);
    
    // Maintain data window
    const cutoffTime = Date.now() - (30 * 1000); // 30 seconds
    const filteredStream = stream.filter(point => point.timestamp > cutoffTime);
    
    // Limit data points for performance
    if (filteredStream.length > this.maxDataPoints) {
      filteredStream.splice(0, filteredStream.length - this.maxDataPoints);
    }
    
    this.dataStreams.set(streamId, filteredStream);
    
    // Update associated chart
    this.updateChart(streamId);
  }

  private updateChart(streamId: string) {
    const chart = this.charts.get(streamId);
    if (!chart) return;
    
    const stream = this.dataStreams.get(streamId);
    if (!stream) return;
    
    const startTime = performance.now();
    
    // Update chart data
    if (chart.data.datasets.length > 0) {
      const dataset = chart.data.datasets[0];
      dataset.data = stream.map(point => ({
        x: point.timestamp,
        y: point.value
      }));
      
      // Use efficient update mode for real-time
      chart.update('none');
    }
    
    const updateTime = performance.now() - startTime;
    
    // Monitor performance and adjust if needed
    if (updateTime > 16) { // 60fps threshold
      this.optimizeChartPerformance(streamId, updateTime);
    }
  }

  private optimizeChartPerformance(streamId: string, updateTime: number) {
    console.warn(`Chart ${streamId} update took ${updateTime.toFixed(2)}ms (>16ms)`);
    
    // Reduce data points if performance is poor
    if (updateTime > 33 && this.maxDataPoints > 100) { // 30fps threshold
      this.maxDataPoints = Math.max(100, Math.floor(this.maxDataPoints * 0.8));
      console.log(`Reduced max data points to ${this.maxDataPoints} for better performance`);
    }
  }

  // Public API
  public registerChart(streamId: string, chart: Chart) {
    this.charts.set(streamId, chart);
    
    // Initialize data stream if it doesn't exist
    if (!this.dataStreams.has(streamId)) {
      this.dataStreams.set(streamId, []);
    }
    
    console.log(`Registered chart for stream: ${streamId}`);
  }

  public unregisterChart(streamId: string) {
    this.charts.delete(streamId);
    
    // Clean up timer if it exists
    const timer = this.updateTimers.get(streamId);
    if (timer) {
      clearInterval(timer);
      this.updateTimers.delete(streamId);
    }
    
    console.log(`Unregistered chart for stream: ${streamId}`);
  }

  public getStreamData(streamId: string): RealTimeDataPoint[] {
    return this.dataStreams.get(streamId) || [];
  }

  public subscribeToMetrics(metrics: string[]) {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({
        type: 'subscribe',
        metrics
      }));
    }
  }

  public setUpdateInterval(interval: number) {
    this.updateInterval = interval;
    
    // Update existing timers
    this.updateTimers.forEach((timer, streamId) => {
      clearInterval(timer);
      this.updateTimers.set(streamId, setInterval(() => {
        this.updateChart(streamId);
      }, interval));
    });
  }

  public disconnect() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    
    // Clear all timers
    this.updateTimers.forEach(timer => clearInterval(timer));
    this.updateTimers.clear();
  }

  public getPerformanceStats() {
    return {
      connectedCharts: this.charts.size,
      activeStreams: this.dataStreams.size,
      maxDataPoints: this.maxDataPoints,
      updateInterval: this.updateInterval,
      websocketStatus: this.websocket?.readyState || 'disconnected'
    };
  }
}

// Singleton instance
export const realTimeUpdater = new RealTimeChartUpdater();

// Initialize WebSocket connection when available
if (typeof window !== 'undefined') {
  realTimeUpdater.subscribeToMetrics([
    'cpu-usage',
    'memory-usage',
    'response-time',
    'tasks',
    'agents'
  ]);
}