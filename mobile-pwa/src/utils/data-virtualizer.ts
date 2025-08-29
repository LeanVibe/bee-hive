/**
 * Intelligent Data Virtualization for Large Datasets
 * Efficiently handles large datasets with smart sampling and aggregation
 */

export interface DataPoint {
  x: number;
  y: number;
  timestamp?: number;
}

export class DataVirtualizer {
  private maxDataPoints: number;
  private samplingAlgorithm: 'lttb' | 'uniform' | 'adaptive';
  private aggregationWindow: number;

  constructor(
    maxDataPoints: number = 1000,
    samplingAlgorithm: 'lttb' | 'uniform' | 'adaptive' = 'lttb',
    aggregationWindow: number = 5000
  ) {
    this.maxDataPoints = maxDataPoints;
    this.samplingAlgorithm = samplingAlgorithm;
    this.aggregationWindow = aggregationWindow;
  }

  // Largest Triangle Three Buckets (LTTB) algorithm for optimal data sampling
  private lttbSample(data: DataPoint[], targetPoints: number): DataPoint[] {
    if (data.length <= targetPoints) {
      return data;
    }

    const sampled: DataPoint[] = [];
    const bucketSize = (data.length - 2) / (targetPoints - 2);
    
    // First point
    sampled.push(data[0]);
    
    let bucketIndex = 0;
    for (let i = 1; i < targetPoints - 1; i++) {
      const bucketStart = Math.floor(bucketIndex * bucketSize) + 1;
      const bucketEnd = Math.floor((bucketIndex + 1) * bucketSize) + 1;
      
      // Calculate area for each point in the bucket
      let maxArea = 0;
      let maxAreaIndex = bucketStart;
      
      const avgNext = this.calculateBucketAverage(data, bucketEnd, Math.min(bucketEnd + bucketSize, data.length));
      
      for (let j = bucketStart; j < bucketEnd; j++) {
        const area = this.calculateTriangleArea(
          sampled[sampled.length - 1],
          data[j],
          avgNext
        );
        
        if (area > maxArea) {
          maxArea = area;
          maxAreaIndex = j;
        }
      }
      
      sampled.push(data[maxAreaIndex]);
      bucketIndex++;
    }
    
    // Last point
    sampled.push(data[data.length - 1]);
    
    return sampled;
  }

  private calculateBucketAverage(data: DataPoint[], start: number, end: number): DataPoint {
    let sumX = 0;
    let sumY = 0;
    const count = end - start;
    
    for (let i = start; i < end; i++) {
      sumX += data[i].x;
      sumY += data[i].y;
    }
    
    return {
      x: sumX / count,
      y: sumY / count
    };
  }

  private calculateTriangleArea(a: DataPoint, b: DataPoint, c: DataPoint): number {
    return Math.abs((a.x - c.x) * (b.y - a.y) - (a.x - b.x) * (c.y - a.y)) / 2;
  }

  // Uniform sampling for predictable data distribution
  private uniformSample(data: DataPoint[], targetPoints: number): DataPoint[] {
    if (data.length <= targetPoints) {
      return data;
    }

    const step = data.length / targetPoints;
    const sampled: DataPoint[] = [];
    
    for (let i = 0; i < targetPoints; i++) {
      const index = Math.floor(i * step);
      sampled.push(data[index]);
    }
    
    return sampled;
  }

  // Adaptive sampling based on data variance
  private adaptiveSample(data: DataPoint[], targetPoints: number): DataPoint[] {
    if (data.length <= targetPoints) {
      return data;
    }

    const variance = this.calculateVariance(data);
    const threshold = variance.mean + variance.std;
    
    // Keep important points (high variance) and sample others
    const importantPoints = data.filter(point => Math.abs(point.y - variance.mean) > threshold);
    const remainingPoints = data.filter(point => Math.abs(point.y - variance.mean) <= threshold);
    
    const availableSlots = targetPoints - importantPoints.length;
    if (availableSlots > 0 && remainingPoints.length > availableSlots) {
      const sampledRemaining = this.uniformSample(remainingPoints, availableSlots);
      return [...importantPoints, ...sampledRemaining].sort((a, b) => a.x - b.x);
    }
    
    return this.uniformSample(data, targetPoints);
  }

  private calculateVariance(data: DataPoint[]): { mean: number; std: number } {
    const values = data.map(point => point.y);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((sum, val) => sum + val, 0) / squaredDiffs.length;
    const std = Math.sqrt(avgSquaredDiff);
    
    return { mean, std };
  }

  // Main virtualization method
  virtualize(data: DataPoint[]): DataPoint[] {
    if (data.length <= this.maxDataPoints) {
      return data;
    }

    const startTime = performance.now();
    let result: DataPoint[];

    switch (this.samplingAlgorithm) {
      case 'lttb':
        result = this.lttbSample(data, this.maxDataPoints);
        break;
      case 'uniform':
        result = this.uniformSample(data, this.maxDataPoints);
        break;
      case 'adaptive':
        result = this.adaptiveSample(data, this.maxDataPoints);
        break;
    }

    const processingTime = performance.now() - startTime;
    
    console.log(`Data virtualized: ${data.length} → ${result.length} points in ${processingTime.toFixed(2)}ms`);
    
    return result;
  }

  // Real-time data aggregation
  aggregateRealTimeData(data: DataPoint[], timeWindow: number = this.aggregationWindow): DataPoint[] {
    if (data.length === 0) return data;

    const now = Date.now();
    const aggregated: DataPoint[] = [];
    const buckets = new Map<number, DataPoint[]>();

    // Group data into time buckets
    data.forEach(point => {
      const timestamp = point.timestamp || point.x;
      if (now - timestamp < timeWindow) {
        const bucketKey = Math.floor(timestamp / 1000) * 1000; // 1-second buckets
        
        if (!buckets.has(bucketKey)) {
          buckets.set(bucketKey, []);
        }
        buckets.get(bucketKey)!.push(point);
      }
    });

    // Aggregate each bucket
    buckets.forEach((bucketData, bucketTime) => {
      if (bucketData.length === 1) {
        aggregated.push(bucketData[0]);
      } else {
        // Calculate average for the bucket
        const avgY = bucketData.reduce((sum, point) => sum + point.y, 0) / bucketData.length;
        aggregated.push({
          x: bucketTime,
          y: avgY,
          timestamp: bucketTime
        });
      }
    });

    return aggregated.sort((a, b) => a.x - b.x);
  }

  // Performance monitoring
  getBenchmarkResults(data: DataPoint[]): {
    originalSize: number;
    virtualizedSize: number;
    compressionRatio: number;
    processingTime: number;
  } {
    const startTime = performance.now();
    const virtualized = this.virtualize(data);
    const processingTime = performance.now() - startTime;

    return {
      originalSize: data.length,
      virtualizedSize: virtualized.length,
      compressionRatio: data.length / virtualized.length,
      processingTime
    };
  }

  // Dynamic adjustment based on performance
  adjustForPerformance(renderTime: number) {
    if (renderTime > 33) { // 30fps threshold
      this.maxDataPoints = Math.max(500, this.maxDataPoints * 0.8);
      console.log(`Reduced max data points to ${this.maxDataPoints} due to performance`);
    } else if (renderTime < 8 && this.maxDataPoints < 2000) { // 120fps threshold
      this.maxDataPoints = Math.min(2000, this.maxDataPoints * 1.1);
      console.log(`Increased max data points to ${this.maxDataPoints} due to good performance`);
    }
  }
}

// Singleton instance with default mobile-optimized settings
export const dataVirtualizer = new DataVirtualizer(
  /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ? 500 : 1000,
  'lttb',
  30000
);