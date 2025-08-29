/**
 * Optimized Chart.js Configuration for Mobile PWA
 * Performance-tuned settings for real-time data visualization
 */

import type { ChartConfiguration, ChartOptions } from 'chart.js';

export class OptimizedChartConfig {
  // Base performance configuration for all charts
  static readonly BASE_PERFORMANCE_CONFIG: Partial<ChartOptions> = {
    responsive: true,
    maintainAspectRatio: false,
    
    // Performance optimizations
    animation: {
      duration: 300, // Reduced from default 1000ms
      easing: 'easeOutQuart',
      resize: {
        duration: 0 // Instant resize for better mobile experience
      }
    },
    
    // Optimize interaction events
    events: ['click', 'touchstart'], // Reduced event listeners
    
    // Optimize legend
    plugins: {
      legend: {
        display: true,
        labels: {
          usePointStyle: true,
          boxWidth: 12,
          padding: 10,
          generateLabels: (chart) => {
            // Optimize legend label generation
            const original = chart.options.plugins?.legend?.labels?.generateLabels;
            if (original && typeof original === 'function') {
              return original(chart).slice(0, 10); // Limit legend items
            }
            return [];
          }
        }
      },
      
      tooltip: {
        enabled: true,
        mode: 'nearest',
        intersect: false,
        animation: {
          duration: 0 // Instant tooltips
        },
        callbacks: {
          // Optimize tooltip rendering
          title: (context) => {
            return context[0]?.label || '';
          },
          label: (context) => {
            return `${context.dataset.label}: ${context.parsed.y}`;
          }
        }
      }
    },
    
    // Optimize scales
    scales: {
      x: {
        type: 'linear',
        display: true,
        ticks: {
          maxTicksLimit: 10, // Limit tick count for performance
          autoSkip: true,
          autoSkipPadding: 10
        },
        grid: {
          display: true,
          drawOnChartArea: true,
          drawTicks: true,
          lineWidth: 1
        }
      },
      y: {
        type: 'linear',
        display: true,
        ticks: {
          maxTicksLimit: 8,
          autoSkip: true
        },
        grid: {
          display: true,
          drawOnChartArea: true,
          drawTicks: true,
          lineWidth: 1
        }
      }
    },
    
    // Layout optimization
    layout: {
      padding: {
        top: 10,
        bottom: 10,
        left: 10,
        right: 10
      }
    }
  };

  // Real-time chart specific configuration
  static readonly REALTIME_CONFIG: Partial<ChartOptions> = {
    ...OptimizedChartConfig.BASE_PERFORMANCE_CONFIG,
    
    animation: {
      duration: 0, // No animation for real-time updates
    },
    
    elements: {
      point: {
        radius: 0, // No points for better performance
        hoverRadius: 4
      },
      line: {
        borderWidth: 2,
        tension: 0.1 // Reduced tension for smoother rendering
      }
    },
    
    scales: {
      x: {
        ...OptimizedChartConfig.BASE_PERFORMANCE_CONFIG.scales?.x,
        type: 'realtime',
        realtime: {
          duration: 30000, // 30 second window
          refresh: 1000,   // 1 second refresh rate
          delay: 1000,     // 1 second delay
          onRefresh: (chart: any) => {
            // Optimized data update callback
            OptimizedChartConfig.updateRealTimeData(chart);
          }
        }
      }
    },
    
    // Optimize for streaming data
    parsing: {
      xAxisKey: 'timestamp',
      yAxisKey: 'value'
    }
  };

  // Mobile-specific configuration
  static readonly MOBILE_CONFIG: Partial<ChartOptions> = {
    ...OptimizedChartConfig.BASE_PERFORMANCE_CONFIG,
    
    // Touch-optimized interactions
    interaction: {
      intersect: false,
      mode: 'index'
    },
    
    // Larger touch targets
    elements: {
      point: {
        radius: 3,
        hoverRadius: 8,
        hitRadius: 15 // Larger touch target
      }
    },
    
    // Mobile-friendly legend
    plugins: {
      ...OptimizedChartConfig.BASE_PERFORMANCE_CONFIG.plugins,
      legend: {
        ...OptimizedChartConfig.BASE_PERFORMANCE_CONFIG.plugins?.legend,
        position: 'bottom',
        labels: {
          ...OptimizedChartConfig.BASE_PERFORMANCE_CONFIG.plugins?.legend?.labels,
          usePointStyle: true,
          boxWidth: 15,
          padding: 15
        }
      }
    }
  };

  // Data virtualization configuration
  static readonly VIRTUALIZED_CONFIG: Partial<ChartOptions> = {
    ...OptimizedChartConfig.BASE_PERFORMANCE_CONFIG,
    
    datasets: {
      line: {
        pointRadius: 0,
        pointHoverRadius: 0,
        pointHitRadius: 0
      }
    },
    
    plugins: {
      ...OptimizedChartConfig.BASE_PERFORMANCE_CONFIG.plugins,
      decimation: {
        enabled: true,
        algorithm: 'lttb', // Largest Triangle Three Buckets algorithm
        samples: 500, // Limit data points for performance
        threshold: 1000 // Apply decimation when data exceeds threshold
      }
    }
  };

  // Get configuration based on chart type and device
  static getOptimizedConfig(
    type: 'line' | 'bar' | 'doughnut' | 'scatter',
    options: {
      realtime?: boolean;
      mobile?: boolean;
      virtualized?: boolean;
      dataPoints?: number;
    } = {}
  ): ChartConfiguration {
    let config = { ...OptimizedChartConfig.BASE_PERFORMANCE_CONFIG };
    
    // Apply real-time optimizations
    if (options.realtime) {
      config = { ...config, ...OptimizedChartConfig.REALTIME_CONFIG };
    }
    
    // Apply mobile optimizations
    if (options.mobile || OptimizedChartConfig.isMobile()) {
      config = { ...config, ...OptimizedChartConfig.MOBILE_CONFIG };
    }
    
    // Apply virtualization for large datasets
    if (options.virtualized || (options.dataPoints && options.dataPoints > 1000)) {
      config = { ...config, ...OptimizedChartConfig.VIRTUALIZED_CONFIG };
    }
    
    return {
      type,
      options: config
    } as ChartConfiguration;
  }

  // Utility methods
  static isMobile(): boolean {
    return /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  }

  static isLowEndDevice(): boolean {
    // Detect low-end devices
    if ('hardwareConcurrency' in navigator) {
      return navigator.hardwareConcurrency <= 2;
    }
    
    // Fallback: assume mobile devices with small screens are low-end
    return window.innerWidth < 768 && OptimizedChartConfig.isMobile();
  }

  // Performance monitoring
  static createPerformanceMonitor() {
    return {
      renderStart: 0,
      renderEnd: 0,
      
      startTiming() {
        this.renderStart = performance.now();
      },
      
      endTiming() {
        this.renderEnd = performance.now();
        const renderTime = this.renderEnd - this.renderStart;
        
        if (renderTime > 16) { // 16ms = 60fps threshold
          console.warn(`Chart render took ${renderTime.toFixed(2)}ms (> 16ms target)`);
        }
        
        return renderTime;
      }
    };
  }

  // Optimized data update for real-time charts
  static updateRealTimeData(chart: any) {
    const now = Date.now();
    const dataAge = 30000; // 30 seconds
    
    // Remove old data points
    chart.data.datasets.forEach((dataset: any) => {
      if (dataset.data && Array.isArray(dataset.data)) {
        dataset.data = dataset.data.filter((point: any) => {
          return (now - point.x) < dataAge;
        });
      }
    });
    
    // Update chart without animation for performance
    chart.update('none');
  }
}

// Export default optimized configurations
export const optimizedLineConfig = OptimizedChartConfig.getOptimizedConfig('line', { realtime: true, mobile: true });
export const optimizedBarConfig = OptimizedChartConfig.getOptimizedConfig('bar', { mobile: true });
export const optimizedDoughnutConfig = OptimizedChartConfig.getOptimizedConfig('doughnut', { mobile: true });