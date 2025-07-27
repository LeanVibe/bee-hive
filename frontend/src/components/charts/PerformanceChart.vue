<template>
  <div class="performance-chart">
    <div class="chart-header mb-4">
      <h3 class="text-lg font-semibold">Performance Metrics</h3>
      <p class="text-sm text-gray-600 dark:text-gray-400">Response time and throughput</p>
    </div>
    <div class="chart-container" style="height: 300px;">
      <canvas ref="chartCanvas"></canvas>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

const chartCanvas = ref<HTMLCanvasElement>()
let chart: ChartJS | null = null

interface PerformanceData {
  timestamp: string
  responseTime: number
  throughput: number
}

const generateMockData = (): PerformanceData[] => {
  const data: PerformanceData[] = []
  const now = new Date()
  
  for (let i = 0; i < 20; i++) {
    const timestamp = new Date(now.getTime() - i * 30 * 1000)
    data.push({
      timestamp: timestamp.toISOString(),
      responseTime: Math.random() * 200 + 50,
      throughput: Math.random() * 100 + 200
    })
  }
  
  return data.reverse()
}

onMounted(() => {
  if (chartCanvas.value) {
    const data = generateMockData()
    
    chart = new ChartJS(chartCanvas.value, {
      type: 'line',
      data: {
        labels: data.map(d => new Date(d.timestamp).toLocaleTimeString()),
        datasets: [
          {
            label: 'Response Time (ms)',
            data: data.map(d => d.responseTime),
            borderColor: 'rgb(239, 68, 68)',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            tension: 0.1,
            yAxisID: 'y'
          },
          {
            label: 'Throughput (req/s)',
            data: data.map(d => d.throughput),
            borderColor: 'rgb(34, 197, 94)',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            tension: 0.1,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index' as const,
          intersect: false,
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'Time'
            }
          },
          y: {
            type: 'linear' as const,
            display: true,
            position: 'left' as const,
            title: {
              display: true,
              text: 'Response Time (ms)'
            }
          },
          y1: {
            type: 'linear' as const,
            display: true,
            position: 'right' as const,
            title: {
              display: true,
              text: 'Throughput (req/s)'
            },
            grid: {
              drawOnChartArea: false,
            },
          }
        },
        plugins: {
          legend: {
            position: 'top' as const,
          },
          title: {
            display: false
          }
        }
      }
    })
  }
})

onUnmounted(() => {
  if (chart) {
    chart.destroy()
  }
})
</script>