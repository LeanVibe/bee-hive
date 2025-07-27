<template>
  <div class="event-timeline-chart">
    <div class="chart-header mb-4">
      <h3 class="text-lg font-semibold">Event Timeline</h3>
      <p class="text-sm text-gray-600 dark:text-gray-400">Recent events over time</p>
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
  Legend,
  TimeScale
} from 'chart.js'
// import { Line } from 'vue-chartjs' // Not used in canvas implementation

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
)

const chartCanvas = ref<HTMLCanvasElement>()
let chart: ChartJS | null = null

interface EventData {
  timestamp: string
  count: number
  type: string
}

const generateMockData = (): EventData[] => {
  const data: EventData[] = []
  const now = new Date()
  
  for (let i = 0; i < 24; i++) {
    const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000)
    data.push({
      timestamp: timestamp.toISOString(),
      count: Math.floor(Math.random() * 50) + 10,
      type: 'events'
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
            label: 'Events per Hour',
            data: data.map(d => d.count),
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true
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