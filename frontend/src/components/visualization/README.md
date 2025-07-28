# Agent Graph Visualization System

A comprehensive real-time visualization system for multi-agent coordination and performance monitoring in the LeanVibe Agent Hive platform.

## Overview

The Agent Graph Visualization System provides:

- **Real-time agent graph visualization** with D3.js force-directed layouts
- **Session-based color coding** for visual consistency across development sessions
- **Performance heatmaps** for identifying bottlenecks and trends
- **Interactive controls** for filtering, zooming, and navigation
- **WebSocket integration** for live updates
- **Responsive design** optimized for desktop and mobile devices

## Components

### 1. AgentGraphVisualization.vue

The main graph visualization component using D3.js force-directed simulation.

**Features:**
- Force-directed, circular, and grid layout algorithms
- Real-time node and link updates
- Session-based color coding
- Performance-based node sizing
- Interactive zoom, pan, and selection
- Node activity indicators
- Performance rings for enhanced visualization

**Props:**
```typescript
interface Props {
  width?: number          // Graph width (default: 800)
  height?: number         // Graph height (default: 600)
  autoLayout?: boolean    // Enable automatic layout (default: true)
  showControls?: boolean  // Show built-in controls (default: true)
  initialZoom?: number    // Initial zoom level (default: 1)
}
```

**Events:**
- `node-selected`: Emitted when a node is clicked
- `graph-updated`: Emitted when graph data changes

### 2. AgentGraphControls.vue

Advanced control panel for graph filtering and layout management.

**Features:**
- Multi-criteria filtering (session, status, performance, activity)
- Layout algorithm selection
- Visualization mode switching
- Real-time zoom controls
- Performance settings
- Filter presets

**Props:**
```typescript
interface Props {
  agents: AgentInfo[]
  sessions: SessionInfo[]
  zoomLevel: number
  nodeCount: number
  linkCount: number
  performanceStats: {
    fps?: number
    updateCount?: number
    renderTime?: number
  }
}
```

### 3. AgentPerformanceHeatmap.vue

Performance heatmap visualization for temporal analysis.

**Features:**
- Time-series performance visualization
- Multiple metrics (performance, activity, errors, latency, memory)
- Anomaly detection and highlighting
- Interactive tooltips and cell selection
- Performance distribution analytics
- Trend analysis

**Props:**
```typescript
interface Props {
  agents: AgentInfo[]
  sessions: SessionInfo[]
  width?: number
  height?: number
  showTrendLines?: boolean
}
```

### 4. AgentGraphDashboard.vue

Comprehensive dashboard integrating all visualization components.

**Features:**
- Multi-view layout (graph, heatmap, combined)
- Real-time metrics overview
- System health monitoring
- Alert management
- Side panel for detailed information
- WebSocket connection management

## Utilities

### SessionColorManager.ts

Provides consistent color coding across all visualization components.

**Key Features:**
- Session-based color schemes
- Performance-based color scales
- Security risk color mapping
- Gradient generation
- Theme support (light/dark)

**Usage:**
```typescript
import { useSessionColors } from '@/utils/SessionColorManager'

const {
  getSessionColor,
  getAgentColor,
  getPerformanceColor,
  getSecurityRiskColor,
  generateGradient,
  createHeatmapScale
} = useSessionColors()
```

### useAgentGraphRealtime.ts

Composable for real-time graph data management.

**Key Features:**
- Efficient real-time updates
- Performance monitoring
- Adaptive throttling
- Event queue management
- Metrics calculation

## Installation & Setup

### Dependencies

```bash
npm install d3 @types/d3 d3-selection d3-force d3-drag d3-zoom d3-scale d3-time-format
```

### Integration

1. **Add route to router:**
```typescript
{
  path: '/agent-graph',
  name: 'AgentGraph',
  component: () => import('@/views/AgentGraphDashboard.vue'),
  meta: {
    title: 'Agent Graph - LeanVibe Agent Hive',
    description: 'Real-time multi-agent coordination visualization',
  },
}
```

2. **Add navigation link:**
```typescript
{ name: 'Agent Graph', to: '/agent-graph', icon: ShareIcon }
```

3. **Ensure WebSocket connection:**
```typescript
// In your main component or store
eventsStore.connectWebSocket()
```

## Configuration

### Performance Settings

```typescript
const config = {
  maxUpdateQueueSize: 1000,      // Maximum queued updates
  batchUpdateInterval: 50,       // Update interval (ms)
  maxBatchSize: 10,             // Maximum updates per batch
  enableAdaptiveThrottling: true, // Enable performance adaptation
  lowPerformanceThreshold: 30    // FPS threshold for throttling
}
```

### Color Schemes

```typescript
const sessionPalettes = [
  ['#3B82F6', '#1E40AF', '#DBEAFE'], // Blue theme
  ['#10B981', '#047857', '#D1FAE5'], // Green theme
  ['#F59E0B', '#D97706', '#FEF3C7'], // Amber theme
  // ... more themes
]
```

## API Integration

### WebSocket Events

The system listens for these WebSocket event types:

```typescript
interface WebSocketHookMessage {
  type: 'hook_event' | 'security_alert' | 'performance_metric' | 'system_status'
  data: HookEvent | SecurityAlert | HookPerformanceMetrics | any
  timestamp: string
}
```

### Data Structures

```typescript
interface GraphNode extends AgentInfo {
  x?: number
  y?: number
  performance: number
  memoryUsage: number
  isActive: boolean
  connections: string[]
  recentEvents: HookEvent[]
  networkLoad: number
  securityLevel: 'safe' | 'warning' | 'danger'
}

interface GraphLink {
  source: GraphNode
  target: GraphNode
  strength: number
  type: 'communication' | 'collaboration' | 'dependency' | 'data_flow'
  eventCount: number
  latency: number
  status: 'active' | 'idle' | 'error'
}
```

## Customization

### Adding New Visualization Modes

1. **Extend the visualization mode enum:**
```typescript
type VisualizationMode = 'session' | 'performance' | 'security' | 'activity' | 'custom'
```

2. **Add color calculation logic:**
```typescript
const getNodeColor = (node: GraphNode): string => {
  switch (visualizationMode.value) {
    case 'custom':
      return calculateCustomColor(node)
    // ... other cases
  }
}
```

3. **Update legend items:**
```typescript
const legendItems = computed(() => {
  switch (visualizationMode.value) {
    case 'custom':
      return getCustomLegendItems()
    // ... other cases
  }
})
```

### Adding New Layout Algorithms

1. **Register the layout type:**
```typescript
type LayoutType = 'force' | 'circle' | 'grid' | 'hierarchical' | 'custom'
```

2. **Implement the layout function:**
```typescript
const applyCustomLayout = () => {
  nodes.value.forEach((node, i) => {
    // Custom positioning logic
    node.fx = calculateCustomX(node, i)
    node.fy = calculateCustomY(node, i)
  })
}
```

### Custom Metrics

1. **Add metric calculation:**
```typescript
const calculateCustomMetric = (agent: AgentInfo): number => {
  // Custom metric calculation
  return customValue
}
```

2. **Update heatmap data generation:**
```typescript
const generateHeatmapData = () => {
  // Include custom metric in data generation
  if (selectedMetric.value === 'custom') {
    value = calculateCustomMetric(agent)
  }
}
```

## Performance Optimization

### Best Practices

1. **Limit Node Count:**
   - Use filtering to reduce visible nodes
   - Implement pagination for large datasets
   - Consider node clustering for dense graphs

2. **Update Throttling:**
   - Enable adaptive throttling
   - Batch updates during high activity
   - Use requestAnimationFrame for smooth animations

3. **Memory Management:**
   - Clean up D3 event listeners
   - Limit stored event history
   - Use object pooling for frequent updates

### Performance Monitoring

The system includes built-in performance monitoring:

```typescript
const performanceStats = {
  updateCount: 0,
  averageUpdateTime: 0,
  droppedUpdates: 0,
  renderFrames: 0,
  lastFrameTime: 0
}
```

## Testing

### Unit Tests

Test individual components:

```typescript
import { mount } from '@vue/test-utils'
import AgentGraphVisualization from '@/components/visualization/AgentGraphVisualization.vue'

describe('AgentGraphVisualization', () => {
  it('renders graph with nodes and links', () => {
    const wrapper = mount(AgentGraphVisualization, {
      props: {
        width: 800,
        height: 600
      }
    })
    
    expect(wrapper.find('svg')).toBeTruthy()
    expect(wrapper.find('.nodes')).toBeTruthy()
    expect(wrapper.find('.links')).toBeTruthy()
  })
})
```

### Integration Tests

Test component interactions:

```typescript
describe('AgentGraphDashboard Integration', () => {
  it('updates graph when filters change', async () => {
    const wrapper = mount(AgentGraphDashboard)
    
    // Simulate filter change
    await wrapper.find('[data-testid="session-filter"]').setValue('session-123')
    
    // Verify graph updates
    expect(wrapper.emitted('filters-changed')).toBeTruthy()
  })
})
```

## Troubleshooting

### Common Issues

1. **Graph not rendering:**
   - Check D3.js dependencies are installed
   - Verify SVG container dimensions
   - Ensure data is properly formatted

2. **Poor performance:**
   - Enable adaptive throttling
   - Reduce visible node count
   - Check for memory leaks in event listeners

3. **WebSocket connection issues:**
   - Verify WebSocket URL configuration
   - Check network connectivity
   - Monitor connection status indicator

4. **Color inconsistencies:**
   - Ensure SessionColorManager is initialized
   - Check theme configuration
   - Verify color palette definitions

### Debug Tools

Enable debug logging:

```typescript
const DEBUG = process.env.NODE_ENV === 'development'

if (DEBUG) {
  console.log('Graph updated:', {
    nodeCount: nodes.value.length,
    linkCount: links.value.length,
    performance: performanceStats
  })
}
```

## Contributing

### Adding New Features

1. Follow the existing component structure
2. Add TypeScript interfaces for new data types
3. Include responsive design considerations
4. Add comprehensive error handling
5. Write unit tests for new functionality
6. Update documentation

### Code Style

- Use TypeScript for type safety
- Follow Vue 3 Composition API patterns
- Use Tailwind CSS for consistent styling
- Implement proper error boundaries
- Add loading states for async operations

## License

This visualization system is part of the LeanVibe Agent Hive project and follows the project's licensing terms.