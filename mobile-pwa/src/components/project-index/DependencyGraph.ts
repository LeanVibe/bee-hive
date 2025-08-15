/**
 * Dependency Graph Component
 * 
 * Visual dependency visualization using D3.js with interactive
 * node clustering, edge styling, and layout algorithms.
 */

import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ProjectIndexStore } from '../../services/project-index-store';
import type {
  DependencyGraph as GraphData,
  DependencyNode,
  DependencyEdge,
  NodeType,
  DependencyType,
  GraphLayout,
  GraphMetrics,
  DependencyGraphState,
  ComponentLoadingState,
  Subscription
} from '../../types/project-index';

// D3.js type definitions (simplified)
interface D3Node extends DependencyNode {
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
  index?: number;
}

interface D3Edge extends DependencyEdge {
  source: D3Node | string;
  target: D3Node | string;
  index?: number;
}

@customElement('dependency-graph')
export class DependencyGraph extends LitElement {
  @property({ type: String }) declare projectId: string;
  @property({ type: Boolean }) declare compact: boolean;
  @property({ type: Boolean }) declare fullScreen: boolean;
  @property({ type: Object }) declare graphData?: GraphData;

  @state() private declare nodes: D3Node[];
  @state() private declare edges: D3Edge[];
  @state() private declare selectedNodes: Set<string>;
  @state() private declare highlightedPaths: string[][];
  @state() private declare currentLayout: GraphLayout;
  @state() private declare metrics: GraphMetrics;
  @state() private declare loadingState: ComponentLoadingState;
  @state() private declare showControls: boolean;
  @state() private declare showMinimap: boolean;
  @state() private declare filterOptions: {
    nodeTypes: Set<NodeType>;
    dependencyTypes: Set<DependencyType>;
    complexityRange: { min: number; max: number };
  };
  @state() private declare viewMode: 'overview' | 'focused' | 'detailed';

  private store: ProjectIndexStore;
  private subscriptions: Subscription[] = [];
  private svgElement?: SVGElement;
  private simulation?: any; // D3 simulation
  private zoom?: any; // D3 zoom behavior
  private containerWidth = 800;
  private containerHeight = 600;
  private isRendering = false;

  constructor() {
    super();
    
    // Initialize properties
    this.projectId = '';
    this.compact = false;
    this.fullScreen = false;
    this.nodes = [];
    this.edges = [];
    this.selectedNodes = new Set();
    this.highlightedPaths = [];
    this.currentLayout = {
      algorithm: 'force',
      width: 800,
      height: 600,
      scale: 1,
      center: { x: 400, y: 300 }
    };
    this.metrics = {
      total_nodes: 0,
      total_edges: 0,
      connected_components: 0,
      cycles: 0,
      max_depth: 0,
      avg_clustering_coefficient: 0,
      density: 0
    };
    this.loadingState = { isLoading: false };
    this.showControls = true;
    this.showMinimap = false;
    this.filterOptions = {
      nodeTypes: new Set(['module', 'class', 'function'] as NodeType[]),
      dependencyTypes: new Set(['import', 'inheritance', 'call'] as DependencyType[]),
      complexityRange: { min: 0, max: 100 }
    };
    this.viewMode = 'overview';
    
    this.store = ProjectIndexStore.getInstance();
    this.setupStoreSubscriptions();
  }

  connectedCallback() {
    super.connectedCallback();
    this.loadDependencyGraph();
    this.setupResizeObserver();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.cleanupSubscriptions();
    this.cleanupD3();
  }

  static styles = css`
    :host {
      display: block;
      position: relative;
      background: var(--surface-color, #ffffff);
      border-radius: 0.5rem;
      overflow: hidden;
      min-height: 400px;
    }

    .graph-container {
      position: relative;
      width: 100%;
      height: 100%;
      background: linear-gradient(45deg, #f8fafc 25%, transparent 25%),
                  linear-gradient(-45deg, #f8fafc 25%, transparent 25%),
                  linear-gradient(45deg, transparent 75%, #f8fafc 75%),
                  linear-gradient(-45deg, transparent 75%, #f8fafc 75%);
      background-size: 20px 20px;
      background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
    }

    .graph-container.compact {
      min-height: 300px;
    }

    .graph-container.full-screen {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 50;
      background: white;
    }

    .graph-header {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border-color, #e5e7eb);
      padding: 1rem;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .graph-title {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-primary-color, #1f2937);
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .graph-controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .control-button {
      background: white;
      border: 1px solid var(--border-color, #e5e7eb);
      color: var(--text-secondary-color, #6b7280);
      padding: 0.5rem;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .control-button:hover {
      background: var(--hover-color, #f3f4f6);
      border-color: var(--border-hover-color, #d1d5db);
    }

    .control-button.active {
      background: var(--primary-color, #3b82f6);
      border-color: var(--primary-color, #3b82f6);
      color: white;
    }

    .control-select {
      padding: 0.5rem;
      border: 1px solid var(--border-color, #e5e7eb);
      border-radius: 0.375rem;
      font-size: 0.875rem;
      background: white;
    }

    .graph-svg {
      width: 100%;
      height: 100%;
      cursor: grab;
    }

    .graph-svg:active {
      cursor: grabbing;
    }

    /* SVG Styles */
    .node {
      cursor: pointer;
      transition: all 0.2s;
    }

    .node:hover {
      stroke-width: 3px;
    }

    .node.selected {
      stroke: var(--primary-color, #3b82f6);
      stroke-width: 3px;
    }

    .node.highlighted {
      stroke: var(--warning-color, #f59e0b);
      stroke-width: 2px;
    }

    .node.module {
      fill: var(--module-color, #3b82f6);
    }

    .node.class {
      fill: var(--class-color, #10b981);
    }

    .node.function {
      fill: var(--function-color, #f59e0b);
    }

    .node.variable {
      fill: var(--variable-color, #8b5cf6);
    }

    .node.external {
      fill: var(--external-color, #6b7280);
      stroke-dasharray: 4,2;
    }

    .edge {
      stroke: var(--edge-color, #d1d5db);
      stroke-width: 1;
      fill: none;
      pointer-events: none;
    }

    .edge.import {
      stroke: var(--import-color, #3b82f6);
    }

    .edge.inheritance {
      stroke: var(--inheritance-color, #10b981);
      stroke-dasharray: 5,5;
    }

    .edge.composition {
      stroke: var(--composition-color, #f59e0b);
      stroke-width: 2;
    }

    .edge.call {
      stroke: var(--call-color, #8b5cf6);
      stroke-width: 1;
    }

    .edge.reference {
      stroke: var(--reference-color, #6b7280);
      stroke-dasharray: 3,3;
    }

    .edge.highlighted {
      stroke: var(--warning-color, #f59e0b);
      stroke-width: 3;
      animation: pulse-edge 2s infinite;
    }

    .node-label {
      font-size: 10px;
      fill: var(--text-primary-color, #1f2937);
      text-anchor: middle;
      pointer-events: none;
      font-weight: 500;
    }

    .edge-label {
      font-size: 8px;
      fill: var(--text-secondary-color, #6b7280);
      text-anchor: middle;
      pointer-events: none;
    }

    /* Sidebar Panel */
    .graph-sidebar {
      position: absolute;
      top: 80px;
      right: 1rem;
      width: 300px;
      background: white;
      border: 1px solid var(--border-color, #e5e7eb);
      border-radius: 0.5rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      z-index: 20;
      transform: translateX(100%);
      transition: transform 0.3s ease;
    }

    .graph-sidebar.open {
      transform: translateX(0);
    }

    .sidebar-header {
      padding: 1rem;
      border-bottom: 1px solid var(--border-color, #e5e7eb);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .sidebar-title {
      font-weight: 600;
      font-size: 0.875rem;
    }

    .sidebar-content {
      padding: 1rem;
      max-height: 400px;
      overflow-y: auto;
    }

    .filter-group {
      margin-bottom: 1.5rem;
    }

    .filter-group:last-child {
      margin-bottom: 0;
    }

    .filter-label {
      display: block;
      font-weight: 500;
      font-size: 0.875rem;
      color: var(--text-primary-color, #1f2937);
      margin-bottom: 0.5rem;
    }

    .filter-checkboxes {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .filter-checkbox {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.875rem;
    }

    .filter-range {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }

    .filter-range input {
      flex: 1;
      padding: 0.25rem 0.5rem;
      border: 1px solid var(--border-color, #e5e7eb);
      border-radius: 0.25rem;
      font-size: 0.875rem;
    }

    /* Minimap */
    .minimap {
      position: absolute;
      bottom: 1rem;
      right: 1rem;
      width: 150px;
      height: 100px;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--border-color, #e5e7eb);
      border-radius: 0.375rem;
      overflow: hidden;
      z-index: 15;
    }

    .minimap-svg {
      width: 100%;
      height: 100%;
    }

    .minimap-viewport {
      fill: rgba(59, 130, 246, 0.2);
      stroke: var(--primary-color, #3b82f6);
      stroke-width: 1;
    }

    /* Legend */
    .graph-legend {
      position: absolute;
      bottom: 1rem;
      left: 1rem;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--border-color, #e5e7eb);
      border-radius: 0.375rem;
      padding: 1rem;
      z-index: 15;
    }

    .legend-title {
      font-weight: 600;
      font-size: 0.875rem;
      margin-bottom: 0.5rem;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.25rem;
      font-size: 0.75rem;
    }

    .legend-symbol {
      width: 12px;
      height: 12px;
      border-radius: 50%;
    }

    .legend-line {
      width: 20px;
      height: 2px;
    }

    /* Loading State */
    .loading-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(255, 255, 255, 0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 30;
    }

    .loading-spinner {
      width: 32px;
      height: 32px;
      border: 3px solid var(--border-color, #e5e7eb);
      border-top: 3px solid var(--primary-color, #3b82f6);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    /* Metrics Panel */
    .metrics-panel {
      position: absolute;
      top: 80px;
      left: 1rem;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--border-color, #e5e7eb);
      border-radius: 0.375rem;
      padding: 1rem;
      z-index: 15;
      min-width: 200px;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }

    .metric-item {
      text-align: center;
    }

    .metric-value {
      font-size: 1.25rem;
      font-weight: 600;
      color: var(--primary-color, #3b82f6);
    }

    .metric-label {
      font-size: 0.75rem;
      color: var(--text-secondary-color, #6b7280);
      margin-top: 0.25rem;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @keyframes pulse-edge {
      0%, 100% { stroke-opacity: 1; }
      50% { stroke-opacity: 0.5; }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .graph-sidebar {
        width: 280px;
        top: 60px;
      }

      .graph-header {
        padding: 0.75rem;
      }

      .metrics-panel {
        top: 60px;
        padding: 0.75rem;
        min-width: 180px;
      }

      .minimap {
        width: 120px;
        height: 80px;
      }
    }
  `;

  private setupStoreSubscriptions(): void {
    this.subscriptions = [
      this.store.onDependencyGraphLoaded((graphData: GraphData) => {
        this.processGraphData(graphData);
      }),
      
      this.store.onStateChanged((state) => {
        this.loadingState = state.loadingStates['dependency-graph'] || { isLoading: false };
      })
    ];
  }

  private cleanupSubscriptions(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
    this.subscriptions = [];
  }

  private setupResizeObserver(): void {
    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        this.containerWidth = entry.contentRect.width;
        this.containerHeight = entry.contentRect.height;
        this.updateLayout();
      }
    });
    
    resizeObserver.observe(this);
  }

  private async loadDependencyGraph(): Promise<void> {
    if (this.projectId) {
      await this.store.loadDependencyGraph(this.projectId);
    }
  }

  private processGraphData(graphData: GraphData): void {
    this.graphData = graphData;
    this.nodes = graphData.nodes.map(node => ({ ...node }));
    this.edges = graphData.edges.map(edge => ({ ...edge }));
    this.metrics = graphData.metrics;
    this.currentLayout = graphData.layout;
    
    this.filterNodes();
    this.initializeD3();
  }

  private filterNodes(): void {
    // Filter nodes by type and complexity
    this.nodes = this.nodes.filter(node => {
      const typeMatch = this.filterOptions.nodeTypes.has(node.type);
      const complexityMatch = node.complexity >= this.filterOptions.complexityRange.min &&
                            node.complexity <= this.filterOptions.complexityRange.max;
      return typeMatch && complexityMatch;
    });
    
    // Filter edges by type and ensure both source and target nodes exist
    const nodeIds = new Set(this.nodes.map(n => n.id));
    this.edges = this.edges.filter(edge => {
      const typeMatch = this.filterOptions.dependencyTypes.has(edge.type);
      const nodesExist = nodeIds.has(edge.source as string) && nodeIds.has(edge.target as string);
      return typeMatch && nodesExist;
    });
  }

  private async initializeD3(): Promise<void> {
    if (this.isRendering || !this.svgElement) return;
    
    this.isRendering = true;
    
    try {
      // Import D3 dynamically to avoid bundle size issues
      const d3 = await import('d3');
      
      const svg = d3.select(this.svgElement);
      svg.selectAll("*").remove();
      
      const width = this.containerWidth;
      const height = this.containerHeight;
      
      // Create zoom behavior
      this.zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
          container.attr('transform', event.transform);
        });
      
      svg.call(this.zoom);
      
      const container = svg.append('g');
      
      // Create simulation
      this.simulation = d3.forceSimulation(this.nodes)
        .force('link', d3.forceLink(this.edges).id((d: any) => d.id).distance(80))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(20));
      
      // Draw edges
      const edge = container.append('g')
        .selectAll('line')
        .data(this.edges)
        .enter().append('line')
        .attr('class', (d: D3Edge) => `edge ${d.type}`)
        .attr('stroke-width', (d: D3Edge) => Math.sqrt(d.weight) || 1);
      
      // Draw nodes
      const node = container.append('g')
        .selectAll('circle')
        .data(this.nodes)
        .enter().append('circle')
        .attr('class', (d: D3Node) => `node ${d.type}`)
        .attr('r', (d: D3Node) => Math.sqrt(d.size) * 2 + 8)
        .call(d3.drag()
          .on('start', (event, d: D3Node) => {
            if (!event.active) this.simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d: D3Node) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d: D3Node) => {
            if (!event.active) this.simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }))
        .on('click', (event, d: D3Node) => {
          this.handleNodeClick(d, event);
        })
        .on('contextmenu', (event, d: D3Node) => {
          event.preventDefault();
          this.handleNodeContextMenu(d, event);
        });
      
      // Add labels
      const label = container.append('g')
        .selectAll('text')
        .data(this.nodes)
        .enter().append('text')
        .attr('class', 'node-label')
        .text((d: D3Node) => d.label)
        .attr('dy', -15);
      
      // Update positions on simulation tick
      this.simulation.on('tick', () => {
        edge
          .attr('x1', (d: any) => d.source.x)
          .attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x)
          .attr('y2', (d: any) => d.target.y);
        
        node
          .attr('cx', (d: D3Node) => d.x!)
          .attr('cy', (d: D3Node) => d.y!);
        
        label
          .attr('x', (d: D3Node) => d.x!)
          .attr('y', (d: D3Node) => d.y!);
      });
      
    } catch (error) {
      console.error('Failed to initialize D3 visualization:', error);
    } finally {
      this.isRendering = false;
    }
  }

  private handleNodeClick(node: D3Node, event: MouseEvent): void {
    if (event.ctrlKey || event.metaKey) {
      // Multi-select
      if (this.selectedNodes.has(node.id)) {
        this.selectedNodes.delete(node.id);
      } else {
        this.selectedNodes.add(node.id);
      }
    } else {
      // Single select
      this.selectedNodes = new Set([node.id]);
    }
    
    this.updateNodeStyles();
    this.store.selectDependencyNodes(Array.from(this.selectedNodes), false);
    
    // Emit node selection event
    this.dispatchEvent(new CustomEvent('node-selected', {
      detail: { node, selected: this.selectedNodes.has(node.id) },
      bubbles: true,
      composed: true
    }));
  }

  private handleNodeContextMenu(node: D3Node, event: MouseEvent): void {
    this.dispatchEvent(new CustomEvent('node-context-menu', {
      detail: { node, position: { x: event.clientX, y: event.clientY } },
      bubbles: true,
      composed: true
    }));
  }

  private updateNodeStyles(): void {
    if (!this.svgElement) return;
    
    // Update node selections (would need D3 reference)
    this.requestUpdate();
  }

  private updateLayout(): void {
    if (this.simulation) {
      this.simulation
        .force('center', d3.forceCenter(this.containerWidth / 2, this.containerHeight / 2))
        .restart();
    }
  }

  private cleanupD3(): void {
    if (this.simulation) {
      this.simulation.stop();
    }
  }

  private handleLayoutChange(algorithm: string): void {
    this.currentLayout.algorithm = algorithm as any;
    this.initializeD3();
  }

  private handleViewModeChange(mode: string): void {
    this.viewMode = mode as any;
    
    switch (mode) {
      case 'overview':
        this.showControls = true;
        this.showMinimap = false;
        break;
      case 'focused':
        this.showControls = true;
        this.showMinimap = true;
        break;
      case 'detailed':
        this.showControls = true;
        this.showMinimap = true;
        break;
    }
  }

  private toggleFilterType(type: string, category: 'nodeTypes' | 'dependencyTypes'): void {
    const filterSet = this.filterOptions[category] as Set<any>;
    
    if (filterSet.has(type)) {
      filterSet.delete(type);
    } else {
      filterSet.add(type);
    }
    
    this.filterNodes();
    this.initializeD3();
  }

  private handleComplexityRangeChange(min: number, max: number): void {
    this.filterOptions.complexityRange = { min, max };
    this.filterNodes();
    this.initializeD3();
  }

  private renderControls(): any {
    if (!this.showControls) return '';
    
    return html`
      <div class="graph-header">
        <h3 class="graph-title">
          <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
          Dependency Graph
          ${this.compact ? html`<span>(${this.nodes.length} nodes)</span>` : ''}
        </h3>
        
        <div class="graph-controls">
          <select class="control-select" @change=${(e: Event) => this.handleLayoutChange((e.target as HTMLSelectElement).value)}>
            <option value="force" ?selected=${this.currentLayout.algorithm === 'force'}>Force Layout</option>
            <option value="hierarchical" ?selected=${this.currentLayout.algorithm === 'hierarchical'}>Hierarchical</option>
            <option value="circular" ?selected=${this.currentLayout.algorithm === 'circular'}>Circular</option>
            <option value="tree" ?selected=${this.currentLayout.algorithm === 'tree'}>Tree</option>
          </select>
          
          <button class="control-button ${this.viewMode === 'overview' ? 'active' : ''}"
                  @click=${() => this.handleViewModeChange('overview')}
                  title="Overview Mode">
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
            </svg>
          </button>
          
          <button class="control-button ${this.showMinimap ? 'active' : ''}"
                  @click=${() => this.showMinimap = !this.showMinimap}
                  title="Toggle Minimap">
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
              <path d="M9 2v3h6V2h6v19H3V2h6zm2 5v10h2V7h-2z"/>
            </svg>
          </button>
          
          ${this.fullScreen ? html`
            <button class="control-button" @click=${() => this.dispatchEvent(new CustomEvent('exit-fullscreen'))}
                    title="Exit Fullscreen">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                <path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z"/>
              </svg>
            </button>
          ` : html`
            <button class="control-button" @click=${() => this.dispatchEvent(new CustomEvent('enter-fullscreen'))}
                    title="Enter Fullscreen">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
              </svg>
            </button>
          `}
        </div>
      </div>
    `;
  }

  private renderSidebar(): any {
    return html`
      <div class="graph-sidebar ${this.showControls ? 'open' : ''}">
        <div class="sidebar-header">
          <span class="sidebar-title">Filters</span>
          <button class="control-button" @click=${() => this.showControls = false}>
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
          </button>
        </div>
        
        <div class="sidebar-content">
          <div class="filter-group">
            <label class="filter-label">Node Types</label>
            <div class="filter-checkboxes">
              ${['module', 'class', 'function', 'variable', 'external'].map(type => html`
                <label class="filter-checkbox">
                  <input type="checkbox" 
                         .checked=${this.filterOptions.nodeTypes.has(type as NodeType)}
                         @change=${() => this.toggleFilterType(type, 'nodeTypes')}>
                  <span>${type}</span>
                </label>
              `)}
            </div>
          </div>
          
          <div class="filter-group">
            <label class="filter-label">Dependency Types</label>
            <div class="filter-checkboxes">
              ${['import', 'inheritance', 'composition', 'call', 'reference'].map(type => html`
                <label class="filter-checkbox">
                  <input type="checkbox" 
                         .checked=${this.filterOptions.dependencyTypes.has(type as DependencyType)}
                         @change=${() => this.toggleFilterType(type, 'dependencyTypes')}>
                  <span>${type}</span>
                </label>
              `)}
            </div>
          </div>
          
          <div class="filter-group">
            <label class="filter-label">Complexity Range</label>
            <div class="filter-range">
              <input type="number" 
                     .value=${this.filterOptions.complexityRange.min}
                     @input=${(e: Event) => this.handleComplexityRangeChange(
                       parseInt((e.target as HTMLInputElement).value) || 0,
                       this.filterOptions.complexityRange.max
                     )}
                     placeholder="Min">
              <span>-</span>
              <input type="number" 
                     .value=${this.filterOptions.complexityRange.max}
                     @input=${(e: Event) => this.handleComplexityRangeChange(
                       this.filterOptions.complexityRange.min,
                       parseInt((e.target as HTMLInputElement).value) || 100
                     )}
                     placeholder="Max">
            </div>
          </div>
        </div>
      </div>
    `;
  }

  private renderMetrics(): any {
    if (this.compact) return '';
    
    return html`
      <div class="metrics-panel">
        <div class="metrics-grid">
          <div class="metric-item">
            <div class="metric-value">${this.metrics.total_nodes}</div>
            <div class="metric-label">Nodes</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">${this.metrics.total_edges}</div>
            <div class="metric-label">Edges</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">${this.metrics.cycles}</div>
            <div class="metric-label">Cycles</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">${Math.round(this.metrics.density * 100)}%</div>
            <div class="metric-label">Density</div>
          </div>
        </div>
      </div>
    `;
  }

  private renderLegend(): any {
    if (this.compact) return '';
    
    return html`
      <div class="graph-legend">
        <div class="legend-title">Node Types</div>
        <div class="legend-item">
          <div class="legend-symbol module"></div>
          <span>Module</span>
        </div>
        <div class="legend-item">
          <div class="legend-symbol class"></div>
          <span>Class</span>
        </div>
        <div class="legend-item">
          <div class="legend-symbol function"></div>
          <span>Function</span>
        </div>
        <div class="legend-item">
          <div class="legend-symbol variable"></div>
          <span>Variable</span>
        </div>
      </div>
    `;
  }

  private renderMinimap(): any {
    if (!this.showMinimap) return '';
    
    return html`
      <div class="minimap">
        <svg class="minimap-svg">
          <!-- Minimap content would be rendered here -->
          <rect class="minimap-viewport" x="10" y="10" width="80" height="60"></rect>
        </svg>
      </div>
    `;
  }

  render() {
    const containerClasses = `graph-container ${this.compact ? 'compact' : ''} ${this.fullScreen ? 'full-screen' : ''}`;
    
    if (this.loadingState.isLoading) {
      return html`
        <div class=${containerClasses}>
          <div class="loading-overlay">
            <div class="loading-spinner"></div>
            <span style="margin-left: 0.5rem">Loading dependency graph...</span>
          </div>
        </div>
      `;
    }

    return html`
      <div class=${containerClasses}>
        ${this.renderControls()}
        
        <svg class="graph-svg" 
             ${(el: SVGElement) => { this.svgElement = el; this.initializeD3(); }}>
        </svg>
        
        ${this.renderSidebar()}
        ${this.renderMetrics()}
        ${this.renderLegend()}
        ${this.renderMinimap()}
      </div>
    `;
  }
}