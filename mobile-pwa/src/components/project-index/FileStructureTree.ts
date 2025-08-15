/**
 * File Structure Tree Component
 * 
 * Interactive file explorer with virtual scrolling, search, filtering,
 * and context menu operations for large project file lists.
 */

import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { classMap } from 'lit/directives/class-map.js';
import { ProjectIndexStore } from '../../services/project-index-store';
import type {
  ProjectFile,
  FileTreeState,
  FileType,
  ComponentLoadingState,
  ProjectIndexState,
  Subscription
} from '../../types/project-index';

interface TreeNode {
  id: string;
  name: string;
  path: string;
  type: FileType;
  size: number;
  language: string;
  extension: string;
  analyzed: boolean;
  children?: TreeNode[];
  parent?: TreeNode;
  level: number;
  isExpanded: boolean;
  isSelected: boolean;
  lastModified: string;
}

@customElement('file-structure-tree')
export class FileStructureTree extends LitElement {
  @property({ type: String }) declare projectId: string;
  @property({ type: Boolean }) declare compact: boolean;
  @property({ type: Boolean }) declare multiSelect: boolean;
  @property({ type: Array }) declare selectedFiles: string[];

  @state() private declare treeNodes: TreeNode[];
  @state() private declare flattenedNodes: TreeNode[];
  @state() private declare filteredNodes: TreeNode[];
  @state() private declare searchTerm: string;
  @state() private declare sortBy: 'name' | 'size' | 'type' | 'modified';
  @state() private declare sortOrder: 'asc' | 'desc';
  @state() private declare expandedNodes: Set<string>;
  @state() private declare selectedNodes: Set<string>;
  @state() private declare loadingState: ComponentLoadingState;
  @state() private declare showContextMenu: boolean;
  @state() private declare contextMenuPosition: { x: number; y: number };
  @state() private declare contextMenuTarget?: TreeNode;
  @state() private declare virtualScrollOffset: number;
  @state() private declare visibleStartIndex: number;
  @state() private declare visibleEndIndex: number;

  private store: ProjectIndexStore;
  private subscriptions: Subscription[] = [];
  private virtualScrollContainer?: HTMLElement;
  private itemHeight = 32;
  private visibleItemCount = 20;
  private scrollTimeout?: number;

  constructor() {
    super();
    
    // Initialize properties
    this.projectId = '';
    this.compact = false;
    this.multiSelect = false;
    this.selectedFiles = [];
    this.treeNodes = [];
    this.flattenedNodes = [];
    this.filteredNodes = [];
    this.searchTerm = '';
    this.sortBy = 'name';
    this.sortOrder = 'asc';
    this.expandedNodes = new Set();
    this.selectedNodes = new Set();
    this.loadingState = { isLoading: false };
    this.showContextMenu = false;
    this.contextMenuPosition = { x: 0, y: 0 };
    this.virtualScrollOffset = 0;
    this.visibleStartIndex = 0;
    this.visibleEndIndex = 0;
    
    this.store = ProjectIndexStore.getInstance();
    this.setupStoreSubscriptions();
  }

  connectedCallback() {
    super.connectedCallback();
    this.loadFiles();
    this.setupEventListeners();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.cleanupSubscriptions();
    this.cleanupEventListeners();
  }

  static styles = css`
    :host {
      display: block;
      height: 100%;
      background: var(--surface-color, #ffffff);
      border-radius: 0.5rem;
      overflow: hidden;
    }

    .tree-container {
      height: 100%;
      display: flex;
      flex-direction: column;
    }

    .tree-header {
      padding: 1rem;
      border-bottom: 1px solid var(--border-color, #e5e7eb);
      background: var(--surface-secondary-color, #f8fafc);
    }

    .tree-header.compact {
      padding: 0.75rem;
    }

    .search-box {
      width: 100%;
      padding: 0.5rem 0.75rem;
      border: 1px solid var(--border-color, #e5e7eb);
      border-radius: 0.375rem;
      font-size: 0.875rem;
      background: white;
      transition: border-color 0.2s;
    }

    .search-box:focus {
      outline: none;
      border-color: var(--primary-color, #3b82f6);
      box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
    }

    .tree-controls {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 0.75rem;
      gap: 0.5rem;
    }

    .tree-controls.compact {
      margin-top: 0.5rem;
    }

    .control-group {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .sort-select {
      padding: 0.25rem 0.5rem;
      border: 1px solid var(--border-color, #e5e7eb);
      border-radius: 0.25rem;
      font-size: 0.75rem;
      background: white;
    }

    .control-button {
      background: none;
      border: 1px solid var(--border-color, #e5e7eb);
      color: var(--text-secondary-color, #6b7280);
      padding: 0.25rem;
      border-radius: 0.25rem;
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

    .tree-stats {
      font-size: 0.75rem;
      color: var(--text-secondary-color, #6b7280);
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .tree-content {
      flex: 1;
      overflow: hidden;
      position: relative;
    }

    .virtual-scroll-container {
      height: 100%;
      overflow-y: auto;
      overflow-x: hidden;
    }

    .virtual-scroll-content {
      position: relative;
    }

    .tree-item {
      display: flex;
      align-items: center;
      padding: 0.375rem 0.75rem;
      cursor: pointer;
      transition: all 0.15s;
      border-bottom: 1px solid transparent;
      position: absolute;
      left: 0;
      right: 0;
      height: 32px;
      box-sizing: border-box;
    }

    .tree-item:hover {
      background: var(--hover-color, #f3f4f6);
    }

    .tree-item.selected {
      background: var(--selection-color, #dbeafe);
      border-bottom-color: var(--primary-color, #3b82f6);
    }

    .tree-item.compact {
      padding: 0.25rem 0.5rem;
      height: 28px;
    }

    .tree-item-content {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      width: 100%;
      min-width: 0;
    }

    .tree-indent {
      flex-shrink: 0;
    }

    .tree-expand-toggle {
      width: 16px;
      height: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: none;
      background: none;
      cursor: pointer;
      color: var(--text-secondary-color, #6b7280);
      transition: transform 0.2s;
      flex-shrink: 0;
    }

    .tree-expand-toggle.expanded {
      transform: rotate(90deg);
    }

    .tree-expand-toggle:disabled {
      opacity: 0;
      cursor: default;
    }

    .file-icon {
      width: 16px;
      height: 16px;
      flex-shrink: 0;
      color: var(--text-secondary-color, #6b7280);
    }

    .file-icon.directory {
      color: var(--directory-color, #3b82f6);
    }

    .file-icon.javascript {
      color: #f7df1e;
    }

    .file-icon.typescript {
      color: #3178c6;
    }

    .file-icon.python {
      color: #3776ab;
    }

    .file-icon.java {
      color: #ed8b00;
    }

    .file-icon.rust {
      color: #ce422b;
    }

    .file-icon.go {
      color: #00add8;
    }

    .file-name {
      flex: 1;
      font-size: 0.875rem;
      font-weight: 500;
      color: var(--text-primary-color, #1f2937);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .file-name.compact {
      font-size: 0.8125rem;
    }

    .file-meta {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.75rem;
      color: var(--text-secondary-color, #6b7280);
      flex-shrink: 0;
    }

    .file-size {
      min-width: 4rem;
      text-align: right;
    }

    .file-status {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .file-status.analyzed {
      background: var(--success-color, #10b981);
    }

    .file-status.pending {
      background: var(--warning-color, #f59e0b);
    }

    .file-status.error {
      background: var(--error-color, #ef4444);
    }

    /* Context Menu */
    .context-menu {
      position: fixed;
      background: white;
      border: 1px solid var(--border-color, #e5e7eb);
      border-radius: 0.5rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      z-index: 50;
      min-width: 160px;
      padding: 0.25rem;
    }

    .context-menu-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 0.75rem;
      cursor: pointer;
      border-radius: 0.25rem;
      font-size: 0.875rem;
      color: var(--text-primary-color, #1f2937);
      transition: background 0.15s;
    }

    .context-menu-item:hover {
      background: var(--hover-color, #f3f4f6);
    }

    .context-menu-separator {
      height: 1px;
      background: var(--border-color, #e5e7eb);
      margin: 0.25rem 0;
    }

    /* Loading State */
    .loading-container {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 200px;
      color: var(--text-secondary-color, #6b7280);
    }

    .loading-spinner {
      width: 24px;
      height: 24px;
      border: 2px solid var(--border-color, #e5e7eb);
      border-top: 2px solid var(--primary-color, #3b82f6);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    /* Empty State */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 200px;
      text-align: center;
      color: var(--text-secondary-color, #6b7280);
    }

    .empty-icon {
      width: 48px;
      height: 48px;
      opacity: 0.5;
      margin-bottom: 1rem;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .tree-header {
        padding: 0.75rem;
      }

      .tree-controls {
        flex-direction: column;
        align-items: stretch;
        gap: 0.5rem;
      }

      .file-meta {
        display: none;
      }

      .tree-item {
        padding: 0.5rem;
      }
    }
  `;

  private setupStoreSubscriptions(): void {
    this.subscriptions = [
      this.store.onFilesLoaded((result: any) => {
        this.buildTreeFromFiles(result.files);
      }),
      
      this.store.onStateChanged((state: ProjectIndexState) => {
        // Update loading state for files
        this.loadingState = state.loadingStates['project-files'] || { isLoading: false };
      })
    ];
  }

  private cleanupSubscriptions(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
    this.subscriptions = [];
  }

  private setupEventListeners(): void {
    document.addEventListener('click', this.handleDocumentClick.bind(this));
    document.addEventListener('keydown', this.handleKeyDown.bind(this));
  }

  private cleanupEventListeners(): void {
    document.removeEventListener('click', this.handleDocumentClick.bind(this));
    document.removeEventListener('keydown', this.handleKeyDown.bind(this));
  }

  private async loadFiles(): Promise<void> {
    if (this.projectId) {
      await this.store.loadProjectFiles(this.projectId);
    }
  }

  private buildTreeFromFiles(files: ProjectFile[]): void {
    const nodeMap = new Map<string, TreeNode>();
    const rootNodes: TreeNode[] = [];

    // Create nodes for all files and directories
    files.forEach(file => {
      const pathParts = file.path.split('/');
      let currentPath = '';
      
      pathParts.forEach((part, index) => {
        const isLast = index === pathParts.length - 1;
        currentPath = currentPath ? `${currentPath}/${part}` : part;
        
        if (!nodeMap.has(currentPath)) {
          const node: TreeNode = {
            id: currentPath,
            name: part,
            path: currentPath,
            type: isLast ? file.type : 'directory',
            size: isLast ? file.size : 0,
            language: isLast ? file.language : '',
            extension: isLast ? file.extension : '',
            analyzed: isLast ? file.analyzed : false,
            children: [],
            level: index,
            isExpanded: this.expandedNodes.has(currentPath),
            isSelected: this.selectedNodes.has(currentPath),
            lastModified: isLast ? file.last_modified : ''
          };
          
          nodeMap.set(currentPath, node);
          
          if (index === 0) {
            rootNodes.push(node);
          } else {
            const parentPath = pathParts.slice(0, index).join('/');
            const parent = nodeMap.get(parentPath);
            if (parent) {
              parent.children!.push(node);
              node.parent = parent;
            }
          }
        }
      });
    });

    this.treeNodes = rootNodes;
    this.flattenTree();
    this.updateFilteredNodes();
  }

  private flattenTree(): void {
    const flattened: TreeNode[] = [];
    
    const traverse = (nodes: TreeNode[], level: number = 0) => {
      nodes.forEach(node => {
        node.level = level;
        flattened.push(node);
        
        if (node.isExpanded && node.children && node.children.length > 0) {
          traverse(node.children, level + 1);
        }
      });
    };
    
    traverse(this.treeNodes);
    this.flattenedNodes = flattened;
  }

  private updateFilteredNodes(): void {
    let filtered = [...this.flattenedNodes];
    
    // Apply search filter
    if (this.searchTerm) {
      const term = this.searchTerm.toLowerCase();
      filtered = filtered.filter(node => 
        node.name.toLowerCase().includes(term) ||
        node.path.toLowerCase().includes(term)
      );
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;
      
      // Directories first
      if (a.type === 'directory' && b.type !== 'directory') return -1;
      if (a.type !== 'directory' && b.type === 'directory') return 1;
      
      switch (this.sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'size':
          comparison = a.size - b.size;
          break;
        case 'type':
          comparison = a.extension.localeCompare(b.extension);
          break;
        case 'modified':
          comparison = new Date(a.lastModified).getTime() - new Date(b.lastModified).getTime();
          break;
      }
      
      return this.sortOrder === 'asc' ? comparison : -comparison;
    });
    
    this.filteredNodes = filtered;
    this.updateVirtualScrollBounds();
  }

  private updateVirtualScrollBounds(): void {
    const totalHeight = this.filteredNodes.length * this.itemHeight;
    const containerHeight = this.virtualScrollContainer?.clientHeight || 400;
    this.visibleItemCount = Math.ceil(containerHeight / this.itemHeight) + 2;
    
    this.visibleStartIndex = Math.floor(this.virtualScrollOffset / this.itemHeight);
    this.visibleEndIndex = Math.min(
      this.visibleStartIndex + this.visibleItemCount,
      this.filteredNodes.length
    );
  }

  private handleScroll(event: Event): void {
    const target = event.target as HTMLElement;
    this.virtualScrollOffset = target.scrollTop;
    
    if (this.scrollTimeout) {
      clearTimeout(this.scrollTimeout);
    }
    
    this.scrollTimeout = window.setTimeout(() => {
      this.updateVirtualScrollBounds();
      this.requestUpdate();
    }, 16);
  }

  private handleSearch(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.searchTerm = target.value;
    this.updateFilteredNodes();
    
    // Update store state
    this.store.updateFileTreeState({ searchTerm: this.searchTerm });
  }

  private handleSortChange(event: Event): void {
    const target = event.target as HTMLSelectElement;
    const [sortBy, sortOrder] = target.value.split('-');
    this.sortBy = sortBy as any;
    this.sortOrder = sortOrder as any;
    this.updateFilteredNodes();
    
    // Update store state
    this.store.updateFileTreeState({ 
      sortBy: this.sortBy, 
      sortOrder: this.sortOrder 
    });
  }

  private handleNodeClick(node: TreeNode, event: MouseEvent): void {
    event.stopPropagation();
    
    if (event.ctrlKey || event.metaKey) {
      // Multi-select
      const selectedNodes = new Set(this.selectedNodes);
      if (selectedNodes.has(node.id)) {
        selectedNodes.delete(node.id);
      } else {
        selectedNodes.add(node.id);
      }
      this.selectedNodes = selectedNodes;
    } else {
      // Single select
      this.selectedNodes = new Set([node.id]);
    }
    
    // Update store state
    this.store.selectFileTreeNodes(Array.from(this.selectedNodes), false);
    
    // Emit file selection event
    this.dispatchEvent(new CustomEvent('file-selected', {
      detail: { node, selected: this.selectedNodes.has(node.id) },
      bubbles: true,
      composed: true
    }));
  }

  private handleNodeToggle(node: TreeNode, event: MouseEvent): void {
    event.stopPropagation();
    
    if (node.type === 'directory') {
      node.isExpanded = !node.isExpanded;
      
      if (node.isExpanded) {
        this.expandedNodes.add(node.id);
      } else {
        this.expandedNodes.delete(node.id);
      }
      
      this.store.toggleFileTreeNode(node.id);
      this.flattenTree();
      this.updateFilteredNodes();
    }
  }

  private handleContextMenu(node: TreeNode, event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
    
    this.contextMenuTarget = node;
    this.contextMenuPosition = { x: event.clientX, y: event.clientY };
    this.showContextMenu = true;
  }

  private handleDocumentClick(): void {
    this.showContextMenu = false;
  }

  private handleKeyDown(event: KeyboardEvent): void {
    if (!this.selectedNodes.size) return;
    
    switch (event.key) {
      case 'ArrowUp':
        event.preventDefault();
        this.navigateSelection(-1);
        break;
      case 'ArrowDown':
        event.preventDefault();
        this.navigateSelection(1);
        break;
      case 'ArrowRight':
        event.preventDefault();
        this.expandSelectedNode();
        break;
      case 'ArrowLeft':
        event.preventDefault();
        this.collapseSelectedNode();
        break;
      case 'Enter':
        event.preventDefault();
        this.openSelectedNode();
        break;
    }
  }

  private navigateSelection(direction: number): void {
    const selectedNode = Array.from(this.selectedNodes)[0];
    const currentIndex = this.filteredNodes.findIndex(node => node.id === selectedNode);
    
    if (currentIndex !== -1) {
      const newIndex = Math.max(0, Math.min(this.filteredNodes.length - 1, currentIndex + direction));
      const newNode = this.filteredNodes[newIndex];
      this.selectedNodes = new Set([newNode.id]);
      this.store.selectFileTreeNodes([newNode.id], false);
    }
  }

  private expandSelectedNode(): void {
    const selectedNode = Array.from(this.selectedNodes)[0];
    const node = this.filteredNodes.find(n => n.id === selectedNode);
    
    if (node && node.type === 'directory' && !node.isExpanded) {
      this.handleNodeToggle(node, new MouseEvent('click'));
    }
  }

  private collapseSelectedNode(): void {
    const selectedNode = Array.from(this.selectedNodes)[0];
    const node = this.filteredNodes.find(n => n.id === selectedNode);
    
    if (node && node.type === 'directory' && node.isExpanded) {
      this.handleNodeToggle(node, new MouseEvent('click'));
    }
  }

  private openSelectedNode(): void {
    const selectedNode = Array.from(this.selectedNodes)[0];
    const node = this.filteredNodes.find(n => n.id === selectedNode);
    
    if (node) {
      this.dispatchEvent(new CustomEvent('file-open', {
        detail: { node },
        bubbles: true,
        composed: true
      }));
    }
  }

  private getFileIcon(node: TreeNode): any {
    if (node.type === 'directory') {
      return html`
        <svg class="file-icon directory" fill="currentColor" viewBox="0 0 24 24">
          <path d="M10 4H4c-1.11 0-2 .89-2 2v12c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2h-8l-2-2z"/>
        </svg>
      `;
    }
    
    const iconClass = `file-icon ${node.language}`;
    
    return html`
      <svg class=${iconClass} fill="currentColor" viewBox="0 0 24 24">
        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>
      </svg>
    `;
  }

  private formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  }

  private renderContextMenu(): any {
    if (!this.showContextMenu || !this.contextMenuTarget) return '';
    
    return html`
      <div class="context-menu" style="left: ${this.contextMenuPosition.x}px; top: ${this.contextMenuPosition.y}px">
        <div class="context-menu-item" @click=${() => this.handleContextAction('open')}>
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
          Open
        </div>
        
        <div class="context-menu-item" @click=${() => this.handleContextAction('analyze')}>
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
          </svg>
          Analyze
        </div>
        
        <div class="context-menu-separator"></div>
        
        <div class="context-menu-item" @click=${() => this.handleContextAction('dependencies')}>
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
          View Dependencies
        </div>
        
        <div class="context-menu-item" @click=${() => this.handleContextAction('context')}>
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
            <path d="M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0L19.2 12l-4.6-4.6L16 6l6 6-6 6-1.4-1.4z"/>
          </svg>
          Add to Context
        </div>
      </div>
    `;
  }

  private handleContextAction(action: string): void {
    if (!this.contextMenuTarget) return;
    
    this.dispatchEvent(new CustomEvent('context-action', {
      detail: { action, node: this.contextMenuTarget },
      bubbles: true,
      composed: true
    }));
    
    this.showContextMenu = false;
  }

  private renderVirtualizedItems(): any {
    const visibleNodes = this.filteredNodes.slice(this.visibleStartIndex, this.visibleEndIndex);
    const totalHeight = this.filteredNodes.length * this.itemHeight;
    const offsetY = this.visibleStartIndex * this.itemHeight;
    
    return html`
      <div class="virtual-scroll-content" style="height: ${totalHeight}px">
        ${repeat(
          visibleNodes,
          node => node.id,
          (node, index) => {
            const actualIndex = this.visibleStartIndex + index;
            const itemClasses = classMap({
              'tree-item': true,
              'compact': this.compact,
              'selected': node.isSelected || this.selectedNodes.has(node.id)
            });
            
            return html`
              <div 
                class=${itemClasses}
                style="top: ${offsetY + (index * this.itemHeight)}px"
                @click=${(e: MouseEvent) => this.handleNodeClick(node, e)}
                @contextmenu=${(e: MouseEvent) => this.handleContextMenu(node, e)}
              >
                <div class="tree-item-content">
                  <div class="tree-indent" style="width: ${node.level * 16}px"></div>
                  
                  <button 
                    class="tree-expand-toggle ${node.isExpanded ? 'expanded' : ''}"
                    ?disabled=${node.type !== 'directory'}
                    @click=${(e: MouseEvent) => this.handleNodeToggle(node, e)}
                  >
                    <svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z"/>
                    </svg>
                  </button>
                  
                  ${this.getFileIcon(node)}
                  
                  <span class="file-name ${this.compact ? 'compact' : ''}">${node.name}</span>
                  
                  ${!this.compact ? html`
                    <div class="file-meta">
                      <span class="file-size">${this.formatFileSize(node.size)}</span>
                      <div class="file-status ${node.analyzed ? 'analyzed' : 'pending'}"></div>
                    </div>
                  ` : ''}
                </div>
              </div>
            `;
          }
        )}
      </div>
    `;
  }

  render() {
    if (this.loadingState.isLoading) {
      return html`
        <div class="tree-container">
          <div class="loading-container">
            <div class="loading-spinner"></div>
            <span style="margin-left: 0.5rem">Loading files...</span>
          </div>
        </div>
      `;
    }

    if (this.filteredNodes.length === 0) {
      return html`
        <div class="tree-container">
          <div class="empty-state">
            <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" 
                    d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/>
            </svg>
            <p>No files found</p>
          </div>
        </div>
      `;
    }

    return html`
      <div class="tree-container">
        <div class="tree-header ${this.compact ? 'compact' : ''}">
          <input 
            type="text" 
            class="search-box"
            placeholder="Search files..."
            .value=${this.searchTerm}
            @input=${this.handleSearch}
          />
          
          <div class="tree-controls ${this.compact ? 'compact' : ''}">
            <div class="control-group">
              <select class="sort-select" @change=${this.handleSortChange} .value=${`${this.sortBy}-${this.sortOrder}`}>
                <option value="name-asc">Name A-Z</option>
                <option value="name-desc">Name Z-A</option>
                <option value="size-asc">Size ↑</option>
                <option value="size-desc">Size ↓</option>
                <option value="type-asc">Type A-Z</option>
                <option value="modified-desc">Modified ↓</option>
              </select>
            </div>
            
            <div class="tree-stats">
              <span>${this.filteredNodes.length} items</span>
              <span>${this.filteredNodes.filter(n => n.analyzed).length} analyzed</span>
            </div>
          </div>
        </div>
        
        <div class="tree-content">
          <div 
            class="virtual-scroll-container"
            @scroll=${this.handleScroll}
            ${(el: HTMLElement) => {
              this.virtualScrollContainer = el;
              this.updateVirtualScrollBounds();
            }}
          >
            ${this.renderVirtualizedItems()}
          </div>
        </div>
        
        ${this.renderContextMenu()}
      </div>
    `;
  }
}