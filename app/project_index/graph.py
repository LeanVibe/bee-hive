"""
Dependency Graph Operations for LeanVibe Agent Hive 2.0

Provides graph algorithms and analysis for code dependency relationships.
Supports cycle detection, path finding, impact analysis, and graph visualization.
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

import structlog

from .models import DependencyResult

logger = structlog.get_logger()


class GraphNodeType(Enum):
    """Types of nodes in the dependency graph."""
    INTERNAL_FILE = "internal_file"
    EXTERNAL_MODULE = "external_module"
    PACKAGE = "package"
    DIRECTORY = "directory"


@dataclass
class GraphNode:
    """Represents a node in the dependency graph."""
    id: str
    name: str
    node_type: GraphNodeType
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.id == other.id


@dataclass
class GraphEdge:
    """Represents an edge in the dependency graph."""
    source: GraphNode
    target: GraphNode
    dependency_type: str
    weight: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __hash__(self):
        return hash((self.source.id, self.target.id, self.dependency_type))


@dataclass
class GraphCycle:
    """Represents a cycle in the dependency graph."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    cycle_length: int
    cycle_weight: float = 0.0


@dataclass
class GraphMetrics:
    """Graph analysis metrics."""
    node_count: int
    edge_count: int
    internal_nodes: int
    external_nodes: int
    strongly_connected_components: int
    cycles_detected: int
    max_depth: int
    average_degree: float
    density: float


class DependencyGraph:
    """
    Dependency graph for code analysis with advanced algorithms.
    
    Provides graph construction, analysis, and manipulation capabilities
    optimized for software dependency analysis.
    """
    
    def __init__(self):
        """Initialize empty dependency graph."""
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._adjacency_list: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)
        
        # Cached analysis results
        self._cycles_cache: Optional[List[GraphCycle]] = None
        self._metrics_cache: Optional[GraphMetrics] = None
        self._scc_cache: Optional[List[List[GraphNode]]] = None
        self._topological_order_cache: Optional[List[GraphNode]] = None
    
    def add_node(self, node: GraphNode) -> bool:
        """
        Add a node to the graph.
        
        Args:
            node: GraphNode to add
            
        Returns:
            True if node was added, False if it already exists
        """
        if node.id in self._nodes:
            # Update existing node
            self._nodes[node.id] = node
            self._invalidate_caches()
            return False
        
        self._nodes[node.id] = node
        self._invalidate_caches()
        
        logger.debug("Node added to graph", node_id=node.id, node_type=node.node_type.value)
        return True
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """
        Add an edge to the graph.
        
        Args:
            edge: GraphEdge to add
            
        Returns:
            True if edge was added, False if it already exists
        """
        # Ensure nodes exist
        if edge.source.id not in self._nodes:
            self.add_node(edge.source)
        if edge.target.id not in self._nodes:
            self.add_node(edge.target)
        
        # Check if edge already exists
        for existing_edge in self._adjacency_list[edge.source.id]:
            if (existing_edge.target.id == edge.target.id and 
                existing_edge.dependency_type == edge.dependency_type):
                return False
        
        # Add edge
        self._edges.append(edge)
        self._adjacency_list[edge.source.id].append(edge)
        self._reverse_adjacency[edge.target.id].append(edge)
        self._invalidate_caches()
        
        logger.debug("Edge added to graph", 
                    source=edge.source.id, 
                    target=edge.target.id,
                    dependency_type=edge.dependency_type)
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its edges from the graph.
        
        Args:
            node_id: ID of node to remove
            
        Returns:
            True if node was removed, False if it didn't exist
        """
        if node_id not in self._nodes:
            return False
        
        # Remove all edges involving this node
        edges_to_remove = []
        for edge in self._edges:
            if edge.source.id == node_id or edge.target.id == node_id:
                edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            self._remove_edge(edge)
        
        # Remove node
        del self._nodes[node_id]
        if node_id in self._adjacency_list:
            del self._adjacency_list[node_id]
        if node_id in self._reverse_adjacency:
            del self._reverse_adjacency[node_id]
        
        self._invalidate_caches()
        
        logger.debug("Node removed from graph", node_id=node_id)
        return True
    
    def _remove_edge(self, edge: GraphEdge) -> None:
        """Remove an edge from the graph."""
        if edge in self._edges:
            self._edges.remove(edge)
        
        # Remove from adjacency lists
        if edge in self._adjacency_list[edge.source.id]:
            self._adjacency_list[edge.source.id].remove(edge)
        
        if edge in self._reverse_adjacency[edge.target.id]:
            self._reverse_adjacency[edge.target.id].remove(edge)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_nodes(self) -> List[GraphNode]:
        """Get all nodes in the graph."""
        return list(self._nodes.values())
    
    def get_edges(self) -> List[GraphEdge]:
        """Get all edges in the graph."""
        return self._edges.copy()
    
    def get_dependencies(self, node_id: str) -> List[GraphEdge]:
        """Get outgoing dependencies (edges) from a node."""
        return self._adjacency_list[node_id].copy()
    
    def get_dependents(self, node_id: str) -> List[GraphEdge]:
        """Get incoming dependencies (edges) to a node."""
        return self._reverse_adjacency[node_id].copy()
    
    # ================== GRAPH ANALYSIS ==================
    
    def detect_cycles(self) -> List[GraphCycle]:
        """
        Detect all cycles in the dependency graph using DFS.
        
        Returns:
            List of GraphCycle objects representing detected cycles
        """
        if self._cycles_cache is not None:
            return self._cycles_cache
        
        cycles = []
        visited = set()
        recursion_stack = set()
        path = []
        
        def dfs(node_id: str, path: List[str]) -> None:
            visited.add(node_id)
            recursion_stack.add(node_id)
            path.append(node_id)
            
            for edge in self._adjacency_list[node_id]:
                target_id = edge.target.id
                
                if target_id in recursion_stack:
                    # Found a cycle
                    cycle_start_idx = path.index(target_id)
                    cycle_nodes = [self._nodes[nid] for nid in path[cycle_start_idx:]]
                    cycle_edges = []
                    
                    # Build cycle edges
                    for i in range(len(cycle_nodes)):
                        source_node = cycle_nodes[i]
                        target_node = cycle_nodes[(i + 1) % len(cycle_nodes)]
                        
                        # Find the edge between these nodes
                        for e in self._adjacency_list[source_node.id]:
                            if e.target.id == target_node.id:
                                cycle_edges.append(e)
                                break
                    
                    cycle = GraphCycle(
                        nodes=cycle_nodes,
                        edges=cycle_edges,
                        cycle_length=len(cycle_nodes),
                        cycle_weight=sum(e.weight for e in cycle_edges)
                    )
                    cycles.append(cycle)
                    
                elif target_id not in visited:
                    dfs(target_id, path)
            
            path.pop()
            recursion_stack.remove(node_id)
        
        # Start DFS from all unvisited nodes
        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id, [])
        
        self._cycles_cache = cycles
        
        logger.info("Cycle detection completed", cycle_count=len(cycles))
        return cycles
    
    def find_strongly_connected_components(self) -> List[List[GraphNode]]:
        """
        Find strongly connected components using Tarjan's algorithm.
        
        Returns:
            List of strongly connected components (each is a list of nodes)
        """
        if self._scc_cache is not None:
            return self._scc_cache
        
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        components = []
        
        def strongconnect(node_id: str):
            index[node_id] = index_counter[0]
            lowlinks[node_id] = index_counter[0]
            index_counter[0] += 1
            stack.append(node_id)
            on_stack[node_id] = True
            
            for edge in self._adjacency_list[node_id]:
                successor = edge.target.id
                
                if successor not in index:
                    strongconnect(successor)
                    lowlinks[node_id] = min(lowlinks[node_id], lowlinks[successor])
                elif on_stack.get(successor, False):
                    lowlinks[node_id] = min(lowlinks[node_id], index[successor])
            
            # If node_id is a root node, pop the stack and create an SCC
            if lowlinks[node_id] == index[node_id]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(self._nodes[w])
                    if w == node_id:
                        break
                components.append(component)
        
        for node_id in self._nodes:
            if node_id not in index:
                strongconnect(node_id)
        
        self._scc_cache = components
        
        logger.info("SCC analysis completed", component_count=len(components))
        return components
    
    def topological_sort(self) -> Optional[List[GraphNode]]:
        """
        Perform topological sort using Kahn's algorithm.
        
        Returns:
            Topologically sorted list of nodes, or None if graph has cycles
        """
        if self._topological_order_cache is not None:
            return self._topological_order_cache
        
        # Calculate in-degrees
        in_degrees = {node_id: 0 for node_id in self._nodes}
        for edge in self._edges:
            in_degrees[edge.target.id] += 1
        
        # Initialize queue with nodes having in-degree 0
        queue = deque([node_id for node_id, degree in in_degrees.items() if degree == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(self._nodes[node_id])
            
            # Remove edges from this node
            for edge in self._adjacency_list[node_id]:
                target_id = edge.target.id
                in_degrees[target_id] -= 1
                if in_degrees[target_id] == 0:
                    queue.append(target_id)
        
        # Check if all nodes were processed (no cycles)
        if len(result) != len(self._nodes):
            logger.warning("Topological sort failed - graph has cycles")
            return None
        
        self._topological_order_cache = result
        
        logger.info("Topological sort completed", node_count=len(result))
        return result
    
    def find_shortest_path(
        self, 
        source_id: str, 
        target_id: str
    ) -> Optional[List[GraphNode]]:
        """
        Find shortest path between two nodes using BFS.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            List of nodes in shortest path, or None if no path exists
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        
        if source_id == target_id:
            return [self._nodes[source_id]]
        
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            for edge in self._adjacency_list[current_id]:
                neighbor_id = edge.target.id
                
                if neighbor_id == target_id:
                    # Found target
                    final_path = path + [neighbor_id]
                    return [self._nodes[nid] for nid in final_path]
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        # No path found
        return None
    
    def find_all_paths(
        self, 
        source_id: str, 
        target_id: str, 
        max_depth: int = 10
    ) -> List[List[GraphNode]]:
        """
        Find all paths between two nodes up to max_depth.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path depth to search
            
        Returns:
            List of paths (each path is a list of nodes)
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return []
        
        if source_id == target_id:
            return [[self._nodes[source_id]]]
        
        all_paths = []
        
        def dfs(current_id: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            
            if current_id == target_id:
                all_paths.append([self._nodes[nid] for nid in path])
                return
            
            for edge in self._adjacency_list[current_id]:
                neighbor_id = edge.target.id
                if neighbor_id not in path:  # Avoid cycles
                    dfs(neighbor_id, path + [neighbor_id], depth + 1)
        
        dfs(source_id, [source_id], 0)
        return all_paths
    
    def calculate_impact_score(self, node_id: str) -> float:
        """
        Calculate impact score based on number of dependents and their weights.
        
        Args:
            node_id: Node ID to calculate impact for
            
        Returns:
            Impact score (higher means more impact if changed)
        """
        if node_id not in self._nodes:
            return 0.0
        
        # Direct dependents
        direct_dependents = len(self._reverse_adjacency[node_id])
        
        # Weighted impact based on edge weights
        weighted_impact = sum(edge.weight for edge in self._reverse_adjacency[node_id])
        
        # Transitive dependents (simplified calculation)
        visited = set()
        
        def count_transitive_dependents(nid: str) -> int:
            if nid in visited:
                return 0
            
            visited.add(nid)
            count = 0
            
            for edge in self._reverse_adjacency[nid]:
                count += 1 + count_transitive_dependents(edge.source.id)
            
            return count
        
        transitive_dependents = count_transitive_dependents(node_id)
        
        # Combine metrics
        impact_score = (direct_dependents * 2.0) + weighted_impact + (transitive_dependents * 0.5)
        
        return impact_score
    
    def get_metrics(self) -> GraphMetrics:
        """
        Calculate comprehensive graph metrics.
        
        Returns:
            GraphMetrics object with analysis results
        """
        if self._metrics_cache is not None:
            return self._metrics_cache
        
        node_count = len(self._nodes)
        edge_count = len(self._edges)
        
        # Count internal vs external nodes
        internal_nodes = sum(1 for node in self._nodes.values() 
                           if node.node_type == GraphNodeType.INTERNAL_FILE)
        external_nodes = node_count - internal_nodes
        
        # Strongly connected components
        scc = self.find_strongly_connected_components()
        scc_count = len(scc)
        
        # Cycles
        cycles = self.detect_cycles()
        cycle_count = len(cycles)
        
        # Calculate max depth
        max_depth = 0
        for node_id in self._nodes:
            depth = self._calculate_max_depth_from_node(node_id)
            max_depth = max(max_depth, depth)
        
        # Calculate average degree
        if node_count > 0:
            total_degree = sum(len(self._adjacency_list[node_id]) + len(self._reverse_adjacency[node_id]) 
                             for node_id in self._nodes)
            average_degree = total_degree / node_count
        else:
            average_degree = 0.0
        
        # Calculate density
        max_edges = node_count * (node_count - 1)
        density = (edge_count / max_edges) if max_edges > 0 else 0.0
        
        metrics = GraphMetrics(
            node_count=node_count,
            edge_count=edge_count,
            internal_nodes=internal_nodes,
            external_nodes=external_nodes,
            strongly_connected_components=scc_count,
            cycles_detected=cycle_count,
            max_depth=max_depth,
            average_degree=average_degree,
            density=density
        )
        
        self._metrics_cache = metrics
        
        logger.info("Graph metrics calculated", 
                   node_count=node_count, 
                   edge_count=edge_count,
                   cycles=cycle_count)
        
        return metrics
    
    def _calculate_max_depth_from_node(self, start_node_id: str) -> int:
        """Calculate maximum depth reachable from a node."""
        visited = set()
        
        def dfs(node_id: str) -> int:
            if node_id in visited:
                return 0
            
            visited.add(node_id)
            max_child_depth = 0
            
            for edge in self._adjacency_list[node_id]:
                child_depth = dfs(edge.target.id)
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth + 1
        
        return dfs(start_node_id)
    
    # ================== GRAPH OPERATIONS ==================
    
    def build_from_dependencies(self, dependencies: List[DependencyResult]) -> None:
        """
        Build graph from dependency analysis results.
        
        Args:
            dependencies: List of DependencyResult objects
        """
        logger.info("Building dependency graph", dependency_count=len(dependencies))
        
        # Clear existing graph
        self.clear()
        
        # Create nodes and edges from dependencies
        for dep in dependencies:
            # Create source node
            source_node = GraphNode(
                id=dep.source_file_path,
                name=dep.source_file_path.split('/')[-1] if '/' in dep.source_file_path else dep.source_file_path,
                node_type=GraphNodeType.INTERNAL_FILE,
                file_path=dep.source_file_path,
                metadata={'file_id': dep.source_file_id}
            )
            
            # Create target node
            if dep.is_external:
                target_node = GraphNode(
                    id=dep.target_name,
                    name=dep.target_name,
                    node_type=GraphNodeType.EXTERNAL_MODULE,
                    metadata={'is_external': True}
                )
            else:
                target_node = GraphNode(
                    id=dep.target_path or dep.target_name,
                    name=dep.target_name,
                    node_type=GraphNodeType.INTERNAL_FILE,
                    file_path=dep.target_path,
                    metadata={'file_id': dep.target_file_id}
                )
            
            # Add nodes
            self.add_node(source_node)
            self.add_node(target_node)
            
            # Create edge
            edge = GraphEdge(
                source=source_node,
                target=target_node,
                dependency_type=dep.dependency_type,
                weight=dep.confidence_score,
                metadata={
                    'line_number': dep.line_number,
                    'column_number': dep.column_number,
                    'source_text': dep.source_text,
                    'is_dynamic': dep.is_dynamic
                }
            )
            
            self.add_edge(edge)
        
        logger.info("Dependency graph built", 
                   node_count=len(self._nodes), 
                   edge_count=len(self._edges))
    
    def filter_internal_only(self) -> 'DependencyGraph':
        """
        Create a new graph containing only internal dependencies.
        
        Returns:
            New DependencyGraph with only internal nodes and edges
        """
        filtered_graph = DependencyGraph()
        
        # Add only internal nodes
        for node in self._nodes.values():
            if node.node_type == GraphNodeType.INTERNAL_FILE:
                filtered_graph.add_node(node)
        
        # Add only edges between internal nodes
        for edge in self._edges:
            if (edge.source.node_type == GraphNodeType.INTERNAL_FILE and
                edge.target.node_type == GraphNodeType.INTERNAL_FILE):
                filtered_graph.add_edge(edge)
        
        logger.info("Internal-only graph created", 
                   original_nodes=len(self._nodes),
                   filtered_nodes=len(filtered_graph._nodes))
        
        return filtered_graph
    
    def get_subgraph(self, node_ids: Set[str]) -> 'DependencyGraph':
        """
        Create a subgraph containing only specified nodes and their edges.
        
        Args:
            node_ids: Set of node IDs to include in subgraph
            
        Returns:
            New DependencyGraph containing only specified nodes
        """
        subgraph = DependencyGraph()
        
        # Add specified nodes
        for node_id in node_ids:
            if node_id in self._nodes:
                subgraph.add_node(self._nodes[node_id])
        
        # Add edges between included nodes
        for edge in self._edges:
            if edge.source.id in node_ids and edge.target.id in node_ids:
                subgraph.add_edge(edge)
        
        logger.info("Subgraph created", 
                   requested_nodes=len(node_ids),
                   subgraph_nodes=len(subgraph._nodes))
        
        return subgraph
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self._nodes.clear()
        self._edges.clear()
        self._adjacency_list.clear()
        self._reverse_adjacency.clear()
        self._invalidate_caches()
        
        logger.debug("Graph cleared")
    
    def _invalidate_caches(self) -> None:
        """Invalidate all cached analysis results."""
        self._cycles_cache = None
        self._metrics_cache = None
        self._scc_cache = None
        self._topological_order_cache = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            'nodes': [
                {
                    'id': node.id,
                    'name': node.name,
                    'type': node.node_type.value,
                    'file_path': node.file_path,
                    'metadata': node.metadata
                }
                for node in self._nodes.values()
            ],
            'edges': [
                {
                    'source': edge.source.id,
                    'target': edge.target.id,
                    'dependency_type': edge.dependency_type,
                    'weight': edge.weight,
                    'metadata': edge.metadata
                }
                for edge in self._edges
            ],
            'metrics': self.get_metrics().__dict__
        }