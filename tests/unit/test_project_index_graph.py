"""
Unit tests for Project Index dependency graph operations.

Tests for dependency graph construction, analysis algorithms,
and graph manipulation operations.
"""

import pytest
from unittest.mock import MagicMock
from typing import List, Set

from app.project_index.graph import (
    GraphNodeType,
    GraphNode,
    GraphEdge,
    GraphCycle,
    GraphMetrics,
    DependencyGraph
)
from app.project_index.models import DependencyResult


class TestGraphNode:
    """Test GraphNode functionality."""
    
    def test_basic_node_creation(self):
        """Test basic node creation."""
        node = GraphNode(
            id="test_node",
            name="Test Node",
            node_type=GraphNodeType.INTERNAL_FILE,
            file_path="/path/to/file.py",
            metadata={"test": True}
        )
        
        assert node.id == "test_node"
        assert node.name == "Test Node"
        assert node.node_type == GraphNodeType.INTERNAL_FILE
        assert node.file_path == "/path/to/file.py"
        assert node.metadata == {"test": True}
    
    def test_node_equality(self):
        """Test node equality based on ID."""
        node1 = GraphNode(
            id="same_id",
            name="Node 1",
            node_type=GraphNodeType.INTERNAL_FILE
        )
        node2 = GraphNode(
            id="same_id",
            name="Node 2",  # Different name
            node_type=GraphNodeType.EXTERNAL_MODULE  # Different type
        )
        node3 = GraphNode(
            id="different_id",
            name="Node 3",
            node_type=GraphNodeType.INTERNAL_FILE
        )
        
        assert node1 == node2  # Same ID
        assert node1 != node3  # Different ID
        assert hash(node1) == hash(node2)  # Same hash for same ID
        assert hash(node1) != hash(node3)  # Different hash for different ID
    
    def test_node_types(self):
        """Test different node types."""
        internal_node = GraphNode(
            id="internal", 
            name="Internal", 
            node_type=GraphNodeType.INTERNAL_FILE
        )
        external_node = GraphNode(
            id="external", 
            name="External", 
            node_type=GraphNodeType.EXTERNAL_MODULE
        )
        package_node = GraphNode(
            id="package", 
            name="Package", 
            node_type=GraphNodeType.PACKAGE
        )
        directory_node = GraphNode(
            id="directory", 
            name="Directory", 
            node_type=GraphNodeType.DIRECTORY
        )
        
        assert internal_node.node_type == GraphNodeType.INTERNAL_FILE
        assert external_node.node_type == GraphNodeType.EXTERNAL_MODULE
        assert package_node.node_type == GraphNodeType.PACKAGE
        assert directory_node.node_type == GraphNodeType.DIRECTORY


class TestGraphEdge:
    """Test GraphEdge functionality."""
    
    def test_basic_edge_creation(self):
        """Test basic edge creation."""
        source = GraphNode(id="source", name="Source", node_type=GraphNodeType.INTERNAL_FILE)
        target = GraphNode(id="target", name="Target", node_type=GraphNodeType.INTERNAL_FILE)
        
        edge = GraphEdge(
            source=source,
            target=target,
            dependency_type="import",
            weight=1.0,
            metadata={"line": 5}
        )
        
        assert edge.source == source
        assert edge.target == target
        assert edge.dependency_type == "import"
        assert edge.weight == 1.0
        assert edge.metadata == {"line": 5}
    
    def test_edge_hash(self):
        """Test edge hashing based on source, target, and type."""
        source = GraphNode(id="source", name="Source", node_type=GraphNodeType.INTERNAL_FILE)
        target = GraphNode(id="target", name="Target", node_type=GraphNodeType.INTERNAL_FILE)
        
        edge1 = GraphEdge(source=source, target=target, dependency_type="import")
        edge2 = GraphEdge(source=source, target=target, dependency_type="import")
        edge3 = GraphEdge(source=source, target=target, dependency_type="require")
        
        assert hash(edge1) == hash(edge2)  # Same source, target, type
        assert hash(edge1) != hash(edge3)  # Different type


class TestGraphCycle:
    """Test GraphCycle functionality."""
    
    def test_cycle_creation(self):
        """Test cycle creation."""
        node1 = GraphNode(id="node1", name="Node 1", node_type=GraphNodeType.INTERNAL_FILE)
        node2 = GraphNode(id="node2", name="Node 2", node_type=GraphNodeType.INTERNAL_FILE)
        
        edge1 = GraphEdge(source=node1, target=node2, dependency_type="import")
        edge2 = GraphEdge(source=node2, target=node1, dependency_type="import")
        
        cycle = GraphCycle(
            nodes=[node1, node2],
            edges=[edge1, edge2],
            cycle_length=2,
            cycle_weight=2.0
        )
        
        assert len(cycle.nodes) == 2
        assert len(cycle.edges) == 2
        assert cycle.cycle_length == 2
        assert cycle.cycle_weight == 2.0


class TestGraphMetrics:
    """Test GraphMetrics functionality."""
    
    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = GraphMetrics(
            node_count=100,
            edge_count=150,
            internal_nodes=80,
            external_nodes=20,
            strongly_connected_components=5,
            cycles_detected=2,
            max_depth=10,
            average_degree=3.0,
            density=0.15
        )
        
        assert metrics.node_count == 100
        assert metrics.edge_count == 150
        assert metrics.internal_nodes == 80
        assert metrics.external_nodes == 20
        assert metrics.cycles_detected == 2
        assert metrics.average_degree == 3.0


class TestDependencyGraph:
    """Test DependencyGraph functionality."""
    
    def test_empty_graph_creation(self):
        """Test creating empty graph."""
        graph = DependencyGraph()
        
        assert len(graph.get_nodes()) == 0
        assert len(graph.get_edges()) == 0
    
    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = DependencyGraph()
        node = GraphNode(id="test", name="Test", node_type=GraphNodeType.INTERNAL_FILE)
        
        # Add new node
        added = graph.add_node(node)
        assert added is True
        assert len(graph.get_nodes()) == 1
        assert graph.get_node("test") == node
        
        # Add same node again (should update)
        added_again = graph.add_node(node)
        assert added_again is False  # Not new
        assert len(graph.get_nodes()) == 1
    
    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = DependencyGraph()
        
        source = GraphNode(id="source", name="Source", node_type=GraphNodeType.INTERNAL_FILE)
        target = GraphNode(id="target", name="Target", node_type=GraphNodeType.INTERNAL_FILE)
        edge = GraphEdge(source=source, target=target, dependency_type="import")
        
        # Add edge (should auto-add nodes)
        added = graph.add_edge(edge)
        assert added is True
        assert len(graph.get_nodes()) == 2
        assert len(graph.get_edges()) == 1
        
        # Add same edge again
        added_again = graph.add_edge(edge)
        assert added_again is False
        assert len(graph.get_edges()) == 1
    
    def test_remove_node(self):
        """Test removing nodes from graph."""
        graph = DependencyGraph()
        
        source = GraphNode(id="source", name="Source", node_type=GraphNodeType.INTERNAL_FILE)
        target = GraphNode(id="target", name="Target", node_type=GraphNodeType.INTERNAL_FILE)
        edge = GraphEdge(source=source, target=target, dependency_type="import")
        
        graph.add_edge(edge)
        assert len(graph.get_nodes()) == 2
        assert len(graph.get_edges()) == 1
        
        # Remove source node (should also remove edge)
        removed = graph.remove_node("source")
        assert removed is True
        assert len(graph.get_nodes()) == 1
        assert len(graph.get_edges()) == 0
        assert graph.get_node("source") is None
        assert graph.get_node("target") is not None
        
        # Try to remove non-existent node
        removed_again = graph.remove_node("nonexistent")
        assert removed_again is False
    
    def test_get_dependencies_and_dependents(self):
        """Test getting dependencies and dependents."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        node_c = GraphNode(id="c", name="C", node_type=GraphNodeType.INTERNAL_FILE)
        
        # A depends on B, B depends on C
        edge_ab = GraphEdge(source=node_a, target=node_b, dependency_type="import")
        edge_bc = GraphEdge(source=node_b, target=node_c, dependency_type="import")
        
        graph.add_edge(edge_ab)
        graph.add_edge(edge_bc)
        
        # Test dependencies (outgoing)
        a_deps = graph.get_dependencies("a")
        assert len(a_deps) == 1
        assert a_deps[0].target.id == "b"
        
        b_deps = graph.get_dependencies("b")
        assert len(b_deps) == 1
        assert b_deps[0].target.id == "c"
        
        c_deps = graph.get_dependencies("c")
        assert len(c_deps) == 0
        
        # Test dependents (incoming)
        a_dependents = graph.get_dependents("a")
        assert len(a_dependents) == 0
        
        b_dependents = graph.get_dependents("b")
        assert len(b_dependents) == 1
        assert b_dependents[0].source.id == "a"
        
        c_dependents = graph.get_dependents("c")
        assert len(c_dependents) == 1
        assert c_dependents[0].source.id == "b"
    
    def test_detect_cycles_no_cycle(self):
        """Test cycle detection with no cycles."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        node_c = GraphNode(id="c", name="C", node_type=GraphNodeType.INTERNAL_FILE)
        
        # Linear chain: A -> B -> C
        edge_ab = GraphEdge(source=node_a, target=node_b, dependency_type="import")
        edge_bc = GraphEdge(source=node_b, target=node_c, dependency_type="import")
        
        graph.add_edge(edge_ab)
        graph.add_edge(edge_bc)
        
        cycles = graph.detect_cycles()
        assert len(cycles) == 0
    
    def test_detect_cycles_with_cycle(self):
        """Test cycle detection with cycles."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        
        # Cycle: A -> B -> A
        edge_ab = GraphEdge(source=node_a, target=node_b, dependency_type="import")
        edge_ba = GraphEdge(source=node_b, target=node_a, dependency_type="import")
        
        graph.add_edge(edge_ab)
        graph.add_edge(edge_ba)
        
        cycles = graph.detect_cycles()
        assert len(cycles) >= 1
        
        # Check first cycle
        cycle = cycles[0]
        assert cycle.cycle_length >= 2
        assert len(cycle.nodes) >= 2
        assert len(cycle.edges) >= 2
    
    def test_topological_sort_no_cycles(self):
        """Test topological sort with no cycles."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        node_c = GraphNode(id="c", name="C", node_type=GraphNodeType.INTERNAL_FILE)
        
        # A -> B -> C
        edge_ab = GraphEdge(source=node_a, target=node_b, dependency_type="import")
        edge_bc = GraphEdge(source=node_b, target=node_c, dependency_type="import")
        
        graph.add_edge(edge_ab)
        graph.add_edge(edge_bc)
        
        sorted_nodes = graph.topological_sort()
        assert sorted_nodes is not None
        assert len(sorted_nodes) == 3
        
        # Check ordering (A should come before B, B before C)
        node_positions = {node.id: i for i, node in enumerate(sorted_nodes)}
        assert node_positions["a"] < node_positions["b"]
        assert node_positions["b"] < node_positions["c"]
    
    def test_topological_sort_with_cycles(self):
        """Test topological sort with cycles (should return None)."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        
        # Cycle: A -> B -> A
        edge_ab = GraphEdge(source=node_a, target=node_b, dependency_type="import")
        edge_ba = GraphEdge(source=node_b, target=node_a, dependency_type="import")
        
        graph.add_edge(edge_ab)
        graph.add_edge(edge_ba)
        
        sorted_nodes = graph.topological_sort()
        assert sorted_nodes is None
    
    def test_find_shortest_path(self):
        """Test shortest path finding."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        node_c = GraphNode(id="c", name="C", node_type=GraphNodeType.INTERNAL_FILE)
        node_d = GraphNode(id="d", name="D", node_type=GraphNodeType.INTERNAL_FILE)
        
        # A -> B -> D and A -> C -> D (two paths from A to D)
        graph.add_edge(GraphEdge(source=node_a, target=node_b, dependency_type="import"))
        graph.add_edge(GraphEdge(source=node_b, target=node_d, dependency_type="import"))
        graph.add_edge(GraphEdge(source=node_a, target=node_c, dependency_type="import"))
        graph.add_edge(GraphEdge(source=node_c, target=node_d, dependency_type="import"))
        
        # Find shortest path from A to D
        path = graph.find_shortest_path("a", "d")
        assert path is not None
        assert len(path) == 3  # A -> B/C -> D
        assert path[0].id == "a"
        assert path[-1].id == "d"
        
        # Find path to non-existent node
        no_path = graph.find_shortest_path("a", "nonexistent")
        assert no_path is None
        
        # Find path to same node
        same_path = graph.find_shortest_path("a", "a")
        assert same_path is not None
        assert len(same_path) == 1
        assert same_path[0].id == "a"
    
    def test_find_all_paths(self):
        """Test finding all paths between nodes."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        node_c = GraphNode(id="c", name="C", node_type=GraphNodeType.INTERNAL_FILE)
        node_d = GraphNode(id="d", name="D", node_type=GraphNodeType.INTERNAL_FILE)
        
        # A -> B -> D and A -> C -> D (two paths from A to D)
        graph.add_edge(GraphEdge(source=node_a, target=node_b, dependency_type="import"))
        graph.add_edge(GraphEdge(source=node_b, target=node_d, dependency_type="import"))
        graph.add_edge(GraphEdge(source=node_a, target=node_c, dependency_type="import"))
        graph.add_edge(GraphEdge(source=node_c, target=node_d, dependency_type="import"))
        
        # Find all paths from A to D
        paths = graph.find_all_paths("a", "d", max_depth=5)
        assert len(paths) == 2  # Two paths: A->B->D and A->C->D
        
        for path in paths:
            assert path[0].id == "a"
            assert path[-1].id == "d"
            assert len(path) == 3
    
    def test_calculate_impact_score(self):
        """Test impact score calculation."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        node_c = GraphNode(id="c", name="C", node_type=GraphNodeType.INTERNAL_FILE)
        
        # B depends on A, C depends on A (A has high impact)
        graph.add_edge(GraphEdge(source=node_b, target=node_a, dependency_type="import", weight=1.0))
        graph.add_edge(GraphEdge(source=node_c, target=node_a, dependency_type="import", weight=1.0))
        
        # A has high impact (2 dependents)
        impact_a = graph.calculate_impact_score("a")
        assert impact_a > 0
        
        # B and C have lower impact (0 dependents)
        impact_b = graph.calculate_impact_score("b")
        impact_c = graph.calculate_impact_score("c")
        
        assert impact_a > impact_b
        assert impact_a > impact_c
        
        # Non-existent node
        impact_none = graph.calculate_impact_score("nonexistent")
        assert impact_none == 0.0
    
    def test_strongly_connected_components(self):
        """Test strongly connected components detection."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        node_c = GraphNode(id="c", name="C", node_type=GraphNodeType.INTERNAL_FILE)
        
        # Create strongly connected component: A <-> B
        graph.add_edge(GraphEdge(source=node_a, target=node_b, dependency_type="import"))
        graph.add_edge(GraphEdge(source=node_b, target=node_a, dependency_type="import"))
        # C is separate
        graph.add_edge(GraphEdge(source=node_c, target=node_a, dependency_type="import"))
        
        components = graph.find_strongly_connected_components()
        assert len(components) >= 2  # At least A-B component and C component
        
        # Find the component containing both A and B
        ab_component = None
        for component in components:
            node_ids = {node.id for node in component}
            if "a" in node_ids and "b" in node_ids:
                ab_component = component
                break
        
        assert ab_component is not None
        assert len(ab_component) == 2
    
    def test_get_metrics(self):
        """Test graph metrics calculation."""
        graph = DependencyGraph()
        
        # Create simple graph
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.EXTERNAL_MODULE)
        
        graph.add_edge(GraphEdge(source=node_a, target=node_b, dependency_type="import"))
        
        metrics = graph.get_metrics()
        
        assert metrics.node_count == 2
        assert metrics.edge_count == 1
        assert metrics.internal_nodes == 1
        assert metrics.external_nodes == 1
        assert metrics.max_depth >= 1
        assert metrics.average_degree > 0
        assert metrics.density > 0
    
    def test_build_from_dependencies(self):
        """Test building graph from dependency results."""
        graph = DependencyGraph()
        
        # Create dependency results
        dep1 = DependencyResult(
            source_file_path="/path/to/a.py",
            source_file_id="file_a",
            target_name="b",
            target_path="/path/to/b.py",
            target_file_id="file_b",
            dependency_type="import",
            is_external=False,
            confidence_score=0.9
        )
        
        dep2 = DependencyResult(
            source_file_path="/path/to/a.py",
            source_file_id="file_a",
            target_name="requests",
            dependency_type="import",
            is_external=True,
            confidence_score=0.8
        )
        
        dependencies = [dep1, dep2]
        graph.build_from_dependencies(dependencies)
        
        # Check nodes were created
        nodes = graph.get_nodes()
        assert len(nodes) == 3  # a.py, b.py, requests
        
        # Check edges were created
        edges = graph.get_edges()
        assert len(edges) == 2
        
        # Check node types
        node_dict = {node.id: node for node in nodes}
        assert "/path/to/a.py" in node_dict
        assert node_dict["/path/to/a.py"].node_type == GraphNodeType.INTERNAL_FILE
        assert "requests" in node_dict
        assert node_dict["requests"].node_type == GraphNodeType.EXTERNAL_MODULE
    
    def test_filter_internal_only(self):
        """Test filtering to internal nodes only."""
        graph = DependencyGraph()
        
        # Create mixed internal/external graph
        internal_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        internal_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        external_c = GraphNode(id="c", name="C", node_type=GraphNodeType.EXTERNAL_MODULE)
        
        graph.add_edge(GraphEdge(source=internal_a, target=internal_b, dependency_type="import"))
        graph.add_edge(GraphEdge(source=internal_a, target=external_c, dependency_type="import"))
        graph.add_edge(GraphEdge(source=internal_b, target=external_c, dependency_type="import"))
        
        # Filter to internal only
        internal_graph = graph.filter_internal_only()
        
        internal_nodes = internal_graph.get_nodes()
        internal_edges = internal_graph.get_edges()
        
        # Should have only internal nodes
        assert len(internal_nodes) == 2
        for node in internal_nodes:
            assert node.node_type == GraphNodeType.INTERNAL_FILE
        
        # Should have only edges between internal nodes
        assert len(internal_edges) == 1
        edge = internal_edges[0]
        assert edge.source.node_type == GraphNodeType.INTERNAL_FILE
        assert edge.target.node_type == GraphNodeType.INTERNAL_FILE
    
    def test_get_subgraph(self):
        """Test getting subgraph for specific nodes."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        node_c = GraphNode(id="c", name="C", node_type=GraphNodeType.INTERNAL_FILE)
        
        graph.add_edge(GraphEdge(source=node_a, target=node_b, dependency_type="import"))
        graph.add_edge(GraphEdge(source=node_b, target=node_c, dependency_type="import"))
        graph.add_edge(GraphEdge(source=node_a, target=node_c, dependency_type="import"))
        
        # Get subgraph with nodes A and B only
        subgraph = graph.get_subgraph({"a", "b"})
        
        sub_nodes = subgraph.get_nodes()
        sub_edges = subgraph.get_edges()
        
        # Should have nodes A and B
        assert len(sub_nodes) == 2
        node_ids = {node.id for node in sub_nodes}
        assert node_ids == {"a", "b"}
        
        # Should have edge A -> B only
        assert len(sub_edges) == 1
        edge = sub_edges[0]
        assert edge.source.id == "a"
        assert edge.target.id == "b"
    
    def test_clear_graph(self):
        """Test clearing graph."""
        graph = DependencyGraph()
        
        node_a = GraphNode(id="a", name="A", node_type=GraphNodeType.INTERNAL_FILE)
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.INTERNAL_FILE)
        graph.add_edge(GraphEdge(source=node_a, target=node_b, dependency_type="import"))
        
        assert len(graph.get_nodes()) == 2
        assert len(graph.get_edges()) == 1
        
        graph.clear()
        
        assert len(graph.get_nodes()) == 0
        assert len(graph.get_edges()) == 0
    
    def test_to_dict(self):
        """Test converting graph to dictionary."""
        graph = DependencyGraph()
        
        node_a = GraphNode(
            id="a", 
            name="A", 
            node_type=GraphNodeType.INTERNAL_FILE,
            file_path="/path/to/a.py",
            metadata={"size": 1000}
        )
        node_b = GraphNode(id="b", name="B", node_type=GraphNodeType.EXTERNAL_MODULE)
        
        edge = GraphEdge(
            source=node_a, 
            target=node_b, 
            dependency_type="import",
            weight=0.9,
            metadata={"line": 5}
        )
        
        graph.add_edge(edge)
        
        graph_dict = graph.to_dict()
        
        assert "nodes" in graph_dict
        assert "edges" in graph_dict
        assert "metrics" in graph_dict
        
        # Check nodes
        assert len(graph_dict["nodes"]) == 2
        node_a_dict = next(n for n in graph_dict["nodes"] if n["id"] == "a")
        assert node_a_dict["name"] == "A"
        assert node_a_dict["type"] == "internal_file"
        assert node_a_dict["file_path"] == "/path/to/a.py"
        assert node_a_dict["metadata"] == {"size": 1000}
        
        # Check edges
        assert len(graph_dict["edges"]) == 1
        edge_dict = graph_dict["edges"][0]
        assert edge_dict["source"] == "a"
        assert edge_dict["target"] == "b"
        assert edge_dict["dependency_type"] == "import"
        assert edge_dict["weight"] == 0.9
        assert edge_dict["metadata"] == {"line": 5}
        
        # Check metrics
        assert isinstance(graph_dict["metrics"], dict)
        assert "node_count" in graph_dict["metrics"]
        assert "edge_count" in graph_dict["metrics"]