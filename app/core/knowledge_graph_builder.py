"""
Knowledge Graph Builder for LeanVibe Agent Hive 2.0

Builds and maintains knowledge graphs that map relationships between agent expertise,
knowledge domains, and collaborative patterns for intelligent knowledge discovery.

Features:
- Agent Expertise Mapping: Graph relationships between agents and their capabilities
- Knowledge Domain Networks: Semantic relationships between knowledge areas
- Collaboration Patterns: Historical collaboration success patterns
- Expertise Evolution Tracking: How agent capabilities develop over time
- Knowledge Flow Analysis: How knowledge spreads through the agent network
- Recommendation Engine: Graph-based recommendations for knowledge sharing
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import logging

import structlog
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .cross_agent_knowledge_manager import AgentExpertise, KnowledgeQualityScore
from .agent_knowledge_manager import KnowledgeItem, KnowledgeType, AccessLevel
from .semantic_embedding_service import get_embedding_service, SemanticEmbeddingService

logger = structlog.get_logger()


# =============================================================================
# GRAPH TYPES AND CONFIGURATIONS
# =============================================================================

class GraphType(str, Enum):
    """Types of knowledge graphs."""
    AGENT_EXPERTISE = "agent_expertise"
    KNOWLEDGE_DOMAIN = "knowledge_domain"
    COLLABORATION = "collaboration"
    KNOWLEDGE_FLOW = "knowledge_flow"
    TEMPORAL_EVOLUTION = "temporal_evolution"


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    AGENT = "agent"
    EXPERTISE_DOMAIN = "expertise_domain"
    CAPABILITY = "capability"
    KNOWLEDGE_ITEM = "knowledge_item"
    COLLABORATION_SESSION = "collaboration_session"
    WORKFLOW = "workflow"


class EdgeType(str, Enum):
    """Types of edges in the knowledge graph."""
    HAS_EXPERTISE = "has_expertise"
    COLLABORATES_WITH = "collaborates_with"
    SHARES_KNOWLEDGE = "shares_knowledge"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    EVOLVES_FROM = "evolves_from"
    CONTRIBUTES_TO = "contributes_to"


class RelationshipStrength(str, Enum):
    """Strength levels for relationships."""
    WEAK = "weak"          # 0.0 - 0.3
    MODERATE = "moderate"  # 0.3 - 0.7
    STRONG = "strong"      # 0.7 - 1.0


@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    node_id: str
    node_type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "properties": self.properties,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class GraphEdge:
    """Edge in the knowledge graph."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def get_strength(self) -> RelationshipStrength:
        """Get relationship strength based on weight."""
        if self.weight >= 0.7:
            return RelationshipStrength.STRONG
        elif self.weight >= 0.3:
            return RelationshipStrength.MODERATE
        else:
            return RelationshipStrength.WEAK
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "strength": self.get_strength().value,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class GraphAnalysis:
    """Analysis results for a knowledge graph."""
    graph_type: GraphType
    analysis_id: str
    timestamp: datetime
    
    # Basic metrics
    node_count: int
    edge_count: int
    density: float
    clustering_coefficient: float
    
    # Centrality measures
    top_central_nodes: List[Tuple[str, float]]
    betweenness_centrality: Dict[str, float]
    eigenvector_centrality: Dict[str, float]
    
    # Community detection
    communities: List[List[str]]
    modularity: float
    
    # Knowledge flow metrics
    knowledge_flow_rate: float
    active_knowledge_paths: int
    bottleneck_nodes: List[str]
    
    # Recommendations
    collaboration_recommendations: List[Dict[str, Any]]
    knowledge_sharing_opportunities: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_type": self.graph_type.value,
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "clustering_coefficient": self.clustering_coefficient,
            "top_central_nodes": self.top_central_nodes,
            "betweenness_centrality": self.betweenness_centrality,
            "eigenvector_centrality": self.eigenvector_centrality,
            "communities": self.communities,
            "modularity": self.modularity,
            "knowledge_flow_rate": self.knowledge_flow_rate,
            "active_knowledge_paths": self.active_knowledge_paths,
            "bottleneck_nodes": self.bottleneck_nodes,
            "collaboration_recommendations": self.collaboration_recommendations,
            "knowledge_sharing_opportunities": self.knowledge_sharing_opportunities
        }


# =============================================================================
# SPECIALIZED GRAPH BUILDERS
# =============================================================================

class AgentExpertiseGraphBuilder:
    """Builds graphs of agent expertise and capabilities."""
    
    def __init__(self, embedding_service: SemanticEmbeddingService):
        self.embedding_service = embedding_service
    
    async def build_expertise_graph(
        self,
        agent_expertise: Dict[str, List[AgentExpertise]]
    ) -> nx.Graph:
        """Build bipartite graph of agents and their expertise."""
        graph = nx.Graph()
        
        # Add agent nodes
        for agent_id in agent_expertise.keys():
            graph.add_node(
                agent_id,
                node_type=NodeType.AGENT.value,
                label=f"Agent {agent_id}",
                properties={"agent_id": agent_id}
            )
        
        # Add expertise domain and capability nodes
        expertise_domains = set()
        capabilities = set()
        
        for expertise_list in agent_expertise.values():
            for expertise in expertise_list:
                expertise_domains.add(expertise.domain)
                capabilities.add(expertise.capability)
        
        # Add domain nodes
        for domain in expertise_domains:
            domain_id = f"domain_{domain.replace(' ', '_').lower()}"
            graph.add_node(
                domain_id,
                node_type=NodeType.EXPERTISE_DOMAIN.value,
                label=domain,
                properties={"domain": domain}
            )
        
        # Add capability nodes
        for capability in capabilities:
            capability_id = f"capability_{capability.replace(' ', '_').lower()}"
            graph.add_node(
                capability_id,
                node_type=NodeType.CAPABILITY.value,
                label=capability,
                properties={"capability": capability}
            )
        
        # Add edges between agents and their expertise
        for agent_id, expertise_list in agent_expertise.items():
            for expertise in expertise_list:
                domain_id = f"domain_{expertise.domain.replace(' ', '_').lower()}"
                capability_id = f"capability_{expertise.capability.replace(' ', '_').lower()}"
                
                # Agent -> Domain edge
                graph.add_edge(
                    agent_id, domain_id,
                    edge_type=EdgeType.HAS_EXPERTISE.value,
                    weight=expertise.proficiency_level,
                    properties={
                        "proficiency": expertise.proficiency_level,
                        "evidence_count": expertise.evidence_count,
                        "success_rate": expertise.success_rate,
                        "last_demonstrated": expertise.last_demonstrated.isoformat()
                    }
                )
                
                # Agent -> Capability edge
                graph.add_edge(
                    agent_id, capability_id,
                    edge_type=EdgeType.HAS_EXPERTISE.value,
                    weight=expertise.proficiency_level,
                    properties={
                        "proficiency": expertise.proficiency_level,
                        "evidence_count": expertise.evidence_count,
                        "success_rate": expertise.success_rate
                    }
                )
                
                # Domain -> Capability edge (if not already exists)
                if not graph.has_edge(domain_id, capability_id):
                    graph.add_edge(
                        domain_id, capability_id,
                        edge_type=EdgeType.CONTRIBUTES_TO.value,
                        weight=0.5,
                        properties={"relationship": "domain_capability"}
                    )
        
        return graph
    
    async def find_expertise_clusters(
        self,
        graph: nx.Graph,
        min_cluster_size: int = 2
    ) -> List[List[str]]:
        """Find clusters of related expertise domains."""
        try:
            # Get only expertise domain nodes
            domain_nodes = [
                n for n, d in graph.nodes(data=True)
                if d.get('node_type') == NodeType.EXPERTISE_DOMAIN.value
            ]
            
            if len(domain_nodes) < min_cluster_size:
                return []
            
            # Create subgraph of domains
            domain_subgraph = graph.subgraph(domain_nodes).copy()
            
            # Use community detection
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(domain_subgraph)
            
            # Filter by minimum size
            clusters = [list(community) for community in communities if len(community) >= min_cluster_size]
            
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to find expertise clusters: {e}")
            return []


class CollaborationGraphBuilder:
    """Builds graphs of collaboration patterns between agents."""
    
    async def build_collaboration_graph(
        self,
        collaboration_history: List[Dict[str, Any]]
    ) -> nx.DiGraph:
        """Build directed graph of collaboration relationships."""
        graph = nx.DiGraph()
        
        # Track collaboration statistics
        collaboration_stats = defaultdict(lambda: {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "total_duration": 0,
            "domains": set(),
            "last_collaboration": None
        })
        
        for collaboration in collaboration_history:
            participating_agents = collaboration.get("participating_agents", [])
            success = collaboration.get("success", False)
            duration = collaboration.get("duration_hours", 0)
            domains = collaboration.get("domains", [])
            timestamp = collaboration.get("timestamp", datetime.utcnow())
            
            # Add agent nodes if not exist
            for agent_id in participating_agents:
                if not graph.has_node(agent_id):
                    graph.add_node(
                        agent_id,
                        node_type=NodeType.AGENT.value,
                        label=f"Agent {agent_id}",
                        properties={"agent_id": agent_id}
                    )
            
            # Add collaboration edges between all pairs
            for i, agent1 in enumerate(participating_agents):
                for agent2 in participating_agents[i+1:]:
                    # Create bidirectional collaboration relationship
                    pair_key = tuple(sorted([agent1, agent2]))
                    
                    # Update statistics
                    collaboration_stats[pair_key]["total_collaborations"] += 1
                    if success:
                        collaboration_stats[pair_key]["successful_collaborations"] += 1
                    collaboration_stats[pair_key]["total_duration"] += duration
                    collaboration_stats[pair_key]["domains"].update(domains)
                    collaboration_stats[pair_key]["last_collaboration"] = timestamp
        
        # Add edges with aggregated statistics
        for (agent1, agent2), stats in collaboration_stats.items():
            success_rate = (stats["successful_collaborations"] / 
                          max(1, stats["total_collaborations"]))
            avg_duration = (stats["total_duration"] / 
                          max(1, stats["total_collaborations"]))
            
            # Edge weight based on success rate and frequency
            weight = (success_rate * 0.7) + (min(1.0, stats["total_collaborations"] / 10) * 0.3)
            
            # Add bidirectional edges
            for source, target in [(agent1, agent2), (agent2, agent1)]:
                graph.add_edge(
                    source, target,
                    edge_type=EdgeType.COLLABORATES_WITH.value,
                    weight=weight,
                    properties={
                        "total_collaborations": stats["total_collaborations"],
                        "success_rate": success_rate,
                        "avg_duration_hours": avg_duration,
                        "shared_domains": list(stats["domains"]),
                        "last_collaboration": stats["last_collaboration"].isoformat() if stats["last_collaboration"] else None
                    }
                )
        
        return graph
    
    async def find_collaboration_opportunities(
        self,
        graph: nx.DiGraph,
        min_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Find potential collaboration opportunities based on graph analysis."""
        opportunities = []
        
        try:
            # Find agents with no direct collaboration but common neighbors
            agents = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == NodeType.AGENT.value]
            
            for i, agent1 in enumerate(agents):
                for agent2 in agents[i+1:]:
                    if not graph.has_edge(agent1, agent2):
                        # Check for common collaborators
                        agent1_collaborators = set(graph.neighbors(agent1))
                        agent2_collaborators = set(graph.neighbors(agent2))
                        common_collaborators = agent1_collaborators & agent2_collaborators
                        
                        if common_collaborators:
                            # Calculate opportunity score
                            common_scores = []
                            for collaborator in common_collaborators:
                                if graph.has_edge(agent1, collaborator):
                                    score1 = graph[agent1][collaborator].get('weight', 0)
                                    score2 = graph[agent2][collaborator].get('weight', 0)
                                    common_scores.append((score1 + score2) / 2)
                            
                            if common_scores:
                                opportunity_score = sum(common_scores) / len(common_scores)
                                
                                if opportunity_score >= min_weight:
                                    opportunities.append({
                                        "agent1": agent1,
                                        "agent2": agent2,
                                        "opportunity_score": opportunity_score,
                                        "common_collaborators": list(common_collaborators),
                                        "reasoning": f"Both agents have successful collaborations with {len(common_collaborators)} common partners"
                                    })
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
            
            return opportunities[:10]  # Top 10 opportunities
            
        except Exception as e:
            logger.error(f"Failed to find collaboration opportunities: {e}")
            return []


class KnowledgeFlowGraphBuilder:
    """Builds graphs of knowledge sharing and flow patterns."""
    
    async def build_knowledge_flow_graph(
        self,
        knowledge_sharing_events: List[Dict[str, Any]]
    ) -> nx.DiGraph:
        """Build directed graph of knowledge flow between agents."""
        graph = nx.DiGraph()
        
        # Track knowledge flow statistics
        flow_stats = defaultdict(lambda: {
            "knowledge_shared": 0,
            "total_quality": 0.0,
            "successful_transfers": 0,
            "domains": Counter(),
            "last_share": None
        })
        
        for event in knowledge_sharing_events:
            source_agent = event.get("source_agent")
            target_agent = event.get("target_agent")
            success = event.get("success", False)
            quality_score = event.get("quality_score", 0.5)
            domain = event.get("domain", "general")
            timestamp = event.get("timestamp", datetime.utcnow())
            
            if source_agent and target_agent:
                # Add agent nodes
                for agent_id in [source_agent, target_agent]:
                    if not graph.has_node(agent_id):
                        graph.add_node(
                            agent_id,
                            node_type=NodeType.AGENT.value,
                            label=f"Agent {agent_id}",
                            properties={"agent_id": agent_id}
                        )
                
                # Update flow statistics
                flow_key = (source_agent, target_agent)
                flow_stats[flow_key]["knowledge_shared"] += 1
                flow_stats[flow_key]["total_quality"] += quality_score
                if success:
                    flow_stats[flow_key]["successful_transfers"] += 1
                flow_stats[flow_key]["domains"][domain] += 1
                flow_stats[flow_key]["last_share"] = timestamp
        
        # Add edges with aggregated flow statistics
        for (source, target), stats in flow_stats.items():
            avg_quality = stats["total_quality"] / max(1, stats["knowledge_shared"])
            success_rate = stats["successful_transfers"] / max(1, stats["knowledge_shared"])
            
            # Flow strength based on volume, quality, and success rate
            flow_strength = (
                min(1.0, stats["knowledge_shared"] / 20) * 0.4 +  # Volume
                avg_quality * 0.3 +  # Quality
                success_rate * 0.3   # Success rate
            )
            
            graph.add_edge(
                source, target,
                edge_type=EdgeType.SHARES_KNOWLEDGE.value,
                weight=flow_strength,
                properties={
                    "knowledge_shared": stats["knowledge_shared"],
                    "avg_quality": avg_quality,
                    "success_rate": success_rate,
                    "top_domains": dict(stats["domains"].most_common(3)),
                    "last_share": stats["last_share"].isoformat() if stats["last_share"] else None
                }
            )
        
        return graph
    
    async def analyze_knowledge_bottlenecks(
        self,
        graph: nx.DiGraph
    ) -> List[Dict[str, Any]]:
        """Identify bottlenecks in knowledge flow."""
        bottlenecks = []
        
        try:
            # Calculate betweenness centrality to find bottleneck nodes
            betweenness = nx.betweenness_centrality(graph, weight='weight')
            
            # Find nodes with high betweenness (potential bottlenecks)
            high_betweenness_nodes = [
                (node, score) for node, score in betweenness.items()
                if score > 0.1  # Threshold for considering as bottleneck
            ]
            
            high_betweenness_nodes.sort(key=lambda x: x[1], reverse=True)
            
            for node, betweenness_score in high_betweenness_nodes[:5]:  # Top 5 bottlenecks
                # Analyze the bottleneck
                in_degree = graph.in_degree(node, weight='weight')
                out_degree = graph.out_degree(node, weight='weight')
                
                # Get connected agents
                predecessors = list(graph.predecessors(node))
                successors = list(graph.successors(node))
                
                bottlenecks.append({
                    "agent_id": node,
                    "betweenness_centrality": betweenness_score,
                    "knowledge_in_flow": in_degree,
                    "knowledge_out_flow": out_degree,
                    "flow_balance": out_degree - in_degree,
                    "knowledge_sources": len(predecessors),
                    "knowledge_targets": len(successors),
                    "bottleneck_type": "sink" if in_degree > out_degree else "source" if out_degree > in_degree else "hub"
                })
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Failed to analyze knowledge bottlenecks: {e}")
            return []


# =============================================================================
# MAIN KNOWLEDGE GRAPH BUILDER
# =============================================================================

class KnowledgeGraphBuilder:
    """
    Main knowledge graph builder that creates and maintains multiple types of
    knowledge graphs for comprehensive agent collaboration and knowledge management.
    """
    
    def __init__(self, embedding_service: Optional[SemanticEmbeddingService] = None):
        """Initialize the knowledge graph builder."""
        self.embedding_service = embedding_service
        
        # Specialized graph builders
        self.expertise_builder = None
        self.collaboration_builder = CollaborationGraphBuilder()
        self.knowledge_flow_builder = KnowledgeFlowGraphBuilder()
        
        # Graph storage
        self.graphs: Dict[GraphType, nx.Graph] = {}
        self.graph_metadata: Dict[GraphType, Dict[str, Any]] = {}
        
        # Analysis cache
        self.analysis_cache: Dict[str, GraphAnalysis] = {}
        self.cache_ttl_hours = 6
        
        # Performance metrics
        self.metrics = {
            "graphs_built": 0,
            "analyses_performed": 0,
            "avg_build_time_ms": 0.0,
            "avg_analysis_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info("Knowledge Graph Builder initialized")
    
    async def initialize(self):
        """Initialize the knowledge graph builder with required services."""
        if not self.embedding_service:
            self.embedding_service = await get_embedding_service()
        
        self.expertise_builder = AgentExpertiseGraphBuilder(self.embedding_service)
        
        logger.info("✅ Knowledge Graph Builder fully initialized")
    
    # =============================================================================
    # GRAPH BUILDING OPERATIONS
    # =============================================================================
    
    async def build_agent_expertise_graph(
        self,
        agent_expertise: Dict[str, List[AgentExpertise]]
    ) -> nx.Graph:
        """Build comprehensive agent expertise graph."""
        start_time = time.time()
        
        try:
            graph = await self.expertise_builder.build_expertise_graph(agent_expertise)
            
            # Store graph
            self.graphs[GraphType.AGENT_EXPERTISE] = graph
            self.graph_metadata[GraphType.AGENT_EXPERTISE] = {
                "built_at": datetime.utcnow(),
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "agent_count": len(agent_expertise)
            }
            
            build_time = (time.time() - start_time) * 1000
            self._update_build_metrics(build_time)
            
            logger.info(
                f"Built agent expertise graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            
            return graph
            
        except Exception as e:
            logger.error(f"Failed to build agent expertise graph: {e}")
            raise
    
    async def build_collaboration_graph(
        self,
        collaboration_history: List[Dict[str, Any]]
    ) -> nx.DiGraph:
        """Build collaboration patterns graph."""
        start_time = time.time()
        
        try:
            graph = await self.collaboration_builder.build_collaboration_graph(collaboration_history)
            
            # Store graph
            self.graphs[GraphType.COLLABORATION] = graph
            self.graph_metadata[GraphType.COLLABORATION] = {
                "built_at": datetime.utcnow(),
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "collaborations_analyzed": len(collaboration_history)
            }
            
            build_time = (time.time() - start_time) * 1000
            self._update_build_metrics(build_time)
            
            logger.info(
                f"Built collaboration graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            
            return graph
            
        except Exception as e:
            logger.error(f"Failed to build collaboration graph: {e}")
            raise
    
    async def build_knowledge_flow_graph(
        self,
        knowledge_sharing_events: List[Dict[str, Any]]
    ) -> nx.DiGraph:
        """Build knowledge flow patterns graph."""
        start_time = time.time()
        
        try:
            graph = await self.knowledge_flow_builder.build_knowledge_flow_graph(knowledge_sharing_events)
            
            # Store graph
            self.graphs[GraphType.KNOWLEDGE_FLOW] = graph
            self.graph_metadata[GraphType.KNOWLEDGE_FLOW] = {
                "built_at": datetime.utcnow(),
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "sharing_events_analyzed": len(knowledge_sharing_events)
            }
            
            build_time = (time.time() - start_time) * 1000
            self._update_build_metrics(build_time)
            
            logger.info(
                f"Built knowledge flow graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            
            return graph
            
        except Exception as e:
            logger.error(f"Failed to build knowledge flow graph: {e}")
            raise
    
    async def build_domain_similarity_graph(
        self,
        knowledge_items: List[KnowledgeItem],
        similarity_threshold: float = 0.7
    ) -> nx.Graph:
        """Build graph of similar knowledge domains."""
        start_time = time.time()
        
        try:
            graph = nx.Graph()
            
            # Extract unique domains from knowledge items
            domains = set()
            domain_items = defaultdict(list)
            
            for item in knowledge_items:
                for tag in item.tags:
                    domains.add(tag)
                    domain_items[tag].append(item)
            
            # Add domain nodes
            for domain in domains:
                graph.add_node(
                    domain,
                    node_type=NodeType.EXPERTISE_DOMAIN.value,
                    label=domain,
                    properties={
                        "domain": domain,
                        "knowledge_count": len(domain_items[domain]),
                        "avg_importance": sum(item.importance_score for item in domain_items[domain]) / len(domain_items[domain])
                    }
                )
            
            # Calculate domain similarities and add edges
            domain_list = list(domains)
            for i, domain1 in enumerate(domain_list):
                for domain2 in domain_list[i+1:]:
                    similarity = await self._calculate_domain_similarity(
                        domain_items[domain1], domain_items[domain2]
                    )
                    
                    if similarity >= similarity_threshold:
                        graph.add_edge(
                            domain1, domain2,
                            edge_type=EdgeType.SIMILAR_TO.value,
                            weight=similarity,
                            properties={
                                "similarity_score": similarity,
                                "relationship_type": "semantic_similarity"
                            }
                        )
            
            # Store graph
            self.graphs[GraphType.KNOWLEDGE_DOMAIN] = graph
            self.graph_metadata[GraphType.KNOWLEDGE_DOMAIN] = {
                "built_at": datetime.utcnow(),
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "domains_analyzed": len(domains),
                "similarity_threshold": similarity_threshold
            }
            
            build_time = (time.time() - start_time) * 1000
            self._update_build_metrics(build_time)
            
            logger.info(
                f"Built domain similarity graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            
            return graph
            
        except Exception as e:
            logger.error(f"Failed to build domain similarity graph: {e}")
            raise
    
    # =============================================================================
    # GRAPH ANALYSIS OPERATIONS
    # =============================================================================
    
    async def analyze_graph(
        self,
        graph_type: GraphType,
        force_refresh: bool = False
    ) -> GraphAnalysis:
        """Perform comprehensive analysis of a knowledge graph."""
        cache_key = f"{graph_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H')}"
        
        # Check cache first
        if not force_refresh and cache_key in self.analysis_cache:
            cached_analysis = self.analysis_cache[cache_key]
            if (datetime.utcnow() - cached_analysis.timestamp).total_seconds() < (self.cache_ttl_hours * 3600):
                self.metrics["cache_hits"] += 1
                return cached_analysis
        
        self.metrics["cache_misses"] += 1
        start_time = time.time()
        
        try:
            graph = self.graphs.get(graph_type)
            if not graph:
                raise ValueError(f"Graph {graph_type.value} not found")
            
            analysis = GraphAnalysis(
                graph_type=graph_type,
                analysis_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
                density=nx.density(graph),
                clustering_coefficient=0.0,
                top_central_nodes=[],
                betweenness_centrality={},
                eigenvector_centrality={},
                communities=[],
                modularity=0.0,
                knowledge_flow_rate=0.0,
                active_knowledge_paths=0,
                bottleneck_nodes=[],
                collaboration_recommendations=[],
                knowledge_sharing_opportunities=[]
            )
            
            # Basic network metrics
            if graph.number_of_nodes() > 0:
                try:
                    analysis.clustering_coefficient = nx.average_clustering(graph)
                except:
                    analysis.clustering_coefficient = 0.0
            
            # Centrality measures
            if graph.number_of_nodes() > 1:
                try:
                    betweenness = nx.betweenness_centrality(graph)
                    analysis.betweenness_centrality = betweenness
                    analysis.top_central_nodes = sorted(
                        betweenness.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                    
                    if not graph.is_directed():
                        eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
                        analysis.eigenvector_centrality = eigenvector
                except:
                    pass
            
            # Community detection
            if graph.number_of_nodes() > 2 and not graph.is_directed():
                try:
                    import networkx.algorithms.community as nx_comm
                    communities = nx_comm.greedy_modularity_communities(graph)
                    analysis.communities = [list(community) for community in communities]
                    analysis.modularity = nx_comm.modularity(graph, communities)
                except:
                    pass
            
            # Graph-specific analysis
            if graph_type == GraphType.COLLABORATION:
                analysis.collaboration_recommendations = await self.collaboration_builder.find_collaboration_opportunities(graph)
            elif graph_type == GraphType.KNOWLEDGE_FLOW:
                bottlenecks = await self.knowledge_flow_builder.analyze_knowledge_bottlenecks(graph)
                analysis.bottleneck_nodes = [b["agent_id"] for b in bottlenecks]
                analysis.knowledge_flow_rate = self._calculate_knowledge_flow_rate(graph)
                analysis.active_knowledge_paths = len([
                    edge for edge in graph.edges(data=True)
                    if edge[2].get('weight', 0) > 0.5
                ])
            elif graph_type == GraphType.AGENT_EXPERTISE:
                analysis.knowledge_sharing_opportunities = await self._find_knowledge_sharing_opportunities(graph)
            
            analysis_time = (time.time() - start_time) * 1000
            self._update_analysis_metrics(analysis_time)
            
            # Cache the analysis
            self.analysis_cache[cache_key] = analysis
            
            logger.info(
                f"Analyzed {graph_type.value} graph",
                analysis_id=analysis.analysis_id,
                processing_time=analysis_time
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            raise
    
    async def find_shortest_knowledge_path(
        self,
        source_agent: str,
        target_agent: str,
        graph_type: GraphType = GraphType.KNOWLEDGE_FLOW
    ) -> List[str]:
        """Find shortest path for knowledge transfer between agents."""
        try:
            graph = self.graphs.get(graph_type)
            if not graph:
                return []
            
            if not graph.has_node(source_agent) or not graph.has_node(target_agent):
                return []
            
            try:
                path = nx.shortest_path(graph, source_agent, target_agent, weight='weight')
                return path
            except nx.NetworkXNoPath:
                return []
                
        except Exception as e:
            logger.error(f"Failed to find knowledge path: {e}")
            return []
    
    async def recommend_knowledge_bridges(
        self,
        domain1: str,
        domain2: str,
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """Recommend agents who could bridge knowledge between domains."""
        try:
            expertise_graph = self.graphs.get(GraphType.AGENT_EXPERTISE)
            if not expertise_graph:
                return []
            
            domain1_id = f"domain_{domain1.replace(' ', '_').lower()}"
            domain2_id = f"domain_{domain2.replace(' ', '_').lower()}"
            
            if not expertise_graph.has_node(domain1_id) or not expertise_graph.has_node(domain2_id):
                return []
            
            # Find agents connected to both domains
            domain1_agents = set(expertise_graph.neighbors(domain1_id))
            domain2_agents = set(expertise_graph.neighbors(domain2_id))
            
            bridge_agents = []
            
            # Agents with expertise in both domains
            for agent in domain1_agents & domain2_agents:
                if expertise_graph.nodes[agent].get('node_type') == NodeType.AGENT.value:
                    weight1 = expertise_graph[agent][domain1_id].get('weight', 0)
                    weight2 = expertise_graph[agent][domain2_id].get('weight', 0)
                    
                    bridge_agents.append({
                        "agent_id": agent,
                        "bridge_score": (weight1 + weight2) / 2,
                        "domain1_expertise": weight1,
                        "domain2_expertise": weight2,
                        "bridge_type": "direct"
                    })
            
            # Agents who could facilitate indirect bridging
            for agent1 in domain1_agents:
                if expertise_graph.nodes[agent1].get('node_type') != NodeType.AGENT.value:
                    continue
                    
                for agent2 in domain2_agents:
                    if (expertise_graph.nodes[agent2].get('node_type') != NodeType.AGENT.value or
                        agent1 == agent2):
                        continue
                    
                    # Check if these agents have collaborated
                    collaboration_graph = self.graphs.get(GraphType.COLLABORATION)
                    if (collaboration_graph and 
                        collaboration_graph.has_edge(agent1, agent2)):
                        
                        collaboration_strength = collaboration_graph[agent1][agent2].get('weight', 0)
                        weight1 = expertise_graph[agent1][domain1_id].get('weight', 0)
                        weight2 = expertise_graph[agent2][domain2_id].get('weight', 0)
                        
                        bridge_score = (weight1 + weight2 + collaboration_strength) / 3
                        
                        bridge_agents.append({
                            "agent_pair": [agent1, agent2],
                            "bridge_score": bridge_score,
                            "collaboration_strength": collaboration_strength,
                            "bridge_type": "collaborative"
                        })
            
            # Sort by bridge score
            bridge_agents.sort(key=lambda x: x["bridge_score"], reverse=True)
            
            return bridge_agents[:max_recommendations]
            
        except Exception as e:
            logger.error(f"Failed to find knowledge bridges: {e}")
            return []
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    async def _calculate_domain_similarity(
        self,
        items1: List[KnowledgeItem],
        items2: List[KnowledgeItem]
    ) -> float:
        """Calculate similarity between two knowledge domains."""
        try:
            # Simple content-based similarity
            content1 = " ".join(item.content for item in items1)
            content2 = " ".join(item.content for item in items2)
            
            # Generate embeddings
            embedding1 = await self.embedding_service.generate_embedding(content1[:1000])  # Limit length
            embedding2 = await self.embedding_service.generate_embedding(content2[:1000])
            
            if embedding1 and embedding2:
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    np.array(embedding1).reshape(1, -1),
                    np.array(embedding2).reshape(1, -1)
                )[0][0]
                return max(0.0, min(1.0, similarity))
            
            # Fallback: tag overlap similarity
            tags1 = set()
            tags2 = set()
            for item in items1:
                tags1.update(item.tags)
            for item in items2:
                tags2.update(item.tags)
            
            if tags1 and tags2:
                return len(tags1 & tags2) / len(tags1 | tags2)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate domain similarity: {e}")
            return 0.0
    
    def _calculate_knowledge_flow_rate(self, graph: nx.DiGraph) -> float:
        """Calculate overall knowledge flow rate in the graph."""
        try:
            if graph.number_of_edges() == 0:
                return 0.0
            
            total_flow = sum(
                data.get('weight', 0) * data.get('properties', {}).get('knowledge_shared', 1)
                for _, _, data in graph.edges(data=True)
            )
            
            max_possible_flow = graph.number_of_edges() * 10  # Assume max 10 knowledge items per edge
            
            return min(1.0, total_flow / max_possible_flow)
            
        except Exception as e:
            logger.error(f"Failed to calculate knowledge flow rate: {e}")
            return 0.0
    
    async def _find_knowledge_sharing_opportunities(
        self,
        graph: nx.Graph
    ) -> List[Dict[str, Any]]:
        """Find opportunities for knowledge sharing based on expertise graph."""
        opportunities = []
        
        try:
            # Find agents with complementary expertise
            agent_nodes = [
                n for n, d in graph.nodes(data=True)
                if d.get('node_type') == NodeType.AGENT.value
            ]
            
            for i, agent1 in enumerate(agent_nodes):
                for agent2 in agent_nodes[i+1:]:
                    # Get their expertise domains
                    agent1_domains = [
                        n for n in graph.neighbors(agent1)
                        if graph.nodes[n].get('node_type') == NodeType.EXPERTISE_DOMAIN.value
                    ]
                    agent2_domains = [
                        n for n in graph.neighbors(agent2)
                        if graph.nodes[n].get('node_type') == NodeType.EXPERTISE_DOMAIN.value
                    ]
                    
                    # Find complementary domains (agent1 has, agent2 doesn't)
                    complementary_domains = []
                    for domain in agent1_domains:
                        if domain not in agent2_domains:
                            expertise_weight = graph[agent1][domain].get('weight', 0)
                            if expertise_weight > 0.6:  # High expertise threshold
                                complementary_domains.append({
                                    "domain": domain,
                                    "expertise_level": expertise_weight
                                })
                    
                    if complementary_domains:
                        opportunity_score = sum(d["expertise_level"] for d in complementary_domains) / len(complementary_domains)
                        
                        opportunities.append({
                            "source_agent": agent1,
                            "target_agent": agent2,
                            "opportunity_score": opportunity_score,
                            "complementary_domains": complementary_domains,
                            "potential_benefit": "Knowledge transfer opportunity"
                        })
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
            
            return opportunities[:10]
            
        except Exception as e:
            logger.error(f"Failed to find knowledge sharing opportunities: {e}")
            return []
    
    def _update_build_metrics(self, build_time_ms: float):
        """Update build performance metrics."""
        self.metrics["graphs_built"] += 1
        if self.metrics["graphs_built"] > 0:
            self.metrics["avg_build_time_ms"] = (
                (self.metrics["avg_build_time_ms"] * (self.metrics["graphs_built"] - 1) + build_time_ms) /
                self.metrics["graphs_built"]
            )
    
    def _update_analysis_metrics(self, analysis_time_ms: float):
        """Update analysis performance metrics."""
        self.metrics["analyses_performed"] += 1
        if self.metrics["analyses_performed"] > 0:
            self.metrics["avg_analysis_time_ms"] = (
                (self.metrics["avg_analysis_time_ms"] * (self.metrics["analyses_performed"] - 1) + analysis_time_ms) /
                self.metrics["analyses_performed"]
            )
    
    # =============================================================================
    # PUBLIC API METHODS
    # =============================================================================
    
    def get_graph_summary(self, graph_type: GraphType) -> Dict[str, Any]:
        """Get summary information about a specific graph."""
        try:
            graph = self.graphs.get(graph_type)
            metadata = self.graph_metadata.get(graph_type, {})
            
            if not graph:
                return {"error": f"Graph {graph_type.value} not found"}
            
            return {
                "graph_type": graph_type.value,
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "density": nx.density(graph),
                "is_directed": graph.is_directed(),
                "metadata": metadata,
                "last_analysis": self._get_latest_analysis_timestamp(graph_type)
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph summary: {e}")
            return {"error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the knowledge graph builder."""
        return {
            **self.metrics,
            "graphs_maintained": len(self.graphs),
            "cached_analyses": len(self.analysis_cache),
            "cache_hit_rate": (
                self.metrics["cache_hits"] / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on knowledge graph builder."""
        try:
            # Test basic graph creation
            test_expertise = {
                "test_agent": [
                    AgentExpertise(
                        agent_id="test_agent",
                        domain="test_domain",
                        capability="test_capability",
                        proficiency_level=0.8,
                        evidence_count=1,
                        success_rate=1.0,
                        last_demonstrated=datetime.utcnow()
                    )
                ]
            }
            
            test_graph = await self.build_agent_expertise_graph(test_expertise)
            
            return {
                "status": "healthy",
                "components": {
                    "expertise_builder": "operational",
                    "collaboration_builder": "operational",
                    "knowledge_flow_builder": "operational",
                    "embedding_service": "operational" if self.embedding_service else "unavailable"
                },
                "test_results": {
                    "graph_creation": test_graph.number_of_nodes() > 0,
                    "graph_analysis": True
                },
                "metrics": self.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Knowledge graph builder health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "components": {
                    "expertise_builder": "unknown",
                    "collaboration_builder": "unknown",
                    "knowledge_flow_builder": "unknown",
                    "embedding_service": "unknown"
                }
            }
    
    def _get_latest_analysis_timestamp(self, graph_type: GraphType) -> Optional[str]:
        """Get timestamp of latest analysis for a graph type."""
        latest_timestamp = None
        for cache_key, analysis in self.analysis_cache.items():
            if analysis.graph_type == graph_type:
                if not latest_timestamp or analysis.timestamp > latest_timestamp:
                    latest_timestamp = analysis.timestamp
        
        return latest_timestamp.isoformat() if latest_timestamp else None


# =============================================================================
# GLOBAL KNOWLEDGE GRAPH BUILDER INSTANCE
# =============================================================================

_graph_builder: Optional[KnowledgeGraphBuilder] = None


async def get_knowledge_graph_builder() -> KnowledgeGraphBuilder:
    """Get global knowledge graph builder instance."""
    global _graph_builder
    
    if _graph_builder is None:
        _graph_builder = KnowledgeGraphBuilder()
        await _graph_builder.initialize()
    
    return _graph_builder


async def cleanup_knowledge_graph_builder():
    """Clean up global knowledge graph builder."""
    global _graph_builder
    _graph_builder = None