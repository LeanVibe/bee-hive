"""
Issue Manager for LeanVibe Agent Hive 2.0

Bi-directional GitHub issue tracking and intelligent agent assignment
with capability-based matching and automated progress tracking.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.orm import selectinload

from ..core.config import get_settings
from ..core.database import get_db_session
from ..models.agent import Agent, AgentStatus
from ..models.github_integration import (
    GitHubRepository, GitHubIssue, IssueState, AgentWorkTree
)
from ..core.github_api_client import GitHubAPIClient


logger = logging.getLogger(__name__)
settings = get_settings()


class IssuePriority(Enum):
    """Issue priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class IssueType(Enum):
    """Issue types for classification."""
    BUG = "bug"
    FEATURE = "feature"
    ENHANCEMENT = "enhancement"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTENANCE = "maintenance"
    QUESTION = "question"


class IssueManagerError(Exception):
    """Custom exception for issue management operations."""
    pass


class IssueClassifier:
    """
    Intelligent issue classification based on content analysis.
    
    Uses pattern matching and NLP techniques to automatically classify
    issues by type, priority, and required capabilities.
    """
    
    def __init__(self):
        self.classification_patterns = {
            IssueType.BUG: [
                r'\b(bug|error|exception|crash|fail|broken|not work)',
                r'\b(stack trace|traceback|null pointer|segfault)',
                r'\b(unexpected behavior|incorrect result|wrong output)'
            ],
            IssueType.FEATURE: [
                r'\b(feature|add|implement|new|create)',
                r'\b(functionality|capability|support for)',
                r'\b(would like|need to|should be able to)'
            ],
            IssueType.ENHANCEMENT: [
                r'\b(improve|enhance|optimize|better|faster)',
                r'\b(performance|efficiency|usability)',
                r'\b(refactor|restructure|redesign)'
            ],
            IssueType.DOCUMENTATION: [
                r'\b(document|readme|docs|documentation)',
                r'\b(tutorial|guide|example|instruction)',
                r'\b(comment|docstring|api documentation)'
            ],
            IssueType.SECURITY: [
                r'\b(security|vulnerability|exploit|attack)',
                r'\b(authentication|authorization|permission)',
                r'\b(injection|xss|csrf|sanitize)'
            ],
            IssueType.PERFORMANCE: [
                r'\b(slow|performance|speed|optimize)',
                r'\b(memory leak|cpu usage|bottleneck)',
                r'\b(latency|response time|throughput)'
            ]
        }
        
        self.priority_patterns = {
            IssuePriority.CRITICAL: [
                r'\b(critical|urgent|blocker|blocking)',
                r'\b(production down|system down|outage)',
                r'\b(data loss|corruption|security breach)'
            ],
            IssuePriority.HIGH: [
                r'\b(high|important|major|significant)',
                r'\b(affects many|widespread|serious)',
                r'\b(deadline|release blocker)'
            ],
            IssuePriority.LOW: [
                r'\b(low|minor|trivial|cosmetic)',
                r'\b(nice to have|when time permits)',
                r'\b(enhancement|improvement)'
            ]
        }
        
        self.capability_patterns = {
            "backend_development": [
                r'\b(api|backend|server|database|sql)',
                r'\b(rest|graphql|microservice|endpoint)',
                r'\b(authentication|authorization|middleware)'
            ],
            "frontend_development": [
                r'\b(ui|frontend|interface|react|vue|angular)',
                r'\b(component|css|html|javascript|typescript)',
                r'\b(responsive|layout|styling)'
            ],
            "testing": [
                r'\b(test|testing|unittest|integration test)',
                r'\b(coverage|assertion|mock|fixture)',
                r'\b(ci|cd|automation|pipeline)'
            ],
            "devops": [
                r'\b(deploy|deployment|infrastructure|docker)',
                r'\b(kubernetes|aws|cloud|monitoring)',
                r'\b(ci/cd|pipeline|automation)'
            ],
            "security": [
                r'\b(security|vulnerability|encryption|ssl)',
                r'\b(authentication|authorization|oauth)',
                r'\b(penetration test|security audit)'
            ],
            "performance": [
                r'\b(performance|optimization|speed|caching)',
                r'\b(memory|cpu|latency|throughput)',
                r'\b(profiling|benchmarking|load test)'
            ],
            "database": [
                r'\b(database|sql|postgresql|mysql|mongodb)',
                r'\b(query|index|migration|schema)',
                r'\b(orm|sqlalchemy|transaction)'
            ],
            "machine_learning": [
                r'\b(ml|machine learning|ai|model|training)',
                r'\b(neural network|deep learning|algorithm)',
                r'\b(prediction|classification|regression)'
            ]
        }
        
    def classify_issue(self, title: str, description: str, labels: List[str] = None) -> Dict[str, Any]:
        """Classify issue based on title, description, and labels."""
        
        labels = labels or []
        text = f"{title} {description}".lower()
        
        classification = {
            "issue_type": self._classify_type(text, labels),
            "priority": self._classify_priority(text, labels),
            "required_capabilities": self._identify_capabilities(text, labels),
            "complexity_score": self._estimate_complexity(text, labels),
            "effort_estimate": self._estimate_effort(text, labels),
            "confidence": 0.0
        }
        
        # Calculate confidence based on pattern matches
        classification["confidence"] = self._calculate_confidence(text, classification)
        
        return classification
        
    def _classify_type(self, text: str, labels: List[str]) -> str:
        """Classify issue type based on content."""
        
        # Check labels first
        label_types = {
            "bug": IssueType.BUG,
            "enhancement": IssueType.ENHANCEMENT,
            "feature": IssueType.FEATURE,
            "documentation": IssueType.DOCUMENTATION,
            "security": IssueType.SECURITY,
            "performance": IssueType.PERFORMANCE
        }
        
        for label in labels:
            label_lower = label.lower()
            for pattern, issue_type in label_types.items():
                if pattern in label_lower:
                    return issue_type.value
                    
        # Pattern matching
        max_matches = 0
        best_type = IssueType.FEATURE  # Default
        
        for issue_type, patterns in self.classification_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
            if matches > max_matches:
                max_matches = matches
                best_type = issue_type
                
        return best_type.value
        
    def _classify_priority(self, text: str, labels: List[str]) -> str:
        """Classify issue priority based on content."""
        
        # Check labels first
        label_priorities = {
            "critical": IssuePriority.CRITICAL,
            "high": IssuePriority.HIGH,
            "low": IssuePriority.LOW,
            "p1": IssuePriority.CRITICAL,
            "p2": IssuePriority.HIGH,
            "p3": IssuePriority.MEDIUM,
            "p4": IssuePriority.LOW
        }
        
        for label in labels:
            label_lower = label.lower()
            for pattern, priority in label_priorities.items():
                if pattern in label_lower:
                    return priority.value
                    
        # Pattern matching
        for priority, patterns in self.priority_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return priority.value
                    
        return IssuePriority.MEDIUM.value  # Default
        
    def _identify_capabilities(self, text: str, labels: List[str]) -> List[str]:
        """Identify required capabilities based on content."""
        
        required_capabilities = []
        
        for capability, patterns in self.capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    required_capabilities.append(capability)
                    break
                    
        # Also check labels for capabilities
        for label in labels:
            label_lower = label.lower()
            if label_lower in self.capability_patterns:
                if label_lower not in required_capabilities:
                    required_capabilities.append(label_lower)
                    
        return required_capabilities or ["general_development"]
        
    def _estimate_complexity(self, text: str, labels: List[str]) -> float:
        """Estimate issue complexity (0.0 to 1.0)."""
        
        complexity_indicators = {
            "high": [
                r'\b(complex|complicated|difficult|challenging)',
                r'\b(multiple|many|several|various)',
                r'\b(architecture|design|refactor|rewrite)',
                r'\b(integration|compatibility|migration)'
            ],
            "medium": [
                r'\b(implement|add|create|build)',
                r'\b(update|modify|change|improve)',
                r'\b(fix|resolve|address)'
            ],
            "low": [
                r'\b(simple|easy|quick|minor)',
                r'\b(typo|spelling|format|style)',
                r'\b(comment|documentation|readme)'
            ]
        }
        
        # Base complexity
        complexity = 0.5
        
        # Check for high complexity indicators
        high_matches = sum(1 for pattern in complexity_indicators["high"] 
                          if re.search(pattern, text, re.IGNORECASE))
        complexity += high_matches * 0.15
        
        # Check for low complexity indicators
        low_matches = sum(1 for pattern in complexity_indicators["low"] 
                         if re.search(pattern, text, re.IGNORECASE))
        complexity -= low_matches * 0.15
        
        # Check text length (longer descriptions often mean more complex issues)
        if len(text) > 1000:
            complexity += 0.1
        elif len(text) < 200:
            complexity -= 0.1
            
        return max(0.0, min(1.0, complexity))
        
    def _estimate_effort(self, text: str, labels: List[str]) -> int:
        """Estimate effort in story points (1-13 Fibonacci scale)."""
        
        complexity = self._estimate_complexity(text, labels)
        
        # Map complexity to story points
        if complexity < 0.2:
            return 1  # Very simple
        elif complexity < 0.4:
            return 2  # Simple
        elif complexity < 0.6:
            return 3  # Medium
        elif complexity < 0.8:
            return 5  # Complex
        else:
            return 8  # Very complex
            
    def _calculate_confidence(self, text: str, classification: Dict[str, Any]) -> float:
        """Calculate confidence in classification."""
        
        confidence_factors = []
        
        # Pattern match strength
        type_patterns = self.classification_patterns.get(IssueType(classification["issue_type"]), [])
        type_matches = sum(1 for pattern in type_patterns if re.search(pattern, text, re.IGNORECASE))
        confidence_factors.append(min(1.0, type_matches / 3))
        
        # Priority pattern matches
        priority_patterns = self.priority_patterns.get(IssuePriority(classification["priority"]), [])
        priority_matches = sum(1 for pattern in priority_patterns if re.search(pattern, text, re.IGNORECASE))
        confidence_factors.append(min(1.0, priority_matches / 2))
        
        # Capability pattern matches
        capability_matches = len(classification["required_capabilities"])
        confidence_factors.append(min(1.0, capability_matches / 3))
        
        # Text length (more text usually means better classification)
        text_factor = min(1.0, len(text) / 500)
        confidence_factors.append(text_factor)
        
        return sum(confidence_factors) / len(confidence_factors)


class AgentMatcher:
    """
    Intelligent agent matching based on capabilities and workload.
    
    Matches issues to agents using capability scores, current workload,
    performance history, and availability.
    """
    
    def __init__(self):
        self.matching_weights = {
            "capability_match": 0.4,
            "workload_factor": 0.25,
            "performance_history": 0.2,
            "availability": 0.15
        }
        
    async def find_best_agent(
        self,
        issue: GitHubIssue,
        required_capabilities: List[str],
        repository_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find the best agent for an issue based on multiple factors."""
        
        async with get_db_session() as session:
            # Get available agents
            result = await session.execute(
                select(Agent).where(
                    and_(
                        Agent.status == AgentStatus.ACTIVE,
                        Agent.capabilities.isnot(None)
                    )
                )
            )
            available_agents = result.scalars().all()
            
            if not available_agents:
                return None
                
            # Score each agent
            agent_scores = []
            for agent in available_agents:
                score = await self._calculate_agent_score(
                    agent, issue, required_capabilities, repository_id, session
                )
                if score["total_score"] > 0:
                    agent_scores.append({
                        "agent": agent,
                        "score": score
                    })
                    
            # Sort by total score
            agent_scores.sort(key=lambda x: x["score"]["total_score"], reverse=True)
            
            if agent_scores:
                best_match = agent_scores[0]
                return {
                    "agent": best_match["agent"],
                    "score": best_match["score"],
                    "alternatives": agent_scores[1:3]  # Top 3 alternatives
                }
                
            return None
            
    async def _calculate_agent_score(
        self,
        agent: Agent,
        issue: GitHubIssue,
        required_capabilities: List[str],
        repository_id: str,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Calculate comprehensive score for agent-issue matching."""
        
        score_details = {
            "capability_score": 0.0,
            "workload_score": 0.0,
            "performance_score": 0.0,
            "availability_score": 0.0,
            "total_score": 0.0
        }
        
        # 1. Capability matching
        score_details["capability_score"] = self._calculate_capability_score(agent, required_capabilities)
        
        # 2. Workload assessment
        score_details["workload_score"] = await self._calculate_workload_score(agent, session)
        
        # 3. Performance history
        score_details["performance_score"] = await self._calculate_performance_score(agent, repository_id, session)
        
        # 4. Availability
        score_details["availability_score"] = self._calculate_availability_score(agent)
        
        # Calculate weighted total
        score_details["total_score"] = (
            score_details["capability_score"] * self.matching_weights["capability_match"] +
            score_details["workload_score"] * self.matching_weights["workload_factor"] +
            score_details["performance_score"] * self.matching_weights["performance_history"] +
            score_details["availability_score"] * self.matching_weights["availability"]
        )
        
        return score_details
        
    def _calculate_capability_score(self, agent: Agent, required_capabilities: List[str]) -> float:
        """Calculate how well agent capabilities match requirements."""
        
        if not agent.capabilities or not required_capabilities:
            return 0.0
            
        total_score = 0.0
        max_possible_score = len(required_capabilities)
        
        for req_cap in required_capabilities:
            best_match_score = 0.0
            
            for agent_cap in agent.capabilities:
                cap_name = agent_cap.get("name", "").lower()
                confidence = agent_cap.get("confidence_level", 0.0)
                areas = agent_cap.get("specialization_areas", [])
                
                # Direct match
                if req_cap.lower() in cap_name:
                    best_match_score = max(best_match_score, confidence)
                # Area match
                elif any(req_cap.lower() in area.lower() for area in areas):
                    best_match_score = max(best_match_score, confidence * 0.8)
                # Partial match
                elif any(word in cap_name for word in req_cap.lower().split('_')):
                    best_match_score = max(best_match_score, confidence * 0.6)
                    
            total_score += best_match_score
            
        return total_score / max_possible_score if max_possible_score > 0 else 0.0
        
    async def _calculate_workload_score(self, agent: Agent, session: AsyncSession) -> float:
        """Calculate agent workload score (higher score = lower workload)."""
        
        # Count currently assigned issues
        result = await session.execute(
            select(func.count(GitHubIssue.id)).where(
                and_(
                    GitHubIssue.assignee_agent_id == agent.id,
                    GitHubIssue.state == IssueState.OPEN
                )
            )
        )
        assigned_issues = result.scalar() or 0
        
        # Context window usage
        context_usage = float(agent.context_window_usage or 0.0)
        
        # Calculate workload score (inverse relationship)
        issue_factor = max(0.0, 1.0 - (assigned_issues / 10))  # Diminishing returns after 10 issues
        context_factor = max(0.0, 1.0 - context_usage)
        
        return (issue_factor + context_factor) / 2
        
    async def _calculate_performance_score(self, agent: Agent, repository_id: str, session: AsyncSession) -> float:
        """Calculate agent performance score based on history."""
        
        # Get recent issue completion rate
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        # Count completed issues
        completed_result = await session.execute(
            select(func.count(GitHubIssue.id)).where(
                and_(
                    GitHubIssue.assignee_agent_id == agent.id,
                    GitHubIssue.state == IssueState.CLOSED,
                    GitHubIssue.updated_at >= cutoff_date
                )
            )
        )
        completed_issues = completed_result.scalar() or 0
        
        # Count total assigned issues in period
        total_result = await session.execute(
            select(func.count(GitHubIssue.id)).where(
                and_(
                    GitHubIssue.assignee_agent_id == agent.id,
                    GitHubIssue.assigned_at >= cutoff_date
                )
            )
        )
        total_issues = total_result.scalar() or 0
        
        # Calculate completion rate
        completion_rate = completed_issues / total_issues if total_issues > 0 else 0.5
        
        # Use agent's task completion metrics
        total_completed = int(agent.total_tasks_completed or 0)
        total_failed = int(agent.total_tasks_failed or 0)
        
        if total_completed + total_failed > 0:
            success_rate = total_completed / (total_completed + total_failed)
        else:
            success_rate = 0.5
            
        return (completion_rate + success_rate) / 2
        
    def _calculate_availability_score(self, agent: Agent) -> float:
        """Calculate agent availability score."""
        
        if agent.status != AgentStatus.ACTIVE:
            return 0.0
            
        # Check last heartbeat
        if agent.last_heartbeat:
            time_since_heartbeat = (datetime.utcnow() - agent.last_heartbeat).total_seconds()
            heartbeat_factor = max(0.0, 1.0 - (time_since_heartbeat / 3600))  # 1 hour decay
        else:
            heartbeat_factor = 0.5
            
        # Check response time
        avg_response_time = float(agent.average_response_time or 0.0)
        response_factor = max(0.0, 1.0 - (avg_response_time / 60))  # 60 second max
        
        return (heartbeat_factor + response_factor) / 2


class IssueManager:
    """
    Comprehensive issue management system for GitHub integration.
    
    Provides bi-directional issue synchronization, intelligent agent assignment,
    and automated progress tracking for multi-agent development workflows.
    """
    
    def __init__(self, github_client: GitHubAPIClient = None):
        self.github_client = github_client or GitHubAPIClient()
        self.classifier = IssueClassifier()
        self.agent_matcher = AgentMatcher()
        
    async def sync_repository_issues(self, repository: GitHubRepository) -> Dict[str, Any]:
        """Sync all issues from GitHub repository."""
        
        repo_parts = repository.repository_full_name.split('/')
        sync_result = {
            "success": False,
            "issues_synced": 0,
            "issues_updated": 0,
            "issues_created": 0,
            "errors": []
        }
        
        try:
            # Get issues from GitHub
            github_issues = await self.github_client.list_issues(
                repo_parts[0], repo_parts[1], state="all"
            )
            
            async with get_db_session() as session:
                for gh_issue in github_issues:
                    try:
                        # Check if issue exists in database
                        result = await session.execute(
                            select(GitHubIssue).where(
                                and_(
                                    GitHubIssue.repository_id == repository.id,
                                    GitHubIssue.github_issue_number == gh_issue["number"]
                                )
                            )
                        )
                        existing_issue = result.scalar_one_or_none()
                        
                        if existing_issue:
                            # Update existing issue
                            updated = await self._update_issue_from_github(existing_issue, gh_issue)
                            if updated:
                                sync_result["issues_updated"] += 1
                        else:
                            # Create new issue
                            await self._create_issue_from_github(repository, gh_issue, session)
                            sync_result["issues_created"] += 1
                            
                        sync_result["issues_synced"] += 1
                        
                    except Exception as e:
                        error_msg = f"Failed to sync issue #{gh_issue['number']}: {str(e)}"
                        sync_result["errors"].append(error_msg)
                        logger.error(error_msg)
                        
                await session.commit()
                
            sync_result["success"] = True
            logger.info(f"Synced {sync_result['issues_synced']} issues from {repository.repository_full_name}")
            
        except Exception as e:
            sync_result["errors"].append(f"Repository sync failed: {str(e)}")
            logger.error(f"Failed to sync issues from {repository.repository_full_name}: {e}")
            
        return sync_result
        
    async def _create_issue_from_github(
        self,
        repository: GitHubRepository,
        gh_issue: Dict[str, Any],
        session: AsyncSession
    ) -> GitHubIssue:
        """Create database issue from GitHub issue data."""
        
        # Extract labels
        labels = [label["name"] for label in gh_issue.get("labels", [])]
        
        # Classify issue
        classification = self.classifier.classify_issue(
            gh_issue["title"],
            gh_issue.get("body", ""),
            labels
        )
        
        # Create issue
        issue = GitHubIssue(
            repository_id=repository.id,
            github_issue_number=gh_issue["number"],
            github_issue_id=gh_issue["id"],
            title=gh_issue["title"],
            description=gh_issue.get("body", ""),
            labels=labels,
            state=IssueState.OPEN if gh_issue["state"] == "open" else IssueState.CLOSED,
            priority=classification["priority"],
            issue_type=classification["issue_type"],
            estimated_effort=classification["effort_estimate"],
            issue_metadata={
                "github_url": gh_issue["html_url"],
                "classification": classification,
                "github_user": gh_issue.get("user", {}).get("login"),
                "created_at_github": gh_issue["created_at"],
                "updated_at_github": gh_issue["updated_at"]
            }
        )
        
        if gh_issue["state"] == "closed":
            issue.closed_at = datetime.fromisoformat(gh_issue["closed_at"].replace('Z', '+00:00'))
            
        session.add(issue)
        return issue
        
    async def _update_issue_from_github(self, issue: GitHubIssue, gh_issue: Dict[str, Any]) -> bool:
        """Update database issue from GitHub issue data."""
        
        updated = False
        
        # Check for changes
        if issue.title != gh_issue["title"]:
            issue.title = gh_issue["title"]
            updated = True
            
        if issue.description != gh_issue.get("body", ""):
            issue.description = gh_issue.get("body", "")
            updated = True
            
        new_labels = [label["name"] for label in gh_issue.get("labels", [])]
        if issue.labels != new_labels:
            issue.labels = new_labels
            updated = True
            
        new_state = IssueState.OPEN if gh_issue["state"] == "open" else IssueState.CLOSED
        if issue.state != new_state:
            issue.state = new_state
            if new_state == IssueState.CLOSED and not issue.closed_at:
                issue.closed_at = datetime.fromisoformat(gh_issue["closed_at"].replace('Z', '+00:00'))
            updated = True
            
        if updated:
            issue.updated_at = datetime.utcnow()
            
        return updated
        
    async def assign_issue_to_agent(
        self,
        issue_id: str,
        agent_id: str = None,
        auto_assign: bool = False
    ) -> Dict[str, Any]:
        """Assign issue to agent or find best agent automatically."""
        
        async with get_db_session() as session:
            # Get issue
            result = await session.execute(
                select(GitHubIssue).options(
                    selectinload(GitHubIssue.repository)
                ).where(GitHubIssue.id == uuid.UUID(issue_id))
            )
            issue = result.scalar_one_or_none()
            
            if not issue:
                raise IssueManagerError(f"Issue {issue_id} not found")
                
            if agent_id:
                # Assign to specific agent
                agent_result = await session.execute(
                    select(Agent).where(Agent.id == uuid.UUID(agent_id))
                )
                agent = agent_result.scalar_one_or_none()
                
                if not agent:
                    raise IssueManagerError(f"Agent {agent_id} not found")
                    
                assignment_result = {
                    "success": True,
                    "agent": agent,
                    "assignment_type": "manual"
                }
            else:
                # Auto-assign to best agent
                if not auto_assign:
                    raise IssueManagerError("Either agent_id must be provided or auto_assign must be True")
                    
                # Get required capabilities from classification
                classification = issue.issue_metadata.get("classification", {})
                required_capabilities = classification.get("required_capabilities", ["general_development"])
                
                # Find best agent
                match_result = await self.agent_matcher.find_best_agent(
                    issue, required_capabilities, str(issue.repository_id)
                )
                
                if not match_result:
                    raise IssueManagerError("No suitable agent found for this issue")
                    
                agent = match_result["agent"]
                assignment_result = {
                    "success": True,
                    "agent": agent,
                    "assignment_type": "automatic",
                    "matching_score": match_result["score"],
                    "alternatives": match_result["alternatives"]
                }
                
            # Perform assignment
            issue.assignee_agent_id = agent.id
            issue.assigned_at = datetime.utcnow()
            
            # Update GitHub
            try:
                repo_parts = issue.repository.repository_full_name.split('/')
                await self.github_client.update_issue(
                    repo_parts[0], repo_parts[1], issue.github_issue_number,
                    assignees=[f"agent-{agent.id}"]  # This would need proper GitHub user mapping
                )
                assignment_result["github_updated"] = True
            except Exception as e:
                logger.warning(f"Failed to update GitHub assignee: {e}")
                assignment_result["github_updated"] = False
                
            await session.commit()
            
            logger.info(f"Assigned issue #{issue.github_issue_number} to agent {agent.name}")
            return assignment_result
            
    async def update_issue_progress(
        self,
        issue_id: str,
        agent_id: str,
        status: str,
        comment: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Update issue progress with agent comment."""
        
        async with get_db_session() as session:
            # Get issue
            result = await session.execute(
                select(GitHubIssue).options(
                    selectinload(GitHubIssue.repository)
                ).where(
                    and_(
                        GitHubIssue.id == uuid.UUID(issue_id),
                        GitHubIssue.assignee_agent_id == uuid.UUID(agent_id)
                    )
                )
            )
            issue = result.scalar_one_or_none()
            
            if not issue:
                raise IssueManagerError(f"Issue {issue_id} not found or not assigned to agent {agent_id}")
                
            # Add progress update
            issue.add_progress_update(status, comment, agent_id)
            
            # Update effort tracking
            if context and "effort_used" in context:
                if not issue.actual_effort:
                    issue.actual_effort = 0
                issue.actual_effort += context["effort_used"]
                
            issue.updated_at = datetime.utcnow()
            
            # Add comment to GitHub
            try:
                repo_parts = issue.repository.repository_full_name.split('/')
                github_comment = f"**Agent Progress Update**\n\n**Status**: {status}\n\n{comment}\n\n---\n*Updated by Agent {agent_id}*"
                
                await self.github_client.add_issue_comment(
                    repo_parts[0], repo_parts[1], issue.github_issue_number, github_comment
                )
                github_updated = True
            except Exception as e:
                logger.warning(f"Failed to add GitHub comment: {e}")
                github_updated = False
                
            await session.commit()
            
            return {
                "success": True,
                "issue_id": issue_id,
                "status": status,
                "github_updated": github_updated,
                "progress_count": len(issue.progress_updates)
            }
            
    async def close_issue(
        self,
        issue_id: str,
        agent_id: str,
        resolution: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Close issue with resolution details."""
        
        async with get_db_session() as session:
            # Get issue
            result = await session.execute(
                select(GitHubIssue).options(
                    selectinload(GitHubIssue.repository)
                ).where(
                    and_(
                        GitHubIssue.id == uuid.UUID(issue_id),
                        GitHubIssue.assignee_agent_id == uuid.UUID(agent_id)
                    )
                )
            )
            issue = result.scalar_one_or_none()
            
            if not issue:
                raise IssueManagerError(f"Issue {issue_id} not found or not assigned to agent {agent_id}")
                
            # Update issue
            issue.state = IssueState.CLOSED
            issue.closed_at = datetime.utcnow()
            issue.add_progress_update("resolved", resolution, agent_id)
            
            # Update metadata with resolution
            if not issue.issue_metadata:
                issue.issue_metadata = {}
            issue.issue_metadata["resolution"] = {
                "resolved_by": agent_id,
                "resolution_time": datetime.utcnow().isoformat(),
                "resolution_description": resolution,
                "context": context or {}
            }
            
            # Close on GitHub
            try:
                repo_parts = issue.repository.repository_full_name.split('/')
                close_comment = f"**Issue Resolved**\n\n{resolution}\n\n---\n*Resolved by Agent {agent_id}*"
                
                # Add resolution comment
                await self.github_client.add_issue_comment(
                    repo_parts[0], repo_parts[1], issue.github_issue_number, close_comment
                )
                
                # Close the issue
                await self.github_client.update_issue(
                    repo_parts[0], repo_parts[1], issue.github_issue_number, state="closed"
                )
                
                github_updated = True
            except Exception as e:
                logger.warning(f"Failed to close GitHub issue: {e}")
                github_updated = False
                
            await session.commit()
            
            logger.info(f"Closed issue #{issue.github_issue_number} - resolved by agent {agent_id}")
            
            return {
                "success": True,
                "issue_id": issue_id,
                "closed_at": issue.closed_at.isoformat(),
                "github_updated": github_updated,
                "resolution": resolution
            }
            
    async def list_agent_issues(
        self,
        agent_id: str,
        state: IssueState = None,
        repository_id: str = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List issues assigned to specific agent."""
        
        async with get_db_session() as session:
            query = select(GitHubIssue).where(
                GitHubIssue.assignee_agent_id == uuid.UUID(agent_id)
            ).options(
                selectinload(GitHubIssue.repository)
            ).order_by(desc(GitHubIssue.updated_at))
            
            if state:
                query = query.where(GitHubIssue.state == state)
            if repository_id:
                query = query.where(GitHubIssue.repository_id == uuid.UUID(repository_id))
                
            query = query.limit(limit)
            
            result = await session.execute(query)
            issues = result.scalars().all()
            
            return [issue.to_dict() for issue in issues]
            
    async def get_issue_recommendations(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get issue recommendations for agent based on capabilities."""
        
        async with get_db_session() as session:
            # Get agent capabilities
            agent_result = await session.execute(
                select(Agent).where(Agent.id == uuid.UUID(agent_id))
            )
            agent = agent_result.scalar_one_or_none()
            
            if not agent or not agent.capabilities:
                return []
                
            # Get unassigned issues
            unassigned_result = await session.execute(
                select(GitHubIssue).where(
                    and_(
                        GitHubIssue.assignee_agent_id.is_(None),
                        GitHubIssue.state == IssueState.OPEN
                    )
                ).options(selectinload(GitHubIssue.repository))
            )
            unassigned_issues = unassigned_result.scalars().all()
            
            # Score and rank issues
            recommendations = []
            for issue in unassigned_issues:
                classification = issue.issue_metadata.get("classification", {})
                required_capabilities = classification.get("required_capabilities", ["general_development"])
                
                score = await self.agent_matcher._calculate_capability_score(agent, required_capabilities)
                
                if score > 0.3:  # Minimum threshold
                    recommendations.append({
                        "issue": issue.to_dict(),
                        "match_score": score,
                        "required_capabilities": required_capabilities,
                        "estimated_effort": issue.estimated_effort,
                        "priority": issue.priority
                    })
                    
            # Sort by score and priority
            recommendations.sort(key=lambda x: (x["match_score"], x["issue"]["priority"]), reverse=True)
            
            return recommendations[:limit]
            
    async def get_issue_statistics(self, repository_id: str = None, days: int = 30) -> Dict[str, Any]:
        """Get issue management statistics."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with get_db_session() as session:
            base_query = select(GitHubIssue).where(GitHubIssue.created_at >= cutoff_date)
            
            if repository_id:
                base_query = base_query.where(GitHubIssue.repository_id == uuid.UUID(repository_id))
                
            # Total issues
            total_result = await session.execute(base_query)
            total_issues = len(total_result.scalars().all())
            
            # Open issues
            open_result = await session.execute(
                base_query.where(GitHubIssue.state == IssueState.OPEN)
            )
            open_issues = len(open_result.scalars().all())
            
            # Closed issues
            closed_result = await session.execute(
                base_query.where(GitHubIssue.state == IssueState.CLOSED)
            )
            closed_issues = len(closed_result.scalars().all())
            
            # Assigned issues
            assigned_result = await session.execute(
                base_query.where(GitHubIssue.assignee_agent_id.isnot(None))
            )
            assigned_issues = len(assigned_result.scalars().all())
            
            # Average resolution time
            resolved_with_times = await session.execute(
                base_query.where(
                    and_(
                        GitHubIssue.state == IssueState.CLOSED,
                        GitHubIssue.assigned_at.isnot(None),
                        GitHubIssue.closed_at.isnot(None)
                    )
                )
            )
            resolved_issues = resolved_with_times.scalars().all()
            
            if resolved_issues:
                resolution_times = [
                    (issue.closed_at - issue.assigned_at).total_seconds() / 3600
                    for issue in resolved_issues
                ]
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
            else:
                avg_resolution_time = 0.0
                
            return {
                "period_days": days,
                "total_issues": total_issues,
                "open_issues": open_issues,
                "closed_issues": closed_issues,
                "assigned_issues": assigned_issues,
                "unassigned_issues": total_issues - assigned_issues,
                "assignment_rate": assigned_issues / total_issues if total_issues > 0 else 0.0,
                "resolution_rate": closed_issues / total_issues if total_issues > 0 else 0.0,
                "average_resolution_time_hours": avg_resolution_time,
                "issues_per_day": total_issues / days
            }