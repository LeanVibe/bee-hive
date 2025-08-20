"""
Human-Friendly ID System for LeanVibe Agent Hive 2.0

Creates memorable, easy-to-type short IDs for all entities including agents.
Inspired by ant-farm patterns and optimized for developer productivity.

Examples:
- Agents: dev-01, qa-02, meta-03, arch-01
- Projects: web-app, mobile-ui, api-core
- Tasks: login-fix, db-opt, ui-tweak
- Sessions: dev-01-sess, qa-02-work
"""

import re
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json

class EntityType(Enum):
    """Enhanced entity types with human-friendly prefixes."""
    # Agents - Role-based short prefixes
    AGENT_DEVELOPER = "dev"       # Backend/fullstack developers
    AGENT_FRONTEND = "fe"         # Frontend specialists  
    AGENT_QA = "qa"              # QA engineers
    AGENT_DEVOPS = "ops"         # DevOps engineers
    AGENT_META = "meta"          # Meta/coordinator agents
    AGENT_ARCHITECT = "arch"     # Architecture specialists
    AGENT_DATA = "data"          # Data engineers
    AGENT_MOBILE = "mob"         # Mobile developers
    
    # Projects and Organization
    PROJECT = "proj"             # Projects: proj-web, proj-mobile
    EPIC = "epic"               # Epics: epic-auth, epic-ui
    PRD = "prd"                 # PRDs: prd-login, prd-dashboard
    TASK = "task"               # Tasks: task-fix, task-impl
    SPRINT = "sp"               # Sprints: sp-24w01, sp-24w02
    
    # Infrastructure
    SESSION = "sess"            # Tmux sessions: dev-01-sess
    WORKSPACE = "ws"            # Workspaces: ws-feature, ws-hotfix
    ENVIRONMENT = "env"         # Environments: env-prod, env-dev
    
    # Workflow
    WORKFLOW = "wf"             # Workflows: wf-deploy, wf-test
    PIPELINE = "pipe"           # Pipelines: pipe-ci, pipe-cd
    DEPLOYMENT = "deploy"       # Deployments: deploy-v1, deploy-hotfix

@dataclass
class HumanFriendlyID:
    """Represents a human-friendly ID with metadata."""
    prefix: str
    counter: int
    suffix: Optional[str] = None
    description: Optional[str] = None
    entity_type: EntityType = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def short_id(self) -> str:
        """Generate the short ID string."""
        base = f"{self.prefix}-{self.counter:02d}"
        if self.suffix:
            return f"{base}-{self.suffix}"
        return base
    
    @property
    def display_name(self) -> str:
        """Generate display name with description."""
        if self.description:
            return f"{self.short_id} ({self.description})"
        return self.short_id

class HumanFriendlyIDGenerator:
    """
    Generates human-friendly, memorable short IDs for all entities.
    
    Features:
    - Role-based agent IDs: dev-01, qa-02, meta-03
    - Descriptive project IDs: web-app, mobile-ui
    - Memorable task IDs: login-fix, db-opt
    - Easy tmux session IDs: dev-01-sess, qa-02-work
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".hive" / "friendly_ids.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Counter tracking for different prefixes
        self.counters: Dict[str, int] = {}
        self.id_registry: Dict[str, HumanFriendlyID] = {}
        self.reverse_lookup: Dict[str, str] = {}  # UUID -> friendly_id
        
        self._load_state()
    
    def _load_state(self):
        """Load saved state from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.counters = data.get('counters', {})
                    
                    # Reconstruct ID registry
                    for short_id, id_data in data.get('registry', {}).items():
                        friendly_id = HumanFriendlyID(
                            prefix=id_data['prefix'],
                            counter=id_data['counter'],
                            suffix=id_data.get('suffix'),
                            description=id_data.get('description'),
                            entity_type=EntityType(id_data['entity_type']) if id_data.get('entity_type') else None,
                            created_at=datetime.fromisoformat(id_data['created_at'])
                        )
                        self.id_registry[short_id] = friendly_id
                    
                    self.reverse_lookup = data.get('reverse_lookup', {})
            except Exception as e:
                print(f"Warning: Could not load friendly ID state: {e}")
    
    def _save_state(self):
        """Save current state to disk."""
        try:
            data = {
                'counters': self.counters,
                'registry': {
                    short_id: {
                        'prefix': fid.prefix,
                        'counter': fid.counter,
                        'suffix': fid.suffix,
                        'description': fid.description,
                        'entity_type': fid.entity_type.value if fid.entity_type else None,
                        'created_at': fid.created_at.isoformat()
                    }
                    for short_id, fid in self.id_registry.items()
                },
                'reverse_lookup': self.reverse_lookup
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save friendly ID state: {e}")
    
    def generate_agent_id(self, 
                         role: str, 
                         description: Optional[str] = None,
                         uuid_mapping: Optional[str] = None) -> str:
        """
        Generate human-friendly agent ID based on role.
        
        Args:
            role: Agent role (developer, qa, frontend, etc.)
            description: Optional description for the agent
            uuid_mapping: Optional UUID to map to this friendly ID
            
        Returns:
            Human-friendly agent ID like 'dev-01', 'qa-02', 'meta-03'
        """
        # Map roles to entity types
        role_mapping = {
            'developer': EntityType.AGENT_DEVELOPER,
            'backend-developer': EntityType.AGENT_DEVELOPER,
            'frontend-developer': EntityType.AGENT_FRONTEND,
            'frontend': EntityType.AGENT_FRONTEND,
            'qa-engineer': EntityType.AGENT_QA,
            'qa': EntityType.AGENT_QA,
            'devops-engineer': EntityType.AGENT_DEVOPS,
            'devops': EntityType.AGENT_DEVOPS,
            'meta-agent': EntityType.AGENT_META,
            'meta': EntityType.AGENT_META,
            'architect': EntityType.AGENT_ARCHITECT,
            'data-engineer': EntityType.AGENT_DATA,
            'data': EntityType.AGENT_DATA,
            'mobile-developer': EntityType.AGENT_MOBILE,
            'mobile': EntityType.AGENT_MOBILE,
        }
        
        entity_type = role_mapping.get(role.lower(), EntityType.AGENT_DEVELOPER)
        return self._generate_id(entity_type, description, uuid_mapping)
    
    def generate_project_id(self, 
                           name: str, 
                           description: Optional[str] = None,
                           uuid_mapping: Optional[str] = None) -> str:
        """
        Generate human-friendly project ID from name.
        
        Examples:
        - "Web Application" -> "web-app-01"
        - "Mobile UI Redesign" -> "mobile-ui-01"
        - "API Core Services" -> "api-core-01"
        """
        # Extract key words and create meaningful prefix
        words = re.findall(r'\b\w+\b', name.lower())
        
        # Create meaningful prefix from name
        if len(words) >= 2:
            prefix = f"{words[0][:3]}-{words[1][:3]}"
        else:
            prefix = words[0][:6] if words else "proj"
        
        # Clean up prefix
        prefix = re.sub(r'[^a-z0-9-]', '', prefix)
        
        return self._generate_id_with_prefix(prefix, EntityType.PROJECT, description, uuid_mapping)
    
    def generate_task_id(self, 
                        title: str, 
                        description: Optional[str] = None,
                        uuid_mapping: Optional[str] = None) -> str:
        """
        Generate memorable task ID from title.
        
        Examples:
        - "Fix login bug" -> "login-fix-01"
        - "Optimize database queries" -> "db-opt-01"
        - "Implement user interface" -> "ui-impl-01"
        """
        # Extract key action and object
        title_lower = title.lower()
        
        # Common action mappings
        action_patterns = {
            r'\b(fix|bug|error|issue)\b': 'fix',
            r'\b(implement|add|create|build)\b': 'impl',
            r'\b(optimize|improve|enhance)\b': 'opt',
            r'\b(update|modify|change)\b': 'upd',
            r'\b(test|testing|verify)\b': 'test',
            r'\b(deploy|deployment|release)\b': 'deploy',
            r'\b(refactor|clean|organize)\b': 'ref',
            r'\b(design|plan|architect)\b': 'design',
        }
        
        # Object/domain patterns
        object_patterns = {
            r'\b(login|auth|authentication)\b': 'login',
            r'\b(ui|interface|frontend)\b': 'ui',
            r'\b(db|database|data)\b': 'db',
            r'\b(api|endpoint|service)\b': 'api',
            r'\b(user|users|account)\b': 'user',
            r'\b(config|settings|environment)\b': 'config',
            r'\b(test|tests|testing)\b': 'test',
            r'\b(doc|docs|documentation)\b': 'docs',
        }
        
        # Find action and object
        action = 'task'
        for pattern, mapped_action in action_patterns.items():
            if re.search(pattern, title_lower):
                action = mapped_action
                break
        
        obj = None
        for pattern, mapped_obj in object_patterns.items():
            if re.search(pattern, title_lower):
                obj = mapped_obj
                break
        
        # Create prefix
        if obj:
            prefix = f"{obj}-{action}"
        else:
            # Fallback: use first significant word
            words = re.findall(r'\b\w{3,}\b', title_lower)
            if words:
                prefix = f"{words[0][:4]}-{action}"
            else:
                prefix = f"task-{action}"
        
        return self._generate_id_with_prefix(prefix, EntityType.TASK, description, uuid_mapping)
    
    def generate_session_id(self, agent_id: str, purpose: str = "work") -> str:
        """
        Generate tmux session ID for an agent.
        
        Examples:
        - dev-01 + "work" -> "dev-01-work"
        - qa-02 + "testing" -> "qa-02-test"
        """
        # Map common purposes to short suffixes
        purpose_mapping = {
            'work': 'work',
            'development': 'dev',
            'testing': 'test',
            'debugging': 'debug',
            'session': 'sess',
            'task': 'task',
        }
        
        suffix = purpose_mapping.get(purpose.lower(), purpose.lower()[:4])
        return f"{agent_id}-{suffix}"
    
    def _generate_id(self, 
                    entity_type: EntityType, 
                    description: Optional[str] = None,
                    uuid_mapping: Optional[str] = None) -> str:
        """Generate ID using entity type prefix."""
        return self._generate_id_with_prefix(entity_type.value, entity_type, description, uuid_mapping)
    
    def _generate_id_with_prefix(self, 
                                prefix: str, 
                                entity_type: EntityType,
                                description: Optional[str] = None,
                                uuid_mapping: Optional[str] = None) -> str:
        """Generate ID with custom prefix."""
        # Get next counter for this prefix
        self.counters[prefix] = self.counters.get(prefix, 0) + 1
        counter = self.counters[prefix]
        
        # Create friendly ID object
        friendly_id = HumanFriendlyID(
            prefix=prefix,
            counter=counter,
            description=description,
            entity_type=entity_type
        )
        
        short_id = friendly_id.short_id
        
        # Store in registry
        self.id_registry[short_id] = friendly_id
        
        # Map UUID if provided
        if uuid_mapping:
            self.reverse_lookup[uuid_mapping] = short_id
        
        # Save state
        self._save_state()
        
        return short_id
    
    def resolve_id(self, id_input: str) -> Optional[HumanFriendlyID]:
        """Resolve a friendly ID to its metadata."""
        # Exact match
        if id_input in self.id_registry:
            return self.id_registry[id_input]
        
        # Partial match
        matches = [
            short_id for short_id in self.id_registry.keys()
            if short_id.startswith(id_input.lower())
        ]
        
        if len(matches) == 1:
            return self.id_registry[matches[0]]
        
        return None
    
    def lookup_by_uuid(self, uuid_str: str) -> Optional[str]:
        """Lookup friendly ID by UUID."""
        return self.reverse_lookup.get(uuid_str)
    
    def list_by_type(self, entity_type: EntityType) -> List[HumanFriendlyID]:
        """List all IDs of a specific type."""
        return [
            fid for fid in self.id_registry.values()
            if fid.entity_type == entity_type
        ]
    
    def list_agents(self) -> List[HumanFriendlyID]:
        """List all agent IDs."""
        agent_types = [
            EntityType.AGENT_DEVELOPER, EntityType.AGENT_FRONTEND,
            EntityType.AGENT_QA, EntityType.AGENT_DEVOPS,
            EntityType.AGENT_META, EntityType.AGENT_ARCHITECT,
            EntityType.AGENT_DATA, EntityType.AGENT_MOBILE
        ]
        
        return [
            fid for fid in self.id_registry.values()
            if fid.entity_type in agent_types
        ]
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about ID generation."""
        stats = {}
        for entity_type in EntityType:
            count = len(self.list_by_type(entity_type))
            if count > 0:
                stats[entity_type.value] = count
        
        return stats

# Global instance
_id_generator: Optional[HumanFriendlyIDGenerator] = None

def get_id_generator() -> HumanFriendlyIDGenerator:
    """Get the global ID generator instance."""
    global _id_generator
    if _id_generator is None:
        _id_generator = HumanFriendlyIDGenerator()
    return _id_generator

# Convenience functions
def generate_agent_id(role: str, description: str = None, uuid_mapping: str = None) -> str:
    """Generate agent ID. Examples: dev-01, qa-02, meta-03"""
    return get_id_generator().generate_agent_id(role, description, uuid_mapping)

def generate_project_id(name: str, description: str = None, uuid_mapping: str = None) -> str:
    """Generate project ID. Examples: web-app-01, mobile-ui-01"""
    return get_id_generator().generate_project_id(name, description, uuid_mapping)

def generate_task_id(title: str, description: str = None, uuid_mapping: str = None) -> str:
    """Generate task ID. Examples: login-fix-01, db-opt-01"""
    return get_id_generator().generate_task_id(title, description, uuid_mapping)

def generate_session_id(agent_id: str, purpose: str = "work") -> str:
    """Generate session ID. Examples: dev-01-work, qa-02-test"""
    return get_id_generator().generate_session_id(agent_id, purpose)

def resolve_friendly_id(id_input: str) -> Optional[HumanFriendlyID]:
    """Resolve friendly ID with partial matching."""
    return get_id_generator().resolve_id(id_input)