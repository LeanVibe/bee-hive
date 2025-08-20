#!/usr/bin/env python3
"""
Project Management System Setup Script

This script sets up the comprehensive project management system for
LeanVibe Agent Hive 2.0, including database migrations, initial data,
and system integration.
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import List, Optional

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import init_database, get_db_session
from app.core.kanban_state_machine import KanbanStateMachine
from app.services.project_management_service import ProjectManagementService
from app.models.project_management import Project, Epic, PRD, Task, KanbanState
from app.models.agent import Agent, AgentStatus, AgentType
from app.core.short_id_generator import get_generator, EntityType
from app.core.logging_service import get_component_logger

logger = get_component_logger("project_management_setup")


class ProjectManagementSetup:
    """Setup manager for project management system."""
    
    def __init__(self):
        self.db_session = None
        self.project_service = None
        self.kanban_machine = None
        self.short_id_generator = None
    
    async def initialize(self):
        """Initialize database and services."""
        try:
            logger.info("🚀 Initializing Project Management System...")
            
            # Initialize database
            await init_database()
            self.db_session = next(get_db_session())
            
            # Initialize services
            self.project_service = ProjectManagementService(self.db_session)
            self.kanban_machine = KanbanStateMachine(self.db_session)
            self.short_id_generator = get_generator()
            
            logger.info("✅ System initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            raise
    
    def create_demo_agents(self) -> List[Agent]:
        """Create demo agents for testing."""
        logger.info("👥 Creating demo agents...")
        
        agents_data = [
            {
                "name": "Project Manager Agent",
                "agent_type": AgentType.CLAUDE,
                "capabilities": ["project_management", "coordination", "planning"],
                "description": "Specialized in project coordination and planning"
            },
            {
                "name": "Backend Developer Agent", 
                "agent_type": AgentType.CLAUDE,
                "capabilities": ["python", "fastapi", "database", "backend_development"],
                "description": "Backend development and API creation"
            },
            {
                "name": "Frontend Developer Agent",
                "agent_type": AgentType.CLAUDE, 
                "capabilities": ["react", "typescript", "ui_ux", "frontend_development"],
                "description": "Frontend development and user interfaces"
            },
            {
                "name": "DevOps Engineer Agent",
                "agent_type": AgentType.CLAUDE,
                "capabilities": ["docker", "kubernetes", "cicd", "infrastructure"],
                "description": "Infrastructure and deployment automation"
            },
            {
                "name": "QA Testing Agent",
                "agent_type": AgentType.CLAUDE,
                "capabilities": ["testing", "quality_assurance", "automation"],
                "description": "Quality assurance and automated testing"
            }
        ]
        
        agents = []
        for agent_data in agents_data:
            agent = Agent(
                name=agent_data["name"],
                status=AgentStatus.ACTIVE,
                agent_type=agent_data["agent_type"],
                capabilities=agent_data["capabilities"],
                current_context_tokens=0,
                max_context_tokens=200000,
                metadata={"description": agent_data["description"]}
            )
            
            self.db_session.add(agent)
            agents.append(agent)
        
        self.db_session.commit()
        logger.info(f"✅ Created {len(agents)} demo agents")
        return agents
    
    def create_demo_project_hierarchy(self, agents: List[Agent]) -> Project:
        """Create a comprehensive demo project hierarchy."""
        logger.info("🏗️ Creating demo project hierarchy...")
        
        pm_agent = agents[0]  # Project Manager
        backend_agent = agents[1]
        frontend_agent = agents[2]
        devops_agent = agents[3]
        qa_agent = agents[4]
        
        # Create project with template
        project, initial_epics = self.project_service.create_project_with_initial_structure(
            name="LeanVibe Agent Hive 2.0 Enhancement",
            description="Comprehensive enhancement of the agent coordination platform with advanced project management capabilities",
            template_type="web_application",
            owner_agent_id=pm_agent.id,
            start_date=None,
            tags=["platform", "agents", "project-management", "coordination"]
        )
        
        logger.info(f"✅ Created project: {project.get_display_id()}")
        
        # Enhance epics with detailed PRDs and tasks
        for epic in initial_epics[:3]:  # Focus on first 3 epics
            logger.info(f"📋 Enhancing epic: {epic.get_display_id()}")
            
            # Create PRDs for each epic
            if "Authentication" in epic.name:
                self._create_auth_prd_and_tasks(epic, backend_agent, qa_agent)
            elif "Core Features" in epic.name:
                self._create_core_features_prd_and_tasks(epic, backend_agent, frontend_agent, qa_agent)
            elif "User Interface" in epic.name:
                self._create_ui_prd_and_tasks(epic, frontend_agent, qa_agent)
        
        # Move some items through workflow to show progress
        self._simulate_project_progress(project)
        
        return project
    
    def _create_auth_prd_and_tasks(self, epic: Epic, backend_agent: Agent, qa_agent: Agent):
        """Create authentication PRD and tasks."""
        prd = PRD(
            title="Enhanced Authentication & Authorization System",
            description="Implement comprehensive authentication with role-based access control",
            epic_id=epic.id,
            priority=epic.priority,
            complexity_score=8,
            estimated_effort_days=12,
            requirements=[
                "Multi-factor authentication support",
                "Role-based access control (RBAC)",
                "OAuth integration for external providers",
                "Session management and security",
                "API key management for agents"
            ],
            technical_requirements=[
                "JWT token-based authentication",
                "Redis session storage",
                "Password hashing with bcrypt",
                "Rate limiting for auth endpoints",
                "Audit logging for security events"
            ],
            acceptance_criteria=[
                "Users can authenticate with username/password",
                "MFA can be enabled/disabled per user",
                "Role permissions are enforced across all endpoints",
                "OAuth flows work with Google and GitHub",
                "API keys can be generated and revoked"
            ]
        )
        prd.ensure_short_id(self.db_session)
        self.db_session.add(prd)
        self.db_session.flush()
        
        # Create specific tasks
        tasks_data = [
            {
                "title": "Design authentication database schema",
                "type": "ARCHITECTURE",
                "agent": backend_agent,
                "effort": 180,
                "priority": "HIGH"
            },
            {
                "title": "Implement JWT token service",
                "type": "FEATURE_DEVELOPMENT", 
                "agent": backend_agent,
                "effort": 240,
                "priority": "HIGH"
            },
            {
                "title": "Create user registration endpoint",
                "type": "FEATURE_DEVELOPMENT",
                "agent": backend_agent,
                "effort": 120,
                "priority": "HIGH"
            },
            {
                "title": "Implement MFA with TOTP",
                "type": "FEATURE_DEVELOPMENT",
                "agent": backend_agent,
                "effort": 300,
                "priority": "MEDIUM"
            },
            {
                "title": "Add OAuth provider integration",
                "type": "FEATURE_DEVELOPMENT",
                "agent": backend_agent,
                "effort": 360,
                "priority": "MEDIUM"
            },
            {
                "title": "Create authentication API tests",
                "type": "TESTING",
                "agent": qa_agent,
                "effort": 180,
                "priority": "HIGH"
            },
            {
                "title": "Security penetration testing",
                "type": "SECURITY",
                "agent": qa_agent,
                "effort": 240,
                "priority": "HIGH"
            }
        ]
        
        self._create_tasks_from_data(prd, tasks_data)
    
    def _create_core_features_prd_and_tasks(self, epic: Epic, backend_agent: Agent, frontend_agent: Agent, qa_agent: Agent):
        """Create core features PRD and tasks."""
        prd = PRD(
            title="Advanced Agent Coordination Features",
            description="Implement sophisticated agent coordination and workflow management",
            epic_id=epic.id,
            priority=epic.priority,
            complexity_score=9,
            estimated_effort_days=20,
            requirements=[
                "Real-time agent communication",
                "Intelligent task routing and assignment",
                "Workflow automation and orchestration", 
                "Performance monitoring and analytics",
                "Resource optimization and load balancing"
            ],
            technical_requirements=[
                "WebSocket-based real-time communication",
                "Redis Streams for event processing",
                "Machine learning for task routing",
                "Prometheus metrics collection",
                "Horizontal scaling architecture"
            ],
            acceptance_criteria=[
                "Agents can communicate in real-time",
                "Tasks are automatically routed to best agents",
                "Workflows can be defined and automated",
                "Performance metrics are collected and visualized",
                "System automatically balances agent workloads"
            ]
        )
        prd.ensure_short_id(self.db_session)
        self.db_session.add(prd)
        self.db_session.flush()
        
        tasks_data = [
            {
                "title": "Design agent communication protocol",
                "type": "ARCHITECTURE",
                "agent": backend_agent,
                "effort": 240,
                "priority": "CRITICAL"
            },
            {
                "title": "Implement WebSocket communication hub",
                "type": "FEATURE_DEVELOPMENT",
                "agent": backend_agent, 
                "effort": 360,
                "priority": "HIGH"
            },
            {
                "title": "Build intelligent task router",
                "type": "FEATURE_DEVELOPMENT",
                "agent": backend_agent,
                "effort": 480,
                "priority": "HIGH"
            },
            {
                "title": "Create workflow orchestration engine",
                "type": "FEATURE_DEVELOPMENT",
                "agent": backend_agent,
                "effort": 600,
                "priority": "HIGH"
            },
            {
                "title": "Implement performance monitoring",
                "type": "FEATURE_DEVELOPMENT",
                "agent": backend_agent,
                "effort": 300,
                "priority": "MEDIUM"
            },
            {
                "title": "Build agent coordination dashboard",
                "type": "FEATURE_DEVELOPMENT",
                "agent": frontend_agent,
                "effort": 480,
                "priority": "MEDIUM"
            },
            {
                "title": "Create load testing framework",
                "type": "TESTING",
                "agent": qa_agent,
                "effort": 360,
                "priority": "MEDIUM"
            },
            {
                "title": "End-to-end coordination testing",
                "type": "TESTING",
                "agent": qa_agent,
                "effort": 240,
                "priority": "HIGH"
            }
        ]
        
        self._create_tasks_from_data(prd, tasks_data)
    
    def _create_ui_prd_and_tasks(self, epic: Epic, frontend_agent: Agent, qa_agent: Agent):
        """Create UI/UX PRD and tasks.""" 
        prd = PRD(
            title="Modern Agent Management Interface",
            description="Create intuitive and responsive interface for agent management",
            epic_id=epic.id,
            priority=epic.priority,
            complexity_score=6,
            estimated_effort_days=15,
            requirements=[
                "Responsive design for desktop and mobile",
                "Real-time dashboard with live updates",
                "Agent status and performance visualization",
                "Task management interface with Kanban boards",
                "Dark/light theme support"
            ],
            technical_requirements=[
                "React 18 with TypeScript",
                "TailwindCSS for styling",
                "WebSocket integration for real-time updates",
                "Chart.js for data visualization",
                "Progressive Web App (PWA) support"
            ],
            acceptance_criteria=[
                "Interface works seamlessly on mobile and desktop",
                "Dashboard updates in real-time without refresh",
                "Users can manage agents and tasks intuitively",
                "Performance charts are clear and informative",
                "Theme switching works without page reload"
            ]
        )
        prd.ensure_short_id(self.db_session)
        self.db_session.add(prd)
        self.db_session.flush()
        
        tasks_data = [
            {
                "title": "Design UI/UX mockups and wireframes",
                "type": "PLANNING",
                "agent": frontend_agent,
                "effort": 240,
                "priority": "HIGH"
            },
            {
                "title": "Set up React TypeScript project structure",
                "type": "ARCHITECTURE",
                "agent": frontend_agent,
                "effort": 120,
                "priority": "HIGH"
            },
            {
                "title": "Implement responsive layout components",
                "type": "FEATURE_DEVELOPMENT",
                "agent": frontend_agent,
                "effort": 300,
                "priority": "HIGH"
            },
            {
                "title": "Build agent dashboard with real-time updates",
                "type": "FEATURE_DEVELOPMENT",
                "agent": frontend_agent,
                "effort": 420,
                "priority": "HIGH"
            },
            {
                "title": "Create Kanban board interface",
                "type": "FEATURE_DEVELOPMENT",
                "agent": frontend_agent,
                "effort": 360,
                "priority": "MEDIUM"
            },
            {
                "title": "Implement data visualization charts",
                "type": "FEATURE_DEVELOPMENT",
                "agent": frontend_agent,
                "effort": 240,
                "priority": "MEDIUM"
            },
            {
                "title": "Add dark/light theme support",
                "type": "FEATURE_DEVELOPMENT",
                "agent": frontend_agent,
                "effort": 180,
                "priority": "LOW"
            },
            {
                "title": "Cross-browser compatibility testing",
                "type": "TESTING",
                "agent": qa_agent,
                "effort": 180,
                "priority": "MEDIUM"
            },
            {
                "title": "Mobile responsiveness testing",
                "type": "TESTING",
                "agent": qa_agent,
                "effort": 120,
                "priority": "MEDIUM"
            }
        ]
        
        self._create_tasks_from_data(prd, tasks_data)
    
    def _create_tasks_from_data(self, prd: PRD, tasks_data: List[dict]):
        """Create tasks from task data."""
        from app.models.project_management import TaskType, TaskPriority
        
        for task_data in tasks_data:
            task = Task(
                title=task_data["title"],
                prd_id=prd.id,
                task_type=getattr(TaskType, task_data["type"]),
                priority=getattr(TaskPriority, task_data["priority"]),
                estimated_effort_minutes=task_data["effort"],
                assigned_agent_id=task_data["agent"].id,
                kanban_state=KanbanState.BACKLOG
            )
            task.ensure_short_id(self.db_session)
            self.db_session.add(task)
        
        self.db_session.commit()
    
    def _simulate_project_progress(self, project: Project):
        """Simulate some progress in the project workflow."""
        logger.info("⚡ Simulating project progress...")
        
        # Move project to active
        result = self.kanban_machine.transition_entity_state(
            project, KanbanState.IN_PROGRESS, 
            reason="Project kickoff - moving to active development"
        )
        
        if result.success:
            logger.info(f"✅ Project moved to IN_PROGRESS")
        
        # Move some epics forward
        for epic in project.epics[:2]:
            result = self.kanban_machine.transition_entity_state(
                epic, KanbanState.IN_PROGRESS,
                reason="Epic development started"
            )
            
            # Move some PRDs forward
            for prd in epic.prds[:1]:
                result = self.kanban_machine.transition_entity_state(
                    prd, KanbanState.IN_PROGRESS,
                    reason="PRD implementation started"
                )
                
                # Complete some tasks
                tasks = prd.tasks[:3]  # First 3 tasks
                for i, task in enumerate(tasks):
                    if i == 0:
                        # Complete first task
                        self.kanban_machine.transition_entity_state(task, KanbanState.READY, reason="Ready to start")
                        self.kanban_machine.transition_entity_state(task, KanbanState.IN_PROGRESS, reason="Work started")
                        self.kanban_machine.transition_entity_state(task, KanbanState.REVIEW, reason="Ready for review")
                        self.kanban_machine.transition_entity_state(task, KanbanState.DONE, reason="Completed")
                    elif i == 1:
                        # Second task in progress
                        self.kanban_machine.transition_entity_state(task, KanbanState.READY, reason="Ready to start")
                        self.kanban_machine.transition_entity_state(task, KanbanState.IN_PROGRESS, reason="Work started")
                    else:
                        # Third task ready
                        self.kanban_machine.transition_entity_state(task, KanbanState.READY, reason="Ready to start")
        
        self.db_session.commit()
        logger.info("✅ Project progress simulation complete")
    
    def display_project_summary(self, project: Project):
        """Display a summary of the created project."""
        logger.info("\n" + "="*60)
        logger.info("🎉 PROJECT MANAGEMENT SYSTEM SETUP COMPLETE!")
        logger.info("="*60)
        
        stats = self.project_service.get_project_hierarchy_stats(project.id)
        
        print(f"""
📊 PROJECT SUMMARY
==================
Project: {project.name} ({project.get_display_id()})
Status: {project.status.value} | State: {project.kanban_state.value}

📈 HIERARCHY STATS
==================
• Epics: {stats.epic_count} (Completed: {stats.completed_epics})
• PRDs: {stats.prd_count} (Completed: {stats.completed_prds})  
• Tasks: {stats.task_count} (Completed: {stats.completed_tasks})

📋 TASK STATE DISTRIBUTION
=========================
• Backlog: {stats.state_distribution[KanbanState.BACKLOG]}
• Ready: {stats.state_distribution[KanbanState.READY]}
• In Progress: {stats.state_distribution[KanbanState.IN_PROGRESS]}
• Review: {stats.state_distribution[KanbanState.REVIEW]} 
• Done: {stats.state_distribution[KanbanState.DONE]}

🚀 NEXT STEPS
==============
1. Run: `hive project list` to see all projects
2. Run: `hive epic list --project {project.get_display_id()}` to see epics
3. Run: `hive task list --state in_progress` to see active tasks
4. Run: `hive board show task --project {project.get_display_id()}` to see Kanban board

💡 TIP: Use short IDs (like {project.get_display_id()}) in CLI commands for easier navigation!
        """)
    
    async def run_setup(self):
        """Run the complete setup process."""
        try:
            await self.initialize()
            
            # Create demo agents
            agents = self.create_demo_agents()
            
            # Create demo project hierarchy
            project = self.create_demo_project_hierarchy(agents)
            
            # Display summary
            self.display_project_summary(project)
            
            logger.info("✅ Project Management System setup completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise
        finally:
            if self.db_session:
                self.db_session.close()


async def main():
    """Main setup function."""
    print("🚀 LeanVibe Agent Hive 2.0 - Project Management System Setup")
    print("=" * 65)
    
    setup = ProjectManagementSetup()
    await setup.run_setup()


if __name__ == "__main__":
    asyncio.run(main())