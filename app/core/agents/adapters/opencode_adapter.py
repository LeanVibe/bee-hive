"""OpenCode Adapter - TODO: Implement"""
from ..universal_agent_interface import *
from typing import *

class OpenCodeAdapter(UniversalAgentInterface):
    def __init__(self, agent_id: str, agent_type: AgentType = AgentType.OPENCODE):
        super().__init__(agent_id, agent_type)
    # TODO: Implement all abstract methods