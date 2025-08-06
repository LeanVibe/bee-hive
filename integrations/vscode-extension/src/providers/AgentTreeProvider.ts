import * as vscode from 'vscode';
import { HiveManager, Agent } from '../hive/HiveManager';

export class AgentTreeProvider implements vscode.TreeDataProvider<AgentTreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<AgentTreeItem | undefined | null | void> = new vscode.EventEmitter<AgentTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<AgentTreeItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor(private hiveManager: HiveManager) {
        hiveManager.on('statusChanged', () => {
            this.refresh();
        });
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: AgentTreeItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: AgentTreeItem): Thenable<AgentTreeItem[]> {
        if (!element) {
            // Root level - show all agents
            const status = this.hiveManager.getStatus();
            if (!status.connected) {
                return Promise.resolve([new AgentTreeItem(
                    'disconnected',
                    'Not Connected',
                    'Connect to Agent Hive to see agents',
                    'warning',
                    vscode.TreeItemCollapsibleState.None
                )]);
            }

            if (status.agents.length === 0) {
                return Promise.resolve([new AgentTreeItem(
                    'no-agents',
                    'No Agents',
                    'Create your first agent to get started',
                    'info',
                    vscode.TreeItemCollapsibleState.None
                )]);
            }

            return Promise.resolve(status.agents.map(agent => 
                new AgentTreeItem(
                    agent.id,
                    agent.name,
                    this.getAgentDescription(agent),
                    this.getAgentIcon(agent.status),
                    vscode.TreeItemCollapsibleState.Collapsed,
                    agent
                )
            ));
        } else if (element.agent) {
            // Agent details level
            const agent = element.agent;
            const details: AgentTreeItem[] = [
                new AgentTreeItem(
                    `${agent.id}-role`,
                    `Role: ${agent.role}`,
                    'Agent role and specialization',
                    'person',
                    vscode.TreeItemCollapsibleState.None
                ),
                new AgentTreeItem(
                    `${agent.id}-status`,
                    `Status: ${agent.status}`,
                    'Current agent status',
                    this.getAgentIcon(agent.status),
                    vscode.TreeItemCollapsibleState.None
                )
            ];

            if (agent.currentTask) {
                details.push(new AgentTreeItem(
                    `${agent.id}-task`,
                    `Current Task: ${agent.currentTask}`,
                    'Currently executing task',
                    'gear',
                    vscode.TreeItemCollapsibleState.None
                ));
            }

            if (agent.performance) {
                details.push(new AgentTreeItem(
                    `${agent.id}-performance`,
                    'Performance',
                    `Tasks: ${agent.performance.tasksCompleted} | Success: ${(agent.performance.successRate * 100).toFixed(1)}%`,
                    'graph',
                    vscode.TreeItemCollapsibleState.Collapsed
                ));
            }

            if (agent.capabilities && agent.capabilities.length > 0) {
                details.push(new AgentTreeItem(
                    `${agent.id}-capabilities`,
                    'Capabilities',
                    `${agent.capabilities.length} capabilities`,
                    'tools',
                    vscode.TreeItemCollapsibleState.Collapsed
                ));
            }

            return Promise.resolve(details);
        } else if (element.id.endsWith('-performance') && element.agent?.performance) {
            // Performance details
            const perf = element.agent.performance;
            return Promise.resolve([
                new AgentTreeItem(
                    `${element.agent.id}-tasks-completed`,
                    `Tasks Completed: ${perf.tasksCompleted}`,
                    '',
                    'check',
                    vscode.TreeItemCollapsibleState.None
                ),
                new AgentTreeItem(
                    `${element.agent.id}-avg-time`,
                    `Average Time: ${perf.averageTime.toFixed(1)}s`,
                    '',
                    'clock',
                    vscode.TreeItemCollapsibleState.None
                ),
                new AgentTreeItem(
                    `${element.agent.id}-success-rate`,
                    `Success Rate: ${(perf.successRate * 100).toFixed(1)}%`,
                    '',
                    'graph',
                    vscode.TreeItemCollapsibleState.None
                )
            ]);
        } else if (element.id.endsWith('-capabilities') && element.agent?.capabilities) {
            // Capabilities list
            return Promise.resolve(element.agent.capabilities.map(capability =>
                new AgentTreeItem(
                    `${element.agent!.id}-cap-${capability}`,
                    capability,
                    'Agent capability',
                    'symbol-method',
                    vscode.TreeItemCollapsibleState.None
                )
            ));
        }

        return Promise.resolve([]);
    }

    private getAgentDescription(agent: Agent): string {
        const parts = [`${agent.role} • ${agent.status}`];
        
        if (agent.currentTask) {
            parts.push(`Working on: ${agent.currentTask}`);
        }
        
        if (agent.performance) {
            parts.push(`${agent.performance.tasksCompleted} tasks completed`);
        }
        
        return parts.join(' | ');
    }

    private getAgentIcon(status: string): string {
        switch (status) {
            case 'active': return 'check-all';
            case 'idle': return 'circle-outline';
            case 'busy': return 'sync';
            case 'error': return 'error';
            case 'stopped': return 'stop-circle';
            default: return 'question';
        }
    }
}

export class AgentTreeItem extends vscode.TreeItem {
    constructor(
        public readonly id: string,
        public readonly label: string,
        public readonly tooltip: string,
        public readonly iconName: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly agent?: Agent
    ) {
        super(label, collapsibleState);
        
        this.tooltip = tooltip;
        this.contextValue = agent ? 'agent' : 'agent-detail';
        this.iconPath = new vscode.ThemeIcon(iconName);
        
        // Add commands for agents
        if (agent) {
            switch (agent.status) {
                case 'stopped':
                case 'error':
                    this.command = {
                        command: 'leanvibe.agent.start',
                        title: 'Start Agent',
                        arguments: [agent.id]
                    };
                    break;
                case 'active':
                case 'busy':
                    this.command = {
                        command: 'leanvibe.agent.stop',
                        title: 'Stop Agent', 
                        arguments: [agent.id]
                    };
                    break;
            }
        }
    }
}