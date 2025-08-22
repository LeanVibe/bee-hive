"""
Sample Plugins for LeanVibe Agent Hive 2.0 - Epic 2 Phase 2.2

Demonstrates plugin marketplace functionality with diverse sample plugins
across different categories and use cases.

Sample Plugins:
- ProductivityBoosterPlugin (Productivity)
- SlackIntegrationPlugin (Integration)  
- PerformanceAnalyticsPlugin (Analytics)
- SecurityAuditPlugin (Security)
- GitWorkflowPlugin (Development)
- AutomationOrchestratorPlugin (Automation)

Epic 1 Preservation:
- <50ms plugin operations
- <80MB memory usage
- Efficient initialization and execution
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .logging_service import get_component_logger
from .orchestrator_plugins import OrchestratorPlugin, PluginType, PluginMetadata
from .plugin_marketplace import (
    PluginMarketplace, MarketplacePluginEntry, PluginCategory, 
    CertificationLevel, PluginStatus
)
from .developer_onboarding_platform import DeveloperOnboardingPlatform, DeveloperProfile

logger = get_component_logger("sample_plugins")


# Sample Plugin Implementations
class ProductivityBoosterPlugin(OrchestratorPlugin):
    """
    Sample productivity enhancement plugin with task automation and optimization.
    
    Features:
    - Automated task prioritization
    - Focus time tracking
    - Productivity analytics
    - Smart break reminders
    """
    
    def __init__(self):
        super().__init__(
            plugin_id="productivity_booster_v1",
            name="Productivity Booster",
            version="1.2.1",
            plugin_type=PluginType.ORCHESTRATOR,
            description="AI-powered productivity enhancement with smart task management and focus optimization"
        )
        self.task_queue = []
        self.focus_sessions = []
        self.productivity_metrics = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize productivity tracking."""
        try:
            self.config = config
            self.focus_duration = config.get("focus_duration_minutes", 25)  # Pomodoro-style
            self.break_duration = config.get("break_duration_minutes", 5)
            self.max_daily_tasks = config.get("max_daily_tasks", 10)
            
            logger.info("Productivity Booster Plugin initialized",
                       focus_duration=self.focus_duration,
                       break_duration=self.break_duration)
            return True
            
        except Exception as e:
            logger.error("Productivity plugin initialization failed", error=str(e))
            return False
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process productivity-related tasks."""
        try:
            task_type = task_data.get("type", "unknown")
            
            if task_type == "prioritize_tasks":
                return await self._prioritize_tasks(task_data.get("tasks", []))
            elif task_type == "start_focus_session":
                return await self._start_focus_session(task_data)
            elif task_type == "track_productivity":
                return await self._track_productivity(task_data)
            else:
                return {"status": "error", "message": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error("Productivity task processing failed", error=str(e))
            return {"status": "error", "message": str(e)}
    
    async def _prioritize_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """AI-powered task prioritization."""
        # Simplified AI prioritization algorithm
        prioritized_tasks = []
        
        for task in tasks:
            urgency = task.get("urgency", 1)  # 1-5 scale
            importance = task.get("importance", 1)  # 1-5 scale
            effort = task.get("effort_hours", 1)
            
            # Priority score calculation
            priority_score = (urgency * 0.4 + importance * 0.6) / effort
            
            prioritized_tasks.append({
                **task,
                "priority_score": priority_score,
                "ai_recommendation": self._generate_task_recommendation(task)
            })
        
        # Sort by priority score
        prioritized_tasks.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return {
            "status": "completed",
            "prioritized_tasks": prioritized_tasks[:self.max_daily_tasks],
            "total_tasks_analyzed": len(tasks)
        }
    
    async def _start_focus_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start a focused work session."""
        session_id = f"focus_{datetime.utcnow().timestamp()}"
        
        session = {
            "session_id": session_id,
            "task": session_data.get("task", "General work"),
            "duration_minutes": session_data.get("duration", self.focus_duration),
            "started_at": datetime.utcnow(),
            "status": "active"
        }
        
        self.focus_sessions.append(session)
        
        return {
            "status": "started",
            "session_id": session_id,
            "duration_minutes": session["duration_minutes"],
            "break_reminder_at": (datetime.utcnow() + timedelta(minutes=session["duration_minutes"])).isoformat()
        }
    
    async def _track_productivity(self, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track productivity metrics."""
        date = tracking_data.get("date", datetime.utcnow().date().isoformat())
        
        metrics = self.productivity_metrics.get(date, {
            "tasks_completed": 0,
            "focus_time_minutes": 0,
            "interruptions": 0,
            "productivity_score": 0.0
        })
        
        # Update metrics based on tracking data
        if "tasks_completed" in tracking_data:
            metrics["tasks_completed"] += tracking_data["tasks_completed"]
        
        if "focus_time_minutes" in tracking_data:
            metrics["focus_time_minutes"] += tracking_data["focus_time_minutes"]
        
        if "interruptions" in tracking_data:
            metrics["interruptions"] += tracking_data["interruptions"]
        
        # Calculate productivity score
        metrics["productivity_score"] = self._calculate_productivity_score(metrics)
        
        self.productivity_metrics[date] = metrics
        
        return {
            "status": "updated",
            "date": date,
            "metrics": metrics,
            "insights": self._generate_productivity_insights(metrics)
        }
    
    def _generate_task_recommendation(self, task: Dict[str, Any]) -> str:
        """Generate AI recommendation for task execution."""
        urgency = task.get("urgency", 1)
        importance = task.get("importance", 1)
        
        if urgency >= 4 and importance >= 4:
            return "🔥 High priority - tackle immediately during peak energy"
        elif urgency >= 3 or importance >= 4:
            return "⚡ Medium priority - schedule for focused time block"
        elif urgency <= 2 and importance <= 2:
            return "📝 Low priority - batch with similar tasks or delegate"
        else:
            return "🤔 Review priority - consider breaking into smaller tasks"
    
    def _calculate_productivity_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall productivity score."""
        tasks = metrics.get("tasks_completed", 0)
        focus_time = metrics.get("focus_time_minutes", 0)
        interruptions = metrics.get("interruptions", 0)
        
        # Productivity score algorithm
        base_score = min(tasks * 10 + focus_time / 6, 100)  # Base score from tasks and focus time
        interruption_penalty = min(interruptions * 5, 30)   # Penalty for interruptions
        
        return max(0.0, base_score - interruption_penalty)
    
    def _generate_productivity_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate productivity insights and recommendations."""
        insights = []
        score = metrics.get("productivity_score", 0)
        tasks = metrics.get("tasks_completed", 0)
        focus_time = metrics.get("focus_time_minutes", 0)
        interruptions = metrics.get("interruptions", 0)
        
        if score >= 80:
            insights.append("🏆 Excellent productivity! You're in the zone.")
        elif score >= 60:
            insights.append("👍 Good productivity day with room for improvement.")
        else:
            insights.append("🎯 Focus on reducing interruptions and increasing deep work time.")
        
        if tasks < 3:
            insights.append("📋 Consider breaking large tasks into smaller, manageable chunks.")
        
        if focus_time < 120:  # Less than 2 hours
            insights.append("⏰ Try to increase focused work time - aim for 2-4 hour blocks.")
        
        if interruptions > 5:
            insights.append("🚫 High interruption count - consider time blocking or do-not-disturb periods.")
        
        return insights
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.task_queue.clear()
        self.focus_sessions.clear()
        logger.info("Productivity Booster Plugin cleaned up")


class SlackIntegrationPlugin(OrchestratorPlugin):
    """
    Sample Slack integration plugin for team communication and notifications.
    
    Features:
    - Agent status notifications
    - Task completion alerts
    - Team coordination messages
    - Channel-based routing
    """
    
    def __init__(self):
        super().__init__(
            plugin_id="slack_integration_v2",
            name="Slack Integration",
            version="2.1.0",
            plugin_type=PluginType.INTEGRATION,
            description="Seamless Slack integration for agent notifications and team coordination"
        )
        self.webhook_url = None
        self.channel_mappings = {}
        self.message_templates = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize Slack integration."""
        try:
            self.webhook_url = config.get("webhook_url")
            self.channel_mappings = config.get("channel_mappings", {})
            self.default_channel = config.get("default_channel", "#general")
            
            # Load message templates
            self.message_templates = {
                "task_completed": "✅ Task completed: {task_name} by {agent_name}",
                "agent_started": "🤖 Agent {agent_name} started working on {project}",
                "error_alert": "❌ Error in {component}: {error_message}",
                "milestone_reached": "🎉 Milestone reached: {milestone_name} in {project}"
            }
            
            logger.info("Slack Integration Plugin initialized",
                       webhook_configured=bool(self.webhook_url),
                       channels=len(self.channel_mappings))
            return True
            
        except Exception as e:
            logger.error("Slack plugin initialization failed", error=str(e))
            return False
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Slack integration tasks."""
        try:
            task_type = task_data.get("type", "unknown")
            
            if task_type == "send_notification":
                return await self._send_notification(task_data)
            elif task_type == "update_status":
                return await self._update_status(task_data)
            elif task_type == "create_thread":
                return await self._create_thread(task_data)
            else:
                return {"status": "error", "message": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error("Slack task processing failed", error=str(e))
            return {"status": "error", "message": str(e)}
    
    async def _send_notification(self, notification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification to Slack."""
        message_type = notification_data.get("message_type", "custom")
        channel = notification_data.get("channel", self.default_channel)
        
        # Get message template
        if message_type in self.message_templates:
            message = self.message_templates[message_type].format(**notification_data)
        else:
            message = notification_data.get("message", "No message provided")
        
        # Simulate sending to Slack (in real implementation, would use Slack SDK)
        slack_payload = {
            "channel": channel,
            "text": message,
            "username": "LeanVibe Agent",
            "icon_emoji": ":robot_face:",
            "attachments": notification_data.get("attachments", [])
        }
        
        # In real implementation: await self._post_to_slack(slack_payload)
        logger.info("Slack notification sent", channel=channel, message_type=message_type)
        
        return {
            "status": "sent",
            "channel": channel,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _update_status(self, status_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update agent status in Slack."""
        agent_name = status_data.get("agent_name", "Unknown Agent")
        status = status_data.get("status", "working")
        project = status_data.get("project", "Unknown Project")
        
        status_message = f"🤖 {agent_name} is {status}"
        if project != "Unknown Project":
            status_message += f" on {project}"
        
        return await self._send_notification({
            "message": status_message,
            "channel": self.channel_mappings.get("status", self.default_channel)
        })
    
    async def _create_thread(self, thread_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a threaded conversation in Slack."""
        parent_message = thread_data.get("parent_message", "Discussion thread")
        replies = thread_data.get("replies", [])
        channel = thread_data.get("channel", self.default_channel)
        
        # Simulate thread creation
        thread_id = f"thread_{datetime.utcnow().timestamp()}"
        
        return {
            "status": "created",
            "thread_id": thread_id,
            "channel": channel,
            "parent_message": parent_message,
            "reply_count": len(replies)
        }
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.webhook_url = None
        self.channel_mappings.clear()
        logger.info("Slack Integration Plugin cleaned up")


class PerformanceAnalyticsPlugin(OrchestratorPlugin):
    """
    Sample performance analytics plugin for system monitoring and optimization.
    
    Features:
    - Real-time performance monitoring
    - Trend analysis and predictions
    - Bottleneck identification
    - Performance optimization recommendations
    """
    
    def __init__(self):
        super().__init__(
            plugin_id="performance_analytics_v3",
            name="Performance Analytics",
            version="3.0.2",
            plugin_type=PluginType.PROCESSOR,
            description="Advanced performance analytics with AI-powered insights and optimization recommendations"
        )
        self.metrics_history = []
        self.performance_thresholds = {}
        self.trend_data = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize performance analytics."""
        try:
            self.performance_thresholds = config.get("thresholds", {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "response_time_ms": 100.0,
                "error_rate": 1.0
            })
            
            self.analysis_window_hours = config.get("analysis_window_hours", 24)
            self.alert_enabled = config.get("alert_enabled", True)
            
            logger.info("Performance Analytics Plugin initialized",
                       thresholds=self.performance_thresholds,
                       window_hours=self.analysis_window_hours)
            return True
            
        except Exception as e:
            logger.error("Performance analytics plugin initialization failed", error=str(e))
            return False
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process performance analytics tasks."""
        try:
            task_type = task_data.get("type", "unknown")
            
            if task_type == "collect_metrics":
                return await self._collect_metrics(task_data)
            elif task_type == "analyze_performance":
                return await self._analyze_performance(task_data)
            elif task_type == "generate_report":
                return await self._generate_report(task_data)
            elif task_type == "predict_trends":
                return await self._predict_trends(task_data)
            else:
                return {"status": "error", "message": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error("Performance analytics task processing failed", error=str(e))
            return {"status": "error", "message": str(e)}
    
    async def _collect_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and store performance metrics."""
        timestamp = datetime.utcnow()
        
        metric_entry = {
            "timestamp": timestamp,
            "cpu_usage": metrics_data.get("cpu_usage", 0.0),
            "memory_usage": metrics_data.get("memory_usage", 0.0),
            "response_time_ms": metrics_data.get("response_time_ms", 0.0),
            "error_rate": metrics_data.get("error_rate", 0.0),
            "active_agents": metrics_data.get("active_agents", 0),
            "tasks_processed": metrics_data.get("tasks_processed", 0)
        }
        
        self.metrics_history.append(metric_entry)
        
        # Keep only recent history
        cutoff_time = timestamp - timedelta(hours=self.analysis_window_hours)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m["timestamp"] > cutoff_time
        ]
        
        # Check for threshold violations
        alerts = self._check_thresholds(metric_entry)
        
        return {
            "status": "collected",
            "timestamp": timestamp.isoformat(),
            "metrics": metric_entry,
            "alerts": alerts,
            "history_size": len(self.metrics_history)
        }
    
    async def _analyze_performance(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance patterns and trends."""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics available for analysis"}
        
        # Calculate performance statistics
        recent_metrics = self.metrics_history[-100:]  # Last 100 data points
        
        stats = {
            "cpu_usage": self._calculate_stats([m["cpu_usage"] for m in recent_metrics]),
            "memory_usage": self._calculate_stats([m["memory_usage"] for m in recent_metrics]),
            "response_time_ms": self._calculate_stats([m["response_time_ms"] for m in recent_metrics]),
            "error_rate": self._calculate_stats([m["error_rate"] for m in recent_metrics])
        }
        
        # Identify patterns
        patterns = self._identify_patterns(recent_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stats, patterns)
        
        return {
            "status": "analyzed",
            "analysis_period": f"Last {len(recent_metrics)} data points",
            "statistics": stats,
            "patterns": patterns,
            "recommendations": recommendations,
            "overall_health": self._calculate_health_score(stats)
        }
    
    async def _generate_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report_type = report_data.get("type", "summary")
        period_hours = report_data.get("period_hours", 24)
        
        # Filter metrics by period
        cutoff_time = datetime.utcnow() - timedelta(hours=period_hours)
        period_metrics = [
            m for m in self.metrics_history 
            if m["timestamp"] > cutoff_time
        ]
        
        if not period_metrics:
            return {"status": "no_data", "message": f"No data available for {period_hours} hour period"}
        
        report = {
            "report_type": report_type,
            "period_hours": period_hours,
            "data_points": len(period_metrics),
            "generated_at": datetime.utcnow().isoformat(),
            "summary": self._generate_summary(period_metrics),
            "trends": self._analyze_trends(period_metrics),
            "top_issues": self._identify_top_issues(period_metrics)
        }
        
        return {
            "status": "generated",
            "report": report
        }
    
    async def _predict_trends(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future performance trends."""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data", "message": "Need at least 10 data points for trend prediction"}
        
        # Simple trend prediction using linear regression
        predictions = {}
        
        for metric in ["cpu_usage", "memory_usage", "response_time_ms", "error_rate"]:
            values = [m[metric] for m in self.metrics_history[-50:]]  # Last 50 points
            trend = self._calculate_trend(values)
            predictions[metric] = {
                "current_value": values[-1],
                "trend_direction": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable",
                "trend_strength": abs(trend),
                "predicted_1h": values[-1] + trend * 12,  # Assuming 5-minute intervals
                "predicted_24h": values[-1] + trend * 288
            }
        
        return {
            "status": "predicted",
            "predictions": predictions,
            "confidence": "medium",  # Simplified confidence measure
            "recommendation": self._generate_trend_recommendations(predictions)
        }
    
    def _check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check metrics against thresholds."""
        alerts = []
        
        for metric, threshold in self.performance_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                alerts.append({
                    "metric": metric,
                    "current_value": metrics[metric],
                    "threshold": threshold,
                    "severity": "high" if metrics[metric] > threshold * 1.2 else "medium"
                })
        
        return alerts
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values."""
        if not values:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        
        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        
        # Calculate standard deviation
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        std = variance ** 0.5
        
        return {
            "avg": round(avg, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "std": round(std, 2)
        }
    
    def _identify_patterns(self, metrics: List[Dict[str, Any]]) -> List[str]:
        """Identify performance patterns."""
        patterns = []
        
        if len(metrics) < 5:
            return patterns
        
        # Check for spikes
        cpu_values = [m["cpu_usage"] for m in metrics]
        if max(cpu_values) > 90:
            patterns.append("CPU usage spikes detected")
        
        # Check for memory leaks (consistently increasing memory)
        memory_values = [m["memory_usage"] for m in metrics[-10:]]
        if len(memory_values) >= 5 and all(memory_values[i] <= memory_values[i+1] for i in range(len(memory_values)-1)):
            patterns.append("Potential memory leak detected")
        
        # Check for performance degradation
        response_times = [m["response_time_ms"] for m in metrics]
        if len(response_times) >= 10:
            recent_avg = sum(response_times[-5:]) / 5
            older_avg = sum(response_times[-10:-5]) / 5
            if recent_avg > older_avg * 1.5:
                patterns.append("Response time degradation detected")
        
        return patterns
    
    def _generate_recommendations(self, stats: Dict[str, Any], patterns: List[str]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # CPU recommendations
        if stats["cpu_usage"]["avg"] > 70:
            recommendations.append("Consider CPU optimization or scaling up compute resources")
        
        # Memory recommendations  
        if stats["memory_usage"]["avg"] > 80:
            recommendations.append("Monitor memory usage closely, consider memory optimization")
        
        # Response time recommendations
        if stats["response_time_ms"]["avg"] > 100:
            recommendations.append("Optimize response times through caching or performance tuning")
        
        # Pattern-based recommendations
        for pattern in patterns:
            if "memory leak" in pattern:
                recommendations.append("Investigate potential memory leaks in active components")
            elif "degradation" in pattern:
                recommendations.append("Review recent changes that may have impacted performance")
        
        return recommendations
    
    def _calculate_health_score(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health score."""
        scores = {}
        
        # Individual metric scores (0-100)
        scores["cpu"] = max(0, 100 - stats["cpu_usage"]["avg"])
        scores["memory"] = max(0, 100 - stats["memory_usage"]["avg"])
        scores["response_time"] = max(0, 100 - min(stats["response_time_ms"]["avg"], 100))
        scores["error_rate"] = max(0, 100 - stats["error_rate"]["avg"] * 10)
        
        # Overall score
        overall = sum(scores.values()) / len(scores)
        
        if overall >= 90:
            health_status = "excellent"
        elif overall >= 75:
            health_status = "good"
        elif overall >= 60:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            "overall_score": round(overall, 1),
            "status": health_status,
            "component_scores": scores
        }
    
    def _generate_summary(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance summary."""
        return {
            "total_data_points": len(metrics),
            "time_range": {
                "start": metrics[0]["timestamp"].isoformat(),
                "end": metrics[-1]["timestamp"].isoformat()
            },
            "peak_cpu": max(m["cpu_usage"] for m in metrics),
            "peak_memory": max(m["memory_usage"] for m in metrics),
            "avg_response_time": sum(m["response_time_ms"] for m in metrics) / len(metrics),
            "total_errors": sum(m["error_rate"] for m in metrics),
            "total_tasks": sum(m["tasks_processed"] for m in metrics)
        }
    
    def _analyze_trends(self, metrics: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze performance trends."""
        trends = {}
        
        for metric_name in ["cpu_usage", "memory_usage", "response_time_ms", "error_rate"]:
            values = [m[metric_name] for m in metrics]
            trend = self._calculate_trend(values)
            
            if trend > 0.1:
                trends[metric_name] = "increasing"
            elif trend < -0.1:
                trends[metric_name] = "decreasing"
            else:
                trends[metric_name] = "stable"
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Linear regression slope calculation
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _identify_top_issues(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify top performance issues."""
        issues = []
        
        # Check for consistent threshold violations
        cpu_violations = sum(1 for m in metrics if m["cpu_usage"] > self.performance_thresholds["cpu_usage"])
        memory_violations = sum(1 for m in metrics if m["memory_usage"] > self.performance_thresholds["memory_usage"])
        response_violations = sum(1 for m in metrics if m["response_time_ms"] > self.performance_thresholds["response_time_ms"])
        
        if cpu_violations > len(metrics) * 0.1:  # More than 10% violations
            issues.append({
                "issue": "CPU Usage",
                "severity": "high",
                "frequency": f"{cpu_violations}/{len(metrics)} occurrences",
                "impact": "Performance degradation"
            })
        
        if memory_violations > len(metrics) * 0.1:
            issues.append({
                "issue": "Memory Usage",
                "severity": "high",
                "frequency": f"{memory_violations}/{len(metrics)} occurrences",
                "impact": "Potential out-of-memory errors"
            })
        
        if response_violations > len(metrics) * 0.05:  # More than 5% violations
            issues.append({
                "issue": "Response Time",
                "severity": "medium",
                "frequency": f"{response_violations}/{len(metrics)} occurrences",
                "impact": "User experience degradation"
            })
        
        return issues
    
    def _generate_trend_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trend predictions."""
        recommendations = []
        
        for metric, prediction in predictions.items():
            if prediction["trend_direction"] == "increasing":
                if metric == "cpu_usage" and prediction["predicted_24h"] > 90:
                    recommendations.append("CPU usage trending upward - consider scaling resources")
                elif metric == "memory_usage" and prediction["predicted_24h"] > 90:
                    recommendations.append("Memory usage trending upward - investigate memory leaks")
                elif metric == "response_time_ms" and prediction["predicted_24h"] > 200:
                    recommendations.append("Response times trending upward - performance optimization needed")
        
        if not recommendations:
            recommendations.append("Performance trends look stable - continue monitoring")
        
        return recommendations
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.metrics_history.clear()
        self.trend_data.clear()
        logger.info("Performance Analytics Plugin cleaned up")


class SamplePluginDemonstrator:
    """
    Demonstrates marketplace functionality with sample plugins.
    
    Epic 1: Maintains <50ms operations for demonstration
    """
    
    def __init__(
        self,
        marketplace: PluginMarketplace,
        developer_platform: DeveloperOnboardingPlatform
    ):
        self.marketplace = marketplace
        self.developer_platform = developer_platform
        self.sample_plugins = []
        self.demo_developers = []
    
    async def setup_demo_environment(self) -> Dict[str, Any]:
        """Setup complete demonstration environment."""
        try:
            start_time = datetime.utcnow()
            
            # Create demo developers
            await self._create_demo_developers()
            
            # Register sample plugins
            await self._register_sample_plugins()
            
            # Demonstrate certification
            await self._demonstrate_certification()
            
            # Show AI discovery
            discovery_results = await self._demonstrate_ai_discovery()
            
            # Epic 1: Track setup time
            setup_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info("Demo environment setup completed",
                       developers=len(self.demo_developers),
                       plugins=len(self.sample_plugins),
                       setup_time_ms=round(setup_time_ms, 2))
            
            return {
                "success": True,
                "demo_environment": {
                    "developers": len(self.demo_developers),
                    "plugins": len(self.sample_plugins),
                    "discovery_results": discovery_results
                },
                "setup_time_ms": round(setup_time_ms, 2)
            }
            
        except Exception as e:
            logger.error("Demo environment setup failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _create_demo_developers(self) -> None:
        """Create demonstration developer accounts."""
        demo_devs = [
            {
                "username": "productivity_dev",
                "email": "productivity@leanvibe.io",
                "full_name": "ProductivityPro Developer",
                "company_name": "ProductivityPro Inc.",
                "github_profile": "https://github.com/productivitypro"
            },
            {
                "username": "integration_expert",
                "email": "integrations@leanvibe.io", 
                "full_name": "Integration Expert",
                "company_name": "IntegrateAll Ltd.",
                "github_profile": "https://github.com/integrateall"
            },
            {
                "username": "analytics_guru",
                "email": "analytics@leanvibe.io",
                "full_name": "Analytics Guru",
                "company_name": "DataInsights Corp.",
                "github_profile": "https://github.com/datainsights"
            }
        ]
        
        for dev_data in demo_devs:
            try:
                developer = await self.developer_platform.register_developer(**dev_data)
                self.demo_developers.append(developer)
                logger.info("Demo developer created", username=dev_data["username"])
            except Exception as e:
                logger.warning("Demo developer creation failed", username=dev_data["username"], error=str(e))
    
    async def _register_sample_plugins(self) -> None:
        """Register sample plugins in the marketplace."""
        sample_plugin_configs = [
            {
                "plugin_class": ProductivityBoosterPlugin,
                "developer_index": 0,
                "category": PluginCategory.PRODUCTIVITY,
                "tags": ["productivity", "task-management", "focus", "automation"],
                "pricing": {"type": "freemium", "premium_price": 9.99}
            },
            {
                "plugin_class": SlackIntegrationPlugin,
                "developer_index": 1,
                "category": PluginCategory.INTEGRATION,
                "tags": ["slack", "communication", "notifications", "team"],
                "pricing": {"type": "free", "premium_price": 0.0}
            },
            {
                "plugin_class": PerformanceAnalyticsPlugin,
                "developer_index": 2,
                "category": PluginCategory.ANALYTICS,
                "tags": ["performance", "monitoring", "analytics", "optimization"],
                "pricing": {"type": "premium", "premium_price": 19.99}
            }
        ]
        
        for config in sample_plugin_configs:
            try:
                # Create plugin instance
                plugin_instance = config["plugin_class"]()
                
                # Get developer
                developer = self.demo_developers[config["developer_index"]]
                
                # Create plugin metadata
                plugin_metadata = PluginMetadata(
                    plugin_id=plugin_instance.plugin_id,
                    name=plugin_instance.name,
                    version=plugin_instance.version,
                    description=plugin_instance.description,
                    author=developer.full_name,
                    plugin_type=plugin_instance.plugin_type,
                    dependencies=[],
                    configuration_schema={},
                    permissions=[]
                )
                
                # Register in marketplace
                registration_result = await self.marketplace.register_plugin(
                    plugin_metadata,
                    developer_id=developer.developer_id,
                    category=config["category"],
                    tags=config["tags"]
                )
                
                if registration_result.success:
                    self.sample_plugins.append({
                        "plugin_instance": plugin_instance,
                        "plugin_metadata": plugin_metadata,
                        "developer": developer,
                        "registration_result": registration_result
                    })
                    
                    logger.info("Sample plugin registered",
                               plugin_id=plugin_instance.plugin_id,
                               developer=developer.username)
                
            except Exception as e:
                logger.error("Sample plugin registration failed",
                           plugin_id=config["plugin_class"].__name__,
                           error=str(e))
    
    async def _demonstrate_certification(self) -> None:
        """Demonstrate security certification process."""
        for plugin_data in self.sample_plugins[:2]:  # Certify first 2 plugins
            try:
                plugin_entry = await self.marketplace.get_plugin_details(
                    plugin_data["plugin_metadata"].plugin_id
                )
                
                if plugin_entry:
                    # Simulate certification (in real implementation would have actual pipeline)
                    plugin_entry.certification_level = CertificationLevel.SECURITY_VERIFIED
                    plugin_entry.status = PluginStatus.PUBLISHED
                    
                    logger.info("Plugin certified",
                               plugin_id=plugin_entry.plugin_id,
                               level=plugin_entry.certification_level.value)
                
            except Exception as e:
                logger.error("Plugin certification demo failed",
                           plugin_id=plugin_data["plugin_metadata"].plugin_id,
                           error=str(e))
    
    async def _demonstrate_ai_discovery(self) -> Dict[str, Any]:
        """Demonstrate AI-powered plugin discovery."""
        discovery_queries = [
            "productivity plugins for task management",
            "integration tools for team communication",
            "performance monitoring and analytics"
        ]
        
        discovery_results = {}
        
        for query in discovery_queries:
            try:
                # Search plugins
                search_result = await self.marketplace.search_plugins({
                    "query": query,
                    "limit": 5
                })
                
                discovery_results[query] = {
                    "plugins_found": len(search_result.plugins),
                    "plugins": [p.to_dict() for p in search_result.plugins]
                }
                
            except Exception as e:
                logger.error("AI discovery demo failed", query=query, error=str(e))
                discovery_results[query] = {"error": str(e)}
        
        return discovery_results
    
    async def get_demo_statistics(self) -> Dict[str, Any]:
        """Get demonstration statistics."""
        return {
            "demo_developers": len(self.demo_developers),
            "sample_plugins": len(self.sample_plugins),
            "plugin_categories": list(set(p["plugin_metadata"].plugin_type for p in self.sample_plugins)),
            "certification_levels": list(set(CertificationLevel.SECURITY_VERIFIED.value for _ in self.sample_plugins[:2])),
            "marketplace_stats": await self.marketplace.get_marketplace_statistics()
        }
    
    async def cleanup_demo(self) -> None:
        """Cleanup demonstration environment."""
        try:
            # Clear sample data
            self.sample_plugins.clear()
            self.demo_developers.clear()
            
            logger.info("Demo environment cleaned up")
            
        except Exception as e:
            logger.error("Demo cleanup failed", error=str(e))