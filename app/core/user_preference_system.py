"""
User Preference System - Phase 3 Intelligence Layer Implementation

This module implements personalized developer experience based on usage patterns
and preferences to optimize workflow and reduce cognitive load.
"""

import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import uuid4

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from app.core.database import get_db
from app.core.redis import get_redis


logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Available notification channels"""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"


class AlertPriority(Enum):
    """User-customizable alert priorities"""
    ALWAYS = "always"
    WORK_HOURS = "work_hours"
    CRITICAL_ONLY = "critical_only"
    NEVER = "never"


class DashboardLayout(Enum):
    """Dashboard layout preferences"""
    COMPACT = "compact"
    DETAILED = "detailed"
    MINIMAL = "minimal"
    CUSTOM = "custom"


class ColorTheme(Enum):
    """Color theme preferences"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    HIGH_CONTRAST = "high_contrast"


@dataclass
class NotificationPreferences:
    """User notification preferences"""
    enabled_channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.DESKTOP])
    alert_thresholds: Dict[str, AlertPriority] = field(default_factory=dict)
    quiet_hours_start: Optional[str] = "22:00"  # 24-hour format
    quiet_hours_end: Optional[str] = "08:00"
    focus_mode_duration: int = 60  # minutes
    escalation_delay: int = 15  # minutes before escalating
    custom_keywords: List[str] = field(default_factory=list)


@dataclass
class DashboardPreferences:
    """Dashboard customization preferences"""
    layout: DashboardLayout = DashboardLayout.DETAILED
    visible_widgets: List[str] = field(default_factory=lambda: ["alerts", "agents", "performance", "tasks"])
    widget_order: List[str] = field(default_factory=list)
    refresh_interval: int = 30  # seconds
    color_theme: ColorTheme = ColorTheme.AUTO
    show_debug_info: bool = False
    mobile_layout: Optional[DashboardLayout] = DashboardLayout.COMPACT


@dataclass
class UsagePattern:
    """User usage pattern tracking"""
    user_id: str
    command_frequency: Dict[str, int] = field(default_factory=dict)
    response_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    peak_activity_hours: List[int] = field(default_factory=list)
    preferred_workflows: List[str] = field(default_factory=list)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PersonalizationInsights:
    """Insights generated from usage patterns"""
    user_id: str
    efficiency_score: float
    most_used_commands: List[Tuple[str, int]]
    improvement_suggestions: List[str]
    workflow_optimizations: List[str]
    productivity_trends: Dict[str, float]
    generated_at: datetime = field(default_factory=datetime.now)


class UserPreferenceSystem:
    """
    Personalization system for developer experience optimization.
    
    Provides:
    - Notification settings and preferences
    - Dashboard layout customization
    - Usage pattern tracking
    - Personalized recommendations
    """
    
    def __init__(self, redis_client: redis.Redis, db_session: AsyncSession = None):
        self.redis = redis_client
        self.db = db_session
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.usage_patterns: Dict[str, UsagePattern] = {}
        
        # Configuration
        self.pattern_analysis_window = timedelta(days=30)
        self.preference_sync_interval = timedelta(minutes=5)
        self.last_sync: Dict[str, datetime] = {}
        
    async def initialize(self) -> None:
        """Initialize the preference system"""
        try:
            # Load user preferences from Redis
            await self._load_user_preferences()
            
            # Load usage patterns
            await self._load_usage_patterns()
            
            logger.info("User Preference System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize User Preference System: {e}")
            raise
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user preferences"""
        try:
            if user_id not in self.user_preferences:
                await self._create_default_preferences(user_id)
            
            preferences = self.user_preferences[user_id]
            
            # Add computed recommendations
            recommendations = await self._generate_recommendations(user_id)
            preferences['recommendations'] = recommendations
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error getting preferences for user {user_id}: {e}")
            return await self._create_default_preferences(user_id)
    
    async def update_notification_preferences(
        self, 
        user_id: str, 
        preferences: NotificationPreferences
    ) -> bool:
        """Update user notification preferences"""
        try:
            if user_id not in self.user_preferences:
                await self._create_default_preferences(user_id)
            
            self.user_preferences[user_id]['notifications'] = asdict(preferences)
            await self._save_preferences(user_id)
            
            logger.info(f"Updated notification preferences for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating notification preferences: {e}")
            return False
    
    async def update_dashboard_preferences(
        self, 
        user_id: str, 
        preferences: DashboardPreferences
    ) -> bool:
        """Update user dashboard preferences"""
        try:
            if user_id not in self.user_preferences:
                await self._create_default_preferences(user_id)
            
            self.user_preferences[user_id]['dashboard'] = asdict(preferences)
            await self._save_preferences(user_id)
            
            logger.info(f"Updated dashboard preferences for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating dashboard preferences: {e}")
            return False
    
    async def track_usage(
        self, 
        user_id: str, 
        command: str, 
        response_time: float = None,
        success: bool = True,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Track user command usage for pattern analysis"""
        try:
            if user_id not in self.usage_patterns:
                self.usage_patterns[user_id] = UsagePattern(user_id=user_id)
            
            pattern = self.usage_patterns[user_id]
            
            # Update command frequency
            pattern.command_frequency[command] = pattern.command_frequency.get(command, 0) + 1
            
            # Track response time
            if response_time is not None:
                pattern.response_times[command].append(response_time)
                # Keep only recent response times
                if len(pattern.response_times[command]) > 100:
                    pattern.response_times[command] = pattern.response_times[command][-100:]
            
            # Track errors
            if not success:
                pattern.error_patterns[command] = pattern.error_patterns.get(command, 0) + 1
            
            # Update activity hour
            current_hour = datetime.now().hour
            if current_hour not in pattern.peak_activity_hours:
                pattern.peak_activity_hours.append(current_hour)
            
            pattern.last_updated = datetime.now()
            
            # Periodically save patterns
            if user_id not in self.last_sync or \
               datetime.now() - self.last_sync[user_id] > self.preference_sync_interval:
                await self._save_usage_pattern(user_id)
                self.last_sync[user_id] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error tracking usage for user {user_id}: {e}")
    
    async def get_personalized_dashboard_config(self, user_id: str) -> Dict[str, Any]:
        """Get personalized dashboard configuration"""
        try:
            preferences = await self.get_user_preferences(user_id)
            dashboard_prefs = preferences.get('dashboard', {})
            
            # Generate adaptive configuration based on usage patterns
            usage_pattern = self.usage_patterns.get(user_id)
            
            config = {
                'layout': dashboard_prefs.get('layout', 'detailed'),
                'theme': dashboard_prefs.get('color_theme', 'auto'),
                'refresh_interval': dashboard_prefs.get('refresh_interval', 30),
                'widgets': await self._get_personalized_widgets(user_id, usage_pattern),
                'alerts': await self._get_personalized_alert_config(user_id),
                'quick_actions': await self._get_personalized_quick_actions(user_id, usage_pattern)
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Error generating personalized dashboard config: {e}")
            return self._get_default_dashboard_config()
    
    async def should_send_notification(
        self, 
        user_id: str, 
        alert_type: str, 
        severity: str
    ) -> Tuple[bool, List[NotificationChannel]]:
        """Determine if notification should be sent and via which channels"""
        try:
            preferences = await self.get_user_preferences(user_id)
            notification_prefs = preferences.get('notifications', {})
            
            # Check quiet hours
            if await self._is_quiet_hours(notification_prefs):
                if severity not in ['critical', 'high']:
                    return False, []
            
            # Check alert thresholds
            alert_threshold = notification_prefs.get('alert_thresholds', {}).get(
                alert_type, 
                AlertPriority.WORK_HOURS.value
            )
            
            if alert_threshold == AlertPriority.NEVER.value:
                return False, []
            elif alert_threshold == AlertPriority.CRITICAL_ONLY.value and severity not in ['critical']:
                return False, []
            
            # Determine channels
            enabled_channels = [
                NotificationChannel(ch) for ch in notification_prefs.get(
                    'enabled_channels', 
                    [NotificationChannel.DESKTOP.value]
                )
            ]
            
            return True, enabled_channels
            
        except Exception as e:
            logger.error(f"Error checking notification preferences: {e}")
            return True, [NotificationChannel.DESKTOP]  # Default to safe option
    
    async def generate_productivity_insights(self, user_id: str) -> PersonalizationInsights:
        """Generate productivity insights for user"""
        try:
            usage_pattern = self.usage_patterns.get(user_id)
            if not usage_pattern:
                return PersonalizationInsights(
                    user_id=user_id,
                    efficiency_score=0.5,
                    most_used_commands=[],
                    improvement_suggestions=["Use the system more to get personalized insights"],
                    workflow_optimizations=[],
                    productivity_trends={}
                )
            
            # Calculate efficiency score
            efficiency_score = await self._calculate_efficiency_score(usage_pattern)
            
            # Get most used commands
            most_used = sorted(
                usage_pattern.command_frequency.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # Generate suggestions
            suggestions = await self._generate_improvement_suggestions(usage_pattern)
            
            # Generate workflow optimizations
            optimizations = await self._generate_workflow_optimizations(usage_pattern)
            
            # Calculate productivity trends
            trends = await self._calculate_productivity_trends(usage_pattern)
            
            return PersonalizationInsights(
                user_id=user_id,
                efficiency_score=efficiency_score,
                most_used_commands=most_used,
                improvement_suggestions=suggestions,
                workflow_optimizations=optimizations,
                productivity_trends=trends
            )
            
        except Exception as e:
            logger.error(f"Error generating productivity insights: {e}")
            return PersonalizationInsights(user_id=user_id, efficiency_score=0.0, most_used_commands=[], improvement_suggestions=[], workflow_optimizations=[], productivity_trends={})
    
    # Private methods
    
    async def _load_user_preferences(self) -> None:
        """Load user preferences from Redis"""
        try:
            preference_keys = await self.redis.keys("user_preferences:*")
            
            for key in preference_keys:
                user_id = key.decode().split(":")[-1]
                preference_data = await self.redis.get(key)
                
                if preference_data:
                    self.user_preferences[user_id] = json.loads(preference_data)
                    
        except Exception as e:
            logger.error(f"Error loading user preferences: {e}")
    
    async def _load_usage_patterns(self) -> None:
        """Load usage patterns from Redis"""
        try:
            pattern_keys = await self.redis.keys("usage_pattern:*")
            
            for key in pattern_keys:
                user_id = key.decode().split(":")[-1]
                pattern_data = await self.redis.get(key)
                
                if pattern_data:
                    data = json.loads(pattern_data)
                    self.usage_patterns[user_id] = UsagePattern(
                        user_id=user_id,
                        command_frequency=data.get('command_frequency', {}),
                        response_times=defaultdict(list, data.get('response_times', {})),
                        peak_activity_hours=data.get('peak_activity_hours', []),
                        preferred_workflows=data.get('preferred_workflows', []),
                        error_patterns=data.get('error_patterns', {}),
                        last_updated=datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat()))
                    )
                    
        except Exception as e:
            logger.error(f"Error loading usage patterns: {e}")
    
    async def _create_default_preferences(self, user_id: str) -> Dict[str, Any]:
        """Create default preferences for new user"""
        default_prefs = {
            'user_id': user_id,
            'notifications': asdict(NotificationPreferences()),
            'dashboard': asdict(DashboardPreferences()),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self.user_preferences[user_id] = default_prefs
        await self._save_preferences(user_id)
        
        return default_prefs
    
    async def _save_preferences(self, user_id: str) -> None:
        """Save user preferences to Redis"""
        try:
            preferences = self.user_preferences[user_id]
            preferences['updated_at'] = datetime.now().isoformat()
            
            await self.redis.setex(
                f"user_preferences:{user_id}",
                86400 * 7,  # 7 days TTL
                json.dumps(preferences, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error saving preferences for user {user_id}: {e}")
    
    async def _save_usage_pattern(self, user_id: str) -> None:
        """Save usage pattern to Redis"""
        try:
            pattern = self.usage_patterns[user_id]
            pattern_data = {
                'user_id': pattern.user_id,
                'command_frequency': pattern.command_frequency,
                'response_times': dict(pattern.response_times),
                'peak_activity_hours': pattern.peak_activity_hours,
                'preferred_workflows': pattern.preferred_workflows,
                'error_patterns': pattern.error_patterns,
                'last_updated': pattern.last_updated.isoformat()
            }
            
            await self.redis.setex(
                f"usage_pattern:{user_id}",
                86400 * 30,  # 30 days TTL
                json.dumps(pattern_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error saving usage pattern for user {user_id}: {e}")
    
    async def _is_quiet_hours(self, notification_prefs: Dict[str, Any]) -> bool:
        """Check if current time is within quiet hours"""
        try:
            quiet_start = notification_prefs.get('quiet_hours_start', '22:00')
            quiet_end = notification_prefs.get('quiet_hours_end', '08:00')
            
            if not quiet_start or not quiet_end:
                return False
            
            now = datetime.now().time()
            start_time = datetime.strptime(quiet_start, '%H:%M').time()
            end_time = datetime.strptime(quiet_end, '%H:%M').time()
            
            if start_time <= end_time:
                # Same day range
                return start_time <= now <= end_time
            else:
                # Overnight range
                return now >= start_time or now <= end_time
                
        except Exception as e:
            logger.error(f"Error checking quiet hours: {e}")
            return False
    
    async def _get_personalized_widgets(
        self, 
        user_id: str, 
        usage_pattern: Optional[UsagePattern]
    ) -> List[Dict[str, Any]]:
        """Generate personalized widget configuration"""
        default_widgets = [
            {'name': 'alerts', 'priority': 1, 'size': 'large'},
            {'name': 'agents', 'priority': 2, 'size': 'medium'},
            {'name': 'performance', 'priority': 3, 'size': 'medium'},
            {'name': 'tasks', 'priority': 4, 'size': 'small'}
        ]
        
        if not usage_pattern:
            return default_widgets
        
        # Customize based on usage patterns
        widgets = default_widgets.copy()
        
        # Prioritize widgets based on command usage
        if usage_pattern.command_frequency:
            most_used_commands = Counter(usage_pattern.command_frequency).most_common(3)
            
            for command, _ in most_used_commands:
                if 'agent' in command.lower():
                    # Increase agent widget priority
                    for widget in widgets:
                        if widget['name'] == 'agents':
                            widget['priority'] = max(1, widget['priority'] - 1)
                elif 'performance' in command.lower() or 'metrics' in command.lower():
                    # Increase performance widget priority
                    for widget in widgets:
                        if widget['name'] == 'performance':
                            widget['priority'] = max(1, widget['priority'] - 1)
        
        return sorted(widgets, key=lambda x: x['priority'])
    
    async def _get_personalized_alert_config(self, user_id: str) -> Dict[str, Any]:
        """Get personalized alert configuration"""
        preferences = await self.get_user_preferences(user_id)
        notification_prefs = preferences.get('notifications', {})
        
        return {
            'thresholds': notification_prefs.get('alert_thresholds', {}),
            'channels': notification_prefs.get('enabled_channels', ['desktop']),
            'escalation_delay': notification_prefs.get('escalation_delay', 15),
            'custom_keywords': notification_prefs.get('custom_keywords', [])
        }
    
    async def _get_personalized_quick_actions(
        self, 
        user_id: str, 
        usage_pattern: Optional[UsagePattern]
    ) -> List[Dict[str, Any]]:
        """Generate personalized quick actions"""
        default_actions = [
            {'name': 'View Agents', 'command': '/hive status', 'icon': 'agents'},
            {'name': 'Check Health', 'command': '/hive health', 'icon': 'health'},
            {'name': 'View Logs', 'command': '/hive logs', 'icon': 'logs'}
        ]
        
        if not usage_pattern or not usage_pattern.command_frequency:
            return default_actions
        
        # Add most used commands as quick actions
        most_used = Counter(usage_pattern.command_frequency).most_common(5)
        personalized_actions = []
        
        for command, frequency in most_used:
            if frequency > 5:  # Only include frequently used commands
                personalized_actions.append({
                    'name': command.replace('_', ' ').title(),
                    'command': f'/hive {command}',
                    'icon': 'command',
                    'usage_count': frequency
                })
        
        # Combine with defaults, prioritizing personalized
        all_actions = personalized_actions + default_actions
        return all_actions[:8]  # Limit to 8 quick actions
    
    def _get_default_dashboard_config(self) -> Dict[str, Any]:
        """Get default dashboard configuration"""
        return {
            'layout': 'detailed',
            'theme': 'auto',
            'refresh_interval': 30,
            'widgets': [
                {'name': 'alerts', 'priority': 1, 'size': 'large'},
                {'name': 'agents', 'priority': 2, 'size': 'medium'},
                {'name': 'performance', 'priority': 3, 'size': 'medium'},
                {'name': 'tasks', 'priority': 4, 'size': 'small'}
            ],
            'alerts': {
                'thresholds': {},
                'channels': ['desktop'],
                'escalation_delay': 15,
                'custom_keywords': []
            },
            'quick_actions': [
                {'name': 'View Agents', 'command': '/hive status', 'icon': 'agents'},
                {'name': 'Check Health', 'command': '/hive health', 'icon': 'health'}
            ]
        }
    
    async def _calculate_efficiency_score(self, usage_pattern: UsagePattern) -> float:
        """Calculate user efficiency score based on usage patterns"""
        try:
            # Factors contributing to efficiency
            factors = []
            
            # Command diversity (using different commands indicates good system knowledge)
            unique_commands = len(usage_pattern.command_frequency)
            if unique_commands > 0:
                diversity_score = min(1.0, unique_commands / 20)  # Normalize to 20 commands
                factors.append(diversity_score)
            
            # Error rate (lower is better)
            total_commands = sum(usage_pattern.command_frequency.values())
            total_errors = sum(usage_pattern.error_patterns.values())
            
            if total_commands > 0:
                error_rate = total_errors / total_commands
                error_score = max(0.0, 1.0 - error_rate * 2)  # 50% error rate = 0 score
                factors.append(error_score)
            
            # Response time consistency (lower variance is better)
            response_time_scores = []
            for command, times in usage_pattern.response_times.items():
                if len(times) > 1:
                    avg_time = sum(times) / len(times)
                    # Normalize based on expected response time (assume 5s is baseline)
                    time_score = max(0.0, min(1.0, 10.0 / max(avg_time, 1.0)))
                    response_time_scores.append(time_score)
            
            if response_time_scores:
                factors.append(sum(response_time_scores) / len(response_time_scores))
            
            # Activity consistency (regular usage indicates engagement)
            if len(usage_pattern.peak_activity_hours) > 0:
                activity_score = min(1.0, len(usage_pattern.peak_activity_hours) / 8)  # 8 hours max
                factors.append(activity_score)
            
            return sum(factors) / len(factors) if factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating efficiency score: {e}")
            return 0.5
    
    async def _generate_improvement_suggestions(self, usage_pattern: UsagePattern) -> List[str]:
        """Generate improvement suggestions based on usage patterns"""
        suggestions = []
        
        try:
            # Analyze command usage
            total_commands = sum(usage_pattern.command_frequency.values())
            unique_commands = len(usage_pattern.command_frequency)
            
            if unique_commands < 5:
                suggestions.append("Explore more system commands to improve efficiency")
            
            # Check error patterns
            total_errors = sum(usage_pattern.error_patterns.values())
            if total_errors > total_commands * 0.1:  # >10% error rate
                suggestions.append("Consider reviewing command syntax to reduce errors")
            
            # Check response times
            slow_commands = []
            for command, times in usage_pattern.response_times.items():
                if times and sum(times) / len(times) > 10:  # Average >10s
                    slow_commands.append(command)
            
            if slow_commands:
                suggestions.append(f"Commands taking longer than expected: {', '.join(slow_commands[:3])}")
            
            # Check activity patterns
            if len(usage_pattern.peak_activity_hours) < 2:
                suggestions.append("Regular usage patterns help us provide better recommendations")
            
            if not suggestions:
                suggestions.append("Great job! Your usage patterns show efficient system interaction")
                
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            suggestions.append("Continue using the system to get personalized suggestions")
        
        return suggestions
    
    async def _generate_workflow_optimizations(self, usage_pattern: UsagePattern) -> List[str]:
        """Generate workflow optimization recommendations"""
        optimizations = []
        
        try:
            # Find commonly used command sequences
            if len(usage_pattern.command_frequency) > 3:
                # Suggest automation for frequently used commands
                most_used = sorted(
                    usage_pattern.command_frequency.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                frequent_commands = [cmd for cmd, count in most_used if count > 10]
                if frequent_commands:
                    optimizations.append(f"Consider creating shortcuts for: {', '.join(frequent_commands)}")
            
            # Suggest dashboard customization based on usage
            if 'status' in usage_pattern.command_frequency and usage_pattern.command_frequency['status'] > 20:
                optimizations.append("Customize dashboard widgets to reduce need for status commands")
            
            # Suggest notification optimization
            if any('alert' in cmd for cmd in usage_pattern.command_frequency.keys()):
                optimizations.append("Configure alert preferences to reduce information overload")
            
            if not optimizations:
                optimizations.append("Your workflow appears optimized - continue current practices")
                
        except Exception as e:
            logger.error(f"Error generating workflow optimizations: {e}")
            optimizations.append("Use the system more to get workflow optimization suggestions")
        
        return optimizations
    
    async def _calculate_productivity_trends(self, usage_pattern: UsagePattern) -> Dict[str, float]:
        """Calculate productivity trends"""
        trends = {}
        
        try:
            # Command usage trend (mock - in real system, compare with historical data)
            total_commands = sum(usage_pattern.command_frequency.values())
            trends['command_usage'] = min(100.0, total_commands)  # Normalized to 100
            
            # Error rate trend
            total_errors = sum(usage_pattern.error_patterns.values())
            error_rate = (total_errors / total_commands * 100) if total_commands > 0 else 0
            trends['error_rate'] = error_rate
            
            # Response time trend
            all_response_times = []
            for times in usage_pattern.response_times.values():
                all_response_times.extend(times)
            
            avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 5.0
            trends['avg_response_time'] = avg_response_time
            
            # Activity consistency
            trends['activity_consistency'] = len(usage_pattern.peak_activity_hours) / 24 * 100  # Percentage of day active
            
        except Exception as e:
            logger.error(f"Error calculating productivity trends: {e}")
        
        return trends
    
    async def _generate_recommendations(self, user_id: str) -> List[str]:
        """Generate general recommendations for user"""
        recommendations = []
        
        try:
            usage_pattern = self.usage_patterns.get(user_id)
            if not usage_pattern:
                recommendations.append("Start using the system to get personalized recommendations")
                return recommendations
            
            # Add efficiency-based recommendations
            efficiency = await self._calculate_efficiency_score(usage_pattern)
            
            if efficiency > 0.8:
                recommendations.append("Excellent system usage! Consider sharing tips with other users")
            elif efficiency > 0.6:
                recommendations.append("Good progress! Try exploring advanced features")
            else:
                recommendations.append("Consider reviewing the user guide to improve efficiency")
            
            # Add usage-specific recommendations
            if len(usage_pattern.command_frequency) < 5:
                recommendations.append("Explore more system commands to unlock additional capabilities")
            
            if sum(usage_pattern.error_patterns.values()) > 5:
                recommendations.append("Check command syntax guide to reduce errors")
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Continue using the system for personalized insights")
        
        return recommendations[:5]  # Limit to 5 recommendations


# Factory function for dependency injection
async def create_user_preference_system() -> UserPreferenceSystem:
    """Create and initialize User Preference System"""
    redis_client = await get_redis()
    system = UserPreferenceSystem(redis_client, None)
    await system.initialize()
    return system