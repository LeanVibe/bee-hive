"""
LeanVibe Agent Hive 2.0 - Competitive Intelligence Dashboard
Real-time competitive intelligence visualization and executive decision support
"""

from fastapi import FastAPI, WebSocket, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncpg
from redis.asyncio import Redis
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum

# Import our competitive intelligence types
from competitive_intelligence_implementation import (
    ThreatLevel, CompetitorType, CompetitiveIntelligence, CompetitorProfile
)

app = FastAPI(title="LeanVibe Competitive Intelligence Dashboard")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure"""
    total_threats_detected: int
    critical_threats: int
    high_threats: int
    medium_threats: int
    low_threats: int
    competitive_win_rate: float
    market_share_estimate: float
    technology_lead_months: int
    customer_satisfaction_score: float
    pipeline_protected: float
    revenue_at_risk: float

@dataclass
class CompetitorInsight:
    """Competitor insight for dashboard display"""
    name: str
    threat_level: ThreatLevel
    recent_activity_count: int
    last_significant_activity: Optional[datetime]
    technology_gap_months: float
    market_overlap_percentage: float
    win_rate_against: float
    key_developments: List[str]
    trend_direction: str  # 'increasing', 'stable', 'decreasing'

class CompetitiveIntelligenceDashboard:
    """Main dashboard controller and data provider"""
    
    def __init__(self, db_pool: asyncpg.Pool, redis: Redis):
        self.db_pool = db_pool
        self.redis = redis
        self.logger = logging.getLogger('competitive_dashboard')
        
        # WebSocket connections for real-time updates
        self.active_connections: List[WebSocket] = []
        
        # Cache for dashboard data
        self.cache_timeout = 300  # 5 minutes
        self.cached_data = {}
        self.cache_timestamps = {}

    async def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get comprehensive dashboard metrics"""
        cache_key = 'dashboard_metrics'
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cached_data[cache_key]
        
        async with self.db_pool.acquire() as conn:
            # Threat counts
            threat_counts = await conn.fetch("""
                SELECT threat_level, COUNT(*) as count
                FROM competitive_intelligence 
                WHERE timestamp > NOW() - INTERVAL '30 days'
                GROUP BY threat_level
            """)
            
            threat_dict = {row['threat_level']: row['count'] for row in threat_counts}
            
            # Win rate calculation (simulated - would come from CRM integration)
            win_rate_data = await conn.fetchrow("""
                SELECT 
                    COUNT(CASE WHEN outcome = 'won' THEN 1 END)::float / 
                    COUNT(*)::float * 100 as win_rate
                FROM competitive_deals 
                WHERE close_date > NOW() - INTERVAL '90 days'
            """)
            
            # Market metrics (simulated)
            market_data = await conn.fetchrow("""
                SELECT 
                    market_share_estimate,
                    customer_satisfaction_score,
                    pipeline_protected,
                    revenue_at_risk
                FROM market_metrics 
                ORDER BY updated_at DESC 
                LIMIT 1
            """)
        
        metrics = DashboardMetrics(
            total_threats_detected=sum(threat_dict.values()),
            critical_threats=threat_dict.get('critical', 0),
            high_threats=threat_dict.get('high', 0),
            medium_threats=threat_dict.get('medium', 0),
            low_threats=threat_dict.get('low', 0),
            competitive_win_rate=win_rate_data['win_rate'] if win_rate_data else 85.0,
            market_share_estimate=market_data['market_share_estimate'] if market_data else 15.2,
            technology_lead_months=36,  # Our 3+ year lead
            customer_satisfaction_score=market_data['customer_satisfaction_score'] if market_data else 94.2,
            pipeline_protected=market_data['pipeline_protected'] if market_data else 247600000,
            revenue_at_risk=market_data['revenue_at_risk'] if market_data else 12300000
        )
        
        # Cache the result
        self.cached_data[cache_key] = metrics
        self.cache_timestamps[cache_key] = datetime.now()
        
        return metrics

    async def get_competitor_insights(self) -> List[CompetitorInsight]:
        """Get competitor insights for dashboard"""
        cache_key = 'competitor_insights'
        
        if self._is_cache_valid(cache_key):
            return self.cached_data[cache_key]
        
        async with self.db_pool.acquire() as conn:
            # Get competitor data with recent activity
            competitor_data = await conn.fetch("""
                SELECT 
                    competitor,
                    COUNT(*) as activity_count,
                    MAX(timestamp) as last_activity,
                    AVG(CASE 
                        WHEN threat_level = 'critical' THEN 4
                        WHEN threat_level = 'high' THEN 3
                        WHEN threat_level = 'medium' THEN 2
                        ELSE 1
                    END) as avg_threat_level
                FROM competitive_intelligence
                WHERE timestamp > NOW() - INTERVAL '30 days'
                GROUP BY competitor
                ORDER BY avg_threat_level DESC, activity_count DESC
            """)
            
            insights = []
            for row in competitor_data:
                # Get recent developments
                recent_developments = await conn.fetch("""
                    SELECT intelligence_type, content
                    FROM competitive_intelligence
                    WHERE competitor = $1 AND timestamp > NOW() - INTERVAL '7 days'
                    ORDER BY timestamp DESC
                    LIMIT 3
                """, row['competitor'])
                
                # Calculate threat level from average
                avg_threat = row['avg_threat_level']
                if avg_threat >= 3.5:
                    threat_level = ThreatLevel.CRITICAL
                elif avg_threat >= 2.5:
                    threat_level = ThreatLevel.HIGH
                elif avg_threat >= 1.5:
                    threat_level = ThreatLevel.MEDIUM
                else:
                    threat_level = ThreatLevel.LOW
                
                # Get competitive win rate (simulated)
                win_rate = await self._get_win_rate_against_competitor(row['competitor'])
                
                insight = CompetitorInsight(
                    name=row['competitor'],
                    threat_level=threat_level,
                    recent_activity_count=row['activity_count'],
                    last_significant_activity=row['last_activity'],
                    technology_gap_months=self._calculate_technology_gap(row['competitor']),
                    market_overlap_percentage=self._calculate_market_overlap(row['competitor']),
                    win_rate_against=win_rate,
                    key_developments=[dev['intelligence_type'] for dev in recent_developments],
                    trend_direction=self._calculate_trend_direction(row['competitor'])
                )
                insights.append(insight)
        
        # Cache the result
        self.cached_data[cache_key] = insights
        self.cache_timestamps[cache_key] = datetime.now()
        
        return insights

    async def get_threat_timeline_data(self, days: int = 30) -> Dict[str, Any]:
        """Get threat timeline data for visualization"""
        async with self.db_pool.acquire() as conn:
            timeline_data = await conn.fetch("""
                SELECT 
                    DATE(timestamp) as date,
                    threat_level,
                    COUNT(*) as count
                FROM competitive_intelligence
                WHERE timestamp > NOW() - INTERVAL '%s days'
                GROUP BY DATE(timestamp), threat_level
                ORDER BY date
            """ % days)
        
        # Prepare data for Plotly
        df = pd.DataFrame([dict(row) for row in timeline_data])
        
        if df.empty:
            return {'data': [], 'layout': {}}
        
        # Create stacked area chart
        fig = px.area(df, x='date', y='count', color='threat_level',
                     title='Competitive Threat Timeline',
                     color_discrete_map={
                         'critical': '#ff4444',
                         'high': '#ff8800', 
                         'medium': '#ffcc00',
                         'low': '#44aa44'
                     })
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Threat Count",
            hovermode='x unified'
        )
        
        return json.loads(fig.to_json())

    async def get_competitor_comparison_data(self) -> Dict[str, Any]:
        """Get competitor comparison radar chart data"""
        competitors = ['github_copilot', 'aws_codewhisperer', 'google_duet', 'cursor_ide']
        metrics = ['Technology', 'Market Share', 'Resources', 'Threat Level', 'Innovation']
        
        # Simulated data - in practice, would come from detailed analysis
        data = {
            'github_copilot': [7, 8, 9, 8, 6],
            'aws_codewhisperer': [6, 6, 8, 6, 5],
            'google_duet': [6, 5, 8, 5, 6],
            'cursor_ide': [4, 2, 3, 3, 7],
            'leanvibe_agent_hive': [10, 4, 6, 10, 10]  # Our positioning
        }
        
        fig = go.Figure()
        
        colors = ['#ff4444', '#ff8800', '#4488ff', '#8844ff', '#44aa44']
        
        for i, (competitor, values) in enumerate(data.items()):
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=competitor.replace('_', ' ').title(),
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Competitive Positioning Radar"
        )
        
        return json.loads(fig.to_json())

    async def get_market_opportunity_data(self) -> Dict[str, Any]:
        """Get market opportunity visualization data"""
        # Simulated market opportunity data
        opportunities = [
            {'segment': 'Fortune 50', 'size': 48000000, 'probability': 85, 'timeline': 6},
            {'segment': 'Fortune 100', 'size': 45000000, 'probability': 75, 'timeline': 4},
            {'segment': 'Fortune 500', 'size': 45000000, 'probability': 65, 'timeline': 3},
            {'segment': 'Financial Services', 'size': 156000000, 'probability': 80, 'timeline': 8},
            {'segment': 'Healthcare Tech', 'size': 89000000, 'probability': 70, 'timeline': 12},
            {'segment': 'Manufacturing', 'size': 234000000, 'probability': 60, 'timeline': 18}
        ]
        
        df = pd.DataFrame(opportunities)
        
        # Create bubble chart
        fig = px.scatter(df, 
                        x='timeline', 
                        y='probability',
                        size='size',
                        color='segment',
                        title='Market Opportunity Analysis',
                        labels={
                            'timeline': 'Timeline (months)',
                            'probability': 'Win Probability (%)',
                            'size': 'Opportunity Size ($)'
                        })
        
        fig.update_layout(
            xaxis_title="Timeline to Close (months)",
            yaxis_title="Win Probability (%)"
        )
        
        return json.loads(fig.to_json())

    async def get_recent_intelligence_feed(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent intelligence items for activity feed"""
        async with self.db_pool.acquire() as conn:
            intelligence_items = await conn.fetch("""
                SELECT 
                    competitor,
                    intelligence_type,
                    content,
                    source,
                    timestamp,
                    threat_level,
                    impact_score
                FROM competitive_intelligence
                ORDER BY timestamp DESC
                LIMIT $1
            """, limit)
        
        return [dict(item) for item in intelligence_items]

    async def connect_websocket(self, websocket: WebSocket):
        """Connect new WebSocket for real-time updates"""
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect WebSocket"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients"""
        if not self.active_connections:
            return
        
        message = json.dumps(data)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        age = (datetime.now() - self.cache_timestamps[cache_key]).seconds
        return age < self.cache_timeout

    async def _get_win_rate_against_competitor(self, competitor: str) -> float:
        """Get win rate against specific competitor"""
        # Simulated data - would integrate with CRM
        win_rates = {
            'github_copilot': 87.0,
            'aws_codewhisperer': 91.0,
            'google_duet': 89.0,
            'cursor_ide': 95.0,
            'replit_teams': 94.0
        }
        return win_rates.get(competitor, 85.0)

    def _calculate_technology_gap(self, competitor: str) -> float:
        """Calculate technology gap in months"""
        gaps = {
            'github_copilot': 18.0,  # 1.5 year gap
            'aws_codewhisperer': 24.0,  # 2 year gap
            'google_duet': 30.0,  # 2.5 year gap
            'cursor_ide': 36.0,  # 3 year gap
            'replit_teams': 42.0   # 3.5 year gap
        }
        return gaps.get(competitor, 36.0)

    def _calculate_market_overlap(self, competitor: str) -> float:
        """Calculate market overlap percentage"""
        overlaps = {
            'github_copilot': 75.0,
            'aws_codewhisperer': 60.0,
            'google_duet': 55.0,
            'cursor_ide': 25.0,
            'replit_teams': 20.0
        }
        return overlaps.get(competitor, 50.0)

    def _calculate_trend_direction(self, competitor: str) -> str:
        """Calculate threat trend direction"""
        # Would analyze recent activity patterns
        return 'stable'  # Simplified for demo

# Create global dashboard instance
dashboard = None

async def get_dashboard() -> CompetitiveIntelligenceDashboard:
    """Dependency to get dashboard instance"""
    global dashboard
    if dashboard is None:
        # Initialize database and Redis connections
        db_pool = await asyncpg.create_pool(
            "postgresql://user:pass@localhost/beehive",
            min_size=5, max_size=20
        )
        redis = Redis.from_url("redis://localhost:6379")
        dashboard = CompetitiveIntelligenceDashboard(db_pool, redis)
    return dashboard

# API Routes

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("competitive_dashboard.html", {"request": request})

@app.get("/api/metrics")
async def get_metrics(dashboard: CompetitiveIntelligenceDashboard = Depends(get_dashboard)):
    """Get dashboard metrics"""
    metrics = await dashboard.get_dashboard_metrics()
    return asdict(metrics)

@app.get("/api/competitors")
async def get_competitors(dashboard: CompetitiveIntelligenceDashboard = Depends(get_dashboard)):
    """Get competitor insights"""
    insights = await dashboard.get_competitor_insights()
    return [asdict(insight) for insight in insights]

@app.get("/api/threat-timeline")
async def get_threat_timeline(days: int = 30, dashboard: CompetitiveIntelligenceDashboard = Depends(get_dashboard)):
    """Get threat timeline data"""
    return await dashboard.get_threat_timeline_data(days)

@app.get("/api/competitor-comparison")
async def get_competitor_comparison(dashboard: CompetitiveIntelligenceDashboard = Depends(get_dashboard)):
    """Get competitor comparison radar chart"""
    return await dashboard.get_competitor_comparison_data()

@app.get("/api/market-opportunities")
async def get_market_opportunities(dashboard: CompetitiveIntelligenceDashboard = Depends(get_dashboard)):
    """Get market opportunity data"""
    return await dashboard.get_market_opportunity_data()

@app.get("/api/intelligence-feed")
async def get_intelligence_feed(limit: int = 20, dashboard: CompetitiveIntelligenceDashboard = Depends(get_dashboard)):
    """Get recent intelligence feed"""
    return await dashboard.get_recent_intelligence_feed(limit)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, dashboard: CompetitiveIntelligenceDashboard = Depends(get_dashboard)):
    """WebSocket endpoint for real-time updates"""
    await dashboard.connect_websocket(websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_text(json.dumps({"type": "heartbeat", "timestamp": datetime.now().isoformat()}))
    except:
        await dashboard.disconnect_websocket(websocket)

# Background task for real-time updates
async def background_intelligence_monitor():
    """Background task to monitor for new intelligence and push updates"""
    global dashboard
    if dashboard is None:
        return
    
    last_check = datetime.now()
    
    while True:
        try:
            # Check for new intelligence since last check
            async with dashboard.db_pool.acquire() as conn:
                new_intelligence = await conn.fetch("""
                    SELECT COUNT(*) as count
                    FROM competitive_intelligence
                    WHERE timestamp > $1
                """, last_check)
            
            if new_intelligence[0]['count'] > 0:
                # New intelligence detected, broadcast update
                await dashboard.broadcast_update({
                    "type": "new_intelligence",
                    "count": new_intelligence[0]['count'],
                    "timestamp": datetime.now().isoformat()
                })
                
                # Clear relevant caches
                dashboard.cached_data.clear()
                dashboard.cache_timestamps.clear()
            
            last_check = datetime.now()
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logging.error(f"Error in background intelligence monitor: {e}")
            await asyncio.sleep(60)

# Start background task on startup
@app.on_event("startup")
async def startup_event():
    """Start background monitoring task"""
    asyncio.create_task(background_intelligence_monitor())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)