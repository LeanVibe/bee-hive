"""
Strategic Monitoring System Integration Tests

Comprehensive integration tests for the strategic monitoring and analytics system:
- Strategic Market Analytics Engine testing
- Performance Monitoring Dashboard validation  
- Strategic Intelligence System verification
- API endpoint testing and validation
- Database model integrity and performance tests
- End-to-end workflow validation

Tests validate the complete strategic monitoring framework for global market expansion.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, date
from typing import Dict, Any, List
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.main import app
from app.core.database import get_async_session
from app.core.strategic_market_analytics import (
    get_strategic_analytics_engine,
    MarketSegment,
    CompetitivePosition,
    TrendDirection,
    RiskLevel
)
from app.core.performance_monitoring_dashboard import get_performance_dashboard
from app.core.strategic_intelligence_system import (
    get_strategic_intelligence_system,
    IntelligenceType,
    ConfidenceLevel,
    AlertSeverity
)
from app.models.strategic_monitoring import (
    MarketIntelligenceData,
    CompetitorAnalysisData,
    StrategicPerformanceMetrics,
    StrategicRecommendations,
    RiskAssessments,
    IntelligenceAlerts,
    BusinessMetrics,
    StrategicOpportunities
)


class TestStrategicMarketAnalyticsEngine:
    """Test suite for Strategic Market Analytics Engine."""
    
    @pytest.mark.asyncio
    async def test_competitive_landscape_analysis(self):
        """Test competitive landscape analysis functionality."""
        analytics_engine = get_strategic_analytics_engine()
        
        # Test competitive landscape analysis
        result = await analytics_engine.analyze_competitive_landscape(
            segment=MarketSegment.ENTERPRISE,
            region="global",
            depth="comprehensive"
        )
        
        # Validate result structure
        assert "analysis_id" in result
        assert "segment" in result
        assert "region" in result
        assert "summary" in result
        assert "competitors" in result
        assert "recommendations" in result
        
        # Validate summary data
        summary = result["summary"]
        assert "total_competitors" in summary
        assert "market_concentration" in summary
        assert "competitive_intensity" in summary
        assert "our_market_position" in summary
        assert "key_insights" in summary
        
        # Validate competitors data
        competitors = result["competitors"]
        assert isinstance(competitors, list)
        assert len(competitors) > 0
        
        for competitor in competitors:
            assert "competitor_name" in competitor
            assert "market_position" in competitor
            assert "market_share_percent" in competitor
            assert "threat_level" in competitor
    
    @pytest.mark.asyncio
    async def test_market_trends_analysis(self):
        """Test market trends analysis functionality."""
        analytics_engine = get_strategic_analytics_engine()
        
        # Test market trends analysis
        result = await analytics_engine.analyze_market_trends(
            time_horizon_months=12,
            categories=["technology", "business_model", "regulatory"]
        )
        
        # Validate result structure
        assert "analysis_id" in result
        assert "time_horizon_months" in result
        assert "categories" in result
        assert "summary" in result
        assert "trends" in result
        assert "predictions" in result
        
        # Validate summary
        summary = result["summary"]
        assert "total_trends_identified" in summary
        assert "high_impact_trends" in summary
        assert "emerging_trends" in summary
        assert "strategic_priority_trends" in summary
        
        # Validate trends data
        trends = result["trends"]
        assert isinstance(trends, list)
        
        for trend in trends:
            assert "trend_id" in trend
            assert "trend_name" in trend
            assert "category" in trend
            assert "direction" in trend
            assert "impact_score" in trend
            assert "confidence_level" in trend
    
    @pytest.mark.asyncio
    async def test_strategic_intelligence_report_generation(self):
        """Test strategic intelligence report generation."""
        analytics_engine = get_strategic_analytics_engine()
        
        # Test strategic intelligence report
        result = await analytics_engine.generate_strategic_intelligence_report(
            focus_areas=["market_expansion", "competitive_positioning"],
            time_horizon_months=6
        )
        
        # Validate result structure
        assert "report_id" in result
        assert "generated_at" in result
        assert "focus_areas" in result
        assert "executive_summary" in result
        assert "competitive_intelligence" in result
        assert "market_trends" in result
        assert "strategic_opportunities" in result
        assert "strategic_recommendations" in result
        
        # Validate executive summary
        executive_summary = result["executive_summary"]
        assert isinstance(executive_summary, dict)
        
        # Validate competitive intelligence
        competitive_intelligence = result["competitive_intelligence"]
        assert isinstance(competitive_intelligence, dict)
        
        # Validate recommendations
        recommendations = result["strategic_recommendations"]
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_market_share_evolution_tracking(self):
        """Test market share evolution tracking."""
        analytics_engine = get_strategic_analytics_engine()
        
        # Test market share evolution
        result = await analytics_engine.track_market_share_evolution(
            segment=MarketSegment.ENTERPRISE,
            region="global",
            periods=12
        )
        
        # Validate result structure
        assert "analysis_id" in result
        assert "segment" in result
        assert "region" in result
        assert "periods_analyzed" in result
        assert "historical_data" in result
        assert "share_evolution" in result
        assert "future_projections" in result
        assert "strategic_insights" in result


class TestPerformanceMonitoringDashboard:
    """Test suite for Performance Monitoring Dashboard."""
    
    @pytest.mark.asyncio
    async def test_performance_dashboard_generation(self):
        """Test comprehensive performance dashboard generation."""
        dashboard = get_performance_dashboard()
        
        # Test dashboard generation
        result = await dashboard.generate_performance_dashboard(
            period="current_month",
            include_forecasts=True
        )
        
        # Validate dashboard structure
        assert result.dashboard_id is not None
        assert result.generated_at is not None
        assert result.overall_performance_score is not None
        assert result.enterprise_metrics is not None
        assert result.community_metrics is not None
        assert result.thought_leadership_metrics is not None
        assert result.revenue_metrics is not None
        assert result.key_insights is not None
        assert result.performance_alerts is not None
        assert result.strategic_recommendations is not None
        
        # Validate enterprise metrics
        enterprise = result.enterprise_metrics
        assert enterprise.total_partnerships is not None
        assert enterprise.conversion_rate_percent is not None
        assert enterprise.pipeline_value_usd is not None
        
        # Validate community metrics
        community = result.community_metrics
        assert community.total_members is not None
        assert community.growth_rate_percent is not None
        assert community.engagement_rate_percent is not None
        
        # Validate thought leadership metrics
        tl = result.thought_leadership_metrics
        assert tl.influence_score is not None
        assert tl.content_pieces_published is not None
        assert tl.brand_sentiment_score is not None
        
        # Validate revenue metrics
        revenue = result.revenue_metrics
        assert revenue.total_revenue_usd is not None
        assert revenue.revenue_growth_rate is not None
        assert revenue.gross_margin_percent is not None
    
    @pytest.mark.asyncio
    async def test_enterprise_partnership_monitoring(self):
        """Test enterprise partnership monitoring."""
        dashboard = get_performance_dashboard()
        
        # Test enterprise partnership monitoring
        metrics = await dashboard.monitor_enterprise_partnerships()
        
        # Validate metrics structure
        assert metrics.total_partnerships is not None
        assert metrics.active_partnerships is not None
        assert metrics.pipeline_value_usd is not None
        assert metrics.conversion_rate_percent is not None
        assert metrics.average_deal_size_usd is not None
        assert metrics.partner_satisfaction_score is not None
        assert metrics.revenue_from_partnerships_usd is not None
        assert metrics.partnership_growth_rate is not None
        assert isinstance(metrics.top_performing_partners, list)
        assert isinstance(metrics.partnership_distribution, dict)
        
        # Validate value ranges
        assert 0 <= metrics.conversion_rate_percent <= 100
        assert 0 <= metrics.partner_satisfaction_score <= 10
        assert 0 <= metrics.churn_rate_percent <= 100
    
    @pytest.mark.asyncio
    async def test_community_growth_monitoring(self):
        """Test community growth monitoring."""
        dashboard = get_performance_dashboard()
        
        # Test community growth monitoring
        metrics = await dashboard.monitor_community_growth()
        
        # Validate metrics structure
        assert metrics.total_members is not None
        assert metrics.active_monthly_users is not None
        assert metrics.daily_active_users is not None
        assert metrics.growth_rate_percent is not None
        assert metrics.engagement_rate_percent is not None
        assert metrics.viral_coefficient is not None
        assert metrics.retention_rate_90_day is not None
        assert metrics.community_health_score is not None
        assert isinstance(metrics.top_contributing_members, list)
        assert isinstance(metrics.geographic_distribution, dict)
        
        # Validate value ranges
        assert 0 <= metrics.engagement_rate_percent <= 100
        assert 0 <= metrics.retention_rate_90_day <= 100
        assert 0 <= metrics.community_health_score <= 10
        assert -1 <= metrics.sentiment_score <= 1
    
    @pytest.mark.asyncio
    async def test_thought_leadership_monitoring(self):
        """Test thought leadership monitoring."""
        dashboard = get_performance_dashboard()
        
        # Test thought leadership monitoring
        metrics = await dashboard.monitor_thought_leadership()
        
        # Validate metrics structure
        assert metrics.content_pieces_published is not None
        assert metrics.total_views is not None
        assert metrics.total_shares is not None
        assert metrics.influence_score is not None
        assert metrics.brand_sentiment_score is not None
        assert metrics.speaking_engagements is not None
        assert isinstance(metrics.content_performance_scores, dict)
        
        # Validate value ranges
        assert 0 <= metrics.influence_score <= 100
        assert -1 <= metrics.brand_sentiment_score <= 1
    
    @pytest.mark.asyncio
    async def test_revenue_metrics_monitoring(self):
        """Test revenue metrics monitoring."""
        dashboard = get_performance_dashboard()
        
        # Test revenue metrics monitoring
        metrics = await dashboard.monitor_revenue_metrics()
        
        # Validate metrics structure
        assert metrics.total_revenue_usd is not None
        assert metrics.recurring_revenue_usd is not None
        assert metrics.revenue_growth_rate is not None
        assert metrics.customer_acquisition_cost_usd is not None
        assert metrics.customer_lifetime_value_usd is not None
        assert metrics.monthly_recurring_revenue is not None
        assert metrics.churn_rate_percent is not None
        assert metrics.gross_margin_percent is not None
        
        # Validate value ranges
        assert 0 <= metrics.churn_rate_percent <= 100
        assert 0 <= metrics.gross_margin_percent <= 100
        assert 0 <= metrics.revenue_forecast_confidence <= 1


class TestStrategicIntelligenceSystem:
    """Test suite for Strategic Intelligence System."""
    
    @pytest.mark.asyncio
    async def test_competitive_intelligence_report_generation(self):
        """Test competitive intelligence report generation."""
        intelligence_system = get_strategic_intelligence_system()
        
        # Test competitive intelligence report
        result = await intelligence_system.generate_competitive_intelligence_report(
            competitor_name="Microsoft",
            analysis_depth="comprehensive"
        )
        
        # Validate result structure
        assert result.analysis_id is not None
        assert result.competitor_name == "Microsoft"
        assert result.analysis_timestamp is not None
        assert isinstance(result.key_findings, list)
        assert isinstance(result.strategic_moves, list)
        assert isinstance(result.opportunities, list)
        assert isinstance(result.risks, list)
        assert result.confidence_level is not None
        assert isinstance(result.data_sources, list)
        assert result.next_analysis_due is not None
        
        # Validate threat assessment
        threat_assessment = result.threat_assessment
        assert isinstance(threat_assessment, dict)
        assert "overall_threat_score" in threat_assessment
        assert "threat_level" in threat_assessment
    
    @pytest.mark.asyncio
    async def test_market_opportunities_analysis(self):
        """Test market opportunities analysis."""
        intelligence_system = get_strategic_intelligence_system()
        
        # Test market opportunities analysis
        result = await intelligence_system.analyze_market_opportunities(
            segment=MarketSegment.ENTERPRISE,
            region="global",
            time_horizon_months=12
        )
        
        # Validate result structure
        assert result.analysis_id is not None
        assert result.market_segment == MarketSegment.ENTERPRISE
        assert result.region == "global"
        assert result.analysis_timestamp is not None
        assert isinstance(result.growth_opportunities, list)
        assert isinstance(result.emerging_trends, list)
        assert isinstance(result.regulatory_changes, list)
        assert isinstance(result.technology_disruptions, list)
        assert result.confidence_level is not None
        assert isinstance(result.impact_assessment, dict)
        assert isinstance(result.strategic_implications, list)
    
    @pytest.mark.asyncio
    async def test_strategic_risk_assessment(self):
        """Test strategic risk assessment."""
        intelligence_system = get_strategic_intelligence_system()
        
        # Test risk assessment
        result = await intelligence_system.perform_strategic_risk_assessment(
            risk_categories=["competitive", "market", "technology"],
            time_horizon_months=12
        )
        
        # Validate result structure
        assert isinstance(result, list)
        assert len(result) > 0
        
        for risk in result:
            assert risk.assessment_id is not None
            assert risk.risk_category is not None
            assert risk.risk_description is not None
            assert 0 <= risk.probability <= 1
            assert 0 <= risk.impact_score <= 100
            assert risk.risk_level is not None
            assert isinstance(risk.mitigation_strategies, list)
            assert isinstance(risk.monitoring_indicators, list)
    
    @pytest.mark.asyncio
    async def test_strategic_recommendations_generation(self):
        """Test strategic recommendations generation."""
        intelligence_system = get_strategic_intelligence_system()
        
        # Test recommendations generation
        result = await intelligence_system.generate_strategic_recommendations(
            focus_areas=["market_expansion", "competitive_positioning"],
            confidence_threshold=0.6
        )
        
        # Validate result structure
        assert isinstance(result, list)
        
        for recommendation in result:
            assert recommendation.recommendation_id is not None
            assert recommendation.title is not None
            assert recommendation.description is not None
            assert recommendation.category is not None
            assert recommendation.priority is not None
            assert recommendation.confidence_level is not None
            assert isinstance(recommendation.success_metrics, list)
            assert isinstance(recommendation.action_items, list)


class TestStrategicMonitoringAPI:
    """Test suite for Strategic Monitoring API endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_competitive_landscape_api(self):
        """Test competitive landscape analysis API."""
        response = self.client.get(
            "/strategic-monitoring/market-intelligence/competitive-landscape",
            params={
                "segment": "enterprise",
                "region": "global",
                "depth": "comprehensive"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "data" in data
        assert "metadata" in data
        
        # Validate data structure
        result_data = data["data"]
        assert "analysis_id" in result_data
        assert "segment" in result_data
        assert "competitors" in result_data
    
    def test_market_trends_api(self):
        """Test market trends analysis API."""
        response = self.client.get(
            "/strategic-monitoring/market-intelligence/trends",
            params={
                "time_horizon_months": 12,
                "categories": "technology,business_model"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "data" in data
        
        # Validate data structure
        result_data = data["data"]
        assert "analysis_id" in result_data
        assert "trends" in result_data
    
    def test_performance_dashboard_api(self):
        """Test performance dashboard API."""
        response = self.client.get(
            "/strategic-monitoring/performance/dashboard",
            params={
                "period": "current_month",
                "include_forecasts": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "data" in data
        
        # Validate dashboard data
        dashboard_data = data["data"]
        assert "dashboard_id" in dashboard_data
        assert "overall_performance_score" in dashboard_data
        assert "enterprise_metrics" in dashboard_data
        assert "community_metrics" in dashboard_data
    
    def test_enterprise_partnerships_api(self):
        """Test enterprise partnerships API."""
        response = self.client.get(
            "/strategic-monitoring/performance/enterprise-partnerships"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "data" in data
        
        # Validate metrics data
        metrics_data = data["data"]
        assert "total_partnerships" in metrics_data
        assert "conversion_rate_percent" in metrics_data
        assert "pipeline_value_usd" in metrics_data
    
    def test_intelligence_alerts_api(self):
        """Test intelligence alerts API."""
        response = self.client.get(
            "/strategic-monitoring/alerts",
            params={
                "unresolved_only": True,
                "limit": 20
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "data" in data
        assert "metadata" in data
        
        # Validate metadata
        metadata = data["metadata"]
        assert "total_alerts" in metadata
        assert "filters_applied" in metadata
    
    def test_system_health_api(self):
        """Test system health API."""
        response = self.client.get("/strategic-monitoring/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "data" in data
        
        # Validate health data
        health_data = data["data"]
        assert "status" in health_data
        assert "components" in health_data
        assert "metrics" in health_data


class TestDatabaseModels:
    """Test suite for Strategic Monitoring database models."""
    
    @pytest.mark.asyncio
    async def test_market_intelligence_data_model(self):
        """Test MarketIntelligenceData model."""
        async for session in get_async_session():
            # Create test data
            intelligence_data = MarketIntelligenceData(
                analysis_id=str(uuid4()),
                market_segment="enterprise",
                region="global",
                market_size_usd=1000000.0,
                growth_rate_percent=15.5,
                key_players=["Microsoft", "Google", "Amazon"],
                adoption_rate_percent=25.0,
                sentiment_score=0.7,
                confidence_level="high",
                data_sources=["market_research", "financial_reports"]
            )
            
            session.add(intelligence_data)
            await session.commit()
            
            # Verify data was saved
            query = select(MarketIntelligenceData).where(
                MarketIntelligenceData.analysis_id == intelligence_data.analysis_id
            )
            result = await session.execute(query)
            saved_data = result.scalar_one_or_none()
            
            assert saved_data is not None
            assert saved_data.market_segment == "enterprise"
            assert saved_data.growth_rate_percent == 15.5
            assert len(saved_data.key_players) == 3
            
            # Clean up
            await session.delete(saved_data)
            await session.commit()
            break
    
    @pytest.mark.asyncio
    async def test_strategic_performance_metrics_model(self):
        """Test StrategicPerformanceMetrics model."""
        async for session in get_async_session():
            # Create test data
            metrics = StrategicPerformanceMetrics(
                metric_id=str(uuid4()),
                metric_name="enterprise_partnerships_count",
                category="enterprise_partnerships",
                current_value=35.0,
                target_value=50.0,
                unit="count",
                status="on_target",
                trend_direction="increasing",
                change_percent=12.5,
                confidence_level=0.85,
                data_sources=["crm", "sales_reports"],
                reporting_period="current_month"
            )
            
            session.add(metrics)
            await session.commit()
            
            # Verify data was saved
            query = select(StrategicPerformanceMetrics).where(
                StrategicPerformanceMetrics.metric_id == metrics.metric_id
            )
            result = await session.execute(query)
            saved_metrics = result.scalar_one_or_none()
            
            assert saved_metrics is not None
            assert saved_metrics.metric_name == "enterprise_partnerships_count"
            assert saved_metrics.current_value == 35.0
            assert saved_metrics.confidence_level == 0.85
            
            # Clean up
            await session.delete(saved_metrics)
            await session.commit()
            break
    
    @pytest.mark.asyncio
    async def test_strategic_recommendations_model(self):
        """Test StrategicRecommendations model."""
        async for session in get_async_session():
            # Create test data
            recommendation = StrategicRecommendations(
                recommendation_id=str(uuid4()),
                title="Accelerate Enterprise Partnership Program",
                description="Increase investment in enterprise partnership development",
                category="partnership_development",
                priority_score=85.0,
                confidence_level="high",
                investment_required_usd=500000.0,
                expected_roi_percent=250.0,
                time_to_impact_months=6,
                risk_assessment="medium",
                success_metrics=["partnership_count", "revenue_growth"],
                action_items=["hire_partnership_manager", "develop_partner_portal"],
                stakeholders=["ceo", "sales_director", "partnership_manager"]
            )
            
            session.add(recommendation)
            await session.commit()
            
            # Verify data was saved
            query = select(StrategicRecommendations).where(
                StrategicRecommendations.recommendation_id == recommendation.recommendation_id
            )
            result = await session.execute(query)
            saved_rec = result.scalar_one_or_none()
            
            assert saved_rec is not None
            assert saved_rec.title == "Accelerate Enterprise Partnership Program"
            assert saved_rec.priority_score == 85.0
            assert len(saved_rec.success_metrics) == 2
            
            # Clean up
            await session.delete(saved_rec)
            await session.commit()
            break


class TestEndToEndWorkflows:
    """Test suite for end-to-end strategic monitoring workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_competitive_analysis_workflow(self):
        """Test complete competitive analysis workflow."""
        # 1. Generate competitive intelligence
        intelligence_system = get_strategic_intelligence_system()
        competitive_intel = await intelligence_system.generate_competitive_intelligence_report(
            competitor_name="Microsoft",
            analysis_depth="comprehensive"
        )
        
        assert competitive_intel.competitor_name == "Microsoft"
        assert len(competitive_intel.key_findings) > 0
        
        # 2. Analyze market opportunities based on competitive intelligence
        market_intel = await intelligence_system.analyze_market_opportunities(
            segment=MarketSegment.ENTERPRISE,
            region="global",
            time_horizon_months=12
        )
        
        assert market_intel.market_segment == MarketSegment.ENTERPRISE
        assert len(market_intel.growth_opportunities) >= 0
        
        # 3. Generate strategic recommendations
        recommendations = await intelligence_system.generate_strategic_recommendations(
            focus_areas=["competitive_positioning", "market_expansion"],
            confidence_threshold=0.6
        )
        
        assert isinstance(recommendations, list)
        
        # 4. Validate workflow integration
        assert competitive_intel.analysis_timestamp is not None
        assert market_intel.analysis_timestamp is not None
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_workflow(self):
        """Test complete performance monitoring workflow."""
        # 1. Generate performance dashboard
        dashboard = get_performance_dashboard()
        dashboard_data = await dashboard.generate_performance_dashboard(
            period="current_month",
            include_forecasts=True
        )
        
        assert dashboard_data.overall_performance_score is not None
        assert dashboard_data.enterprise_metrics is not None
        
        # 2. Track individual KPIs
        kpi_metric = await dashboard.track_kpi_performance(
            kpi_name="enterprise_partnerships_count",
            category="enterprise_partnerships",
            time_range_days=30
        )
        
        assert kpi_metric.name == "enterprise_partnerships_count"
        assert kpi_metric.current_value is not None
        
        # 3. Monitor specific metrics
        enterprise_metrics = await dashboard.monitor_enterprise_partnerships()
        community_metrics = await dashboard.monitor_community_growth()
        
        assert enterprise_metrics.total_partnerships is not None
        assert community_metrics.total_members is not None
        
        # 4. Validate integration
        assert dashboard_data.generated_at is not None
        assert kpi_metric.last_updated is not None


# Performance and Load Tests

class TestPerformanceAndLoad:
    """Test suite for performance and load testing."""
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_performance(self):
        """Test performance with concurrent analysis requests."""
        analytics_engine = get_strategic_analytics_engine()
        
        # Create multiple concurrent analysis tasks
        tasks = []
        for i in range(5):
            task = analytics_engine.analyze_competitive_landscape(
                segment=MarketSegment.ENTERPRISE,
                region="global",
                depth="basic"
            )
            tasks.append(task)
        
        # Execute tasks concurrently
        start_time = datetime.utcnow()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.utcnow()
        
        # Validate results
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 30  # Should complete within 30 seconds
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5
        
        for result in successful_results:
            assert "analysis_id" in result
            assert "competitors" in result
    
    @pytest.mark.asyncio
    async def test_api_response_time_performance(self):
        """Test API response time performance."""
        client = TestClient(app)
        
        # Test multiple API endpoints
        endpoints = [
            "/strategic-monitoring/performance/dashboard",
            "/strategic-monitoring/performance/enterprise-partnerships",
            "/strategic-monitoring/performance/community-growth",
            "/strategic-monitoring/health"
        ]
        
        for endpoint in endpoints:
            start_time = datetime.utcnow()
            response = client.get(endpoint)
            end_time = datetime.utcnow()
            
            response_time = (end_time - start_time).total_seconds()
            
            assert response.status_code == 200
            assert response_time < 5.0  # Should respond within 5 seconds
            
            data = response.json()
            assert data["status"] == "success"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])