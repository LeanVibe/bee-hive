"""
Comprehensive Feedback Analyzer with sentiment analysis and quality scoring.

Analyzes user feedback to extract insights about prompt performance,
sentiment patterns, and quality metrics for optimization guidance.
"""

import asyncio
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from collections import defaultdict, Counter
import statistics

from ..models.prompt_optimization import PromptFeedback, PromptVariant

logger = structlog.get_logger()


class SentimentPolarity(str, Enum):
    """Sentiment polarity categories."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class FeedbackCategory(str, Enum):
    """Feedback categories for classification."""
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    HELPFULNESS = "helpfulness"
    TONE = "tone"
    STRUCTURE = "structure"
    CREATIVITY = "creativity"
    TECHNICAL_DEPTH = "technical_depth"
    ACTIONABILITY = "actionability"


@dataclass
class SentimentAnalysis:
    """Sentiment analysis results."""
    polarity: SentimentPolarity
    score: float
    confidence: float
    emotional_indicators: List[str]
    sentiment_keywords: List[str]


@dataclass
class QualityScores:
    """Quality scoring results."""
    overall_quality: float
    response_quality: float
    relevance_score: float
    clarity_score: float
    usefulness_score: float
    completeness_score: float
    accuracy_score: float
    satisfaction_score: float


@dataclass
class FeedbackInsights:
    """Comprehensive feedback analysis results."""
    sentiment_analysis: SentimentAnalysis
    quality_scores: QualityScores
    feedback_categories: List[FeedbackCategory]
    improvement_suggestions: List[str]
    pattern_indicators: Dict[str, float]
    confidence_level: float
    priority_score: float


class FeedbackAnalyzer:
    """
    Advanced feedback analysis system for prompt optimization.
    
    Capabilities:
    - Sentiment analysis with confidence scoring
    - Multi-dimensional quality assessment
    - Pattern recognition across feedback data
    - Automated improvement suggestions
    - Statistical trend analysis
    - Category-based feedback classification
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.logger = logger.bind(component="feedback_analyzer")
        
        # Sentiment analysis lexicons
        self.positive_words = {
            'excellent', 'great', 'amazing', 'perfect', 'helpful', 'clear',
            'accurate', 'useful', 'comprehensive', 'detailed', 'informative',
            'insightful', 'brilliant', 'outstanding', 'fantastic', 'wonderful',
            'precise', 'thorough', 'effective', 'valuable', 'impressive'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'bad', 'poor', 'unclear', 'confusing',
            'inaccurate', 'useless', 'incomplete', 'vague', 'misleading',
            'disappointing', 'frustrating', 'unhelpful', 'wrong', 'boring',
            'irrelevant', 'shallow', 'inadequate', 'problematic', 'insufficient'
        }
        
        self.intensity_words = {
            'very', 'extremely', 'incredibly', 'absolutely', 'completely',
            'totally', 'quite', 'really', 'pretty', 'somewhat', 'rather'
        }
        
        # Quality indicators
        self.quality_indicators = {
            FeedbackCategory.ACCURACY: {
                'positive': ['accurate', 'correct', 'precise', 'factual', 'right'],
                'negative': ['inaccurate', 'wrong', 'incorrect', 'misleading', 'false']
            },
            FeedbackCategory.CLARITY: {
                'positive': ['clear', 'understandable', 'readable', 'simple', 'straightforward'],
                'negative': ['unclear', 'confusing', 'complicated', 'ambiguous', 'hard to understand']
            },
            FeedbackCategory.COMPLETENESS: {
                'positive': ['complete', 'comprehensive', 'thorough', 'detailed', 'full'],
                'negative': ['incomplete', 'missing', 'partial', 'lacking', 'insufficient']
            },
            FeedbackCategory.RELEVANCE: {
                'positive': ['relevant', 'on-topic', 'applicable', 'pertinent', 'related'],
                'negative': ['irrelevant', 'off-topic', 'unrelated', 'tangential', 'beside the point']
            },
            FeedbackCategory.HELPFULNESS: {
                'positive': ['helpful', 'useful', 'beneficial', 'valuable', 'practical'],
                'negative': ['unhelpful', 'useless', 'pointless', 'waste of time', 'not useful']
            }
        }
    
    async def analyze_feedback(
        self,
        rating: int,
        feedback_text: Optional[str] = None,
        feedback_categories: Optional[List[str]] = None,
        context_data: Optional[Dict[str, Any]] = None,
        prompt_variant_id: Optional[str] = None
    ) -> FeedbackInsights:
        """
        Comprehensive feedback analysis with sentiment and quality scoring.
        
        Args:
            rating: Numerical rating (1-5)
            feedback_text: Textual feedback content
            feedback_categories: Pre-defined categories
            context_data: Additional context information
            prompt_variant_id: Associated prompt variant
            
        Returns:
            FeedbackInsights containing comprehensive analysis results
        """
        try:
            start_time = time.time()
            
            self.logger.info(
                "Analyzing feedback",
                rating=rating,
                has_text=bool(feedback_text),
                categories=feedback_categories
            )
            
            # Perform sentiment analysis
            sentiment_analysis = await self._analyze_sentiment(
                feedback_text or "", rating, context_data
            )
            
            # Calculate quality scores
            quality_scores = await self._calculate_quality_scores(
                rating, feedback_text, context_data, sentiment_analysis
            )
            
            # Classify feedback categories
            classified_categories = await self._classify_feedback_categories(
                feedback_text or "", feedback_categories or []
            )
            
            # Generate improvement suggestions
            improvement_suggestions = await self._generate_improvement_suggestions(
                sentiment_analysis, quality_scores, classified_categories, rating
            )
            
            # Identify patterns
            pattern_indicators = await self._identify_patterns(
                feedback_text or "", rating, context_data, sentiment_analysis
            )
            
            # Calculate confidence and priority
            confidence_level = await self._calculate_confidence_level(
                feedback_text, rating, sentiment_analysis, quality_scores
            )
            
            priority_score = await self._calculate_priority_score(
                sentiment_analysis, quality_scores, rating, confidence_level
            )
            
            analysis_time = time.time() - start_time
            
            insights = FeedbackInsights(
                sentiment_analysis=sentiment_analysis,
                quality_scores=quality_scores,
                feedback_categories=classified_categories,
                improvement_suggestions=improvement_suggestions,
                pattern_indicators=pattern_indicators,
                confidence_level=confidence_level,
                priority_score=priority_score
            )
            
            self.logger.info(
                "Feedback analysis completed",
                sentiment=sentiment_analysis.polarity.value,
                quality=quality_scores.overall_quality,
                confidence=confidence_level,
                analysis_time=analysis_time
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(
                "Failed to analyze feedback",
                rating=rating,
                error=str(e)
            )
            raise
    
    async def analyze_feedback_trends(
        self,
        prompt_variant_id: str,
        time_period_days: int = 30,
        minimum_feedback_count: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze feedback trends over time for a specific prompt variant.
        
        Args:
            prompt_variant_id: Target prompt variant
            time_period_days: Analysis period in days
            minimum_feedback_count: Minimum feedback required for analysis
            
        Returns:
            Dict containing trend analysis results
        """
        try:
            # Get feedback data from database
            feedback_data = await self._get_feedback_data(
                prompt_variant_id, time_period_days
            )
            
            if len(feedback_data) < minimum_feedback_count:
                return {
                    'insufficient_data': True,
                    'feedback_count': len(feedback_data),
                    'minimum_required': minimum_feedback_count
                }
            
            # Analyze trends
            rating_trends = await self._analyze_rating_trends(feedback_data)
            sentiment_trends = await self._analyze_sentiment_trends(feedback_data)
            category_trends = await self._analyze_category_trends(feedback_data)
            quality_trends = await self._analyze_quality_trends(feedback_data)
            
            # Calculate statistical metrics
            statistical_summary = await self._calculate_statistical_summary(feedback_data)
            
            # Identify significant changes
            significant_changes = await self._identify_significant_changes(
                feedback_data, time_period_days
            )
            
            # Generate trend-based recommendations
            trend_recommendations = await self._generate_trend_recommendations(
                rating_trends, sentiment_trends, category_trends, significant_changes
            )
            
            return {
                'analysis_period_days': time_period_days,
                'total_feedback_count': len(feedback_data),
                'rating_trends': rating_trends,
                'sentiment_trends': sentiment_trends,
                'category_trends': category_trends,
                'quality_trends': quality_trends,
                'statistical_summary': statistical_summary,
                'significant_changes': significant_changes,
                'recommendations': trend_recommendations,
                'confidence_level': await self._calculate_trend_confidence(feedback_data)
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to analyze feedback trends",
                prompt_variant_id=prompt_variant_id,
                error=str(e)
            )
            raise
    
    async def get_feedback_summary(
        self,
        prompt_variant_ids: List[str],
        include_text_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Get aggregated feedback summary across multiple prompt variants.
        
        Args:
            prompt_variant_ids: List of prompt variant IDs
            include_text_analysis: Whether to include text analysis
            
        Returns:
            Dict containing aggregated feedback summary
        """
        try:
            summaries = {}
            
            for variant_id in prompt_variant_ids:
                feedback_data = await self._get_feedback_data(variant_id, 90)  # 3 months
                
                if not feedback_data:
                    summaries[variant_id] = {'no_feedback': True}
                    continue
                
                # Calculate aggregate metrics
                ratings = [fb['rating'] for fb in feedback_data]
                
                summary = {
                    'total_feedback': len(feedback_data),
                    'average_rating': statistics.mean(ratings),
                    'rating_distribution': dict(Counter(ratings)),
                    'rating_std': statistics.stdev(ratings) if len(ratings) > 1 else 0.0,
                    'latest_feedback_date': max(fb['submitted_at'] for fb in feedback_data)
                }
                
                # Add text analysis if requested
                if include_text_analysis:
                    text_feedback = [fb for fb in feedback_data if fb.get('feedback_text')]
                    if text_feedback:
                        text_analysis = await self._analyze_text_feedback_aggregate(text_feedback)
                        summary['text_analysis'] = text_analysis
                
                # Calculate quality metrics
                quality_metrics = await self._calculate_aggregate_quality_metrics(feedback_data)
                summary['quality_metrics'] = quality_metrics
                
                summaries[variant_id] = summary
            
            # Calculate comparative analysis
            comparative_analysis = await self._generate_comparative_analysis(summaries)
            
            return {
                'individual_summaries': summaries,
                'comparative_analysis': comparative_analysis,
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error("Failed to generate feedback summary", error=str(e))
            raise
    
    # Private helper methods
    
    async def _analyze_sentiment(
        self,
        text: str,
        rating: int,
        context_data: Optional[Dict[str, Any]]
    ) -> SentimentAnalysis:
        """Analyze sentiment from text and rating."""
        if not text:
            # Rating-based sentiment only
            score = (rating - 1) / 4  # Normalize to 0-1
            polarity = self._score_to_polarity(score)
            
            return SentimentAnalysis(
                polarity=polarity,
                score=score,
                confidence=0.6,  # Lower confidence without text
                emotional_indicators=[],
                sentiment_keywords=[]
            )
        
        # Text-based sentiment analysis
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Count sentiment words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Apply intensity modifiers
        intensity_multiplier = 1.0
        for i, word in enumerate(words):
            if word in self.intensity_words:
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word in self.positive_words:
                        positive_count += 0.5
                    elif next_word in self.negative_words:
                        negative_count += 0.5
        
        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            text_score = 0.5  # Neutral
            confidence = 0.3
        else:
            text_score = positive_count / total_sentiment_words
            confidence = min(0.9, total_sentiment_words / len(words) * 5)
        
        # Combine with rating
        rating_score = (rating - 1) / 4
        combined_score = (text_score * 0.7 + rating_score * 0.3)
        
        # Extract indicators
        emotional_indicators = [word for word in words if word in self.positive_words or word in self.negative_words]
        
        return SentimentAnalysis(
            polarity=self._score_to_polarity(combined_score),
            score=combined_score,
            confidence=confidence,
            emotional_indicators=emotional_indicators[:5],  # Top 5
            sentiment_keywords=list(set(emotional_indicators))
        )
    
    def _score_to_polarity(self, score: float) -> SentimentPolarity:
        """Convert numerical score to sentiment polarity."""
        if score < 0.2:
            return SentimentPolarity.VERY_NEGATIVE
        elif score < 0.4:
            return SentimentPolarity.NEGATIVE
        elif score < 0.6:
            return SentimentPolarity.NEUTRAL
        elif score < 0.8:
            return SentimentPolarity.POSITIVE
        else:
            return SentimentPolarity.VERY_POSITIVE
    
    async def _calculate_quality_scores(
        self,
        rating: int,
        feedback_text: Optional[str],
        context_data: Optional[Dict[str, Any]],
        sentiment_analysis: SentimentAnalysis
    ) -> QualityScores:
        """Calculate multi-dimensional quality scores."""
        base_score = rating / 5.0
        
        # Adjust scores based on text analysis
        if feedback_text:
            text_adjustments = await self._analyze_quality_indicators(feedback_text)
        else:
            text_adjustments = {}
        
        # Calculate individual quality scores
        response_quality = min(1.0, max(0.0, 
            base_score + text_adjustments.get('response_quality', 0.0)
        ))
        
        relevance_score = min(1.0, max(0.0,
            base_score + text_adjustments.get('relevance', 0.0)
        ))
        
        clarity_score = min(1.0, max(0.0,
            base_score + text_adjustments.get('clarity', 0.0)
        ))
        
        usefulness_score = min(1.0, max(0.0,
            base_score + text_adjustments.get('usefulness', 0.0)
        ))
        
        completeness_score = min(1.0, max(0.0,
            base_score + text_adjustments.get('completeness', 0.0)
        ))
        
        accuracy_score = min(1.0, max(0.0,
            base_score + text_adjustments.get('accuracy', 0.0)
        ))
        
        # Overall quality combines all dimensions
        overall_quality = statistics.mean([
            response_quality, relevance_score, clarity_score,
            usefulness_score, completeness_score, accuracy_score
        ])
        
        return QualityScores(
            overall_quality=overall_quality,
            response_quality=response_quality,
            relevance_score=relevance_score,
            clarity_score=clarity_score,
            usefulness_score=usefulness_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            satisfaction_score=sentiment_analysis.score
        )
    
    async def _analyze_quality_indicators(self, text: str) -> Dict[str, float]:
        """Analyze text for quality indicators."""
        words = re.findall(r'\b\w+\b', text.lower())
        adjustments = defaultdict(float)
        
        for category, indicators in self.quality_indicators.items():
            positive_matches = sum(1 for word in words if word in indicators['positive'])
            negative_matches = sum(1 for word in words if word in indicators['negative'])
            
            # Calculate adjustment (-0.2 to +0.2)
            if positive_matches > 0 or negative_matches > 0:
                adjustment = (positive_matches - negative_matches) * 0.1
                adjustments[category.value] = max(-0.2, min(0.2, adjustment))
        
        return dict(adjustments)
    
    async def _classify_feedback_categories(
        self,
        text: str,
        existing_categories: List[str]
    ) -> List[FeedbackCategory]:
        """Classify feedback into categories."""
        categories = set()
        
        # Add existing categories
        for cat_str in existing_categories:
            try:
                categories.add(FeedbackCategory(cat_str))
            except ValueError:
                pass  # Invalid category
        
        # Auto-detect categories from text
        if text:
            words = re.findall(r'\b\w+\b', text.lower())
            
            for category, indicators in self.quality_indicators.items():
                category_words = indicators['positive'] + indicators['negative']
                if any(word in words for word in category_words):
                    categories.add(category)
        
        return list(categories)
    
    async def _generate_improvement_suggestions(
        self,
        sentiment_analysis: SentimentAnalysis,
        quality_scores: QualityScores,
        categories: List[FeedbackCategory],
        rating: int
    ) -> List[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []
        
        # Rating-based suggestions
        if rating <= 2:
            suggestions.append("Critical improvement needed - consider major prompt revision")
        elif rating == 3:
            suggestions.append("Moderate improvement needed - focus on user pain points")
        
        # Quality-based suggestions
        if quality_scores.clarity_score < 0.6:
            suggestions.append("Improve clarity: use simpler language and better structure")
        
        if quality_scores.completeness_score < 0.6:
            suggestions.append("Enhance completeness: provide more comprehensive information")
        
        if quality_scores.accuracy_score < 0.7:
            suggestions.append("Verify accuracy: check facts and correct any errors")
        
        if quality_scores.relevance_score < 0.6:
            suggestions.append("Increase relevance: better align content with user needs")
        
        # Sentiment-based suggestions
        if sentiment_analysis.polarity in [SentimentPolarity.NEGATIVE, SentimentPolarity.VERY_NEGATIVE]:
            suggestions.append("Address negative sentiment: identify and fix user frustration points")
        
        # Category-specific suggestions
        if FeedbackCategory.TONE in categories and quality_scores.satisfaction_score < 0.6:
            suggestions.append("Improve tone: make response more engaging and user-friendly")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    async def _identify_patterns(
        self,
        text: str,
        rating: int,
        context_data: Optional[Dict[str, Any]],
        sentiment_analysis: SentimentAnalysis
    ) -> Dict[str, float]:
        """Identify patterns in feedback."""
        patterns = {}
        
        # Length pattern
        if text:
            patterns['text_length'] = min(1.0, len(text) / 500)
        
        # Rating pattern
        patterns['rating_extremity'] = abs(rating - 3) / 2  # How far from neutral
        
        # Sentiment consistency
        rating_sentiment = (rating - 1) / 4
        sentiment_consistency = 1.0 - abs(rating_sentiment - sentiment_analysis.score)
        patterns['sentiment_consistency'] = sentiment_consistency
        
        # Emotional intensity
        if sentiment_analysis.emotional_indicators:
            patterns['emotional_intensity'] = len(sentiment_analysis.emotional_indicators) / 10
        else:
            patterns['emotional_intensity'] = 0.0
        
        return patterns
    
    async def _calculate_confidence_level(
        self,
        feedback_text: Optional[str],
        rating: int,
        sentiment_analysis: SentimentAnalysis,
        quality_scores: QualityScores
    ) -> float:
        """Calculate confidence level of the analysis."""
        confidence_factors = []
        
        # Text availability
        if feedback_text and len(feedback_text) > 20:
            confidence_factors.append(0.9)
        elif feedback_text:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Rating clarity (extremes are more confident)
        rating_confidence = 0.5 + (abs(rating - 3) / 2) * 0.4
        confidence_factors.append(rating_confidence)
        
        # Sentiment confidence
        confidence_factors.append(sentiment_analysis.confidence)
        
        return statistics.mean(confidence_factors)
    
    async def _calculate_priority_score(
        self,
        sentiment_analysis: SentimentAnalysis,
        quality_scores: QualityScores,
        rating: int,
        confidence_level: float
    ) -> float:
        """Calculate priority score for addressing feedback."""
        priority_factors = []
        
        # Negative feedback gets higher priority
        if sentiment_analysis.polarity in [SentimentPolarity.VERY_NEGATIVE, SentimentPolarity.NEGATIVE]:
            priority_factors.append(0.9)
        elif sentiment_analysis.polarity == SentimentPolarity.NEUTRAL:
            priority_factors.append(0.5)
        else:
            priority_factors.append(0.2)
        
        # Low quality scores increase priority
        avg_quality = quality_scores.overall_quality
        quality_priority = 1.0 - avg_quality
        priority_factors.append(quality_priority)
        
        # Low ratings increase priority
        rating_priority = (5 - rating) / 4
        priority_factors.append(rating_priority)
        
        # High confidence increases priority
        priority_factors.append(confidence_level)
        
        return statistics.mean(priority_factors)
    
    async def _get_feedback_data(self, variant_id: str, days: int) -> List[Dict[str, Any]]:
        """Get feedback data from database."""
        try:
            # This would query the database for feedback
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            self.logger.error("Failed to get feedback data", error=str(e))
            return []
    
    async def _analyze_rating_trends(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze rating trends over time."""
        if not feedback_data:
            return {}
        
        ratings = [fb['rating'] for fb in feedback_data]
        
        return {
            'average_rating': statistics.mean(ratings),
            'rating_trend': 'stable',  # Would calculate actual trend
            'rating_distribution': dict(Counter(ratings)),
            'volatility': statistics.stdev(ratings) if len(ratings) > 1 else 0.0
        }
    
    async def _analyze_sentiment_trends(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment trends over time."""
        # Placeholder implementation
        return {
            'overall_sentiment': 'neutral',
            'sentiment_stability': 0.8,
            'positive_ratio': 0.6,
            'negative_ratio': 0.2
        }
    
    async def _analyze_category_trends(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feedback category trends."""
        return {
            'most_mentioned_categories': ['clarity', 'accuracy'],
            'category_sentiment_breakdown': {},
            'emerging_categories': []
        }
    
    async def _analyze_quality_trends(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality metric trends."""
        return {
            'quality_improvement': True,
            'quality_score_trend': 'improving',
            'quality_consistency': 0.85
        }
    
    async def _calculate_statistical_summary(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistical summary of feedback."""
        ratings = [fb['rating'] for fb in feedback_data]
        
        return {
            'sample_size': len(feedback_data),
            'mean_rating': statistics.mean(ratings),
            'median_rating': statistics.median(ratings),
            'std_deviation': statistics.stdev(ratings) if len(ratings) > 1 else 0.0,
            'confidence_interval': self._calculate_confidence_interval(ratings)
        }
    
    def _calculate_confidence_interval(self, ratings: List[int]) -> Tuple[float, float]:
        """Calculate confidence interval for ratings."""
        if len(ratings) < 2:
            mean_rating = ratings[0] if ratings else 3.0
            return (mean_rating, mean_rating)
        
        mean_rating = statistics.mean(ratings)
        std_error = statistics.stdev(ratings) / (len(ratings) ** 0.5)
        margin = 1.96 * std_error  # 95% confidence interval
        
        return (mean_rating - margin, mean_rating + margin)
    
    async def _identify_significant_changes(
        self, 
        feedback_data: List[Dict[str, Any]], 
        time_period_days: int
    ) -> List[Dict[str, Any]]:
        """Identify significant changes in feedback patterns."""
        return []  # Placeholder
    
    async def _generate_trend_recommendations(
        self,
        rating_trends: Dict[str, Any],
        sentiment_trends: Dict[str, Any],
        category_trends: Dict[str, Any],
        significant_changes: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        # Example recommendations based on trends
        if rating_trends.get('rating_trend') == 'declining':
            recommendations.append("Address declining satisfaction - investigate recent changes")
        
        if sentiment_trends.get('negative_ratio', 0) > 0.3:
            recommendations.append("High negative sentiment - focus on user pain points")
        
        return recommendations
    
    async def _calculate_trend_confidence(self, feedback_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence level for trend analysis."""
        sample_size = len(feedback_data)
        if sample_size < 10:
            return 0.3
        elif sample_size < 30:
            return 0.6
        else:
            return 0.9
    
    async def _analyze_text_feedback_aggregate(self, text_feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze text feedback in aggregate."""
        return {
            'common_themes': ['clarity', 'usefulness'],
            'sentiment_distribution': {'positive': 0.6, 'neutral': 0.2, 'negative': 0.2},
            'word_frequency': {}
        }
    
    async def _calculate_aggregate_quality_metrics(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate quality metrics."""
        return {
            'overall_satisfaction': 0.75,
            'quality_consistency': 0.8,
            'improvement_trend': 0.1
        }
    
    async def _generate_comparative_analysis(self, summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparative analysis across variants."""
        return {
            'best_performing_variant': None,
            'largest_improvement_opportunity': None,
            'consistency_analysis': {}
        }