"""
FastAPI endpoints for Prompt Optimization System.

Provides comprehensive API for prompt generation, optimization, evaluation,
A/B testing, and feedback collection.
"""

import uuid
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, func
from sqlalchemy.orm import selectinload

import structlog

from ...core.database import get_session_dependency
from ...core.prompt_optimizer import PromptOptimizer
from ...models.prompt_optimization import (
    PromptTemplate, OptimizationExperiment, PromptVariant, PromptEvaluation,
    ABTestResult, PromptFeedback, PromptTestCase, OptimizationMetric,
    PromptStatus, ExperimentStatus, OptimizationMethod
)
from ...schemas.prompt_optimization import (
    PromptTemplateCreate, PromptTemplateUpdate, PromptTemplateResponse, PromptTemplateListResponse,
    OptimizationExperimentCreate, OptimizationExperimentUpdate, OptimizationExperimentResponse,
    OptimizationExperimentListResponse, PromptVariantResponse, PromptVariantListResponse,
    PromptEvaluationCreate, PromptEvaluationResponse, PromptEvaluationBatchResponse,
    ABTestCreate, ABTestResponse, PromptFeedbackCreate, PromptFeedbackResponse,
    PromptTestCaseCreate, PromptTestCaseResponse, PromptGenerationRequest, PromptGenerationResponse,
    OptimizationMetricResponse, SystemMetricsResponse, PromptOptimizationError,
    BulkEvaluationRequest, BulkEvaluationResponse, PromptSearchRequest
)

logger = structlog.get_logger()
router = APIRouter()


# Prompt Template endpoints
@router.post("/templates", response_model=PromptTemplateResponse, status_code=201)
async def create_prompt_template(
    template_data: PromptTemplateCreate,
    db: AsyncSession = Depends(get_session_dependency)
) -> PromptTemplateResponse:
    """Create a new prompt template."""
    try:
        template = PromptTemplate(
            name=template_data.name,
            task_type=template_data.task_type,
            domain=template_data.domain,
            template_content=template_data.template_content,
            template_variables=template_data.template_variables,
            description=template_data.description,
            tags=template_data.tags,
            metadata=template_data.metadata,
            created_by=template_data.created_by
        )
        
        db.add(template)
        await db.commit()
        await db.refresh(template)
        
        logger.info("Created prompt template", template_id=str(template.id))
        return PromptTemplateResponse.from_orm(template)
        
    except Exception as e:
        logger.error("Failed to create prompt template", error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create prompt template")


@router.get("/templates", response_model=PromptTemplateListResponse)
async def list_prompt_templates(
    domain: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    status: Optional[PromptStatus] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_session_dependency)
) -> PromptTemplateListResponse:
    """List prompt templates with filtering."""
    try:
        query = select(PromptTemplate)
        
        if domain:
            query = query.where(PromptTemplate.domain == domain)
        if task_type:
            query = query.where(PromptTemplate.task_type == task_type)
        if status:
            query = query.where(PromptTemplate.status == status)
        
        query = query.offset(offset).limit(limit).order_by(PromptTemplate.created_at.desc())
        
        result = await db.execute(query)
        templates = result.scalars().all()
        
        # Get total count
        count_query = select(func.count(PromptTemplate.id))
        if domain:
            count_query = count_query.where(PromptTemplate.domain == domain)
        if task_type:
            count_query = count_query.where(PromptTemplate.task_type == task_type)
        if status:
            count_query = count_query.where(PromptTemplate.status == status)
        
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        return PromptTemplateListResponse(
            templates=[PromptTemplateResponse.from_orm(t) for t in templates],
            total=total,
            offset=offset,
            limit=limit
        )
        
    except Exception as e:
        logger.error("Failed to list prompt templates", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve templates")


@router.get("/templates/{template_id}", response_model=PromptTemplateResponse)
async def get_prompt_template(
    template_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> PromptTemplateResponse:
    """Get a specific prompt template."""
    try:
        result = await db.execute(
            select(PromptTemplate).where(PromptTemplate.id == template_id)
        )
        template = result.scalar_one_or_none()
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return PromptTemplateResponse.from_orm(template)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get template", template_id=str(template_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve template")


# Prompt Generation endpoints
@router.post("/generate", response_model=PromptGenerationResponse)
async def generate_prompts(
    request: PromptGenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency)
) -> PromptGenerationResponse:
    """Generate optimized prompt candidates."""
    try:
        optimizer = PromptOptimizer(db)
        
        result = await optimizer.generate_prompts(
            task_description=request.task_description,
            domain=request.domain,
            performance_goals=request.performance_goals,
            baseline_examples=request.baseline_examples,
            constraints=request.constraints,
            num_candidates=5
        )
        
        logger.info("Generated prompts", experiment_id=result['experiment_id'])
        
        return PromptGenerationResponse(
            prompt_candidates=result['prompt_candidates'],
            experiment_id=uuid.UUID(result['experiment_id']),
            generation_metadata=result['generation_metadata']
        )
        
    except Exception as e:
        logger.error("Failed to generate prompts", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate prompts")


# Optimization Experiment endpoints
@router.post("/experiments", response_model=OptimizationExperimentResponse, status_code=201)
async def create_optimization_experiment(
    experiment_data: OptimizationExperimentCreate,
    db: AsyncSession = Depends(get_session_dependency)
) -> OptimizationExperimentResponse:
    """Create a new optimization experiment."""
    try:
        optimizer = PromptOptimizer(db)
        
        experiment_id = await optimizer.create_experiment(
            experiment_name=experiment_data.experiment_name,
            base_prompt_id=experiment_data.base_prompt_id,
            optimization_method=experiment_data.optimization_method,
            target_metrics=experiment_data.target_metrics,
            experiment_config=experiment_data.experiment_config,
            max_iterations=experiment_data.max_iterations,
            created_by=experiment_data.created_by
        )
        
        # Get the created experiment
        result = await db.execute(
            select(OptimizationExperiment).where(OptimizationExperiment.id == experiment_id)
        )
        experiment = result.scalar_one()
        
        return OptimizationExperimentResponse.from_orm(experiment)
        
    except Exception as e:
        logger.error("Failed to create experiment", error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create experiment")


@router.post("/experiments/{experiment_id}/run")
async def run_optimization_experiment(
    experiment_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency)
) -> dict:
    """Run an optimization experiment."""
    try:
        optimizer = PromptOptimizer(db)
        
        # Run experiment in background
        background_tasks.add_task(
            optimizer.run_experiment,
            experiment_id
        )
        
        return {
            "message": "Experiment started",
            "experiment_id": str(experiment_id),
            "status": "running"
        }
        
    except Exception as e:
        logger.error("Failed to start experiment", experiment_id=str(experiment_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start experiment")


@router.get("/experiments/{experiment_id}/status")
async def get_experiment_status(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> dict:
    """Get experiment status and progress."""
    try:
        optimizer = PromptOptimizer(db)
        status = await optimizer.get_experiment_status(experiment_id)
        return status
        
    except Exception as e:
        logger.error("Failed to get experiment status", experiment_id=str(experiment_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get experiment status")


# Evaluation endpoints
@router.post("/evaluate", response_model=PromptEvaluationBatchResponse)
async def evaluate_prompt(
    evaluation_request: PromptEvaluationCreate,
    db: AsyncSession = Depends(get_session_dependency)
) -> PromptEvaluationBatchResponse:
    """Evaluate a prompt variant's performance."""
    try:
        optimizer = PromptOptimizer(db)
        
        result = await optimizer.evaluate_prompt(
            prompt_variant_id=evaluation_request.prompt_variant_id,
            test_cases=None,  # Will use default test cases
            evaluation_metrics=evaluation_request.evaluation_metrics
        )
        
        # Create evaluation records (simplified)
        evaluations = []
        for metric, score in result['detailed_metrics'].items():
            evaluation = PromptEvaluationResponse(
                id=uuid.uuid4(),
                prompt_variant_id=evaluation_request.prompt_variant_id,
                test_case_id=evaluation_request.test_case_id,
                metric_name=metric,
                metric_value=score,
                raw_output=None,
                expected_output=None,
                evaluation_context=result['context'],
                evaluation_method="automated",
                evaluation_time_seconds=result['evaluation_time'],
                token_usage=result['token_usage'],
                cost_estimate=None,
                error_details=None,
                evaluated_by="prompt_optimizer",
                evaluated_at=datetime.utcnow()
            )
            evaluations.append(evaluation)
        
        return PromptEvaluationBatchResponse(
            evaluations=evaluations,
            performance_score=result['performance_score'],
            detailed_metrics=result['detailed_metrics']
        )
        
    except Exception as e:
        logger.error("Failed to evaluate prompt", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to evaluate prompt")


# A/B Testing endpoints
@router.post("/ab-tests", response_model=ABTestResponse)
async def create_ab_test(
    ab_test_data: ABTestCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency)
) -> ABTestResponse:
    """Create and run an A/B test."""
    try:
        optimizer = PromptOptimizer(db)
        
        result = await optimizer.run_ab_test(
            experiment_id=ab_test_data.experiment_id,
            prompt_a_id=ab_test_data.prompt_a_id,
            prompt_b_id=ab_test_data.prompt_b_id,
            sample_size=ab_test_data.sample_size,
            significance_level=ab_test_data.significance_level
        )
        
        # Create response object
        ab_test_response = ABTestResponse(
            id=uuid.UUID(result['test_id']),
            experiment_id=ab_test_data.experiment_id,
            test_name=ab_test_data.test_name,
            prompt_a_id=ab_test_data.prompt_a_id,
            prompt_b_id=ab_test_data.prompt_b_id,
            sample_size=result['sample_size'],
            significance_level=result['significance_level'],
            p_value=result['p_value'],
            effect_size=result['effect_size'],
            confidence_interval_lower=result['confidence_interval'][0] if result['confidence_interval'] else None,
            confidence_interval_upper=result['confidence_interval'][1] if result['confidence_interval'] else None,
            winner_variant_id=uuid.UUID(result['winner_variant_id']) if result['winner_variant_id'] else None,
            test_power=result['test_power'],
            mean_a=result['detailed_results'].get('mean_a'),
            mean_b=result['detailed_results'].get('mean_b'),
            std_a=result['detailed_results'].get('std_a'),
            std_b=result['detailed_results'].get('std_b'),
            test_statistic=result['detailed_results'].get('test_statistic'),
            degrees_of_freedom=result['detailed_results'].get('degrees_of_freedom'),
            statistical_notes=result['detailed_results'].get('notes'),
            test_completed_at=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        return ab_test_response
        
    except Exception as e:
        logger.error("Failed to create A/B test", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create A/B test")


# Feedback endpoints
@router.post("/feedback", response_model=PromptFeedbackResponse)
async def submit_feedback(
    feedback_data: PromptFeedbackCreate,
    db: AsyncSession = Depends(get_session_dependency)
) -> PromptFeedbackResponse:
    """Submit user feedback for a prompt variant."""
    try:
        optimizer = PromptOptimizer(db)
        
        result = await optimizer.record_feedback(
            prompt_variant_id=feedback_data.prompt_variant_id,
            user_id=feedback_data.user_id,
            rating=feedback_data.rating,
            feedback_text=feedback_data.feedback_text,
            feedback_categories=feedback_data.feedback_categories,
            context_data=feedback_data.context_data
        )
        
        # Create response
        response = PromptFeedbackResponse(
            id=uuid.UUID(result['feedback_id']),
            prompt_variant_id=feedback_data.prompt_variant_id,
            user_id=feedback_data.user_id,
            session_id=feedback_data.session_id,
            rating=feedback_data.rating,
            feedback_text=feedback_data.feedback_text,
            feedback_categories=feedback_data.feedback_categories,
            context_data=feedback_data.context_data,
            response_quality_score=result['quality_scores'].get('response_quality'),
            relevance_score=result['quality_scores'].get('relevance'),
            clarity_score=result['quality_scores'].get('clarity'),
            usefulness_score=result['quality_scores'].get('usefulness'),
            sentiment_score=result['quality_scores'].get('sentiment'),
            feedback_weight=result['influence_weight'],
            is_validated=False,
            validation_notes=None,
            submitted_at=datetime.utcnow(),
            status=result['status'],
            influence_weight=result['influence_weight']
        )
        
        return response
        
    except Exception as e:
        logger.error("Failed to submit feedback", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


# Test Case endpoints
@router.post("/test-cases", response_model=PromptTestCaseResponse, status_code=201)
async def create_test_case(
    test_case_data: PromptTestCaseCreate,
    db: AsyncSession = Depends(get_session_dependency)
) -> PromptTestCaseResponse:
    """Create a new test case."""
    try:
        test_case = PromptTestCase(
            name=test_case_data.name,
            description=test_case_data.description,
            domain=test_case_data.domain,
            task_type=test_case_data.task_type,
            input_data=test_case_data.input_data,
            expected_output=test_case_data.expected_output,
            evaluation_criteria=test_case_data.evaluation_criteria,
            difficulty_level=test_case_data.difficulty_level,
            tags=test_case_data.tags,
            metadata=test_case_data.metadata,
            created_by=test_case_data.created_by
        )
        
        db.add(test_case)
        await db.commit()
        await db.refresh(test_case)
        
        return PromptTestCaseResponse.from_orm(test_case)
        
    except Exception as e:
        logger.error("Failed to create test case", error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create test case")


# System metrics and monitoring
@router.get("/metrics/system", response_model=SystemMetricsResponse)
async def get_system_metrics(
    db: AsyncSession = Depends(get_session_dependency)
) -> SystemMetricsResponse:
    """Get system-wide optimization metrics."""
    try:
        # Get active experiments count
        active_experiments_result = await db.execute(
            select(func.count(OptimizationExperiment.id))
            .where(OptimizationExperiment.status == ExperimentStatus.RUNNING)
        )
        active_experiments = active_experiments_result.scalar()
        
        # Get total prompts optimized
        total_prompts_result = await db.execute(
            select(func.count(PromptTemplate.id))
        )
        total_prompts = total_prompts_result.scalar()
        
        # Get completed experiments
        completed_experiments_result = await db.execute(
            select(func.count(OptimizationExperiment.id))
            .where(OptimizationExperiment.status == ExperimentStatus.COMPLETED)
        )
        completed_experiments = completed_experiments_result.scalar()
        
        # Get failed experiments
        failed_experiments_result = await db.execute(
            select(func.count(OptimizationExperiment.id))
            .where(OptimizationExperiment.status == ExperimentStatus.FAILED)
        )
        failed_experiments = failed_experiments_result.scalar()
        
        # Calculate average improvement (simplified)
        avg_improvement = 15.0  # Placeholder
        
        return SystemMetricsResponse(
            active_experiments=active_experiments,
            total_prompts_optimized=total_prompts,
            average_improvement_percentage=avg_improvement,
            successful_optimizations=completed_experiments,
            failed_optimizations=failed_experiments,
            total_evaluations_performed=total_prompts * 5,  # Estimate
            average_optimization_time_hours=2.5,
            metrics_by_method={
                'evolutionary': {'success_rate': 0.85, 'avg_improvement': 0.18},
                'meta_prompting': {'success_rate': 0.92, 'avg_improvement': 0.15},
                'few_shot': {'success_rate': 0.78, 'avg_improvement': 0.12}
            }
        )
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


# Bulk operations
@router.post("/evaluate/bulk", response_model=BulkEvaluationResponse)
async def bulk_evaluate_prompts(
    request: BulkEvaluationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency)
) -> BulkEvaluationResponse:
    """Evaluate multiple prompt variants in bulk."""
    try:
        if len(request.prompt_variant_ids) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 variants per bulk request")
        
        optimizer = PromptOptimizer(db)
        
        # Process in background for large requests
        if len(request.prompt_variant_ids) > 10:
            background_tasks.add_task(
                _process_bulk_evaluation,
                optimizer,
                request.prompt_variant_ids,
                request.evaluation_metrics
            )
            
            return BulkEvaluationResponse(
                total_evaluations=len(request.prompt_variant_ids),
                successful_evaluations=0,
                failed_evaluations=0,
                evaluation_results=[],
                error_details=[]
            )
        
        # Process immediately for small requests
        results = []
        errors = []
        
        for variant_id in request.prompt_variant_ids:
            try:
                result = await optimizer.evaluate_prompt(
                    prompt_variant_id=variant_id,
                    evaluation_metrics=request.evaluation_metrics
                )
                
                # Convert to response format (simplified)
                evaluation = PromptEvaluationResponse(
                    id=uuid.uuid4(),
                    prompt_variant_id=variant_id,
                    test_case_id=None,
                    metric_name="overall",
                    metric_value=result['performance_score'],
                    raw_output=None,
                    expected_output=None,
                    evaluation_context=result['context'],
                    evaluation_method="bulk_automated",
                    evaluation_time_seconds=result['evaluation_time'],
                    token_usage=result['token_usage'],
                    cost_estimate=None,
                    error_details=None,
                    evaluated_by="bulk_evaluator",
                    evaluated_at=datetime.utcnow()
                )
                results.append(evaluation)
                
            except Exception as e:
                errors.append({
                    'variant_id': str(variant_id),
                    'error': str(e)
                })
        
        return BulkEvaluationResponse(
            total_evaluations=len(request.prompt_variant_ids),
            successful_evaluations=len(results),
            failed_evaluations=len(errors),
            evaluation_results=results,
            error_details=errors
        )
        
    except Exception as e:
        logger.error("Failed bulk evaluation", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to process bulk evaluation")


async def _process_bulk_evaluation(
    optimizer: PromptOptimizer,
    variant_ids: List[uuid.UUID],
    metrics: List[str]
):
    """Process bulk evaluation in background."""
    for variant_id in variant_ids:
        try:
            await optimizer.evaluate_prompt(
                prompt_variant_id=variant_id,
                evaluation_metrics=metrics
            )
        except Exception as e:
            logger.error("Bulk evaluation failed for variant", variant_id=str(variant_id), error=str(e))


# Health check
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "prompt_optimization",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }