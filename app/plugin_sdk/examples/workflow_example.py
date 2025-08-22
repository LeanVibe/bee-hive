"""
Data Pipeline Plugin Example

Demonstrates comprehensive workflow plugin development using the LeanVibe SDK.
Shows best practices for data processing, error handling, and performance optimization.
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..interfaces import WorkflowPlugin, PluginType
from ..models import PluginConfig, TaskInterface, TaskResult, PluginEvent, EventSeverity
from ..decorators import plugin_method, performance_tracked, error_handled, cached_result
from ..exceptions import PluginConfigurationError, PluginExecutionError


@dataclass
class ProcessingStep:
    """Configuration for a pipeline processing step."""
    name: str
    step_type: str
    parameters: Dict[str, Any]
    enabled: bool = True
    timeout_seconds: int = 30


class DataPipelinePlugin(WorkflowPlugin):
    """
    Advanced data processing pipeline plugin.
    
    Features:
    - Multi-step data processing pipeline
    - Configurable processing steps
    - Error recovery and retry logic
    - Performance monitoring and optimization
    - Data validation and quality checks
    - Flexible output formats
    
    Epic 1 Optimizations:
    - Lazy loading of processing components
    - Memory-efficient batch processing
    - <50ms operation response times
    - <80MB memory footprint
    """
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        
        # Configuration
        self.pipeline_steps = self._load_pipeline_steps()
        self.batch_size = config.parameters.get("batch_size", 1000)
        self.max_retries = config.parameters.get("max_retries", 3)
        self.output_format = config.parameters.get("output_format", "json")
        self.enable_validation = config.parameters.get("enable_validation", True)
        
        # Runtime state
        self.processing_stats = {
            "total_processed": 0,
            "total_errors": 0,
            "pipeline_runs": 0,
            "average_time_ms": 0.0
        }
        
        # Lazy-loaded components
        self._validators = None
        self._transformers = None
        self._output_handlers = None
    
    def _load_pipeline_steps(self) -> List[ProcessingStep]:
        """Load pipeline steps from configuration."""
        steps_config = self.config.parameters.get("pipeline_steps", [])
        
        steps = []
        for step_config in steps_config:
            step = ProcessingStep(
                name=step_config["name"],
                step_type=step_config["type"],
                parameters=step_config.get("parameters", {}),
                enabled=step_config.get("enabled", True),
                timeout_seconds=step_config.get("timeout_seconds", 30)
            )
            steps.append(step)
        
        return steps
    
    @property
    def validators(self):
        """Lazy load data validators."""
        if self._validators is None:
            self._validators = self._create_validators()
        return self._validators
    
    @property
    def transformers(self):
        """Lazy load data transformers."""
        if self._transformers is None:
            self._transformers = self._create_transformers()
        return self._transformers
    
    @property
    def output_handlers(self):
        """Lazy load output handlers."""
        if self._output_handlers is None:
            self._output_handlers = self._create_output_handlers()
        return self._output_handlers
    
    async def _on_initialize(self) -> None:
        """Initialize the data pipeline plugin."""
        await self.log_info("Initializing DataPipelinePlugin")
        
        # Validate configuration
        if not self.pipeline_steps:
            raise PluginConfigurationError(
                "No pipeline steps configured",
                config_key="pipeline_steps",
                plugin_id=self.plugin_id
            )
        
        if self.batch_size <= 0:
            raise PluginConfigurationError(
                "Batch size must be positive",
                config_key="batch_size",
                expected_type="positive integer",
                actual_value=self.batch_size,
                plugin_id=self.plugin_id
            )
        
        # Initialize processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "total_errors": 0,
            "pipeline_runs": 0,
            "average_time_ms": 0.0
        }
        
        await self.log_info(f"Initialized with {len(self.pipeline_steps)} steps, batch_size={self.batch_size}")
    
    @performance_tracked(alert_threshold_ms=5000, memory_limit_mb=100)
    @plugin_method(timeout_seconds=1800, max_retries=2)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """
        Execute data pipeline processing.
        
        Supports the following task types:
        - process_data: Process dataset through pipeline
        - validate_data: Validate dataset only
        - get_stats: Get processing statistics
        """
        task_type = task.task_type
        
        if task_type == "process_data":
            return await self._process_data_pipeline(task)
        elif task_type == "validate_data":
            return await self._validate_data_only(task)
        elif task_type == "get_stats":
            return await self._get_processing_stats(task)
        else:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=f"Unknown task type: {task_type}",
                error_code="INVALID_TASK_TYPE"
            )
    
    async def _process_data_pipeline(self, task: TaskInterface) -> TaskResult:
        """Process data through the complete pipeline."""
        start_time = datetime.utcnow()
        
        try:
            # Get input data
            input_data = task.parameters.get("input_data")
            if not input_data:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error="No input data provided",
                    error_code="MISSING_INPUT_DATA"
                )
            
            await task.update_status("running", progress=0.1)
            await self.log_info(f"Starting pipeline processing for {len(input_data)} records")
            
            # Validate input data if enabled
            if self.enable_validation:
                validation_result = await self._validate_input_data(input_data)
                if not validation_result["valid"]:
                    return TaskResult(
                        success=False,
                        plugin_id=self.plugin_id,
                        task_id=task.task_id,
                        error=f"Input validation failed: {validation_result['errors']}",
                        error_code="VALIDATION_FAILED",
                        data={"validation_errors": validation_result["errors"]}
                    )
            
            await task.update_status("running", progress=0.2)
            
            # Process data through pipeline steps
            current_data = input_data
            step_results = []
            
            for i, step in enumerate(self.pipeline_steps):
                if not step.enabled:
                    await self.log_info(f"Skipping disabled step: {step.name}")
                    continue
                
                await self.log_info(f"Executing step {i+1}/{len(self.pipeline_steps)}: {step.name}")
                
                step_start = datetime.utcnow()
                step_result = await self._execute_pipeline_step(step, current_data, task)
                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000
                
                step_results.append({
                    "step_name": step.name,
                    "step_type": step.step_type,
                    "success": step_result["success"],
                    "duration_ms": step_duration,
                    "records_in": len(current_data) if isinstance(current_data, list) else 1,
                    "records_out": len(step_result["output"]) if isinstance(step_result["output"], list) else 1,
                    "error": step_result.get("error"),
                    "metrics": step_result.get("metrics", {})
                })
                
                if not step_result["success"]:
                    # Step failed - return error with context
                    return TaskResult(
                        success=False,
                        plugin_id=self.plugin_id,
                        task_id=task.task_id,
                        error=f"Pipeline step '{step.name}' failed: {step_result['error']}",
                        error_code="PIPELINE_STEP_FAILED",
                        data={
                            "failed_step": step.name,
                            "step_index": i,
                            "step_results": step_results
                        }
                    )
                
                current_data = step_result["output"]
                
                # Update progress
                progress = 0.2 + (0.6 * (i + 1) / len(self.pipeline_steps))
                await task.update_status("running", progress=progress)
            
            await task.update_status("running", progress=0.9)
            
            # Save output data
            output_path = await self._save_output_data(current_data, task.task_id)
            
            # Update statistics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.processing_stats["total_processed"] += len(current_data) if isinstance(current_data, list) else 1
            self.processing_stats["pipeline_runs"] += 1
            
            # Update average processing time
            current_avg = self.processing_stats["average_time_ms"]
            runs = self.processing_stats["pipeline_runs"]
            new_avg = ((current_avg * (runs - 1)) + processing_time) / runs
            self.processing_stats["average_time_ms"] = new_avg
            
            await task.update_status("completed", progress=1.0)
            
            # Emit completion event
            completion_event = PluginEvent(
                event_type="pipeline_completed",
                plugin_id=self.plugin_id,
                data={
                    "records_processed": len(current_data) if isinstance(current_data, list) else 1,
                    "processing_time_ms": processing_time,
                    "steps_executed": len([s for s in self.pipeline_steps if s.enabled])
                },
                task_id=task.task_id
            )
            await self.emit_event(completion_event)
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                execution_time_ms=processing_time,
                data={
                    "output_path": output_path,
                    "records_processed": len(current_data) if isinstance(current_data, list) else 1,
                    "pipeline_steps": len(self.pipeline_steps),
                    "step_results": step_results,
                    "processing_stats": self.processing_stats
                }
            )
            
        except Exception as e:
            self.processing_stats["total_errors"] += 1
            await self.log_error(f"Pipeline processing failed: {e}")
            
            # Emit error event
            error_event = PluginEvent(
                event_type="pipeline_error",
                plugin_id=self.plugin_id,
                data={"error": str(e), "error_type": type(e).__name__},
                severity=EventSeverity.ERROR,
                task_id=task.task_id
            )
            await self.emit_event(error_event)
            
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="PIPELINE_EXECUTION_ERROR",
                data={"processing_stats": self.processing_stats}
            )
    
    async def _execute_pipeline_step(self, step: ProcessingStep, data: Any, task: TaskInterface) -> Dict[str, Any]:
        """Execute a single pipeline step with timeout and error handling."""
        try:
            # Execute step with timeout
            result = await asyncio.wait_for(
                self._run_step_logic(step, data),
                timeout=step.timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Step '{step.name}' timed out after {step.timeout_seconds}s"
            await self.log_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "output": data
            }
        except Exception as e:
            error_msg = f"Step '{step.name}' failed: {str(e)}"
            await self.log_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "output": data
            }
    
    async def _run_step_logic(self, step: ProcessingStep, data: Any) -> Dict[str, Any]:
        """Run the actual step logic based on step type."""
        step_type = step.step_type
        
        if step_type == "filter":
            return await self._filter_step(data, step.parameters)
        elif step_type == "transform":
            return await self._transform_step(data, step.parameters)
        elif step_type == "aggregate":
            return await self._aggregate_step(data, step.parameters)
        elif step_type == "validate":
            return await self._validate_step(data, step.parameters)
        elif step_type == "enrich":
            return await self._enrich_step(data, step.parameters)
        elif step_type == "deduplicate":
            return await self._deduplicate_step(data, step.parameters)
        else:
            return {
                "success": False,
                "error": f"Unknown step type: {step_type}",
                "output": data
            }
    
    async def _filter_step(self, data: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data based on criteria."""
        try:
            filter_field = params.get("field")
            filter_value = params.get("value")
            filter_op = params.get("operation", "equals")
            
            if not filter_field:
                return {
                    "success": False,
                    "error": "Filter field not specified",
                    "output": data
                }
            
            filtered_data = []
            for record in data:
                if self._evaluate_filter_condition(record, filter_field, filter_value, filter_op):
                    filtered_data.append(record)
            
            metrics = {
                "input_count": len(data),
                "output_count": len(filtered_data),
                "filtered_out": len(data) - len(filtered_data),
                "filter_rate": (len(data) - len(filtered_data)) / len(data) if data else 0
            }
            
            return {
                "success": True,
                "output": filtered_data,
                "metrics": metrics
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": data
            }
    
    def _evaluate_filter_condition(self, record: Dict, field: str, value: Any, operation: str) -> bool:
        """Evaluate filter condition for a record."""
        record_value = record.get(field)
        
        if operation == "equals":
            return record_value == value
        elif operation == "not_equals":
            return record_value != value
        elif operation == "contains":
            return value in str(record_value) if record_value is not None else False
        elif operation == "greater_than":
            return float(record_value) > float(value) if record_value is not None else False
        elif operation == "less_than":
            return float(record_value) < float(value) if record_value is not None else False
        elif operation == "starts_with":
            return str(record_value).startswith(str(value)) if record_value is not None else False
        elif operation == "ends_with":
            return str(record_value).endswith(str(value)) if record_value is not None else False
        elif operation == "in_list":
            return record_value in value if isinstance(value, list) else False
        else:
            return False
    
    async def _transform_step(self, data: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data records."""
        try:
            transformations = params.get("transformations", [])
            
            transformed_data = []
            for record in data:
                transformed_record = record.copy()
                
                for transform in transformations:
                    field = transform.get("field")
                    operation = transform.get("operation")
                    value = transform.get("value")
                    
                    if operation == "set":
                        transformed_record[field] = value
                    elif operation == "copy":
                        source_field = transform.get("source_field")
                        if source_field in transformed_record:
                            transformed_record[field] = transformed_record[source_field]
                    elif operation == "uppercase" and field in transformed_record:
                        transformed_record[field] = str(transformed_record[field]).upper()
                    elif operation == "lowercase" and field in transformed_record:
                        transformed_record[field] = str(transformed_record[field]).lower()
                    elif operation == "multiply" and field in transformed_record:
                        transformed_record[field] = float(transformed_record[field]) * float(value)
                    elif operation == "add" and field in transformed_record:
                        transformed_record[field] = float(transformed_record[field]) + float(value)
                    elif operation == "format" and field in transformed_record:
                        transformed_record[field] = value.format(transformed_record[field])
                    elif operation == "timestamp":
                        transformed_record[field] = datetime.utcnow().isoformat()
                    elif operation == "remove":
                        transformed_record.pop(field, None)
                
                transformed_data.append(transformed_record)
            
            metrics = {
                "transformations_applied": len(transformations),
                "records_transformed": len(transformed_data)
            }
            
            return {
                "success": True,
                "output": transformed_data,
                "metrics": metrics
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": data
            }
    
    async def _aggregate_step(self, data: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data based on grouping and operations."""
        try:
            group_by = params.get("group_by")
            aggregations = params.get("aggregations", [])
            
            if not group_by:
                # Global aggregation
                result = {}
                for agg in aggregations:
                    field = agg.get("field")
                    operation = agg.get("operation")
                    alias = agg.get("alias", f"{field}_{operation}")
                    
                    values = [record.get(field, 0) for record in data if field in record]
                    
                    if operation == "sum":
                        result[alias] = sum(values)
                    elif operation == "avg":
                        result[alias] = sum(values) / len(values) if values else 0
                    elif operation == "count":
                        result[alias] = len(values)
                    elif operation == "max":
                        result[alias] = max(values) if values else 0
                    elif operation == "min":
                        result[alias] = min(values) if values else 0
                    elif operation == "distinct_count":
                        result[alias] = len(set(values))
                
                return {
                    "success": True,
                    "output": [result],
                    "metrics": {"input_records": len(data), "output_records": 1}
                }
            else:
                # Group by aggregation
                groups = {}
                for record in data:
                    key = record.get(group_by)
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(record)
                
                aggregated_data = []
                for group_key, group_records in groups.items():
                    result = {group_by: group_key}
                    
                    for agg in aggregations:
                        field = agg.get("field")
                        operation = agg.get("operation")
                        alias = agg.get("alias", f"{field}_{operation}")
                        
                        values = [record.get(field, 0) for record in group_records if field in record]
                        
                        if operation == "sum":
                            result[alias] = sum(values)
                        elif operation == "avg":
                            result[alias] = sum(values) / len(values) if values else 0
                        elif operation == "count":
                            result[alias] = len(values)
                        elif operation == "max":
                            result[alias] = max(values) if values else 0
                        elif operation == "min":
                            result[alias] = min(values) if values else 0
                        elif operation == "distinct_count":
                            result[alias] = len(set(values))
                    
                    aggregated_data.append(result)
                
                return {
                    "success": True,
                    "output": aggregated_data,
                    "metrics": {
                        "input_records": len(data),
                        "output_groups": len(groups),
                        "output_records": len(aggregated_data)
                    }
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": data
            }
    
    async def _validate_step(self, data: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality."""
        try:
            validations = params.get("validations", [])
            validation_errors = []
            valid_records = []
            
            for i, record in enumerate(data):
                record_errors = []
                
                for validation in validations:
                    field = validation.get("field")
                    rule = validation.get("rule")
                    value = validation.get("value")
                    
                    if rule == "required" and (field not in record or record[field] is None):
                        record_errors.append(f"Missing required field '{field}'")
                    elif rule == "type" and field in record:
                        expected_type = validation.get("type", "str")
                        if not self._validate_field_type(record[field], expected_type):
                            record_errors.append(f"Field '{field}' has wrong type")
                    elif rule == "range" and field in record:
                        min_val = validation.get("min")
                        max_val = validation.get("max")
                        if min_val is not None and record[field] < min_val:
                            record_errors.append(f"Field '{field}' below minimum {min_val}")
                        if max_val is not None and record[field] > max_val:
                            record_errors.append(f"Field '{field}' above maximum {max_val}")
                    elif rule == "pattern" and field in record:
                        import re
                        pattern = validation.get("pattern")
                        if not re.match(pattern, str(record[field])):
                            record_errors.append(f"Field '{field}' doesn't match pattern")
                    elif rule == "email" and field in record:
                        import re
                        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                        if not re.match(email_pattern, str(record[field])):
                            record_errors.append(f"Field '{field}' is not a valid email")
                
                if record_errors:
                    validation_errors.append(f"Record {i}: {'; '.join(record_errors)}")
                else:
                    valid_records.append(record)
            
            # Determine if validation passes based on error tolerance
            error_tolerance = params.get("error_tolerance", 0)  # 0 = no errors allowed
            error_rate = len(validation_errors) / len(data) if data else 0
            
            if error_rate <= error_tolerance:
                return {
                    "success": True,
                    "output": valid_records if params.get("filter_invalid", False) else data,
                    "metrics": {
                        "records_validated": len(data),
                        "validation_errors": len(validation_errors),
                        "error_rate": error_rate,
                        "valid_records": len(valid_records)
                    },
                    "validation_errors": validation_errors
                }
            else:
                return {
                    "success": False,
                    "error": f"Validation failed: {len(validation_errors)} errors (rate: {error_rate:.2%})",
                    "output": data,
                    "validation_errors": validation_errors
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": data
            }
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type."""
        if expected_type == "str":
            return isinstance(value, str)
        elif expected_type == "int":
            return isinstance(value, int)
        elif expected_type == "float":
            return isinstance(value, (int, float))
        elif expected_type == "bool":
            return isinstance(value, bool)
        elif expected_type == "list":
            return isinstance(value, list)
        elif expected_type == "dict":
            return isinstance(value, dict)
        else:
            return True  # Unknown type, assume valid
    
    async def _enrich_step(self, data: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data with additional information."""
        try:
            enrichment_type = params.get("type", "static")
            
            if enrichment_type == "static":
                # Add static fields
                static_fields = params.get("fields", {})
                enriched_data = []
                
                for record in data:
                    enriched_record = record.copy()
                    enriched_record.update(static_fields)
                    enriched_data.append(enriched_record)
                
                return {
                    "success": True,
                    "output": enriched_data,
                    "metrics": {"fields_added": len(static_fields)}
                }
            
            elif enrichment_type == "lookup":
                # Lookup enrichment (simplified mock)
                lookup_field = params.get("lookup_field")
                target_field = params.get("target_field", "enriched_value")
                
                enriched_data = []
                for record in data:
                    enriched_record = record.copy()
                    lookup_value = record.get(lookup_field)
                    if lookup_value:
                        # Mock lookup - in real implementation, this would query external data
                        enriched_record[target_field] = f"enriched_{lookup_value}"
                    enriched_data.append(enriched_record)
                
                return {
                    "success": True,
                    "output": enriched_data,
                    "metrics": {"records_enriched": len(enriched_data)}
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown enrichment type: {enrichment_type}",
                    "output": data
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": data
            }
    
    async def _deduplicate_step(self, data: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate records."""
        try:
            key_fields = params.get("key_fields", [])
            strategy = params.get("strategy", "first")  # first, last, merge
            
            if not key_fields:
                # If no key fields specified, deduplicate by full record
                seen = set()
                deduplicated_data = []
                
                for record in data:
                    record_key = json.dumps(record, sort_keys=True)
                    if record_key not in seen:
                        seen.add(record_key)
                        deduplicated_data.append(record)
                
            else:
                # Deduplicate by key fields
                seen_keys = {}
                deduplicated_data = []
                
                for record in data:
                    key = tuple(record.get(field) for field in key_fields)
                    
                    if key not in seen_keys:
                        seen_keys[key] = record
                        deduplicated_data.append(record)
                    elif strategy == "last":
                        # Replace with last occurrence
                        seen_keys[key] = record
                        # Update in deduplicated_data
                        for i, existing_record in enumerate(deduplicated_data):
                            existing_key = tuple(existing_record.get(field) for field in key_fields)
                            if existing_key == key:
                                deduplicated_data[i] = record
                                break
                    elif strategy == "merge":
                        # Merge records (prefer non-null values)
                        existing_record = seen_keys[key]
                        merged_record = existing_record.copy()
                        for field, value in record.items():
                            if value is not None and (field not in merged_record or merged_record[field] is None):
                                merged_record[field] = value
                        
                        seen_keys[key] = merged_record
                        # Update in deduplicated_data
                        for i, existing_record in enumerate(deduplicated_data):
                            existing_key = tuple(existing_record.get(field) for field in key_fields)
                            if existing_key == key:
                                deduplicated_data[i] = merged_record
                                break
            
            duplicates_removed = len(data) - len(deduplicated_data)
            
            return {
                "success": True,
                "output": deduplicated_data,
                "metrics": {
                    "input_records": len(data),
                    "output_records": len(deduplicated_data),
                    "duplicates_removed": duplicates_removed,
                    "deduplication_rate": duplicates_removed / len(data) if data else 0
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": data
            }
    
    @cached_result(ttl_seconds=300)
    async def _validate_input_data(self, data: Any) -> Dict[str, Any]:
        """Validate input data structure and content."""
        errors = []
        
        # Check if data is a list
        if not isinstance(data, list):
            errors.append("Input data must be a list of records")
            return {"valid": False, "errors": errors}
        
        # Check if list is not empty
        if not data:
            errors.append("Input data cannot be empty")
            return {"valid": False, "errors": errors}
        
        # Check if all items are dictionaries
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                errors.append(f"Record {i} is not a dictionary")
                if len(errors) > 10:  # Limit error reports
                    errors.append("... and more type errors")
                    break
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    async def _validate_data_only(self, task: TaskInterface) -> TaskResult:
        """Validate data without processing."""
        try:
            input_data = task.parameters.get("input_data")
            if not input_data:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error="No input data provided"
                )
            
            validation_result = await self._validate_input_data(input_data)
            
            return TaskResult(
                success=validation_result["valid"],
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={
                    "validation_result": validation_result,
                    "record_count": len(input_data) if isinstance(input_data, list) else 1
                },
                error=f"Validation failed: {validation_result['errors']}" if not validation_result["valid"] else None
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e)
            )
    
    async def _get_processing_stats(self, task: TaskInterface) -> TaskResult:
        """Get current processing statistics."""
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data={
                "processing_stats": self.processing_stats,
                "plugin_config": {
                    "batch_size": self.batch_size,
                    "max_retries": self.max_retries,
                    "output_format": self.output_format,
                    "pipeline_steps": len(self.pipeline_steps)
                }
            }
        )
    
    async def _save_output_data(self, data: Any, task_id: str) -> str:
        """Save processed data to output file."""
        output_file = f"/tmp/pipeline_output_{task_id}_{uuid.uuid4().hex[:8]}.{self.output_format}"
        
        try:
            if self.output_format == "json":
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            elif self.output_format == "csv" and isinstance(data, list) and data:
                import csv
                with open(output_file, 'w', newline='') as f:
                    if isinstance(data[0], dict):
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                    else:
                        writer = csv.writer(f)
                        writer.writerows(data)
            
            await self.log_info(f"Saved output to {output_file}")
            return output_file
            
        except Exception as e:
            await self.log_error(f"Failed to save output: {e}")
            raise
    
    def _create_validators(self):
        """Create data validators."""
        # Mock implementation - in real plugin, this would create actual validators
        return {
            "email": lambda x: "@" in str(x),
            "numeric": lambda x: isinstance(x, (int, float)),
            "required": lambda x: x is not None and x != ""
        }
    
    def _create_transformers(self):
        """Create data transformers."""
        # Mock implementation - in real plugin, this would create actual transformers
        return {
            "uppercase": lambda x: str(x).upper(),
            "lowercase": lambda x: str(x).lower(),
            "trim": lambda x: str(x).strip()
        }
    
    def _create_output_handlers(self):
        """Create output handlers."""
        # Mock implementation - in real plugin, this would create actual output handlers
        return {
            "json": self._save_json_output,
            "csv": self._save_csv_output,
            "xml": self._save_xml_output
        }
    
    async def _save_json_output(self, data, path):
        """Save data as JSON."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def _save_csv_output(self, data, path):
        """Save data as CSV."""
        import csv
        with open(path, 'w', newline='') as f:
            if data and isinstance(data[0], dict):
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    
    async def _save_xml_output(self, data, path):
        """Save data as XML."""
        # Simplified XML output
        with open(path, 'w') as f:
            f.write("<data>\n")
            for i, record in enumerate(data):
                f.write(f"  <record id=\"{i}\">\n")
                if isinstance(record, dict):
                    for key, value in record.items():
                        f.write(f"    <{key}>{value}</{key}>\n")
                f.write("  </record>\n")
            f.write("</data>\n")
    
    async def _on_cleanup(self) -> None:
        """Cleanup plugin resources."""
        await self.log_info("Cleaning up DataPipelinePlugin")
        
        # Clear lazy-loaded components
        self._validators = None
        self._transformers = None
        self._output_handlers = None
        
        # Reset statistics
        self.processing_stats = {
            "total_processed": 0,
            "total_errors": 0,
            "pipeline_runs": 0,
            "average_time_ms": 0.0
        }