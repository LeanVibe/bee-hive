"""
MessagePack Event Serialization for LeanVibe Agent Hive 2.0

High-performance binary serialization for observability events with <5ms overhead.
Provides efficient event encoding/decoding for real-time multi-agent coordination.
"""

import uuid
import msgpack
import structlog
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum

from app.schemas.observability import (
    BaseObservabilityEvent,
    ObservabilityEvent,
    EventCategory,
    PerformanceMetrics,
    EventMetadata,
)

logger = structlog.get_logger()


class SerializationFormat(str, Enum):
    """Supported serialization formats."""
    JSON = "json"
    MSGPACK = "msgpack"
    MSGPACK_COMPRESSED = "msgpack_compressed"


class EventSerializer:
    """
    High-performance event serializer with MessagePack optimization.
    
    Provides efficient binary serialization with optional compression for
    observability events in multi-agent systems.
    """
    
    def __init__(
        self,
        format: SerializationFormat = SerializationFormat.MSGPACK,
        enable_compression: bool = False,
        max_payload_size: int = 50000
    ):
        """
        Initialize event serializer.
        
        Args:
            format: Serialization format to use
            enable_compression: Whether to enable compression
            max_payload_size: Maximum payload size in bytes
        """
        self.format = format
        self.enable_compression = enable_compression
        self.max_payload_size = max_payload_size
        
        # Configure MessagePack with custom handlers
        self._msgpack_kwargs = {
            'default': self._msgpack_encoder,
            'use_bin_type': True,
            'strict_types': False
        }
        
        logger.info(
            "Event serializer initialized",
            format=format.value,
            compression=enable_compression,
            max_payload_size=max_payload_size
        )
    
    def _msgpack_encoder(self, obj: Any) -> Any:
        """Custom MessagePack encoder for complex types."""
        if isinstance(obj, datetime):
            return {'__datetime__': obj.isoformat()}
        elif isinstance(obj, uuid.UUID):
            return {'__uuid__': str(obj)}
        elif isinstance(obj, Enum):
            return {'__enum__': obj.value}
        elif hasattr(obj, 'dict'):
            # Handle Pydantic models
            return obj.dict()
        else:
            raise TypeError(f"Object of type {type(obj)} is not serializable")
    
    def _msgpack_decoder(self, obj: Dict[str, Any]) -> Any:
        """Custom MessagePack decoder for complex types."""
        if '__datetime__' in obj:
            return datetime.fromisoformat(obj['__datetime__'])
        elif '__uuid__' in obj:
            return uuid.UUID(obj['__uuid__'])
        elif '__enum__' in obj:
            return obj['__enum__']
        return obj
    
    def serialize_event(self, event: ObservabilityEvent) -> Tuple[bytes, Dict[str, Any]]:
        """
        Serialize observability event to binary format.
        
        Args:
            event: Observability event to serialize
            
        Returns:
            Tuple of (serialized_data, metadata)
        """
        start_time = datetime.utcnow()
        
        try:
            # Convert event to dictionary
            if hasattr(event, 'dict'):
                event_dict = event.dict()
            else:
                event_dict = event
            
            # Optimize payload size
            if self._estimate_size(event_dict) > self.max_payload_size:
                event_dict = self._optimize_payload(event_dict)
            
            # Serialize based on format
            if self.format == SerializationFormat.JSON:
                import json
                serialized_data = json.dumps(event_dict, default=str).encode('utf-8')
            elif self.format in [SerializationFormat.MSGPACK, SerializationFormat.MSGPACK_COMPRESSED]:
                serialized_data = msgpack.packb(event_dict, **self._msgpack_kwargs)
                
                # Apply compression if enabled
                if self.enable_compression or self.format == SerializationFormat.MSGPACK_COMPRESSED:
                    import zlib
                    serialized_data = zlib.compress(serialized_data, level=6)
            else:
                raise ValueError(f"Unsupported serialization format: {self.format}")
            
            # Calculate performance metrics
            end_time = datetime.utcnow()
            serialization_time_ms = (end_time - start_time).total_seconds() * 1000
            
            metadata = {
                'serialization_format': self.format.value,
                'compressed': self.enable_compression or self.format == SerializationFormat.MSGPACK_COMPRESSED,
                'serialized_size_bytes': len(serialized_data),
                'serialization_time_ms': serialization_time_ms,
                'original_event_type': event_dict.get('event_type', 'unknown'),
                'event_category': event_dict.get('event_category', 'unknown'),
                'schema_version': event_dict.get('metadata', {}).get('schema_version', '1.0.0')
            }
            
            logger.debug(
                "Event serialized",
                event_type=metadata['original_event_type'],
                format=self.format.value,
                size_bytes=metadata['serialized_size_bytes'],
                time_ms=serialization_time_ms
            )
            
            return serialized_data, metadata
            
        except Exception as e:
            logger.error(
                "Event serialization failed",
                error=str(e),
                format=self.format.value,
                exc_info=True
            )
            raise
    
    def deserialize_event(self, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> ObservabilityEvent:
        """
        Deserialize binary data to observability event.
        
        Args:
            data: Serialized event data
            metadata: Optional serialization metadata
            
        Returns:
            Deserialized observability event
        """
        start_time = datetime.utcnow()
        
        try:
            # Determine format from metadata or use configured format
            format_str = metadata.get('serialization_format', self.format.value) if metadata else self.format.value
            is_compressed = metadata.get('compressed', False) if metadata else False
            
            # Decompress if needed
            if is_compressed:
                import zlib
                data = zlib.decompress(data)
            
            # Deserialize based on format
            if format_str == SerializationFormat.JSON.value:
                import json
                event_dict = json.loads(data.decode('utf-8'))
            elif format_str in [SerializationFormat.MSGPACK.value, SerializationFormat.MSGPACK_COMPRESSED.value]:
                event_dict = msgpack.unpackb(
                    data,
                    object_hook=self._msgpack_decoder,
                    strict_map_key=False
                )
            else:
                raise ValueError(f"Unsupported deserialization format: {format_str}")
            
            # Convert back to proper types
            event_dict = self._restore_types(event_dict)
            
            # Calculate performance metrics
            end_time = datetime.utcnow()
            deserialization_time_ms = (end_time - start_time).total_seconds() * 1000
            
            logger.debug(
                "Event deserialized",
                event_type=event_dict.get('event_type', 'unknown'),
                format=format_str,
                time_ms=deserialization_time_ms
            )
            
            return event_dict
            
        except Exception as e:
            logger.error(
                "Event deserialization failed",
                error=str(e),
                format=format_str if 'format_str' in locals() else 'unknown',
                exc_info=True
            )
            raise
    
    def _estimate_size(self, data: Dict[str, Any]) -> int:
        """Estimate size of data structure."""
        import json
        return len(json.dumps(data, default=str))
    
    def _optimize_payload(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize event payload for size constraints."""
        optimized = event_dict.copy()
        
        # Truncate large payload fields
        if 'payload' in optimized:
            payload = optimized['payload']
            
            # Truncate long strings
            for key, value in payload.items():
                if isinstance(value, str) and len(value) > 10000:
                    payload[f"{key}_truncated"] = True
                    payload[f"{key}_original_size"] = len(value)
                    payload[key] = value[:10000] + "... (truncated)"
            
            # Remove large objects
            if 'large_data' in payload:
                payload['large_data_removed'] = True
                del payload['large_data']
        
        # Truncate semantic embeddings if too large
        if 'semantic_embedding' in optimized and optimized['semantic_embedding']:
            if len(optimized['semantic_embedding']) > 1536:
                optimized['semantic_embedding'] = optimized['semantic_embedding'][:1536]
                optimized['embedding_truncated'] = True
        
        logger.debug(
            "Event payload optimized",
            original_size=self._estimate_size(event_dict),
            optimized_size=self._estimate_size(optimized)
        )
        
        return optimized
    
    def _restore_types(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Restore proper types from deserialized data."""
        restored = event_dict.copy()
        
        # Convert string UUIDs back to UUID objects where expected
        uuid_fields = ['event_id', 'workflow_id', 'agent_id', 'session_id', 'context_id', 'message_id']
        for field in uuid_fields:
            if field in restored and isinstance(restored[field], str):
                try:
                    restored[field] = uuid.UUID(restored[field])
                except (ValueError, TypeError):
                    pass  # Keep as string if not valid UUID
        
        # Convert datetime strings back to datetime objects
        if 'timestamp' in restored and isinstance(restored['timestamp'], str):
            try:
                restored['timestamp'] = datetime.fromisoformat(restored['timestamp'])
            except (ValueError, TypeError):
                pass  # Keep as string if not valid datetime
        
        return restored
    
    def serialize_batch(self, events: List[ObservabilityEvent]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Serialize multiple events efficiently in batch.
        
        Args:
            events: List of observability events
            
        Returns:
            Tuple of (serialized_data, metadata)
        """
        start_time = datetime.utcnow()
        
        try:
            # Convert all events to dictionaries
            event_dicts = []
            for event in events:
                if hasattr(event, 'dict'):
                    event_dicts.append(event.dict())
                else:
                    event_dicts.append(event)
            
            # Create batch structure
            batch_data = {
                'batch_id': str(uuid.uuid4()),
                'batch_timestamp': datetime.utcnow().isoformat(),
                'event_count': len(events),
                'events': event_dicts
            }
            
            # Serialize batch
            serialized_data, _ = self.serialize_event(batch_data)
            
            # Calculate performance metrics
            end_time = datetime.utcnow()
            batch_time_ms = (end_time - start_time).total_seconds() * 1000
            
            metadata = {
                'batch_serialization': True,
                'event_count': len(events),
                'batch_size_bytes': len(serialized_data),
                'batch_time_ms': batch_time_ms,
                'avg_time_per_event_ms': batch_time_ms / len(events) if events else 0,
                'serialization_format': self.format.value
            }
            
            logger.debug(
                "Event batch serialized",
                event_count=len(events),
                batch_size_bytes=len(serialized_data),
                time_ms=batch_time_ms
            )
            
            return serialized_data, metadata
            
        except Exception as e:
            logger.error(
                "Batch serialization failed",
                error=str(e),
                event_count=len(events),
                exc_info=True
            )
            raise
    
    def benchmark_performance(self, sample_events: List[ObservabilityEvent], iterations: int = 1000) -> Dict[str, float]:
        """
        Benchmark serialization performance.
        
        Args:
            sample_events: Sample events for benchmarking
            iterations: Number of iterations to run
            
        Returns:
            Performance metrics dictionary
        """
        if not sample_events:
            raise ValueError("Sample events required for benchmarking")
        
        logger.info(f"Starting serialization benchmark with {iterations} iterations")
        
        # Benchmark serialization
        start_time = datetime.utcnow()
        serialize_times = []
        deserialize_times = []
        
        for i in range(iterations):
            event = sample_events[i % len(sample_events)]
            
            # Serialize
            ser_start = datetime.utcnow()
            serialized_data, metadata = self.serialize_event(event)
            ser_end = datetime.utcnow()
            serialize_times.append((ser_end - ser_start).total_seconds() * 1000)
            
            # Deserialize
            deser_start = datetime.utcnow()
            self.deserialize_event(serialized_data, metadata)
            deser_end = datetime.utcnow()
            deserialize_times.append((deser_end - deser_start).total_seconds() * 1000)
        
        end_time = datetime.utcnow()
        total_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Calculate statistics
        avg_serialize_ms = sum(serialize_times) / len(serialize_times)
        avg_deserialize_ms = sum(deserialize_times) / len(deserialize_times)
        max_serialize_ms = max(serialize_times)
        max_deserialize_ms = max(deserialize_times)
        min_serialize_ms = min(serialize_times)
        min_deserialize_ms = min(deserialize_times)
        
        # Calculate percentiles
        serialize_times.sort()
        deserialize_times.sort()
        p95_serialize = serialize_times[int(0.95 * len(serialize_times))]
        p95_deserialize = deserialize_times[int(0.95 * len(deserialize_times))]
        
        results = {
            'total_time_ms': total_time_ms,
            'iterations': iterations,
            'avg_serialize_ms': avg_serialize_ms,
            'avg_deserialize_ms': avg_deserialize_ms,
            'max_serialize_ms': max_serialize_ms,
            'max_deserialize_ms': max_deserialize_ms,
            'min_serialize_ms': min_serialize_ms,
            'min_deserialize_ms': min_deserialize_ms,
            'p95_serialize_ms': p95_serialize,
            'p95_deserialize_ms': p95_deserialize,
            'avg_total_ms': avg_serialize_ms + avg_deserialize_ms,
            'events_per_second': (iterations * 1000) / total_time_ms,
            'format': self.format.value,
            'compression_enabled': self.enable_compression
        }
        
        logger.info(
            "Serialization benchmark completed",
            **results
        )
        
        return results


# Global serializer instances for different use cases
_high_performance_serializer = None
_compressed_serializer = None
_json_serializer = None


def get_high_performance_serializer() -> EventSerializer:
    """Get high-performance MessagePack serializer."""
    global _high_performance_serializer
    if _high_performance_serializer is None:
        _high_performance_serializer = EventSerializer(
            format=SerializationFormat.MSGPACK,
            enable_compression=False,
            max_payload_size=50000
        )
    return _high_performance_serializer


def get_compressed_serializer() -> EventSerializer:
    """Get compressed MessagePack serializer for storage."""
    global _compressed_serializer
    if _compressed_serializer is None:
        _compressed_serializer = EventSerializer(
            format=SerializationFormat.MSGPACK_COMPRESSED,
            enable_compression=True,
            max_payload_size=100000
        )
    return _compressed_serializer


def get_json_serializer() -> EventSerializer:
    """Get JSON serializer for debugging and compatibility."""
    global _json_serializer
    if _json_serializer is None:
        _json_serializer = EventSerializer(
            format=SerializationFormat.JSON,
            enable_compression=False,
            max_payload_size=50000
        )
    return _json_serializer


def serialize_for_stream(event: ObservabilityEvent) -> Tuple[bytes, Dict[str, Any]]:
    """
    Serialize event for Redis stream with optimal performance.
    
    Args:
        event: Event to serialize
        
    Returns:
        Tuple of (serialized_data, metadata)
    """
    return get_high_performance_serializer().serialize_event(event)


def serialize_for_storage(event: ObservabilityEvent) -> Tuple[bytes, Dict[str, Any]]:
    """
    Serialize event for long-term storage with compression.
    
    Args:
        event: Event to serialize
        
    Returns:
        Tuple of (serialized_data, metadata)
    """
    return get_compressed_serializer().serialize_event(event)


def deserialize_from_stream(data: bytes, metadata: Optional[Dict[str, Any]] = None) -> ObservabilityEvent:
    """
    Deserialize event from Redis stream.
    
    Args:
        data: Serialized event data
        metadata: Optional serialization metadata
        
    Returns:
        Deserialized event
    """
    return get_high_performance_serializer().deserialize_event(data, metadata)


def deserialize_from_storage(data: bytes, metadata: Optional[Dict[str, Any]] = None) -> ObservabilityEvent:
    """
    Deserialize event from storage.
    
    Args:
        data: Serialized event data
        metadata: Optional serialization metadata
        
    Returns:
        Deserialized event
    """
    return get_compressed_serializer().deserialize_event(data, metadata)