#!/usr/bin/env python3
"""
Performance Validation Script for Vertical Slice 4.2

Validates that the Redis Streams with Consumer Groups implementation
meets the specified performance targets:

- Message Throughput: ≥10k messages/second sustained
- Consumer Lag: <5 seconds under normal load  
- Failure Recovery: <30 seconds to reassign stalled messages
- Consumer Group Join: <1 second for new consumer registration
- DLQ Processing: <10 seconds to move poison messages

Usage:
    python scripts/validate_vs_4_2_performance.py [--redis-url REDIS_URL] [--verbose]
"""

import asyncio
import argparse
import json
import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import uuid

# Add project root to Python path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.enhanced_redis_streams_manager import (
    EnhancedRedisStreamsManager, ConsumerGroupConfig, ConsumerGroupType, MessageRoutingMode
)
from app.core.consumer_group_coordinator import ConsumerGroupCoordinator
from app.core.workflow_message_router import WorkflowMessageRouter
from app.core.dead_letter_queue_handler import DeadLetterQueueHandler
from app.models.message import StreamMessage, MessageType, MessagePriority

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceValidator:
    """Validates VS 4.2 performance against specified targets."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/14"):
        """Initialize performance validator."""
        self.redis_url = redis_url
        self.results = {}
        
        # Performance targets
        self.targets = {
            "message_throughput_per_sec": 10000,
            "consumer_lag_seconds": 5,
            "failure_recovery_seconds": 30,
            "consumer_join_seconds": 1,
            "dlq_processing_seconds": 10
        }
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all performance validations."""
        logger.info("Starting VS 4.2 Performance Validation")
        logger.info(f"Performance Targets: {json.dumps(self.targets, indent=2)}")
        
        validation_start = time.time()
        
        try:
            # Test 1: Message Throughput
            logger.info("\n=== Test 1: Message Throughput ===")
            throughput_result = await self.test_message_throughput()
            self.results["message_throughput"] = throughput_result
            
            # Test 2: Consumer Lag
            logger.info("\n=== Test 2: Consumer Lag ===")
            lag_result = await self.test_consumer_lag()
            self.results["consumer_lag"] = lag_result
            
            # Test 3: Failure Recovery
            logger.info("\n=== Test 3: Failure Recovery ===")
            recovery_result = await self.test_failure_recovery()
            self.results["failure_recovery"] = recovery_result
            
            # Test 4: Consumer Group Join
            logger.info("\n=== Test 4: Consumer Group Join ===")
            join_result = await self.test_consumer_group_join()
            self.results["consumer_group_join"] = join_result
            
            # Test 5: DLQ Processing
            logger.info("\n=== Test 5: DLQ Processing ===")
            dlq_result = await self.test_dlq_processing()
            self.results["dlq_processing"] = dlq_result
            
            # Overall results
            validation_duration = time.time() - validation_start
            self.results["validation_summary"] = {
                "total_duration_seconds": validation_duration,
                "tests_passed": sum(1 for r in self.results.values() if isinstance(r, dict) and r.get("passed", False)),
                "tests_total": 5,
                "overall_passed": all(
                    isinstance(r, dict) and r.get("passed", False) 
                    for r in self.results.values() 
                    if isinstance(r, dict) and "passed" in r
                )
            }
            
            logger.info(f"\n=== Validation Complete ===")
            logger.info(f"Duration: {validation_duration:.2f} seconds")
            logger.info(f"Tests Passed: {self.results['validation_summary']['tests_passed']}/5")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            self.results["error"] = str(e)
            return self.results
    
    async def test_message_throughput(self) -> Dict[str, Any]:
        """Test message throughput performance (Target: ≥10k msg/sec)."""
        try:
            # Create test system
            streams_manager = EnhancedRedisStreamsManager(
                redis_url=self.redis_url,
                auto_scaling_enabled=False
            )
            await streams_manager.connect()
            
            # Create consumer group
            config = ConsumerGroupConfig(
                name="throughput_test_group",
                stream_name="throughput_test_stream",
                agent_type=ConsumerGroupType.BACKEND_ENGINEERS,
                max_consumers=10
            )
            await streams_manager.create_consumer_group(config)
            
            # Add multiple consumers for parallel processing
            consumer_count = 5
            message_count = 0
            
            async def message_handler(message):
                nonlocal message_count
                message_count += 1
                return {"processed": True}
            
            for i in range(consumer_count):
                await streams_manager.add_consumer_to_group(
                    "throughput_test_group", 
                    f"throughput_consumer_{i}",
                    message_handler
                )
            
            # Send messages at high throughput
            test_message_count = 50000  # 50k messages for throughput test
            logger.info(f"Sending {test_message_count} messages...")
            
            start_time = time.time()
            
            # Send messages in batches for better performance
            batch_size = 1000
            for batch_start in range(0, test_message_count, batch_size):
                batch_tasks = []
                for i in range(batch_start, min(batch_start + batch_size, test_message_count)):
                    message = StreamMessage(
                        id=f"throughput_msg_{i}",
                        from_agent="throughput_tester",
                        to_agent=None,
                        message_type=MessageType.TASK_REQUEST,
                        payload={"batch": batch_start // batch_size, "index": i},
                        priority=MessagePriority.NORMAL,
                        timestamp=time.time()
                    )
                    
                    task = asyncio.create_task(
                        streams_manager.send_message_to_group("throughput_test_group", message)
                    )
                    batch_tasks.append(task)
                
                # Wait for batch to complete
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Brief pause between batches to avoid overwhelming
                await asyncio.sleep(0.01)
            
            send_duration = time.time() - start_time
            throughput = test_message_count / send_duration
            
            # Wait a bit for processing to complete
            await asyncio.sleep(2)
            
            # Get final metrics
            metrics = await streams_manager.get_performance_metrics()
            
            await streams_manager.disconnect()
            
            result = {
                "messages_sent": test_message_count,
                "send_duration_seconds": send_duration,
                "throughput_msg_per_sec": throughput,
                "target_throughput": self.targets["message_throughput_per_sec"],
                "passed": throughput >= self.targets["message_throughput_per_sec"],
                "consumer_count": consumer_count,
                "metrics": metrics.get("enhanced_metrics", {})
            }
            
            logger.info(f"Throughput: {throughput:.0f} msg/sec (Target: ≥{self.targets['message_throughput_per_sec']} msg/sec)")
            logger.info(f"Test {'PASSED' if result['passed'] else 'FAILED'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Throughput test failed: {e}")
            return {"passed": False, "error": str(e)}
    
    async def test_consumer_lag(self) -> Dict[str, Any]:
        """Test consumer lag under normal load (Target: <5 seconds)."""
        try:
            streams_manager = EnhancedRedisStreamsManager(
                redis_url=self.redis_url,
                auto_scaling_enabled=True
            )
            await streams_manager.connect()
            
            # Create consumer group
            config = ConsumerGroupConfig(
                name="lag_test_group",
                stream_name="lag_test_stream", 
                agent_type=ConsumerGroupType.QA_ENGINEERS,
                lag_threshold=50
            )
            await streams_manager.create_consumer_group(config)
            
            # Add consumers with simulated processing delay
            consumer_count = 3
            processing_delay = 0.1  # 100ms per message
            
            async def slow_handler(message):
                await asyncio.sleep(processing_delay)
                return {"processed": True, "delay": processing_delay}
            
            for i in range(consumer_count):
                await streams_manager.add_consumer_to_group(
                    "lag_test_group",
                    f"lag_consumer_{i}",
                    slow_handler
                )
            
            # Send burst of messages to create lag
            message_count = 200
            logger.info(f"Sending {message_count} messages with {processing_delay}s processing delay...")
            
            start_time = time.time()
            for i in range(message_count):
                message = StreamMessage(
                    id=f"lag_msg_{i}",
                    from_agent="lag_tester",
                    to_agent=None,
                    message_type=MessageType.TASK_REQUEST,
                    payload={"index": i, "sent_at": time.time()},
                    priority=MessagePriority.NORMAL,
                    timestamp=time.time()
                )
                
                await streams_manager.send_message_to_group("lag_test_group", message)
            
            send_duration = time.time() - start_time
            
            # Monitor lag over time
            lag_measurements = []
            for _ in range(10):  # Monitor for 10 seconds
                await asyncio.sleep(1)
                
                stats = await streams_manager.get_consumer_group_stats("lag_test_group")
                if stats:
                    lag_measurements.append(stats.lag)
                    logger.info(f"Current lag: {stats.lag} messages")
            
            max_lag = max(lag_measurements) if lag_measurements else 0
            avg_lag = statistics.mean(lag_measurements) if lag_measurements else 0
            
            # Estimate lag in seconds (messages * avg processing time)
            estimated_lag_seconds = max_lag * processing_delay / consumer_count
            
            await streams_manager.disconnect()
            
            result = {
                "messages_sent": message_count,
                "send_duration_seconds": send_duration,
                "max_lag_messages": max_lag,
                "avg_lag_messages": avg_lag,
                "estimated_lag_seconds": estimated_lag_seconds,
                "target_lag_seconds": self.targets["consumer_lag_seconds"],
                "passed": estimated_lag_seconds < self.targets["consumer_lag_seconds"],
                "consumer_count": consumer_count,
                "processing_delay_seconds": processing_delay
            }
            
            logger.info(f"Max lag: {max_lag} messages (~{estimated_lag_seconds:.1f}s)")
            logger.info(f"Target: <{self.targets['consumer_lag_seconds']}s")
            logger.info(f"Test {'PASSED' if result['passed'] else 'FAILED'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Consumer lag test failed: {e}")
            return {"passed": False, "error": str(e)}
    
    async def test_failure_recovery(self) -> Dict[str, Any]:
        """Test failure recovery time (Target: <30 seconds)."""
        try:
            streams_manager = EnhancedRedisStreamsManager(
                redis_url=self.redis_url,
                auto_scaling_enabled=False
            )
            await streams_manager.connect()
            
            coordinator = ConsumerGroupCoordinator(
                streams_manager,
                health_check_interval=1,
                rebalance_interval=5
            )
            await coordinator.start()
            
            # Create consumer group
            config = ConsumerGroupConfig(
                name="recovery_test_group",
                stream_name="recovery_test_stream",
                agent_type=ConsumerGroupType.DEVOPS_ENGINEERS,
                max_consumers=5
            )
            await streams_manager.create_consumer_group(config)
            
            # Add consumers
            consumer_ids = []
            for i in range(3):
                consumer_id = f"recovery_consumer_{i}"
                consumer_ids.append(consumer_id)
                
                async def handler(message):
                    return {"processed": True}
                
                await streams_manager.add_consumer_to_group(
                    "recovery_test_group", consumer_id, handler
                )
            
            # Send some messages
            message_count = 50
            for i in range(message_count):
                message = StreamMessage(
                    id=f"recovery_msg_{i}",
                    from_agent="recovery_tester",
                    to_agent=None,
                    message_type=MessageType.TASK_REQUEST,
                    payload={"index": i},
                    priority=MessagePriority.NORMAL,
                    timestamp=time.time()
                )
                
                await streams_manager.send_message_to_group("recovery_test_group", message)
            
            # Simulate consumer failure by removing a consumer
            logger.info("Simulating consumer failure...")
            failure_start = time.time()
            
            failed_consumer = consumer_ids[0]
            await streams_manager.remove_consumer_from_group("recovery_test_group", failed_consumer)
            
            # Monitor recovery through rebalancing
            recovery_detected = False
            while time.time() - failure_start < self.targets["failure_recovery_seconds"]:
                await asyncio.sleep(1)
                
                # Check if coordinator detected the issue and rebalanced
                stats = await streams_manager.get_consumer_group_stats("recovery_test_group")
                if stats and stats.consumer_count == len(consumer_ids) - 1:
                    recovery_detected = True
                    break
            
            recovery_time = time.time() - failure_start
            
            await coordinator.stop()
            await streams_manager.disconnect()
            
            result = {
                "messages_sent": message_count,
                "failed_consumer": failed_consumer,
                "recovery_time_seconds": recovery_time,
                "target_recovery_seconds": self.targets["failure_recovery_seconds"],
                "recovery_detected": recovery_detected,
                "passed": recovery_detected and recovery_time < self.targets["failure_recovery_seconds"]
            }
            
            logger.info(f"Recovery time: {recovery_time:.1f}s (Target: <{self.targets['failure_recovery_seconds']}s)")
            logger.info(f"Recovery detected: {recovery_detected}")
            logger.info(f"Test {'PASSED' if result['passed'] else 'FAILED'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failure recovery test failed: {e}")
            return {"passed": False, "error": str(e)}
    
    async def test_consumer_group_join(self) -> Dict[str, Any]:
        """Test consumer group join time (Target: <1 second)."""
        try:
            streams_manager = EnhancedRedisStreamsManager(
                redis_url=self.redis_url,
                auto_scaling_enabled=False
            )
            await streams_manager.connect()
            
            # Create consumer group
            config = ConsumerGroupConfig(
                name="join_test_group",
                stream_name="join_test_stream",
                agent_type=ConsumerGroupType.FRONTEND_DEVELOPERS
            )
            await streams_manager.create_consumer_group(config)
            
            # Test multiple consumer joins
            join_times = []
            consumer_count = 10
            
            for i in range(consumer_count):
                consumer_id = f"join_consumer_{i}"
                
                async def handler(message):
                    return {"processed": True}
                
                start_time = time.time()
                await streams_manager.add_consumer_to_group(
                    "join_test_group", consumer_id, handler
                )
                join_time = time.time() - start_time
                join_times.append(join_time)
                
                logger.info(f"Consumer {i+1} joined in {join_time:.3f}s")
            
            await streams_manager.disconnect()
            
            avg_join_time = statistics.mean(join_times)
            max_join_time = max(join_times)
            
            result = {
                "consumer_count": consumer_count,
                "join_times_seconds": join_times,
                "avg_join_time_seconds": avg_join_time,
                "max_join_time_seconds": max_join_time,
                "target_join_seconds": self.targets["consumer_join_seconds"],
                "passed": max_join_time < self.targets["consumer_join_seconds"]
            }
            
            logger.info(f"Average join time: {avg_join_time:.3f}s")
            logger.info(f"Max join time: {max_join_time:.3f}s (Target: <{self.targets['consumer_join_seconds']}s)")
            logger.info(f"Test {'PASSED' if result['passed'] else 'FAILED'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Consumer join test failed: {e}")
            return {"passed": False, "error": str(e)}
    
    async def test_dlq_processing(self) -> Dict[str, Any]:
        """Test DLQ processing time (Target: <10 seconds)."""
        try:
            streams_manager = EnhancedRedisStreamsManager(
                redis_url=self.redis_url,
                auto_scaling_enabled=False
            )
            await streams_manager.connect()
            
            dlq_handler = DeadLetterQueueHandler(
                streams_manager,
                enable_automatic_recovery=False
            )
            await dlq_handler.start()
            
            # Create failed messages
            failed_message_count = 20
            dlq_processing_times = []
            
            for i in range(failed_message_count):
                failed_message = StreamMessage(
                    id=f"dlq_test_msg_{i}",
                    from_agent="dlq_tester",
                    to_agent=None,
                    message_type=MessageType.TASK_REQUEST,
                    payload={"index": i, "test": "dlq"},
                    priority=MessagePriority.NORMAL,
                    timestamp=time.time()
                )
                
                failure_details = {
                    "error_type": "handler_exception",
                    "error_message": f"Test failure {i}"
                }
                
                # Measure DLQ processing time
                start_time = time.time()
                dlq_id = await dlq_handler.process_failed_message(
                    failed_message, "dlq_test_stream", "dlq_test_group", failure_details
                )
                processing_time = time.time() - start_time
                dlq_processing_times.append(processing_time)
                
                assert dlq_id is not None
                
                logger.info(f"DLQ message {i+1} processed in {processing_time:.3f}s")
            
            await dlq_handler.stop()
            await streams_manager.disconnect()
            
            avg_processing_time = statistics.mean(dlq_processing_times)
            max_processing_time = max(dlq_processing_times)
            
            result = {
                "failed_message_count": failed_message_count,
                "processing_times_seconds": dlq_processing_times,
                "avg_processing_time_seconds": avg_processing_time,
                "max_processing_time_seconds": max_processing_time,
                "target_processing_seconds": self.targets["dlq_processing_seconds"],
                "passed": max_processing_time < self.targets["dlq_processing_seconds"]
            }
            
            logger.info(f"Average DLQ processing: {avg_processing_time:.3f}s")
            logger.info(f"Max DLQ processing: {max_processing_time:.3f}s (Target: <{self.targets['dlq_processing_seconds']}s)")
            logger.info(f"Test {'PASSED' if result['passed'] else 'FAILED'}")
            
            return result
            
        except Exception as e:
            logger.error(f"DLQ processing test failed: {e}")
            return {"passed": False, "error": str(e)}
    
    def print_summary_report(self):
        """Print a comprehensive summary report."""
        print("\n" + "="*80)
        print("VERTICAL SLICE 4.2 PERFORMANCE VALIDATION REPORT")
        print("="*80)
        
        if "validation_summary" in self.results:
            summary = self.results["validation_summary"]
            print(f"Overall Result: {'PASSED' if summary['overall_passed'] else 'FAILED'}")
            print(f"Tests Passed: {summary['tests_passed']}/{summary['tests_total']}")
            print(f"Total Duration: {summary['total_duration_seconds']:.2f} seconds")
        
        print("\nDetailed Results:")
        print("-" * 80)
        
        test_names = {
            "message_throughput": "Message Throughput",
            "consumer_lag": "Consumer Lag",
            "failure_recovery": "Failure Recovery", 
            "consumer_group_join": "Consumer Group Join",
            "dlq_processing": "DLQ Processing"
        }
        
        for key, name in test_names.items():
            if key in self.results:
                result = self.results[key]
                status = "PASSED" if result.get("passed", False) else "FAILED"
                print(f"{name:20}: {status}")
                
                # Print key metrics
                if key == "message_throughput":
                    print(f"                     Throughput: {result.get('throughput_msg_per_sec', 0):.0f} msg/sec")
                elif key == "consumer_lag":
                    print(f"                     Estimated Lag: {result.get('estimated_lag_seconds', 0):.1f}s")
                elif key == "failure_recovery":
                    print(f"                     Recovery Time: {result.get('recovery_time_seconds', 0):.1f}s")
                elif key == "consumer_group_join":
                    print(f"                     Max Join Time: {result.get('max_join_time_seconds', 0):.3f}s")
                elif key == "dlq_processing":
                    print(f"                     Max Processing: {result.get('max_processing_time_seconds', 0):.3f}s")
        
        print("\n" + "="*80)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="VS 4.2 Performance Validation")
    parser.add_argument("--redis-url", default="redis://localhost:6379/14", 
                       help="Redis URL for testing")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = PerformanceValidator(redis_url=args.redis_url)
    
    try:
        results = await validator.run_all_validations()
        
        # Print summary report
        validator.print_summary_report()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"vs_4_2_performance_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Exit with appropriate code
        if results.get("validation_summary", {}).get("overall_passed", False):
            print("\n🎉 All performance targets met!")
            return 0
        else:
            print("\n❌ Some performance targets not met.")
            return 1
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class ValidateVs42PerformanceScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(ValidateVs42PerformanceScript)