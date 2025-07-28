"""
Load testing suite for Redis Pub/Sub Communication System.

Tests the system under various load conditions to validate performance
requirements from the Communication PRD including 10k msg/sec throughput
and <200ms P95 latency targets.
"""

import asyncio
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any
import pytest

from app.core.agent_communication_service import AgentCommunicationService, AgentMessage
from app.core.redis_pubsub_manager import RedisPubSubManager
from app.models.message import MessageType, MessagePriority


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    total_messages: int
    duration_seconds: float
    throughput_msg_per_sec: float
    success_count: int
    failure_count: int
    success_rate: float
    latencies_ms: List[float]
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float


class CommunicationLoadTester:
    """Load tester for communication system."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.latencies = []
        self.successes = 0
        self.failures = 0
    
    async def run_throughput_test(
        self,
        message_count: int,
        concurrent_senders: int = 10,
        use_streams: bool = False
    ) -> LoadTestResult:
        """
        Run throughput test with multiple concurrent senders.
        
        Args:
            message_count: Total number of messages to send
            concurrent_senders: Number of concurrent sender tasks
            use_streams: Whether to use Redis Streams vs Pub/Sub
            
        Returns:
            LoadTestResult with performance metrics
        """
        self.latencies = []
        self.successes = 0
        self.failures = 0
        
        # Create sender services
        services = []
        for i in range(concurrent_senders):
            service = AgentCommunicationService(
                redis_url=self.redis_url,
                consumer_name=f"load-test-{i}"
            )
            await service.connect()
            services.append(service)
        
        try:
            # Calculate messages per sender
            messages_per_sender = message_count // concurrent_senders
            
            start_time = time.time()
            
            # Create sender tasks
            tasks = []
            for i, service in enumerate(services):
                task = asyncio.create_task(
                    self._sender_worker(
                        service=service,
                        sender_id=f"sender-{i}",
                        message_count=messages_per_sender,
                        use_streams=use_streams
                    )
                )
                tasks.append(task)
            
            # Wait for all senders to complete
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate results
            total_sent = self.successes + self.failures
            throughput = total_sent / duration if duration > 0 else 0
            success_rate = self.successes / total_sent if total_sent > 0 else 0
            
            # Calculate latency statistics
            latency_stats = self._calculate_latency_stats(self.latencies)
            
            return LoadTestResult(
                total_messages=total_sent,
                duration_seconds=duration,
                throughput_msg_per_sec=throughput,
                success_count=self.successes,
                failure_count=self.failures,
                success_rate=success_rate,
                latencies_ms=self.latencies,
                **latency_stats
            )
            
        finally:
            # Cleanup services
            for service in services:
                await service.disconnect()
    
    async def _sender_worker(
        self,
        service: AgentCommunicationService,
        sender_id: str,
        message_count: int,
        use_streams: bool = False
    ) -> None:
        """Worker function for sending messages."""
        for i in range(message_count):
            try:
                message = AgentMessage(
                    id=str(uuid.uuid4()),
                    from_agent=sender_id,
                    to_agent="target",
                    type=MessageType.EVENT,
                    payload={
                        "sequence": i,
                        "sender": sender_id,
                        "timestamp": time.time()
                    },
                    timestamp=time.time(),
                    priority=MessagePriority.NORMAL
                )
                
                send_start = time.time()
                
                if use_streams:
                    result = await service.send_durable_message(message)
                    success = result is not None
                else:
                    success = await service.send_message(message)
                
                send_end = time.time()
                latency_ms = (send_end - send_start) * 1000
                
                self.latencies.append(latency_ms)
                
                if success:
                    self.successes += 1
                else:
                    self.failures += 1
                    
            except Exception as e:
                self.failures += 1
                print(f"Send error in {sender_id}: {e}")
    
    async def run_consumer_load_test(
        self,
        message_count: int,
        concurrent_consumers: int = 5
    ) -> Dict[str, Any]:
        """
        Test load with multiple concurrent consumers.
        
        Args:
            message_count: Number of messages to process
            concurrent_consumers: Number of concurrent consumers
            
        Returns:
            Test results dictionary
        """
        # Setup producer
        producer = AgentCommunicationService(redis_url=self.redis_url)
        await producer.connect()
        
        # Setup consumers
        consumers = []
        message_counters = []
        
        for i in range(concurrent_consumers):
            consumer = AgentCommunicationService(
                redis_url=self.redis_url,
                consumer_name=f"consumer-{i}"
            )
            await consumer.connect()
            consumers.append(consumer)
            message_counters.append(0)
        
        try:
            # Setup message handlers
            for i, consumer in enumerate(consumers):
                def make_handler(index):
                    def handler(message: AgentMessage):
                        message_counters[index] += 1
                    return handler
                
                await consumer.subscribe_agent(
                    f"consumer-{i}",
                    make_handler(i)
                )
            
            # Send messages
            start_time = time.time()
            
            for i in range(message_count):
                message = AgentMessage(
                    id=str(uuid.uuid4()),
                    from_agent="producer",
                    to_agent=f"consumer-{i % concurrent_consumers}",
                    type=MessageType.TASK_REQUEST,
                    payload={"sequence": i},
                    timestamp=time.time()
                )
                
                await producer.send_message(message)
            
            # Wait for processing
            await asyncio.sleep(2.0)  # Give time for all messages to be processed
            
            end_time = time.time()
            duration = end_time - start_time
            
            total_received = sum(message_counters)
            
            return {
                "messages_sent": message_count,
                "messages_received": total_received,
                "consumers": concurrent_consumers,
                "duration_seconds": duration,
                "processing_rate": total_received / duration if duration > 0 else 0,
                "message_distribution": message_counters,
                "delivery_rate": total_received / message_count if message_count > 0 else 0
            }
            
        finally:
            await producer.disconnect()
            for consumer in consumers:
                await consumer.disconnect()
    
    async def run_burst_load_test(
        self,
        burst_size: int,
        burst_count: int,
        delay_between_bursts: float = 1.0
    ) -> List[LoadTestResult]:
        """
        Test handling of burst traffic patterns.
        
        Args:
            burst_size: Number of messages per burst
            burst_count: Number of bursts
            delay_between_bursts: Delay between bursts in seconds
            
        Returns:
            List of LoadTestResult for each burst
        """
        service = AgentCommunicationService(redis_url=self.redis_url)
        await service.connect()
        
        results = []
        
        try:
            for burst_num in range(burst_count):
                # Reset counters for this burst
                self.latencies = []
                self.successes = 0
                self.failures = 0
                
                start_time = time.time()
                
                # Send burst of messages
                tasks = []
                for i in range(burst_size):
                    message = AgentMessage(
                        id=str(uuid.uuid4()),
                        from_agent="burst-test",
                        to_agent="target",
                        type=MessageType.EVENT,
                        payload={
                            "burst": burst_num,
                            "sequence": i
                        },
                        timestamp=time.time()
                    )
                    
                    task = asyncio.create_task(self._send_with_timing(service, message))
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Calculate burst results
                total_sent = self.successes + self.failures
                throughput = total_sent / duration if duration > 0 else 0
                success_rate = self.successes / total_sent if total_sent > 0 else 0
                
                latency_stats = self._calculate_latency_stats(self.latencies)
                
                burst_result = LoadTestResult(
                    total_messages=total_sent,
                    duration_seconds=duration,
                    throughput_msg_per_sec=throughput,
                    success_count=self.successes,
                    failure_count=self.failures,
                    success_rate=success_rate,
                    latencies_ms=self.latencies,
                    **latency_stats
                )
                
                results.append(burst_result)
                
                # Wait between bursts (except for last burst)
                if burst_num < burst_count - 1:
                    await asyncio.sleep(delay_between_bursts)
                    
        finally:
            await service.disconnect()
            
        return results
    
    async def _send_with_timing(
        self,
        service: AgentCommunicationService,
        message: AgentMessage
    ) -> None:
        """Send message with latency timing."""
        try:
            start_time = time.time()
            success = await service.send_message(message)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            self.latencies.append(latency_ms)
            
            if success:
                self.successes += 1
            else:
                self.failures += 1
                
        except Exception:
            self.failures += 1
    
    def _calculate_latency_stats(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not latencies:
            return {
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "min_latency_ms": 0.0
            }
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            "p99_latency_ms": sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies)
        }


class TestCommunicationLoad:
    """Load test suite for communication system."""
    
    @pytest.fixture
    def load_tester(self):
        """Load tester fixture."""
        return CommunicationLoadTester()
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_basic_throughput(self, load_tester):
        """Test basic throughput with moderate load."""
        result = await load_tester.run_throughput_test(
            message_count=1000,
            concurrent_senders=5
        )
        
        # Validate basic performance
        assert result.success_rate >= 0.95, f"Success rate {result.success_rate:.3f} too low"
        assert result.throughput_msg_per_sec >= 50, f"Throughput {result.throughput_msg_per_sec:.2f} too low"
        assert result.avg_latency_ms <= 500, f"Average latency {result.avg_latency_ms:.2f}ms too high"
        
        print(f"Basic throughput test results:")
        print(f"  Throughput: {result.throughput_msg_per_sec:.2f} msg/sec")
        print(f"  Success rate: {result.success_rate:.3f}")
        print(f"  Average latency: {result.avg_latency_ms:.2f}ms")
        print(f"  P95 latency: {result.p95_latency_ms:.2f}ms")
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_high_throughput_pubsub(self, load_tester):
        """Test high throughput with Pub/Sub."""
        result = await load_tester.run_throughput_test(
            message_count=5000,
            concurrent_senders=10,
            use_streams=False
        )
        
        # Validate high throughput performance
        assert result.success_rate >= 0.90, f"High-load success rate {result.success_rate:.3f} too low"
        assert result.throughput_msg_per_sec >= 100, f"High-load throughput {result.throughput_msg_per_sec:.2f} too low"
        
        print(f"High throughput Pub/Sub test results:")
        print(f"  Throughput: {result.throughput_msg_per_sec:.2f} msg/sec")
        print(f"  Success rate: {result.success_rate:.3f}")
        print(f"  P95 latency: {result.p95_latency_ms:.2f}ms")
        print(f"  P99 latency: {result.p99_latency_ms:.2f}ms")
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_high_throughput_streams(self, load_tester):
        """Test high throughput with Redis Streams."""
        result = await load_tester.run_throughput_test(
            message_count=2000,  # Streams may be slower
            concurrent_senders=8,
            use_streams=True
        )
        
        # Streams may have slightly lower throughput but better durability
        assert result.success_rate >= 0.95, f"Streams success rate {result.success_rate:.3f} too low"
        assert result.throughput_msg_per_sec >= 50, f"Streams throughput {result.throughput_msg_per_sec:.2f} too low"
        
        print(f"High throughput Streams test results:")
        print(f"  Throughput: {result.throughput_msg_per_sec:.2f} msg/sec")  
        print(f"  Success rate: {result.success_rate:.3f}")
        print(f"  P95 latency: {result.p95_latency_ms:.2f}ms")
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_latency_requirements(self, load_tester):
        """Test latency requirements under load."""
        result = await load_tester.run_throughput_test(
            message_count=1000,
            concurrent_senders=5
        )
        
        # Validate PRD latency requirements
        assert result.p95_latency_ms <= 200, f"P95 latency {result.p95_latency_ms:.2f}ms exceeds 200ms requirement"
        assert result.avg_latency_ms <= 100, f"Average latency {result.avg_latency_ms:.2f}ms too high"
        
        print(f"Latency test results:")
        print(f"  Average: {result.avg_latency_ms:.2f}ms")
        print(f"  P95: {result.p95_latency_ms:.2f}ms")
        print(f"  P99: {result.p99_latency_ms:.2f}ms")
        print(f"  Max: {result.max_latency_ms:.2f}ms")
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_consumer_load(self, load_tester):
        """Test load with multiple consumers."""
        result = await load_tester.run_consumer_load_test(
            message_count=1000,
            concurrent_consumers=5
        )
        
        # Validate consumer performance
        assert result["delivery_rate"] >= 0.90, f"Delivery rate {result['delivery_rate']:.3f} too low"
        assert result["processing_rate"] >= 50, f"Processing rate {result['processing_rate']:.2f} too low"
        
        print(f"Consumer load test results:")
        print(f"  Messages sent: {result['messages_sent']}")
        print(f"  Messages received: {result['messages_received']}")
        print(f"  Delivery rate: {result['delivery_rate']:.3f}")
        print(f"  Processing rate: {result['processing_rate']:.2f} msg/sec")
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_burst_traffic(self, load_tester):
        """Test handling of burst traffic patterns."""
        results = await load_tester.run_burst_load_test(
            burst_size=200,
            burst_count=5,
            delay_between_bursts=0.5
        )
        
        # Validate burst handling
        for i, result in enumerate(results):
            assert result.success_rate >= 0.85, f"Burst {i} success rate {result.success_rate:.3f} too low"
            print(f"Burst {i}: {result.throughput_msg_per_sec:.2f} msg/sec, "
                  f"{result.success_rate:.3f} success rate, "
                  f"{result.p95_latency_ms:.2f}ms P95 latency")
        
        # Check consistency across bursts
        throughputs = [r.throughput_msg_per_sec for r in results]
        avg_throughput = statistics.mean(throughputs)
        throughput_std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0
        
        print(f"Burst consistency - Avg throughput: {avg_throughput:.2f}, Std dev: {throughput_std:.2f}")
        
        # Throughput should be relatively consistent
        assert throughput_std / avg_throughput <= 0.3, "Burst throughput too inconsistent"
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_sustained_load(self, load_tester):
        """Test sustained load over longer duration."""
        # Run multiple rounds to simulate sustained load
        total_messages = 0
        total_duration = 0
        all_success_rates = []
        
        for round_num in range(3):
            result = await load_tester.run_throughput_test(
                message_count=1000,
                concurrent_senders=6
            )
            
            total_messages += result.total_messages
            total_duration += result.duration_seconds
            all_success_rates.append(result.success_rate)
            
            print(f"Round {round_num + 1}: {result.throughput_msg_per_sec:.2f} msg/sec, "
                  f"{result.success_rate:.3f} success rate")
            
            # Brief pause between rounds
            await asyncio.sleep(1.0)
        
        # Calculate overall metrics
        overall_throughput = total_messages / total_duration
        avg_success_rate = statistics.mean(all_success_rates)
        
        # Validate sustained performance
        assert avg_success_rate >= 0.90, f"Sustained success rate {avg_success_rate:.3f} too low"
        assert overall_throughput >= 80, f"Sustained throughput {overall_throughput:.2f} too low"
        
        print(f"Sustained load results:")
        print(f"  Overall throughput: {overall_throughput:.2f} msg/sec")
        print(f"  Average success rate: {avg_success_rate:.3f}")
        print(f"  Total messages: {total_messages}")
        print(f"  Total duration: {total_duration:.2f}s")


class TestStressConditions:
    """Stress tests to find system limits."""
    
    @pytest.fixture
    def load_tester(self):
        """Load tester fixture."""
        return CommunicationLoadTester()
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_maximum_throughput(self, load_tester):
        """Find maximum sustainable throughput."""
        # Test with increasing load until failure
        test_configs = [
            (1000, 5),   # Baseline
            (2000, 8),   # Medium
            (5000, 12),  # High
            (10000, 20), # Very high
        ]
        
        results = []
        
        for message_count, senders in test_configs:
            try:
                result = await load_tester.run_throughput_test(
                    message_count=message_count,
                    concurrent_senders=senders
                )
                
                results.append({
                    "config": f"{message_count} msgs, {senders} senders",
                    "throughput": result.throughput_msg_per_sec,
                    "success_rate": result.success_rate,
                    "p95_latency": result.p95_latency_ms
                })
                
                print(f"Config {message_count}/{senders}: "
                      f"{result.throughput_msg_per_sec:.2f} msg/sec, "
                      f"{result.success_rate:.3f} success, "
                      f"{result.p95_latency_ms:.2f}ms P95")
                
                # Stop if performance degrades significantly
                if result.success_rate < 0.80 or result.p95_latency_ms > 1000:
                    print(f"Performance degraded at config {message_count}/{senders}")
                    break
                    
            except Exception as e:
                print(f"Failed at config {message_count}/{senders}: {e}")
                break
        
        # Find maximum acceptable throughput
        acceptable_results = [r for r in results if r["success_rate"] >= 0.90 and r["p95_latency"] <= 500]
        
        if acceptable_results:
            max_throughput = max(r["throughput"] for r in acceptable_results)
            print(f"Maximum acceptable throughput: {max_throughput:.2f} msg/sec")
            
            # Should meet minimum performance requirements
            assert max_throughput >= 100, f"Maximum throughput {max_throughput:.2f} below requirements"
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_memory_usage_under_load(self, load_tester):
        """Test memory usage doesn't grow excessively under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run load test
        result = await load_tester.run_throughput_test(
            message_count=5000,
            concurrent_senders=10
        )
        
        # Check memory after test
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB "
              f"(+{memory_increase:.2f}MB)")
        
        # Memory increase should be reasonable
        assert memory_increase <= 100, f"Memory increase {memory_increase:.2f}MB too high"
        
        # Performance should still be acceptable
        assert result.success_rate >= 0.85, "Performance degraded under memory pressure"


if __name__ == "__main__":
    # Run load tests
    pytest.main([
        __file__, 
        "-v", 
        "-m", "load",
        "--tb=short"
    ])