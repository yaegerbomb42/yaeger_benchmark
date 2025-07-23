"""
Stream Source - Data input simulation for the pipeline
"""
import time
import random
import threading
from dataclasses import dataclass
from typing import List, Callable, Any, Dict
import queue
import json

@dataclass
class DataRecord:
    """Represents a single data record in the stream."""
    id: str
    timestamp: float
    data: Dict[str, Any]
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "data": self.data
        }
    
    def to_json(self):
        return json.dumps(self.to_dict())

class StreamSource:
    """Simulates a high-throughput data stream source."""
    
    def __init__(self, stream_name: str, target_rate: int = 1000):
        self.stream_name = stream_name
        self.target_rate = target_rate  # Records per second
        self.is_running = False
        self.subscribers = []
        self.record_queue = queue.Queue(maxsize=10000)
        self.stats = {
            "records_generated": 0,
            "records_sent": 0,
            "errors": 0
        }
    
    def subscribe(self, callback: Callable[[DataRecord], None]):
        """Subscribe to receive records from this stream."""
        self.subscribers.append(callback)
    
    def start(self):
        """Start generating and sending data records."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start generator thread
        generator_thread = threading.Thread(target=self._generate_records)
        generator_thread.daemon = True
        generator_thread.start()
        
        # Start sender thread
        sender_thread = threading.Thread(target=self._send_records)
        sender_thread.daemon = True
        sender_thread.start()
    
    def stop(self):
        """Stop the stream source."""
        self.is_running = False
    
    def _generate_records(self):
        """Generate records at the target rate."""
        record_id = 0
        
        while self.is_running:
            try:
                # Calculate delay to maintain target rate
                delay = 1.0 / self.target_rate
                
                record = DataRecord(
                    id=f"{self.stream_name}_{record_id}",
                    timestamp=time.time(),
                    data=self._generate_sample_data(record_id)
                )
                
                self.record_queue.put(record, timeout=1.0)
                self.stats["records_generated"] += 1
                record_id += 1
                
                time.sleep(delay)
                
            except queue.Full:
                self.stats["errors"] += 1
                # Skip record if queue is full (simulates backpressure)
            except Exception as e:
                self.stats["errors"] += 1
    
    def _send_records(self):
        """Send records to subscribers."""
        while self.is_running:
            try:
                record = self.record_queue.get(timeout=1.0)
                
                for callback in self.subscribers:
                    try:
                        callback(record)
                        self.stats["records_sent"] += 1
                    except Exception as e:
                        self.stats["errors"] += 1
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.stats["errors"] += 1
    
    def _generate_sample_data(self, record_id: int) -> Dict[str, Any]:
        """Generate sample data for testing."""
        event_types = ["click", "view", "purchase", "signup", "logout"]
        
        return {
            "user_id": random.randint(1, 100000),
            "session_id": f"session_{random.randint(1, 10000)}",
            "event_type": random.choice(event_types),
            "value": round(random.uniform(0, 1000), 2),
            "product_id": f"prod_{random.randint(1, 1000)}",
            "category": random.choice(["electronics", "clothing", "books", "food"]),
            "location": {
                "country": random.choice(["US", "UK", "CA", "AU", "DE"]),
                "city": random.choice(["New York", "London", "Toronto", "Sydney", "Berlin"])
            },
            "device": {
                "type": random.choice(["desktop", "mobile", "tablet"]),
                "os": random.choice(["Windows", "iOS", "Android", "macOS"])
            },
            "metadata": {
                "source": self.stream_name,
                "version": "1.0",
                "batch_id": record_id // 1000
            }
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get stream statistics."""
        return self.stats.copy()

class MultiStreamSource:
    """Manages multiple concurrent data streams."""
    
    def __init__(self):
        self.streams = {}
        self.global_callback = None
    
    def add_stream(self, stream_name: str, target_rate: int = 1000) -> StreamSource:
        """Add a new stream source."""
        stream = StreamSource(stream_name, target_rate)
        self.streams[stream_name] = stream
        
        if self.global_callback:
            stream.subscribe(self.global_callback)
        
        return stream
    
    def subscribe_all(self, callback: Callable[[DataRecord], None]):
        """Subscribe to all streams."""
        self.global_callback = callback
        
        for stream in self.streams.values():
            stream.subscribe(callback)
    
    def start_all(self):
        """Start all streams."""
        for stream in self.streams.values():
            stream.start()
    
    def stop_all(self):
        """Stop all streams."""
        for stream in self.streams.values():
            stream.stop()
    
    def get_total_stats(self) -> Dict[str, int]:
        """Get aggregated statistics across all streams."""
        total_stats = {
            "records_generated": 0,
            "records_sent": 0,
            "errors": 0,
            "active_streams": 0
        }
        
        for stream in self.streams.values():
            stats = stream.get_stats()
            total_stats["records_generated"] += stats["records_generated"]
            total_stats["records_sent"] += stats["records_sent"]
            total_stats["errors"] += stats["errors"]
            
            if stream.is_running:
                total_stats["active_streams"] += 1
        
        return total_stats
