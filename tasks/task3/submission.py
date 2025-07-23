"""
Dynamic Data Pipeline Implementation

Implement a high-throughput stream processing system that can handle 1M records/second.
Your solution will be assessed on:
1. Throughput (1M+ records/second)
2. Memory efficiency (<1GB for large datasets)
3. Fault tolerance and error handling
4. Data consistency and accuracy

API Usage:
- DataPipeline: Main pipeline orchestrator
- StreamProcessor: Process individual records or batches
- DataTransformer: Transform, validate, and enrich data

Performance Requirements:
- Handle 1M+ records per second throughput
- Process streaming data with <100ms latency
- Maintain <1GB memory usage for large datasets
- Provide 99.9% data accuracy and consistency
"""

from codebase.stream_source import StreamSource, DataRecord, MultiStreamSource
from codebase.stream_sink import StreamSink, MemorySink, MultiSink, AsyncBatchSink
from codebase.pipeline_metrics import PipelineMetrics
import time
import threading
from typing import List, Dict, Any, Optional, Callable
import queue
import json

class DataPipeline:
    """Main data pipeline orchestrator for high-throughput stream processing."""
    
    def __init__(self):
        self.sinks = MultiSink()
        self.transformer = DataTransformer()
        self.processor = StreamProcessor()
        self.metrics = PipelineMetrics()
        self.is_running = False
        
        # Configuration
        self.batch_size = 1000
        self.worker_threads = 4
        self.max_queue_size = 10000
        
        # Internal state
        self.input_queue = queue.Queue(maxsize=self.max_queue_size)
        self.workers = []
        
    def add_sink(self, sink: StreamSink, name: str = None):
        """Add a data sink to the pipeline."""
        if name is None:
            name = f"sink_{len(self.sinks.sinks)}"
        self.sinks.add_sink(name, sink)
    
    def start(self):
        """Start the pipeline processing."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker threads
        for i in range(self.worker_threads):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop the pipeline processing."""
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
    
    def process_batch(self, records: List[Any]) -> int:
        """Process a batch of records synchronously."""
        start_time = time.time()
        processed_count = 0
        
        try:
            # Basic processing - transform and write to sinks
            for record in records:
                try:
                    # Transform the record
                    transformed = self.transformer.transform_record(record)
                    if transformed:
                        # Write to sinks
                        successful_writes = self.sinks.write(transformed)
                        if successful_writes > 0:
                            processed_count += 1
                        
                        # Record metrics
                        record_latency = time.time() - start_time
                        self.metrics.record_processing(
                            latency=record_latency,
                            success=successful_writes > 0,
                            bytes_size=len(str(record))
                        )
                except Exception as e:
                    self.metrics.record_processing(
                        latency=time.time() - start_time,
                        success=False,
                        error_type=type(e).__name__
                    )
            
            return processed_count
            
        except Exception as e:
            self.metrics.record_processing(
                latency=time.time() - start_time,
                success=False,
                error_type=type(e).__name__
            )
            return processed_count
    
    def process_stream(self, stream_id: str, records: List[Any]) -> int:
        """Process records from a specific stream."""
        return self.process_batch(records)
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop for processing queued records."""
        while self.is_running:
            try:
                # Get batch from queue
                batch = self.input_queue.get(timeout=1.0)
                
                # Process the batch
                self.process_batch(batch)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.metrics.record_processing(
                    latency=0,
                    success=False,
                    error_type=type(e).__name__
                )

class StreamProcessor:
    """Handles individual record and batch processing logic."""
    
    def __init__(self):
        self.filters = []
        self.enrichers = []
        
    def add_filter(self, filter_func: Callable[[Any], bool]):
        """Add a filter function to the processor."""
        self.filters.append(filter_func)
    
    def add_enricher(self, enricher_func: Callable[[Any], Any]):
        """Add an enricher function to the processor."""
        self.enrichers.append(enricher_func)
    
    def process_record(self, record: Any) -> Optional[Any]:
        """Process a single record through filters and enrichers."""
        try:
            # Apply filters
            for filter_func in self.filters:
                if not filter_func(record):
                    return None  # Record filtered out
            
            # Apply enrichers
            for enricher_func in self.enrichers:
                record = enricher_func(record)
                if record is None:
                    break
            
            return record
            
        except Exception:
            return None  # Failed processing
    
    def process_batch(self, records: List[Any]) -> List[Any]:
        """Process a batch of records."""
        processed = []
        
        for record in records:
            result = self.process_record(record)
            if result is not None:
                processed.append(result)
        
        return processed

class DataTransformer:
    """Handles data transformation, validation, and enrichment."""
    
    def __init__(self):
        self.validation_rules = []
        self.transformation_rules = []
        
    def transform_record(self, record: Any) -> Optional[Dict[str, Any]]:
        """Transform a single record."""
        try:
            # Convert to dict if needed
            if hasattr(record, 'to_dict'):
                data = record.to_dict()
            elif hasattr(record, '__dict__'):
                data = record.__dict__
            elif isinstance(record, dict):
                data = record.copy()
            else:
                # Try to convert to dict
                data = {"value": record, "timestamp": time.time()}
            
            # Basic validation and cleaning
            cleaned = self.clean_record(data)
            if not cleaned:
                return None
            
            # Add processing metadata
            cleaned["processed_at"] = time.time()
            cleaned["pipeline_version"] = "1.0"
            
            return cleaned
            
        except Exception:
            return None
    
    def clean_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and validate a record."""
        if not isinstance(record, dict):
            return None
        
        cleaned = {}
        
        for key, value in record.items():
            # Skip None values
            if value is None:
                continue
            
            # Clean string values
            if isinstance(value, str):
                value = value.strip()
                if not value:  # Skip empty strings
                    continue
            
            # Validate numbers
            if isinstance(value, (int, float)):
                if value < 0 and key == "value":
                    continue  # Skip negative values for 'value' field
            
            cleaned[key] = value
        
        # Require at least some content
        return cleaned if cleaned else None
    
    def clean_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean a batch of records."""
        cleaned = []
        
        for record in records:
            clean_record = self.clean_record(record)
            if clean_record:
                cleaned.append(clean_record)
        
        return cleaned
    
    def transform_batch(self, records: List[Any]) -> List[Dict[str, Any]]:
        """Transform a batch of records."""
        transformed = []
        
        for record in records:
            result = self.transform_record(record)
            if result:
                transformed.append(result)
        
        return transformed
    
    def enrich_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich a batch of records with additional data."""
        enriched = []
        
        for record in records:
            # Add timestamp if missing
            if "timestamp" not in record:
                record["timestamp"] = time.time()
            
            # Add processing metadata
            record["processed_at"] = time.time()
            
            enriched.append(record)
        
        return enriched
    
    def deduplicate(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate records, keeping the most recent."""
        seen = {}
        
        for record in records:
            record_id = record.get("id")
            if record_id is None:
                continue
            
            timestamp = record.get("timestamp", 0)
            
            if record_id not in seen or timestamp > seen[record_id]["timestamp"]:
                seen[record_id] = record
        
        return list(seen.values())
    
    def window_aggregate(self, records: List[Dict[str, Any]], window_size: int = 300) -> List[Dict[str, Any]]:
        """Aggregate records into time windows."""
        if not records:
            return []
        
        # Sort by timestamp
        sorted_records = sorted(records, key=lambda x: x.get("timestamp", 0))
        
        windows = []
        current_window = []
        window_start = sorted_records[0].get("timestamp", 0)
        
        for record in sorted_records:
            timestamp = record.get("timestamp", 0)
            
            if timestamp - window_start >= window_size:
                # Finish current window
                if current_window:
                    window_summary = self._summarize_window(current_window, window_start, timestamp)
                    windows.append(window_summary)
                
                # Start new window
                current_window = []
                window_start = timestamp
            
            current_window.append(record)
        
        # Finish final window
        if current_window:
            final_timestamp = current_window[-1].get("timestamp", window_start)
            window_summary = self._summarize_window(current_window, window_start, final_timestamp)
            windows.append(window_summary)
        
        return windows
    
    def _summarize_window(self, records: List[Dict[str, Any]], start_time: float, end_time: float) -> Dict[str, Any]:
        """Summarize a window of records."""
        values = [r.get("value", 0) for r in records if isinstance(r.get("value"), (int, float))]
        
        return {
            "start_time": start_time,
            "end_time": end_time,
            "count": len(records),
            "sum": sum(values) if values else 0,
            "avg": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0
        }

def process_stream(data_stream):
    """
    Process incoming data stream and perform real-time aggregations.
    
    Args:
        data_stream: Iterator of JSON records
        
    Returns:
        Processed results
    """
    # TODO: Implement high-performance stream processing
    pass

def query_aggregations(metric_name, time_window):
    """
    Query aggregated metrics for specified time window.
    
    Args:
        metric_name: Name of metric to query
        time_window: Time window for aggregation
        
    Returns:
        Aggregated results
    """
    # TODO: Implement fast aggregation queries
    pass

if __name__ == "__main__":
    # TODO: Add test/demo code
    print("Dynamic Data Pipeline - Implementation needed")
