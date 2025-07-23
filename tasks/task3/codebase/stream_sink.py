"""
Stream Sink - Data output destinations for the pipeline
"""
import time
import threading
import queue
from typing import Any, Dict, List
from abc import ABC, abstractmethod

class StreamSink(ABC):
    """Abstract base class for stream data sinks."""
    
    @abstractmethod
    def write(self, record: Any) -> bool:
        """Write a record to the sink. Returns True if successful."""
        pass
    
    @abstractmethod
    def flush(self) -> bool:
        """Flush any buffered data. Returns True if successful."""
        pass
    
    @abstractmethod
    def close(self) -> bool:
        """Close the sink and cleanup resources."""
        pass

class MemorySink(StreamSink):
    """In-memory sink for testing purposes."""
    
    def __init__(self, max_size: int = 100000):
        self.records = []
        self.max_size = max_size
        self.stats = {
            "writes": 0,
            "failures": 0,
            "bytes_written": 0
        }
        self._lock = threading.Lock()
    
    def write(self, record: Any) -> bool:
        """Write record to memory buffer."""
        try:
            with self._lock:
                if len(self.records) >= self.max_size:
                    # Remove oldest record to make room
                    self.records.pop(0)
                
                self.records.append(record)
                self.stats["writes"] += 1
                
                # Estimate bytes written
                if hasattr(record, '__sizeof__'):
                    self.stats["bytes_written"] += record.__sizeof__()
                else:
                    self.stats["bytes_written"] += len(str(record))
                
                return True
        except Exception:
            self.stats["failures"] += 1
            return False
    
    def flush(self) -> bool:
        """Memory sink doesn't need flushing."""
        return True
    
    def close(self) -> bool:
        """Clear memory buffer."""
        with self._lock:
            self.records.clear()
        return True
    
    def get_records(self) -> List[Any]:
        """Get all stored records."""
        with self._lock:
            return self.records.copy()
    
    def get_stats(self) -> Dict[str, int]:
        """Get sink statistics."""
        return self.stats.copy()

class FileSink(StreamSink):
    """File-based sink for persistent storage."""
    
    def __init__(self, filename: str, buffer_size: int = 1000):
        self.filename = filename
        self.buffer_size = buffer_size
        self.buffer = []
        self.file_handle = None
        self.stats = {
            "writes": 0,
            "failures": 0,
            "bytes_written": 0,
            "flushes": 0
        }
        self._lock = threading.Lock()
        
        try:
            self.file_handle = open(filename, 'w')
        except Exception:
            self.file_handle = None
    
    def write(self, record: Any) -> bool:
        """Write record to file buffer."""
        if not self.file_handle:
            self.stats["failures"] += 1
            return False
        
        try:
            with self._lock:
                record_str = str(record) + '\n'
                self.buffer.append(record_str)
                self.stats["writes"] += 1
                self.stats["bytes_written"] += len(record_str)
                
                # Auto-flush when buffer is full
                if len(self.buffer) >= self.buffer_size:
                    return self._flush_buffer()
                
                return True
        except Exception:
            self.stats["failures"] += 1
            return False
    
    def flush(self) -> bool:
        """Flush buffer to file."""
        with self._lock:
            return self._flush_buffer()
    
    def _flush_buffer(self) -> bool:
        """Internal method to flush buffer (assumes lock is held)."""
        if not self.file_handle or not self.buffer:
            return True
        
        try:
            self.file_handle.writelines(self.buffer)
            self.file_handle.flush()
            self.buffer.clear()
            self.stats["flushes"] += 1
            return True
        except Exception:
            self.stats["failures"] += 1
            return False
    
    def close(self) -> bool:
        """Close file handle after flushing."""
        success = self.flush()
        
        if self.file_handle:
            try:
                self.file_handle.close()
                self.file_handle = None
            except Exception:
                success = False
        
        return success
    
    def get_stats(self) -> Dict[str, int]:
        """Get sink statistics."""
        return self.stats.copy()

class AsyncBatchSink(StreamSink):
    """Asynchronous batching sink for high-throughput scenarios."""
    
    def __init__(self, target_sink: StreamSink, batch_size: int = 1000, 
                 flush_interval: float = 5.0):
        self.target_sink = target_sink
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch_queue = queue.Queue()
        self.current_batch = []
        self.stats = {
            "writes": 0,
            "batches_sent": 0,
            "failures": 0
        }
        self.is_running = False
        self._lock = threading.Lock()
        
        # Start background thread for batch processing
        self.worker_thread = threading.Thread(target=self._batch_worker)
        self.worker_thread.daemon = True
    
    def start(self):
        """Start the async batch processing."""
        if not self.is_running:
            self.is_running = True
            self.worker_thread.start()
    
    def write(self, record: Any) -> bool:
        """Add record to current batch."""
        try:
            with self._lock:
                self.current_batch.append(record)
                self.stats["writes"] += 1
                
                # Send batch if it's full
                if len(self.current_batch) >= self.batch_size:
                    batch = self.current_batch.copy()
                    self.current_batch.clear()
                    self.batch_queue.put(batch, timeout=1.0)
                
                return True
        except Exception:
            self.stats["failures"] += 1
            return False
    
    def _batch_worker(self):
        """Background worker to process batches."""
        last_flush = time.time()
        
        while self.is_running:
            try:
                # Check for time-based flush
                current_time = time.time()
                if (current_time - last_flush) >= self.flush_interval:
                    self._flush_current_batch()
                    last_flush = current_time
                
                # Process any queued batches
                try:
                    batch = self.batch_queue.get(timeout=0.1)
                    self._send_batch(batch)
                except queue.Empty:
                    continue
                    
            except Exception:
                self.stats["failures"] += 1
    
    def _flush_current_batch(self):
        """Flush the current batch if it has records."""
        with self._lock:
            if self.current_batch:
                batch = self.current_batch.copy()
                self.current_batch.clear()
                try:
                    self.batch_queue.put(batch, timeout=0.1)
                except queue.Full:
                    self.stats["failures"] += 1
    
    def _send_batch(self, batch: List[Any]):
        """Send a batch to the target sink."""
        try:
            for record in batch:
                self.target_sink.write(record)
            
            self.target_sink.flush()
            self.stats["batches_sent"] += 1
        except Exception:
            self.stats["failures"] += 1
    
    def flush(self) -> bool:
        """Flush current batch and target sink."""
        self._flush_current_batch()
        
        # Wait for queue to drain
        while not self.batch_queue.empty():
            time.sleep(0.01)
        
        return self.target_sink.flush()
    
    def close(self) -> bool:
        """Stop processing and close target sink."""
        self.is_running = False
        
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        self.flush()
        return self.target_sink.close()
    
    def get_stats(self) -> Dict[str, int]:
        """Get combined statistics."""
        stats = self.stats.copy()
        if hasattr(self.target_sink, 'get_stats'):
            target_stats = self.target_sink.get_stats()
            stats["target_writes"] = target_stats.get("writes", 0)
            stats["target_failures"] = target_stats.get("failures", 0)
        
        return stats

class MultiSink:
    """Writes to multiple sinks simultaneously."""
    
    def __init__(self):
        self.sinks = {}
        self.stats = {
            "writes": 0,
            "sink_failures": 0
        }
    
    def add_sink(self, name: str, sink: StreamSink):
        """Add a sink with a given name."""
        self.sinks[name] = sink
    
    def write(self, record: Any) -> int:
        """Write to all sinks, return number of successful writes."""
        successful_writes = 0
        
        for name, sink in self.sinks.items():
            try:
                if sink.write(record):
                    successful_writes += 1
                else:
                    self.stats["sink_failures"] += 1
            except Exception:
                self.stats["sink_failures"] += 1
        
        self.stats["writes"] += 1
        return successful_writes
    
    def flush_all(self) -> Dict[str, bool]:
        """Flush all sinks, return success status for each."""
        results = {}
        
        for name, sink in self.sinks.items():
            try:
                results[name] = sink.flush()
            except Exception:
                results[name] = False
        
        return results
    
    def close_all(self) -> Dict[str, bool]:
        """Close all sinks, return success status for each."""
        results = {}
        
        for name, sink in self.sinks.items():
            try:
                results[name] = sink.close()
            except Exception:
                results[name] = False
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        stats = self.stats.copy()
        stats["sink_stats"] = {}
        
        for name, sink in self.sinks.items():
            if hasattr(sink, 'get_stats'):
                stats["sink_stats"][name] = sink.get_stats()
        
        return stats
