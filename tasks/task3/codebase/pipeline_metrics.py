"""
Pipeline Metrics - Monitoring and observability for the data pipeline
"""
import time
import threading
import statistics
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import psutil
import os

class PipelineMetrics:
    """Comprehensive metrics collection and reporting for data pipelines."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.start_time = time.time()
        
        # Core metrics
        self.records_processed = 0
        self.records_failed = 0
        self.bytes_processed = 0
        
        # Latency tracking
        self.latencies = deque(maxlen=window_size)
        
        # Throughput tracking
        self.throughput_samples = deque(maxlen=100)  # Last 100 samples
        self.last_sample_time = time.time()
        self.last_record_count = 0
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=window_size)
        
        # Resource tracking
        self.cpu_samples = deque(maxlen=60)  # 1 minute of samples
        self.memory_samples = deque(maxlen=60)
        
        # Alert thresholds
        self.thresholds = {}
        self.active_alerts = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start background monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._background_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def record_processing(self, latency: float, success: bool = True, 
                         bytes_size: int = 0, error_type: str = None):
        """Record processing metrics for a single operation."""
        with self._lock:
            if success:
                self.records_processed += 1
                self.latencies.append(latency)
                self.bytes_processed += bytes_size
            else:
                self.records_failed += 1
                if error_type:
                    self.error_counts[error_type] += 1
                    self.error_history.append({
                        "timestamp": time.time(),
                        "error_type": error_type,
                        "latency": latency
                    })
    
    def record_batch_processing(self, batch_size: int, total_latency: float, 
                              successful_records: int, failed_records: int = 0):
        """Record metrics for batch processing."""
        with self._lock:
            self.records_processed += successful_records
            self.records_failed += failed_records
            
            # Record average latency per record
            if batch_size > 0:
                avg_latency = total_latency / batch_size
                for _ in range(successful_records):
                    self.latencies.append(avg_latency)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            current_time = time.time()
            runtime = current_time - self.start_time
            
            # Calculate throughput
            total_records = self.records_processed + self.records_failed
            overall_throughput = total_records / runtime if runtime > 0 else 0
            
            # Calculate current throughput (last sample)
            current_throughput = 0
            if self.throughput_samples:
                current_throughput = self.throughput_samples[-1]
            
            # Calculate error rate
            error_rate = 0
            if total_records > 0:
                error_rate = self.records_failed / total_records
            
            # Calculate latency stats
            latency_stats = self._calculate_latency_stats()
            
            return {
                "timestamp": current_time,
                "runtime_seconds": runtime,
                "records_processed": self.records_processed,
                "records_failed": self.records_failed,
                "bytes_processed": self.bytes_processed,
                "throughput": overall_throughput,
                "current_throughput": current_throughput,
                "error_rate": error_rate,
                "latency_avg": latency_stats["mean"],
                "latency_p50": latency_stats["p50"],
                "latency_p95": latency_stats["p95"],
                "latency_p99": latency_stats["p99"],
                "top_errors": dict(list(self.error_counts.items())[:5])
            }
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get detailed latency percentile breakdown."""
        with self._lock:
            return self._calculate_latency_stats()
    
    def _calculate_latency_stats(self) -> Dict[str, float]:
        """Calculate latency statistics from current samples."""
        if not self.latencies:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        
        latency_list = list(self.latencies)
        latency_list.sort()
        
        return {
            "mean": statistics.mean(latency_list),
            "p50": self._percentile(latency_list, 50),
            "p95": self._percentile(latency_list, 95),
            "p99": self._percentile(latency_list, 99),
            "min": min(latency_list),
            "max": max(latency_list)
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value from sorted data."""
        if not data:
            return 0.0
        
        index = (len(data) - 1) * (percentile / 100.0)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(data) - 1)
        
        if lower_index == upper_index:
            return data[lower_index]
        
        # Linear interpolation
        weight = index - lower_index
        return data[lower_index] * (1 - weight) + data[upper_index] * weight
    
    def get_resource_metrics(self) -> Dict[str, float]:
        """Get current resource utilization metrics."""
        try:
            process = psutil.Process(os.getpid())
            
            # CPU usage (as percentage)
            cpu_percent = process.cpu_percent()
            
            # Memory usage (in MB)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Memory percentage of available system memory
            memory_percent = process.memory_percent()
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory_mb,
                "memory_percent": memory_percent,
                "threads": process.num_threads(),
                "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0
            }
        except Exception:
            return {
                "cpu_usage": 0,
                "memory_usage": 0,
                "memory_percent": 0,
                "threads": 0,
                "open_files": 0
            }
    
    def _background_monitor(self):
        """Background thread for continuous monitoring."""
        while self.monitoring:
            try:
                # Sample throughput
                current_time = time.time()
                time_delta = current_time - self.last_sample_time
                
                if time_delta >= 1.0:  # Sample every second
                    with self._lock:
                        record_delta = (self.records_processed + self.records_failed) - self.last_record_count
                        current_throughput = record_delta / time_delta if time_delta > 0 else 0
                        
                        self.throughput_samples.append(current_throughput)
                        self.last_sample_time = current_time
                        self.last_record_count = self.records_processed + self.records_failed
                
                # Sample resource usage
                resource_metrics = self.get_resource_metrics()
                self.cpu_samples.append(resource_metrics["cpu_usage"])
                self.memory_samples.append(resource_metrics["memory_usage"])
                
                # Check alert thresholds
                self._check_alerts()
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception:
                # Continue monitoring even if individual samples fail
                time.sleep(1.0)
    
    def set_threshold(self, metric_name: str, threshold_value: float, 
                     comparison: str = "greater"):
        """Set alert threshold for a metric."""
        self.thresholds[metric_name] = {
            "value": threshold_value,
            "comparison": comparison  # "greater", "less", "equal"
        }
    
    def _check_alerts(self):
        """Check if any metrics exceed their thresholds."""
        if not self.thresholds:
            return
        
        current_metrics = self.get_current_metrics()
        new_alerts = []
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in current_metrics:
                metric_value = current_metrics[metric_name]
                threshold_value = threshold["value"]
                comparison = threshold["comparison"]
                
                alert_triggered = False
                
                if comparison == "greater" and metric_value > threshold_value:
                    alert_triggered = True
                elif comparison == "less" and metric_value < threshold_value:
                    alert_triggered = True
                elif comparison == "equal" and abs(metric_value - threshold_value) < 0.001:
                    alert_triggered = True
                
                if alert_triggered:
                    alert = {
                        "timestamp": time.time(),
                        "metric": metric_name,
                        "value": metric_value,
                        "threshold": threshold_value,
                        "comparison": comparison
                    }
                    new_alerts.append(alert)
        
        # Update active alerts (keep only recent ones)
        current_time = time.time()
        self.active_alerts = [
            alert for alert in self.active_alerts 
            if current_time - alert["timestamp"] < 300  # Keep alerts for 5 minutes
        ]
        
        self.active_alerts.extend(new_alerts)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        with self._lock:
            return self.active_alerts.copy()
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get detailed error analysis."""
        with self._lock:
            recent_errors = [
                error for error in self.error_history
                if time.time() - error["timestamp"] < 3600  # Last hour
            ]
            
            error_rate_trend = []
            if len(recent_errors) >= 10:
                # Calculate error rate trend over time
                sorted_errors = sorted(recent_errors, key=lambda x: x["timestamp"])
                window_size = max(1, len(sorted_errors) // 10)
                
                for i in range(0, len(sorted_errors), window_size):
                    window = sorted_errors[i:i+window_size]
                    if window:
                        window_start = window[0]["timestamp"]
                        window_end = window[-1]["timestamp"]
                        window_duration = max(1, window_end - window_start)
                        error_rate = len(window) / window_duration
                        
                        error_rate_trend.append({
                            "timestamp": window_start,
                            "error_rate": error_rate,
                            "error_count": len(window)
                        })
            
            return {
                "total_errors": self.records_failed,
                "error_types": dict(self.error_counts),
                "recent_errors": len(recent_errors),
                "error_rate_trend": error_rate_trend
            }
    
    def reset_metrics(self):
        """Reset all metrics to initial state."""
        with self._lock:
            self.records_processed = 0
            self.records_failed = 0
            self.bytes_processed = 0
            self.latencies.clear()
            self.throughput_samples.clear()
            self.error_counts.clear()
            self.error_history.clear()
            self.active_alerts.clear()
            self.start_time = time.time()
            self.last_sample_time = time.time()
            self.last_record_count = 0
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
    
    def export_metrics(self, format: str = "json") -> Any:
        """Export metrics in specified format."""
        all_metrics = {
            "current": self.get_current_metrics(),
            "latency": self.get_latency_percentiles(),
            "resources": self.get_resource_metrics(),
            "errors": self.get_error_analysis(),
            "alerts": self.get_active_alerts()
        }
        
        if format == "json":
            import json
            return json.dumps(all_metrics, indent=2)
        else:
            return all_metrics
