"""
Cache Node - Individual node in the distributed cache cluster with advanced features
"""
import time
import threading
import hashlib
import json
import socket
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pickle
import zlib
from enum import Enum
import uuid
import statistics

class ConsistencyLevel(Enum):
    """Consistency levels for cache operations."""
    ONE = "one"          # At least 1 replica
    QUORUM = "quorum"    # Majority of replicas  
    ALL = "all"          # All replicas

class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    TTL = "ttl"          # Time To Live based
    RANDOM = "random"    # Random eviction

@dataclass
class CacheEntry:
    """Represents a single cache entry with comprehensive metadata."""
    value: Any
    created_at: float
    ttl: Optional[int] = None
    access_count: int = 0
    last_accessed: float = 0
    version: int = 1
    checksum: str = ""
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.last_accessed == 0:
            self.last_accessed = self.created_at
        if not self.checksum:
            self.checksum = self._calculate_checksum()
        if self.size_bytes == 0:
            self.size_bytes = self._estimate_size()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for data integrity."""
        try:
            data = pickle.dumps(self.value)
            return hashlib.md5(data).hexdigest()
        except:
            return hashlib.md5(str(self.value).encode()).hexdigest()
    
    def _estimate_size(self) -> int:
        """Estimate memory size of the entry."""
        try:
            return len(pickle.dumps(self.value))
        except:
            return len(str(self.value).encode('utf-8'))
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def validate_integrity(self) -> bool:
        """Validate data integrity using checksum."""
        current_checksum = self._calculate_checksum()
        return current_checksum == self.checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create CacheEntry from dictionary."""
        return cls(**data)

class ReplicationManager:
    """Manages data replication across cache nodes."""
    
    def __init__(self, node_id: str, replication_factor: int = 3):
        self.node_id = node_id
        self.replication_factor = replication_factor
        self.replica_nodes: Set[str] = set()
        self.ring_positions: Dict[str, int] = {}
        self.virtual_nodes = 100  # Virtual nodes per physical node
        self.ring_size = 2**32
        
    def add_node(self, node_id: str):
        """Add a node to the consistent hash ring."""
        self.replica_nodes.add(node_id)
        
        # Add virtual nodes to the ring
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_obj = hashlib.md5(virtual_key.encode())
            position = int(hash_obj.hexdigest(), 16) % self.ring_size
            self.ring_positions[virtual_key] = position
    
    def remove_node(self, node_id: str):
        """Remove a node from the hash ring."""
        self.replica_nodes.discard(node_id)
        
        # Remove virtual nodes
        keys_to_remove = [k for k in self.ring_positions.keys() if k.startswith(f"{node_id}:")]
        for key in keys_to_remove:
            del self.ring_positions[key]
    
    def get_replica_nodes(self, key: str) -> List[str]:
        """Get ordered list of replica nodes for a key."""
        if not self.replica_nodes:
            return []
        
        # Hash key to ring position
        hash_obj = hashlib.md5(key.encode())
        key_position = int(hash_obj.hexdigest(), 16) % self.ring_size
        
        # Find closest virtual nodes clockwise
        sorted_positions = sorted(self.ring_positions.items(), key=lambda x: x[1])
        
        # Find position in ring
        replicas = set()
        for virtual_key, position in sorted_positions:
            if position >= key_position:
                node_id = virtual_key.split(':')[0]
                replicas.add(node_id)
                if len(replicas) >= self.replication_factor:
                    break
        
        # Wrap around if needed
        if len(replicas) < self.replication_factor:
            for virtual_key, position in sorted_positions:
                node_id = virtual_key.split(':')[0]
                replicas.add(node_id)
                if len(replicas) >= self.replication_factor:
                    break
        
        return list(replicas)[:self.replication_factor]

class VectorClock:
    """Vector clock for conflict resolution in distributed cache."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.clock: Dict[str, int] = {node_id: 0}
        self.lock = threading.Lock()
    
    def increment(self):
        """Increment this node's logical clock."""
        with self.lock:
            self.clock[self.node_id] += 1
    
    def update(self, other_clock: Dict[str, int]):
        """Update clock with another vector clock."""
        with self.lock:
            for node, timestamp in other_clock.items():
                self.clock[node] = max(self.clock.get(node, 0), timestamp)
            self.increment()
    
    def compare(self, other_clock: Dict[str, int]) -> str:
        """Compare with another vector clock."""
        with self.lock:
            self_dominates = False
            other_dominates = False
            
            all_nodes = set(self.clock.keys()) | set(other_clock.keys())
            
            for node in all_nodes:
                self_time = self.clock.get(node, 0)
                other_time = other_clock.get(node, 0)
                
                if self_time > other_time:
                    self_dominates = True
                elif self_time < other_time:
                    other_dominates = True
            
            if self_dominates and not other_dominates:
                return "after"
            elif other_dominates and not self_dominates:
                return "before"
            elif not self_dominates and not other_dominates:
                return "equal"
            else:
                return "concurrent"
    
    def copy(self) -> Dict[str, int]:
        """Get a copy of the current clock."""
        with self.lock:
            return self.clock.copy()

class MemoryStorage:
    """Advanced in-memory storage backend with compression and optimization."""
    
    def __init__(self, max_memory_mb: int = 1024, compression_enabled: bool = True):
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.data: Dict[str, CacheEntry] = {}
        self.memory_usage = 0
        self.compression_enabled = compression_enabled
        self.compression_threshold = 1024  # Compress entries > 1KB
        
        # LRU tracking
        self.access_order = deque()  # For LRU eviction
        self.access_times: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            "compressed_entries": 0,
            "compression_ratio": 0.0,
            "memory_saved": 0
        }
        
        self.lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()
    
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Store an entry with optional compression."""
        try:
            # Compress large values if enabled
            if self.compression_enabled and entry.size_bytes > self.compression_threshold:
                entry = self._compress_entry(entry)
            
            with self.lock:
                # Remove existing entry if present
                if key in self.data:
                    old_entry = self.data[key]
                    self.memory_usage -= old_entry.size_bytes
                    self.access_order.remove(key)
                
                # Check if we need to evict
                while (self.memory_usage + entry.size_bytes > self.max_memory and 
                       len(self.data) > 0):
                    self._evict_lru()
                
                # Store new entry
                self.data[key] = entry
                self.memory_usage += entry.size_bytes
                self.access_order.append(key)
                self.access_times[key] = time.time()
                
                return True
                
        except Exception:
            return False
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve an entry and update access statistics."""
        with self.lock:
            if key not in self.data:
                return None
            
            entry = self.data[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                return None
            
            # Update access statistics
            entry.access()
            self.access_times[key] = time.time()
            
            # Move to end of LRU queue
            self.access_order.remove(key)
            self.access_order.append(key)
            
            # Decompress if needed
            if hasattr(entry, '_compressed') and entry._compressed:
                entry = self._decompress_entry(entry)
            
            return entry
    
    def delete(self, key: str) -> bool:
        """Delete an entry from storage."""
        with self.lock:
            if key in self.data:
                self._remove_entry(key)
                return True
            return False
    
    def _remove_entry(self, key: str):
        """Internal method to remove entry (assumes lock held)."""
        entry = self.data[key]
        del self.data[key]
        self.memory_usage -= entry.size_bytes
        self.access_order.remove(key)
        self.access_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_order:
            return
        
        lru_key = self.access_order.popleft()
        entry = self.data[lru_key]
        del self.data[lru_key]
        self.memory_usage -= entry.size_bytes
        self.access_times.pop(lru_key, None)
    
    def _compress_entry(self, entry: CacheEntry) -> CacheEntry:
        """Compress entry value if beneficial."""
        try:
            original_data = pickle.dumps(entry.value)
            compressed_data = zlib.compress(original_data)
            
            if len(compressed_data) < len(original_data):
                # Compression beneficial
                entry.value = compressed_data
                entry._compressed = True
                entry._original_size = len(original_data)
                entry.size_bytes = len(compressed_data)
                
                # Update stats
                self.stats["compressed_entries"] += 1
                self.stats["memory_saved"] += len(original_data) - len(compressed_data)
                
                if self.stats["compressed_entries"] > 0:
                    self.stats["compression_ratio"] = (
                        self.stats["memory_saved"] / 
                        (self.stats["memory_saved"] + sum(e.size_bytes for e in self.data.values()))
                    )
            
            return entry
            
        except Exception:
            return entry  # Return original if compression fails
    
    def _decompress_entry(self, entry: CacheEntry) -> CacheEntry:
        """Decompress entry value."""
        try:
            if hasattr(entry, '_compressed') and entry._compressed:
                compressed_data = entry.value
                original_data = zlib.decompress(compressed_data)
                entry.value = pickle.loads(original_data)
                entry._compressed = False
                entry.size_bytes = getattr(entry, '_original_size', len(original_data))
            
            return entry
            
        except Exception:
            return entry  # Return as-is if decompression fails
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self.lock:
            return {
                "used_memory": self.memory_usage,
                "max_memory": self.max_memory,
                "memory_utilization": self.memory_usage / self.max_memory,
                "entry_count": len(self.data),
                "avg_entry_size": self.memory_usage / len(self.data) if self.data else 0,
                "compression_stats": self.stats.copy()
            }
        self._lock = threading.RWLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry by key."""
        with self._lock.reader():
            entry = self.data.get(key)
            if entry and not entry.is_expired():
                entry.access()
                return entry
            elif entry and entry.is_expired():
                # Remove expired entry
                self._remove_entry(key, entry)
        return None
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Store entry with key."""
        try:
            # Estimate memory usage
            entry_size = self._estimate_size(key, entry)
            
            with self._lock.writer():
                # Check if key already exists
                if key in self.data:
                    old_entry = self.data[key]
                    old_size = self._estimate_size(key, old_entry)
                    self.memory_usage -= old_size
                
                # Check memory limits
                if self.memory_usage + entry_size > self.max_memory:
                    if not self._evict_entries(entry_size):
                        return False  # Couldn't free enough memory
                
                self.data[key] = entry
                self.memory_usage += entry_size
                return True
                
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete entry by key."""
        with self._lock.writer():
            if key in self.data:
                entry = self.data[key]
                self._remove_entry(key, entry)
                return True
        return False
    
    def _remove_entry(self, key: str, entry: CacheEntry):
        """Remove entry and update memory usage."""
        del self.data[key]
        entry_size = self._estimate_size(key, entry)
        self.memory_usage -= entry_size
    
    def _estimate_size(self, key: str, entry: CacheEntry) -> int:
        """Estimate memory usage of key-value pair."""
        try:
            # Rough estimation using pickle
            serialized = pickle.dumps((key, entry))
            return len(serialized)
        except Exception:
            # Fallback estimation
            return len(key) * 2 + len(str(entry.value)) * 2 + 100
    
    def _evict_entries(self, needed_space: int) -> bool:
        """Evict entries using LRU policy to free space."""
        # Sort by last accessed time (LRU)
        entries_by_access = sorted(
            self.data.items(),
            key=lambda x: x[1].last_accessed
        )
        
        freed_space = 0
        entries_to_remove = []
        
        for key, entry in entries_by_access:
            entry_size = self._estimate_size(key, entry)
            entries_to_remove.append((key, entry, entry_size))
            freed_space += entry_size
            
            if freed_space >= needed_space:
                break
        
        # Remove selected entries
        for key, entry, size in entries_to_remove:
            self._remove_entry(key, entry)
        
        return freed_space >= needed_space
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock.reader():
            expired_count = sum(1 for entry in self.data.values() if entry.is_expired())
            
            return {
                "total_entries": len(self.data),
                "memory_usage_bytes": self.memory_usage,
                "memory_usage_mb": self.memory_usage / (1024 * 1024),
                "memory_limit_mb": self.max_memory / (1024 * 1024),
                "memory_utilization": self.memory_usage / self.max_memory,
                "expired_entries": expired_count
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        removed_count = 0
        current_time = time.time()
        
        with self._lock.writer():
            expired_keys = [
                key for key, entry in self.data.items()
                if entry.ttl and (current_time > entry.created_at + entry.ttl)
            ]
            
            for key in expired_keys:
                entry = self.data[key]
                self._remove_entry(key, entry)
                removed_count += 1
        
        return removed_count

class CacheNode:
    """Individual cache node in distributed system."""
    
    def __init__(self, node_id: str, address: str, port: int = 8000, 
                 max_memory_mb: int = 1024):
        self.node_id = node_id
        self.address = address
        self.port = port
        self.storage = MemoryStorage(max_memory_mb)
        
        # Node state
        self.is_healthy = True
        self.peers: Set[str] = set()
        self.replication_factor = 3
        
        # Statistics
        self.stats = {
            "requests_handled": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "replications_sent": 0,
            "replications_received": 0,
            "start_time": time.time()
        }
        
        # Threading
        self._lock = threading.Lock()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # Cleanup expired entries every 60 seconds
        cleanup_thread = threading.Thread(target=self._cleanup_loop)
        cleanup_thread.daemon = True
        cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background task to cleanup expired entries."""
        while self.is_healthy:
            try:
                removed = self.storage.cleanup_expired()
                if removed > 0:
                    print(f"Node {self.node_id}: Cleaned up {removed} expired entries")
                time.sleep(60)  # Run every minute
            except Exception as e:
                print(f"Node {self.node_id}: Cleanup error: {e}")
                time.sleep(60)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        with self._lock:
            self.stats["requests_handled"] += 1
            
            entry = self.storage.get(key)
            if entry:
                self.stats["cache_hits"] += 1
                return entry.value
            else:
                self.stats["cache_misses"] += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            replicate: bool = True) -> bool:
        """Set key-value pair with optional TTL."""
        try:
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl
            )
            
            with self._lock:
                self.stats["requests_handled"] += 1
                success = self.storage.set(key, entry)
                
                if success and replicate:
                    # Replicate to peers asynchronously
                    self._replicate_async(key, value, ttl)
                
                return success
                
        except Exception:
            return False
    
    def delete(self, key: str, replicate: bool = True) -> bool:
        """Delete key from cache."""
        with self._lock:
            self.stats["requests_handled"] += 1
            success = self.storage.delete(key)
            
            if success and replicate:
                # Replicate deletion to peers
                self._replicate_delete_async(key)
            
            return success
    
    def _replicate_async(self, key: str, value: Any, ttl: Optional[int]):
        """Asynchronously replicate data to peer nodes."""
        def replicate_worker():
            try:
                # Simple replication - in real system would use proper consensus
                for peer in list(self.peers):
                    try:
                        # Simulate network call to peer
                        # In real implementation, this would be HTTP/gRPC call
                        self.stats["replications_sent"] += 1
                    except Exception:
                        pass  # Handle peer failure
            except Exception:
                pass
        
        thread = threading.Thread(target=replicate_worker)
        thread.daemon = True
        thread.start()
    
    def _replicate_delete_async(self, key: str):
        """Asynchronously replicate deletion to peers."""
        def delete_worker():
            try:
                for peer in list(self.peers):
                    try:
                        # Simulate deletion replication
                        self.stats["replications_sent"] += 1
                    except Exception:
                        pass
            except Exception:
                pass
        
        thread = threading.Thread(target=delete_worker)
        thread.daemon = True
        thread.start()
    
    def add_peer(self, peer_address: str):
        """Add peer node to replication set."""
        with self._lock:
            self.peers.add(peer_address)
    
    def remove_peer(self, peer_address: str):
        """Remove peer node from replication set."""
        with self._lock:
            self.peers.discard(peer_address)
    
    def health_check(self) -> Dict[str, Any]:
        """Return node health and statistics."""
        storage_stats = self.storage.get_stats()
        uptime = time.time() - self.stats["start_time"]
        
        return {
            "node_id": self.node_id,
            "address": f"{self.address}:{self.port}",
            "is_healthy": self.is_healthy,
            "uptime_seconds": uptime,
            "peers_count": len(self.peers),
            "requests_handled": self.stats["requests_handled"],
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(1, self.stats["requests_handled"])
            ),
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "replications_sent": self.stats["replications_sent"],
            "replications_received": self.stats["replications_received"],
            "storage": storage_stats
        }
    
    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys in single operation."""
        results = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                results[key] = value
        return results
    
    def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, bool]:
        """Set multiple key-value pairs."""
        results = {}
        for key, value in items.items():
            results[key] = self.set(key, value, ttl)
        return results
    
    def get_keys_by_pattern(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern (simplified implementation)."""
        # Simple pattern matching - in production would use proper regex
        with self._lock:
            if pattern == "*":
                return list(self.storage.data.keys())
            else:
                # Basic prefix matching
                if pattern.endswith("*"):
                    prefix = pattern[:-1]
                    return [key for key in self.storage.data.keys() if key.startswith(prefix)]
                else:
                    return [key for key in self.storage.data.keys() if key == pattern]
    
    def shutdown(self):
        """Gracefully shutdown the node."""
        self.is_healthy = False
        print(f"Node {self.node_id}: Shutting down gracefully")

# Helper class for RWLock (since Python doesn't have built-in RWLock)
class RWLock:
    """Reader-Writer lock implementation."""
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
    
    def reader(self):
        return self._ReaderContext(self)
    
    def writer(self):
        return self._WriterContext(self)
    
    class _ReaderContext:
        def __init__(self, lock):
            self._lock = lock
        
        def __enter__(self):
            self._lock._read_ready.acquire()
            try:
                self._lock._readers += 1
            finally:
                self._lock._read_ready.release()
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self._lock._read_ready.acquire()
            try:
                self._lock._readers -= 1
                if self._lock._readers == 0:
                    self._lock._read_ready.notifyAll()
            finally:
                self._lock._read_ready.release()
    
    class _WriterContext:
        def __init__(self, lock):
            self._lock = lock
        
        def __enter__(self):
            self._lock._read_ready.acquire()
            while self._lock._readers > 0:
                self._lock._read_ready.wait()
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self._lock._read_ready.release()

# Monkey patch the RWLock into threading module
threading.RWLock = RWLock


class CacheNode:
    """Main distributed cache node with replication and consistency."""
    
    def __init__(self, node_id: str, max_memory_mb: int = 1024, 
                 replication_factor: int = 3):
        self.node_id = node_id
        self.storage = MemoryStorage(max_memory_mb)
        self.replication_manager = ReplicationManager(node_id, replication_factor)
        self.vector_clock = VectorClock(node_id)
        
        # Configuration
        self.default_ttl = 3600  # 1 hour default TTL
        self.eviction_policy = EvictionPolicy.LRU
        
        # Statistics
        self.stats = {
            "requests_total": 0,
            "requests_get": 0,
            "requests_put": 0,
            "requests_delete": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "replication_successes": 0,
            "replication_failures": 0,
            "consistency_violations": 0,
            "network_timeouts": 0
        }
        
        # Background tasks
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        # Network simulation (in real implementation, would be actual network layer)
        self._network_reliability = 0.95  # 95% success rate for network operations
        
    def get(self, key: str, consistency_level: ConsistencyLevel = ConsistencyLevel.ONE) -> Optional[Any]:
        """Get value from cache with specified consistency level."""
        self.stats["requests_total"] += 1
        self.stats["requests_get"] += 1
        
        try:
            # Try local storage first
            entry = self.storage.get(key)
            
            if entry is not None:
                # Validate data integrity
                if not entry.validate_integrity():
                    self.stats["consistency_violations"] += 1
                    self.storage.delete(key)
                    entry = None
                else:
                    self.stats["cache_hits"] += 1
                    return entry.value
            
            # For higher consistency levels, check replicas
            if consistency_level != ConsistencyLevel.ONE:
                replica_value = self._get_from_replicas(key, consistency_level)
                if replica_value is not None:
                    self.stats["cache_hits"] += 1
                    return replica_value
            
            self.stats["cache_misses"] += 1
            return None
            
        except Exception as e:
            self.stats["requests_total"] -= 1  # Don't count failed requests
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None,
            consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM) -> bool:
        """Put value in cache with replication."""
        self.stats["requests_total"] += 1
        self.stats["requests_put"] += 1
        
        try:
            # Create cache entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl or self.default_ttl,
                version=self._get_next_version(key)
            )
            
            # Store locally
            local_success = self.storage.put(key, entry)
            if not local_success:
                return False
            
            # Update vector clock
            self.vector_clock.increment()
            
            # Replicate to other nodes
            if consistency_level != ConsistencyLevel.ONE:
                replication_success = self._replicate_to_nodes(key, entry, consistency_level)
                if replication_success:
                    self.stats["replication_successes"] += 1
                else:
                    self.stats["replication_failures"] += 1
                    
                return replication_success
            
            self.stats["replication_successes"] += 1
            return True
            
        except Exception as e:
            self.stats["requests_total"] -= 1
            return False
    
    def delete(self, key: str, consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM) -> bool:
        """Delete key from cache and replicas."""
        self.stats["requests_total"] += 1
        self.stats["requests_delete"] += 1
        
        try:
            # Delete locally
            local_success = self.storage.delete(key)
            
            # Update vector clock
            self.vector_clock.increment()
            
            # Delete from replicas
            if consistency_level != ConsistencyLevel.ONE:
                replica_success = self._delete_from_replicas(key, consistency_level)
                return local_success or replica_success
            
            return local_success
            
        except Exception as e:
            self.stats["requests_total"] -= 1
            return False
    
    def _get_next_version(self, key: str) -> int:
        """Get next version number for a key."""
        entry = self.storage.get(key)
        if entry:
            return entry.version + 1
        return 1
    
    def _get_from_replicas(self, key: str, consistency_level: ConsistencyLevel) -> Optional[Any]:
        """Get value from replica nodes."""
        replica_nodes = self.replication_manager.get_replica_nodes(key)
        required_responses = self._get_required_responses(len(replica_nodes), consistency_level)
        
        successful_reads = 0
        latest_entry = None
        latest_version = 0
        
        for node_id in replica_nodes:
            if node_id == self.node_id:
                continue
                
            try:
                # Simulate network call to replica
                entry = self._simulate_replica_read(node_id, key)
                
                if entry and entry.version > latest_version:
                    latest_entry = entry
                    latest_version = entry.version
                
                successful_reads += 1
                
                if successful_reads >= required_responses:
                    break
                    
            except Exception:
                self.stats["network_timeouts"] += 1
                continue
        
        if successful_reads >= required_responses and latest_entry:
            # Store locally for future reads
            self.storage.put(key, latest_entry)
            return latest_entry.value
        
        return None
    
    def _replicate_to_nodes(self, key: str, entry: CacheEntry, 
                           consistency_level: ConsistencyLevel) -> bool:
        """Replicate entry to other nodes."""
        replica_nodes = self.replication_manager.get_replica_nodes(key)
        required_responses = self._get_required_responses(len(replica_nodes), consistency_level)
        
        successful_writes = 1  # Local write already succeeded
        
        for node_id in replica_nodes:
            if node_id == self.node_id:
                continue
                
            try:
                # Simulate network call to replica
                success = self._simulate_replica_write(node_id, key, entry)
                
                if success:
                    successful_writes += 1
                    
                if successful_writes >= required_responses:
                    break
                    
            except Exception:
                self.stats["network_timeouts"] += 1
                continue
        
        return successful_writes >= required_responses
    
    def _delete_from_replicas(self, key: str, consistency_level: ConsistencyLevel) -> bool:
        """Delete key from replica nodes."""
        replica_nodes = self.replication_manager.get_replica_nodes(key)
        required_responses = self._get_required_responses(len(replica_nodes), consistency_level)
        
        successful_deletes = 0
        
        for node_id in replica_nodes:
            if node_id == self.node_id:
                continue
                
            try:
                # Simulate network call to replica
                success = self._simulate_replica_delete(node_id, key)
                
                if success:
                    successful_deletes += 1
                    
                if successful_deletes >= required_responses:
                    break
                    
            except Exception:
                self.stats["network_timeouts"] += 1
                continue
        
        return successful_deletes >= required_responses
    
    def _get_required_responses(self, total_nodes: int, consistency_level: ConsistencyLevel) -> int:
        """Calculate required responses for consistency level."""
        if consistency_level == ConsistencyLevel.ONE:
            return 1
        elif consistency_level == ConsistencyLevel.ALL:
            return total_nodes
        else:  # QUORUM
            return (total_nodes // 2) + 1
    
    def _simulate_replica_read(self, node_id: str, key: str) -> Optional[CacheEntry]:
        """Simulate reading from a replica node."""
        import random
        
        # Simulate network reliability
        if random.random() > self._network_reliability:
            raise Exception("Network timeout")
        
        # Simulate data presence (80% chance if key exists locally)
        local_entry = self.storage.get(key)
        if local_entry and random.random() < 0.8:
            # Return slightly older version with some probability
            version_offset = random.choice([0, 0, 0, -1, -2])  # Mostly current version
            return CacheEntry(
                value=local_entry.value,
                created_at=local_entry.created_at - random.uniform(0, 300),
                ttl=local_entry.ttl,
                version=max(1, local_entry.version + version_offset)
            )
        
        return None
    
    def _simulate_replica_write(self, node_id: str, key: str, entry: CacheEntry) -> bool:
        """Simulate writing to a replica node."""
        import random
        
        # Simulate network reliability
        if random.random() > self._network_reliability:
            raise Exception("Network timeout")
        
        # Simulate write success (95% success rate)
        return random.random() < 0.95
    
    def _simulate_replica_delete(self, node_id: str, key: str) -> bool:
        """Simulate deleting from a replica node."""
        import random
        
        # Simulate network reliability
        if random.random() > self._network_reliability:
            raise Exception("Network timeout")
        
        # Simulate delete success (95% success rate)
        return random.random() < 0.95
    
    def _background_cleanup(self):
        """Background thread for cleanup tasks."""
        while True:
            try:
                time.sleep(60)  # Run every minute
                
                # Clean up expired entries
                self._cleanup_expired_entries()
                
                # Trigger garbage collection if memory usage is high
                memory_stats = self.storage.get_memory_stats()
                if memory_stats["memory_utilization"] > 0.9:
                    self._trigger_eviction()
                    
            except Exception:
                continue
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from storage."""
        # This would be implemented in the storage layer
        pass
    
    def _trigger_eviction(self):
        """Trigger cache eviction based on policy."""
        # This would be implemented based on eviction policy
        pass
    
    def add_replica_node(self, node_id: str):
        """Add a replica node to the cluster."""
        self.replication_manager.add_node(node_id)
    
    def remove_replica_node(self, node_id: str):
        """Remove a replica node from the cluster."""
        self.replication_manager.remove_node(node_id)
    
    def get_node_stats(self) -> Dict[str, Any]:
        """Get comprehensive node statistics."""
        memory_stats = self.storage.get_memory_stats()
        
        # Calculate derived metrics
        total_requests = self.stats["requests_total"]
        hit_rate = 0
        if total_requests > 0:
            hit_rate = self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
        
        replication_success_rate = 0
        total_replications = self.stats["replication_successes"] + self.stats["replication_failures"]
        if total_replications > 0:
            replication_success_rate = self.stats["replication_successes"] / total_replications
        
        return {
            "node_id": self.node_id,
            "memory_stats": memory_stats,
            "cache_hit_rate": hit_rate,
            "replication_success_rate": replication_success_rate,
            "requests": self.stats.copy(),
            "replica_nodes": list(self.replication_manager.replica_nodes),
            "vector_clock": self.vector_clock.copy()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        memory_stats = self.storage.get_memory_stats()
        
        status = "healthy"
        issues = []
        
        # Check memory usage
        if memory_stats["memory_utilization"] > 0.95:
            status = "warning"
            issues.append("High memory usage")
        
        # Check replication health
        if self.stats["replication_failures"] > self.stats["replication_successes"]:
            status = "critical"
            issues.append("Replication failures exceed successes")
        
        # Check consistency violations
        if self.stats["consistency_violations"] > 10:
            status = "warning"
            issues.append("Multiple consistency violations detected")
        
        return {
            "status": status,
            "issues": issues,
            "uptime": time.time(),  # Simplified uptime
            "memory_utilization": memory_stats["memory_utilization"],
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
        }