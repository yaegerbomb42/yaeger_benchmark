"""
Distributed Cache System Implementation

Implement a fault-tolerant distributed caching system with strong consistency.
Your solution will be assessed on:
1. Correctness (distributed operations, consistency)
2. Performance (throughput, latency)
3. Fault tolerance (node failures, split-brain scenarios)
4. Memory efficiency and scalability

API Usage:
- DistributedCache: Main cache interface
- CacheNode: Individual cache node
- ConsistentHashRing: Data distribution

Performance Requirements:
- Handle 100,000+ operations per second
- <10ms latency for cache hits
- Support adding/removing nodes dynamically
- Maintain consistency during node failures
"""

from codebase.cache_node import CacheNode, MemoryStorage, CacheEntry
from codebase.distributed_cache import DistributedCache as CoreDistributedCache, ConsistentHashRing
import time
import threading
from typing import List, Dict, Any, Optional, Set
import random

class DistributedCache:
    """Main distributed cache system coordinating multiple nodes."""
    
    def __init__(self, nodes: List[str], replication_factor: int = 3):
        self.replication_factor = replication_factor
        self.hash_ring = ConsistentHashRing()
        
        # Create actual cache nodes
        self.cache_nodes: Dict[str, CacheNode] = {}
        for node_addr in nodes:
            node_id = node_addr.split(':')[0]
            host = node_addr.split(':')[0] 
            port = int(node_addr.split(':')[1]) if ':' in node_addr else 8000
            
            cache_node = CacheNode(node_id, host, port)
            self.cache_nodes[node_addr] = cache_node
            self.hash_ring.add_node(node_addr)
        
        # Add peers to each node for replication
        for node_addr, node in self.cache_nodes.items():
            other_nodes = [addr for addr in nodes if addr != node_addr]
            for peer in other_nodes:
                node.add_peer(peer)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "start_time": time.time()
        }
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key from the cache."""
        with self._lock:
            self.stats["total_requests"] += 1
        
        start_time = time.time()
        
        try:
            # Route request to appropriate nodes using consistent hashing
            target_nodes = self.hash_ring.get_nodes(key, self.replication_factor)
            
            for node_addr in target_nodes:
                if node_addr in self.cache_nodes:
                    node = self.cache_nodes[node_addr]
                    if node.is_healthy:
                        value = node.get(key)
                        
                        if value is not None:
                            with self._lock:
                                self.stats["successful_requests"] += 1
                            return value
            
            # No value found in any node
            with self._lock:
                self.stats["successful_requests"] += 1  # Still a successful operation
            
            return None
            
        except Exception as e:
            with self._lock:
                self.stats["failed_requests"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set key-value pair with optional TTL."""
        with self._lock:
            self.stats["total_requests"] += 1
        
        try:
            # Route to replica nodes for writes
            target_nodes = self.hash_ring.get_nodes(key, self.replication_factor)
            
            successful_writes = 0
            required_writes = max(1, len(target_nodes) // 2 + 1)  # Majority
            
            for node_addr in target_nodes:
                if node_addr in self.cache_nodes:
                    node = self.cache_nodes[node_addr]
                    if node.is_healthy:
                        # Write to node (without triggering replication to avoid loops)
                        if node.set(key, value, ttl, replicate=False):
                            successful_writes += 1
            
            success = successful_writes >= required_writes
            
            with self._lock:
                if success:
                    self.stats["successful_requests"] += 1
                else:
                    self.stats["failed_requests"] += 1
            
            return success
            
        except Exception as e:
            with self._lock:
                self.stats["failed_requests"] += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            self.stats["total_requests"] += 1
        
        try:
            # Route to replica nodes for deletion
            target_nodes = self.hash_ring.get_nodes(key, self.replication_factor)
            
            successful_deletes = 0
            
            for node_addr in target_nodes:
                if node_addr in self.cache_nodes:
                    node = self.cache_nodes[node_addr]
                    if node.is_healthy:
                        if node.delete(key, replicate=False):
                            successful_deletes += 1
            
            success = successful_deletes > 0
            
            with self._lock:
                if success:
                    self.stats["successful_requests"] += 1
                else:
                    self.stats["failed_requests"] += 1
            
            return success
            
        except Exception as e:
            with self._lock:
                self.stats["failed_requests"] += 1
            return False
    
    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys in single operation."""
        results = {}
        
        for key in keys:
            value = self.get(key)
            if value is not None:
                results[key] = value
        
        return results
    
    def batch_set(self, items: Dict[str, Any], ttl: int = None) -> bool:
        """Set multiple key-value pairs."""
        successful_sets = 0
        
        for key, value in items.items():
            if self.set(key, value, ttl):
                successful_sets += 1
        
        # Return True if majority of sets succeeded
        return successful_sets >= len(items) // 2 + 1
    
    def add_node(self, node_address: str) -> bool:
        """Add new node to cluster."""
        try:
            # Create cache node
            node_id = node_address.split(':')[0]
            host = node_address.split(':')[0]
            port = int(node_address.split(':')[1]) if ':' in node_address else 8000
            
            new_node = CacheNode(node_id, host, port)
            self.cache_nodes[node_address] = new_node
            self.hash_ring.add_node(node_address)
            
            # Add existing nodes as peers
            for existing_addr in self.cache_nodes:
                if existing_addr != node_address:
                    new_node.add_peer(existing_addr)
                    self.cache_nodes[existing_addr].add_peer(node_address)
            
            return True
            
        except Exception:
            return False
    
    def remove_node(self, node_address: str) -> bool:
        """Remove node from cluster."""
        try:
            # Shutdown and remove cache node
            if node_address in self.cache_nodes:
                node = self.cache_nodes[node_address]
                node.shutdown()
                del self.cache_nodes[node_address]
                self.hash_ring.remove_node(node_address)
                
                # Remove from other nodes' peer lists
                for other_node in self.cache_nodes.values():
                    other_node.remove_peer(node_address)
            
            return True
            
        except Exception:
            return False
    
    def simulate_node_failure(self, node_address: str):
        """Simulate node failure for testing."""
        if node_address in self.cache_nodes:
            # Simulate node becoming unresponsive
            node = self.cache_nodes[node_address]
            node.is_healthy = False
    
    def recover_node(self, node_address: str):
        """Recover a failed node."""
        if node_address in self.cache_nodes:
            # Restore node health
            node = self.cache_nodes[node_address]
            node.is_healthy = True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Aggregate stats from all nodes
        total_cache_hits = 0
        total_cache_misses = 0
        total_memory_usage = 0
        node_health = {}
        
        for node_addr, node in self.cache_nodes.items():
            node_stats = node.health_check()
            total_cache_hits += node_stats["cache_hits"]
            total_cache_misses += node_stats["cache_misses"]
            total_memory_usage += node_stats["storage"]["memory_usage_mb"]
            node_health[node_addr] = node_stats["is_healthy"]
        
        total_requests = total_cache_hits + total_cache_misses
        hit_rate = total_cache_hits / max(1, total_requests)
        
        uptime = time.time() - self.stats["start_time"]
        
        return {
            "cluster_stats": {
                "total_nodes": len(self.cache_nodes),
                "healthy_nodes": sum(1 for healthy in node_health.values() if healthy),
                "uptime_seconds": uptime,
                "replication_factor": self.replication_factor
            },
            "performance_stats": {
                "total_requests": self.stats["total_requests"],
                "successful_requests": self.stats["successful_requests"],
                "failed_requests": self.stats["failed_requests"],
                "success_rate": self.stats["successful_requests"] / max(1, self.stats["total_requests"]),
                "cache_hit_rate": hit_rate,
                "total_cache_hits": total_cache_hits,
                "total_cache_misses": total_cache_misses
            },
            "resource_stats": {
                "total_memory_usage_mb": total_memory_usage,
                "average_memory_per_node": total_memory_usage / max(1, len(self.cache_nodes))
            },
            "node_health": node_health
        }

class CacheCluster:
    """Helper class for managing a cluster of cache nodes."""
    
    def __init__(self, cluster_size: int = 3, base_port: int = 8000):
        self.cluster_size = cluster_size
        self.base_port = base_port
        self.nodes = []
        
        # Create node addresses
        for i in range(cluster_size):
            node_addr = f"node{i}:{base_port + i}"
            self.nodes.append(node_addr)
        
        self.cache = DistributedCache(self.nodes)
    
    def get_cache(self) -> DistributedCache:
        """Get the distributed cache instance."""
        return self.cache
    
    def expand_cluster(self, additional_nodes: int = 1) -> List[str]:
        """Add more nodes to the cluster."""
        new_nodes = []
        
        for i in range(additional_nodes):
            node_id = len(self.nodes) + i
            node_addr = f"node{node_id}:{self.base_port + node_id}"
            
            if self.cache.add_node(node_addr):
                self.nodes.append(node_addr)
                new_nodes.append(node_addr)
        
        return new_nodes
    
    def shrink_cluster(self, nodes_to_remove: int = 1) -> List[str]:
        """Remove nodes from the cluster."""
        removed_nodes = []
        
        for _ in range(min(nodes_to_remove, len(self.nodes) - 1)):  # Keep at least 1 node
            if self.nodes:
                node_addr = self.nodes.pop()
                if self.cache.remove_node(node_addr):
                    removed_nodes.append(node_addr)
        
        return removed_nodes

def create_cache_cluster(nodes: List[str] = None, replication_factor: int = 3) -> DistributedCache:
    """Create a distributed cache cluster."""
    if nodes is None:
        nodes = ["node0:8000", "node1:8001", "node2:8002"]
    
    return DistributedCache(nodes, replication_factor)

if __name__ == "__main__":
    # Demo of distributed cache
    print("Distributed Cache System Demo")
    
    # Create cluster
    cluster = CacheCluster(cluster_size=3)
    cache = cluster.get_cache()
    
    # Test basic operations
    print("\n1. Testing basic operations...")
    cache.set("user:123", {"name": "Alice", "age": 30})
    cache.set("user:456", {"name": "Bob", "age": 25})
    
    user_data = cache.get("user:123")
    print(f"Retrieved user data: {user_data}")
    
    # Test batch operations
    print("\n2. Testing batch operations...")
    batch_data = {
        "session:abc": {"user_id": 123, "expires": 1234567890},
        "session:def": {"user_id": 456, "expires": 1234567890}
    }
    cache.batch_set(batch_data)
    
    sessions = cache.batch_get(["session:abc", "session:def"])
    print(f"Retrieved sessions: {sessions}")
    
    # Test cluster expansion
    print("\n3. Testing cluster expansion...")
    new_nodes = cluster.expand_cluster(1)
    print(f"Added nodes: {new_nodes}")
    
    # Test node failure simulation
    print("\n4. Testing fault tolerance...")
    cache.simulate_node_failure("node0:8000")
    
    # Operations should still work
    cache.set("test:key", "test_value")
    value = cache.get("test:key")
    print(f"Retrieved after node failure: {value}")
    
    # Recover node
    cache.recover_node("node0:8000")
    
    # Show final stats
    print("\n5. Final cluster stats:")
    stats = cache.get_stats()
    print(f"Total nodes: {stats['cluster_stats']['total_nodes']}")
    print(f"Healthy nodes: {stats['cluster_stats']['healthy_nodes']}")
    print(f"Success rate: {stats['performance_stats']['success_rate']:.2%}")
    print(f"Cache hit rate: {stats['performance_stats']['cache_hit_rate']:.2%}")
