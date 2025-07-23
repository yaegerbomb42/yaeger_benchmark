"""
Distributed Cache Cluster - Manages multiple cache nodes with consistency
"""
import time
import threading
import hashlib
from typing import Dict, Any, Optional, List, Set
from enum import Enum
import random
import uuid
from .cache_node import CacheNode, ConsistencyLevel, CacheEntry

class ClusterState(Enum):
    """Cluster operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    FAILED = "failed"

class DistributedCache:
    """Main distributed cache cluster coordinator."""
    
    def __init__(self, cluster_name: str = "default", 
                 replication_factor: int = 3, 
                 min_nodes: int = 3):
        self.cluster_name = cluster_name
        self.replication_factor = replication_factor
        self.min_nodes = min_nodes
        
        # Node management
        self.nodes: Dict[str, CacheNode] = {}
        self.node_health: Dict[str, Dict[str, Any]] = {}
        self.cluster_state = ClusterState.FAILED  # Start as failed until nodes added
        
        # Cluster configuration
        self.default_consistency_level = ConsistencyLevel.QUORUM
        self.auto_rebalance_enabled = True
        self.failure_detection_enabled = True
        
        # Statistics
        self.cluster_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "consistency_violations": 0,
            "node_failures": 0,
            "rebalance_operations": 0
        }
        
        # Background monitoring
        self._monitor_thread = threading.Thread(target=self._cluster_monitor, daemon=True)
        self._monitor_running = True
        self._monitor_thread.start()
        
        self._lock = threading.Lock()
    
    def add_node(self, node_id: str, max_memory_mb: int = 1024) -> bool:
        """Add a new node to the cluster."""
        try:
            with self._lock:
                if node_id in self.nodes:
                    return False
                
                # Create new cache node
                node = CacheNode(node_id, max_memory_mb, self.replication_factor)
                
                # Add existing nodes as replicas
                for existing_node_id in self.nodes:
                    node.add_replica_node(existing_node_id)
                    self.nodes[existing_node_id].add_replica_node(node_id)
                
                self.nodes[node_id] = node
                self.node_health[node_id] = {"status": "healthy", "last_check": time.time()}
                
                # Update cluster state
                self._update_cluster_state()
                
                # Trigger rebalancing if enabled
                if self.auto_rebalance_enabled and len(self.nodes) > 1:
                    self._trigger_rebalance()
                
                return True
                
        except Exception:
            return False
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the cluster."""
        try:
            with self._lock:
                if node_id not in self.nodes:
                    return False
                
                # Remove node from all other nodes' replica lists
                for other_node_id, node in self.nodes.items():
                    if other_node_id != node_id:
                        node.remove_replica_node(node_id)
                
                # Remove the node
                del self.nodes[node_id]
                del self.node_health[node_id]
                
                # Update cluster state
                self._update_cluster_state()
                
                # Trigger rebalancing
                if self.auto_rebalance_enabled:
                    self._trigger_rebalance()
                
                return True
                
        except Exception:
            return False
    
    def get(self, key: str, consistency_level: Optional[ConsistencyLevel] = None) -> Optional[Any]:
        """Get value from the distributed cache."""
        self.cluster_stats["total_requests"] += 1
        
        try:
            if not self.nodes:
                self.cluster_stats["failed_requests"] += 1
                return None
            
            consistency = consistency_level or self.default_consistency_level
            
            # Find responsible nodes using consistent hashing
            responsible_nodes = self._get_responsible_nodes(key)
            
            # Try to read from responsible nodes
            for node_id in responsible_nodes:
                if node_id in self.nodes and self._is_node_healthy(node_id):
                    try:
                        value = self.nodes[node_id].get(key, consistency)
                        if value is not None:
                            self.cluster_stats["successful_requests"] += 1
                            return value
                    except Exception:
                        self.cluster_stats["consistency_violations"] += 1
                        continue
            
            self.cluster_stats["failed_requests"] += 1
            return None
            
        except Exception:
            self.cluster_stats["failed_requests"] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None,
            consistency_level: Optional[ConsistencyLevel] = None) -> bool:
        """Put value into the distributed cache."""
        self.cluster_stats["total_requests"] += 1
        
        try:
            if not self.nodes:
                self.cluster_stats["failed_requests"] += 1
                return False
            
            consistency = consistency_level or self.default_consistency_level
            
            # Find responsible nodes
            responsible_nodes = self._get_responsible_nodes(key)
            
            successful_writes = 0
            required_writes = self._get_required_writes(len(responsible_nodes), consistency)
            
            # Write to responsible nodes
            for node_id in responsible_nodes:
                if node_id in self.nodes and self._is_node_healthy(node_id):
                    try:
                        success = self.nodes[node_id].put(key, value, ttl, ConsistencyLevel.ONE)
                        if success:
                            successful_writes += 1
                            
                        if successful_writes >= required_writes:
                            break
                            
                    except Exception:
                        continue
            
            if successful_writes >= required_writes:
                self.cluster_stats["successful_requests"] += 1
                return True
            else:
                self.cluster_stats["failed_requests"] += 1
                return False
                
        except Exception:
            self.cluster_stats["failed_requests"] += 1
            return False
    
    def delete(self, key: str, consistency_level: Optional[ConsistencyLevel] = None) -> bool:
        """Delete key from the distributed cache."""
        self.cluster_stats["total_requests"] += 1
        
        try:
            if not self.nodes:
                self.cluster_stats["failed_requests"] += 1
                return False
            
            consistency = consistency_level or self.default_consistency_level
            
            # Find responsible nodes
            responsible_nodes = self._get_responsible_nodes(key)
            
            successful_deletes = 0
            required_deletes = self._get_required_writes(len(responsible_nodes), consistency)
            
            # Delete from responsible nodes
            for node_id in responsible_nodes:
                if node_id in self.nodes and self._is_node_healthy(node_id):
                    try:
                        success = self.nodes[node_id].delete(key, ConsistencyLevel.ONE)
                        if success:
                            successful_deletes += 1
                            
                        if successful_deletes >= required_deletes:
                            break
                            
                    except Exception:
                        continue
            
            if successful_deletes >= required_deletes:
                self.cluster_stats["successful_requests"] += 1
                return True
            else:
                self.cluster_stats["failed_requests"] += 1
                return False
                
        except Exception:
            self.cluster_stats["failed_requests"] += 1
            return False
    
    def _get_responsible_nodes(self, key: str) -> List[str]:
        """Get list of nodes responsible for a key using consistent hashing."""
        if not self.nodes:
            return []
        
        # Simple consistent hashing implementation
        hash_obj = hashlib.md5(key.encode())
        key_hash = int(hash_obj.hexdigest(), 16)
        
        # Sort nodes by their hash
        node_hashes = []
        for node_id in self.nodes:
            node_hash_obj = hashlib.md5(node_id.encode())
            node_hash = int(node_hash_obj.hexdigest(), 16)
            node_hashes.append((node_hash, node_id))
        
        node_hashes.sort()
        
        # Find the first node whose hash is >= key hash
        responsible_nodes = []
        for node_hash, node_id in node_hashes:
            if node_hash >= key_hash:
                responsible_nodes.append(node_id)
                break
        
        # If no node found, wrap around to first node
        if not responsible_nodes:
            responsible_nodes.append(node_hashes[0][1])
        
        # Add additional replica nodes
        start_idx = node_hashes.index((node_hashes[0][0], responsible_nodes[0]))
        for i in range(1, min(self.replication_factor, len(node_hashes))):
            next_idx = (start_idx + i) % len(node_hashes)
            responsible_nodes.append(node_hashes[next_idx][1])
        
        return responsible_nodes[:self.replication_factor]
    
    def _get_required_writes(self, total_nodes: int, consistency_level: ConsistencyLevel) -> int:
        """Calculate required successful writes for consistency level."""
        if consistency_level == ConsistencyLevel.ONE:
            return 1
        elif consistency_level == ConsistencyLevel.ALL:
            return total_nodes
        else:  # QUORUM
            return (total_nodes // 2) + 1
    
    def _is_node_healthy(self, node_id: str) -> bool:
        """Check if a node is healthy."""
        health = self.node_health.get(node_id, {})
        return health.get("status") == "healthy"
    
    def _update_cluster_state(self):
        """Update overall cluster state based on node health."""
        total_nodes = len(self.nodes)
        healthy_nodes = sum(1 for health in self.node_health.values() 
                          if health.get("status") == "healthy")
        
        if total_nodes == 0:
            self.cluster_state = ClusterState.FAILED
        elif healthy_nodes < self.min_nodes:
            self.cluster_state = ClusterState.CRITICAL
        elif healthy_nodes < total_nodes * 0.8:  # Less than 80% healthy
            self.cluster_state = ClusterState.DEGRADED
        else:
            self.cluster_state = ClusterState.HEALTHY
    
    def _cluster_monitor(self):
        """Background thread for cluster monitoring."""
        while self._monitor_running:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Health check all nodes
                for node_id, node in self.nodes.items():
                    try:
                        health = node.health_check()
                        self.node_health[node_id] = {
                            "status": health["status"],
                            "last_check": time.time(),
                            "details": health
                        }
                        
                        # Detect node failures
                        if health["status"] in ["critical", "failed"]:
                            self.cluster_stats["node_failures"] += 1
                            
                    except Exception:
                        # Mark node as unhealthy if health check fails
                        self.node_health[node_id] = {
                            "status": "failed",
                            "last_check": time.time(),
                            "details": {"error": "Health check failed"}
                        }
                        self.cluster_stats["node_failures"] += 1
                
                # Update cluster state
                self._update_cluster_state()
                
                # Trigger rebalancing if needed
                if (self.auto_rebalance_enabled and 
                    self.cluster_state in [ClusterState.HEALTHY, ClusterState.DEGRADED]):
                    self._check_rebalance_needed()
                
            except Exception:
                continue
    
    def _trigger_rebalance(self):
        """Trigger cluster rebalancing operation."""
        # Simplified rebalancing - in a real implementation this would be much more complex
        self.cluster_stats["rebalance_operations"] += 1
    
    def _check_rebalance_needed(self):
        """Check if cluster rebalancing is needed."""
        # Simple heuristic: rebalance if node count changed significantly
        # In a real implementation, this would check data distribution
        pass
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics."""
        total_memory = 0
        used_memory = 0
        total_entries = 0
        
        node_stats = {}
        for node_id, node in self.nodes.items():
            stats = node.get_node_stats()
            node_stats[node_id] = stats
            
            memory_stats = stats["memory_stats"]
            total_memory += memory_stats["max_memory"]
            used_memory += memory_stats["used_memory"]
            total_entries += memory_stats["entry_count"]
        
        # Calculate cluster-wide metrics
        success_rate = 0
        if self.cluster_stats["total_requests"] > 0:
            success_rate = (self.cluster_stats["successful_requests"] / 
                          self.cluster_stats["total_requests"])
        
        return {
            "cluster_name": self.cluster_name,
            "cluster_state": self.cluster_state.value,
            "node_count": len(self.nodes),
            "healthy_nodes": sum(1 for h in self.node_health.values() 
                               if h.get("status") == "healthy"),
            "replication_factor": self.replication_factor,
            "total_memory_mb": total_memory // (1024 * 1024),
            "used_memory_mb": used_memory // (1024 * 1024),
            "memory_utilization": used_memory / total_memory if total_memory > 0 else 0,
            "total_entries": total_entries,
            "success_rate": success_rate,
            "cluster_stats": self.cluster_stats.copy(),
            "node_stats": node_stats,
            "node_health": self.node_health.copy()
        }
    
    def shutdown(self):
        """Gracefully shutdown the cluster."""
        self._monitor_running = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        # Clear all nodes
        self.nodes.clear()
        self.node_health.clear()
        self.cluster_state = ClusterState.FAILED
