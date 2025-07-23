"""
Consistent Hashing - Distribute keys across cache nodes
"""
import hashlib
import bisect
from typing import List, Dict, Any, Optional, Set
import threading

class ConsistentHashRing:
    """Consistent hashing ring for distributing keys across nodes."""
    
    def __init__(self, nodes: List[str] = None, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes  # Number of virtual nodes per physical node
        self.ring: Dict[int, str] = {}  # Hash -> Node mapping
        self.sorted_hashes: List[int] = []  # Sorted list of hash values
        self.nodes: Set[str] = set()  # Set of physical nodes
        self._lock = threading.RLock()
        
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def _hash(self, key: str) -> int:
        """Generate hash value for a key."""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
    
    def add_node(self, node: str) -> bool:
        """Add a node to the hash ring."""
        with self._lock:
            if node in self.nodes:
                return False  # Node already exists
            
            self.nodes.add(node)
            
            # Add virtual nodes for this physical node
            for i in range(self.virtual_nodes):
                virtual_key = f"{node}:{i}"
                hash_value = self._hash(virtual_key)
                
                self.ring[hash_value] = node
                bisect.insort(self.sorted_hashes, hash_value)
            
            return True
    
    def remove_node(self, node: str) -> bool:
        """Remove a node from the hash ring."""
        with self._lock:
            if node not in self.nodes:
                return False  # Node doesn't exist
            
            self.nodes.remove(node)
            
            # Remove all virtual nodes for this physical node
            hashes_to_remove = []
            for hash_value, ring_node in self.ring.items():
                if ring_node == node:
                    hashes_to_remove.append(hash_value)
            
            for hash_value in hashes_to_remove:
                del self.ring[hash_value]
                self.sorted_hashes.remove(hash_value)
            
            return True
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a given key."""
        with self._lock:
            if not self.ring:
                return None
            
            hash_value = self._hash(key)
            
            # Find the first node with hash >= key hash (clockwise on ring)
            idx = bisect.bisect_right(self.sorted_hashes, hash_value)
            
            if idx == len(self.sorted_hashes):
                # Wrap around to the first node
                idx = 0
            
            return self.ring[self.sorted_hashes[idx]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """Get multiple nodes for a key (for replication)."""
        with self._lock:
            if not self.ring or count <= 0:
                return []
            
            hash_value = self._hash(key)
            
            # Find starting position
            idx = bisect.bisect_right(self.sorted_hashes, hash_value)
            if idx == len(self.sorted_hashes):
                idx = 0
            
            nodes = []
            seen_nodes = set()
            
            # Collect unique nodes going clockwise
            for _ in range(len(self.sorted_hashes)):
                node = self.ring[self.sorted_hashes[idx]]
                
                if node not in seen_nodes:
                    nodes.append(node)
                    seen_nodes.add(node)
                    
                    if len(nodes) >= count:
                        break
                
                idx = (idx + 1) % len(self.sorted_hashes)
            
            return nodes
    
    def get_node_distribution(self) -> Dict[str, int]:
        """Get the distribution of virtual nodes across physical nodes."""
        with self._lock:
            distribution = {}
            for node in self.ring.values():
                distribution[node] = distribution.get(node, 0) + 1
            return distribution
    
    def rebalance_keys(self, old_ring: 'ConsistentHashRing', 
                      key_migration_callback: callable = None) -> Dict[str, List[str]]:
        """Calculate which keys need to be migrated when nodes change."""
        migrations = {}  # old_node -> [keys_to_migrate]
        
        # This would typically work with a list of all keys
        # For now, return the structure for the migration logic
        return migrations
    
    def get_ring_info(self) -> Dict[str, Any]:
        """Get information about the current ring state."""
        with self._lock:
            return {
                "total_nodes": len(self.nodes),
                "virtual_nodes_per_physical": self.virtual_nodes,
                "total_virtual_nodes": len(self.ring),
                "node_distribution": self.get_node_distribution(),
                "ring_coverage": len(self.sorted_hashes) > 0,
                "nodes": list(self.nodes)
            }

class LoadBalancer:
    """Load balancer using consistent hashing for request distribution."""
    
    def __init__(self, nodes: List[str] = None, replication_factor: int = 3):
        self.hash_ring = ConsistentHashRing(nodes)
        self.replication_factor = replication_factor
        self.node_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Initialize stats for existing nodes
        if nodes:
            for node in nodes:
                self._init_node_stats(node)
    
    def _init_node_stats(self, node: str):
        """Initialize statistics for a node."""
        self.node_stats[node] = {
            "requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "last_request_time": 0.0,
            "is_healthy": True
        }
    
    def add_node(self, node: str) -> bool:
        """Add a node to the load balancer."""
        with self._lock:
            success = self.hash_ring.add_node(node)
            if success:
                self._init_node_stats(node)
            return success
    
    def remove_node(self, node: str) -> bool:
        """Remove a node from the load balancer."""
        with self._lock:
            success = self.hash_ring.remove_node(node)
            if success and node in self.node_stats:
                del self.node_stats[node]
            return success
    
    def get_primary_node(self, key: str) -> Optional[str]:
        """Get the primary node for a key."""
        return self.hash_ring.get_node(key)
    
    def get_replica_nodes(self, key: str) -> List[str]:
        """Get all replica nodes for a key (including primary)."""
        return self.hash_ring.get_nodes(key, self.replication_factor)
    
    def route_request(self, key: str, operation: str = "read") -> List[str]:
        """Route a request and return list of nodes to contact."""
        with self._lock:
            if operation == "read":
                # For reads, try primary first, fallback to replicas
                primary = self.get_primary_node(key)
                if primary and self.node_stats[primary]["is_healthy"]:
                    return [primary]
                else:
                    # Primary unhealthy, try replicas
                    replicas = self.get_replica_nodes(key)
                    healthy_replicas = [
                        node for node in replicas 
                        if node in self.node_stats and self.node_stats[node]["is_healthy"]
                    ]
                    return healthy_replicas[:1]  # Return one healthy replica
            
            elif operation == "write":
                # For writes, contact all replicas
                replicas = self.get_replica_nodes(key)
                healthy_replicas = [
                    node for node in replicas 
                    if node in self.node_stats and self.node_stats[node]["is_healthy"]
                ]
                return healthy_replicas
            
            else:
                # Unknown operation, return primary
                primary = self.get_primary_node(key)
                return [primary] if primary else []
    
    def record_request(self, node: str, success: bool, response_time: float = 0.0):
        """Record statistics for a request to a node."""
        with self._lock:
            if node not in self.node_stats:
                self._init_node_stats(node)
            
            stats = self.node_stats[node]
            stats["requests"] += 1
            stats["last_request_time"] = response_time
            
            if success:
                stats["successful_requests"] += 1
            else:
                stats["failed_requests"] += 1
            
            # Update rolling average response time
            total_requests = stats["requests"]
            current_avg = stats["average_response_time"]
            stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
    
    def mark_node_unhealthy(self, node: str):
        """Mark a node as unhealthy."""
        with self._lock:
            if node in self.node_stats:
                self.node_stats[node]["is_healthy"] = False
    
    def mark_node_healthy(self, node: str):
        """Mark a node as healthy."""
        with self._lock:
            if node in self.node_stats:
                self.node_stats[node]["is_healthy"] = True
    
    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            total_requests = sum(stats["requests"] for stats in self.node_stats.values())
            healthy_nodes = sum(1 for stats in self.node_stats.values() if stats["is_healthy"])
            
            return {
                "total_nodes": len(self.node_stats),
                "healthy_nodes": healthy_nodes,
                "total_requests": total_requests,
                "replication_factor": self.replication_factor,
                "node_stats": self.node_stats.copy(),
                "ring_info": self.hash_ring.get_ring_info()
            }
    
    def rebalance_cluster(self) -> Dict[str, Any]:
        """Trigger cluster rebalancing and return migration plan."""
        # In a real implementation, this would:
        # 1. Calculate optimal data distribution
        # 2. Generate migration plan
        # 3. Coordinate data movement between nodes
        
        ring_info = self.hash_ring.get_ring_info()
        
        return {
            "rebalance_needed": False,  # Placeholder
            "migration_plan": {},
            "estimated_migration_time": 0,
            "ring_info": ring_info
        }
