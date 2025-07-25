"""
Hardware Security Module (HSM) Integration for EXTREME Portfolio Optimizer
Implements cryptographic order signing and authentication
"""

import time
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import threading
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import json

class HSMStatus(Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"

class KeyType(Enum):
    SIGNING = "SIGNING"
    ENCRYPTION = "ENCRYPTION"
    AUTHENTICATION = "AUTHENTICATION"

@dataclass
class CryptoKey:
    key_id: str
    key_type: KeyType
    algorithm: str
    key_size: int
    created_time: float
    expiry_time: Optional[float]
    usage_count: int
    max_usage: Optional[int]

@dataclass
class SignatureResult:
    signature: str
    key_id: str
    algorithm: str
    timestamp: float
    verification_data: Dict[str, str]

@dataclass
class HSMMetrics:
    operations_per_second: float
    latency_microseconds: float
    error_rate: float
    key_rotations: int
    security_events: int

class HardwareSecurityModule:
    """
    Simulated Hardware Security Module for cryptographic operations
    In production, this would interface with actual HSM hardware
    """
    
    def __init__(self, hsm_id: str = "HSM_001"):
        self.hsm_id = hsm_id
        self.status = HSMStatus.ONLINE
        
        # Key storage (in real HSM, keys never leave the hardware)
        self.keys: Dict[str, CryptoKey] = {}
        self.private_keys: Dict[str, any] = {}  # Simulated secure storage
        self.public_keys: Dict[str, any] = {}
        
        # Authentication and access control
        self.authenticated_users: Set[str] = set()
        self.access_policies: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.metrics = HSMMetrics(0.0, 0.0, 0.0, 0, 0)
        self.operation_log: List[Dict] = []
        
        # Security configuration
        self.max_sign_operations_per_second = 100000  # 100K ops/sec
        self.key_rotation_interval = 86400  # 24 hours
        
        # Threading for concurrent operations
        self.lock = threading.RLock()
        self.operation_counter = 0
        
        # Initialize with master keys
        self._initialize_master_keys()
        
        # Start background maintenance
        self._start_maintenance_thread()
    
    def _initialize_master_keys(self):
        """Initialize master signing and encryption keys"""
        # Master signing key
        signing_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        
        signing_key_id = "MASTER_SIGN_001"
        self.keys[signing_key_id] = CryptoKey(
            key_id=signing_key_id,
            key_type=KeyType.SIGNING,
            algorithm="RSA-PSS-SHA256",
            key_size=4096,
            created_time=time.time(),
            expiry_time=time.time() + (365 * 24 * 3600),  # 1 year
            usage_count=0,
            max_usage=1000000  # 1M signatures
        )
        self.private_keys[signing_key_id] = signing_key
        self.public_keys[signing_key_id] = signing_key.public_key()
        
        # Session signing keys (rotated frequently)
        for i in range(10):
            session_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            session_key_id = f"SESSION_SIGN_{i:03d}"
            
            self.keys[session_key_id] = CryptoKey(
                key_id=session_key_id,
                key_type=KeyType.SIGNING,
                algorithm="RSA-PSS-SHA256",
                key_size=2048,
                created_time=time.time(),
                expiry_time=time.time() + 3600,  # 1 hour
                usage_count=0,
                max_usage=10000  # 10K signatures
            )
            self.private_keys[session_key_id] = session_key
            self.public_keys[session_key_id] = session_key.public_key()
    
    def authenticate_user(self, user_id: str, credentials: Dict[str, str]) -> bool:
        """Authenticate user for HSM access"""
        # Simplified authentication (in production: multi-factor, certificates, etc.)
        expected_token = hashlib.sha256(f"{user_id}:HSM_SECRET".encode()).hexdigest()
        provided_token = credentials.get("auth_token", "")
        
        if hmac.compare_digest(expected_token, provided_token):
            with self.lock:
                self.authenticated_users.add(user_id)
                self._log_operation("AUTHENTICATE", user_id, "SUCCESS")
            return True
        else:
            self._log_operation("AUTHENTICATE", user_id, "FAILED")
            return False
    
    def sign_order(self, order_data: Dict[str, any], 
                   user_id: str = "TRADING_SYSTEM",
                   key_id: Optional[str] = None) -> SignatureResult:
        """
        Sign trading order with HSM private key
        """
        start_time = time.perf_counter()
        
        # Check authentication
        if user_id not in self.authenticated_users:
            raise ValueError(f"User {user_id} not authenticated")
        
        # Check HSM status
        if self.status != HSMStatus.ONLINE:
            raise RuntimeError(f"HSM {self.hsm_id} is {self.status.value}")
        
        # Select signing key
        if not key_id:
            key_id = self._select_signing_key()
        
        if key_id not in self.keys:
            raise ValueError(f"Key {key_id} not found")
        
        key_info = self.keys[key_id]
        
        # Check key validity
        current_time = time.time()
        if key_info.expiry_time and current_time > key_info.expiry_time:
            raise ValueError(f"Key {key_id} has expired")
        
        if key_info.max_usage and key_info.usage_count >= key_info.max_usage:
            raise ValueError(f"Key {key_id} has reached maximum usage")
        
        with self.lock:
            try:
                # Prepare signing data
                signing_data = self._prepare_signing_data(order_data)
                
                # Sign with private key
                private_key = self.private_keys[key_id]
                signature_bytes = private_key.sign(
                    signing_data.encode('utf-8'),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                # Encode signature
                signature_b64 = base64.b64encode(signature_bytes).decode('utf-8')
                
                # Update key usage
                key_info.usage_count += 1
                self.operation_counter += 1
                
                # Calculate latency
                latency_us = (time.perf_counter() - start_time) * 1_000_000
                
                # Update metrics
                self._update_metrics(latency_us)
                
                # Create signature result
                result = SignatureResult(
                    signature=signature_b64,
                    key_id=key_id,
                    algorithm=key_info.algorithm,
                    timestamp=current_time,
                    verification_data={
                        "hsm_id": self.hsm_id,
                        "user_id": user_id,
                        "operation_id": str(self.operation_counter),
                        "data_hash": hashlib.sha256(signing_data.encode()).hexdigest()
                    }
                )
                
                # Log operation
                self._log_operation("SIGN_ORDER", user_id, "SUCCESS", {
                    "key_id": key_id,
                    "latency_us": latency_us,
                    "data_size": len(signing_data)
                })
                
                return result
                
            except Exception as e:
                self._log_operation("SIGN_ORDER", user_id, "ERROR", {"error": str(e)})
                self.metrics.error_rate += 0.01
                raise
    
    def verify_signature(self, signature: str, order_data: Dict[str, any],
                        key_id: str, verification_data: Dict[str, str]) -> bool:
        """
        Verify order signature using public key
        """
        try:
            # Prepare signing data (same as signing)
            signing_data = self._prepare_signing_data(order_data)
            
            # Verify data hash
            expected_hash = hashlib.sha256(signing_data.encode()).hexdigest()
            if verification_data.get("data_hash") != expected_hash:
                return False
            
            # Get public key
            if key_id not in self.public_keys:
                return False
            
            public_key = self.public_keys[key_id]
            signature_bytes = base64.b64decode(signature.encode())
            
            # Verify signature
            public_key.verify(
                signature_bytes,
                signing_data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception:
            return False
    
    def _prepare_signing_data(self, order_data: Dict[str, any]) -> str:
        """
        Prepare canonical order data for signing
        """
        # Extract critical order fields
        critical_fields = {
            "symbol": order_data.get("symbol", ""),
            "side": order_data.get("side", ""),
            "quantity": order_data.get("quantity", 0),
            "price": order_data.get("price", 0.0),
            "order_type": order_data.get("order_type", ""),
            "timestamp": order_data.get("timestamp", time.time()),
            "user_id": order_data.get("user_id", ""),
            "account_id": order_data.get("account_id", "")
        }
        
        # Create canonical JSON (sorted keys)
        canonical_json = json.dumps(critical_fields, sort_keys=True, separators=(',', ':'))
        
        return canonical_json
    
    def _select_signing_key(self) -> str:
        """
        Select best available signing key
        """
        current_time = time.time()
        
        # Prefer session keys that are still valid and not overused
        valid_keys = []
        for key_id, key_info in self.keys.items():
            if (key_info.key_type == KeyType.SIGNING and
                (not key_info.expiry_time or current_time < key_info.expiry_time) and
                (not key_info.max_usage or key_info.usage_count < key_info.max_usage)):
                valid_keys.append((key_id, key_info))
        
        if not valid_keys:
            raise RuntimeError("No valid signing keys available")
        
        # Select key with lowest usage
        selected_key = min(valid_keys, key=lambda x: x[1].usage_count)
        return selected_key[0]
    
    def rotate_keys(self, force: bool = False) -> List[str]:
        """
        Rotate expired or overused keys
        """
        current_time = time.time()
        rotated_keys = []
        
        with self.lock:
            keys_to_rotate = []
            
            for key_id, key_info in self.keys.items():
                should_rotate = (
                    force or
                    (key_info.expiry_time and current_time > key_info.expiry_time) or
                    (key_info.max_usage and key_info.usage_count >= key_info.max_usage * 0.9)
                )
                
                if should_rotate and key_info.key_type == KeyType.SIGNING:
                    keys_to_rotate.append(key_id)
            
            # Generate new keys
            for old_key_id in keys_to_rotate:
                new_key_id = f"SESSION_SIGN_{int(time.time())}"
                
                # Generate new RSA key
                new_private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
                
                # Create new key record
                self.keys[new_key_id] = CryptoKey(
                    key_id=new_key_id,
                    key_type=KeyType.SIGNING,
                    algorithm="RSA-PSS-SHA256",
                    key_size=2048,
                    created_time=current_time,
                    expiry_time=current_time + 3600,  # 1 hour
                    usage_count=0,
                    max_usage=10000
                )
                
                self.private_keys[new_key_id] = new_private_key
                self.public_keys[new_key_id] = new_private_key.public_key()
                
                # Remove old key (keep for verification grace period)
                old_key = self.keys[old_key_id]
                old_key.expiry_time = current_time + 300  # 5 minute grace period
                
                rotated_keys.append(new_key_id)
                self.metrics.key_rotations += 1
                
                self._log_operation("ROTATE_KEY", "SYSTEM", "SUCCESS", {
                    "old_key_id": old_key_id,
                    "new_key_id": new_key_id
                })
        
        return rotated_keys
    
    def get_public_key(self, key_id: str) -> Optional[str]:
        """
        Get public key for verification (PEM format)
        """
        if key_id not in self.public_keys:
            return None
        
        public_key = self.public_keys[key_id]
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return pem.decode('utf-8')
    
    def _update_metrics(self, latency_us: float):
        """Update HSM performance metrics"""
        # Exponential moving average
        alpha = 0.1
        self.metrics.latency_microseconds = (
            (1 - alpha) * self.metrics.latency_microseconds + 
            alpha * latency_us
        )
        
        # Operations per second (simplified)
        self.metrics.operations_per_second = min(
            self.max_sign_operations_per_second,
            1_000_000 / max(1, self.metrics.latency_microseconds)
        )
    
    def _log_operation(self, operation_type: str, user_id: str, 
                      status: str, metadata: Dict = None):
        """Log HSM operation for audit trail"""
        log_entry = {
            "timestamp": time.time(),
            "hsm_id": self.hsm_id,
            "operation_type": operation_type,
            "user_id": user_id,
            "status": status,
            "metadata": metadata or {}
        }
        
        self.operation_log.append(log_entry)
        
        # Keep last 10K operations
        if len(self.operation_log) > 10000:
            self.operation_log.pop(0)
    
    def _start_maintenance_thread(self):
        """Start background maintenance thread"""
        import threading
        
        def maintenance_loop():
            while True:
                try:
                    time.sleep(300)  # 5 minutes
                    
                    # Auto-rotate keys if needed
                    self.rotate_keys()
                    
                    # Clean up expired keys
                    self._cleanup_expired_keys()
                    
                except Exception as e:
                    self._log_operation("MAINTENANCE", "SYSTEM", "ERROR", {"error": str(e)})
        
        maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        maintenance_thread.start()
    
    def _cleanup_expired_keys(self):
        """Remove expired keys past grace period"""
        current_time = time.time()
        
        with self.lock:
            expired_keys = []
            for key_id, key_info in self.keys.items():
                if (key_info.expiry_time and 
                    current_time > key_info.expiry_time + 300):  # 5 minute grace
                    expired_keys.append(key_id)
            
            for key_id in expired_keys:
                del self.keys[key_id]
                if key_id in self.private_keys:
                    del self.private_keys[key_id]
                if key_id in self.public_keys:
                    del self.public_keys[key_id]
    
    def get_status(self) -> Dict:
        """Get HSM status and metrics"""
        with self.lock:
            return {
                "hsm_id": self.hsm_id,
                "status": self.status.value,
                "metrics": {
                    "operations_per_second": self.metrics.operations_per_second,
                    "latency_microseconds": self.metrics.latency_microseconds,
                    "error_rate": self.metrics.error_rate,
                    "key_rotations": self.metrics.key_rotations
                },
                "key_count": len(self.keys),
                "active_keys": sum(1 for k in self.keys.values() 
                                 if not k.expiry_time or time.time() < k.expiry_time),
                "authenticated_users": len(self.authenticated_users),
                "total_operations": self.operation_counter
            }
    
    def emergency_shutdown(self, reason: str):
        """Emergency HSM shutdown"""
        with self.lock:
            self.status = HSMStatus.OFFLINE
            self.authenticated_users.clear()
            
            self._log_operation("EMERGENCY_SHUTDOWN", "SYSTEM", "EXECUTED", {
                "reason": reason,
                "shutdown_time": time.time()
            })

# Global HSM instance for the trading system
hsm_instance = HardwareSecurityModule("TRADING_HSM_001")

def initialize_hsm() -> bool:
    """Initialize HSM for trading system"""
    try:
        # Authenticate trading system
        credentials = {
            "auth_token": hashlib.sha256("TRADING_SYSTEM:HSM_SECRET".encode()).hexdigest()
        }
        
        success = hsm_instance.authenticate_user("TRADING_SYSTEM", credentials)
        return success
        
    except Exception as e:
        print(f"HSM initialization failed: {e}")
        return False

def sign_trading_order(order_data: Dict[str, any]) -> SignatureResult:
    """Sign trading order using HSM"""
    return hsm_instance.sign_order(order_data, "TRADING_SYSTEM")

def verify_trading_order(signature: str, order_data: Dict[str, any],
                        key_id: str, verification_data: Dict[str, str]) -> bool:
    """Verify trading order signature"""
    return hsm_instance.verify_signature(signature, order_data, key_id, verification_data)
