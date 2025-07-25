"""
Compliance Module for EXTREME Portfolio Optimizer
Implements MiFID II, Dodd-Frank, and other regulatory requirements
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import threading
from datetime import datetime, timedelta

class RegulationType(Enum):
    MIFID_II = "MIFID_II"
    DODD_FRANK = "DODD_FRANK"
    VOLCKER_RULE = "VOLCKER_RULE"
    BASEL_III = "BASEL_III"
    CFTC = "CFTC"
    SEC = "SEC"

class ViolationSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    BREACH = "BREACH"

@dataclass
class RegulatoryLimit:
    rule_name: str
    regulation_type: RegulationType
    limit_value: float
    limit_type: str  # "percentage", "absolute", "ratio"
    description: str
    enforcement_start: datetime
    penalties: Dict[str, float]  # {"warning": 10000, "breach": 100000}

@dataclass
class ComplianceEvent:
    event_id: str
    rule_name: str
    event_type: str  # "VIOLATION", "NEAR_BREACH", "CLEARED"
    severity: ViolationSeverity
    current_value: float
    limit_value: float
    timestamp: datetime
    description: str
    required_action: str
    deadline: datetime
    estimated_penalty: float

@dataclass
class TradeCompliance:
    trade_id: str
    symbol: str
    quantity: int
    price: float
    side: str
    timestamp: datetime
    compliance_checks: Dict[str, bool]
    violations: List[str]
    approved: bool

class ComplianceMonitor:
    """
    Real-time regulatory compliance monitoring and reporting
    """
    
    def __init__(self):
        # Regulatory limits
        self.regulatory_limits = self._initialize_regulatory_limits()
        
        # Compliance state
        self.active_violations: Dict[str, ComplianceEvent] = {}
        self.compliance_history: List[ComplianceEvent] = []
        self.trade_log: List[TradeCompliance] = []
        
        # Position tracking for regulatory purposes
        self.position_snapshots: List[Dict] = []
        self.large_trader_positions: Dict[str, float] = {}
        
        # Threading
        self.lock = threading.RLock()
        self.last_check_time = time.time()
        
        # Reporting requirements
        self.mifid_report_buffer: List[Dict] = []
        self.dodd_frank_report_buffer: List[Dict] = []
        
    def _initialize_regulatory_limits(self) -> Dict[str, RegulatoryLimit]:
        """Initialize regulatory limits and thresholds"""
        limits = {}
        
        # MiFID II Limits
        limits["mifid_position_limit"] = RegulatoryLimit(
            rule_name="MiFID II Position Limit",
            regulation_type=RegulationType.MIFID_II,
            limit_value=0.05,  # 5% of portfolio
            limit_type="percentage",
            description="Maximum position size as percentage of portfolio",
            enforcement_start=datetime(2018, 1, 3),
            penalties={"warning": 50000, "breach": 500000}
        )
        
        limits["mifid_large_trader"] = RegulatoryLimit(
            rule_name="MiFID II Large Trader Reporting",
            regulation_type=RegulationType.MIFID_II,
            limit_value=50000000,  # €50M
            limit_type="absolute",
            description="Large trader position reporting threshold",
            enforcement_start=datetime(2018, 1, 3),
            penalties={"warning": 25000, "breach": 250000}
        )
        
        # Dodd-Frank Limits
        limits["dodd_frank_prop_trading"] = RegulatoryLimit(
            rule_name="Dodd-Frank Proprietary Trading",
            regulation_type=RegulationType.DODD_FRANK,
            limit_value=0.03,  # 3% of Tier 1 capital
            limit_type="percentage",
            description="Volcker Rule proprietary trading limit",
            enforcement_start=datetime(2015, 7, 21),
            penalties={"warning": 100000, "breach": 1000000}
        )
        
        limits["dodd_frank_swap_margin"] = RegulatoryLimit(
            rule_name="Dodd-Frank Swap Margin",
            regulation_type=RegulationType.DODD_FRANK,
            limit_value=0.20,  # 20% initial margin
            limit_type="percentage",
            description="Minimum initial margin for non-cleared swaps",
            enforcement_start=datetime(2016, 9, 1),
            penalties={"warning": 75000, "breach": 750000}
        )
        
        # Basel III Limits
        limits["basel_leverage_ratio"] = RegulatoryLimit(
            rule_name="Basel III Leverage Ratio",
            regulation_type=RegulationType.BASEL_III,
            limit_value=0.03,  # 3% minimum
            limit_type="ratio",
            description="Minimum leverage ratio (Tier 1 capital / total exposure)",
            enforcement_start=datetime(2018, 1, 1),
            penalties={"warning": 200000, "breach": 2000000}
        )
        
        limits["basel_liquidity_ratio"] = RegulatoryLimit(
            rule_name="Basel III Liquidity Coverage Ratio",
            regulation_type=RegulationType.BASEL_III,
            limit_value=1.0,  # 100% minimum
            limit_type="ratio",
            description="Minimum liquidity coverage ratio",
            enforcement_start=datetime(2019, 1, 1),
            penalties={"warning": 150000, "breach": 1500000}
        )
        
        # SEC Limits
        limits["sec_net_capital"] = RegulatoryLimit(
            rule_name="SEC Net Capital Rule",
            regulation_type=RegulationType.SEC,
            limit_value=250000,  # $250K minimum
            limit_type="absolute",
            description="Minimum net capital requirement",
            enforcement_start=datetime(1975, 1, 1),
            penalties={"warning": 50000, "breach": 500000}
        )
        
        # CFTC Limits
        limits["cftc_position_limit"] = RegulatoryLimit(
            rule_name="CFTC Position Limits",
            regulation_type=RegulationType.CFTC,
            limit_value=0.25,  # 25% of deliverable supply
            limit_type="percentage",
            description="Position limits for physical commodity derivatives",
            enforcement_start=datetime(2013, 10, 14),
            penalties={"warning": 75000, "breach": 1000000}
        )
        
        return limits
    
    def check_pre_trade_compliance(self, symbol: str, quantity: int, 
                                 price: float, side: str, 
                                 current_positions: Dict[str, float],
                                 portfolio_value: float) -> TradeCompliance:
        """
        Check compliance before executing trade
        """
        trade_id = f"TRADE_{int(time.time() * 1000000)}"
        
        compliance_checks = {}
        violations = []
        
        # Calculate new position after trade
        current_position = current_positions.get(symbol, 0.0)
        if side == "BUY":
            new_position_value = (current_position * portfolio_value) + (quantity * price)
        else:
            new_position_value = (current_position * portfolio_value) - (quantity * price)
        
        new_position_weight = new_position_value / portfolio_value if portfolio_value > 0 else 0.0
        
        # MiFID II Position Limit Check
        mifid_limit = self.regulatory_limits["mifid_position_limit"]
        if abs(new_position_weight) > mifid_limit.limit_value:
            compliance_checks["mifid_position_limit"] = False
            violations.append(f"Position would exceed MiFID II limit of {mifid_limit.limit_value:.1%}")
        else:
            compliance_checks["mifid_position_limit"] = True
        
        # Large Trader Reporting Check
        large_trader_limit = self.regulatory_limits["mifid_large_trader"]
        if abs(new_position_value) > large_trader_limit.limit_value:
            compliance_checks["large_trader_reporting"] = True  # Not a violation, just reporting requirement
            self.large_trader_positions[symbol] = new_position_value
        else:
            compliance_checks["large_trader_reporting"] = True
        
        # Dodd-Frank Proprietary Trading Check (simplified)
        dodd_frank_limit = self.regulatory_limits["dodd_frank_prop_trading"]
        # This would require more complex logic in practice
        compliance_checks["dodd_frank_prop_trading"] = True
        
        # Concentration Risk Check
        if abs(new_position_weight) > 0.10:  # 10% concentration limit
            compliance_checks["concentration_risk"] = False
            violations.append("Position would exceed 10% concentration limit")
        else:
            compliance_checks["concentration_risk"] = True
        
        # Determine if trade is approved
        approved = len(violations) == 0
        
        trade_compliance = TradeCompliance(
            trade_id=trade_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            side=side,
            timestamp=datetime.now(),
            compliance_checks=compliance_checks,
            violations=violations,
            approved=approved
        )
        
        # Log trade
        with self.lock:
            self.trade_log.append(trade_compliance)
            if len(self.trade_log) > 10000:  # Keep last 10K trades
                self.trade_log.pop(0)
        
        return trade_compliance
    
    def check_portfolio_compliance(self, positions: Dict[str, float],
                                 portfolio_value: float,
                                 risk_metrics: Dict[str, float]) -> List[ComplianceEvent]:
        """
        Check portfolio-level compliance
        """
        current_time = datetime.now()
        violations = []
        
        with self.lock:
            # Position Size Compliance (MiFID II)
            for symbol, weight in positions.items():
                if abs(weight) > self.regulatory_limits["mifid_position_limit"].limit_value:
                    event_id = f"MIFID_POS_{symbol}_{int(time.time())}"
                    
                    severity = ViolationSeverity.CRITICAL if abs(weight) > 0.07 else ViolationSeverity.WARNING
                    
                    violation = ComplianceEvent(
                        event_id=event_id,
                        rule_name="MiFID II Position Limit",
                        event_type="VIOLATION",
                        severity=severity,
                        current_value=abs(weight),
                        limit_value=self.regulatory_limits["mifid_position_limit"].limit_value,
                        timestamp=current_time,
                        description=f"Position in {symbol} exceeds MiFID II limit",
                        required_action="Reduce position size",
                        deadline=current_time + timedelta(hours=2),
                        estimated_penalty=self._calculate_penalty("mifid_position_limit", severity)
                    )
                    violations.append(violation)
                    self.active_violations[event_id] = violation
            
            # Leverage Compliance (Basel III)
            leverage_ratio = risk_metrics.get("leverage", 0.0)
            basel_limit = self.regulatory_limits["basel_leverage_ratio"]
            
            if leverage_ratio > (1.0 / basel_limit.limit_value):  # Leverage > 33x (1/3%)
                event_id = f"BASEL_LEV_{int(time.time())}"
                
                severity = ViolationSeverity.BREACH if leverage_ratio > 50 else ViolationSeverity.CRITICAL
                
                violation = ComplianceEvent(
                    event_id=event_id,
                    rule_name="Basel III Leverage Ratio",
                    event_type="VIOLATION", 
                    severity=severity,
                    current_value=leverage_ratio,
                    limit_value=(1.0 / basel_limit.limit_value),
                    timestamp=current_time,
                    description=f"Leverage ratio {leverage_ratio:.1f}x exceeds Basel III limit",
                    required_action="Reduce leverage immediately",
                    deadline=current_time + timedelta(hours=1),
                    estimated_penalty=self._calculate_penalty("basel_leverage_ratio", severity)
                )
                violations.append(violation)
                self.active_violations[event_id] = violation
            
            # VaR Compliance
            var_95 = risk_metrics.get("var_95", 0.0)
            if var_95 > 0.15:  # 15% VaR limit
                event_id = f"VAR_LIMIT_{int(time.time())}"
                
                severity = ViolationSeverity.CRITICAL if var_95 > 0.20 else ViolationSeverity.WARNING
                
                violation = ComplianceEvent(
                    event_id=event_id,
                    rule_name="VaR Limit",
                    event_type="VIOLATION",
                    severity=severity,
                    current_value=var_95,
                    limit_value=0.15,
                    timestamp=current_time,
                    description=f"Portfolio VaR {var_95:.1%} exceeds limit",
                    required_action="Reduce portfolio risk",
                    deadline=current_time + timedelta(hours=4),
                    estimated_penalty=50000 if severity == ViolationSeverity.WARNING else 200000
                )
                violations.append(violation)
                self.active_violations[event_id] = violation
            
            # Update compliance history
            for violation in violations:
                self.compliance_history.append(violation)
                if len(self.compliance_history) > 1000:
                    self.compliance_history.pop(0)
        
        return violations
    
    def _calculate_penalty(self, rule_name: str, severity: ViolationSeverity) -> float:
        """Calculate estimated regulatory penalty"""
        rule = self.regulatory_limits.get(rule_name)
        if not rule:
            return 10000  # Default penalty
        
        base_penalty = rule.penalties.get("warning", 10000)
        
        multipliers = {
            ViolationSeverity.INFO: 0.0,
            ViolationSeverity.WARNING: 1.0,
            ViolationSeverity.CRITICAL: 3.0,
            ViolationSeverity.BREACH: 10.0
        }
        
        return base_penalty * multipliers.get(severity, 1.0)
    
    def generate_mifid_report(self, positions: Dict[str, float],
                            portfolio_value: float) -> Dict:
        """
        Generate MiFID II regulatory report
        """
        report_data = {
            "report_id": f"MIFID_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "report_date": datetime.now().isoformat(),
            "reporting_entity": "EXTREME_TRADING_LLC",
            "total_portfolio_value": portfolio_value,
            "currency": "USD",
            "positions": [],
            "large_positions": [],
            "violations": []
        }
        
        # Position reporting
        for symbol, weight in positions.items():
            position_value = weight * portfolio_value
            
            position_report = {
                "symbol": symbol,
                "position_value": position_value,
                "position_weight": weight,
                "isin": f"US{symbol}000000",  # Simplified ISIN
                "classification": "EQUITY",
                "exchange": "NASDAQ"
            }
            report_data["positions"].append(position_report)
            
            # Large position reporting
            if abs(position_value) > self.regulatory_limits["mifid_large_trader"].limit_value:
                large_position = {
                    "symbol": symbol,
                    "position_value": position_value,
                    "threshold_exceeded": "LARGE_TRADER",
                    "notification_required": True
                }
                report_data["large_positions"].append(large_position)
        
        # Violation reporting
        for event in self.active_violations.values():
            if event.rule_name.startswith("MiFID"):
                violation_report = {
                    "violation_id": event.event_id,
                    "rule_violated": event.rule_name,
                    "severity": event.severity.value,
                    "current_value": event.current_value,
                    "limit_value": event.limit_value,
                    "timestamp": event.timestamp.isoformat(),
                    "remedial_action": event.required_action
                }
                report_data["violations"].append(violation_report)
        
        # Buffer for batch reporting
        with self.lock:
            self.mifid_report_buffer.append(report_data)
            if len(self.mifid_report_buffer) > 100:
                self.mifid_report_buffer.pop(0)
        
        return report_data
    
    def generate_dodd_frank_report(self, positions: Dict[str, float],
                                 portfolio_value: float,
                                 swap_positions: Dict[str, float] = None) -> Dict:
        """
        Generate Dodd-Frank regulatory report
        """
        swap_positions = swap_positions or {}
        
        report_data = {
            "report_id": f"DODD_FRANK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "report_date": datetime.now().isoformat(),
            "reporting_entity": "EXTREME_TRADING_LLC",
            "entity_type": "SWAP_DEALER",
            "total_portfolio_value": portfolio_value,
            "proprietary_trading": [],
            "swap_positions": [],
            "margin_requirements": [],
            "violations": []
        }
        
        # Proprietary trading analysis
        total_prop_trading = sum(abs(weight) * portfolio_value for weight in positions.values())
        prop_trading_ratio = total_prop_trading / (portfolio_value * 25)  # Assuming 25x Tier 1 capital ratio
        
        prop_trading_report = {
            "total_proprietary_trading": total_prop_trading,
            "tier_1_capital_ratio": prop_trading_ratio,
            "volcker_rule_limit": self.regulatory_limits["dodd_frank_prop_trading"].limit_value,
            "compliant": prop_trading_ratio <= self.regulatory_limits["dodd_frank_prop_trading"].limit_value
        }
        report_data["proprietary_trading"].append(prop_trading_report)
        
        # Swap position reporting
        for instrument, notional in swap_positions.items():
            swap_report = {
                "instrument": instrument,
                "notional_amount": notional,
                "currency": "USD",
                "maturity": "1Y",  # Simplified
                "counterparty": "BANK_A",
                "cleared": True,
                "margin_posted": notional * 0.02  # 2% margin
            }
            report_data["swap_positions"].append(swap_report)
        
        # Buffer for batch reporting
        with self.lock:
            self.dodd_frank_report_buffer.append(report_data)
            if len(self.dodd_frank_report_buffer) > 100:
                self.dodd_frank_report_buffer.pop(0)
        
        return report_data
    
    def check_real_time_limits(self, positions: Dict[str, float],
                             portfolio_value: float) -> List[str]:
        """
        Fast real-time compliance check (for high-frequency trading)
        """
        violations = []
        
        # Quick position size check
        for symbol, weight in positions.items():
            if abs(weight) > 0.05:  # 5% limit
                violations.append(f"Position {symbol} exceeds 5% limit")
        
        # Quick leverage check
        total_exposure = sum(abs(weight) for weight in positions.values())
        if total_exposure > 2.0:  # 2x leverage limit
            violations.append("Portfolio leverage exceeds 2x limit")
        
        return violations
    
    def clear_violation(self, event_id: str, resolution_note: str):
        """Clear a compliance violation"""
        with self.lock:
            if event_id in self.active_violations:
                event = self.active_violations[event_id]
                event.event_type = "CLEARED"
                event.description += f" | RESOLVED: {resolution_note}"
                del self.active_violations[event_id]
    
    def get_compliance_summary(self) -> Dict:
        """Get summary of compliance status"""
        with self.lock:
            active_critical = sum(1 for v in self.active_violations.values() 
                                if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.BREACH])
            
            active_warnings = sum(1 for v in self.active_violations.values()
                                if v.severity == ViolationSeverity.WARNING)
            
            total_penalties = sum(v.estimated_penalty for v in self.active_violations.values())
            
            return {
                "total_violations": len(self.active_violations),
                "critical_violations": active_critical,
                "warnings": active_warnings,
                "estimated_penalties": total_penalties,
                "last_report_time": datetime.now().isoformat(),
                "compliance_score": max(0, 100 - (active_critical * 20) - (active_warnings * 5)),
                "next_reporting_deadlines": self._get_next_deadlines()
            }
    
    def _get_next_deadlines(self) -> List[Dict]:
        """Get upcoming compliance deadlines"""
        deadlines = []
        current_time = datetime.now()
        
        for event in self.active_violations.values():
            if event.deadline > current_time:
                deadlines.append({
                    "event_id": event.event_id,
                    "rule_name": event.rule_name,
                    "deadline": event.deadline.isoformat(),
                    "hours_remaining": (event.deadline - current_time).total_seconds() / 3600
                })
        
        # Sort by deadline
        deadlines.sort(key=lambda x: x["deadline"])
        
        return deadlines[:10]  # Return next 10 deadlines
