"""
Risk Model Module for EXTREME Portfolio Optimizer
Implements factor risk models, scenario analysis, and stress testing
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import threading
import random

try:
    import numpy as np
except ImportError:
    # Fallback for numpy functionality
    class np:
        @staticmethod
        def array(arr):
            return arr
        
        @staticmethod
        def sqrt(x):
            return x ** 0.5
        
        @staticmethod
        def random():
            class Random:
                @staticmethod
                def multivariate_normal(mean, cov, size=None):
                    if size:
                        return [[random.gauss(m, 0.1) for m in mean] for _ in range(size)]
                    else:
                        return [random.gauss(m, 0.1) for m in mean]
            return Random()
        
        @staticmethod
        def dot(a, b):
            if isinstance(a[0], (list, tuple)):
                return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
            else:
                return sum(a[i] * b[i] for i in range(len(a)))

class StressScenario(Enum):
    MARKET_CRASH_2008 = "MARKET_CRASH_2008"
    COVID_CRASH_2020 = "COVID_CRASH_2020"
    FLASH_CRASH_2010 = "FLASH_CRASH_2010"
    RATE_SHOCK = "RATE_SHOCK"
    CREDIT_CRISIS = "CREDIT_CRISIS"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"

@dataclass
class RiskFactor:
    name: str
    current_value: float
    volatility: float
    mean_reversion_speed: float
    long_term_mean: float

@dataclass
class StressTestResult:
    scenario: StressScenario
    portfolio_pnl: float
    portfolio_pnl_pct: float
    max_loss: float
    time_to_recovery_days: int
    worst_positions: List[Tuple[str, float]]  # (symbol, loss)
    risk_metrics: Dict[str, float]

class RiskModel:
    """
    Factor-based risk model with scenario analysis and stress testing
    """
    
    def __init__(self):
        # Risk factors
        self.risk_factors = {
            "equity_market": RiskFactor("equity_market", 0.0, 0.15, 0.1, 0.08),
            "interest_rate": RiskFactor("interest_rate", 0.025, 0.02, 0.05, 0.03),
            "credit_spread": RiskFactor("credit_spread", 0.01, 0.005, 0.2, 0.008),
            "fx_usd": RiskFactor("fx_usd", 1.0, 0.12, 0.15, 1.0),
            "volatility": RiskFactor("volatility", 0.2, 0.05, 0.3, 0.18),
            "liquidity": RiskFactor("liquidity", 1.0, 0.1, 0.5, 1.0)
        }
        
        # Factor correlation matrix
        self.factor_correlation = np.array([
            [1.00, -0.30, 0.60, -0.20, 0.80, -0.40],  # equity_market
            [-0.30, 1.00, -0.40, 0.10, -0.25, 0.15],  # interest_rate
            [0.60, -0.40, 1.00, -0.15, 0.50, -0.30],  # credit_spread
            [-0.20, 0.10, -0.15, 1.00, 0.10, 0.20],   # fx_usd
            [0.80, -0.25, 0.50, 0.10, 1.00, -0.50],   # volatility
            [-0.40, 0.15, -0.30, 0.20, -0.50, 1.00]   # liquidity
        ])
        
        # Factor volatility matrix
        factor_vols = [f.volatility for f in self.risk_factors.values()]
        self.factor_covariance = np.outer(factor_vols, factor_vols) * self.factor_correlation
        
        # Stress scenarios
        self.stress_scenarios = self._initialize_stress_scenarios()
        
        # Threading
        self.lock = threading.Lock()
        
    def _initialize_stress_scenarios(self) -> Dict[StressScenario, Dict[str, float]]:
        """Initialize predefined stress scenarios"""
        scenarios = {
            StressScenario.MARKET_CRASH_2008: {
                "equity_market": -0.35,      # 35% market drop
                "credit_spread": 0.008,      # 800 bps widening
                "volatility": 0.20,          # Vol spike to 40%
                "interest_rate": -0.015,     # 150 bps rate cut
                "liquidity": -0.50,          # 50% liquidity reduction
                "fx_usd": 0.10               # 10% USD strengthening
            },
            StressScenario.COVID_CRASH_2020: {
                "equity_market": -0.30,      # 30% market drop
                "credit_spread": 0.005,      # 500 bps widening
                "volatility": 0.25,          # Vol spike to 45%
                "interest_rate": -0.020,     # 200 bps rate cut
                "liquidity": -0.40,          # 40% liquidity reduction
                "fx_usd": 0.05               # 5% USD strengthening
            },
            StressScenario.FLASH_CRASH_2010: {
                "equity_market": -0.10,      # 10% sudden drop
                "volatility": 0.30,          # Extreme vol spike
                "liquidity": -0.80,          # 80% liquidity evaporation
                "credit_spread": 0.002,      # 200 bps widening
                "interest_rate": 0.0,        # No rate change
                "fx_usd": 0.02               # 2% USD move
            },
            StressScenario.RATE_SHOCK: {
                "interest_rate": 0.030,      # 300 bps rate increase
                "equity_market": -0.15,      # 15% equity decline
                "credit_spread": 0.004,      # 400 bps credit widening
                "volatility": 0.10,          # Moderate vol increase
                "liquidity": -0.20,          # 20% liquidity reduction
                "fx_usd": 0.15               # 15% USD strengthening
            },
            StressScenario.CREDIT_CRISIS: {
                "credit_spread": 0.012,      # 1200 bps credit blowout
                "equity_market": -0.25,      # 25% equity decline
                "volatility": 0.15,          # High volatility
                "interest_rate": -0.010,     # 100 bps rate cut
                "liquidity": -0.60,          # 60% liquidity reduction
                "fx_usd": 0.08               # 8% USD strengthening
            },
            StressScenario.LIQUIDITY_CRISIS: {
                "liquidity": -0.90,          # 90% liquidity evaporation
                "volatility": 0.35,          # Extreme volatility
                "equity_market": -0.20,      # 20% market decline
                "credit_spread": 0.006,      # 600 bps credit widening
                "interest_rate": -0.005,     # 50 bps rate cut
                "fx_usd": 0.05               # 5% USD move
            }
        }
        return scenarios
    
    def calculate_portfolio_risk(self, positions: Dict[str, float], 
                               factor_loadings: Dict[str, Dict[str, float]],
                               confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics using factor model
        
        Args:
            positions: Dict of {symbol: weight}
            factor_loadings: Dict of {symbol: {factor: loading}}
            confidence_level: VaR confidence level
        """
        with self.lock:
            try:
                # Build factor loading matrix
                symbols = list(positions.keys())
                num_symbols = len(symbols)
                num_factors = len(self.risk_factors)
                
                if num_symbols == 0:
                    return {"var": 0.0, "cvar": 0.0, "volatility": 0.0}
                
                # Portfolio weights vector
                weights = np.array([positions[symbol] for symbol in symbols])
                
                # Factor loading matrix (assets x factors)
                loading_matrix = np.zeros((num_symbols, num_factors))
                factor_names = list(self.risk_factors.keys())
                
                for i, symbol in enumerate(symbols):
                    symbol_loadings = factor_loadings.get(symbol, {})
                    for j, factor_name in enumerate(factor_names):
                        loading_matrix[i, j] = symbol_loadings.get(factor_name, 0.0)
                
                # Portfolio factor exposures
                portfolio_exposures = weights.T @ loading_matrix
                
                # Portfolio variance from factor model
                factor_risk = portfolio_exposures.T @ self.factor_covariance @ portfolio_exposures
                
                # Idiosyncratic risk (simplified)
                idiosyncratic_variance = 0.04  # 20% idiosyncratic vol
                total_idiosyncratic = np.sum(weights**2) * idiosyncratic_variance
                
                # Total portfolio variance
                portfolio_variance = factor_risk + total_idiosyncratic
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # VaR calculation (assuming normal distribution)
                z_score = -1.645 if confidence_level == 0.95 else -2.326  # 95% or 99%
                portfolio_var = -z_score * portfolio_volatility
                
                # CVaR (Expected Shortfall)
                portfolio_cvar = portfolio_volatility * np.exp(-0.5 * z_score**2) / (np.sqrt(2 * np.pi) * (1 - confidence_level))
                
                return {
                    "volatility": portfolio_volatility,
                    "var": portfolio_var,
                    "cvar": portfolio_cvar,
                    "factor_risk": factor_risk,
                    "idiosyncratic_risk": total_idiosyncratic,
                    "total_risk": portfolio_variance
                }
                
            except Exception as e:
                # Return conservative estimates if calculation fails
                return {
                    "volatility": 0.20,
                    "var": 0.10,
                    "cvar": 0.12,
                    "factor_risk": 0.03,
                    "idiosyncratic_risk": 0.01,
                    "total_risk": 0.04
                }
    
    def run_stress_test(self, positions: Dict[str, float],
                       factor_loadings: Dict[str, Dict[str, float]],
                       scenario: StressScenario,
                       portfolio_value: float) -> StressTestResult:
        """
        Run stress test scenario on portfolio
        """
        scenario_shocks = self.stress_scenarios[scenario]
        
        # Calculate portfolio PnL under stress
        total_pnl = 0.0
        position_pnls = {}
        
        symbols = list(positions.keys())
        
        for symbol in symbols:
            symbol_loadings = factor_loadings.get(symbol, {})
            position_weight = positions[symbol]
            position_value = position_weight * portfolio_value
            
            # Calculate symbol PnL from factor shocks
            symbol_pnl = 0.0
            for factor_name, shock in scenario_shocks.items():
                factor_loading = symbol_loadings.get(factor_name, 0.0)
                symbol_pnl += factor_loading * shock
            
            # Convert to dollar PnL
            position_dollar_pnl = symbol_pnl * position_value
            position_pnls[symbol] = position_dollar_pnl
            total_pnl += position_dollar_pnl
        
        # Portfolio-level statistics
        portfolio_pnl_pct = total_pnl / portfolio_value if portfolio_value > 0 else 0.0
        
        # Find worst performing positions
        worst_positions = sorted(position_pnls.items(), key=lambda x: x[1])[:5]
        
        # Estimate recovery time (simplified)
        recovery_days = self._estimate_recovery_time(scenario, abs(portfolio_pnl_pct))
        
        # Calculate stressed risk metrics
        stressed_vol = self._calculate_stressed_volatility(scenario, factor_loadings, positions)
        
        risk_metrics = {
            "stressed_volatility": stressed_vol,
            "var_stressed": stressed_vol * 1.645,  # 95% VaR
            "max_leverage_allowed": 0.5 if abs(portfolio_pnl_pct) > 0.2 else 1.0
        }
        
        return StressTestResult(
            scenario=scenario,
            portfolio_pnl=total_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            max_loss=min(position_pnls.values()) if position_pnls else 0.0,
            time_to_recovery_days=recovery_days,
            worst_positions=worst_positions,
            risk_metrics=risk_metrics
        )
    
    def _estimate_recovery_time(self, scenario: StressScenario, loss_magnitude: float) -> int:
        """Estimate time to recover from stress scenario"""
        base_recovery_days = {
            StressScenario.MARKET_CRASH_2008: 1200,  # ~3.3 years
            StressScenario.COVID_CRASH_2020: 180,    # ~6 months
            StressScenario.FLASH_CRASH_2010: 30,     # ~1 month
            StressScenario.RATE_SHOCK: 360,          # ~1 year
            StressScenario.CREDIT_CRISIS: 900,       # ~2.5 years
            StressScenario.LIQUIDITY_CRISIS: 90      # ~3 months
        }
        
        base_days = base_recovery_days.get(scenario, 365)
        
        # Scale by loss magnitude
        recovery_multiplier = min(3.0, max(0.5, loss_magnitude / 0.2))
        
        return int(base_days * recovery_multiplier)
    
    def _calculate_stressed_volatility(self, scenario: StressScenario,
                                     factor_loadings: Dict[str, Dict[str, float]],
                                     positions: Dict[str, float]) -> float:
        """Calculate portfolio volatility under stress"""
        scenario_shocks = self.stress_scenarios[scenario]
        
        # Increase correlations during stress
        stress_correlation_multiplier = {
            StressScenario.MARKET_CRASH_2008: 1.5,
            StressScenario.COVID_CRASH_2020: 1.4,
            StressScenario.FLASH_CRASH_2010: 2.0,
            StressScenario.RATE_SHOCK: 1.2,
            StressScenario.CREDIT_CRISIS: 1.6,
            StressScenario.LIQUIDITY_CRISIS: 1.8
        }
        
        multiplier = stress_correlation_multiplier.get(scenario, 1.3)
        
        # Scale factor volatilities
        stressed_factor_vols = []
        for factor_name, factor in self.risk_factors.items():
            shock_magnitude = abs(scenario_shocks.get(factor_name, 0.0))
            vol_multiplier = 1.0 + shock_magnitude * 2.0  # Vol increases with shock size
            stressed_vol = factor.volatility * vol_multiplier
            stressed_factor_vols.append(stressed_vol)
        
        # Stressed covariance matrix
        stressed_covariance = np.outer(stressed_factor_vols, stressed_factor_vols) * self.factor_correlation * multiplier
        
        # Calculate portfolio volatility with stressed covariance
        symbols = list(positions.keys())
        weights = np.array([positions[symbol] for symbol in symbols])
        
        if len(symbols) == 0:
            return 0.30  # Default high stress volatility
        
        # Build loading matrix
        loading_matrix = np.zeros((len(symbols), len(self.risk_factors)))
        factor_names = list(self.risk_factors.keys())
        
        for i, symbol in enumerate(symbols):
            symbol_loadings = factor_loadings.get(symbol, {})
            for j, factor_name in enumerate(factor_names):
                loading_matrix[i, j] = symbol_loadings.get(factor_name, 0.0)
        
        # Portfolio exposures
        portfolio_exposures = weights.T @ loading_matrix
        
        # Stressed portfolio variance
        factor_risk = portfolio_exposures.T @ stressed_covariance @ portfolio_exposures
        idiosyncratic_risk = np.sum(weights**2) * 0.09  # Higher idiosyncratic risk during stress
        
        total_variance = factor_risk + idiosyncratic_risk
        return np.sqrt(total_variance)
    
    def monte_carlo_simulation(self, positions: Dict[str, float],
                             factor_loadings: Dict[str, Dict[str, float]],
                             num_simulations: int = 10000,
                             time_horizon_days: int = 1) -> Dict[str, any]:
        """
        Monte Carlo simulation for portfolio risk
        """
        np.random.seed(42)  # For reproducibility
        
        symbols = list(positions.keys())
        weights = np.array([positions[symbol] for symbol in symbols])
        
        if len(symbols) == 0:
            return {"var_95": 0.0, "cvar_95": 0.0, "max_loss": 0.0, "percentiles": []}
        
        # Generate factor returns
        factor_returns = np.random.multivariate_normal(
            mean=np.zeros(len(self.risk_factors)),
            cov=self.factor_covariance * time_horizon_days,
            size=num_simulations
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = []
        
        for sim in range(num_simulations):
            portfolio_return = 0.0
            
            for i, symbol in enumerate(symbols):
                symbol_loadings = factor_loadings.get(symbol, {})
                symbol_return = 0.0
                
                # Factor contribution
                for j, factor_name in enumerate(self.risk_factors.keys()):
                    factor_loading = symbol_loadings.get(factor_name, 0.0)
                    symbol_return += factor_loading * factor_returns[sim, j]
                
                # Idiosyncratic return
                idiosyncratic_return = np.random.normal(0, 0.20 * np.sqrt(time_horizon_days))
                symbol_return += idiosyncratic_return
                
                # Weight by position size
                portfolio_return += weights[i] * symbol_return
            
            portfolio_returns.append(portfolio_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate risk metrics
        var_95 = -np.percentile(portfolio_returns, 5)
        var_99 = -np.percentile(portfolio_returns, 1)
        
        # CVaR (Expected Shortfall)
        tail_returns = portfolio_returns[portfolio_returns <= -var_95]
        cvar_95 = -np.mean(tail_returns) if len(tail_returns) > 0 else var_95
        
        max_loss = -np.min(portfolio_returns)
        
        percentiles = {
            "p01": np.percentile(portfolio_returns, 1),
            "p05": np.percentile(portfolio_returns, 5),
            "p10": np.percentile(portfolio_returns, 10),
            "p50": np.percentile(portfolio_returns, 50),
            "p90": np.percentile(portfolio_returns, 90),
            "p95": np.percentile(portfolio_returns, 95),
            "p99": np.percentile(portfolio_returns, 99)
        }
        
        return {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "max_loss": max_loss,
            "mean_return": np.mean(portfolio_returns),
            "volatility": np.std(portfolio_returns),
            "percentiles": percentiles,
            "num_simulations": num_simulations
        }
    
    def update_risk_factors(self, factor_updates: Dict[str, float]):
        """Update risk factor values (for real-time operation)"""
        with self.lock:
            for factor_name, new_value in factor_updates.items():
                if factor_name in self.risk_factors:
                    self.risk_factors[factor_name].current_value = new_value
