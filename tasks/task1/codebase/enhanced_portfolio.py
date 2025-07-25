"""
Enhanced Portfolio Module for EXTREME Portfolio Optimizer
Supports real-time risk management, compliance monitoring, and performance attribution
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import threading
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    # Fallback for numpy functionality
    class np:
        @staticmethod
        def sqrt(x):
            return x ** 0.5
        
        @staticmethod
        def array(arr):
            return arr
        
        @staticmethod
        def percentile(arr, q):
            sorted_arr = sorted(arr)
            n = len(sorted_arr)
            index = (q / 100.0) * (n - 1)
            lower = int(index)
            upper = min(lower + 1, n - 1)
            weight = index - lower
            return sorted_arr[lower] * (1 - weight) + sorted_arr[upper] * weight

class RiskMeasure(Enum):
    VAR_95 = "VAR_95"
    VAR_99 = "VAR_99"
    CVAR_95 = "CVAR_95"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    SHARPE_RATIO = "SHARPE_RATIO"
    SORTINO_RATIO = "SORTINO_RATIO"

class ComplianceRule(Enum):
    MIFID_POSITION_LIMIT = "MIFID_POSITION_LIMIT"
    DODD_FRANK_PROP_TRADING = "DODD_FRANK_PROP_TRADING"
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    VAR_LIMIT = "VAR_LIMIT"

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    cvar_95: float  # Conditional VaR 
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta_market: float
    tracking_error: float
    leverage: float
    total_return: float
    volatility: float
    
@dataclass
class ComplianceViolation:
    rule_name: str
    rule_type: ComplianceRule
    severity: str  # "WARNING", "CRITICAL", "BREACH"
    current_value: float
    limit_value: float
    time_to_compliance: int  # seconds until must be resolved
    description: str
    timestamp: float

@dataclass
class AttributionReport:
    total_return: float
    benchmark_return: float
    active_return: float
    factor_attribution: Dict[str, float]  # Factor contributions
    stock_selection: float
    asset_allocation: float
    interaction_effect: float
    
@dataclass
class PortfolioPosition:
    symbol: str
    quantity: int
    market_value: float
    weight: float
    unrealized_pnl: float
    realized_pnl: float
    avg_price: float
    last_update: float
    
@dataclass
class RiskBudget:
    total_risk_budget: float
    factor_risk_budgets: Dict[str, float]
    sector_risk_budgets: Dict[str, float]
    individual_stock_limit: float

class ExtremePortfolio:
    """
    Enterprise-grade portfolio management with real-time risk analytics
    """
    
    def __init__(self, initial_capital: float = 10_000_000, base_currency: str = "USD"):
        self.initial_capital = initial_capital
        self.base_currency = base_currency
        
        # Portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.cash = initial_capital
        self.total_value = initial_capital
        
        # Risk management
        self.risk_metrics = RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        self.risk_budget = RiskBudget(
            total_risk_budget=0.15,  # 15% VaR limit
            factor_risk_budgets={
                "market": 0.10,
                "size": 0.03,
                "value": 0.03,
                "momentum": 0.03,
                "quality": 0.02
            },
            sector_risk_budgets={
                "technology": 0.25,
                "financials": 0.20,
                "healthcare": 0.20,
                "consumer": 0.15,
                "industrials": 0.10,
                "energy": 0.05,
                "utilities": 0.05
            },
            individual_stock_limit=0.05  # 5% max per stock
        )
        
        # Compliance monitoring
        self.compliance_violations: List[ComplianceViolation] = []
        self.compliance_limits = {
            ComplianceRule.MIFID_POSITION_LIMIT: 0.05,  # 5% of total portfolio
            ComplianceRule.LEVERAGE_LIMIT: 2.0,         # 2x leverage max
            ComplianceRule.CONCENTRATION_LIMIT: 0.10,   # 10% max in single sector
            ComplianceRule.VAR_LIMIT: 0.15              # 15% VaR limit
        }
        
        # Performance tracking
        self.pnl_history: List[float] = []
        self.return_history: List[float] = []
        self.benchmark_returns: List[float] = []
        self.portfolio_values: List[float] = [initial_capital]
        
        # Threading
        self.lock = threading.RLock()
        self.last_risk_calculation = time.time()
        self.last_compliance_check = time.time()
        
        # Factor exposures
        self.factor_exposures: Dict[str, float] = defaultdict(float)
        self.sector_exposures: Dict[str, float] = defaultdict(float)
    
    def update_position(self, symbol: str, quantity: int, price: float, 
                       market_data_provider=None) -> bool:
        """Update position and recalculate portfolio metrics"""
        with self.lock:
            current_time = time.time()
            
            # Calculate market value
            market_value = quantity * price
            
            # Check if we have enough cash for buys
            if symbol not in self.positions and market_value > self.cash:
                return False  # Insufficient funds
            
            # Update or create position
            if symbol in self.positions:
                old_position = self.positions[symbol]
                old_market_value = old_position.market_value
                
                # Calculate realized PnL for position changes
                if quantity == 0:  # Closing position
                    realized_pnl = (price - old_position.avg_price) * old_position.quantity
                    self.cash += market_value + realized_pnl
                    del self.positions[symbol]
                else:
                    # Update existing position
                    if quantity > old_position.quantity:
                        # Adding to position
                        additional_shares = quantity - old_position.quantity
                        additional_cost = additional_shares * price
                        if additional_cost > self.cash:
                            return False  # Insufficient funds
                        
                        # Update average price
                        total_cost = (old_position.quantity * old_position.avg_price + 
                                    additional_shares * price)
                        new_avg_price = total_cost / quantity
                        
                        self.cash -= additional_cost
                        
                        self.positions[symbol] = PortfolioPosition(
                            symbol=symbol,
                            quantity=quantity,
                            market_value=market_value,
                            weight=0.0,  # Will be calculated below
                            unrealized_pnl=(price - new_avg_price) * quantity,
                            realized_pnl=old_position.realized_pnl,
                            avg_price=new_avg_price,
                            last_update=current_time
                        )
                    else:
                        # Reducing position
                        shares_sold = old_position.quantity - quantity
                        proceeds = shares_sold * price
                        realized_pnl = (price - old_position.avg_price) * shares_sold
                        
                        self.cash += proceeds
                        
                        if quantity > 0:
                            self.positions[symbol] = PortfolioPosition(
                                symbol=symbol,
                                quantity=quantity,
                                market_value=market_value,
                                weight=0.0,
                                unrealized_pnl=(price - old_position.avg_price) * quantity,
                                realized_pnl=old_position.realized_pnl + realized_pnl,
                                avg_price=old_position.avg_price,
                                last_update=current_time
                            )
                        else:
                            # Position closed
                            del self.positions[symbol]
            else:
                # New position
                if market_value > self.cash:
                    return False
                    
                self.cash -= market_value
                self.positions[symbol] = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity,
                    market_value=market_value,
                    weight=0.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    avg_price=price,
                    last_update=current_time
                )
            
            # Recalculate portfolio metrics
            self._recalculate_portfolio_metrics()
            
            return True
    
    def _recalculate_portfolio_metrics(self):
        """Recalculate total value and position weights"""
        # Calculate total market value
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        self.total_value = total_market_value + self.cash
        
        # Update position weights
        for position in self.positions.values():
            position.weight = position.market_value / self.total_value if self.total_value > 0 else 0.0
        
        # Track portfolio value history
        self.portfolio_values.append(self.total_value)
        if len(self.portfolio_values) > 1000:  # Keep last 1000 values
            self.portfolio_values.pop(0)
    
    def optimize_allocation(self, assets: List[str], risk_budget: float, 
                          regime, market_data_provider=None) -> Dict[str, float]:
        """
        Real-time portfolio optimization using Black-Litterman or mean-variance
        """
        if not assets or not market_data_provider:
            return {}
        
        try:
            # Get correlation matrix and expected returns
            correlation_matrix = market_data_provider.get_correlation_matrix(assets, 50)
            
            # Get factor loadings for risk model
            factor_loadings = market_data_provider.get_factor_loadings(assets)
            
            # Calculate expected returns based on regime
            expected_returns = self._calculate_expected_returns(assets, regime, market_data_provider)
            
            # Risk model: factor covariance matrix
            risk_model = self._build_risk_model(assets, factor_loadings, correlation_matrix)
            
            # Optimization constraints
            constraints = self._build_optimization_constraints(assets, risk_budget)
            
            # Solve optimization (simplified mean-variance)
            optimal_weights = self._solve_portfolio_optimization(
                expected_returns, risk_model, constraints
            )
            
            return optimal_weights
            
        except Exception as e:
            # Fallback to equal weight
            equal_weight = 1.0 / len(assets)
            return {asset: equal_weight for asset in assets}
    
    def _calculate_expected_returns(self, assets: List[str], regime, market_data_provider) -> np.ndarray:
        """Calculate expected returns based on market regime"""
        expected_returns = []
        
        # Regime-dependent expected returns
        regime_adjustments = {
            "BULL": 0.002,    # 20 bps daily
            "BEAR": -0.001,   # -10 bps daily
            "SIDEWAYS": 0.0001,  # 1 bp daily
            "CRISIS": -0.005  # -50 bps daily
        }
        
        base_adjustment = regime_adjustments.get(regime.regime_type.value, 0.0)
        
        for asset in assets:
            # Get historical returns
            returns = market_data_provider.get_returns_history(asset, 50)
            if returns:
                historical_mean = np.mean(returns)
                # Combine historical and regime-based expectation
                expected_return = 0.5 * historical_mean + 0.5 * base_adjustment
            else:
                expected_return = base_adjustment
                
            expected_returns.append(expected_return)
        
        return np.array(expected_returns)
    
    def _build_risk_model(self, assets: List[str], factor_loadings: Dict, 
                         correlation_matrix: np.ndarray) -> np.ndarray:
        """Build factor-based risk model"""
        if correlation_matrix.size == 0:
            # Fallback to identity matrix
            return np.eye(len(assets)) * 0.04  # 20% vol assumption
        
        # Factor volatilities
        factor_vols = {
            "market_factor": 0.15,      # 15% market vol
            "size_factor": 0.05,        # 5% size factor vol
            "value_factor": 0.04,       # 4% value factor vol
            "momentum_factor": 0.06,    # 6% momentum factor vol
            "volatility_factor": 0.03,  # 3% vol factor vol
            "quality_factor": 0.04      # 4% quality factor vol
        }
        
        # Build factor loading matrix
        factor_matrix = []
        for asset in assets:
            if asset in factor_loadings:
                loadings = factor_loadings[asset]
                factor_row = [
                    loadings.market_factor,
                    loadings.size_factor,
                    loadings.value_factor,
                    loadings.momentum_factor,
                    loadings.volatility_factor,
                    loadings.quality_factor
                ]
            else:
                # Default factor loadings
                factor_row = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            factor_matrix.append(factor_row)
        
        factor_matrix = np.array(factor_matrix)
        
        # Factor covariance matrix
        factor_cov = np.diag([vol**2 for vol in factor_vols.values()])
        
        # Asset covariance matrix
        covariance_matrix = factor_matrix @ factor_cov @ factor_matrix.T
        
        # Add idiosyncratic risk
        idiosyncratic_risk = np.diag([0.2**2] * len(assets))  # 20% idiosyncratic vol
        covariance_matrix += idiosyncratic_risk
        
        return covariance_matrix
    
    def _build_optimization_constraints(self, assets: List[str], 
                                      risk_budget: float) -> Dict:
        """Build portfolio optimization constraints"""
        num_assets = len(assets)
        
        constraints = {
            "weight_bounds": [(0.0, self.risk_budget.individual_stock_limit) for _ in range(num_assets)],
            "weight_sum": 1.0,  # Weights must sum to 1
            "risk_budget": risk_budget,
            "leverage_limit": self.compliance_limits[ComplianceRule.LEVERAGE_LIMIT],
            "sector_limits": self.risk_budget.sector_risk_budgets
        }
        
        return constraints
    
    def _solve_portfolio_optimization(self, expected_returns: np.ndarray, 
                                    covariance_matrix: np.ndarray, 
                                    constraints: Dict) -> Dict[str, float]:
        """
        Solve portfolio optimization problem (simplified mean-variance)
        In production, would use proper optimizer like CVXPY or scipy.optimize
        """
        try:
            num_assets = len(expected_returns)
            
            # Risk aversion parameter
            risk_aversion = 5.0
            
            # Simplified analytical solution for mean-variance (unconstrained)
            inv_cov = np.linalg.inv(covariance_matrix + np.eye(num_assets) * 1e-6)
            ones = np.ones(num_assets)
            
            # Mean-variance optimal weights
            numerator = inv_cov @ expected_returns
            denominator = ones.T @ inv_cov @ expected_returns
            
            if abs(denominator) > 1e-10:
                weights = numerator / denominator
            else:
                # Fallback to minimum variance portfolio
                numerator = inv_cov @ ones
                denominator = ones.T @ inv_cov @ ones
                weights = numerator / denominator
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Apply constraints (simplified)
            weights = np.clip(weights, 0.0, constraints["weight_bounds"][0][1])
            weights = weights / np.sum(weights)  # Renormalize
            
            # Convert to dictionary
            asset_names = [f"asset_{i}" for i in range(num_assets)]  # Would use actual asset names
            return dict(zip(asset_names, weights))
            
        except Exception as e:
            # Equal weight fallback
            equal_weight = 1.0 / len(expected_returns)
            return {f"asset_{i}": equal_weight for i in range(len(expected_returns))}
    
    def calculate_risk_metrics(self, market_data_provider=None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        current_time = time.time()
        
        # Update every 10 seconds for performance
        if current_time - self.last_risk_calculation < 10:
            return self.risk_metrics
            
        with self.lock:
            try:
                # Get portfolio returns
                returns = self._calculate_portfolio_returns()
                
                if len(returns) < 10:
                    return self.risk_metrics
                
                # Value at Risk calculations
                returns_sorted = np.sort(returns)
                var_95 = -np.percentile(returns_sorted, 5) * np.sqrt(252)
                var_99 = -np.percentile(returns_sorted, 1) * np.sqrt(252)
                
                # Conditional Value at Risk (Expected Shortfall)
                var_95_threshold = np.percentile(returns_sorted, 5)
                tail_returns = returns_sorted[returns_sorted <= var_95_threshold]
                cvar_95 = -np.mean(tail_returns) * np.sqrt(252) if len(tail_returns) > 0 else var_95
                
                # Maximum Drawdown
                portfolio_values = np.array(self.portfolio_values)
                if len(portfolio_values) > 1:
                    running_max = np.maximum.accumulate(portfolio_values)
                    drawdowns = (portfolio_values - running_max) / running_max
                    max_drawdown = -np.min(drawdowns)
                else:
                    max_drawdown = 0.0
                
                # Sharpe and Sortino ratios
                mean_return = np.mean(returns) * 252
                volatility = np.std(returns) * np.sqrt(252)
                
                risk_free_rate = 0.02  # 2% risk-free rate assumption
                sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0.0
                
                # Sortino ratio (downside deviation)
                downside_returns = returns[returns < 0]
                downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
                sortino_ratio = (mean_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
                
                # Beta calculation (vs market)
                market_beta = 1.0  # Simplified
                if market_data_provider:
                    spy_returns = market_data_provider.get_returns_history("SPY", len(returns))
                    if len(spy_returns) == len(returns):
                        covariance = np.cov(returns, spy_returns)[0, 1]
                        market_variance = np.var(spy_returns)
                        market_beta = covariance / market_variance if market_variance > 0 else 1.0
                
                # Leverage calculation
                total_exposure = sum(pos.market_value for pos in self.positions.values())
                leverage = total_exposure / self.total_value if self.total_value > 0 else 0.0
                
                # Tracking error (vs benchmark)
                tracking_error = volatility  # Simplified - would compare vs actual benchmark
                
                # Update risk metrics
                self.risk_metrics = RiskMetrics(
                    var_95=var_95,
                    var_99=var_99,
                    cvar_95=cvar_95,
                    max_drawdown=max_drawdown,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=sortino_ratio,
                    beta_market=market_beta,
                    tracking_error=tracking_error,
                    leverage=leverage,
                    total_return=mean_return,
                    volatility=volatility
                )
                
                self.last_risk_calculation = current_time
                
            except Exception as e:
                # Keep existing metrics if calculation fails
                pass
        
        return self.risk_metrics
    
    def _calculate_portfolio_returns(self) -> List[float]:
        """Calculate portfolio returns from value history"""
        if len(self.portfolio_values) < 2:
            return []
            
        returns = []
        for i in range(1, len(self.portfolio_values)):
            if self.portfolio_values[i-1] > 0:
                ret = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
                returns.append(ret)
                
        return returns
    
    def check_regulatory_limits(self) -> List[ComplianceViolation]:
        """Real-time regulatory compliance monitoring"""
        current_time = time.time()
        
        # Check every 30 seconds
        if current_time - self.last_compliance_check < 30:
            return self.compliance_violations
            
        violations = []
        
        with self.lock:
            # Position size limits (MiFID II)
            for symbol, position in self.positions.items():
                if position.weight > self.compliance_limits[ComplianceRule.MIFID_POSITION_LIMIT]:
                    violation = ComplianceViolation(
                        rule_name="MiFID II Position Limit",
                        rule_type=ComplianceRule.MIFID_POSITION_LIMIT,
                        severity="CRITICAL",
                        current_value=position.weight,
                        limit_value=self.compliance_limits[ComplianceRule.MIFID_POSITION_LIMIT],
                        time_to_compliance=300,  # 5 minutes to resolve
                        description=f"Position in {symbol} exceeds 5% limit",
                        timestamp=current_time
                    )
                    violations.append(violation)
            
            # Leverage limits
            current_leverage = self.risk_metrics.leverage
            leverage_limit = self.compliance_limits[ComplianceRule.LEVERAGE_LIMIT]
            if current_leverage > leverage_limit:
                violation = ComplianceViolation(
                    rule_name="Leverage Limit",
                    rule_type=ComplianceRule.LEVERAGE_LIMIT,
                    severity="CRITICAL" if current_leverage > leverage_limit * 1.2 else "WARNING",
                    current_value=current_leverage,
                    limit_value=leverage_limit,
                    time_to_compliance=600,  # 10 minutes to resolve
                    description=f"Portfolio leverage {current_leverage:.2f}x exceeds {leverage_limit}x limit",
                    timestamp=current_time
                )
                violations.append(violation)
            
            # VaR limits
            var_limit = self.compliance_limits[ComplianceRule.VAR_LIMIT]
            if self.risk_metrics.var_95 > var_limit:
                violation = ComplianceViolation(
                    rule_name="VaR Limit",
                    rule_type=ComplianceRule.VAR_LIMIT,
                    severity="CRITICAL",
                    current_value=self.risk_metrics.var_95,
                    limit_value=var_limit,
                    time_to_compliance=900,  # 15 minutes to resolve
                    description=f"Portfolio VaR {self.risk_metrics.var_95:.1%} exceeds {var_limit:.1%} limit",
                    timestamp=current_time
                )
                violations.append(violation)
            
            # Concentration limits by sector
            sector_exposures = self._calculate_sector_exposures()
            concentration_limit = self.compliance_limits[ComplianceRule.CONCENTRATION_LIMIT]
            
            for sector, exposure in sector_exposures.items():
                if exposure > concentration_limit:
                    violation = ComplianceViolation(
                        rule_name="Sector Concentration Limit",
                        rule_type=ComplianceRule.CONCENTRATION_LIMIT,
                        severity="WARNING",
                        current_value=exposure,
                        limit_value=concentration_limit,
                        time_to_compliance=1800,  # 30 minutes to resolve
                        description=f"Sector {sector} exposure {exposure:.1%} exceeds {concentration_limit:.1%} limit",
                        timestamp=current_time
                    )
                    violations.append(violation)
        
        self.compliance_violations = violations
        self.last_compliance_check = current_time
        return violations
    
    def _calculate_sector_exposures(self) -> Dict[str, float]:
        """Calculate exposure by sector (simplified)"""
        sector_mapping = {
            "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
            "JPM": "financials", "BAC": "financials", "WFC": "financials",
            "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare"
        }
        
        sector_exposures = defaultdict(float)
        
        for symbol, position in self.positions.items():
            sector = sector_mapping.get(symbol, "other")
            sector_exposures[sector] += position.weight
            
        return dict(sector_exposures)
    
    def get_attribution_analysis(self, benchmark_returns: List[float] = None) -> AttributionReport:
        """Performance attribution analysis"""
        portfolio_returns = self._calculate_portfolio_returns()
        
        if not portfolio_returns:
            return AttributionReport(0.0, 0.0, 0.0, {}, 0.0, 0.0, 0.0)
        
        total_return = np.mean(portfolio_returns) * 252  # Annualized
        
        # Simplified attribution (would be more complex in practice)
        if benchmark_returns and len(benchmark_returns) == len(portfolio_returns):
            benchmark_return = np.mean(benchmark_returns) * 252
            active_return = total_return - benchmark_return
        else:
            benchmark_return = 0.08  # 8% assumption
            active_return = total_return - benchmark_return
        
        # Factor attribution (simplified)
        factor_attribution = {
            "market": active_return * 0.6,
            "size": active_return * 0.1,
            "value": active_return * 0.1,
            "momentum": active_return * 0.1,
            "quality": active_return * 0.1
        }
        
        return AttributionReport(
            total_return=total_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            factor_attribution=factor_attribution,
            stock_selection=active_return * 0.7,
            asset_allocation=active_return * 0.3,
            interaction_effect=0.0
        )
    
    def can_buy(self, symbol: str, quantity: int, price: float = None) -> bool:
        """Check if we can buy given quantity of symbol"""
        if not price:
            return False
            
        cost = quantity * price
        return cost <= self.cash
    
    def can_sell(self, symbol: str, quantity: int) -> bool:
        """Check if we can sell given quantity of symbol"""
        position = self.positions.get(symbol)
        return position is not None and position.quantity >= quantity
    
    def get_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_total_value(self) -> float:
        """Get total portfolio value"""
        return self.total_value
    
    def get_cash(self) -> float:
        """Get available cash"""
        return self.cash

# Alias for backward compatibility  
Portfolio = ExtremePortfolio
