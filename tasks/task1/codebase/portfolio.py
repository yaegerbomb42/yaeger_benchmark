from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class Portfolio:
    """Portfolio management and risk tracking."""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> quantity
        self.trade_history = []
        self.daily_pnl = 0.0
        self.max_position_size = 1000  # Max shares per symbol
        self.max_daily_loss = 5000.0   # Max daily loss ($)
        self.stop_loss_pct = 0.05      # 5% stop loss
        
    def get_buying_power(self) -> float:
        """Get available buying power."""
        return self.cash
    
    def get_position(self, symbol: str) -> int:
        """Get current position size for symbol."""
        return self.positions.get(symbol, 0)
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Get total portfolio value."""
        total_value = self.cash
        
        for symbol, quantity in self.positions.items():
            if symbol in current_prices and quantity != 0:
                total_value += current_prices[symbol] * quantity
                
        return total_value
    
    def can_buy(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if we can buy the specified quantity."""
        # Check cash availability
        total_cost = quantity * price
        if total_cost > self.cash:
            return False
        
        # Check position limits
        current_position = self.get_position(symbol)
        if current_position + quantity > self.max_position_size:
            return False
        
        return True
    
    def can_sell(self, symbol: str, quantity: int) -> bool:
        """Check if we can sell the specified quantity."""
        current_position = self.get_position(symbol)
        return current_position >= quantity
    
    def execute_trade(self, symbol: str, quantity: int, price: float, side: str) -> bool:
        """Execute a trade and update portfolio."""
        if side == "BUY":
            if not self.can_buy(symbol, quantity, price):
                return False
            
            total_cost = quantity * price
            self.cash -= total_cost
            current_position = self.get_position(symbol)
            self.positions[symbol] = current_position + quantity
            
        elif side == "SELL":
            if not self.can_sell(symbol, quantity):
                return False
            
            proceeds = quantity * price
            self.cash += proceeds
            current_position = self.get_position(symbol)
            new_position = current_position - quantity
            
            if new_position == 0:
                del self.positions[symbol]
            else:
                self.positions[symbol] = new_position
        
        # Record trade
        self.trade_history.append({
            'timestamp': time.time(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price
        })
        
        return True
    
    def calculate_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate unrealized and realized PnL."""
        unrealized_pnl = 0.0
        
        # Calculate unrealized PnL from current positions
        for symbol, quantity in self.positions.items():
            if symbol in current_prices and quantity != 0:
                current_value = current_prices[symbol] * quantity
                
                # Estimate cost basis from trade history
                cost_basis = self._get_cost_basis(symbol)
                unrealized_pnl += current_value - (cost_basis * abs(quantity))
        
        # Total PnL = cash + position values - initial cash
        total_value = self.get_total_value(current_prices)
        total_pnl = total_value - self.initial_cash
        realized_pnl = total_pnl - unrealized_pnl
        
        return {
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_pnl': total_pnl
        }
    
    def _get_cost_basis(self, symbol: str) -> float:
        """Estimate average cost basis for a symbol."""
        buy_trades = [t for t in self.trade_history 
                     if t['symbol'] == symbol and t['side'] == 'BUY']
        
        if not buy_trades:
            return 0.0
        
        total_cost = sum(t['quantity'] * t['price'] for t in buy_trades)
        total_quantity = sum(t['quantity'] for t in buy_trades)
        
        return total_cost / total_quantity if total_quantity > 0 else 0.0
    
    def check_risk_limits(self, current_prices: Dict[str, float]) -> List[str]:
        """Check if any risk limits are violated."""
        violations = []
        
        # Check daily loss limit
        pnl = self.calculate_pnl(current_prices)
        if pnl['total_pnl'] < -self.max_daily_loss:
            violations.append("Daily loss limit exceeded")
        
        # Check position sizes
        for symbol, quantity in self.positions.items():
            if abs(quantity) > self.max_position_size:
                violations.append(f"Position size limit exceeded for {symbol}")
        
        # Check stop losses
        for symbol, quantity in self.positions.items():
            if symbol in current_prices and quantity != 0:
                current_price = current_prices[symbol]
                cost_basis = self._get_cost_basis(symbol)
                
                if quantity > 0:  # Long position
                    loss_pct = (cost_basis - current_price) / cost_basis
                    if loss_pct > self.stop_loss_pct:
                        violations.append(f"Stop loss triggered for {symbol}")
                        
                else:  # Short position
                    loss_pct = (current_price - cost_basis) / cost_basis
                    if loss_pct > self.stop_loss_pct:
                        violations.append(f"Stop loss triggered for {symbol}")
        
        return violations
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get comprehensive portfolio summary."""
        pnl = self.calculate_pnl(current_prices)
        total_value = self.get_total_value(current_prices)
        
        return {
            'cash': self.cash,
            'total_value': total_value,
            'positions': self.positions.copy(),
            'pnl': pnl,
            'trade_count': len(self.trade_history),
            'risk_violations': self.check_risk_limits(current_prices)
        }
