"""
ETF Data Fetcher using Yahoo Finance.

Fetches historical data for ETFs by ISIN or ticker symbol.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ISIN to Yahoo Finance ticker mapping for common ETFs
ISIN_TO_TICKER = {
    'IE00BK5BQT80': 'VWCE.DE',  # Vanguard FTSE All-World UCITS ETF
    'IE00B4L5Y983': 'IWDA.AS',  # iShares Core MSCI World
    'IE00B3RBWM25': 'VUSA.L',   # Vanguard S&P 500
    'IE00BKX55T58': 'VWRL.L',   # Vanguard FTSE All-World (distributing)
    'LU0392494562': 'EXSA.DE', # iShares STOXX Europe 600
    'IE00B5BMR087': 'CSPX.L',   # iShares Core S&P 500
    'IE00BJ0KDQ92': 'XDWL.DE',  # Xtrackers MSCI World
    'IE00B3XXRP09': 'VUSA.AS',  # Vanguard S&P 500 (EUR)
}


@dataclass
class ETFData:
    """Container for ETF historical data and statistics."""
    isin: str
    ticker: str
    name: str
    annual_return: float  # Annualized return (decimal, e.g., 0.085 = 8.5%)
    annual_volatility: float  # Annualized std dev
    years_of_data: int
    last_price: float
    currency: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'isin': self.isin,
            'ticker': self.ticker,
            'name': self.name,
            'annual_return': round(self.annual_return * 100, 2),  # As percentage
            'annual_volatility': round(self.annual_volatility * 100, 2),
            'years_of_data': self.years_of_data,
            'years': self.years_of_data, # Alias for frontend
            'last_price': round(self.last_price, 2),
            'currency': self.currency
        }


def isin_to_ticker(isin: str) -> Optional[str]:
    """
    Convert ISIN to Yahoo Finance ticker.
    
    Args:
        isin: ISIN code (e.g., 'IE00BK5BQT80')
        
    Returns:
        Yahoo Finance ticker or None if not found.
    """
    isin = isin.upper().strip()
    return ISIN_TO_TICKER.get(isin)


@lru_cache(maxsize=20)
def fetch_etf_data(isin: str, years: int = 15) -> Optional[ETFData]:
    """
    Fetch ETF data from Yahoo Finance.
    
    Args:
        isin: ISIN code
        years: Years of historical data to fetch
        
    Returns:
        ETFData object or None if not found.
    """
    ticker_symbol = isin_to_ticker(isin)
    if not ticker_symbol:
        # Try using ISIN directly (sometimes works)
        ticker_symbol = isin
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get historical data
        hist = ticker.history(period=f"{years}y")
        if hist.empty or len(hist) < 252:  # Need at least 1 year of data
            return None
        
        # Calculate daily returns
        daily_returns = hist['Close'].pct_change().dropna()
        
        # Annualized return (geometric mean)
        total_return = (1 + daily_returns).prod()
        trading_days = len(daily_returns)
        years_actual = trading_days / 252
        annual_return = total_return ** (1 / years_actual) - 1
        
        # Annualized volatility
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        # Get ETF info
        info = ticker.info
        name = info.get('longName', info.get('shortName', ticker_symbol))
        currency = info.get('currency', 'EUR')
        last_price = hist['Close'].iloc[-1]
        
        return ETFData(
            isin=isin.upper(),
            ticker=ticker_symbol,
            name=name,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            years_of_data=int(years_actual),
            last_price=last_price,
            currency=currency
        )
        
    except Exception as e:
        print(f"Error fetching {isin}: {e}")
        return None


def validate_portfolio(etfs: list[tuple[str, float]]) -> tuple[bool, str]:
    """
    Validate a portfolio allocation.
    
    Args:
        etfs: List of (isin, allocation_percent) tuples
        
    Returns:
        (is_valid, error_message)
    """
    # Filter out empty entries
    etfs = [(isin, pct) for isin, pct in etfs if isin and isin.strip()]
    
    if not etfs:
        return False, "At least one ETF is required"
    
    total_allocation = sum(pct for _, pct in etfs)
    if abs(total_allocation - 100) > 0.1:
        return False, f"Allocations must sum to 100% (currently {total_allocation}%)"
    
    for isin, pct in etfs:
        if pct < 0 or pct > 100:
            return False, f"Invalid allocation for {isin}: {pct}%"
    
    return True, ""


if __name__ == "__main__":
    # Test with VWCE
    print("Testing VWCE (IE00BK5BQT80)...")
    data = fetch_etf_data("IE00BK5BQT80")
    if data:
        print(f"  Name: {data.name}")
        print(f"  Annual Return: {data.annual_return*100:.2f}%")
        print(f"  Annual Volatility: {data.annual_volatility*100:.2f}%")
        print(f"  Years of Data: {data.years_of_data}")
    else:
        print("  Failed to fetch data")
