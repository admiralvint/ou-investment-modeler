"""
Monte Carlo Simulation Engine for Investment Projections.

Uses historical return and volatility to simulate future outcomes.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results."""
    years: list[int]
    p2: list[float]   # 2nd percentile (extreme pessimistic)
    p10: list[float]  # 10th percentile (pessimistic)
    p50: list[float]  # 50th percentile (median/expected)
    p90: list[float]  # 90th percentile (optimistic)
    p98: list[float]  # 98th percentile (extreme optimistic)
    mean: list[float]  # Mean across all simulations
    percentiles: dict[str, list[float]]  # Full percentile range p0-p100
    payouts_p50: list[float]  # Median annual payout
    all_paths: Optional[np.ndarray] = None  # Full simulation data if requested
    
    def to_dict(self, include_paths: bool = False) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'years': self.years,
            'p2': [round(v, 2) for v in self.p2],
            'p10': [round(v, 2) for v in self.p10],
            'p50': [round(v, 2) for v in self.p50],
            'p90': [round(v, 2) for v in self.p90],
            'p98': [round(v, 2) for v in self.p98],
            'mean': [round(v, 2) for v in self.mean],
            'payouts_p50': [round(v, 2) for v in self.payouts_p50],
            'percentiles': {k: [round(v, 2) for v in vals] for k, vals in self.percentiles.items()}
        }
        if include_paths and self.all_paths is not None:
            result['paths'] = self.all_paths.tolist()
        return result


@dataclass
class PortfolioETF:
    """Single ETF in a portfolio."""
    isin: str
    allocation: float  # Decimal (0.6 = 60%)
    annual_return: float  # Decimal
    annual_volatility: float  # Decimal


@dataclass
class RentalProperty:
    """Rental property configuration."""
    include: bool
    sell: bool
    sale_year: int
    mortgage_2026: float  # Starting mortgage balance
    monthly_payment: float  # Mortgage payment
    monthly_income: float  # Net rental income to OÜ
    interest_rate: float
    mart_share: float  # Mart's share of property value
    kerli_share: float  # Kerli's share


@dataclass 
class Contribution:
    """Monthly contribution from a person."""
    name: str
    monthly_amount: float


class MonteCarloSimulator:
    """Monte Carlo simulation engine for investment projections."""
    
    def __init__(
        self,
        portfolio: list[PortfolioETF],
        contributions: list[Contribution],
        rental: Optional[RentalProperty] = None,
        n_simulations: int = 10000,
        seed: Optional[int] = None
    ):
        """
        Initialize the simulator.
        
        Args:
            portfolio: List of ETFs with allocations
            contributions: Monthly contributions from each person
            rental: Optional rental property configuration
            n_simulations: Number of simulation paths
            seed: Random seed for reproducibility
        """
        self.portfolio = portfolio
        self.contributions = contributions
        self.rental = rental
        self.n_simulations = n_simulations
        
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate portfolio-level return and volatility
        self.portfolio_return = self._calculate_portfolio_return()
        self.portfolio_volatility = self._calculate_portfolio_volatility()
    
    def _calculate_portfolio_return(self) -> float:
        """Calculate weighted average return of portfolio."""
        total = sum(etf.allocation * etf.annual_return for etf in self.portfolio)
        return total
    
    def _calculate_portfolio_volatility(self) -> float:
        """
        Calculate portfolio volatility.
        
        Simplified: assumes no correlation between ETFs (conservative).
        """
        # Var(portfolio) = sum(w_i^2 * var_i) assuming 0 correlation
        variance = sum(
            (etf.allocation ** 2) * (etf.annual_volatility ** 2) 
            for etf in self.portfolio
        )
        return np.sqrt(variance)
    
    def _calculate_mortgage_balance(self, year: int) -> float:
        """Calculate remaining mortgage balance at start of year."""
        if not self.rental or not self.rental.include:
            return 0.0
        
        months = (year - 2026) * 12
        balance = self.rental.mortgage_2026
        monthly_rate = self.rental.interest_rate / 12
        
        for _ in range(months):
            if balance <= 0:
                break
            interest = balance * monthly_rate
            principal = self.rental.monthly_payment - interest
            balance -= principal
            balance = max(0, balance)
        
        return balance
    
    def simulate(
        self,
        start_year: int = 2026,
        start_month: int = 1,  # 1=January, 12=December
        end_year: int = 2040,
        starting_capital: float = 82964.0,
        starting_loans: Optional[dict[str, float]] = None,
        annual_costs: float = 0.0,
        withdrawal_rate: float = 0.0,  # 0.04 = 4%
        withdrawal_start_year: int = 2035,
        withdrawal_mode: str = 'loan',  # 'loan' or 'dividend'
        contribution_end_year: Optional[int] = None,
        contribution_change_year: Optional[int] = None,
        contribution_change_factor: float = 1.0
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            start_year: First year of simulation
            start_month: Month to start (1-12, default 1=January)
            end_year: Last year of simulation
            starting_capital: Initial capital in OÜ
            starting_loans: Initial loans per person
            
        Returns:
            SimulationResult with percentile paths
        """
        years = list(range(start_year, end_year + 1))
        n_years = len(years)
        
        # Initialize paths array: (n_simulations, n_years)
        # Store ending balance for each year
        paths = np.zeros((self.n_simulations, n_years))
        payouts_paths = np.zeros((self.n_simulations, n_years))
        
        # Annual contribution total
        annual_contribution = sum(c.monthly_amount * 12 for c in self.contributions)
        
        # Monthly volatility
        monthly_vol = self.portfolio_volatility / np.sqrt(12)
        monthly_return = self.portfolio_return / 12
        
        for sim in range(self.n_simulations):
            balance = starting_capital
            
            for year_idx, year in enumerate(years):
                # Monthly simulation for this year
                year_start_balance = balance
                
                # First year: start from start_month, otherwise full year
                first_month = (start_month - 1) if year == start_year else 0
                
                # Calculate annual withdrawal budget for this year
                year_withdrawal_budget = 0.0
                if year >= withdrawal_start_year and withdrawal_rate > 0:
                    year_withdrawal_budget = balance * withdrawal_rate
                
                year_payout_gross = 0.0
                

                for month in range(first_month, 12):
                    # Monthly contribution calculation
                    current_monthly_contrib = sum(c.monthly_amount for c in self.contributions)
                    
                    # Logic 1: Stop Contributions
                    if contribution_end_year is not None and year >= contribution_end_year:
                        current_monthly_contrib = 0.0
                    
                    # Logic 2: Change Contribution Rate
                    elif (contribution_change_year is not None and 
                          year >= contribution_change_year):
                        current_monthly_contrib *= contribution_change_factor
                    
                    balance += current_monthly_contrib
                    
                    # Deduct monthly share of annual costs
                    balance -= (annual_costs / 12)
                    
                    # Deduct monthly withdrawal
                    if year_withdrawal_budget > 0:
                        monthly_wd = year_withdrawal_budget / 12
                        balance -= monthly_wd
                        year_payout_gross += monthly_wd
                    
                    # Rental income logic (mid-year timing for sale year):
                    # If sell=True (OÜ pays mortgage): OÜ gets full rental income after repayment
                    # If sell=False (personal payment): OÜ gets nothing
                    if self.rental and self.rental.include and self.rental.sell:
                        if year > self.rental.sale_year:
                             # Full year after repayment - get all months
                            balance += self.rental.monthly_income
                        elif year == self.rental.sale_year and month >= 6:
                            # Sale year - rental income starts from July (month 6)
                            balance += self.rental.monthly_income
                    
                    # Mid-year mortgage deduction (July 1 = month 6)
                    if self.rental and self.rental.include and self.rental.sell:
                        if year == self.rental.sale_year and month == 6:
                            # Deduct mortgage from balance mid-year
                            balance -= (self.rental.mart_share + self.rental.kerli_share)
                    
                    # Investment return (with randomness)
                    random_return = np.random.normal(monthly_return, monthly_vol)
                    balance *= (1 + random_return)
                
                # Store ending balance
                paths[sim, year_idx] = balance
                
                # Store annual payout (Net)
                if withdrawal_mode == 'dividend':
                    # 22/78 rule: Net = Gross * 0.78 (approx)
                    # Actually if 22/78 is ratio, then Net = Gross * (78/100)
                    payouts_paths[sim, year_idx] = year_payout_gross * 0.78
                else:
                     # Loan repayment: Net = Gross
                    payouts_paths[sim, year_idx] = year_payout_gross
        
        # Calculate percentiles
        p2 = np.percentile(paths, 2, axis=0).tolist()
        p10 = np.percentile(paths, 10, axis=0).tolist()
        p50 = np.percentile(paths, 50, axis=0).tolist()
        p90 = np.percentile(paths, 90, axis=0).tolist()
        p98 = np.percentile(paths, 98, axis=0).tolist()
        mean = np.mean(paths, axis=0).tolist()
        
        payouts_p50 = np.percentile(payouts_paths, 50, axis=0).tolist()
        
        # Calculate full range of percentiles
        percentiles = {}
        # Calculate in steps of 1%
        for i in range(101):
            key = f'p{i}'
            values = np.percentile(paths, i, axis=0).tolist()
            percentiles[key] = values

        return SimulationResult(
            years=years,
            p2=p2,
            p10=p10,
            p50=p50,
            p90=p90,
            p98=p98,
            mean=mean,
            percentiles=percentiles,
            payouts_p50=payouts_p50,
            all_paths=paths
        )


def calculate_loan_evolution(
    starting_loans: dict[str, float],
    contributions: list[Contribution],
    years: list[int],
    rental: Optional[RentalProperty] = None,
    start_month: int = 1,
    payouts: list[float] = None,

    withdrawal_mode: str = 'dividend',
    contribution_end_year: Optional[int] = None,
    contribution_change_year: Optional[int] = None,
    contribution_change_factor: float = 1.0
) -> dict[str, list[float]]:
    """
    Calculate evolution of loans (principal + contributions).
    
    Args:
        starting_loans: Initial loan amounts
        contributions: Monthly contributions
        years: List of simulation years
        rental: Rental property config
        start_month: Start month (1-12) for the first year
        
    Returns:
        Dictionary of loan balances per person per year
    """
    # Map contributions to persons
    contrib_map = {c.name: c.monthly_amount for c in contributions}
    
    loan_evolution = {name: [] for name in starting_loans.keys()}
    
    # Initialize current loans
    current_loans = starting_loans.copy()
    
    for year_idx, year in enumerate(years):
        # Calculate contribution months for this year
        months = 12
        if year == years[0]:
            months = 12 - start_month + 1
        
        # 1. Add Contributions
        for person in current_loans:
            monthly = contrib_map.get(person, 0)
            
            # Logic: Stop/Change Contributions (Same as simulate)
            if contribution_end_year is not None and year >= contribution_end_year:
                monthly = 0.0
            elif (contribution_change_year is not None and 
                  year >= contribution_change_year):
                monthly *= contribution_change_factor
                
            current_loans[person] += monthly * months
            
        # 2. Subtract Rental Repayment (Mart/Kerli)
        if rental and rental.include and rental.sell and year == rental.sale_year:
            if 'Mart' in current_loans:
                current_loans['Mart'] -= rental.mart_share
            if 'Kerli' in current_loans:
                current_loans['Kerli'] -= rental.kerli_share
        
        # 3. Subtract Loan Repayment (Withdrawals) if mode='loan'
        if withdrawal_mode == 'loan' and payouts and year_idx < len(payouts):
            payout_amount = payouts[year_idx]
            if payout_amount > 0:
                total_loan = sum(current_loans.values())
                if total_loan > 0:
                    # Reduce properly proportionally
                    # If payout > total_loan, we reduce to 0 (excess is dividend logic, but here we just zero out)
                    factor = max(0.0, (total_loan - payout_amount) / total_loan)
                    for person in current_loans:
                        current_loans[person] *= factor
        
        # Store snapshot
        for person in current_loans:
            loan_evolution[person].append(current_loans[person])
            
    return loan_evolution


if __name__ == "__main__":
    # Test simulation with VWCE-like returns
    portfolio = [
        PortfolioETF(
            isin="IE00BK5BQT80",
            allocation=1.0,
            annual_return=0.085,  # 8.5%
            annual_volatility=0.15  # 15%
        )
    ]
    
    contributions = [
        Contribution("Mart", 850),
        Contribution("Kerli", 850),
        Contribution("Laps1", 150),
        Contribution("Laps2", 150),
    ]
    
    simulator = MonteCarloSimulator(
        portfolio=portfolio,
        contributions=contributions,
        n_simulations=10000,
        seed=42
    )
    
    result = simulator.simulate()
    
    print("Monte Carlo Simulation Results (2026-2040):")
    print("-" * 60)
    for i, year in enumerate(result.years):
        print(f"{year}: P10=€{result.p10[i]:,.0f}  P50=€{result.p50[i]:,.0f}  P90=€{result.p90[i]:,.0f}")
