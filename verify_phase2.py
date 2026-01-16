
from monte_carlo import MonteCarloSimulator, PortfolioETF, Contribution
import numpy as np

def verify():
    # Setup Portfolio
    portfolio = [PortfolioETF("TSLA", 100, 0.15, 0.3)] # 15% return, 30% vol
    contributions = [Contribution("Test", 1000)]
    
    # 1. Test Costs
    print("Running Simulation with Costs €1200/yr...")
    sim_costs = MonteCarloSimulator(portfolio, contributions, n_simulations=100)
    res_costs = sim_costs.simulate(
        start_year=2026, end_year=2030,
        starting_capital=10000,
        annual_costs=1200 # €100/mo
    )
    
    # Check impact
    # Contribution €12000/yr. Costs €1200/yr. Net ~ €10800 added.
    # Start €10000. 5 years.
    # End approx 10k + 5*10.8k + returns.
    print(f"Mean End Balance (Costs): €{res_costs.mean[-1]:,.0f}")
    
    # 2. Test Withdrawals (Loan)
    print("\nRunning Simulation with Withdrawal (Loan)...")
    sim_wd = MonteCarloSimulator(portfolio, contributions, n_simulations=100)
    res_wd = sim_wd.simulate(
        start_year=2026, end_year=2030,
        starting_capital=100000,
        withdrawal_rate=0.05, # 5%
        withdrawal_start_year=2026,
        withdrawal_mode='loan'
    )
    
    payout_p50 = res_wd.payouts_p50
    print(f"P50 Payouts: {[round(x) for x in payout_p50]}")
    # Should be approx 5% of balance.
    # 5% of 100k = 5000.
    
    # 3. Test Withdrawals (Dividend)
    print("\nRunning Simulation with Withdrawal (Dividend 22/78)...")
    res_div = sim_wd.simulate(
        start_year=2026, end_year=2030,
        starting_capital=100000,
        withdrawal_rate=0.05,
        withdrawal_start_year=2026,
        withdrawal_mode='dividend'
    )
    payout_div_p50 = res_div.payouts_p50
    print(f"P50 Payouts (Div): {[round(x) for x in payout_div_p50]}")
    
    # Dividend payout should be roughly 78% of Loan payout
    ratio = payout_div_p50[0] / payout_p50[0]
    print(f"Dividend/Loan Ratio (First Year): {ratio:.2f} (Expected ~0.78)")
    
    if 0.77 < ratio < 0.79:
        print("SUCCESS: Tax logic verified.")
    else:
        print("FAILURE: Tax logic mismatch.")

if __name__ == "__main__":
    verify()
