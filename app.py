"""
Flask Web Application for OÜ Investment Modeler.

Generic version - all configuration values are user-inputable.
Ready for GitHub upload.
"""

import json
from flask import Flask, render_template, request, jsonify

from etf_fetcher import fetch_etf_data, validate_portfolio
from monte_carlo import (
    MonteCarloSimulator,
    PortfolioETF,
    Contribution,
    RentalProperty,
    calculate_loan_evolution
)


app = Flask(__name__)


# Default configuration (can be overridden by user input)
DEFAULT_CONFIG = {
    'contributions': {
        'Person1': 500,
        'Person2': 500,
        'Person3': 100,
        'Person4': 100
    },
    'starting_capital': 50000.0,
    'starting_loans': {
        'Person1': 10000.0,
        'Person2': 10000.0,
        'Person3': 5000.0,
        'Person4': 5000.0
    },
    'rental': {
        'include': False,
        'mortgage_2026': 100000.0,
        'monthly_payment': 500.0,
        'monthly_income': 800.0,
        'interest_rate': 0.04,
        'person1_share': 50000.0,
        'person2_share': 50000.0
    },
    'etfs': [
        {'isin': 'IE00BK5BQT80', 'allocation': 100}  # VWCE
    ],
    'start_year': 2026,
    'end_year': 2040,
    'repay_mortgage': False,
    'repayment_year': 2030
}


@app.route('/')
def index():
    """Render the main UI."""
    return render_template('index.html', config=DEFAULT_CONFIG)


@app.route('/api/etf/<isin>')
def get_etf_info(isin: str):
    """Fetch ETF information by ISIN."""
    data = fetch_etf_data(isin)
    if data:
        return jsonify({'success': True, 'data': data.to_dict()})
    return jsonify({'success': False, 'error': f'ETF not found: {isin}'})


@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    """Run Monte Carlo simulation with provided parameters."""
    try:
        params = request.json
        
        # Parse contributions from dynamic persons array
        persons_data = params.get('persons', [])
        contributions = []
        starting_loans = {}
        
        if not persons_data:
            # Fallback to defaults if no persons provided
            for i, (name, monthly) in enumerate(DEFAULT_CONFIG['contributions'].items()):
                contributions.append(Contribution(name, monthly))
                starting_loans[name] = DEFAULT_CONFIG['starting_loans'].get(name, 0)
        else:
            for p in persons_data:
                name = p.get('name', f'Person{len(contributions)+1}')
                monthly = float(p.get('monthly', 0))
                loan = float(p.get('loan', 0))
                contributions.append(Contribution(name, monthly))
                starting_loans[name] = loan
        
        # Parse starting capital - auto-calculate from loans if not specified
        starting_capital_input = params.get('starting_capital', 0)
        if starting_capital_input and float(starting_capital_input) > 0:
            starting_capital = float(starting_capital_input)
        else:
            # Auto-calculate: sum of all starting loans
            starting_capital = sum(starting_loans.values())
        
        # Parse ETFs
        etf_list = []
        if 'etfs' in params and isinstance(params['etfs'], list):
            # Dynamic list from Phase 3
            for item in params['etfs']:
                isin = item.get('isin', '').strip()
                allocation = float(item.get('allocation', 0))
                if isin and allocation > 0:
                    etf_list.append((isin, allocation))
        else:
            # Legacy fallback
            for i in range(1, 4):
                isin = params.get(f'etf{i}_isin', '').strip()
                allocation = float(item.get(f'etf{i}_allocation', 0))
                if isin and allocation > 0:
                    etf_list.append((isin, allocation))
        
        # Validate portfolio
        is_valid, error_msg = validate_portfolio(etf_list)
        if not is_valid:
            return jsonify({'success': False, 'error': error_msg})
        
        # Fetch ETF data and build portfolio
        portfolio = []
        etf_info = []
        for isin, allocation in etf_list:
            data = fetch_etf_data(isin)
            if not data:
                return jsonify({'success': False, 'error': f'Could not fetch data for {isin}'})
            
            portfolio.append(PortfolioETF(
                isin=isin,
                allocation=allocation / 100,  # Convert to decimal
                annual_return=data.annual_return,
                annual_volatility=data.annual_volatility
            ))
            etf_info.append(data.to_dict())
        
        # Parse rental configuration
        include_rental = params.get('include_rental', False)
        repay_mortgage = params.get('repay_mortgage', False)
        repayment_year = int(params.get('sale_year') or 2030)
        
        rental = None
        if include_rental:
            mortgage_balance = float(params.get('mortgage_balance') or 0)
            monthly_payment = float(params.get('mortgage_payment') or 0)
            rental_income = float(params.get('rental_income') or 0)
            interest_rate = float(params.get('mortgage_rate') or 0) / 100
            adult1_share = float(params.get('adult1_mortgage_share') or mortgage_balance / 2)
            adult2_share = float(params.get('adult2_mortgage_share') or mortgage_balance / 2)
            
            rental = RentalProperty(
                include=True,
                sell=repay_mortgage,
                sale_year=repayment_year,
                mortgage_2026=mortgage_balance,
                monthly_payment=monthly_payment,
                monthly_income=rental_income,
                interest_rate=interest_rate,
                mart_share=adult1_share,
                kerli_share=adult2_share
            )
        
        # Run simulation
        simulator = MonteCarloSimulator(
            portfolio=portfolio,
            contributions=contributions,
            rental=rental,
            n_simulations=10000,
            seed=None  # Random each time
        )
        
        start_year = int(params.get('start_year') or 2026)
        start_month = int(params.get('start_month') or 1)
        end_year = int(params.get('end_year') or 2040)
        
        result = simulator.simulate(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            starting_capital=starting_capital,
            starting_loans=starting_loans,
            annual_costs=float(params.get('annual_costs') or 0),
            withdrawal_rate=float(params.get('withdrawal_rate') or 0) / 100,
            withdrawal_start_year=int(params.get('withdrawal_start_year') or 2035),
            withdrawal_mode=params.get('withdrawal_mode', 'loan'),
            contribution_end_year=int(params['contribution_end_year']) if params.get('contribution_end_year') else None,
            contribution_change_year=int(params['contribution_change_year']) if params.get('contribution_change_year') else None,
            contribution_change_factor=float(params.get('contribution_change_factor') or 1.0)
        )
        
        # Calculate loan evolution
        loan_evolution = calculate_loan_evolution(
            starting_loans=starting_loans,
            contributions=contributions,
            years=result.years,
            rental=rental,
            start_month=start_month,
            payouts=result.payouts_p50,

            withdrawal_mode=params.get('withdrawal_mode', 'dividend'),
            contribution_end_year=int(params['contribution_end_year']) if params.get('contribution_end_year') else None,
            contribution_change_year=int(params['contribution_change_year']) if params.get('contribution_change_year') else None,
            contribution_change_factor=float(params.get('contribution_change_factor') or 1.0)
        )
        
        # Calculate total loans per year
        total_loans = []
        for i in range(len(result.years)):
            year_total = sum(loan_evolution[person][i] for person in loan_evolution)
            total_loans.append(round(year_total, 2))
        
        # Calculate profits (P50 balance - loans)
        profits_p50 = [round(result.p50[i] - total_loans[i], 2) for i in range(len(result.years))]
        
        # Calculate balance breakdown using Monte Carlo P50 values
        balance_breakdown = []
        annual_contrib = sum(c.monthly_amount * 12 for c in contributions)
        annual_costs_val = float(params.get('annual_costs', 0))
        rental_annual = rental.monthly_income * 12 if rental else 0
        mortgage_deduction = (rental.mart_share + rental.kerli_share) if rental and rental.sell else 0
        
        for i, year in enumerate(result.years):
            row = {'year': year}
            
            # Start balance
            if i == 0:
                row['start_balance'] = round(starting_capital, 0)
            else:
                row['start_balance'] = round(result.p50[i-1], 0)
            
            # Contributions & Costs for this year
            if year == result.years[0]:
                months_first_year = 12 - start_month + 1
                year_contrib = sum(c.monthly_amount for c in contributions) * months_first_year
                year_costs = (annual_costs_val / 12) * months_first_year
            else:
                year_contrib = annual_contrib
                year_costs = annual_costs_val
            
            row['contributions'] = round(year_contrib, 0)
            row['costs'] = round(-year_costs, 0)  # Store as negative for display
            
            # Payouts (Withdrawals) - P50
            year_payout = result.payouts_p50[i]
            row['payouts'] = round(-year_payout, 0) # Store as negative
            
            # Rental income
            if rental and rental.include and rental.sell and year >= rental.sale_year:
                if year == rental.sale_year:
                     row['rental_income'] = round(rental.monthly_income * 6, 0) # July-Dec
                else:
                    row['rental_income'] = round(rental_annual, 0)
            else:
                row['rental_income'] = 0
            
            # Mortgage deduction
            if rental and rental.include and rental.sell and year == rental.sale_year:
                row['mortgage_deduction'] = round(-mortgage_deduction, 0)
            else:
                row['mortgage_deduction'] = 0
            
            # End balance
            row['end_balance'] = round(result.p50[i], 0)
            
            # Back-calculate effective investment return
            # known_changes = contributions + rental + mortgage_deduction + costs + payouts
            known_changes = (row['contributions'] + row['rental_income'] + 
                           row['mortgage_deduction'] + row['costs'] + row['payouts'])
            
            row['investment_return'] = round(row['end_balance'] - row['start_balance'] - known_changes, 0)
            
            balance_breakdown.append(row)
        
        return jsonify({
            'success': True,
            'simulation': result.to_dict(),
            'loan_evolution': loan_evolution,
            'total_loans': total_loans,
            'profits_p50': profits_p50,
            'balance_breakdown': balance_breakdown,
            'portfolio': {
                'expected_return': round(simulator.portfolio_return * 100, 2),
                'expected_volatility': round(simulator.portfolio_volatility * 100, 2)
            },
            'etf_info': etf_info,
            'starting_capital': starting_capital
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("OÜ Investment Modeler - Starting Web Server")
    print("="*60)
    print("\n🌐 Open http://localhost:5021 in your browser\n")
    app.run(debug=True, port=5021)
