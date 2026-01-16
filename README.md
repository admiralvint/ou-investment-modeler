# OÜ Investment Modeler

A Monte Carlo simulation tool for projecting investment company (OÜ) growth. Built with Flask and Chart.js.

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- **Monte Carlo Simulation**: 10,000 random paths based on historical ETF returns
- **Multiple Percentiles**: View P2, P10, P50, P90, P98 scenarios or deterministic average
- **Multi-ETF Portfolio**: Support for up to 3 ETFs with custom allocations
- **Rental Property Integration**: Optional rental income with mortgage repayment options
- **Loan Tracking**: Track tax-free withdrawal amounts for each person
- **Interactive Charts**: Real-time projections and profit/loss visualization

## Screenshot

[Add screenshot here]

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ou-investment-modeler.git
cd ou-investment-modeler
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install flask pandas numpy yfinance
```

4. Run the application:
```bash
python app.py
```

5. Open http://localhost:5020 in your browser.

## Usage

### Basic Setup

1. **Starting Capital**: Enter your company's initial balance
2. **Monthly Contributions**: Set monthly contributions for up to 4 people
3. **Starting Loans**: Set initial loan amounts from company to each person
4. **ETF Portfolio**: Enter ISIN codes and allocation percentages (must total 100%)
5. **Projection Period**: Choose start and end years

### Rental Property (Optional)

If you have a rental property owned by the company:

1. Check "Include rental property"
2. Enter mortgage details (balance, payment, income, rate)
3. Choose between:
   - **Pay mortgage personally**: Rental income covers mortgage (company doesn't receive rental income)
   - **Repay mortgage from company**: Company uses loan balances to pay off mortgage, then receives rental income

### Understanding Results

- **P2/P10/P50/P90/P98**: Different percentile outcomes from 10,000 simulations
- **Average (deterministic)**: What would happen with exactly the historical average return
- **Profit = Balance - Total Loans**: What remains after repaying all loans

## Configuration

Default values in `app.py` can be customized:

```python
DEFAULT_CONFIG = {
    'contributions': {
        'Person1': 500,
        'Person2': 500,
        'Person3': 100,
        'Person4': 100
    },
    'starting_capital': 50000.0,
    ...
}
```

## How It Works

### Monte Carlo Simulation

The simulator:
1. Fetches historical ETF data from Yahoo Finance
2. Calculates annualized return and volatility
3. For each of 10,000 paths:
   - Each month: adds contributions and applies random return (normal distribution)
   - Handles rental income and mortgage repayment when applicable
4. Computes percentiles across all paths

### ETF Data

- Data fetched from Yahoo Finance via `yfinance`
- ISIN codes are mapped to tickers (cached for performance)
- Returns/volatility calculated from last 10 years of daily data

## Dependencies

- Flask
- pandas
- numpy
- yfinance

## License

MIT License - see LICENSE file for details.

## Contributing

Pull requests welcome! Please open an issue first to discuss proposed changes.
