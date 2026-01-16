import requests
import json
import sys

def verify_phase3():
    print("Testing Phase 3 Features...")
    
    url = "http://127.0.0.1:5021/api/simulate"
    
    # 1. Test Single Stock + ETF (15 Years)
    # Using TSLA (should work if history > 10yrs) and VWCE (IE00BK5BQT80)
    # We will check start_year of data in ETF Fetcher if possible, or just result success.
    
    payload = {
        "starting_capital": 10000,
        "persons": [],
        "etfs": [
            {"isin": "TSLA", "allocation": 50},
            {"isin": "IE00BK5BQT80", "allocation": 50}
        ],
        "include_rental": False,
        "start_year": 2026,
        "end_year": 2040,
        # Withdrawals Enabled
        "withdrawal_rate": 5,
        "withdrawal_start_year": 2035,
        "withdrawal_mode": "loan"
    }
    
    try:
        print("Sending POST request...")
        response = requests.post(url, json=payload)
        data = response.json()
        
        if not data.get("success"):
            print(f"FAILED: Simulation error: {data.get('error')}")
            sys.exit(1)
            
        print("SUCCESS: Simulation ran.")
        
        # Check Balance Breakdown for Payouts
        breakdown = data.get("balance_breakdown", [])
        payouts_found = False
        for row in breakdown:
            if row.get("year") >= 2035:
                payout = row.get("payouts", 0)
                if abs(payout) > 0:
                    payouts_found = True
                    print(f"Year {row['year']} Payout: {payout}")
                    
        if payouts_found:
            print("SUCCESS: Payouts found in breakdown.")
        else:
            print("FAILED: No payouts found in breakdown despite withdrawal plan.")
            
        # Check ETF Data Info (years)
        portfolio = data.get("portfolio", {})
        # Note: API might not return 'years' directly in portfolio summary, 
        # but we can check individual ETF endpoints.
        
        print("Checking ETF History length...")
        etf_resp = requests.get("http://127.0.0.1:5021/api/etf/TSLA")
        etf_data = etf_resp.json()
        if etf_data.get("success"):
            years = etf_data["data"].get("years", 0)
            print(f"TSLA History Years: {years}")
            if years > 10:
                print("SUCCESS: History > 10 years (likely 15).")
            else:
                print("WARNING: History <= 10 years.")
        else:
            print("FAILED: Could not fetch TSLA info.")

    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_phase3()
