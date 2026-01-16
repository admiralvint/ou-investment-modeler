
from etf_fetcher import fetch_etf_data

def test_stock():
    print("Testing TSLA...")
    data = fetch_etf_data("TSLA")
    if data:
        print(f"Success! Name: {data.name}")
        print(f"Type: {data.ticker}") # Should be TSLA
        print(f"Return: {data.annual_return:.2%}")
    else:
        print("Failed to fetch TSLA")

    print("\nTesting SWED-A.ST (Swedbank)...")
    data2 = fetch_etf_data("SWED-A.ST")
    if data2:
         print(f"Success! Name: {data2.name}")
    else:
         print("Failed SWED-A.ST")

if __name__ == "__main__":
    test_stock()
