from etf_fetcher import fetch_etf_data

def test():
    print("Fetching TSLA data...")
    data = fetch_etf_data("TSLA")
    if data:
        d = data.to_dict()
        print("Keys:", d.keys())
        if 'years' in d:
            print(f"SUCCESS: 'years' key found: {d['years']}")
        else:
            print("FAILED: 'years' key missing")
    else:
        print("FAILED: Could not fetch TSLA")

if __name__ == "__main__":
    test()
