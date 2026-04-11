import requests
import os
import json

BASE_URL = "https://wis2boxstdby.imd.gov.in/oapi/collections/urn:wmo:md:in-imd:surface-based-observations.synop/items"
OUTPUT_DIR = "imd_surface_observations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_sample_items(limit=10):
    # Add limit parameter to restrict number of records
    url = f"{BASE_URL}?limit={limit}&sortby=-reportTime"
    try:
        resp = requests.get(url, verify=False, timeout=30)  # replace with cert path once SSL fixed
        resp.raise_for_status()
        data = resp.json()

        # Save to file
        file_path = os.path.join(OUTPUT_DIR, f"sample_{limit}_observations.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data.get('features', []))} records to {file_path}")
    except Exception as e:
        print(f"Error fetching items: {e}")

if __name__ == "__main__":
    fetch_sample_items(limit=10)
