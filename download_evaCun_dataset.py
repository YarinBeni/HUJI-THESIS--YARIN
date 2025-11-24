"""
EvaCun 2025 Dataset Downloader and Loader (Verified Real Data)
============================================================

Downloads and verifies REAL eBL dataset from Zenodo.
Compares to stats from both EvaCun and source paper.
"""

import requests
import json
import pandas as pd
from pathlib import Path

class EvaCunDatasetDownloader:
    def __init__(self, data_dir="evaCun_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Real Zenodo file URL
        self.zenodo_url = "https://zenodo.org/records/10018951/files/fragments.json?download=1"
        
        # Expected stats from papers
        self.expected = {
            'total_fragments': 25000,  # Source paper ~25,000
            'total_lines': 350000,     # Source paper >350,000
            'evacun_subset': 10214     # EvaCun Table 1
        }
    
    def download_real_dataset(self):
        filepath = self.data_dir / "eBL_fragments.json"
        if filepath.exists():
            print(f"✓ Exists: {filepath}")
            return filepath
        
        print("📥 Downloading...")
        response = requests.get(self.zenodo_url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"✓ Downloaded: {filepath}")
        return filepath
    
    def load_and_verify(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"✓ Loaded {len(df)} fragments")
        
        # Compute real stats
        real_fragments = len(df)
        real_lines = sum(len(str(row.get('atf', '')).splitlines()) for _, row in df.iterrows())
        
        # Verify
        print("\n🧪 Verification:")
        print(f"Fragments: Real {real_fragments} | Expected ~{self.expected['total_fragments']} | Match: {abs(real_fragments - self.expected['total_fragments']) < 2500}")
        print(f"Lines: Real {real_lines} | Expected >{self.expected['total_lines']} | Match: {real_lines > self.expected['total_lines'] - 10000}")
        print(f"EvaCun Subset Check: Real larger than {self.expected['evacun_subset']}? {real_fragments > self.expected['evacun_subset']}")

def main():
    downloader = EvaCunDatasetDownloader()
    file = downloader.download_real_dataset()
    downloader.load_and_verify(file)

if __name__ == "__main__":
    main()