#!/usr/bin/env python3
"""
Download ORACC project ZIPs.

This script downloads all public ORACC projects as JSON ZIP files.
Based on Akk/preprocessing/scraping.py
"""

import io
import json
import zipfile
from pathlib import Path
import requests
import warnings

# Suppress SSL warnings for old academic websites
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=''):
        print(f"{desc}...")
        return iterable


def download_projects_jsons(output_dir: str):
    """
    Download ORACC projects as JSON files.

    Args:
        output_dir: Directory to save the project ZIPs
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("Fetching list of ORACC projects...")
    # Disable SSL verification for old academic websites
    res = requests.get('http://oracc.museum.upenn.edu/projects.json', verify=False)
    projects_list = json.loads(res.content)['public']

    print(f"Found {len(projects_list)} public projects")
    print(f"Downloading to: {output_path.absolute()}\n")

    successful = 0
    failed = []

    for project_name in tqdm(projects_list, desc='Downloading projects'):
        project_json_url = f'http://oracc.org/{project_name}/json'
        try:
            r = requests.get(project_json_url, stream=True, timeout=30, verify=False)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as zip_ref:
                    zip_ref.extractall(output_path)
                successful += 1
            else:
                failed.append(f"{project_name} (HTTP {r.status_code})")
        except zipfile.BadZipFile:
            failed.append(f"{project_name} (Bad ZIP)")
        except Exception as e:
            failed.append(f"{project_name} ({str(e)})")

    print(f"\n✓ Successfully downloaded: {successful} projects")
    if failed:
        print(f"✗ Failed to download: {len(failed)} projects")
        print("Failed projects:", ", ".join(failed[:10]))
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download ORACC project ZIPs')
    parser.add_argument(
        '--output_dir',
        default='../v_1/data/downloaded/oracc',
        help='Directory to save ORACC projects (default: ../v_1/data/downloaded/oracc)'
    )

    args = parser.parse_args()
    download_projects_jsons(args.output_dir)
