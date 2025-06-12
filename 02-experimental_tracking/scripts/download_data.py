#!/usr/bin/env python
"""
NYC Taxi Data Downloader
========================

Utility script to download NYC Taxi Trip data from the official TLC website.
Handles downloading multiple months of data for green and yellow taxis.

Examples:
    # Download January 2021 green taxi data
    python -m scripts.download_data --year 2021 --months 1 --taxi green

    # Download multiple months of yellow taxi data
    python -m scripts.download_data --year 2021 --months 1 2 3 --taxi yellow

    # Download both green and yellow taxi data for a quarter
    python -m scripts.download_data --year 2021 --months 1 2 3 --taxi green yellow

Author: Habeeb Babatunde
Date: May 14, 2025
"""

import os
import argparse
import logging
from typing import List, Optional
import urllib.request
import time
import sys
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    """Progress bar for downloads using tqdm"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update progress bar
        
        Args:
            b: Number of blocks transferred
            bsize: Size of each block
            tsize: Total size
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str) -> None:
    """
    Download file with progress bar
    
    Args:
        url: URL to download from
        output_path: Path to save the file
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_taxi_data(
    year: int,
    months: List[int],
    taxi_types: List[str] = ['green'],
    data_dir: str = './data',
    max_retries: int = 3,
    retry_delay: int = 5
) -> None:
    """
    Download NYC Taxi trip data for specified months and taxi types
    
    Args:
        year: Year to download data for
        months: List of months to download (1-12)
        taxi_types: List of taxi types ('green', 'yellow')
        data_dir: Directory to save data files
        max_retries: Maximum number of download retries
        retry_delay: Delay between retries in seconds
    """
    # Validate inputs
    if not all(1 <= month <= 12 for month in months):
        raise ValueError("Months must be between 1 and 12")
    
    if not all(taxi_type in ['green', 'yellow'] for taxi_type in taxi_types):
        raise ValueError("Taxi types must be 'green' or 'yellow'")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Base URL for NYC TLC data
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    
    # Track successful and failed downloads
    successful = []
    failed = []
    
    # Download data for each combination of taxi type and month
    for taxi_type in taxi_types:
        for month in months:
            # Format month with leading zero
            month_str = f"{month:02d}"
            
            # Construct filename and URL
            filename = f"{taxi_type}_tripdata_{year}-{month_str}.parquet"
            url = f"{base_url}/{filename}"
            output_path = os.path.join(data_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(output_path):
                logger.info(f"File already exists: {output_path}")
                successful.append(filename)
                continue
            
            # Try to download with retries
            for attempt in range(max_retries):
                try:
                    logger.info(f"Downloading {url} to {output_path}")
                    download_file(url, output_path)
                    successful.append(filename)
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to download {filename} after {max_retries} attempts")
                        failed.append(filename)
    
    # Summary report
    logger.info(f"Download summary:")
    logger.info(f"  Successfully downloaded: {len(successful)} files")
    if successful:
        logger.info(f"    - " + ", ".join(successful))
    
    if failed:
        logger.warning(f"  Failed to download: {len(failed)} files")
        logger.warning(f"    - " + ", ".join(failed))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download NYC Taxi Trip Data")
    
    parser.add_argument("--year", type=int, required=True,
                       help="Year to download data for (e.g., 2021)")
    
    parser.add_argument("--months", type=int, required=True, nargs="+",
                       help="Months to download (1-12)")
    
    parser.add_argument("--taxi", type=str, default=["green"], nargs="+",
                       choices=["green", "yellow"],
                       help="Taxi types to download")
    
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Directory to save data files")
    
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum number of download retries")
    
    parser.add_argument("--retry-delay", type=int, default=5,
                       help="Delay between retries in seconds")
    
    return parser.parse_args()


def main() -> int:
    """Main entry point"""
    try:
        args = parse_args()
        
        download_taxi_data(
            year=args.year,
            months=args.months,
            taxi_types=args.taxi,
            data_dir=args.data_dir,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        
        return 0
    
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())