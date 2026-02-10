#!/usr/bin/env python3
"""
Script to collect CORE ranking information from CORE portal
for all conferences across all editions
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
import csv
from io import StringIO


class COREPortalScraper:
    """Scraper for CORE portal to collect ranking information"""
    
    BASE_URL = "https://portal.core.edu.au"
    SEARCH_URL = f"{BASE_URL}/conf-ranks"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_available_years(self):
        """Get list of available ranking sources (ICORE2026, CORE2023, etc.)"""
        try:
            response = self.session.get(self.SEARCH_URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the source dropdown
            source_select = soup.find('select', {'name': 'source'})
            sources = []
            
            if source_select:
                options = source_select.find_all('option')
                for option in options:
                    value = option.get('value', '').strip()
                    # Skip 'all' option and extract actual source values
                    if value and value != 'all':
                        sources.append(value)
            
            return sources
        except Exception as e:
            print(f"Error fetching available sources: {e}")
            # Return a default list if we can't fetch
            return ['ICORE2026', 'CORE2023', 'CORE2021', 'CORE2020', 'CORE2018', 'CORE2017', 'CORE2014', 'CORE2013', 'ERA2010', 'CORE2008']
    
    def search_for_source(self, source=None):
        """
        Download all conference rankings for a given source/edition via CSV export
        
        Args:
            source: Specific source to query (e.g., 'ICORE2026', 'CORE2023')
        
        Returns:
            List of conference entries with their rankings
        """
        params = {
            'search': '',
            'by': 'all',
            'sort': 'atitle',
            'page': '1',
            'do': 'Export'
        }
        
        if source:
            params['source'] = source
            
        try:
            response = self.session.get(self.SEARCH_URL, params=params)
            response.raise_for_status()
            print(f"\n[DEBUG] Retrieved CSV for source {source}")
            print(f"[DEBUG] Response status: {response.status_code}")
            print(f"[DEBUG] Response length: {len(response.text)} characters")
            print(f"[DEBUG] First 500 chars of CSV:\n{response.text[:500]}")
            return self.parse_csv_results(response.text, source)
        except Exception as e:
            print(f"Error downloading data for source {source}: {e}")
            return []
    
    def parse_csv_results(self, csv_content, source):
        """Parse CSV results to extract conference rankings
        
        CSV columns (no header):
        0: ID
        1: Title
        2: Acronym
        3: Source
        4: Rank
        5: Yes/No flag
        6+: FoR codes
        """
        results = []
        
        try:
            # Parse CSV content using regular reader (no headers)
            csv_file = StringIO(csv_content)
            reader = csv.reader(csv_file)
            
            row_count = 0
            for row in reader:
                row_count += 1
                
                # Skip empty rows
                if not row or len(row) < 5:
                    continue
                
                # Extract fields by column index
                # Extract FoR codes (column 6 onwards, semicolon-separated)
                for_codes = []
                if len(row) > 6:
                    for_codes = [code.strip() for code in row[6:] if code.strip()]
                
                entry = {
                    'id': row[0].strip() if len(row) > 0 else '',
                    'title': row[1].strip() if len(row) > 1 else '',
                    'acronym': row[2].strip() if len(row) > 2 else '',
                    'rank': row[4].strip() if len(row) > 4 else '',
                    'source': source,
                    'for_codes': for_codes
                }
                
                # Debug: print first few rows
                if row_count <= 3:
                    print(f"[DEBUG] Row {row_count}: {entry}")
                
                # Only add entries that have at least a title
                if entry['title']:
                    results.append(entry)
            
            print(f"[DEBUG] Total rows parsed: {row_count}, entries added: {len(results)}")
        
        except Exception as e:
            print(f"Error parsing CSV for source {source}: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def collect_all_years(self, output_file="rankings_data.json"):
        """
        Collect rankings for all available sources/editions and save to file
        
        Args:
            output_file: Output JSON file path
        """
        print(f"Collecting all CORE rankings...")
        
        sources = self.get_available_years()
        print(f"Found sources: {sources}")
        
        all_data = []
        
        for source in sources:
            print(f"Fetching data for source {source}...")
            results = self.search_for_source(source)
            all_data.extend(results)
            time.sleep(1)  # Be polite to the server
        
        # Save to file
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        with open(output_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"Data saved to {output_path}")
        print(f"Total entries collected: {len(all_data)}")
        
        return all_data


def main():
    """Main execution function"""
    scraper = COREPortalScraper()
    
    # Collect rankings for all conferences across all editions
    data = scraper.collect_all_years(output_file="rankings_data.json")
    
    print("\nSample of collected data:")
    if data:
        for entry in data[:5]:
            print(f"  {entry.get('source', 'N/A')}: {entry.get('title', 'N/A')} - Rank: {entry.get('rank', 'N/A')}")


if __name__ == "__main__":
    main()
