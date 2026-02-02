#!/usr/bin/env python3
"""
Script to collect CORE ranking information from CORE portal
for Field Of Research: 4613 - Theory of computation
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime


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
        """Get list of available ranking years"""
        try:
            response = self.session.get(self.SEARCH_URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find year selector or links
            # This needs to be adapted based on actual site structure
            years = []
            # Look for year options in dropdowns or links
            year_elements = soup.find_all(['option', 'a'], text=lambda t: t and t.strip().isdigit() and len(t.strip()) == 4)
            
            for elem in year_elements:
                year = elem.get_text(strip=True)
                # Use current year - 1 as upper bound since current year rankings may not be published yet
                if year.isdigit() and 2000 <= int(year) <= datetime.now().year - 1:
                    years.append(year)
            
            return sorted(list(set(years)))
        except Exception as e:
            print(f"Error fetching available years: {e}")
            # Return a default range if we can't fetch
            return [str(y) for y in range(2008, datetime.now().year + 1)]
    
    def search_for_code(self, for_code="4613", year=None):
        """
        Search for conferences with the given Field of Research code
        
        Args:
            for_code: Field of Research code (default: 4613 - Theory of computation)
            year: Specific year to query (optional)
        
        Returns:
            List of conference entries with their rankings
        """
        params = {
            'by': 'for',
            'for': for_code,
        }
        
        if year:
            params['year'] = year
            
        try:
            response = self.session.get(self.SEARCH_URL, params=params)
            response.raise_for_status()
            return self.parse_results(response.content, year)
        except Exception as e:
            print(f"Error searching for code {for_code} (year: {year}): {e}")
            return []
    
    def parse_results(self, html_content, year):
        """Parse HTML results to extract conference rankings"""
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        # Find the results table
        # The exact structure depends on the website, adapting to common patterns
        table = soup.find('table', class_=['table', 'results', 'rankings'])
        
        if table:
            rows = table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 3:
                    entry = {
                        'title': cells[0].get_text(strip=True),
                        'acronym': cells[1].get_text(strip=True) if len(cells) > 1 else '',
                        'rank': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                        'year': year
                    }
                    results.append(entry)
        
        return results
    
    def collect_all_years(self, for_code="4613", output_file="rankings_data.json"):
        """
        Collect rankings for all available years and save to file
        
        Args:
            for_code: Field of Research code
            output_file: Output JSON file path
        """
        print(f"Collecting CORE rankings for FoR {for_code}...")
        
        years = self.get_available_years()
        print(f"Found years: {years}")
        
        all_data = []
        
        for year in years:
            print(f"Fetching data for year {year}...")
            results = self.search_for_code(for_code, year)
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
    
    # Collect rankings for FoR 4613 - Theory of computation
    data = scraper.collect_all_years(for_code="4613", output_file="rankings_data.json")
    
    print("\nSample of collected data:")
    if data:
        for entry in data[:5]:
            print(f"  {entry.get('year', 'N/A')}: {entry.get('title', 'N/A')} - Rank: {entry.get('rank', 'N/A')}")


if __name__ == "__main__":
    main()
