#!/usr/bin/env python3
"""
Generate sample CORE ranking data for testing and demonstration purposes.
This is useful when the CORE portal is not accessible or for testing the visualization.
"""

import json
import random
from datetime import datetime


def generate_sample_data():
    """Generate sample CORE ranking data for FoR 4613"""
    
    # Sample conferences in Theory of Computation
    conferences = [
        ("STOC", "ACM Symposium on Theory of Computing"),
        ("FOCS", "IEEE Symposium on Foundations of Computer Science"),
        ("SODA", "ACM-SIAM Symposium on Discrete Algorithms"),
        ("ICALP", "International Colloquium on Automata, Languages and Programming"),
        ("LICS", "ACM/IEEE Symposium on Logic in Computer Science"),
        ("CCC", "IEEE Conference on Computational Complexity"),
        ("MFCS", "Mathematical Foundations of Computer Science"),
        ("STACS", "Symposium on Theoretical Aspects of Computer Science"),
        ("ESA", "European Symposium on Algorithms"),
        ("APPROX", "International Workshop on Approximation Algorithms"),
        ("RANDOM", "International Workshop on Randomization and Computation"),
        ("FSTTCS", "Foundations of Software Technology and Theoretical Computer Science"),
        ("ISAAC", "International Symposium on Algorithms and Computation"),
        ("LATIN", "Latin American Symposium on Theoretical Informatics"),
        ("WADS", "Algorithms and Data Structures Symposium"),
        ("IWPEC", "International Workshop on Parameterized and Exact Computation"),
        ("SWAT", "Scandinavian Workshop on Algorithm Theory"),
        ("WG", "International Workshop on Graph-Theoretic Concepts in Computer Science"),
        ("COCOON", "International Computing and Combinatorics Conference"),
        ("CSR", "Computer Science in Russia"),
    ]
    
    # Generate data for years 2010-2024
    years = list(range(2010, 2025))
    
    data = []
    
    for conf_acronym, conf_title in conferences:
        # Each conference has a "base" rank that might change over time
        base_ranks = ['A*', 'A', 'B', 'C']
        
        # Top conferences
        if conf_acronym in ['STOC', 'FOCS', 'SODA']:
            rank_pool = ['A*'] * 10 + ['A'] * 1  # Mostly A*
        elif conf_acronym in ['ICALP', 'LICS', 'CCC']:
            rank_pool = ['A*'] * 5 + ['A'] * 5  # Mix of A* and A
        elif conf_acronym in ['MFCS', 'STACS', 'ESA', 'APPROX']:
            rank_pool = ['A'] * 8 + ['B'] * 2  # Mostly A
        elif conf_acronym in ['RANDOM', 'FSTTCS', 'ISAAC']:
            rank_pool = ['A'] * 4 + ['B'] * 6  # Mostly B
        else:
            rank_pool = ['B'] * 5 + ['C'] * 5  # Mix of B and C
        
        current_rank = random.choice(rank_pool)
        
        for year in years:
            # Small chance of rank change each year
            if random.random() < 0.15:  # 15% chance of change
                # Prefer small changes
                rank_idx = base_ranks.index(current_rank) if current_rank in base_ranks else 2
                
                if random.random() < 0.5 and rank_idx > 0:
                    current_rank = base_ranks[rank_idx - 1]  # Improve
                elif rank_idx < len(base_ranks) - 1:
                    current_rank = base_ranks[rank_idx + 1]  # Decline
            
            entry = {
                "title": conf_title,
                "acronym": conf_acronym,
                "rank": current_rank,
                "year": str(year)
            }
            data.append(entry)
    
    return data


def main():
    """Generate and save sample data"""
    print("Generating sample CORE ranking data...")
    
    data = generate_sample_data()
    
    # Save to file
    output_file = "rankings_data.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {len(data)} sample entries")
    print(f"Data saved to {output_file}")
    
    # Show some statistics
    years = set(entry['year'] for entry in data)
    conferences = set(entry['acronym'] for entry in data)
    ranks = {}
    for entry in data:
        rank = entry['rank']
        ranks[rank] = ranks.get(rank, 0) + 1
    
    print(f"\nStatistics:")
    print(f"  Years: {min(years)} to {max(years)}")
    print(f"  Conferences: {len(conferences)}")
    print(f"  Rank distribution:")
    for rank in sorted(ranks.keys()):
        print(f"    {rank}: {ranks[rank]} ({ranks[rank]/len(data)*100:.1f}%)")


if __name__ == "__main__":
    main()
