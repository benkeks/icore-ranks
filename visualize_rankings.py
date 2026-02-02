#!/usr/bin/env python3
"""
Script to visualize CORE ranking changes over the years
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


class RankingVisualizer:
    """Visualize CORE ranking changes over time"""
    
    RANK_COLORS = {
        'A*': '#1f77b4',  # Blue
        'A': '#2ca02c',    # Green
        'B': '#ff7f0e',    # Orange
        'C': '#d62728',    # Red
        'Australasian': '#9467bd',  # Purple
        'Unranked': '#7f7f7f'  # Gray
    }
    
    RANK_ORDER = ['A*', 'A', 'B', 'C', 'Australasian', 'Unranked']
    
    def __init__(self, data_file="rankings_data.json"):
        """Initialize with path to data file"""
        self.data_file = os.path.join(os.path.dirname(__file__), data_file)
        self.data = self.load_data()
        
    def load_data(self):
        """Load ranking data from JSON file"""
        if not os.path.exists(self.data_file):
            print(f"Error: Data file {self.data_file} not found!")
            return []
        
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} ranking entries")
        return data
    
    def prepare_dataframe(self):
        """Convert data to pandas DataFrame for analysis"""
        if not self.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data)
        
        # Clean and standardize the data
        if 'year' in df.columns:
            df['year'] = df['year'].astype(str)
        
        if 'rank' in df.columns:
            # Normalize rank values
            df['rank'] = df['rank'].str.strip().str.upper()
            # Handle variations
            df['rank'] = df['rank'].replace({
                'A STAR': 'A*',
                'ASTAR': 'A*',
            })
        
        return df
    
    def plot_rank_distribution_over_time(self, output_file="rank_distribution.png"):
        """Create a stacked bar chart showing rank distribution over years"""
        df = self.prepare_dataframe()
        
        if df.empty:
            print("No data to plot!")
            return
        
        # Count ranks per year
        rank_counts = df.groupby(['year', 'rank']).size().unstack(fill_value=0)
        
        # Reorder columns by rank hierarchy
        available_ranks = [r for r in self.RANK_ORDER if r in rank_counts.columns]
        rank_counts = rank_counts[available_ranks]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create stacked bar chart
        rank_counts.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=[self.RANK_COLORS.get(r, '#7f7f7f') for r in available_ranks],
            width=0.8
        )
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Conferences', fontsize=12)
        ax.set_title('CORE Rankings Distribution Over Time\nField of Research: 4613 - Theory of Computation', 
                     fontsize=14, fontweight='bold')
        ax.legend(title='Rank', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        
        plt.close()
        
    def plot_individual_conference_changes(self, output_file="conference_rank_changes.png", top_n=20):
        """Plot how individual conferences' ranks changed over time"""
        df = self.prepare_dataframe()
        
        if df.empty:
            print("No data to plot!")
            return
        
        # Find conferences that appear in multiple years
        conference_counts = df.groupby('acronym').size()
        multi_year_conferences = conference_counts[conference_counts > 1].index
        
        df_multi = df[df['acronym'].isin(multi_year_conferences)]
        
        if df_multi.empty:
            print("No conferences found with data across multiple years")
            return
        
        # Select top N conferences by frequency
        top_conferences = df_multi['acronym'].value_counts().head(top_n).index
        df_top = df_multi[df_multi['acronym'].isin(top_conferences)]
        
        # Create rank to numeric mapping for plotting
        rank_to_num = {rank: i for i, rank in enumerate(reversed(self.RANK_ORDER))}
        df_top['rank_num'] = df_top['rank'].map(rank_to_num)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot each conference
        for conf in top_conferences[:10]:  # Limit to 10 for readability
            conf_data = df_top[df_top['acronym'] == conf].sort_values('year')
            if len(conf_data) > 1:
                ax.plot(conf_data['year'], conf_data['rank_num'], 
                       marker='o', label=conf, linewidth=2, markersize=6)
        
        # Set y-axis labels to rank names
        ax.set_yticks(range(len(self.RANK_ORDER)))
        ax.set_yticklabels(reversed(self.RANK_ORDER))
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('CORE Rank', fontsize=12)
        ax.set_title('Conference Ranking Changes Over Time\nField of Research: 4613 - Theory of Computation',
                     fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        
        plt.close()
    
    def plot_rank_transitions(self, output_file="rank_transitions.png"):
        """Plot a heatmap of rank transitions year-over-year"""
        df = self.prepare_dataframe()
        
        if df.empty:
            print("No data to plot!")
            return
        
        # Find conferences with consecutive year data
        transitions = []
        
        for conf in df['acronym'].unique():
            conf_data = df[df['acronym'] == conf].sort_values('year')
            if len(conf_data) > 1:
                for i in range(len(conf_data) - 1):
                    from_rank = conf_data.iloc[i]['rank']
                    to_rank = conf_data.iloc[i + 1]['rank']
                    year = conf_data.iloc[i + 1]['year']
                    transitions.append({
                        'from': from_rank,
                        'to': to_rank,
                        'year': year,
                        'conference': conf
                    })
        
        if not transitions:
            print("No rank transitions found")
            return
        
        # Count transitions
        transition_counts = defaultdict(int)
        for t in transitions:
            transition_counts[(t['from'], t['to'])] += 1
        
        # Create summary statistics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for display
        transition_text = []
        for (from_rank, to_rank), count in sorted(transition_counts.items(), 
                                                   key=lambda x: x[1], reverse=True)[:15]:
            transition_text.append(f"{from_rank} → {to_rank}: {count}")
        
        # Create text-based visualization
        y_pos = range(len(transition_text))
        counts = [transition_counts[(line.split(':')[0].split(' → ')[0].strip(), 
                                     line.split(':')[0].split(' → ')[1].strip())] 
                  for line in transition_text]
        
        ax.barh(y_pos, counts, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(transition_text)
        ax.set_xlabel('Number of Transitions', fontsize=12)
        ax.set_title('Top Rank Transitions Year-over-Year\nField of Research: 4613 - Theory of Computation',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        
        plt.close()
    
    def generate_summary_stats(self):
        """Print summary statistics about the rankings"""
        df = self.prepare_dataframe()
        
        if df.empty:
            print("No data available for statistics")
            return
        
        print("\n" + "="*60)
        print("CORE Rankings Summary Statistics")
        print("Field of Research: 4613 - Theory of Computation")
        print("="*60)
        
        if 'year' in df.columns:
            print(f"\nYears covered: {df['year'].min()} to {df['year'].max()}")
            print(f"Total entries: {len(df)}")
            print(f"Unique conferences: {df['acronym'].nunique()}")
        
        if 'rank' in df.columns:
            print("\nRank distribution:")
            rank_dist = df['rank'].value_counts().sort_index()
            for rank, count in rank_dist.items():
                print(f"  {rank}: {count} ({count/len(df)*100:.1f}%)")
        
        # Conferences that changed rank
        conferences_with_changes = 0
        for conf in df['acronym'].unique():
            conf_data = df[df['acronym'] == conf]
            if conf_data['rank'].nunique() > 1:
                conferences_with_changes += 1
        
        print(f"\nConferences with rank changes: {conferences_with_changes}")
        print("="*60 + "\n")


def main():
    """Main execution function"""
    visualizer = RankingVisualizer("rankings_data.json")
    
    # Generate summary statistics
    visualizer.generate_summary_stats()
    
    # Create visualizations
    print("Generating visualizations...")
    visualizer.plot_rank_distribution_over_time("rank_distribution.png")
    visualizer.plot_individual_conference_changes("conference_rank_changes.png")
    visualizer.plot_rank_transitions("rank_transitions.png")
    
    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()
