#!/usr/bin/env python3
"""
Script to visualize CORE ranking changes over the years
"""

import json
import os
import numpy as np
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
        if 'source' in df.columns:
            df['source'] = df['source'].astype(str)
        
        if 'rank' in df.columns:
            # Normalize rank values
            df['rank'] = df['rank'].str.strip()
            # Group similar ranks together
            df['rank_group'] = df['rank'].apply(self._categorize_rank)
        
        return df
    
    def _categorize_rank(self, rank):
        """Categorize rank values into main categories"""
        if not rank:
            return 'Unranked'
        
        rank_upper = rank.upper().strip()
        
        if rank_upper.startswith('A*'):
            return 'A*'
        elif rank_upper.startswith('A'):
            return 'A'
        elif rank_upper.startswith('B'):
            return 'B'
        elif rank_upper.startswith('C'):
            return 'C'
        elif 'AUSTRALASIAN' in rank_upper or 'REGIONAL' in rank_upper:
            return 'Australasian'
        elif 'NATIONAL' in rank_upper:
            return 'National'
        else:
            return 'Unranked'
    
    def plot_rank_distribution_over_time(self, output_file="rank_distribution.png"):
        """Create a stacked bar chart showing rank distribution over sources"""
        df = self.prepare_dataframe()
        
        if df.empty:
            print("No data to plot!")
            return
        
        # Count ranks per source
        rank_counts = df.groupby(['source', 'rank_group']).size().unstack(fill_value=0)
        
        # Reorder columns by rank hierarchy
        available_ranks = [r for r in self.RANK_ORDER if r in rank_counts.columns]
        rank_counts = rank_counts[available_ranks]
        
        # Sort sources chronologically
        source_order = sorted(rank_counts.index)
        rank_counts = rank_counts.loc[source_order]
        
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
        
        ax.set_xlabel('Edition/Source', fontsize=12)
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
        """Plot how individual conferences' ranks changed across sources
        
        Includes all conferences with at least one A*, A, or B rank, with visual 
        disambiguation for conferences at the same rank level.
        """
        df = self.prepare_dataframe()
        
        if df.empty:
            print("No data to plot!")
            return
        
        # Find conferences that have at least one A*, A, or B rank at any point
        top_ranks = df[df['rank_group'].isin(['A*', 'A', 'B'])]
        target_conferences = top_ranks['acronym'].unique()
        
        # Get all data for these conferences (including times they drop below B)
        df_target = df[df['acronym'].isin(target_conferences)].copy()
        
        if df_target.empty:
            print("No A*, A, or B conferences found")
            return
        
        # Create rank to numeric mapping for plotting, excluding Australasian from display
        display_rank_order = [r for r in self.RANK_ORDER if r != 'Australasian']
        rank_to_num = {rank: i for i, rank in enumerate(reversed(display_rank_order))}
        df_target['rank_num'] = df_target['rank_group'].map(rank_to_num)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(18, 12))
        
        # Convert source to numeric for plotting
        unique_sources = sorted(df_target['source'].unique())
        source_to_num = {s: i for i, s in enumerate(unique_sources)}
        df_target['source_num'] = df_target['source'].map(source_to_num)
        
        # Line styles, markers, and line widths for disambiguation
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', '+']
        line_widths = [1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5]
        
        # Create mapping of conference to color (cycle through extended palette)
        sorted_conferences = sorted(target_conferences)
        n_confs = len(sorted_conferences)
        colors_list = plt.cm.tab20c(range(n_confs % 20))
        if n_confs > 20:
            colors_list = plt.cm.get_cmap('hsv')(np.linspace(0, 0.95, n_confs))
        
        # Calculate jitter for each rank level, accounting for number of conferences
        # but ensuring no collision between rank levels
        rank_jitters = {}
        rank_counts = {}
        
        # First pass: count conferences at each rank
        for rank in display_rank_order:
            confs_at_rank = [c for c in sorted_conferences if df_target[(df_target['acronym'] == c) & (df_target['rank_group'] == rank)].shape[0] > 0]
            rank_counts[rank] = len(confs_at_rank)
        
        # Second pass: assign jitter values with balanced range
        for rank in display_rank_order:
            confs_at_rank = [c for c in sorted_conferences if df_target[(df_target['acronym'] == c) & (df_target['rank_group'] == rank)].shape[0] > 0]
            n_at_rank = len(confs_at_rank)
            
            if n_at_rank > 0:
                # Max jitter range is 0.45 to allow good spread while avoiding rank level collision
                # Scale within this range based on number of conferences
                max_jitter = 0.45
                jitter_scale = np.sqrt(n_at_rank) / np.sqrt(64)  # Normalize by typical max (64 at C rank)
                jitter_scale = min(1.0, jitter_scale)  # Cap at 1.0
                effective_jitter_range = max_jitter * jitter_scale
                
                # Spread conferences evenly around their rank level
                jitter_values = np.linspace(-effective_jitter_range, effective_jitter_range, n_at_rank)
                for conf, jitter in zip(confs_at_rank, jitter_values):
                    if conf not in rank_jitters:
                        rank_jitters[conf] = {}
                    rank_jitters[conf][rank] = jitter
        
        # Plot each conference
        for idx, conf in enumerate(sorted_conferences):
            conf_data = df_target[df_target['acronym'] == conf].sort_values('source_num')
            if len(conf_data) > 0:
                # Apply rank-specific jitter
                y_values = []
                for _, row in conf_data.iterrows():
                    base_y = row['rank_num']
                    jitter = rank_jitters.get(conf, {}).get(row['rank_group'], 0)
                    y_values.append(base_y + jitter)
                
                line_style = line_styles[idx % len(line_styles)]
                marker = markers[idx % len(markers)]
                line_width = line_widths[idx % len(line_widths)]
                color = colors_list[idx % len(colors_list)]
                
                ax.plot(conf_data['source_num'].values, y_values, 
                       marker=marker, linestyle=line_style, 
                       label=conf, linewidth=line_width, markersize=4, 
                       color=color, alpha=0.85)
        
        # Set y-axis labels to rank names (excluding Australasian)
        ax.set_yticks(range(len(display_rank_order)))
        ax.set_yticklabels(reversed(display_rank_order), fontsize=13)
        
        # Set x-axis labels to sources
        ax.set_xticks(range(len(unique_sources)))
        ax.set_xticklabels(unique_sources, rotation=45, fontsize=13)
        
        ax.set_xlabel('Edition/Source', fontsize=16)
        ax.set_ylabel('CORE Rank', fontsize=16)
        ax.set_title(f'Conference Ranking Changes Over Time (All A*, A, B Conferences)\nField of Research: 4613 - Theory of Computation',
                     fontsize=20, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Add note about jitter
        ax.text(0.02, 0.02, 'Note: Y-positions within one rank level are jittered to avoid overlap', 
               transform=ax.transAxes, fontsize=12, style='italic', color='gray')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path} ({len(target_conferences)} A*, A, B conferences)")
        
        plt.close()
    
    def plot_rank_transitions(self, output_file="rank_transitions.png"):
        """Plot a heatmap of rank transitions source-to-source"""
        df = self.prepare_dataframe()
        
        if df.empty:
            print("No data to plot!")
            return
        
        # Find conferences with consecutive source data
        transitions = []
        
        for conf in df['acronym'].unique():
            conf_data = df[df['acronym'] == conf].sort_values('source')
            if len(conf_data) > 1:
                for i in range(len(conf_data) - 1):
                    from_rank = conf_data.iloc[i]['rank_group']
                    to_rank = conf_data.iloc[i + 1]['rank_group']
                    source = conf_data.iloc[i + 1]['source']
                    transitions.append({
                        'from': from_rank,
                        'to': to_rank,
                        'source': source,
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
        ax.set_title('Top Rank Transitions Across Editions\nField of Research: 4613 - Theory of Computation',
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
        
        if 'source' in df.columns:
            print(f"\nEditions covered: {sorted(df['source'].unique())}")
            print(f"Total entries: {len(df)}")
            print(f"Unique conferences: {df['acronym'].nunique()}")
        
        if 'rank_group' in df.columns:
            print("\nRank distribution:")
            rank_dist = df['rank_group'].value_counts().sort_index()
            for rank, count in rank_dist.items():
                print(f"  {rank}: {count} ({count/len(df)*100:.1f}%)")
        
        # Conferences that changed rank
        conferences_with_changes = 0
        for conf in df['acronym'].unique():
            conf_data = df[df['acronym'] == conf]
            if conf_data['rank_group'].nunique() > 1:
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
