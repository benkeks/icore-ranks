# CORE Rankings Collection and Visualization

This repository contains Python scripts to collect and visualize CORE (CORE Conference Ranking) information from the CORE portal (https://portal.core.edu.au/) for Field Of Research: 4613 - Theory of computation.

## Features

- **Data Collection**: Automatically scrape CORE portal for ranking information across multiple years
- **Data Visualization**: Generate multiple plots showing:
  - Rank distribution over time (stacked bar chart)
  - Individual conference ranking changes (line plot)
  - Rank transitions year-over-year (bar chart)
- **Summary Statistics**: Generate comprehensive statistics about the collected data

## Installation

1. Clone this repository:
```bash
git clone https://github.com/benkeks/icore-ranks.git
cd icore-ranks
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Collect Rankings Data

Run the collection script to scrape CORE portal for ranking information:

```bash
python collect_rankings.py
```

This will:
- Fetch ranking data for FoR 4613 (Theory of computation) across all available years
- Save the data to `rankings_data.json`

**Note**: The script is designed to be polite to the server with rate limiting. The collection process may take a few minutes depending on the number of years available.

### Step 2: Visualize the Data

Generate visualizations from the collected data:

```bash
python visualize_rankings.py
```

This will create three plots:
- `rank_distribution.png`: Shows how many conferences had each rank over the years
- `conference_rank_changes.png`: Tracks individual conferences' ranking changes
- `rank_transitions.png`: Shows the most common rank transitions

The script also prints summary statistics to the console.

## Output Files

After running both scripts, you'll have:

- `rankings_data.json`: Raw collected data in JSON format
- `rank_distribution.png`: Stacked bar chart of rank distribution over time
- `conference_rank_changes.png`: Line plot of individual conference rankings
- `rank_transitions.png`: Bar chart of rank transitions

## Data Structure

The collected data is stored in JSON format with the following structure:

```json
[
  {
    "title": "Conference Full Name",
    "acronym": "CONF",
    "rank": "A*",
    "year": "2023"
  },
  ...
]
```

## CORE Ranking Levels

The CORE rankings use the following tiers (from highest to lowest):
- **A\***: Flagship conferences - top ~5% of conferences
- **A**: Excellent conferences - next ~15% of conferences  
- **B**: Good conferences - next ~30% of conferences
- **C**: Other ranked conferences - remaining ~50% of conferences
- **Australasian**: Conferences of particular interest to Australasian community
- **Unranked**: Conferences not yet ranked or no longer ranked

## Requirements

- Python 3.7+
- requests
- beautifulsoup4
- pandas
- matplotlib
- lxml

## Notes

- The scraping script is designed to respect the CORE portal's server with appropriate delays between requests
- If the portal structure changes, the parsing logic in `collect_rankings.py` may need to be updated
- The scripts are focused on FoR 4613 (Theory of computation), but can be adapted for other fields by modifying the `for_code` parameter

## License

This project is open source. The CORE ranking data belongs to CORE (Computing Research and Education Association of Australasia).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.