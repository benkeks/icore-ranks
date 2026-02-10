# CORE Rankings Collection and Visualization

Scripts to collect and visualize CORE conference rankings from https://portal.core.edu.au/.

## Features
- Collect all conferences across all editions (CSV export)
- Store FoR codes per conference
- Visualize ranking trends for a filtered subset (default: FoR 4613 conferences present in ICORE2026)
- Exclude ERA2010 from visualizations (incompatible ranking)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/benkeks/icore-ranks.git
cd icore-ranks
```

2. Set up the project Python environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Activate the environment for future sessions:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Usage

### 1) Collect rankings
```bash
python3 collect_rankings.py
```
Outputs `rankings_data.json` (all editions, all fields). Rate limiting is built in.

### 2) Generate visualizations
```bash
python3 visualize_rankings.py
```
Creates plots for FoR 4613 conferences that appear in ICORE2026.

## Output Files
- `rankings_data.json`
- `rank_distribution.png`
- `conference_rank_changes.png`
- `rank_transitions.png`
- `rank_migrations.png`

## Data Structure (JSON)
```json
{
  "id": "2172",
  "title": "Conference Full Name",
  "acronym": "CONF",
  "rank": "A*",
  "source": "ICORE2026",
  "for_codes": ["4613", "4605"]
}
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
- Visualizations ignore ERA2010.
- Adjust filtering inside `visualize_rankings.py` if you want a different FoR or edition.

## License

This project is open source. The CORE ranking data belongs to CORE (Computing Research and Education Association of Australasia).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.