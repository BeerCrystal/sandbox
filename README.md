# Upstate New York Hospital Cost Analysis

This tool analyzes hospital admission costs, comparing Q1 2025 performance against Q4 2024 and all quarters in 2024, with a focus on Upstate New York. It produces both summary statistics and executive-level visualizations, and supports both sample and real data.

## Features
- **Automated sample data generation** (for demo/testing)
- **Claim line rollup** to claim level (by Claim ID)
- **Quarterly and provider-level analysis**
- **Top 5 inpatient provider deep dive** (outlier and statistical analysis)
- **Executive summary visualizations** (PNG charts)
- **All outputs saved in timestamped folders**

## Setup

1. **Install Python 3.8+** (if not already installed)
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### To run with sample data (default):
```bash
python analysis_runs/YYYYMMDD_HHMMSS_Q1_2025_Cost_Analysis/analyze_costs.py
```
- Each run creates a new folder in `analysis_runs/` with all outputs.

### To use your own data:
- Replace or adapt the `generate_sample_data` method in `analyze_costs.py` to load your file (CSV/Excel) and ensure it has the following columns:
  - `Claim_ID` (unique per claim)
  - `Provider_ID`
  - `Admission_Date` (YYYY-MM-DD)
  - `DRG`, `DRG_Weight`, `Total_Paid`, `Discharge_Status`, `Patient_Type`, `Service_Type`
- If you need help with this, ask for a custom loader function.

## Outputs
Each run creates a timestamped folder in `analysis_runs/` containing:
- `quarterly_metrics.csv` — Metrics by quarter and provider
- `summary_statistics.csv` — Summary stats by quarter
- `top_providers_detailed.csv` — Deep dive on top 5 inpatient providers
- `top_providers_outliers.csv` — Outlier analysis for top providers
- `top_providers_statistical_tests.csv` — Statistical test results for top providers
- **Visualizations (PNGs):**
  - `quarterly_cost_trends.png` — Average cost per admission by quarter
  - `provider_cost_distribution.png` — Cost distribution for top 5 providers
  - `drg_weight_vs_cost.png` — DRG weight vs. total cost
  - `q4_q1_comparison.png` — Q4 2024 vs Q1 2025 cost comparison
  - `base_rate_trends.png` — Average base rate by quarter
  - `service_type_costs.png` — Cost by service type

## Analysis Details
- **Claim-level rollup:** All claim lines are summed by `Claim_ID`.
- **Metrics:** Average DRG, base rate (total paid/DRG weight), and more.
- **Top 5 inpatient providers:** Outlier detection and t-tests for cost changes.
- **Visuals:** All major trends and comparisons are visualized for executive review.

## Portability
- All dependencies are in `requirements.txt`.
- No hardcoded or OS-specific paths.
- Works on any OS with Python 3.8+ and listed packages.

## Customization
- To use real data, adapt the data loading section in `analyze_costs.py`.
- To add new metrics or visuals, extend the relevant methods in the script.

## Support
If you need help adapting the script for your data or want to add new features, just ask Brad!