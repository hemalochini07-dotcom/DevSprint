# Campaign Sentinel

Adaptive Campaign Anomaly Detection and Intelligent Alert Management System.

## Run in VS Code

1. Open this folder in VS Code.
2. Open Terminal → New Terminal.
3. Create and activate a virtual environment (Windows PowerShell):

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL shown in the terminal, usually http://localhost:8501.

## Pages

- Overview
- Campaign Monitoring
- Smart Alerts
- Incident Analysis
- Alert Performance
- Settings

## Data

The app starts with labelled synthetic data for a demonstration. Upload a CSV from the sidebar to use your own data. Expected columns:

- date
- campaign_id
- spend
- clicks
- impressions
- conversions

Optional column: `ground_truth_incident` (true/false, 1/0, yes/no).

The detector uses a campaign-specific rolling historical median and MAD-based deviation rule. Treat this as a prototype baseline; evaluate carefully before any production use.


## Organiser-provided dataset
The project includes `data/daily_campaign_metrics.csv` and loads it automatically on startup. It contains 3,600 rows (20 campaigns × 180 days) with campaign metadata and USD metrics. The app uses date, campaign_id, spend, clicks, impressions, and conversions for detection; it retains channel, vertical, landing_page_group, campaign_name, and revenue for analysis. Since this dataset does not include ground-truth incident labels, precision/recall cannot be measured honestly and will display N/A.
