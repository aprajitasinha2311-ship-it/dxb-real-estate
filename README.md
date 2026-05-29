# 🏙️ Dubai Real Estate as an Inflation Hedge — Dashboard

**Research:** *Real Estate as an Inflation Hedge: An Empirical Analysis of Dubai's Housing Market (2010–2025)*  
**Author:** Aprajita Sinha · MS25GF019 · MGB OCT 2025  

---

## Project Scope

This interactive Streamlit dashboard supports postgraduate research examining whether
Dubai's residential real estate market acts as an effective inflation hedge. The study
covers 2010–2025 using time-series data on property prices, CPI, FDI, transaction
volumes, and global monetary conditions.

**Core Finding:** Dubai RE is a *partial, conditional* inflation hedge. FDI inflows and
transaction volume are empirically stronger price drivers than domestic CPI.

---

## Repository Architecture

```
dubai_real_estate_dashboard/
├── app.py                      # Streamlit multi-tab application
├── data_gen.py                 # Dataset generator & validation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── dubai_realestate_2010_2025.csv    # Calibrated synthetic dataset
├── src/
│   ├── eda.py                  # EDA module (descriptive stats, heatmaps)
│   └── models.py               # ML pipeline (OLS, Logit, K-Means, Assoc.)
└── figures/                    # Auto-generated EDA charts (after running eda.py)
```

---

## Installation & Local Run

```bash
# 1. Clone or unzip the repository
cd dubai_real_estate_dashboard

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset
python data_gen.py

# 5. (Optional) Run EDA to generate figures
python src/eda.py

# 6. Launch the dashboard
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| 🏠 Project Overview | Hypotheses, variable dictionary, economic timeline |
| 📊 Descriptive Analytics | Dynamic KPIs, interactive charts, correlation heatmap, data table |
| 🤖 Modelling Results | OLS regression, logistic classification, K-Means clustering, association rules |
| 🧠 AI Tools Hub | AI Purchase Predictor + AI Analyst Chat (Claude / offline fallback) |
| 💼 Business Recommendations | Executive guidance for developers, investors, fintechs, policymakers |

---

## AI Chat — Anthropic API (Optional)

1. Obtain an API key from [console.anthropic.com](https://console.anthropic.com)
2. Enter it in the **sidebar → API Key** field
3. The chat upgrades from offline keyword-matching to Claude 3.5 Sonnet

Without an API key, the chat uses a built-in keyword-matching engine pre-loaded with
research-specific responses covering all key topics.

---

## Streamlit Community Cloud Deployment

1. Push this repository to a public GitHub repo
2. Visit [share.streamlit.io](https://share.streamlit.io) → New App
3. Select `app.py` as the entrypoint
4. (Optional) Add `ANTHROPIC_API_KEY` as a Secret in Settings → Secrets:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Deploy – the app will be live at `your-app.streamlit.app`

---

## Key Data Sources

| Variable | Source |
|----------|--------|
| RPPI (AED/sqft) | Dubai Land Department; CBRE (2024) |
| Rental Index | CBRE; Dubai Statistics Center |
| CPI Inflation | Federal Competitiveness & Statistics Centre (2024) |
| Fed Funds Rate | FRED; IMF World Economic Outlook |
| FDI Inflows | IMF (2024); UAE Central Bank |
| Transactions | Dubai Land Department; Primo Capital (2024) |

---

## References (Key)

- Muckenhaupt, Hoesli & Zhu (2025) – Real estate as an inflation hedge
- Aveline-Dubach (2015) – Global capital flows & urban real estate
- Douissa et al. (2025) – Media narratives & Dubai RE investment
- Maghyereh (2024) – Real estate bubbles & systemic risk in UAE
- Fisher (1930); Fama & Schwert (1977) – Inflation hedging theory
