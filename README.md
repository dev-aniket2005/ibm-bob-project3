# 🏠 House Price Prediction & Business Intelligence Dashboard

> **Executive Summary:** An end-to-end real estate analytics platform that transforms raw housing data into actionable business intelligence. The system ingests, cleans, and engineers features from a 50,000-row housing dataset, trains a RandomForestRegressor achieving **R² > 0.99**, and surfaces insights through a four-section interactive Streamlit dashboard covering KPI monitoring, AI price prediction, exploratory analysis, and strategic decision support.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Tech Stack](#tech-stack)
4. [Dataset](#dataset)
5. [Quick Start](#quick-start)
6. [Application Sections](#application-sections)
7. [ML Model Details](#ml-model-details)
8. [Project Structure](#project-structure)
9. [Business Value](#business-value)

---

## 🎯 Project Overview

| Attribute       | Detail                                                   |
|-----------------|----------------------------------------------------------|
| **Domain**      | Real Estate & House Price Prediction                     |
| **Objective**   | Clean dirty housing data → EDA → ML Model → BI Dashboard |
| **Model**       | RandomForestRegressor (200 trees, depth 12)              |
| **UI**          | Streamlit multi-page interactive dashboard               |
| **Output**      | Price prediction, KPI metrics, strategic recommendations |

---

## 🏗️ System Architecture

```
house_price_regression_dataset.csv
            │
            ▼
    ┌─────────────────┐
    │  Data Cleaning   │  ← Duplicate removal, median imputation,
    │  & Engineering   │    outlier capping, domain validation
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Feature Engineer │  ← House Age, Price/SqFt, Luxury Index,
    │                  │    Total Rooms, SqFt per Room
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │  RandomForestRegressor          │
    │  (n_estimators=200, depth=12)   │
    │  Train 80% │ Test 20%           │
    └────────┬────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │       Streamlit Dashboard       │
    │  • Executive BI Dashboard       │
    │  • AI Price Predictor           │
    │  • Exploratory Analysis         │
    │  • Strategic Decision Matrix    │
    └─────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer            | Technology                                        |
|------------------|---------------------------------------------------|
| **Language**     | Python 3.10+                                      |
| **Frontend UI**  | Streamlit 1.32+                                   |
| **Data Wrangling** | Pandas, NumPy                                   |
| **ML Framework** | scikit-learn (RandomForestRegressor)              |
| **Visualizations** | Plotly Express, Plotly Graph Objects            |
| **Statistical**  | SciPy, Seaborn (supporting analysis)              |
| **Deployment**   | Local / Streamlit Community Cloud                 |

---

## 📊 Dataset

**Source:** Kaggle — House Price Regression Dataset  
**Direct Link:** [https://www.kaggle.com/datasets/prokshitha/home-value-insights-house-price-regression](https://www.kaggle.com/datasets/prokshitha/home-value-insights-house-price-regression)

| Column                 | Type    | Description                              |
|------------------------|---------|------------------------------------------|
| `Square_Footage`       | float   | Interior living area in sq ft            |
| `Num_Bedrooms`         | int     | Number of bedrooms                       |
| `Num_Bathrooms`        | int     | Number of bathrooms                      |
| `Year_Built`           | int     | Year of original construction            |
| `Lot_Size`             | float   | Land area in acres                       |
| `Garage_Size`          | int     | Garage capacity (0–3 cars)               |
| `Neighborhood_Quality` | int     | Quality score 1–10                       |
| `House_Price`          | float   | **Target variable** — market value (USD) |

### Engineered Features

| Feature           | Formula                                      |
|-------------------|----------------------------------------------|
| `House_Age`       | `2024 - Year_Built`                          |
| `Price_Per_SqFt`  | `House_Price / Square_Footage`               |
| `Total_Rooms`     | `Num_Bedrooms + Num_Bathrooms`               |
| `Sqft_Per_Room`   | `Square_Footage / (Total_Rooms + 1)`         |
| `Luxury_Index`    | `NQ × 0.4 + Garage × 10 + Square_Footage/100` |
| `Neighborhood_Tier` | Categorical bin: Budget/Mid-Range/Premium/Luxury |

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# 1. Clone or download the project files
#    Ensure house_price_regression_dataset.csv is in the same directory

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run house_price_app.py
```

The app opens automatically at **http://localhost:8501**

### File Structure Required

```
📁 project-directory/
├── house_price_app.py               ← Main application (run this)
├── house_price_regression_dataset.csv ← Dataset (must be present)
├── requirements.txt                 ← Python dependencies
├── README.md                        ← This file
└── project_report.docx              ← Project report
```

---

## 📱 Application Sections

### 1. 📊 Executive Dashboard
Top-level KPI cards showing **Average Price**, **Price/SqFt**, **Median Price**, **Model R² Score**, and **Mean Absolute Error**. Accompanied by four Plotly charts: price distribution histogram, neighborhood tier bar chart, price vs. square footage scatter, and actual vs. predicted model validation.

### 2. 🤖 AI Price Predictor
Interactive sliders for all 7 property attributes. Real-time price valuation on button click, including:
- Predicted price with confidence range
- Price gauge showing market percentile position
- Business advisory banners (buy signal, premium flag, renovation warning)

### 3. 📈 Exploratory Analysis
Four-tab deep-dive:
- **Distributions:** Box plots and violin charts by tier, garage, bedrooms, age
- **Correlations:** Full feature correlation heatmap + top drivers bar chart
- **Neighborhood Analysis:** NQ score rankings, bubble chart, tier summary table
- **Feature Importance:** RandomForest importance bar chart with key drivers

### 4. 🧭 Strategic Decision Matrix
Business intelligence layer:
- Revenue opportunities (3 cards)
- Market risks (3 cards)
- Strategic recommendations (3 cards)
- Price trend by construction decade
- Full decision matrix table

---

## 🤖 ML Model Details

```
Model:          RandomForestRegressor
n_estimators:   200
max_depth:      12
min_samples_split: 5
min_samples_leaf: 2
Train/Test:     80% / 20% split (random_state=42)
```

| Metric | Description                        |
|--------|------------------------------------|
| **R²** | Proportion of variance explained   |
| **MAE** | Mean Absolute Error (USD)         |
| **RMSE** | Root Mean Squared Error (USD)    |
| **MAPE** | Mean Absolute Percentage Error   |

---

## 💼 Business Value

| Stakeholder       | Value Delivered                                                    |
|-------------------|--------------------------------------------------------------------|
| **Investors**     | AI-assisted valuation, below-market opportunity detection          |
| **Developers**    | Identify high-ROI segments (NQ uplift, renovation plays)           |
| **Agents**        | Reduce time-on-market via accurate pricing (within ±MAPE%)         |
| **Policymakers**  | Track aging housing stock, affordability gaps, tier distribution   |

---

## 📄 License

This project is created for educational and business intelligence demonstration purposes.

---

*Built with ❤️ using Python, Streamlit, and scikit-learn.*
