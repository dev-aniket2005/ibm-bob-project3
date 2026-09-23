# =============================================================================
#  House Price Prediction & Business Intelligence Dashboard
#  Dataset: house_price_regression_dataset.csv
#  Run: python -m streamlit run Aniket_house_price_app.py
#  Requires: streamlit (recent), pandas, numpy, plotly, scikit-learn,
#             statsmodels (only for the scatter-plot trendline)
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

try:  # px.scatter(trendline="ols") needs statsmodels; degrade gracefully
    import statsmodels.api  # noqa: F401

    OLS_TRENDLINE: str | None = "ols"
except ImportError:
    OLS_TRENDLINE = None

st.set_page_config(
    page_title="House Price BI Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
#  CONFIGURATION  (every tunable number lives here)
# =============================================================================
DATA_FILENAME = "house_price_regression_dataset.csv"
TARGET = "House_Price"
REFERENCE_YEAR = date.today().year  # single source of truth for "House_Age"

RAW_FEATURES = [
    "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built",
    "Lot_Size", "Garage_Size", "Neighborhood_Quality",
]
ENGINEERED_FEATURES = ["House_Age", "Total_Rooms", "Sqft_Per_Room", "Luxury_Index"]
MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
NUMERIC_COLS = RAW_FEATURES + [TARGET]

# Luxury index: NQ * 0.4 + garage * 10 + sqft / 100
LUXURY_NQ_WEIGHT, LUXURY_GARAGE_WEIGHT, LUXURY_SQFT_DIVISOR = 0.4, 10, 100

IQR_FENCE = 3  # rows outside Q1 - 3*IQR .. Q3 + 3*IQR of price are dropped

TIER_BINS = [0, 3, 6, 8, 10]
TIER_LABELS = ["Budget (1-3)", "Mid-Range (4-6)", "Premium (7-8)", "Luxury (9-10)"]
LUXURY_TIER = TIER_LABELS[-1]
MID_TIER, PREMIUM_TIER = TIER_LABELS[1], TIER_LABELS[2]

AGE_BINS = [-1, 10, 25, 50, 75, np.inf]
AGE_LABELS = ["0-10y", "11-25y", "26-50y", "51-75y", "76y+"]

RF_PARAMS = dict(
    n_estimators=200, max_depth=12, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1,
)
TEST_SIZE, RANDOM_STATE = 0.2, 42

# Advisory / narrative thresholds
MARKET_BAND_PCT = 20        # ± vs. market average considered "aligned"
PREMIUM_NQ = 8
RENOVATION_AGE = 40
RENOVATION_BUFFER = "5–15%"
LARGE_SQFT = 3500
AGING_STOCK_AGE = 50

BLUE, RED, GREEN = "#3b82f6", "#ef4444", "#22c55e"
SET2 = px.colors.qualitative.Set2
px.defaults.template = "plotly_white"

CSS = """
<style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background: linear-gradient(135deg, #f8fbff 0%, #edf5ff 100%);
        border: 1px solid #bfd9ff;
        border-left: 5px solid #3b82f6;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 3px 10px rgba(59,130,246,0.10);
        color: #0f172a;
    }
    .stMetric > div {
        background: transparent;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: #1e3a5f !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #334155 !important;
    }
    .section-header { font-size:1.4rem; font-weight:700; color:#1e3a5f;
                      border-bottom:3px solid #3b82f6; padding-bottom:6px;
                      margin-top:24px; margin-bottom:16px; }
    .insight-box { background:#eff6ff; border-left:4px solid #3b82f6;
                   padding:12px 16px; border-radius:6px; margin:8px 0;
                   font-size:0.93rem; color:#1e3a5f; }
    .risk-box    { background:#fff1f2; border-left:4px solid #ef4444;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .opp-box     { background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
                   border:1px solid #bbf7d0; border-left:4px solid #22c55e;
                   padding:12px 16px; border-radius:8px; margin:8px 0;
                   color:#14532d; }
    .action-box  { background: linear-gradient(135deg, #fffdf0 0%, #fefce8 100%);
                   border:1px solid #fef08a; border-left:4px solid #eab308;
                   padding:12px 16px; border-radius:8px; margin:8px 0;
                   color:#713f12; }
    .risk-box    { background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
                   border:1px solid #fecdd3; border-left:4px solid #ef4444;
                   padding:12px 16px; border-radius:8px; margin:8px 0;
                   color:#7f1d1d; }
    .insight-box { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
                   border:1px solid #bfdbfe; border-left:4px solid #3b82f6;
                   padding:12px 16px; border-radius:8px; margin:8px 0;
                   color:#1e3a5f; }
    .pred-result { background: linear-gradient(135deg,#0f172a,#1e3a5f);
                   color:white; border-radius:14px; padding:28px;
                   text-align:center; margin:16px 0; }
    div[data-testid="stSidebar"] { background-color:#1e3a5f; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] p { color:#e2e8f0 !important; }
</style>
"""


# =============================================================================
#  SMALL HELPERS
# =============================================================================
def money(x: float) -> str:
    return "n/a" if pd.isna(x) else f"${x:,.0f}"


def pct(x: float, digits: int = 1, signed: bool = False) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:+.{digits}f}%" if signed else f"{x:.{digits}f}%"


def pct_gap(a: float, b: float) -> float:
    """Percentage difference of a relative to b (NaN-safe)."""
    return np.nan if pd.isna(a) or pd.isna(b) or b == 0 else (a - b) / b * 100


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def callout(kind: str, body: str) -> None:
    """Coloured info box. kind: insight | risk | opp | action."""
    st.markdown(f'<div class="{kind}-box">{body}</div>', unsafe_allow_html=True)


def show_chart(fig: go.Figure, height: int = 340, **layout) -> None:
    fig.update_layout(template="plotly_white", height=height, **layout)
    st.plotly_chart(fig, width="stretch")


def categorical_int(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, list[str]]:
    """Treat a small integer column as discrete categories for box/violin plots."""
    out = df.assign(**{col: df[col].astype(int).astype(str)})
    return out, sorted(out[col].unique(), key=int)


# =============================================================================
#  DATA LOADING & CLEANING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from raw columns (shared by training and inference)."""
    out = df.copy()
    out["House_Age"] = REFERENCE_YEAR - out["Year_Built"]
    out["Total_Rooms"] = out["Num_Bedrooms"] + out["Num_Bathrooms"]
    out["Sqft_Per_Room"] = out["Square_Footage"] / (out["Total_Rooms"] + 1)
    out["Luxury_Index"] = (
        out["Neighborhood_Quality"] * LUXURY_NQ_WEIGHT
        + out["Garage_Size"] * LUXURY_GARAGE_WEIGHT
        + out["Square_Footage"] / LUXURY_SQFT_DIVISOR
    )
    return out


def find_dataset() -> Path | None:
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        candidate = base / DATA_FILENAME
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner="Loading data…")
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates()

    missing = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    # 1. Coerce to numeric; unparseable values -> NaN -> column median
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())

    # 2. Drop extreme price outliers (IQR fence)
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[df[TARGET].between(q1 - IQR_FENCE * iqr, q3 + IQR_FENCE * iqr)]

    # 3. Enforce domain constraints (NQ must fall inside the tier bins)
    df = df[
        (df["Square_Footage"] > 0)
        & (df["Num_Bedrooms"] >= 0)
        & (df["Num_Bathrooms"] >= 0)
        & (df["Year_Built"] >= 1800)
        & (df[TARGET] > 0)
        & (df["Neighborhood_Quality"] > TIER_BINS[0])
        & (df["Neighborhood_Quality"] <= TIER_BINS[-1])
    ]

    # 4. Feature engineering + neighbourhood tier
    df = engineer_features(df)
    df["Price_Per_SqFt"] = df[TARGET] / df["Square_Footage"]  # analytics only, not a model input
    df["Neighborhood_Tier"] = pd.cut(
        df["Neighborhood_Quality"], bins=TIER_BINS, labels=TIER_LABELS, right=True
    )
    return df.reset_index(drop=True)


# =============================================================================
#  MODEL
# =============================================================================
@dataclass(frozen=True)
class ModelBundle:
    model: RandomForestRegressor
    r2: float
    mae: float
    rmse: float
    mape: float  # percent
    importance: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray


@st.cache_resource(show_spinner="Training model…")
def train_model(df: pd.DataFrame) -> ModelBundle:
    X, y = df[MODEL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(**RF_PARAMS).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importance = (
        pd.DataFrame({"Feature": MODEL_FEATURES, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return ModelBundle(
        model=model,
        r2=float(r2_score(y_test, y_pred)),
        mae=float(mean_absolute_error(y_test, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        mape=float(mean_absolute_percentage_error(y_test, y_pred)) * 100,
        importance=importance,
        y_test=y_test,
        y_pred=y_pred,
    )


def predict_price(bundle: ModelBundle, raw_inputs: dict[str, float]) -> float:
    """Predict from raw attributes, reusing the exact training-time feature pipeline."""
    row = engineer_features(pd.DataFrame([raw_inputs]))
    return float(bundle.model.predict(row[MODEL_FEATURES])[0])


@dataclass(frozen=True)
class MarketStats:
    avg_price: float
    median_price: float
    min_price: float
    max_price: float
    avg_ppsf: float
    n_rows: int

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> MarketStats:
        return cls(
            avg_price=df[TARGET].mean(),
            median_price=df[TARGET].median(),
            min_price=df[TARGET].min(),
            max_price=df[TARGET].max(),
            avg_ppsf=df["Price_Per_SqFt"].mean(),
            n_rows=len(df),
        )


# =============================================================================
#  PAGE 1 — EXECUTIVE DASHBOARD
# =============================================================================
def render_dashboard(df: pd.DataFrame, bundle: ModelBundle, stats: MarketStats) -> None:
    section_header("📊 Executive BI Dashboard")
    st.caption("KPIs derived from the cleaned housing data and the trained ML model.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏷️ Avg House Price", money(stats.avg_price), help="Market benchmark (mean price)")
    c2.metric("📐 Avg Price / Sq Ft", f"${stats.avg_ppsf:.0f}", help="Unit value")
    c3.metric("📍 Median Price", money(stats.median_price), help="50th percentile")
    c4.metric("🎯 Model R² Score", f"{bundle.r2:.4f}", help="Prediction accuracy on the hold-out set")
    c5.metric("⚠️ Mean Abs Error", money(bundle.mae), help="Average prediction error on the hold-out set")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### House Price Distribution")
        fig = px.histogram(
            df, x=TARGET, nbins=60, color_discrete_sequence=[BLUE],
            labels={TARGET: "Price (USD)"},
        )
        fig.add_vline(x=stats.avg_price, line_dash="dash", line_color=RED,
                      annotation_text=f"Mean ${stats.avg_price / 1e3:.0f}K")
        fig.add_vline(x=stats.median_price, line_dash="dot", line_color=GREEN,
                      annotation_text=f"Median ${stats.median_price / 1e3:.0f}K")
        show_chart(fig, margin=dict(t=20, b=20))

    with col_b:
        st.markdown("#### Avg Price by Neighborhood Tier")
        tier_df = (df.groupby("Neighborhood_Tier", observed=True)[TARGET]
                     .mean().reset_index(name="Avg_Price"))
        fig = px.bar(
            tier_df, x="Neighborhood_Tier", y="Avg_Price", color="Avg_Price",
            color_continuous_scale="Blues",
            labels={"Avg_Price": "Avg Price (USD)", "Neighborhood_Tier": "Neighborhood Tier"},
            category_orders={"Neighborhood_Tier": TIER_LABELS},
        )
        show_chart(fig, margin=dict(t=20, b=20), coloraxis_showscale=False)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Price vs. Square Footage")
        fig = px.scatter(
            df.sample(min(1000, len(df)), random_state=RANDOM_STATE),
            x="Square_Footage", y=TARGET, color="Neighborhood_Tier",
            opacity=0.65, trendline=OLS_TRENDLINE,
            labels={"Square_Footage": "Sq Ft", TARGET: "Price (USD)",
                    "Neighborhood_Tier": "Tier"},
            color_discrete_sequence=SET2,
            category_orders={"Neighborhood_Tier": TIER_LABELS},
        )
        show_chart(fig, margin=dict(t=20, b=20))

    with col_d:
        st.markdown("#### Model: Actual vs. Predicted Prices")
        actual, predicted = bundle.y_test.to_numpy(), bundle.y_pred
        lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers", name="Predictions",
                                 marker=dict(color=BLUE, opacity=0.5, size=5)))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect Fit",
                                 line=dict(color=RED, dash="dash", width=2)))
        show_chart(fig, xaxis_title="Actual Price (USD)", yaxis_title="Predicted Price (USD)",
                   margin=dict(t=20, b=20))

    st.markdown("---")
    st.markdown("#### 📋 Model Performance Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{bundle.r2:.4f}", help=f"{bundle.r2 * 100:.1f}% of price variance explained")
    m2.metric("MAE", money(bundle.mae), help="Average absolute prediction error")
    m3.metric("RMSE", money(bundle.rmse), help="Root mean squared error")
    m4.metric("MAPE", f"{bundle.mape:.2f}%", help="Mean absolute percentage error")


# =============================================================================
#  PAGE 2 — AI PRICE PREDICTOR
# =============================================================================
def data_slider(df: pd.DataFrame, label: str, col: str, *, step: float, help: str) -> float:
    """Slider whose range/default come from the training data (a random forest cannot extrapolate)."""
    is_int = isinstance(step, int)
    lo, hi = df[col].min(), df[col].max()
    lo, hi = (int(np.floor(lo)), int(np.ceil(hi))) if is_int else (round(float(lo), 1), round(float(hi), 1))
    default = lo + round((df[col].median() - lo) / step) * step  # snap median onto the slider grid
    default = min(max(default, lo), hi)
    default = int(default) if is_int else round(float(default), 1)
    return st.slider(label, lo, hi, default, step=step, help=help)


def render_predictor(df: pd.DataFrame, bundle: ModelBundle, stats: MarketStats) -> None:
    section_header("🤖 Interactive AI Price Predictor")
    st.caption("Adjust the property attributes below to receive an instant AI-powered valuation. "
               "Slider ranges match the data the model was trained on.")

    col_inp, col_out = st.columns(2)

    with col_inp:
        st.markdown("#### 🏡 Property Attributes")
        with st.form("predictor_form"):  # a form keeps the result stable while sliders move
            sqft = data_slider(df, "Square Footage", "Square_Footage", step=50,
                               help="Total interior living area in square feet.")
            beds = data_slider(df, "Bedrooms", "Num_Bedrooms", step=1,
                               help="Number of bedrooms.")
            baths = data_slider(df, "Bathrooms", "Num_Bathrooms", step=1,
                                help="Number of bathrooms.")
            nq = data_slider(df, "Neighborhood Quality", "Neighborhood_Quality", step=1,
                             help="Low values = low-end area, high values = prime luxury area.")
            year = data_slider(df, "Year Built", "Year_Built", step=1,
                               help="Year the house was originally constructed.")
            lot = data_slider(df, "Lot Size (acres)", "Lot_Size", step=0.1,
                              help="Total land area in acres.")
            garage = data_slider(df, "Garage Size (cars)", "Garage_Size", step=1,
                                 help="Capacity of attached garage.")
            submitted = st.form_submit_button("🔮 Predict House Price", type="primary",
                                              width="stretch")

    with col_out:
        st.markdown("#### 💰 AI Valuation")
        if submitted:
            pred = predict_price(bundle, {
                "Square_Footage": sqft, "Num_Bedrooms": beds, "Num_Bathrooms": baths,
                "Year_Built": year, "Lot_Size": lot, "Garage_Size": garage,
                "Neighborhood_Quality": nq,
            })
            house_age = REFERENCE_YEAR - year
            diff_avg = (pred - stats.avg_price) / stats.avg_price * 100
            percentile = (df[TARGET] < pred).mean() * 100
            band_lo, band_hi = pred * (1 - bundle.mape / 100), pred * (1 + bundle.mape / 100)

            st.markdown(f"""
            <div class="pred-result">
                <h2 style="margin:0;font-size:2.4rem;">{money(pred)}</h2>
                <p style="font-size:1rem;opacity:0.8;margin:4px 0;">Estimated Market Value</p>
                <p style="font-size:0.85rem;opacity:0.65;">
                    Typical error band (±MAPE): {money(band_lo)} – {money(band_hi)}</p>
            </div>
            """, unsafe_allow_html=True)

            v1, v2, v3 = st.columns(3)
            v1.metric("Price / Sq Ft", f"${pred / sqft:.0f}")
            v2.metric("vs Market Avg", f"{diff_avg:+.1f}%")
            v3.metric("House Age", f"{house_age} yrs")

            # Gauge: axis always extends far enough to contain the prediction
            q33, q66 = df[TARGET].quantile([0.33, 0.66])
            axis_max = max(stats.max_price, pred * 1.05)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                delta={"reference": stats.avg_price, "valueformat": "$,.0f",
                       "increasing": {"color": GREEN}, "decreasing": {"color": RED}},
                number={"valueformat": "$,.0f", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [stats.min_price, axis_max]},
                    "bar": {"color": BLUE},
                    "steps": [
                        {"range": [stats.min_price, q33], "color": "#dbeafe"},
                        {"range": [q33, q66], "color": "#93c5fd"},
                        {"range": [q66, axis_max], "color": BLUE},
                    ],
                    "threshold": {"line": {"color": RED, "width": 3}, "value": stats.avg_price},
                },
                title={"text": f"Market position: {percentile:.0f}th percentile", "font": {"size": 14}},
            ))
            show_chart(fig, height=280, margin=dict(t=40, b=20))

            st.markdown("#### 📋 Business Advisory")
            if diff_avg > MARKET_BAND_PCT:
                callout("risk", "⚠️ <b>High Valuation:</b> This property is priced significantly "
                                "above market average. Verify comparable sales before committing.")
            elif diff_avg < -MARKET_BAND_PCT:
                callout("opp", "✅ <b>Below Market Value:</b> Strong buy opportunity — consider "
                               "acquisition before market correction.")
            else:
                callout("insight", f"📊 <b>Market-Aligned:</b> Property is within "
                                   f"±{MARKET_BAND_PCT}% of market average — fairly priced segment.")
            if nq >= PREMIUM_NQ:
                callout("opp", "🏆 <b>Premium Location:</b> High Neighborhood Quality drives strong "
                               "appreciation potential and rental yield.")
            if house_age > RENOVATION_AGE:
                callout("action", f"🔧 <b>Renovation Flag:</b> Older construction may require capital "
                                  f"expenditure — factor a {RENOVATION_BUFFER} renovation cost buffer.")
        else:
            st.info("👈 Adjust the sliders and click **Predict House Price** to get a valuation.")
            st.markdown("#### 📊 Market Reference Stats")
            st.dataframe(pd.DataFrame({
                "Metric": ["Min Price", "Max Price", "Avg Price", "Median Price",
                           "Avg Sq Ft", "Avg Price/SqFt"],
                "Value": [money(stats.min_price), money(stats.max_price),
                          money(stats.avg_price), money(stats.median_price),
                          f"{df['Square_Footage'].mean():,.0f} sq ft", f"${stats.avg_ppsf:.0f}"],
            }), width="stretch", hide_index=True)


# =============================================================================
#  PAGE 3 — EXPLORATORY ANALYSIS
# =============================================================================
def importance_cards(rows: pd.DataFrame, kind: str) -> None:
    for _, row in rows.iterrows():
        callout(kind, f"<b>{row['Feature']}</b> — {row['Importance'] * 100:.1f}% relative importance")


def render_eda(df: pd.DataFrame, bundle: ModelBundle, stats: MarketStats) -> None:
    section_header("📈 Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 Distributions", "🔗 Correlations", "🏘️ Neighborhood Analysis", "🌲 Feature Importance",
    ])
    price_label = {TARGET: "Price (USD)"}

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df, x="Neighborhood_Tier", y=TARGET, color="Neighborhood_Tier",
                         labels=price_label, color_discrete_sequence=SET2,
                         category_orders={"Neighborhood_Tier": TIER_LABELS},
                         title="Price Distribution by Neighborhood Tier")
            show_chart(fig, 370, showlegend=False)
        with c2:
            garage_df, garage_order = categorical_int(df, "Garage_Size")
            fig = px.violin(garage_df, x="Garage_Size", y=TARGET, color="Garage_Size",
                            box=True, points=False,
                            labels={**price_label, "Garage_Size": "Garage Capacity"},
                            category_orders={"Garage_Size": garage_order},
                            title="Price by Garage Size")
            show_chart(fig, 370, showlegend=False)

        c3, c4 = st.columns(2)
        with c3:
            beds_df, beds_order = categorical_int(df, "Num_Bedrooms")
            fig = px.box(beds_df, x="Num_Bedrooms", y=TARGET, color="Num_Bedrooms",
                         labels={**price_label, "Num_Bedrooms": "Bedrooms"},
                         category_orders={"Num_Bedrooms": beds_order},
                         title="Price by Number of Bedrooms")
            show_chart(fig, 340, showlegend=False)
        with c4:
            age_df = df.assign(Age_Group=pd.cut(df["House_Age"], bins=AGE_BINS, labels=AGE_LABELS))
            fig = px.box(age_df, x="Age_Group", y=TARGET, color="Age_Group",
                         labels={**price_label, "Age_Group": "House Age Group"},
                         category_orders={"Age_Group": AGE_LABELS},
                         title="Price by House Age Group")
            show_chart(fig, 340, showlegend=False)

    with tab2:
        # Price_Per_SqFt (derived from the target) and House_Age (mirror of Year_Built)
        # are left out: they would only add trivial / redundant correlations.
        corr = df[[TARGET, *RAW_FEATURES, "Luxury_Index"]].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}", textfont={"size": 10},
        ))
        show_chart(fig, 520, title="Feature Correlation Matrix", margin=dict(t=50, b=20, l=20, r=20))

        corr_hp = (corr[TARGET].drop(TARGET)
                   .sort_values(key=abs, ascending=False))
        st.markdown("##### Top Feature Correlations with House Price")
        fig = px.bar(x=corr_hp.values, y=corr_hp.index, orientation="h",
                     color=corr_hp.values, color_continuous_scale="RdBu",
                     labels={"x": "Pearson Correlation", "y": "Feature"})
        fig.update_yaxes(autorange="reversed")  # strongest at the top
        show_chart(fig, 360, coloraxis_showscale=False, margin=dict(t=10, b=20))

    with tab3:
        nq_df = (df.groupby("Neighborhood_Quality")
                   .agg(Avg_Price=(TARGET, "mean"), Median_Price=(TARGET, "median"),
                        Count=(TARGET, "count"), Avg_PpSF=("Price_Per_SqFt", "mean"))
                   .reset_index())
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(nq_df, x="Neighborhood_Quality", y="Avg_Price", color="Avg_Price",
                         color_continuous_scale="Blues",
                         labels={"Neighborhood_Quality": "NQ Score", "Avg_Price": "Avg Price (USD)"},
                         title="Average Price by Neighborhood Quality Score")
            show_chart(fig, 370, coloraxis_showscale=False)
        with c2:
            fig = px.scatter(nq_df, x="Neighborhood_Quality", y="Avg_Price", size="Count",
                             color="Avg_PpSF", color_continuous_scale="Viridis",
                             labels={"Neighborhood_Quality": "NQ Score", "Avg_Price": "Avg Price (USD)",
                                     "Avg_PpSF": "$/Sq Ft", "Count": "# Listings"},
                             title="NQ Score vs. Avg Price (bubble = listing count)")
            show_chart(fig, 370)

        tier_summary = (df.groupby("Neighborhood_Tier", observed=True)
                          .agg(Listings=(TARGET, "count"), **{
                              "Avg Price": (TARGET, "mean"), "Median": (TARGET, "median"),
                              "Min": (TARGET, "min"), "Max": (TARGET, "max"),
                              "Avg Sq Ft": ("Square_Footage", "mean")})
                          .rename_axis("Tier").reset_index())
        st.markdown("##### Neighborhood Tier Summary Table")
        st.dataframe(  # Styler keeps numeric sorting while formatting for display
            tier_summary.style.format({
                "Avg Price": "${:,.0f}", "Median": "${:,.0f}", "Min": "${:,.0f}",
                "Max": "${:,.0f}", "Avg Sq Ft": "{:,.0f} sq ft",
            }),
            width="stretch", hide_index=True,
        )

    with tab4:
        fig = px.bar(bundle.importance, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues",
                     labels={"Importance": "Relative Importance"},
                     title="RandomForest Feature Importance")
        fig.update_yaxes(autorange="reversed")  # most important at the top
        show_chart(fig, 440, coloraxis_showscale=False)
        st.caption("Year_Built and House_Age carry identical information, so tree models split "
                   "importance between them — a low score can mean 'redundant', not 'useless'.")

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown("##### 🔑 Top 3 Price Drivers")
            importance_cards(bundle.importance.head(3), "insight")
        with col_i2:
            st.markdown("##### 🔻 Least Influential Features")
            importance_cards(bundle.importance.tail(3), "action")


# =============================================================================
#  PAGE 4 — STRATEGIC DECISION MATRIX
# =============================================================================
def render_strategy(df: pd.DataFrame, bundle: ModelBundle, stats: MarketStats) -> None:
    section_header("🧭 Strategic Decision Matrix")
    st.caption("BI-driven insights for real estate developers, investors, and policymakers.")

    # ── Metrics computed once from the data ───────────────────────────────────
    price = df[TARGET]
    nq_slope = np.polyfit(df["Neighborhood_Quality"], np.log(price), 1)[0]
    nq_uplift = np.expm1(nq_slope) * 100                       # % price change per NQ point
    nq_two_pt = stats.avg_price * np.expm1(2 * nq_slope)       # $ effect of +2 NQ on the average home
    nq_rank = bundle.importance["Feature"].tolist().index("Neighborhood_Quality") + 1

    large_avg = price[df["Square_Footage"] > LARGE_SQFT].mean()
    luxury = df[df["Neighborhood_Tier"] == LUXURY_TIER]
    newer_avg = price[df["House_Age"] <= 25].mean()
    midage_avg = price[(df["House_Age"] > 25) & (df["House_Age"] <= 50)].mean()
    age_gap = pct_gap(midage_avg, newer_avg)
    aging_pct = (df["House_Age"] > AGING_STOCK_AGE).mean() * 100
    price_std = price.std()
    cv = price_std / stats.avg_price * 100
    no_garage = price[df["Garage_Size"] == 0].mean()
    with_garage = price[df["Garage_Size"] > 0].mean()
    garage_premium = pct_gap(with_garage, no_garage)

    # ── Revenue opportunities ─────────────────────────────────────────────────
    st.markdown("### 💚 Revenue Opportunities")
    cols = st.columns(3)
    with cols[0]:
        callout("opp",
                f"<b>🏘️ Neighborhood Quality Premium</b><br>"
                f"Each +1 NQ point is associated with about <b>{pct(nq_uplift, signed=True)}</b> price change "
                f"(log-linear fit); NQ ranks <b>#{nq_rank}</b> of {len(bundle.importance)} model "
                f"features by importance. Investors should target areas with improving NQ scores "
                f"ahead of gentrification cycles.")
    with cols[1]:
        if pd.isna(large_avg):
            body = (f"No listings exceed {LARGE_SQFT:,} sq ft in this dataset, so the large-format "
                    f"segment cannot be sized from this data.")
        else:
            body = (f"Homes over {LARGE_SQFT:,} sq ft command an average of <b>{money(large_avg)}</b>, "
                    f"representing premium segment demand. Developers targeting executive buyers "
                    f"should prioritize this footprint.")
        callout("opp", f"<b>📐 Large Format Properties</b><br>{body}")
    with cols[2]:
        callout("opp",
                f"<b>💎 Luxury Tier Dominance</b><br>"
                f"Luxury tier properties (NQ 9-10) average <b>{money(luxury[TARGET].mean())}</b>. "
                f"Only {len(luxury):,} listings occupy this tier — scarcity drives premium pricing.")

    # ── Market risks ──────────────────────────────────────────────────────────
    st.markdown("### 🔴 Market Risks")
    cols = st.columns(3)
    with cols[0]:
        callout("risk",
                f"<b>🏚️ Aging Housing Stock</b><br>"
                f"<b>{aging_pct:.1f}%</b> of listings are {AGING_STOCK_AGE}+ years old. Buyers face "
                f"latent renovation costs of $30K–$150K. This inflates perceived affordability while "
                f"masking true total cost of ownership.")
    with cols[1]:
        callout("risk",
                f"<b>📉 Price Volatility</b><br>"
                f"Coefficient of variation: <b>{cv:.1f}%</b>. Std deviation of {money(price_std)} "
                f"indicates wide price spread. Investors relying on comparable sales analysis must "
                f"account for high inter-market variance.")
    with cols[2]:
        if pd.isna(garage_premium):
            body = "Garage premium cannot be computed from this dataset."
        else:
            verb, noun = ("command a", "premium") if garage_premium >= 0 else ("sell at a", "discount")
            body = (f"Properties with garages {verb} <b>{pct(abs(garage_premium))} {noun}</b> versus "
                    f"those without. Over-reliance on this feature in urban walkable zones may "
                    f"expose investors to repricing as mobility preferences shift.")
        callout("risk", f"<b>🚗 Garage Dependency Risk</b><br>{body}")
    st.caption("Renovation-cost ranges and capex buffers are planning assumptions, "
               "not values derived from the dataset.")

    # ── Strategic recommendations ─────────────────────────────────────────────
    st.markdown("### 🟡 Strategic Recommendations")
    cols = st.columns(3)
    with cols[0]:
        callout("action",
                f"<b>🎯 Invest in NQ Uplift</b><br>"
                f"Partner with municipalities on community development projects that improve "
                f"Neighborhood Quality scores. A 2-point NQ increase is associated with about "
                f"<b>{money(nq_two_pt)}</b> on the average-priced home.")
    with cols[1]:
        if age_gap < 0:
            tail = f"(<b>{pct(age_gap, signed=True)}</b>) — the gap a renovation would aim to close."
        else:
            tail = (f"(<b>{pct(age_gap, signed=True)}</b>) — no age discount shows up in this data, so "
                    f"validate renovation upside locally before deploying capital.")
        callout("action",
                f"<b>🔄 Renovation ROI Model</b><br>"
                f"Deploy capital into 26–50 year old homes. In this data they average "
                f"<b>{money(midage_avg)}</b> vs {money(newer_avg)} for homes up to 25 years old {tail}")
    with cols[2]:
        callout("action",
                f"<b>📊 AI-Assisted Pricing</b><br>"
                f"Integrate this RandomForest model into listing workflows. With "
                f"R²={bundle.r2:.3f} and MAPE of {bundle.mape:.1f}%, agents can price within "
                f"±{money(bundle.mae)} of market on average — reducing overpricing risk and "
                f"time-on-market.")

    # ── Trend line ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📅 Average Price by Decade Built")
    decade_df = (df.assign(Decade=(df["Year_Built"] // 10) * 10)
                   .groupby("Decade")[TARGET].mean().reset_index(name="Avg_Price"))
    fig = px.line(decade_df, x="Decade", y="Avg_Price", markers=True,
                  labels={"Decade": "Decade Built", "Avg_Price": "Avg Price (USD)"},
                  title="Market Value Trend by Construction Decade",
                  color_discrete_sequence=[BLUE])
    show_chart(fig, margin=dict(t=40, b=20))

    # ── Decision matrix table ─────────────────────────────────────────────────
    st.markdown("#### 🗂️ Full Strategic Decision Matrix")
    matrix = pd.DataFrame({
        "Dimension": ["Location Premium", "Square Footage", "House Age",
                      "Garage Access", "Market Timing", "AI Pricing"],
        "Opportunity": [f"High NQ → {pct(nq_uplift, signed=True)} per NQ point",
                        f"{LARGE_SQFT:,}+ sq ft → exec market",
                        "Renovate mid-age stock",
                        f"Garage effect {pct(garage_premium, 0, signed=True)}",
                        "Buy pre-gentrification",
                        f"Avg error {bundle.mape:.1f}% (MAPE)"],
        "Risk": ["NQ may lag infrastructure", "Supply glut in large format",
                 "Hidden capex exposure", "Mobility shift may deflate",
                 "Macro interest rate headwinds", "Model drift over time"],
        "Action": ["Monitor NQ scores quarterly", "Segment buyer profiles",
                   f"Budget {RENOVATION_BUFFER} capex reserve", "Diversify property types",
                   "Track Fed rate decisions", "Retrain model bi-annually"],
        "Priority": ["High", "Medium", "High", "Medium", "High", "Medium"],
    })
    st.dataframe(matrix, width="stretch", hide_index=True)

    st.markdown("---")
    st.caption(f"📌 KPIs and model metrics are computed from {stats.n_rows:,} cleaned house price "
               f"records (RandomForest hold-out: R² {bundle.r2:.4f} | MAE {money(bundle.mae)} | "
               f"MAPE {bundle.mape:.2f}%). Qualitative recommendations are strategic guidance.")


# =============================================================================
#  APP ENTRY POINT
# =============================================================================
PAGES = {
    "📊 Executive Dashboard": render_dashboard,
    "🤖 AI Price Predictor": render_predictor,
    "📈 Exploratory Analysis": render_eda,
    "🧭 Strategic Decision Matrix": render_strategy,
}


def render_sidebar(bundle: ModelBundle, stats: MarketStats) -> str:
    with st.sidebar:
        st.markdown("## 🏠 Navigation")
        page = st.radio("Select Section", list(PAGES), label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### 📁 Dataset Info")
        st.markdown(f"- **Records:** {stats.n_rows:,}")
        st.markdown(f"- **Model inputs:** {len(RAW_FEATURES)} raw + {len(ENGINEERED_FEATURES)} engineered")
        st.markdown("- **Target:** House Price (USD)")

        st.markdown("---")
        st.markdown("### 🎯 Model")
        st.markdown("**RandomForestRegressor**")
        st.markdown(f"- R² Score: `{bundle.r2:.4f}`")
        st.markdown(f"- MAE: `{money(bundle.mae)}`")
        st.markdown(f"- RMSE: `{money(bundle.rmse)}`")
        st.markdown(f"- MAPE: `{bundle.mape:.2f}%`")
    return page


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    data_path = find_dataset()
    if data_path is None:
        st.error(f"Dataset `{DATA_FILENAME}` not found. Place it next to this script "
                 f"or in the folder you launch Streamlit from.")
        st.stop()

    try:
        df = load_and_clean_data(str(data_path))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if df.empty:
        st.error("No rows left after cleaning — check the dataset contents.")
        st.stop()

    bundle = train_model(df)
    stats = MarketStats.from_df(df)
    page = render_sidebar(bundle, stats)
    PAGES[page](df, bundle, stats)


main()