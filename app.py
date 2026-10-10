import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from data_utils import make_demo_data, validate_data
from detector import detect_anomalies, evaluate_detection

st.set_page_config(
    page_title="Campaign Sentinel",
    page_icon="◉",
    layout="wide"
)

from ui_theme import apply_theme
apply_theme()


# --------------------------------
# CACHE: Reuse detection results
# --------------------------------
@st.cache_data(show_spinner=False)
def get_campaign_results(df):
    alerts = detect_anomalies(df)
    metrics = evaluate_detection(df, alerts)
    return alerts, metrics


# --------------------------------
# SIDEBAR
# --------------------------------
st.sidebar.markdown("# ◉ Campaign Sentinel")
st.sidebar.caption("Adaptive anomaly detection")
st.sidebar.markdown("---")
st.sidebar.markdown("### Data source")

uploaded = st.sidebar.file_uploader(
    "Upload campaign CSV",
    type=["csv"],
    help="CSV should contain a date, campaign identifier and campaign metrics."
)

if uploaded is not None:
    try:
        source_df = pd.read_csv(uploaded)
        source_df, messages = validate_data(source_df)

        st.session_state["campaign_df"] = source_df
        st.session_state["data_source"] = "Uploaded CSV"

        for message in messages:
            st.sidebar.info(message)

    except Exception as exc:
        st.sidebar.error(f"Could not read this CSV: {exc}")


# --------------------------------
# LOAD DATASET
# --------------------------------
if "campaign_df" not in st.session_state:

    dataset_path = (
        Path(__file__).parent
        / "data"
        / "daily_campaign_metrics.csv"
    )

    if dataset_path.exists():
        try:
            st.session_state["campaign_df"] = pd.read_csv(
                dataset_path
            )
            st.session_state["data_source"] = (
                "Organiser-provided PS-04 dataset"
            )

        except Exception as exc:
            st.sidebar.error(
                f"Could not load the included dataset: {exc}"
            )
            st.session_state["campaign_df"] = make_demo_data()
            st.session_state["data_source"] = (
                "Synthetic demo fallback"
            )

    else:
        st.session_state["campaign_df"] = make_demo_data()
        st.session_state["data_source"] = (
            "Synthetic demo fallback"
        )


# --------------------------------
# VALIDATE DATA AND GET RESULTS
# --------------------------------
df, validation_messages = validate_data(
    st.session_state["campaign_df"].copy()
)

st.session_state["campaign_df"] = df

# Cached results avoid repeating expensive calculations
# when the same dataset is used again.
alerts, metrics = get_campaign_results(df)


# --------------------------------
# HERO SECTION
# --------------------------------
st.markdown("""
<div class="cs-hero">
  <div class="cs-eyebrow">
    PS-04 · Intelligent Marketing Operations
  </div>
  <h1>Campaign <span>Sentinel</span></h1>
  <div class="cs-pill">
    Adaptive anomaly alerting · without alert fatigue
  </div>
  <p>
    See beyond the noise. Detect unusual campaign behaviour,
    understand what changed, and prioritise the incidents
    that deserve attention — powered by your organiser-provided
    campaign dataset.
  </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------
# MAIN METRICS
# --------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Campaigns",
    f"{df['campaign_id'].nunique():,}"
)

c2.metric(
    "Total spend (USD)",
    f"${df['spend'].sum():,.0f}"
)

c3.metric(
    "Total conversions",
    f"{int(df['conversions'].sum()):,}"
)

c4.metric(
    "Unique alerts",
    f"{len(alerts):,}"
)


# --------------------------------
# ALERT METRICS
# --------------------------------
c5, c6, c7, c8 = st.columns(4)

c5.metric(
    "Critical",
    int((alerts["severity"] == "Critical").sum())
    if not alerts.empty else 0
)

c6.metric(
    "Warnings",
    int((alerts["severity"] == "Warning").sum())
    if not alerts.empty else 0
)

c7.metric(
    "Precision",
    f"{metrics['precision']:.1%}"
    if metrics["precision"] is not None else "N/A"
)

c8.metric(
    "Recall",
    f"{metrics['recall']:.1%}"
    if metrics["recall"] is not None else "N/A"
)


# --------------------------------
# CHARTS
# --------------------------------
left, right = st.columns([1.5, 1])

with left:
    st.subheader("Spend and conversion trend")

    trend = df.groupby(
        "date",
        as_index=False
    ).agg(
        spend=("spend", "sum"),
        conversions=("conversions", "sum")
    )

    fig = px.line(
        trend,
        x="date",
        y=["spend", "conversions"],
        markers=False,
        color_discrete_sequence=["#38bdf8", "#34d399"]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text=""
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with right:
    st.subheader("Alerts by severity")

    if alerts.empty:
        st.success(
            "No anomalies detected in the current data."
        )

    else:
        sev = (
            alerts["severity"]
            .value_counts()
            .rename_axis("severity")
            .reset_index(name="count")
        )

        fig2 = px.pie(
            sev,
            names="severity",
            values="count",
            hole=0.58,
            color="severity",
            color_discrete_map={
                "Critical": "#fb7185",
                "Warning": "#fbbf24",
                "Informational": "#38bdf8"
            }
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10)
        )

        st.plotly_chart(
            fig2,
            width="stretch"
        )


# --------------------------------
# LATEST ALERTS
# --------------------------------
st.subheader("Latest alerts")

if alerts.empty:
    st.info(
        "No alerts to show. Upload data or use the demo data to explore the system."
    )

else:
    st.dataframe(
        alerts.head(8),
        width="stretch",
        hide_index=True
    )


# --------------------------------
# ABOUT THE DATA
# --------------------------------
with st.expander("About the demo data and evaluation"):

    st.write(
        f"Current source: "
        f"**{st.session_state.get('data_source', 'Unknown')}**."
    )

    st.write(
        "If you have not uploaded the organiser's dataset, "
        "the application uses labelled synthetic data. "
        "The default source is the organiser-provided "
        "daily_campaign_metrics.csv. This dataset has no "
        "ground-truth incident labels, so precision and recall "
        "are shown as N/A rather than guessed."
    )

    st.write(
        "Use the pages in the left navigation to explore "
        "campaign monitoring, smart alerts, incident analysis, "
        "and evaluation."
    )
# --------------------------------
# ADDITIONAL CAMPAIGN GRAPHS
# --------------------------------

st.subheader("📊 Campaign Performance Analytics")

# Channel-wise performance
if "channel" in df.columns:

    left, right = st.columns(2)

    with left:
        st.markdown("#### 💰 Spend by Channel")

        channel_spend = (
            df.groupby("channel", as_index=False)
            .agg(spend=("spend", "sum"))
            .sort_values("spend", ascending=False)
        )

        fig_channel = px.bar(
            channel_spend,
            x="channel",
            y="spend",
            color="channel",
            title="Total Spend Across Channels",
            labels={
                "channel": "Marketing Channel",
                "spend": "Total Spend (USD)"
            },
            template="plotly_dark"
        )

        fig_channel.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_channel,
            width="stretch"
        )

    with right:
        if "revenue" in df.columns:

            st.markdown("#### 🎯 Revenue by Channel")

            channel_revenue = (
                df.groupby("channel", as_index=False)
                .agg(revenue=("revenue", "sum"))
                .sort_values("revenue", ascending=False)
            )

            fig_revenue = px.bar(
                channel_revenue,
                x="channel",
                y="revenue",
                color="channel",
                title="Total Revenue Across Channels",
                labels={
                    "channel": "Marketing Channel",
                    "revenue": "Total Revenue (USD)"
                },
                template="plotly_dark"
            )

            fig_revenue.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig_revenue,
                width="stretch"
            )


# Campaign-wise anomaly count
if not alerts.empty:

    st.markdown("#### 🚨 Campaign Alert Ranking")

    campaign_alerts = (
        alerts.groupby("campaign_id", as_index=False)
        .size()
        .rename(columns={"size": "alert_count"})
        .sort_values("alert_count", ascending=False)
        .head(10)
    )

    fig_alerts = px.bar(
        campaign_alerts,
        x="alert_count",
        y="campaign_id",
        orientation="h",
        color="alert_count",
        title="Top 10 Campaigns by Alert Count",
        labels={
            "campaign_id": "Campaign",
            "alert_count": "Number of Alerts"
        },
        template="plotly_dark"
    )

    fig_alerts.update_layout(
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig_alerts,
        width="stretch"
    )