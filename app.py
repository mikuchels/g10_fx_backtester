"""
Phase 6 — Streamlit Dashboard.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    DEFAULT_START, DEFAULT_END, DAYS_PER_YEAR,
    CARRY_LONG_N, CARRY_SHORT_N,
    MOMENTUM_LOOKBACKS, MR_WINDOW, MR_ZSCORE_THRESHOLD,
    TARGET_VOLATILITY,
)
from data import load_all_data
from strategies import carry_signals, momentum_signals, mean_reversion_signals
from engine import (
    run_backtest,
    performance_table,
    drawdown_series,
    monthly_returns_matrix,
    strategy_correlation,
    combine_strategies,
)


# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title="G10 FX Backtester", page_icon="💱", layout="wide")
st.title("💱 G10 FX Strategy Backtester")
st.caption("Carry · Time-Series Momentum · Mean Reversion")

STRATEGY_COLORS = {
    "Carry": "#1f77b4",
    "Momentum": "#ff7f0e",
    "Mean Reversion": "#2ca02c",
    "Combined": "#d62728",
}


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    fred_key = st.text_input(
        "FRED API Key",
        type="password",
        help="Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
    )

    st.subheader("Date Range")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start", value=pd.Timestamp(DEFAULT_START))
    end_date = c2.date_input("End", value=pd.Timestamp(DEFAULT_END))

    st.subheader("Carry")
    carry_long = st.slider("Long top N", 1, 5, CARRY_LONG_N)
    carry_short = st.slider("Short bottom N", 1, 5, CARRY_SHORT_N)

    st.subheader("Momentum")
    use_1m = st.checkbox("1-month lookback (21d)", value=True)
    use_3m = st.checkbox("3-month lookback (63d)", value=True)
    use_12m = st.checkbox("12-month lookback (252d)", value=True)

    st.subheader("Mean Reversion")
    mr_win = st.slider("Window (days)", 5, 30, MR_WINDOW)
    mr_z = st.slider("Z-score threshold", 0.5, 3.0, MR_ZSCORE_THRESHOLD, 0.1)

    st.subheader("Backtest Settings")
    use_tc = st.checkbox("Apply transaction costs", value=True)
    use_vol = st.checkbox("Volatility targeting", value=True)
    tgt_vol = st.slider("Target vol (%)", 5, 25, int(TARGET_VOLATILITY * 100)) / 100.0
    combo = st.selectbox("Combination method", ["equal", "inverse_vol"])

    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)


# ── Guard: need API key ─────────────────────────────────────────
if not fred_key:
    st.info(
        "👈 Enter your **FRED API key** in the sidebar to get started.  \n"
        "Get a free key → [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)"
    )
    st.stop()


# ── Data loading (cached) ───────────────────────────────────────
@st.cache_data(show_spinner="Downloading market data…")
def _load(api_key, start, end):
    return load_all_data(api_key, str(start), str(end), use_cache=True)


# ── Run backtest ─────────────────────────────────────────────────
if run_btn or "results" not in st.session_state:
    try:
        data = _load(fred_key, start_date, end_date)
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    returns_df = data["returns"]
    carry_df = data["carry"]

    # Momentum lookbacks
    mom_lbs = []
    if use_1m:
        mom_lbs.append(21)
    if use_3m:
        mom_lbs.append(63)
    if use_12m:
        mom_lbs.append(252)
    if not mom_lbs:
        mom_lbs = [63]

    # Signals
    sig_carry = carry_signals(carry_df, carry_long, carry_short)
    sig_mom = momentum_signals(returns_df, mom_lbs)
    sig_mr = mean_reversion_signals(returns_df, mr_win, mr_z)

    # Backtests
    bt_carry = run_backtest(sig_carry, returns_df, carry_df, use_tc, use_vol, tgt_vol)
    bt_mom = run_backtest(sig_mom, returns_df, carry_df, use_tc, use_vol, tgt_vol)
    bt_mr = run_backtest(sig_mr, returns_df, carry_df, use_tc, use_vol, tgt_vol)

    results = {"Carry": bt_carry, "Momentum": bt_mom, "Mean Reversion": bt_mr}
    combined = combine_strategies(results, method=combo)

    st.session_state["data"] = data
    st.session_state["results"] = results
    st.session_state["combined"] = combined

# Retrieve
data = st.session_state.get("data")
results = st.session_state.get("results")
combined = st.session_state.get("combined")
if not results:
    st.stop()


# ── KPI cards ────────────────────────────────────────────────────
st.markdown("---")
kpi_cols = st.columns(4)

for idx, (name, color) in enumerate(STRATEGY_COLORS.items()):
    r = (combined["net_returns"] if name == "Combined" else results[name]["net_returns"]).dropna()
    ann_ret = r.mean() * DAYS_PER_YEAR
    ann_vol = r.std() * np.sqrt(DAYS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = (1 + r).cumprod()
    mdd = (eq / eq.cummax() - 1).min()

    with kpi_cols[idx]:
        st.metric(name, f"Sharpe {sharpe:.2f}")
        st.caption(f"Ret {ann_ret:.1%} · Vol {ann_vol:.1%} · DD {mdd:.1%}")


# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Equity Curves", "📊 Tear Sheet", "🗓️ Monthly Returns", "🔗 Correlations", "💱 Positions"]
)


# ────────────────────────── TAB 1: Equity Curves ─────────────────
with tab1:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("Equity Curves (Net of Costs)", "Drawdowns"),
    )

    for name, color in STRATEGY_COLORS.items():
        eq = combined["equity_net"] if name == "Combined" else results[name]["equity_net"]
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq.values, name=name, line=dict(color=color, width=1.5)),
            row=1, col=1,
        )
        dd = drawdown_series(eq)
        fig.add_trace(
            go.Scatter(
                x=dd.index, y=dd.values, name=f"{name} DD",
                fill="tozeroy", line=dict(color=color, width=0.5),
                showlegend=False, opacity=0.4,
            ),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(height=650, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Optional gross vs net overlay
    if st.checkbox("Show gross vs net comparison"):
        fig2 = go.Figure()
        for name, color in list(STRATEGY_COLORS.items())[:3]:
            fig2.add_trace(go.Scatter(
                x=results[name]["equity_gross"].index,
                y=results[name]["equity_gross"].values,
                name=f"{name} Gross", line=dict(color=color, width=1.5),
            ))
            fig2.add_trace(go.Scatter(
                x=results[name]["equity_net"].index,
                y=results[name]["equity_net"].values,
                name=f"{name} Net", line=dict(color=color, width=1.5, dash="dot"),
            ))
        fig2.update_layout(height=400, title="Gross vs Net — Impact of Transaction Costs")
        st.plotly_chart(fig2, use_container_width=True)


# ────────────────────────── TAB 2: Tear Sheet ────────────────────
with tab2:
    rows = []
    for name in ["Carry", "Momentum", "Mean Reversion"]:
        rows.append(performance_table(results[name]["net_returns"], name))
    rows.append(performance_table(combined["net_returns"], "Combined"))

    perf_df = pd.DataFrame(rows).set_index("Name")
    st.dataframe(perf_df, use_container_width=True)

    # TC summary
    st.subheader("Transaction Cost Drag")
    tc_cols = st.columns(3)
    for i, name in enumerate(["Carry", "Momentum", "Mean Reversion"]):
        tc = results[name]["tc_daily"]
        tc_cols[i].metric(name, f"{tc.sum():.4f} cumulative", f"{tc.mean() * DAYS_PER_YEAR:.2%} p.a.")

    # Export
    st.download_button(
        "📥 Download performance metrics (CSV)",
        perf_df.to_csv(), "performance_metrics.csv", "text/csv",
    )


# ────────────────────────── TAB 3: Monthly Returns ───────────────
with tab3:
    sel = st.selectbox(
        "Select strategy", ["Carry", "Momentum", "Mean Reversion", "Combined"], key="monthly_sel"
    )
    rets = combined["net_returns"] if sel == "Combined" else results[sel]["net_returns"]
    mtx = monthly_returns_matrix(rets)

    # Heatmap (exclude Annual column)
    z = mtx.iloc[:, :-1].values * 100
    x_labels = mtx.columns[:-1].tolist()
    y_labels = mtx.index.astype(str).tolist()
    text = [[f"{v:.1f}%" for v in row] for row in z]

    fig_hm = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels, text=text, texttemplate="%{text}",
        colorscale="RdYlGn", zmid=0, colorbar=dict(title="Return %"),
    ))
    fig_hm.update_layout(title=f"{sel} — Monthly Returns (%)", height=500)
    st.plotly_chart(fig_hm, use_container_width=True)

    # Annual bar chart
    annual = mtx["Annual"] * 100
    fig_ann = go.Figure(go.Bar(
        x=annual.index.astype(str), y=annual.values,
        marker_color=["#2ca02c" if v > 0 else "#d62728" for v in annual.values],
        text=[f"{v:.1f}%" for v in annual.values], textposition="auto",
    ))
    fig_ann.update_layout(title=f"{sel} — Annual Returns (%)", height=350, yaxis_title="Return %")
    st.plotly_chart(fig_ann, use_container_width=True)


# ────────────────────────── TAB 4: Correlations ──────────────────
with tab4:
    st.subheader("Daily Return Correlation Matrix")

    all_rets = {**{n: r for n, r in results.items()}, "Combined": combined}
    corr_df = pd.DataFrame(
        {n: r["net_returns"] for n, r in all_rets.items()}
    ).dropna().corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(), y=corr_df.index.tolist(),
        text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
        texttemplate="%{text}",
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
    ))
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Rolling correlations
    st.subheader("Rolling 252-Day Correlations")
    names = ["Carry", "Momentum", "Mean Reversion"]
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    pair_colors = ["#1f77b4", "#ff7f0e", "#9467bd"]

    fig_rc = go.Figure()
    for idx, (s1, s2) in enumerate(pairs):
        r1 = results[s1]["net_returns"]
        r2 = results[s2]["net_returns"]
        aligned = pd.concat([r1.rename(s1), r2.rename(s2)], axis=1).dropna()
        rc = aligned[s1].rolling(252).corr(aligned[s2])
        fig_rc.add_trace(go.Scatter(
            x=rc.index, y=rc.values, name=f"{s1} × {s2}",
            line=dict(color=pair_colors[idx], width=1.5),
        ))
    fig_rc.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_rc.update_layout(height=400, yaxis_title="Correlation")
    st.plotly_chart(fig_rc, use_container_width=True)

    # Sharpe comparison bar chart
    st.subheader("Diversification Benefit")
    sharpes = {}
    for n in names:
        r = results[n]["net_returns"].dropna()
        sharpes[n] = (r.mean() * DAYS_PER_YEAR) / (r.std() * np.sqrt(DAYS_PER_YEAR)) if r.std() > 0 else 0
    r_c = combined["net_returns"].dropna()
    sharpes["Combined"] = (r_c.mean() * DAYS_PER_YEAR) / (r_c.std() * np.sqrt(DAYS_PER_YEAR)) if r_c.std() > 0 else 0

    fig_sh = go.Figure(go.Bar(
        x=list(sharpes.keys()), y=list(sharpes.values()),
        marker_color=[STRATEGY_COLORS[n] for n in sharpes],
        text=[f"{v:.2f}" for v in sharpes.values()], textposition="auto",
    ))
    fig_sh.update_layout(height=350, yaxis_title="Sharpe Ratio",
                         title="Individual vs Combined Sharpe Ratios")
    st.plotly_chart(fig_sh, use_container_width=True)


# ────────────────────────── TAB 5: Positions ─────────────────────
with tab5:
    pos_sel = st.selectbox(
        "Select strategy", ["Carry", "Momentum", "Mean Reversion"], key="pos_sel"
    )
    positions = results[pos_sel]["positions"]

    # Weekly position heatmap
    pos_w = positions.resample("W").last()
    fig_pos = go.Figure(go.Heatmap(
        z=pos_w.T.values,
        x=pos_w.index.strftime("%Y-%m-%d").tolist(),
        y=pos_w.columns.tolist(),
        colorscale="RdBu", zmid=0, colorbar=dict(title="Weight"),
    ))
    fig_pos.update_layout(height=400, title=f"{pos_sel} — Position Weights Over Time",
                          xaxis_title="Date", yaxis_title="Currency")
    st.plotly_chart(fig_pos, use_container_width=True)

    # Latest positions
    st.subheader("Most Recent Positions")
    latest = positions.iloc[-1].sort_values(ascending=False)
    fig_lat = go.Figure(go.Bar(
        x=latest.index.tolist(), y=latest.values,
        marker_color=["#2ca02c" if v > 0 else "#d62728" for v in latest.values],
        text=[f"{v:.3f}" for v in latest.values], textposition="auto",
    ))
    fig_lat.update_layout(height=300, yaxis_title="Weight")
    st.plotly_chart(fig_lat, use_container_width=True)

    # PnL attribution (stacked area)
    st.subheader("Cumulative PnL Attribution by Currency")
    cum_pnl = results[pos_sel]["pnl_by_ccy"].cumsum()
    fig_attr = go.Figure()
    for ccy in cum_pnl.columns:
        fig_attr.add_trace(go.Scatter(
            x=cum_pnl.index, y=cum_pnl[ccy].values,
            name=ccy, line=dict(width=1.5), stackgroup="one",
        ))
    fig_attr.update_layout(height=400, yaxis_title="Cumulative Return")
    st.plotly_chart(fig_attr, use_container_width=True)

    # Export
    st.download_button(
        "📥 Download positions (CSV)",
        positions.to_csv(), f"{pos_sel.lower().replace(' ', '_')}_positions.csv", "text/csv",
    )


# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "G10 FX Strategy Backtester · Data: Yahoo Finance (FX Spot), FRED (Interest Rates) · "
    "Strategies: Carry, TSMOM, Mean Reversion · Built with Python, Streamlit, Plotly"
)