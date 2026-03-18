import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------- Page Config ----------------------------
st.set_page_config(page_title="Portfolio Risk Monitor & Rebalancing Assistant", layout="wide")
st.title("📊 Portfolio Risk Monitor & Rebalancing Assistant")
st.markdown("Enter your stock/ETF holdings to get real-time risk metrics and rebalancing suggestions.")

# ---------------------------- Sidebar Input ----------------------------
with st.sidebar:
    st.header("Holdings Information")
    tickers_input = st.text_input(
        "Stock Codes (separated by commas)",
        "000001.SZ,600519.SS,000858.SZ",
        help="For A-shares, suffix required: .SZ for Shenzhen, .SS for Shanghai"
    )
    weights_input = st.text_input(
        "Target Weights (%, separated by commas)",
        "40,30,30",
        help="e.g., 40,30,30 means target weights of 40%, 30%, 30% respectively"
    )
    shares_input = st.text_input(
        "Number of Shares Held (separated by commas)",
        "100,50,20",
        help="Enter the number of shares for each stock"
    )

    threshold = st.slider(
        "Rebalancing Threshold (deviation %)",
        min_value=1, max_value=20, value=5, step=1,
        help="When a stock's weight deviates from target beyond this threshold, a rebalancing alert is triggered"
    )

    st.markdown("---")
    st.header("TuShare Settings")
    tushare_token = st.text_input("Enter your TuShare token", type="password")
    if not tushare_token:
        st.warning("Please enter your TuShare token to fetch real-time data")
        st.stop()
    # Strip any accidental whitespace
    tushare_token = tushare_token.strip()
    ts.set_token(tushare_token)
    pro = ts.pro_api()

    st.markdown("---")
    st.caption(
        "💡 Target weights are your intended asset allocation. Rebalancing brings actual weights back to target to control risk.")

# ---------------------------- Parse Input ----------------------------
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
try:
    target_weights = [float(w.strip()) / 100 for w in weights_input.split(",") if w.strip()]
    shares = [float(s.strip()) for s in shares_input.split(",") if s.strip()]
except:
    st.error("❌ Please ensure target weights and share numbers are numeric and separated by commas.")
    st.stop()

if not (len(tickers) == len(target_weights) == len(shares)):
    st.error("❌ Number of stock codes, target weights, and share counts must be equal!")
    st.stop()

if abs(sum(target_weights) - 1.0) > 0.01:
    st.warning("⚠️ Target weights should sum to approximately 100%. Please check your input.")


# ---------------------------- Data Fetching Functions (with caching) ----------------------------
@st.cache_data(ttl=3600)
def fetch_tushare_data(tickers, start_date, end_date):
    """Fetch daily close prices for the given tickers from TuShare."""
    all_data = {}
    for ticker in tickers:
        try:
            df = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date, fields='trade_date,close')
            if df.empty:
                st.warning(f"⚠️ No data for {ticker}, please check the code.")
                continue
            df = df.set_index('trade_date')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            all_data[ticker] = df['close']
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch data for {ticker}: {e}")
            continue
    if not all_data:
        return None
    return pd.DataFrame(all_data)


@st.cache_data(ttl=3600)
def fetch_benchmark(start_date, end_date):
    """Fetch CSI300 index data for Beta calculation (may be N/A if no permission)."""
    try:
        df = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date, fields='trade_date,close')
        if df.empty:
            return None
        df = df.set_index('trade_date')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df['close']
    except:
        return None


# ---------------------------- Fetch Data ----------------------------
end_date = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')

with st.spinner("Fetching real-time data from TuShare..."):
    price_data = fetch_tushare_data(tickers, start_date, end_date)
    if price_data is None:
        st.error("❌ Unable to fetch any stock data. Please check your token, network, or stock codes.")
        st.stop()
    benchmark = fetch_benchmark(start_date, end_date)

# Get latest prices
latest_prices = price_data.iloc[-1]
if isinstance(latest_prices, pd.Series):
    latest_prices = latest_prices.values
else:
    latest_prices = [latest_prices]

# Calculate current market value and weights
current_values = shares * latest_prices
total_value = sum(current_values)
if total_value == 0:
    st.error("❌ Total market value is zero. Please check share counts or prices.")
    st.stop()
current_weights = [v / total_value for v in current_values]

# ---------------------------- Risk Metrics Calculation ----------------------------
returns = price_data.pct_change().dropna()
if returns.empty:
    st.error("❌ Unable to calculate returns – insufficient data points.")
    st.stop()

portfolio_returns = (returns * current_weights).sum(axis=1)

# Annualized volatility
annual_vol = portfolio_returns.std() * np.sqrt(252)

# Maximum drawdown
cumulative = (1 + portfolio_returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

# Sharpe ratio
sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

# Concentration (HHI)
hhi = sum(w ** 2 for w in current_weights)

# Beta (if benchmark available)
if benchmark is not None:
    benchmark_returns = benchmark.pct_change().dropna()
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
    if len(aligned) > 0:
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        var_market = aligned.iloc[:, 1].var()
        beta = cov / var_market if var_market != 0 else np.nan
    else:
        beta = np.nan
else:
    beta = np.nan

# VaR (95%)
var_95 = np.percentile(portfolio_returns, 5)

# ---------------------------- Display Results ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Current Holdings")
    df_display = pd.DataFrame({
        "Stock": tickers,
        "Target Weight": [f"{w * 100:.1f}%" for w in target_weights],
        "Current Weight": [f"{w * 100:.1f}%" for w in current_weights],
        "Deviation": [f"{(c - t) * 100:+.1f}%" for c, t in zip(current_weights, target_weights)],
        "Shares": shares,
        "Latest Price": [f"¥{p:.2f}" if 'SS' in t or 'SZ' in t else f"${p:.2f}" for p, t in
                         zip(latest_prices, tickers)],
        "Market Value": [f"¥{v:.2f}" if 'SS' in t or 'SZ' in t else f"${v:.2f}" for v, t in
                         zip(current_values, tickers)]
    })
    st.dataframe(df_display, use_container_width=True)
    st.metric(
        "Total Portfolio Value",
        f"¥{total_value:.2f}" if any('SS' in t or 'SZ' in t for t in tickers) else f"${total_value:.2f}",
        help="Total current market value of all holdings"
    )

with col2:
    st.subheader("📈 Risk Metrics")
    col2a, col2b, col2c = st.columns(3)
    col2a.metric("Annual Volatility", f"{annual_vol:.2%}", help="Measures price fluctuation; higher means more risk")
    col2b.metric("Max Drawdown", f"{max_drawdown:.2%}", help="Largest peak-to-trough decline in history")
    col2c.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", help="Excess return per unit of risk (higher is better)")

    col2d, col2e, col2f = st.columns(3)
    col2d.metric("Concentration (HHI)", f"{hhi:.3f}",
                 help="Close to 1 means highly concentrated; near 0 means diversified")
    col2e.metric("Beta", f"{beta:.2f}" if not np.isnan(beta) else "N/A",
                 help="Sensitivity to market (requires index permission)")
    col2f.metric("VaR (95%, daily)", f"{var_95:.2%}", help="Maximum expected loss in a day with 95% confidence")

# ---------------------------- Weight Visualization ----------------------------
st.subheader("🥧 Weight Comparison")
col_pie1, col_pie2 = st.columns(2)
with col_pie1:
    fig_target = px.pie(names=tickers, values=target_weights, title="Target Weights", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_target, use_container_width=True)
with col_pie2:
    fig_current = px.pie(names=tickers, values=current_weights, title="Current Weights", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_current, use_container_width=True)

deviations = [c - t for c, t in zip(current_weights, target_weights)]
fig_bar = go.Figure(
    data=[go.Bar(x=tickers, y=deviations, marker_color=['red' if d > 0 else 'green' for d in deviations])])
fig_bar.update_layout(title="Deviation (Current - Target)", xaxis_title="Stock", yaxis_title="Deviation",
                      yaxis_tickformat='.1%')
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------- Rebalancing Suggestions ----------------------------
st.subheader("⚖️ Rebalancing Suggestions")
deviations_detail = [(t, target, current, current - target) for t, target, current in
                     zip(tickers, target_weights, current_weights)]
alerts = [(t, tgt, cur, dev) for t, tgt, cur, dev in deviations_detail if abs(dev) > threshold / 100]

if alerts:
    st.warning(f"The following assets deviate from target by more than {threshold}%. Consider rebalancing:")
    for ticker, target, current, dev in alerts:
        action = "SELL" if dev > 0 else "BUY"
        st.markdown(
            f"- **{ticker}**: target {target:.1%}, current {current:.1%}, deviation {dev:+.2%} → suggested **{action}** to return to target.")

    # Simple simulation
    st.markdown("**📉 What if you do NOT rebalance?**")
    annual_returns = (price_data.iloc[-1] / price_data.iloc[0] - 1).values
    future_values = current_values * (1 + annual_returns)
    future_total = sum(future_values)
    future_weights = future_values / future_total
    st.markdown("Assuming each stock repeats its past year's performance, weights in one year could become:")
    for t, fw in zip(tickers, future_weights):
        st.markdown(f"- {t}: {fw:.1%}")
    st.info("If deviations widen, portfolio risk may significantly deviate from your original intention.")
else:
    st.success("All assets are within the target weight range. No rebalancing needed.")

# ---------------------------- Historical Charts ----------------------------
st.subheader("📉 Portfolio Historical NAV (Normalized)")
st.line_chart(cumulative / cumulative.iloc[0])

st.subheader("📈 Individual Stock Prices (Normalized)")
st.line_chart(price_data / price_data.iloc[0])

# ---------------------------- Footer ----------------------------
st.markdown("---")
st.caption(
    "Disclaimer: This tool is for informational purposes only and does not constitute investment advice. Data source: TuShare. Beta may be N/A if index data is unavailable.")