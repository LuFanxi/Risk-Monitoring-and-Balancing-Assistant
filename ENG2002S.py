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
st.markdown("Enter your stock/ETF holdings to get real‑time risk metrics and rebalancing suggestions.")

# ---------------------------- Read TuShare Token from Secrets ----------------------------
try:
    tushare_token = st.secrets["tushare_token"]
    ts.set_token(tushare_token)
    pro = ts.pro_api()
except KeyError:
    st.error("❌ tushare_token not found in Streamlit Secrets. Please set it during deployment.")
    st.stop()
except Exception as e:
    st.error(f"❌ TuShare initialization failed: {e}")
    st.stop()

# ---------------------------- Sidebar Input ----------------------------
with st.sidebar:
    st.header("Holdings Information")
    tickers_input = st.text_input(
        "Stock codes (comma‑separated)",
        "000001.SZ,600519.SS,000858.SZ",
        help="For A‑shares, add suffix: .SZ for Shenzhen, .SS for Shanghai"
    )
    weights_input = st.text_input(
        "Target weights (%, comma‑separated)",
        "40,30,30",
        help="e.g. 40,30,30 means 40%, 30%, 30% for the three stocks"
    )
    shares_input = st.text_input(
        "Number of shares held (comma‑separated)",
        "100,50,20",
        help="Enter the number of shares for each stock"
    )

    threshold = st.slider(
        "Rebalancing threshold (deviation %)",
        min_value=1, max_value=20, value=5, step=1,
        help="Alert when a stock's weight deviates from its target by more than this percentage"
    )

    st.markdown("---")
    st.caption(
        "💡 Target weights are your intended asset allocation. Rebalancing brings actual weights back to target to control risk.")

# ---------------------------- Parse Input ----------------------------
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
try:
    target_weights = [float(w.strip()) / 100 for w in weights_input.split(",") if w.strip()]
    shares = [float(s.strip()) for s in shares_input.split(",") if s.strip()]
except:
    st.error("❌ Please ensure target weights and share counts are numbers and separated by commas.")
    st.stop()

if not (len(tickers) == len(target_weights) == len(shares)):
    st.error("❌ The number of stock codes, target weights, and share counts must be the same!")
    st.stop()

if abs(sum(target_weights) - 1.0) > 0.01:
    st.warning("⚠️ Target weights should sum to approximately 100%. Please check your input.")


# ---------------------------- Data Fetching Functions (with caching) ----------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour to avoid redundant API calls
def fetch_tushare_data(tickers, start_date, end_date):
    """
    Fetch daily closing prices for a list of stocks from TuShare for the last year.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    all_data = {}
    for ticker in tickers:
        try:
            df = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date, fields='trade_date,close')
            if df.empty:
                st.warning(f"⚠️ No data for {ticker}. Please check the code.")
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
    """Fetch CSI 300 index (000300.SH) data for Beta calculation."""
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

with st.spinner("Fetching real‑time data from TuShare..."):
    price_data = fetch_tushare_data(tickers, start_date, end_date)
    if price_data is None:
        st.error("❌ Unable to retrieve any stock data. Please check your token, network, or stock codes.")
        st.stop()
    benchmark = fetch_benchmark(start_date, end_date)  # Optional; if fails, Beta will show N/A

# Latest prices
latest_prices = price_data.iloc[-1]
if isinstance(latest_prices, pd.Series):
    latest_prices = latest_prices.values
else:
    latest_prices = [latest_prices]

# Calculate current market value and weights
current_values = shares * latest_prices
total_value = sum(current_values)
if total_value == 0:
    st.error("❌ Total market value is zero. Check share counts or prices.")
    st.stop()
current_weights = [v / total_value for v in current_values]

# ---------------------------- Risk Metrics Calculation ----------------------------
# Daily returns
returns = price_data.pct_change().dropna()
if returns.empty:
    st.error("❌ Unable to calculate returns – insufficient data points.")
    st.stop()

# Portfolio returns (weighted by current weights)
portfolio_returns = (returns * current_weights).sum(axis=1)

# Annualized volatility
annual_vol = portfolio_returns.std() * np.sqrt(252)

# Maximum drawdown
cumulative = (1 + portfolio_returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

# Sharpe ratio (assuming risk‑free rate = 0)
sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

# Asset concentration (Herfindahl‑Hirschman Index)
hhi = sum(w ** 2 for w in current_weights)

# Beta (relative to CSI 300)
if benchmark is not None:
    benchmark_returns = benchmark.pct_change().dropna()
    # Align dates
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
    if len(aligned) > 0:
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        var_market = aligned.iloc[:, 1].var()
        beta = cov / var_market if var_market != 0 else np.nan
    else:
        beta = np.nan
else:
    beta = np.nan

# Value at Risk (95% confidence)
var_95 = np.percentile(portfolio_returns, 5)  # worst 5% of daily returns
var_95_annual = var_95 * np.sqrt(252)  # rough annualized

# ---------------------------- Display Results ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Current Holdings")
    df_display = pd.DataFrame({
        "Stock": tickers,
        "Target Weight": [f"{w * 100:.1f}%" for w in target_weights],
        "Current Weight": [f"{w * 100:.1f}%" for w in current_weights],
        "Deviation": [f"{(c - t) * 100:+.1f}%" for c, t in zip(current_weights, target_weights)],
        "Shares Held": shares,
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
    col2b.metric("Max Drawdown", f"{max_drawdown:.2%}", help="Largest peak‑to‑trough decline in history")
    col2c.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", help="Excess return per unit of risk (higher is better)")

    col2d, col2e, col2f = st.columns(3)
    col2d.metric("Concentration (HHI)", f"{hhi:.3f}",
                 help="Close to 1 means highly concentrated, close to 0 means well diversified")
    col2e.metric("Beta", f"{beta:.2f}" if not np.isnan(beta) else "N/A", help="Sensitivity to the market (CSI 300)")
    col2f.metric("VaR (95%, daily)", f"{var_95:.2%}", help="Maximum expected daily loss at 95% confidence level")

# ---------------------------- Weight Visualization ----------------------------
st.subheader("🥧 Weight Comparison")
col_pie1, col_pie2 = st.columns(2)
with col_pie1:
    fig_target = px.pie(
        names=tickers, values=target_weights, title="Target Weights",
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_target, use_container_width=True)
with col_pie2:
    fig_current = px.pie(
        names=tickers, values=current_weights, title="Current Weights",
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_current, use_container_width=True)

# Deviation bar chart
deviations = [c - t for c, t in zip(current_weights, target_weights)]
fig_bar = go.Figure(data=[
    go.Bar(x=tickers, y=deviations, marker_color=['red' if d > 0 else 'green' for d in deviations])
])
fig_bar.update_layout(
    title="Deviation (Current - Target)",
    xaxis_title="Stock",
    yaxis_title="Deviation",
    yaxis_tickformat='.1%'
)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------- Rebalancing Suggestions ----------------------------
st.subheader("⚖️ Rebalancing Suggestions")
deviations_detail = []
for ticker, target, current in zip(tickers, target_weights, current_weights):
    dev = current - target
    deviations_detail.append((ticker, target, current, dev))

alerts = [(t, tgt, cur, dev) for t, tgt, cur, dev in deviations_detail if abs(dev) > threshold / 100]

if alerts:
    st.warning(f"⚠️ The following assets deviate by more than {threshold}%. Consider rebalancing:")
    for ticker, target, current, dev in alerts:
        action = "SELL" if dev > 0 else "BUY"
        st.markdown(
            f"- **{ticker}**: target {target:.1%}, current {current:.1%}, deviation {dev:+.2%} → **{action}** to return to target.")

    # Simple rebalancing simulation (optional)
    st.markdown("**📉 What if you don't rebalance?**")
    # Assume future returns equal past year's returns
    annual_returns = (price_data.iloc[-1] / price_data.iloc[0] - 1).values
    future_values = current_values * (1 + annual_returns)
    future_total = sum(future_values)
    future_weights = future_values / future_total
    st.markdown("Based on past year's returns, one year from now the weights might become:")
    for t, fw in zip(tickers, future_weights):
        st.markdown(f"- {t}: {fw:.1%}")
    st.info("If deviation continues to widen, your portfolio risk may significantly deviate from your expectations.")
else:
    st.success("✅ All asset weights are within the acceptable range. No rebalancing needed.")

# ---------------------------- Historical Charts ----------------------------
st.subheader("📉 Portfolio Historical NAV (Normalized)")
normalized = cumulative / cumulative.iloc[0]
st.line_chart(normalized)

# Individual stock history
st.subheader("📈 Individual Stock Prices (Normalized)")
normalized_stocks = price_data / price_data.iloc[0]
st.line_chart(normalized_stocks)

# ---------------------------- Footer ----------------------------
st.markdown("---")
st.caption(
    "Disclaimer: This tool is for informational purposes only and does not constitute investment advice. Data source: TuShare.")