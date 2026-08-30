import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats
import warnings
import time
import io

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ultimate Stock Analysis (NSE + BSE)", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""<style>
.main-header{font-size:2.2rem;font-weight:700;color:#1f77b4;text-align:center;margin-bottom:1rem}
.metric-card{background:#f8f9fb;padding:0.6rem 0.9rem;border-radius:8px;border-left:4px solid #1f77b4;box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.metric-title{font-size:0.9rem;color:#333;margin-bottom:6px}
.metric-value{font-size:1.4rem;font-weight:700;color:#111}
.rec-buy{background:#d4edda;border-left:4px solid #28a745;padding:1rem;border-radius:8px;margin:0.5rem 0}
.rec-hold{background:#fff3cd;border-left:4px solid #ffc107;padding:1rem;border-radius:8px;margin:0.5rem 0}
.rec-sell{background:#f8d7da;border-left:4px solid #dc3545;padding:1rem;border-radius:8px;margin:0.5rem 0}
</style>""", unsafe_allow_html=True)

# Exchange configuration: suffix + default benchmark index ticker/name
EXCHANGES = {
    "NSE": {"suffix": ".NS", "benchmark_ticker": "^NSEI", "benchmark_name": "NIFTY 50"},
    "BSE": {"suffix": ".BO", "benchmark_ticker": "^BSESN", "benchmark_name": "SENSEX"},
}

@st.cache_data(ttl=3600)
def fetch_data(code, suffix, benchmark_ticker, start, end, freq='1d', retries=3):
    full_ticker = f"{code}{suffix}"
    for i in range(retries):
        try:
            if i > 0: time.sleep(2 ** i)
            s = yf.download(full_ticker, start, end, interval=freq, progress=False)
            time.sleep(0.5)
            n = yf.download(benchmark_ticker, start, end, interval=freq, progress=False)
            if s.empty or n.empty: raise ValueError("No data")
            if isinstance(s.columns, pd.MultiIndex): s.columns = s.columns.droplevel(1)
            if isinstance(n.columns, pd.MultiIndex): n.columns = n.columns.droplevel(1)
            return s, n, full_ticker
        except:
            if i == retries-1: raise
    raise Exception("Failed")

def calc_returns(stock, bench):
    sr = stock['Close'].pct_change() * 100
    br = bench['Close'].pct_change() * 100
    df = pd.DataFrame({'Stock_Return': sr, 'Bench_Return': br}).dropna()
    if len(df) < 30: raise ValueError("Need 30+ data points")
    return df

def calc_metrics(stock, bench, rets, rf=6.5):
    r = {}
    y, X = rets['Stock_Return'], add_constant(rets['Bench_Return'])
    m = OLS(y, X).fit()
    r.update({'beta': m.params['Bench_Return'], 'alpha': m.params['const'], 'r_squared': m.rsquared,
              'std_error': m.bse['Bench_Return'], 'p_value': m.pvalues['Bench_Return'],
              'conf_int_lower': m.conf_int().loc['Bench_Return',0], 'conf_int_upper': m.conf_int().loc['Bench_Return',1]})
    r['volatility'] = rets['Stock_Return'].std()
    r['annual_vol'] = r['volatility'] * np.sqrt(252)
    r['mean_ret'] = rets['Stock_Return'].mean()
    r['annual_ret'] = r['mean_ret'] * 252
    r['cum_ret'] = ((1 + rets['Stock_Return']/100).prod() - 1) * 100
    r['mkt_mean'] = rets['Bench_Return'].mean()
    r['mkt_annual'] = r['mkt_mean'] * 252
    r['mkt_vol'] = rets['Bench_Return'].std()
    r['mkt_annual_vol'] = r['mkt_vol'] * np.sqrt(252)
    r['excess'] = r['mean_ret'] - r['mkt_mean']
    daily_rf = rf/252
    r['sharpe'] = (r['mean_ret'] - daily_rf) / r['volatility']
    r['annual_sharpe'] = r['sharpe'] * np.sqrt(252)
    r['treynor'] = (r['annual_ret'] - rf) / r['beta']
    r['jensen'] = r['annual_ret'] - (rf + r['beta'] * (r['mkt_annual'] - rf))
    te = (rets['Stock_Return'] - rets['Bench_Return']).std()
    r['info_ratio'] = r['excess'] / te if te != 0 else 0
    r['tracking_error'] = te * np.sqrt(252)
    neg = rets['Stock_Return'][rets['Stock_Return'] < 0]
    r['downside_dev'] = neg.std() if len(neg) > 0 else 0
    r['sortino'] = (r['mean_ret'] - daily_rf) / r['downside_dev'] if r['downside_dev'] != 0 else 0
    cum = (1 + rets['Stock_Return']/100).cumprod()
    rmax = cum.expanding().max()
    dd = (cum - rmax) / rmax * 100
    r['max_dd'] = dd.min()
    r['var_95'] = np.percentile(rets['Stock_Return'], 5)
    r['cvar_95'] = rets['Stock_Return'][rets['Stock_Return'] <= r['var_95']].mean()
    r['skew'] = stats.skew(rets['Stock_Return'])
    r['kurt'] = stats.kurtosis(rets['Stock_Return'])
    pos = (rets['Stock_Return'] > 0).sum()
    r['win_rate'] = (pos / len(rets)) * 100
    r['avg_win'] = rets['Stock_Return'][rets['Stock_Return'] > 0].mean()
    r['avg_loss'] = rets['Stock_Return'][rets['Stock_Return'] < 0].mean()
    r['wl_ratio'] = abs(r['avg_win'] / r['avg_loss']) if r['avg_loss'] != 0 else 0
    r['price'] = stock['Close'].iloc[-1]
    r['high52'] = stock['High'].iloc[-252:].max() if len(stock) >= 252 else stock['High'].max()
    r['low52'] = stock['Low'].iloc[-252:].min() if len(stock) >= 252 else stock['Low'].min()
    r['pct_high'] = (r['price'] / r['high52']) * 100
    r['corr'] = rets['Stock_Return'].corr(rets['Bench_Return'])
    r['obs'] = len(rets)

    # --- Systematic vs Unsystematic risk decomposition (sigma_i^2 = beta^2*sigma_m^2 + sigma_eps^2) ---
    r['total_var'] = r['annual_vol'] ** 2
    r['systematic_var'] = (r['beta'] ** 2) * (r['mkt_annual_vol'] ** 2)
    r['unsystematic_var'] = max(r['total_var'] - r['systematic_var'], 0)
    r['systematic_risk'] = np.sqrt(r['systematic_var'])          # beta * sigma_m, annualized
    r['unsystematic_risk'] = np.sqrt(r['unsystematic_var'])      # sqrt(sigma_i^2 - beta^2*sigma_m^2)
    r['systematic_share'] = r['systematic_var'] / r['total_var'] if r['total_var'] != 0 else 0  # ~ R^2
    r['unsystematic_share'] = 1 - r['systematic_share']
    return r, m

def recommend(r):
    recs, score = [], 0
    if 0.8 <= r['beta'] <= 1.2: recs.append("✅ Beta ~1.0: Moderate market risk"); score += 1
    elif r['beta'] > 1.5: recs.append("⚠️ High beta >1.5: High volatility"); score -= 1
    elif r['beta'] < 0.5: recs.append("✅ Low beta: Defensive"); score += 1
    if r['annual_sharpe'] > 1.5: recs.append("✅ Excellent Sharpe >1.5"); score += 2
    elif r['annual_sharpe'] > 1.0: recs.append("✅ Good Sharpe >1.0"); score += 1
    elif r['annual_sharpe'] < 0: recs.append("⚠️ Negative Sharpe"); score -= 2
    if r['jensen'] > 2: recs.append("✅ Strong alpha >2%"); score += 2
    elif r['jensen'] < -2: recs.append("⚠️ Negative alpha"); score -= 2
    if r['annual_vol'] > 40: recs.append("⚠️ High volatility >40%"); score -= 1
    elif r['annual_vol'] < 20: recs.append("✅ Low volatility <20%"); score += 1
    if r['max_dd'] < -30: recs.append("⚠️ Severe drawdown >30%"); score -= 2
    elif r['max_dd'] > -15: recs.append("✅ Moderate drawdown <15%"); score += 1
    if r['win_rate'] > 60: recs.append("✅ High win rate >60%"); score += 1
    elif r['win_rate'] < 45: recs.append("⚠️ Low win rate <45%"); score -= 1
    if score >= 4: final = ("BUY", "Strong metrics", "rec-buy")
    elif score >= 1: final = ("HOLD", "Moderate metrics", "rec-hold")
    else: final = ("AVOID", "Weak metrics", "rec-sell")
    return recs, final, score

def plot_reg(rets, m, t, bench_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rets['Bench_Return'], y=rets['Stock_Return'],
                            mode='markers', name='Returns', marker=dict(size=5, opacity=0.5)))
    x = np.linspace(rets['Bench_Return'].min(), rets['Bench_Return'].max(), 100)
    y = m.params['const'] + m.params['Bench_Return'] * x
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Regression'))
    fig.update_layout(title=f'{t} Regression', xaxis_title=f'{bench_name} %', yaxis_title=f'{t} %', height=500)
    return fig

def plot_dist(rets, t, bench_name):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rets['Stock_Return'], name=t, nbinsx=50, opacity=0.7))
    fig.add_trace(go.Histogram(x=rets['Bench_Return'], name=bench_name, nbinsx=50, opacity=0.7))
    fig.update_layout(title='Returns Distribution', barmode='overlay', height=500)
    return fig

def plot_cum(rets, t, bench_name):
    s = (1 + rets['Stock_Return']/100).cumprod() * 100
    n = (1 + rets['Bench_Return']/100).cumprod() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, mode='lines', name=t))
    fig.add_trace(go.Scatter(x=n.index, y=n, mode='lines', name=bench_name))
    fig.update_layout(title='Cumulative Returns', height=500, hovermode='x unified')
    return fig

def plot_dd(rets, t):
    cum = (1 + rets['Stock_Return']/100).cumprod()
    rmax = cum.expanding().max()
    dd = (cum - rmax) / rmax * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', name='Drawdown', line=dict(color='red')))
    fig.update_layout(title='Drawdown', height=500)
    return fig

def plot_risk_decomp(r, t):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'xy'}]],
                         subplot_titles=('Variance Share', 'Annualized Risk (%)'))
    fig.add_trace(go.Pie(labels=['Systematic', 'Unsystematic'],
                          values=[r['systematic_share']*100, r['unsystematic_share']*100],
                          hole=0.55, marker=dict(colors=['#1f77b4', '#ff7f0e']),
                          textinfo='label+percent'), row=1, col=1)
    fig.add_trace(go.Bar(x=['Total', 'Systematic', 'Unsystematic'],
                          y=[r['annual_vol'], r['systematic_risk'], r['unsystematic_risk']],
                          marker_color=['#6c757d', '#1f77b4', '#ff7f0e'],
                          text=[f"{v:.2f}%" for v in [r['annual_vol'], r['systematic_risk'], r['unsystematic_risk']]],
                          textposition='outside'), row=1, col=2)
    fig.update_layout(title=f'{t}: Systematic vs Unsystematic Risk', height=450, showlegend=False)
    return fig

def plot_rolling(rets, ws, t):
    fig = go.Figure()
    for w in ws:
        if len(rets) >= w:
            rb, dates = [], []
            for i in range(w, len(rets)):
                sub = rets.iloc[i-w:i]
                try:
                    m = OLS(sub['Stock_Return'], add_constant(sub['Bench_Return'])).fit()
                    rb.append(m.params['Bench_Return'])
                    dates.append(sub.index[-1])
                except: rb.append(np.nan); dates.append(sub.index[-1])
            fig.add_trace(go.Scatter(x=dates, y=rb, mode='lines', name=f'{w}D Beta'))
    fig.update_layout(title='Rolling Beta', height=500, hovermode='x unified')
    return fig

# Main
page = st.sidebar.selectbox("Navigation", ["Stock Analysis", "Portfolio/MF", "Formulas"])

if page == "Stock Analysis":
    st.markdown('<p class="main-header">📊 Ultimate Stock Analysis</p>', unsafe_allow_html=True)
    st.markdown("Comprehensive analysis with 30+ metrics and recommendations — NSE & BSE")

    st.sidebar.header("Configuration")
    exchange = st.sidebar.radio("Exchange", list(EXCHANGES.keys()), horizontal=True)
    ex_conf = EXCHANGES[exchange]

    if exchange == "NSE":
        code_label = "NSE Ticker"
        code_default = "RELIANCE"
        code_help = "e.g. RELIANCE, TCS, INFY"
    else:
        code_label = "BSE Code"
        code_default = "500325"
        code_help = "Numeric BSE scrip code, e.g. 500325 (Reliance), 531212 (HDFC AMC)"

    code = st.sidebar.text_input(code_label, code_default, help=code_help).strip().upper()

    # Optional override: let advanced users pick the benchmark independent of exchange
    bench_choice = st.sidebar.selectbox(
        "Benchmark Index",
        [f"Auto ({ex_conf['benchmark_name']})", "NIFTY 50", "SENSEX"]
    )
    if bench_choice.startswith("Auto"):
        benchmark_ticker = ex_conf["benchmark_ticker"]
        benchmark_name = ex_conf["benchmark_name"]
    elif bench_choice == "NIFTY 50":
        benchmark_ticker, benchmark_name = "^NSEI", "NIFTY 50"
    else:
        benchmark_ticker, benchmark_name = "^BSESN", "SENSEX"

    freq_map = {'Daily': '1d', 'Weekly': '1wk', 'Monthly': '1mo'}
    freq = freq_map[st.sidebar.selectbox("Frequency", list(freq_map.keys()))]

    period_type = st.sidebar.radio("Period", ["Predefined", "Custom"])
    if period_type == "Predefined":
        days = {'1Y': 365, '3Y': 1095, '5Y': 1825}[st.sidebar.selectbox("Range", ['1Y','3Y','5Y'])]
        end = datetime.now()
        start = end - timedelta(days=days)
    else:
        start = st.sidebar.date_input("Start", datetime.now()-timedelta(days=1095))
        end = st.sidebar.date_input("End", datetime.now())

    rf = st.sidebar.number_input("Risk-Free %", 5.0, 10.0, 6.5, 0.1)

    if st.sidebar.button("🚀 Analyze", type="primary", use_container_width=True):
        try:
            prog = st.progress(0); stat = st.empty()
            stat.info(f'Fetching {code} ({exchange})...'); prog.progress(20)
            s, n, ft = fetch_data(code, ex_conf["suffix"], benchmark_ticker, start, end, freq)
            stat.info('Calculating...'); prog.progress(50)
            rets = calc_returns(s, n)
            stat.info('Computing metrics...'); prog.progress(70)
            r, m = calc_metrics(s, n, rets, rf)
            stat.info('Generating recs...'); prog.progress(90)
            recs, final, score = recommend(r)
            prog.progress(100); stat.success(f'✅ Done for {ft} vs {benchmark_name}')
            time.sleep(1); stat.empty(); prog.empty()

            st.header(f"📈 Analysis: {ft}  ·  Benchmark: {benchmark_name}")

            # Recommendation
            st.subheader("🎯 Recommendation")
            st.markdown(f'<div class="{final[2]}"><h3>{final[0]}</h3><p>{final[1]}</p><p>Score: {score}/10</p></div>',
                       unsafe_allow_html=True)

            with st.expander("📋 Details", expanded=True):
                for rec in recs: st.markdown(f"- {rec}")

            # Metrics
            st.subheader("📊 Key Metrics")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Beta", f"{r['beta']:.4f}")
            c2.metric("Annual Return", f"{r['annual_ret']:.2f}%")
            c3.metric("Sharpe Ratio", f"{r['annual_sharpe']:.4f}")
            c4.metric("Alpha", f"{r['jensen']:.2f}%")

            st.markdown("### Risk")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Volatility", f"{r['annual_vol']:.2f}%")
            c2.metric("Max Drawdown", f"{r['max_dd']:.2f}%")
            c3.metric("VaR 95%", f"{r['var_95']:.2f}%")
            c4.metric("Sortino", f"{r['sortino']:.4f}")

            st.markdown("### 🧩 Risk Decomposition (Systematic vs Unsystematic)")
            st.caption(r"σᵢ² = β²σₘ² + σε²  ·  Systematic share ≈ R²")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Systematic Risk", f"{r['systematic_risk']:.2f}%", help="β × σ(market), annualized")
            c2.metric("Unsystematic Risk", f"{r['unsystematic_risk']:.2f}%", help="√(σᵢ² − β²σₘ²), annualized")
            c3.metric("Systematic Share", f"{r['systematic_share']*100:.1f}%", help="Systematic variance / Total variance ≈ R²")
            c4.metric("Unsystematic Share", f"{r['unsystematic_share']*100:.1f}%", help="1 − Systematic Share")

            st.markdown("### Performance")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Win Rate", f"{r['win_rate']:.1f}%")
            c2.metric("Avg Win", f"{r['avg_win']:.2f}%")
            c3.metric("Avg Loss", f"{r['avg_loss']:.2f}%")
            c4.metric("W/L Ratio", f"{r['wl_ratio']:.2f}")

            # Full stats table
            st.subheader("📋 Complete Statistics")
            stats_df = pd.DataFrame({
                'Category': ['Beta']*4 + ['Returns']*4 + ['Risk']*6 + ['Risk Decomposition']*4 + ['Ratios']*4 + ['Dist']*3 + ['Price']*4,
                'Metric': ['Beta','Alpha','R²','Correlation',
                          'Daily Return','Annual Return','Cumulative','Excess',
                          'Daily Vol','Annual Vol','Downside Dev','Max DD','VaR 95%','CVaR 95%',
                          'Systematic Risk','Unsystematic Risk','Systematic Share','Unsystematic Share',
                          'Sharpe','Sortino','Treynor','Info Ratio',
                          'Skewness','Kurtosis','Tracking Error',
                          'Current','52W High','52W Low','% to High'],
                'Value': [f"{r['beta']:.6f}", f"{r['alpha']:.6f}%", f"{r['r_squared']:.6f}", f"{r['corr']:.6f}",
                         f"{r['mean_ret']:.4f}%", f"{r['annual_ret']:.2f}%", f"{r['cum_ret']:.2f}%", f"{r['excess']:.4f}%",
                         f"{r['volatility']:.4f}%", f"{r['annual_vol']:.2f}%", f"{r['downside_dev']:.4f}%",
                         f"{r['max_dd']:.2f}%", f"{r['var_95']:.2f}%", f"{r['cvar_95']:.2f}%",
                         f"{r['systematic_risk']:.2f}%", f"{r['unsystematic_risk']:.2f}%",
                         f"{r['systematic_share']*100:.1f}%", f"{r['unsystematic_share']*100:.1f}%",
                         f"{r['annual_sharpe']:.4f}", f"{r['sortino']:.4f}", f"{r['treynor']:.2f}", f"{r['info_ratio']:.4f}",
                         f"{r['skew']:.4f}", f"{r['kurt']:.4f}", f"{r['tracking_error']:.2f}%",
                         f"₹{r['price']:.2f}", f"₹{r['high52']:.2f}", f"₹{r['low52']:.2f}", f"{r['pct_high']:.1f}%"]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            # Charts
            st.header("📊 Visualizations")
            t1,t2,t3,t4,t5,t6 = st.tabs(["Regression", "Distribution", "Cumulative", "Drawdown", "Rolling", "Risk Decomposition"])
            with t1: st.plotly_chart(plot_reg(rets, m, ft, benchmark_name), use_container_width=True)
            with t2: st.plotly_chart(plot_dist(rets, ft, benchmark_name), use_container_width=True)
            with t3: st.plotly_chart(plot_cum(rets, ft, benchmark_name), use_container_width=True)
            with t4: st.plotly_chart(plot_dd(rets, ft), use_container_width=True)
            with t5:
                ws = [30, 90, 180] if len(rets) >= 180 else ([30, 90] if len(rets) >= 90 else [30])
                if len(rets) >= 30: st.plotly_chart(plot_rolling(rets, ws, ft), use_container_width=True)
                else: st.warning("Need 30+ points for rolling")
            with t6:
                st.plotly_chart(plot_risk_decomp(r, ft), use_container_width=True)
                st.markdown(
                    f"**{ft}** total annualized risk of **{r['annual_vol']:.2f}%** splits into "
                    f"**{r['systematic_risk']:.2f}%** systematic (market-driven, from β = {r['beta']:.3f} vs "
                    f"{benchmark_name}'s {r['mkt_annual_vol']:.2f}% annual volatility) and "
                    f"**{r['unsystematic_risk']:.2f}%** unsystematic (stock-specific/diversifiable) risk."
                )

            # Downloads
            st.header("💾 Downloads")
            c1, c2, c3 = st.columns(3)
            with c1:
                full_df = pd.DataFrame({
                    'Date': rets.index,
                    'Stock_Return': rets['Stock_Return'],
                    'Benchmark_Return': rets['Bench_Return'],
                    'Stock_Price': s['Close'].reindex(rets.index),
                    'Benchmark_Price': n['Close'].reindex(rets.index),
                    'Stock_Volume': s['Volume'].reindex(rets.index)
                })
                st.download_button("📥 Returns + Prices", full_df.to_csv(index=False),
                                  f"{ft}_full_data.csv", "text/csv")
            with c2:
                st.download_button("📥 Statistics", stats_df.to_csv(index=False),
                                  f"{ft}_stats.csv", "text/csv")
            with c3:
                raw_df = pd.DataFrame({
                    'Date': s.index,
                    'Open': s['Open'],
                    'High': s['High'],
                    'Low': s['Low'],
                    'Close': s['Close'],
                    'Volume': s['Volume']
                })
                st.download_button("📥 Raw OHLCV", raw_df.to_csv(index=False),
                                  f"{ft}_ohlcv.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ {e}")
            if 'rate limit' in str(e).lower():
                st.warning("💡 Try: Wait 1-2 min, use Weekly/Monthly, shorter range, or clear cache")
    else:
        st.info("👈 Configure and click Analyze")
        st.subheader("Example Codes")
        c1,c2,c3 = st.columns(3)
        c1.markdown("**NSE Large Cap**\n\nRELIANCE\nTCS\nHDFCBANK\nINFY")
        c2.markdown("**NSE Mid Cap**\n\nADANIPORTS\nLT\nAXISBANK\nM&M")
        c3.markdown("**BSE Codes**\n\n500325 (Reliance)\n532540 (TCS)\n500180 (HDFC Bank)\n531212 (HDFC AMC)")

elif page == "Portfolio/MF":
    st.markdown('<p class="main-header">📊 Portfolio & Mutual Fund Beta</p>', unsafe_allow_html=True)
    st.info("🚧 Feature coming soon: Analyze portfolio beta and mutual fund holdings")
    st.markdown("""
    **Planned Features:**
    - Multi-stock portfolio beta calculation (NSE + BSE)
    - Weighted portfolio metrics
    - Mutual fund holdings analysis (via AMFI data)
    - Portfolio optimization suggestions
    - Correlation matrix for holdings
    - Sector exposure analysis
    """)

    st.subheader("Manual Portfolio Beta (Prototype)")
    st.markdown("Enter ticker symbols and weights:")

    num_stocks = st.number_input("Number of stocks", 2, 10, 3)
    tickers, weights = [], []

    for i in range(num_stocks):
        c1, c2 = st.columns(2)
        tickers.append(c1.text_input(f"Stock {i+1}", f"STOCK{i+1}"))
        weights.append(c2.number_input(f"Weight {i+1} (%)", 0.0, 100.0, 100.0/num_stocks))

    if st.button("Calculate Portfolio Beta"):
        if sum(weights) != 100:
            st.error("⚠️ Weights must sum to 100%")
        else:
            st.warning("🚧 This feature requires full implementation with API integration")

else:  # Formulas
    st.markdown('<p class="main-header">📘 Formula Reference</p>', unsafe_allow_html=True)

    st.subheader("1. Returns")
    st.latex(r"R_t = \frac{P_t - P_{t-1}}{P_{t-1}} \times 100")

    st.subheader("2. Beta (Regression)")
    st.latex(r"R_{stock} = \alpha + \beta R_{market} + \epsilon")
    st.latex(r"\beta = \frac{Cov(R_s, R_m)}{Var(R_m)}")

    st.subheader("2b. Total Risk = Systematic + Unsystematic Risk")
    st.latex(r"\sigma_i^2 = \beta_i^2\sigma_m^2 + \sigma_\epsilon^2")
    st.markdown("**Systematic risk** — the portion driven by overall market movements:")
    st.latex(r"\text{Systematic Variance} = \beta^2\sigma_m^2 \qquad \text{Systematic Risk} = \beta\sigma_m")
    st.markdown("**Unsystematic risk** — the stock-specific/diversifiable residual, found by subtracting *variances* (never subtract standard deviations directly):")
    st.latex(r"\sigma_\epsilon = \sqrt{\sigma_i^2 - \beta^2\sigma_m^2}")
    st.markdown("**R²-based decomposition** — from the regression, R² is exactly the systematic share of total variance:")
    st.latex(r"R^2 = \frac{\text{Systematic Variance}}{\text{Total Variance}} \qquad \text{Unsystematic Share} = 1 - R^2")
    st.markdown("Both approaches are computed for every stock analyzed on this app — see the *Risk Decomposition* tab under Visualizations.")

    st.subheader("3. Sharpe Ratio")
    st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")

    st.subheader("4. Sortino Ratio")
    st.latex(r"Sortino = \frac{R_p - R_f}{\sigma_{downside}}")

    st.subheader("5. Treynor Ratio")
    st.latex(r"Treynor = \frac{R_p - R_f}{\beta_p}")

    st.subheader("6. Jensen's Alpha")
    st.latex(r"\alpha_J = R_p - [R_f + \beta(R_m - R_f)]")

    st.subheader("7. Information Ratio")
    st.latex(r"IR = \frac{R_p - R_m}{TE}")
    st.markdown("Where TE = Tracking Error (std of excess returns)")

    st.subheader("8. Maximum Drawdown")
    st.latex(r"MDD = \min\left(\frac{P_t - P_{peak}}{P_{peak}}\right)")

    st.subheader("9. Value at Risk (VaR)")
    st.markdown("VaR₉₅ = 5th percentile of return distribution")

    st.subheader("10. Conditional VaR (CVaR)")
    st.markdown("CVaR₉₅ = Mean of returns below VaR₉₅")

    st.markdown("---")
    st.markdown("💡 All annualized metrics use √252 for daily data scaling. Benchmark = NIFTY 50 for NSE, SENSEX for BSE (overridable).")

st.markdown("---")
st.markdown("""<div style='text-align:center;color:#666;'>
<p>Built with Streamlit | Data: Yahoo Finance | Stats: Statsmodels & SciPy</p>
<p style='font-size:0.8rem;'>⚠️ Educational purposes only. Not financial advice.</p>
</div>""", unsafe_allow_html=True)
