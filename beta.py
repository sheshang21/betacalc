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
import re
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ultimate NSE Stock Analysis", page_icon="📊", layout="wide")

# Enhanced CSS
st.markdown("""<style>
.main-header{font-size:2.2rem;font-weight:700;color:#1f77b4;text-align:center;margin-bottom:1rem}
.sub-header{font-size:1.5rem;font-weight:600;color:#2c3e50;margin:1.5rem 0 0.8rem 0;border-bottom:2px solid #1f77b4;padding-bottom:0.5rem}
.metric-card{background:#f8f9fb;padding:0.6rem 0.9rem;border-radius:8px;border-left:4px solid #1f77b4;box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.metric-title{font-size:0.9rem;color:#333;margin-bottom:6px}
.metric-value{font-size:1.4rem;font-weight:700;color:#111}
.rec-buy{background:#d4edda;border-left:4px solid #28a745;padding:1rem;border-radius:8px;margin:0.5rem 0}
.rec-hold{background:#fff3cd;border-left:4px solid #ffc107;padding:1rem;border-radius:8px;margin:0.5rem 0}
.rec-sell{background:#f8d7da;border-left:4px solid #dc3545;padding:1rem;border-radius:8px;margin:0.5rem 0}
.formula-box{background:#f0f8ff;padding:1.5rem;border-radius:8px;border-left:4px solid #4682b4;margin:1rem 0}
.interpretation-box{background:#fff9e6;padding:1rem;border-radius:8px;border-left:4px solid #ffa500;margin:0.8rem 0}
.example-box{background:#f0fff0;padding:1rem;border-radius:8px;border-left:4px solid #32cd32;margin:0.8rem 0}
</style>""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end, freq='1d', retries=3):
    for i in range(retries):
        try:
            if i > 0: time.sleep(2 ** i)
            s = yf.download(f"{ticker}.NS", start, end, interval=freq, progress=False)
            time.sleep(0.5)
            n = yf.download("^NSEI", start, end, interval=freq, progress=False)
            if s.empty or n.empty: raise ValueError("No data")
            if isinstance(s.columns, pd.MultiIndex): s.columns = s.columns.droplevel(1)
            if isinstance(n.columns, pd.MultiIndex): n.columns = n.columns.droplevel(1)
            return s, n, f"{ticker}.NS"
        except: 
            if i == retries-1: raise
    raise Exception("Failed")

@st.cache_data(ttl=3600)
def scrape_mf_holdings(groww_url):
    """Scrape mutual fund holdings from Groww"""
    try:
        # Extract scheme code from URL
        match = re.search(r'/([^/]+)$', groww_url)
        if not match:
            raise ValueError("Invalid Groww URL format")
        
        # Fetch page
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(groww_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find holdings table (Groww's structure may vary)
        holdings = []
        
        # Look for common patterns in Groww's HTML structure
        # This is a simplified parser - actual implementation may need adjustments
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header
                cols = row.find_all('td')
                if len(cols) >= 2:
                    # Extract stock name and weight
                    stock_name = cols[0].get_text(strip=True)
                    weight_text = cols[-1].get_text(strip=True)
                    
                    # Try to extract percentage
                    weight_match = re.search(r'([\d.]+)%?', weight_text)
                    if weight_match and stock_name:
                        weight = float(weight_match.group(1))
                        
                        # Try to map to NSE ticker (simplified)
                        # In production, you'd need a proper company name to ticker mapping
                        ticker = stock_name.upper().replace(' ', '').replace('LIMITED', '').replace('LTD', '')[:10]
                        
                        holdings.append({
                            'Stock': stock_name,
                            'Ticker': ticker,
                            'Weight': weight
                        })
        
        if not holdings:
            # Fallback: Create sample holdings for demonstration
            st.warning("⚠️ Could not auto-fetch holdings. Using sample portfolio.")
            holdings = [
                {'Stock': 'NTPC Ltd', 'Ticker': 'NTPC', 'Weight': 8.5},
                {'Stock': 'Power Grid Corp', 'Ticker': 'POWERGRID', 'Weight': 7.2},
                {'Stock': 'Coal India', 'Ticker': 'COALINDIA', 'Weight': 6.8},
                {'Stock': 'GAIL India', 'Ticker': 'GAIL', 'Weight': 5.9},
                {'Stock': 'Bharat Electronics', 'Ticker': 'BEL', 'Weight': 5.5},
            ]
        
        return pd.DataFrame(holdings)
    
    except Exception as e:
        st.error(f"Error fetching holdings: {e}")
        # Return sample data
        return pd.DataFrame([
            {'Stock': 'NTPC Ltd', 'Ticker': 'NTPC', 'Weight': 8.5},
            {'Stock': 'Power Grid Corp', 'Ticker': 'POWERGRID', 'Weight': 7.2},
            {'Stock': 'Coal India', 'Ticker': 'COALINDIA', 'Weight': 6.8},
            {'Stock': 'GAIL India', 'Ticker': 'GAIL', 'Weight': 5.9},
            {'Stock': 'Bharat Electronics', 'Ticker': 'BEL', 'Weight': 5.5},
        ])

def calc_returns(stock, nifty):
    sr = stock['Close'].pct_change() * 100
    nr = nifty['Close'].pct_change() * 100
    df = pd.DataFrame({'Stock_Return': sr, 'Nifty_Return': nr}).dropna()
    if len(df) < 30: raise ValueError("Need 30+ data points")
    return df

def calc_metrics(stock, nifty, rets, rf=6.5):
    r = {}
    y, X = rets['Stock_Return'], add_constant(rets['Nifty_Return'])
    m = OLS(y, X).fit()
    r.update({'beta': m.params['Nifty_Return'], 'alpha': m.params['const'], 'r_squared': m.rsquared,
              'std_error': m.bse['Nifty_Return'], 'p_value': m.pvalues['Nifty_Return'],
              'conf_int_lower': m.conf_int().loc['Nifty_Return',0], 'conf_int_upper': m.conf_int().loc['Nifty_Return',1]})
    r['volatility'] = rets['Stock_Return'].std()
    r['annual_vol'] = r['volatility'] * np.sqrt(252)
    r['mean_ret'] = rets['Stock_Return'].mean()
    r['annual_ret'] = r['mean_ret'] * 252
    r['cum_ret'] = ((1 + rets['Stock_Return']/100).prod() - 1) * 100
    r['mkt_mean'] = rets['Nifty_Return'].mean()
    r['mkt_annual'] = r['mkt_mean'] * 252
    r['excess'] = r['mean_ret'] - r['mkt_mean']
    daily_rf = rf/252
    r['sharpe'] = (r['mean_ret'] - daily_rf) / r['volatility']
    r['annual_sharpe'] = r['sharpe'] * np.sqrt(252)
    r['treynor'] = (r['annual_ret'] - rf) / r['beta']
    r['jensen'] = r['annual_ret'] - (rf + r['beta'] * (r['mkt_annual'] - rf))
    te = (rets['Stock_Return'] - rets['Nifty_Return']).std()
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
    r['corr'] = rets['Stock_Return'].corr(rets['Nifty_Return'])
    r['obs'] = len(rets)
    return r, m

def calc_portfolio_beta(holdings_df, start, end, rf=6.5):
    """Calculate weighted portfolio beta"""
    total_weight = holdings_df['Weight'].sum()
    if abs(total_weight - 100) > 0.1:
        holdings_df['Weight'] = holdings_df['Weight'] / total_weight * 100
    
    portfolio_metrics = []
    failed_tickers = []
    
    for _, row in holdings_df.iterrows():
        try:
            ticker = row['Ticker']
            weight = row['Weight'] / 100
            
            s, n, _ = fetch_data(ticker, start, end)
            rets = calc_returns(s, n)
            metrics, _ = calc_metrics(s, n, rets, rf)
            
            portfolio_metrics.append({
                'Ticker': ticker,
                'Stock': row['Stock'],
                'Weight': row['Weight'],
                'Beta': metrics['beta'],
                'Weighted_Beta': metrics['beta'] * weight,
                'Annual_Return': metrics['annual_ret'],
                'Volatility': metrics['annual_vol'],
                'Sharpe': metrics['annual_sharpe']
            })
        except:
            failed_tickers.append(row['Ticker'])
            continue
    
    if not portfolio_metrics:
        raise ValueError("Could not fetch data for any holdings")
    
    df = pd.DataFrame(portfolio_metrics)
    
    # Calculate portfolio-level metrics
    portfolio_beta = df['Weighted_Beta'].sum()
    portfolio_return = (df['Annual_Return'] * df['Weight'] / 100).sum()
    portfolio_vol = np.sqrt((df['Volatility']**2 * (df['Weight']/100)**2).sum())
    
    return df, portfolio_beta, portfolio_return, portfolio_vol, failed_tickers

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

def plot_reg(rets, m, t):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rets['Nifty_Return'], y=rets['Stock_Return'], 
                            mode='markers', name='Returns', marker=dict(size=5, opacity=0.5)))
    x = np.linspace(rets['Nifty_Return'].min(), rets['Nifty_Return'].max(), 100)
    y = m.params['const'] + m.params['Nifty_Return'] * x
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Regression'))
    fig.update_layout(title=f'{t} Regression', xaxis_title='NIFTY %', yaxis_title=f'{t} %', height=500)
    return fig

def plot_dist(rets, t):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rets['Stock_Return'], name=t, nbinsx=50, opacity=0.7))
    fig.add_trace(go.Histogram(x=rets['Nifty_Return'], name='NIFTY', nbinsx=50, opacity=0.7))
    fig.update_layout(title='Returns Distribution', barmode='overlay', height=500)
    return fig

def plot_cum(rets, t):
    s = (1 + rets['Stock_Return']/100).cumprod() * 100
    n = (1 + rets['Nifty_Return']/100).cumprod() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, mode='lines', name=t))
    fig.add_trace(go.Scatter(x=n.index, y=n, mode='lines', name='NIFTY'))
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

def plot_rolling(rets, ws, t):
    fig = go.Figure()
    for w in ws:
        if len(rets) >= w:
            rb, dates = [], []
            for i in range(w, len(rets)):
                sub = rets.iloc[i-w:i]
                try:
                    m = OLS(sub['Stock_Return'], add_constant(sub['Nifty_Return'])).fit()
                    rb.append(m.params['Nifty_Return'])
                    dates.append(sub.index[-1])
                except: rb.append(np.nan); dates.append(sub.index[-1])
            fig.add_trace(go.Scatter(x=dates, y=rb, mode='lines', name=f'{w}D Beta'))
    fig.update_layout(title='Rolling Beta', height=500, hovermode='x unified')
    return fig

# Main Navigation
page = st.sidebar.selectbox("📍 Navigation", ["Stock Analysis", "Portfolio/MF Beta", "📚 Metrics Encyclopedia"])

if page == "Stock Analysis":
    st.markdown('<p class="main-header">📊 Ultimate Stock Analysis</p>', unsafe_allow_html=True)
    st.markdown("Comprehensive analysis with 30+ metrics and recommendations")
    
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("NSE Ticker", "RELIANCE").upper()
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
            stat.info(f'Fetching {ticker}...'); prog.progress(20)
            s, n, ft = fetch_data(ticker, start, end, freq)
            stat.info('Calculating...'); prog.progress(50)
            rets = calc_returns(s, n)
            stat.info('Computing metrics...'); prog.progress(70)
            r, m = calc_metrics(s, n, rets, rf)
            stat.info('Generating recs...'); prog.progress(90)
            recs, final, score = recommend(r)
            prog.progress(100); stat.success(f'✅ Done for {ft}')
            time.sleep(1); stat.empty(); prog.empty()
            
            st.header(f"📈 Analysis: {ticker}")
            
            st.subheader("🎯 Recommendation")
            st.markdown(f'<div class="{final[2]}"><h3>{final[0]}</h3><p>{final[1]}</p><p>Score: {score}/10</p></div>', 
                       unsafe_allow_html=True)
            
            with st.expander("📋 Details", expanded=True):
                for rec in recs: st.markdown(f"- {rec}")
            
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
            
            st.markdown("### Performance")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Win Rate", f"{r['win_rate']:.1f}%")
            c2.metric("Avg Win", f"{r['avg_win']:.2f}%")
            c3.metric("Avg Loss", f"{r['avg_loss']:.2f}%")
            c4.metric("W/L Ratio", f"{r['wl_ratio']:.2f}")
            
            st.subheader("📋 Complete Statistics")
            stats_df = pd.DataFrame({
                'Category': ['Beta']*4 + ['Returns']*4 + ['Risk']*6 + ['Ratios']*4 + ['Dist']*3 + ['Price']*4,
                'Metric': ['Beta','Alpha','R²','Correlation',
                          'Daily Return','Annual Return','Cumulative','Excess',
                          'Daily Vol','Annual Vol','Downside Dev','Max DD','VaR 95%','CVaR 95%',
                          'Sharpe','Sortino','Treynor','Info Ratio',
                          'Skewness','Kurtosis','Tracking Error',
                          'Current','52W High','52W Low','% to High'],
                'Value': [f"{r['beta']:.6f}", f"{r['alpha']:.6f}%", f"{r['r_squared']:.6f}", f"{r['corr']:.6f}",
                         f"{r['mean_ret']:.4f}%", f"{r['annual_ret']:.2f}%", f"{r['cum_ret']:.2f}%", f"{r['excess']:.4f}%",
                         f"{r['volatility']:.4f}%", f"{r['annual_vol']:.2f}%", f"{r['downside_dev']:.4f}%",
                         f"{r['max_dd']:.2f}%", f"{r['var_95']:.2f}%", f"{r['cvar_95']:.2f}%",
                         f"{r['annual_sharpe']:.4f}", f"{r['sortino']:.4f}", f"{r['treynor']:.2f}", f"{r['info_ratio']:.4f}",
                         f"{r['skew']:.4f}", f"{r['kurt']:.4f}", f"{r['tracking_error']:.2f}%",
                         f"₹{r['price']:.2f}", f"₹{r['high52']:.2f}", f"₹{r['low52']:.2f}", f"{r['pct_high']:.1f}%"]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            st.header("📊 Visualizations")
            t1,t2,t3,t4,t5 = st.tabs(["Regression", "Distribution", "Cumulative", "Drawdown", "Rolling"])
            with t1: st.plotly_chart(plot_reg(rets, m, ticker), use_container_width=True)
            with t2: st.plotly_chart(plot_dist(rets, ticker), use_container_width=True)
            with t3: st.plotly_chart(plot_cum(rets, ticker), use_container_width=True)
            with t4: st.plotly_chart(plot_dd(rets, ticker), use_container_width=True)
            with t5:
                ws = [30, 90, 180] if len(rets) >= 180 else ([30, 90] if len(rets) >= 90 else [30])
                if len(rets) >= 30: st.plotly_chart(plot_rolling(rets, ws, ticker), use_container_width=True)
                else: st.warning("Need 30+ points for rolling")
            
            st.header("💾 Downloads")
            c1, c2, c3 = st.columns(3)
            with c1:
                full_df = pd.DataFrame({
                    'Date': rets.index, 'Stock_Return': rets['Stock_Return'], 'Nifty_Return': rets['Nifty_Return'],
                    'Stock_Price': s['Close'].reindex(rets.index), 'Nifty_Price': n['Close'].reindex(rets.index),
                    'Stock_Volume': s['Volume'].reindex(rets.index)
                })
                st.download_button("📥 Returns + Prices", full_df.to_csv(index=False),
                                  f"{ticker}_full_data.csv", "text/csv")
            with c2:
                st.download_button("📥 Statistics", stats_df.to_csv(index=False),
                                  f"{ticker}_stats.csv", "text/csv")
            with c3:
                raw_df = pd.DataFrame({
                    'Date': s.index, 'Open': s['Open'], 'High': s['High'],
                    'Low': s['Low'], 'Close': s['Close'], 'Volume': s['Volume']
                })
                st.download_button("📥 Raw OHLCV", raw_df.to_csv(index=False),
                                  f"{ticker}_ohlcv.csv", "text/csv")
            
        except Exception as e:
            st.error(f"❌ {e}")
    else:
        st.info("👈 Configure and click Analyze")
        st.subheader("Example Tickers")
        c1,c2,c3 = st.columns(3)
        c1.markdown("**Large Cap**\n\nRELIANCE\nTCS\nHDFCBANK\nINFY")
        c2.markdown("**Mid Cap**\n\nADANIPORTS\nLT\nAXISBANK\nM&M")
        c3.markdown("**IT**\n\nWIPRO\nHCLTECH\nTECHM\nLTIM")

elif page == "Portfolio/MF Beta":
    st.markdown('<p class="main-header">📊 Portfolio & Mutual Fund Beta Calculator</p>', unsafe_allow_html=True)
    
    analysis_type = st.radio("Select Analysis Type", ["Mutual Fund (Groww Link)", "Manual Portfolio Entry"])
    
    if analysis_type == "Mutual Fund (Groww Link)":
        st.subheader("🔗 Auto-Fetch from Groww")
        st.markdown("Paste the Groww mutual fund URL to automatically fetch holdings and calculate portfolio beta")
        
        groww_url = st.text_input(
            "Groww Fund URL",
            placeholder="https://groww.in/mutual-funds/aditya-birla-sun-life-psu-equity-fund-direct-growth"
        )
        
        st.markdown("**Example URLs:**")
        st.code("""
https://groww.in/mutual-funds/aditya-birla-sun-life-psu-equity-fund-direct-growth
https://groww.in/mutual-funds/sbi-bluechip-fund-direct-growth
https://groww.in/mutual-funds/hdfc-mid-cap-opportunities-fund-direct-growth
        """)
        
    else:
        st.subheader("✏️ Manual Portfolio Entry")
        st.markdown("Enter stock tickers and their weights manually")
        
        num_stocks = st.number_input("Number of stocks", 2, 20, 5)
        
        manual_data = []
        for i in range(num_stocks):
            c1, c2, c3 = st.columns([2, 2, 1])
            stock_name = c1.text_input(f"Stock Name {i+1}", f"Stock {i+1}", key=f"name_{i}")
            ticker = c2.text_input(f"NSE Ticker {i+1}", f"STOCK{i+1}", key=f"tick_{i}")
            weight = c3.number_input(f"Weight % {i+1}", 0.0, 100.0, 100.0/num_stocks, key=f"wt_{i}")
            manual_data.append({'Stock': stock_name, 'Ticker': ticker.upper(), 'Weight': weight})
        
        holdings_df = pd.DataFrame(manual_data)
        st.markdown("**Preview:**")
        st.dataframe(holdings_df, hide_index=True)
    
    # Common inputs for both types
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        pf_start = st.date_input("Analysis Start Date", datetime.now()-timedelta(days=1095), key="pf_start")
    with col2:
        pf_end = st.date_input("Analysis End Date", datetime.now(), key="pf_end")
    
    pf_rf = st.number_input("Risk-Free Rate (%)", 5.0, 10.0, 6.5, 0.1, key="pf_rf")
    
    if st.button("📊 Calculate Portfolio Beta", type="primary", use_container_width=True):
        try:
            # Get holdings data
            if analysis_type == "Mutual Fund (Groww Link)":
                if not groww_url:
                    st.error("⚠️ Please enter a Groww URL")
                    st.stop()
                
                with st.spinner("🔍 Fetching mutual fund holdings..."):
                    holdings_df = scrape_mf_holdings(groww_url)
                
                st.success(f"✅ Fetched {len(holdings_df)} holdings")
                st.dataframe(holdings_df, hide_index=True)
            
            # Calculate portfolio metrics
            with st.spinner("📈 Calculating portfolio beta and metrics..."):
                metrics_df, port_beta, port_ret, port_vol, failed = calc_portfolio_beta(
                    holdings_df, pf_start, pf_end, pf_rf
                )
            
            # Display results
            st.success("✅ Portfolio analysis complete!")
            
            if failed:
                st.warning(f"⚠️ Could not fetch data for: {', '.join(failed)}")
            
            # Key metrics
            st.markdown('<p class="sub-header">📊 Portfolio Metrics</p>', unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio Beta", f"{port_beta:.4f}")
            c2.metric("Expected Annual Return", f"{port_ret:.2f}%")
            c3.metric("Portfolio Volatility", f"{port_vol:.2f}%")
            sharpe = (port_ret - pf_rf) / port_vol if port_vol != 0 else 0
            c4.metric("Sharpe Ratio", f"{sharpe:.4f}")
            
            # Interpretation
            st.markdown('<p class="sub-header">💡 Portfolio Interpretation</p>', unsafe_allow_html=True)
            
            if port_beta > 1.2:
                st.markdown(f"""
                <div class="interpretation-box">
                <b>Aggressive Portfolio (Beta: {port_beta:.2f})</b><br>
                Your portfolio is significantly more volatile than the market. For every 1% move in NIFTY, 
                expect your portfolio to move approximately {port_beta:.2f}%. Suitable for high risk tolerance.
                </div>
                """, unsafe_allow_html=True)
            elif port_beta >= 0.8:
                st.markdown(f"""
                <div class="interpretation-box">
                <b>Moderate Portfolio (Beta: {port_beta:.2f})</b><br>
                Your portfolio moves roughly in line with the market. Balanced risk-return profile 
                suitable for most investors.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="interpretation-box">
                <b>Defensive Portfolio (Beta: {port_beta:.2f})</b><br>
                Your portfolio is less volatile than the market. Lower risk but potentially lower returns 
                during bull markets. Good for capital preservation.
                </div>
                """, unsafe_allow_html=True)
            
            # Holdings breakdown
            st.markdown('<p class="sub-header">📋 Holdings Analysis</p>', unsafe_allow_html=True)
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            # Visualizations
            st.markdown('<p class="sub-header">📊 Portfolio Visualizations</p>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["Beta Distribution", "Risk-Return Map", "Weight Analysis"])
            
            with tab1:
                # Beta distribution chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=metrics_df['Ticker'],
                    y=metrics_df['Beta'],
                    text=metrics_df['Beta'].round(2),
                    textposition='outside',
                    marker_color=['red' if b > 1.5 else 'orange' if b > 1 else 'green' 
                                 for b in metrics_df['Beta']]
                ))
                fig.add_hline(y=1, line_dash="dash", line_color="blue", 
                             annotation_text="Market Beta = 1")
                fig.add_hline(y=port_beta, line_dash="dot", line_color="purple",
                             annotation_text=f"Portfolio Beta = {port_beta:.2f}")
                fig.update_layout(title="Stock Beta Distribution", 
                                xaxis_title="Stock", yaxis_title="Beta", height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Risk-return scatter
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=metrics_df['Volatility'],
                    y=metrics_df['Annual_Return'],
                    mode='markers+text',
                    text=metrics_df['Ticker'],
                    textposition='top center',
                    marker=dict(
                        size=metrics_df['Weight'],
                        sizemode='diameter',
                        sizeref=2,
                        color=metrics_df['Sharpe'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Sharpe")
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Return: %{y:.2f}%<br>' +
                                 'Volatility: %{x:.2f}%<br>' +
                                 '<extra></extra>'
                ))
                fig.update_layout(title="Risk-Return Map (Bubble Size = Weight)",
                                xaxis_title="Volatility (%)", 
                                yaxis_title="Annual Return (%)", height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Weight distribution pie
                fig = go.Figure(data=[go.Pie(
                    labels=metrics_df['Ticker'],
                    values=metrics_df['Weight'],
                    hole=0.4,
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>' +
                                 'Weight: %{value:.1f}%<br>' +
                                 '<extra></extra>'
                )])
                fig.update_layout(title="Portfolio Weight Distribution", height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Download
            st.markdown('<p class="sub-header">💾 Download Portfolio Data</p>', unsafe_allow_html=True)
            
            summary_df = pd.DataFrame({
                'Metric': ['Portfolio Beta', 'Expected Annual Return', 'Portfolio Volatility', 
                          'Sharpe Ratio', 'Number of Holdings', 'Analysis Period'],
                'Value': [f"{port_beta:.4f}", f"{port_ret:.2f}%", f"{port_vol:.2f}%",
                         f"{sharpe:.4f}", str(len(metrics_df)), 
                         f"{pf_start} to {pf_end}"]
            })
            
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 Holdings Details", metrics_df.to_csv(index=False),
                                  "portfolio_holdings.csv", "text/csv")
            with c2:
                st.download_button("📥 Summary Metrics", summary_df.to_csv(index=False),
                                  "portfolio_summary.csv", "text/csv")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("💡 Make sure all ticker symbols are valid NSE stocks")

else:  # Metrics Encyclopedia
    st.markdown('<p class="main-header">📚 Complete Metrics Encyclopedia</p>', unsafe_allow_html=True)
    st.markdown("Comprehensive guide to all financial metrics, formulas, and interpretations")
    
    metric_category = st.selectbox("Select Category", [
        "Beta & Regression Metrics",
        "Return Metrics", 
        "Risk Metrics",
        "Risk-Adjusted Ratios",
        "Distribution Metrics",
        "Performance Metrics",
        "Price Metrics",
        "Chart Explanations"
    ])
    
    if metric_category == "Beta & Regression Metrics":
        st.markdown('<p class="sub-header">Beta (β)</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box">
        <b>Formula:</b><br>
        β = Cov(R<sub>stock</sub>, R<sub>market</sub>) / Var(R<sub>market</sub>)
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"\beta = \frac{\text{Cov}(R_s, R_m)}{\text{Var}(R_m)} = \frac{\sum (R_s - \bar{R}_s)(R_m - \bar{R}_m)}{\sum (R_m - \bar{R}_m)^2}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Beta measures how sensitive a stock's returns are to market movements. It's the slope of the regression line.
        
        <ul>
        <li><b>β = 1.0:</b> Stock moves exactly with the market (average risk)</li>
        <li><b>β > 1.0:</b> Stock is more volatile than market (amplifies market moves)</li>
        <li><b>β < 1.0:</b> Stock is less volatile (defensive)</li>
        <li><b>β < 0:</b> Stock moves opposite to market (hedge)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-box">
        <b>Example:</b><br>
        If RELIANCE has β = 1.5 and NIFTY rises 10%, we expect RELIANCE to rise 15% (1.5 × 10%).
        If NIFTY falls 5%, RELIANCE would fall approximately 7.5%.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Alpha (α) - Regression Intercept</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box">
        <b>Formula:</b><br>
        From regression: R<sub>stock</sub> = α + β × R<sub>market</sub> + ε
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"R_{\text{stock}} = \alpha + \beta R_{\text{market}} + \epsilon")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Alpha is the intercept of the regression line. It represents the average return of the stock 
        that is NOT explained by market movements.
        
        <ul>
        <li><b>Positive α:</b> Stock has excess returns above what beta predicts</li>
        <li><b>Negative α:</b> Stock underperforms its beta-predicted return</li>
        <li><b>α ≈ 0:</b> Stock returns are fully explained by market movements</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">R-Squared (R²)</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box">
        <b>Formula:</b><br>
        R² = 1 - (SS<sub>residual</sub> / SS<sub>total</sub>)
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        R² measures how much of the stock's return variation is explained by market movements.
        
        <ul>
        <li><b>R² = 1.0:</b> 100% of stock returns explained by market (perfect correlation)</li>
        <li><b>R² = 0.7:</b> 70% explained by market, 30% by other factors</li>
        <li><b>R² = 0.3:</b> Only 30% explained by market (stock has its own drivers)</li>
        </ul>
        
        High R² = Stock closely tracks the market<br>
        Low R² = Stock has independent factors driving returns
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Correlation</p>', unsafe_allow_html=True)
        st.latex(r"\rho = \frac{\text{Cov}(R_s, R_m)}{\sigma_s \sigma_m}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Correlation measures the strength and direction of the linear relationship between stock and market.
        
        <ul>
        <li><b>+1:</b> Perfect positive correlation (move together)</li>
        <li><b>0:</b> No linear relationship</li>
        <li><b>-1:</b> Perfect negative correlation (move opposite)</li>
        </ul>
        
        Note: R² = Correlation²
        </div>
        """, unsafe_allow_html=True)
    
    elif metric_category == "Return Metrics":
        st.markdown('<p class="sub-header">Simple Return</p>', unsafe_allow_html=True)
        st.latex(r"R_t = \frac{P_t - P_{t-1}}{P_{t-1}} \times 100\%")
        
        st.markdown("""
        <div class="interpretation-box">
        Percentage change in price from one period to the next.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Annualized Return</p>', unsafe_allow_html=True)
        st.latex(r"\text{Annualized Return} = \bar{R}_{\text{daily}} \times 252")
        
        st.markdown("""
        <div class="interpretation-box">
        Average daily return scaled to annual basis (252 trading days per year).
        This assumes returns are additive and provides expected annual return.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Cumulative Return</p>', unsafe_allow_html=True)
        st.latex(r"\text{Cumulative Return} = \left[\prod_{t=1}^{n}(1 + R_t)\right] - 1")
        
        st.markdown("""
        <div class="interpretation-box">
        Total return over the entire period, accounting for compounding.
        
        <b>Example:</b> +5% followed by +3% = (1.05 × 1.03) - 1 = 8.15% (not 8%)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Excess Return</p>', unsafe_allow_html=True)
        st.latex(r"\text{Excess Return} = R_{\text{stock}} - R_{\text{market}}")
        
        st.markdown("""
        <div class="interpretation-box">
        How much the stock outperforms (or underperforms) the market.
        
        <ul>
        <li><b>Positive:</b> Stock beats market</li>
        <li><b>Negative:</b> Stock lags market</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif metric_category == "Risk Metrics":
        st.markdown('<p class="sub-header">Volatility (Standard Deviation)</p>', unsafe_allow_html=True)
        st.latex(r"\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(R_i - \bar{R})^2}")
        st.latex(r"\text{Annualized: } \sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Volatility measures the dispersion of returns. Higher volatility = more uncertainty.
        
        <ul>
        <li><b>Low (< 20%):</b> Stable, predictable (e.g., large caps, utilities)</li>
        <li><b>Medium (20-40%):</b> Moderate fluctuation (most stocks)</li>
        <li><b>High (> 40%):</b> Highly unpredictable (small caps, speculative)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Maximum Drawdown (MDD)</p>', unsafe_allow_html=True)
        st.latex(r"\text{MDD} = \min_{t}\left(\frac{P_t - P_{\text{peak}}}{P_{\text{peak}}}\right) \times 100\%")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Largest peak-to-trough decline during the period. Measures worst-case loss.
        
        <ul>
        <li><b>-10%:</b> Minor correction (acceptable for most)</li>
        <li><b>-20%:</b> Significant drop (requires risk tolerance)</li>
        <li><b>-30%+:</b> Severe decline (high risk, recovery takes time)</li>
        </ul>
        
        <b>Important:</b> Tells you how much capital you could have lost if you bought at the worst time.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Downside Deviation</p>', unsafe_allow_html=True)
        st.latex(r"\sigma_{\text{downside}} = \sqrt{\frac{1}{n}\sum_{\{i:R_i<0\}}R_i^2}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Volatility calculated using only negative returns. Focuses on downside risk only.
        
        Better than standard volatility because upside volatility is actually good!
        Used in Sortino Ratio calculation.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Value at Risk (VaR) - 95%</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box">
        VaR<sub>95</sub> = 5th percentile of return distribution
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        With 95% confidence, your losses will not exceed VaR on any given day.
        
        <b>Example:</b> VaR₉₅ = -2.5% means:<br>
        "95% of the time, daily losses will be less than 2.5%"<br>
        OR "Only 5% of days will have losses exceeding 2.5%"
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Conditional VaR (CVaR)</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box">
        CVaR<sub>95</sub> = Average of returns below VaR<sub>95</sub>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Expected loss GIVEN that you're in the worst 5% of outcomes. Also called "Expected Shortfall".
        
        More informative than VaR because it tells you HOW BAD the tail losses are, not just the threshold.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Tracking Error</p>', unsafe_allow_html=True)
        st.latex(r"\text{TE} = \sigma(R_{\text{stock}} - R_{\text{market}}) \times \sqrt{252}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Standard deviation of excess returns. Measures how closely the stock tracks the benchmark.
        
        <ul>
        <li><b>Low TE (< 5%):</b> Closely tracks market (index-like)</li>
        <li><b>Medium TE (5-10%):</b> Moderate divergence</li>
        <li><b>High TE (> 10%):</b> Significant independent movement</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif metric_category == "Risk-Adjusted Ratios":
        st.markdown('<p class="sub-header">Sharpe Ratio</p>', unsafe_allow_html=True)
        st.latex(r"\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Excess return per unit of total risk. Higher is better.
        
        <b>Interpretation:</b>
        <ul>
        <li><b>< 0:</b> Losing money vs risk-free rate (bad)</li>
        <li><b>0 - 1:</b> Subpar risk-adjusted returns</li>
        <li><b>1 - 2:</b> Good risk-adjusted returns</li>
        <li><b>> 2:</b> Excellent (rare in equity markets)</li>
        </ul>
        
        <b>Rule of Thumb:</b> Sharpe > 1 is acceptable, > 1.5 is excellent
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Sortino Ratio</p>', unsafe_allow_html=True)
        st.latex(r"\text{Sortino} = \frac{R_p - R_f}{\sigma_{\text{downside}}}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Similar to Sharpe but penalizes only downside volatility (negative returns).
        
        <b>Why Better than Sharpe:</b><br>
        Investors don't mind upside volatility! Sortino focuses only on harmful volatility.
        
        Generally higher than Sharpe ratio for the same asset. Interpret similarly to Sharpe.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Treynor Ratio</p>', unsafe_allow_html=True)
        st.latex(r"\text{Treynor} = \frac{R_p - R_f}{\beta_p}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Excess return per unit of systematic risk (beta). Best for diversified portfolios.
        
        <b>Sharpe vs Treynor:</b>
        <ul>
        <li><b>Sharpe:</b> Uses total risk (σ) - better for standalone evaluation</li>
        <li><b>Treynor:</b> Uses systematic risk (β) - better for portfolio additions</li>
        </ul>
        
        Higher Treynor = Better compensation for market risk taken
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Jensen's Alpha</p>', unsafe_allow_html=True)
        st.latex(r"\alpha_J = R_p - [R_f + \beta(R_m - R_f)]")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Excess return AFTER adjusting for systematic risk (beta). The "true" outperformance.
        
        <ul>
        <li><b>α > 0:</b> Outperforming risk-adjusted expectations (manager skill)</li>
        <li><b>α = 0:</b> Fair return for risk taken</li>
        <li><b>α < 0:</b> Underperforming (would be better off in index)</li>
        </ul>
        
        <b>CAPM Equation:</b> Jensen's Alpha is the intercept when regressing against expected CAPM returns
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Information Ratio</p>', unsafe_allow_html=True)
        st.latex(r"\text{IR} = \frac{R_p - R_m}{\text{Tracking Error}}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Excess return per unit of active risk (deviation from benchmark).
        
        <b>Interpretation:</b>
        <ul>
        <li><b>IR > 0.5:</b> Good active management</li>
        <li><b>IR > 1.0:</b> Excellent (top quartile of active managers)</li>
        <li><b>IR < 0:</b> Active bets are destroying value</li>
        </ul>
        
        Most important metric for active fund managers!
        </div>
        """, unsafe_allow_html=True)
    
    elif metric_category == "Distribution Metrics":
        st.markdown('<p class="sub-header">Skewness</p>', unsafe_allow_html=True)
        st.latex(r"\text{Skewness} = \frac{E[(R - \mu)^3]}{\sigma^3}")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Measures asymmetry of return distribution.
        
        <ul>
        <li><b>Skew = 0:</b> Symmetric (normal distribution)</li>
        <li><b>Skew > 0:</b> Positive skew - more frequent small losses, occasional large gains (good!)</li>
        <li><b>Skew < 0:</b> Negative skew - more frequent small gains, occasional large losses (risky!)</li>
        </ul>
        
        <b>Investor Preference:</b> Positive skewness is preferred (lottery-like payoffs)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Kurtosis</p>', unsafe_allow_html=True)
        st.latex(r"\text{Kurtosis} = \frac{E[(R - \mu)^4]}{\sigma^4} - 3")
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Measures "fat tails" - likelihood of extreme events.
        
        <ul>
        <li><b>Kurt = 0:</b> Normal distribution (Gaussian)</li>
        <li><b>Kurt > 0:</b> Fat tails - more extreme events than normal (crash risk!)</li>
        <li><b>Kurt < 0:</b> Thin tails - fewer extremes than normal</li>
        </ul>
        
        <b>Important:</b> High kurtosis means VaR/CVaR may underestimate tail risk!
        Most stocks have positive kurtosis (crashes happen more than normal distribution predicts)
        </div>
        """, unsafe_allow_html=True)
    
    elif metric_category == "Performance Metrics":
        st.markdown('<p class="sub-header">Win Rate</p>', unsafe_allow_html=True)
        st.latex(r"\text{Win Rate} = \frac{\text{Number of Positive Return Days}}{\text{Total Days}} \times 100\%")
        
        st.markdown("""
        <div class="interpretation-box">
        Percentage of days with positive returns.
        
        <ul>
        <li><b>> 60%:</b> Very consistent (but check if wins are small)</li>
        <li><b>50-60%:</b> Normal for equity markets</li>
        <li><b>< 45%:</b> More losing days (needs high win/loss ratio to compensate)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">Win/Loss Ratio</p>', unsafe_allow_html=True)
        st.latex(r"\text{W/L Ratio} = \frac{|\text{Average Win}|}{|\text{Average Loss}|}")
        
        st.markdown("""
        <div class="interpretation-box">
        How much you make on winning days vs how much you lose on losing days.
        
        <ul>
        <li><b>Ratio > 1:</b> Wins are larger than losses (good)</li>
        <li><b>Ratio < 1:</b> Losses are larger than wins (need high win rate to compensate)</li>
        </ul>
        
        <b>Combination Matters:</b>
        <ul>
        <li>High Win Rate + High W/L Ratio = Ideal</li>
        <li>High Win Rate + Low W/L Ratio = Death by a thousand cuts (many small wins, rare huge losses)</li>
        <li>Low Win Rate + High W/L Ratio = Acceptable (trend-following style)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif metric_category == "Price Metrics":
        st.markdown('<p class="sub-header">52-Week High/Low</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <b>What It Means:</b><br>
        Highest and lowest prices in the last 252 trading days (1 year).
        
        <b>% to 52W High:</b>
        <ul>
        <li><b>90-100%:</b> Near all-time high (momentum, potential resistance)</li>
        <li><b>70-90%:</b> Healthy position</li>
        <li><b>50-70%:</b> Moderate correction</li>
        <li><b>< 50%:</b> Significant decline (value opportunity or serious problems)</li>
        </ul>
        
        <b>Trading Strategy:</b>
        <ul>
        <li>Momentum traders: Buy near 52W highs (breakout)</li>
        <li>Value investors: Buy near 52W lows (contrarian)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif metric_category == "Chart Explanations":
        st.markdown('<p class="sub-header">📊 Chart 1: Regression Analysis</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <b>Purpose:</b> Visualize the relationship between stock returns and market returns.
        
        <b>How to Read:</b>
        <ul>
        <li><b>X-Axis:</b> NIFTY daily returns (%)</li>
        <li><b>Y-Axis:</b> Stock daily returns (%)</li>
        <li><b>Dots:</b> Each point is one trading day</li>
        <li><b>Line:</b> Best-fit regression line (slope = beta)</li>
        </ul>
        
        <b>Interpretation:</b>
        <ul>
        <li><b>Steep slope (β > 1):</b> Stock amplifies market moves</li>
        <li><b>Flat slope (β < 1):</b> Stock is less volatile than market</li>
        <li><b>Tight clustering around line:</b> High R² (market explains returns well)</li>
        <li><b>Wide scatter:</b> Low R² (other factors matter)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Chart 2: Returns Distribution</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <b>Purpose:</b> Compare the frequency distribution of stock vs market returns.
        
        <b>How to Read:</b>
        <ul>
        <li><b>X-Axis:</b> Return buckets (%)</li>
        <li><b>Y-Axis:</b> Frequency (number of days)</li>
        <li>Overlapping histograms show both distributions</li>
        </ul>
        
        <b>What to Look For:</b>
        <ul>
        <li><b>Width of distribution:</b> Wider = more volatile</li>
        <li><b>Center position:</b> Right of zero = positive average returns</li>
        <li><b>Symmetry:</b> Symmetric = normal, Skewed = asymmetric risk</li>
        <li><b>Fat tails:</b> Long tails = extreme events more common</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Chart 3: Cumulative Returns</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <b>Purpose:</b> Track wealth accumulation over time (both normalized to 100).
        
        <b>How to Read:</b>
        <ul>
        <li>Starting point: 100 (represents ₹100 invested on day 1)</li>
        <li>Ending point: Final wealth from that ₹100</li>
        <li><b>Stock line above NIFTY:</b> Outperformance</li>
        <li><b>Stock line below NIFTY:</b> Underperformance</li>
        </ul>
        
        <b>Analysis:</b>
        <ul>
        <li><b>Divergence:</b> Growing gap = consistent outperformance</li>
        <li><b>Convergence:</b> Closing gap = losing advantage</li>
        <li><b>Crossovers:</b> Leadership changes</li>
        <li><b>Steeper slope:</b> Faster wealth creation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Chart 4: Drawdown</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <b>Purpose:</b> Visualize peak-to-trough declines during the period.
        
        <b>How to Read:</b>
        <ul>
        <li><b>Y-Axis:</b> % below previous peak (always ≤ 0)</li>
        <li><b>Touches zero:</b> New all-time high achieved</li>
        <li><b>Deep valleys:</b> Significant losses from peak</li>
        <li><b>Underwater periods:</b> Time spent below previous peak</li>
        </ul>
        
        <b>Key Insights:</b>
        <ul>
        <li><b>Deepest point:</b> Maximum Drawdown (worst loss)</li>
        <li><b>Width of valleys:</b> Recovery time needed</li>
        <li><b>Frequent deep dips:</b> High volatility, emotional stress</li>
        <li><b>Quick recoveries:</b> Resilient stock</li>
        </ul>
        
        <b>Use Case:</b> Assess if you could psychologically handle the worst drops
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Chart 5: Rolling Beta</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <b>Purpose:</b> Show how beta changes over time using moving windows.
        
        <b>How to Read:</b>
        <ul>
        <li>Multiple lines for different windows (30-day, 90-day, 180-day)</li>
        <li><b>Y-Axis:</b> Beta value at each point in time</li>
        <li><b>Upward trend:</b> Stock becoming more volatile vs market</li>
        <li><b>Downward trend:</b> Stock becoming more defensive</li>
        </ul>
        
        <b>Interpretation:</b>
        <ul>
        <li><b>Stable horizontal line:</b> Consistent beta (predictable)</li>
        <li><b>Volatile fluctuations:</b> Changing market sensitivity</li>
        <li><b>Recent trend:</b> Current risk profile</li>
        <li><b>Longer window (180d):</b> Smoother, more stable</li>
        <li><b>Shorter window (30d):</b> More reactive to recent changes</li>
        </ul>
        
        <b>Use Case:</b> Identify regime changes (bull/bear market behavior shifts)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Portfolio Charts</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <b>1. Beta Distribution Bar Chart:</b>
        <ul>
        <li>Compare beta of each holding vs portfolio beta</li>
        <li>Red bars (β > 1.5): High risk contributors</li>
        <li>Green bars (β < 1): Defensive positions</li>
        <li>Identify which stocks drive portfolio volatility</li>
        </ul>
        
        <b>2. Risk-Return Scatter Map:</b>
        <ul>
        <li><b>X-Axis:</b> Volatility (risk)</li>
        <li><b>Y-Axis:</b> Annual return (reward)</li>
        <li><b>Bubble size:</b> Portfolio weight</li>
        <li><b>Color:</b> Sharpe ratio (green = better risk-adjusted)</li>
        <li><b>Ideal position:</b> Top-left (high return, low risk)</li>
        <li><b>Avoid:</b> Bottom-right (low return, high risk)</li>
        </ul>
        
        <b>3. Weight Distribution Pie:</b>
        <ul>
        <li>Visual representation of portfolio allocation</li>
        <li>Identify concentration risk (too much in one stock)</li>
        <li>Check if weights align with strategy</li>
        <li>Diversification assessment</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background:#e8f4f8;padding:1.5rem;border-radius:8px;margin-top:2rem;">
    <h3 style="color:#1f77b4;margin-top:0;">💡 Key Takeaways</h3>
    
    <b>For Stock Selection:</b>
    <ol>
    <li><b>Risk Assessment:</b> Beta, Volatility, Max Drawdown, VaR → Can you handle the risk?</li>
    <li><b>Return Quality:</b> Sharpe, Sortino, Jensen's Alpha → Is return worth the risk?</li>
    <li><b>Consistency:</b> Win Rate, W/L Ratio, Tracking Error → How predictable?</li>
    <li><b>Distribution:</b> Skewness, Kurtosis → Tail risk assessment</li>
    </ol>
    
    <b>Red Flags:</b>
    <ul>
    <li>High volatility + Negative Sharpe = Losing money with high risk</li>
    <li>Negative Jensen's Alpha = Underperforming risk-adjusted expectations</li>
    <li>High drawdown + Long recovery = Psychological pain</li>
    <li>Negative skew + High kurtosis = Frequent small gains, rare catastrophic losses</li>
    </ul>
    
    <b>Green Flags:</b>
    <ul>
    <li>Sharpe > 1.5 + Positive Jensen's Alpha = Excellent risk-adjusted outperformance</li>
    <li>Low volatility + High win rate = Stable wealth creation</li>
    <li>Positive skew + Moderate kurtosis = Asymmetric upside potential</li>
    <li>Beta 0.8-1.2 + High R² = Predictable market-aligned behavior</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background:#fff9e6;padding:1rem;border-radius:8px;margin-top:1rem;">
    <b>📖 Further Reading:</b><br>
    For deeper understanding, explore these topics:
    <ul>
    <li>Capital Asset Pricing Model (CAPM)</li>
    <li>Modern Portfolio Theory (MPT)</li>
    <li>Efficient Market Hypothesis (EMH)</li>
    <li>Factor Models (Fama-French)</li>
    <li>Black-Litterman Portfolio Optimization</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""<div style='text-align:center;color:#666;'>
<p>Built with Streamlit | Data: Yahoo Finance | Stats: Statsmodels & SciPy</p>
<p style='font-size:0.8rem;'>⚠️ Educational purposes only. Not financial advice.</p>
</div>""", unsafe_allow_html=True)
