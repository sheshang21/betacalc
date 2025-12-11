import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
import time

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="NSE Stock Beta Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (includes tooltip + simple metric card styles) ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fb;
        padding: 0.6rem 0.9rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #333;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111;
    }
    .tooltip {
      position: relative;
      display: inline-block;
      cursor: help;
      color: #1f77b4;
      font-weight: 700;
      margin-left: 6px;
    }
    .tooltip .tooltiptext {
      visibility: hidden;
      width: 260px;
      background-color: #1f77b4;
      color: #fff;
      text-align: left;
      border-radius: 6px;
      padding: 10px;
      position: absolute;
      z-index: 9999;
      bottom: 125%;
      left: 50%;
      margin-left: -130px;
      opacity: 0;
      transition: opacity 0.25s;
      font-size: 0.85rem;
      line-height: 1.2;
    }
    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
    .small-muted {
        color: #666;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------
# Functions
# ----------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data_with_retry(ticker, start_date, end_date, frequency='1d', max_retries=3):
    """
    Fetch stock and NIFTY data with retry logic and exponential backoff
    """
    stock_ticker = f"{ticker}.NS"
    nifty_ticker = "^NSEI"
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            
            stock_data = yf.download(
                stock_ticker,
                start=start_date,
                end=end_date,
                interval=frequency,
                progress=False
            )
            time.sleep(0.5)
            nifty_data = yf.download(
                nifty_ticker,
                start=start_date,
                end=end_date,
                interval=frequency,
                progress=False
            )
            
            if stock_data.empty:
                raise ValueError(f"No data found for ticker {stock_ticker}. Please verify the ticker symbol is correct.")
            if nifty_data.empty:
                raise ValueError("Unable to fetch NIFTY index data. Please try again later.")
            
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(1)
            if isinstance(nifty_data.columns, pd.MultiIndex):
                nifty_data.columns = nifty_data.columns.droplevel(1)
            
            return stock_data, nifty_data, stock_ticker
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or 'too many requests' in error_msg or '429' in error_msg:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise Exception(
                        "Yahoo Finance rate limit exceeded. Please try one of these solutions:\n"
                        "1. Wait 1-2 minutes and try again\n"
                        "2. Use a longer time period (Weekly/Monthly instead of Daily)\n"
                        "3. Reduce the date range\n"
                        "4. Try a different stock ticker first"
                    )
            else:
                raise Exception(f"Error fetching data: {str(e)}")
    
    raise Exception("Failed to fetch data after multiple attempts. Please try again later.")

def calculate_returns(stock_data, nifty_data):
    """
    Calculate percentage returns and align data
    """
    try:
        stock_returns = stock_data['Close'].pct_change() * 100
        nifty_returns = nifty_data['Close'].pct_change() * 100
        
        returns_df = pd.DataFrame({
            'Stock_Return': stock_returns,
            'Nifty_Return': nifty_returns
        })
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:
            raise ValueError("Insufficient data points for regression. Need at least 30 observations.")
        
        return returns_df
    except Exception as e:
        raise Exception(f"Error calculating returns: {str(e)}")

def calculate_beta(returns_df):
    """
    Calculate Beta using OLS regression
    """
    try:
        y = returns_df['Stock_Return']
        X = add_constant(returns_df['Nifty_Return'])
        model = OLS(y, X).fit()
        results_dict = {
            'beta': model.params['Nifty_Return'],
            'alpha': model.params['const'],
            'r_squared': model.rsquared,
            'std_error': model.bse['Nifty_Return'],
            'p_value': model.pvalues['Nifty_Return'],
            'conf_int_lower': model.conf_int().loc['Nifty_Return', 0],
            'conf_int_upper': model.conf_int().loc['Nifty_Return', 1],
            'observations': len(returns_df)
        }
        return results_dict, model
    except Exception as e:
        raise Exception(f"Error in regression calculation: {str(e)}")

def plot_regression(returns_df, model, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns_df['Nifty_Return'],
        y=returns_df['Stock_Return'],
        mode='markers',
        name='Returns',
        marker=dict(size=5, opacity=0.5)
    ))
    x_range = np.linspace(returns_df['Nifty_Return'].min(), returns_df['Nifty_Return'].max(), 100)
    y_pred = model.params['const'] + model.params['Nifty_Return'] * x_range
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='Regression Line',
        line=dict(width=2)
    ))
    fig.update_layout(
        title=f'{ticker} vs NIFTY - Regression Analysis',
        xaxis_title='NIFTY Returns (%)',
        yaxis_title=f'{ticker} Returns (%)',
        hovermode='closest',
        height=500
    )
    return fig

def calculate_rolling_beta(returns_df, window):
    rolling_beta = []
    dates = []
    for i in range(window, len(returns_df)):
        subset = returns_df.iloc[i-window:i]
        y = subset['Stock_Return']
        X = add_constant(subset['Nifty_Return'])
        try:
            model = OLS(y, X).fit()
            rolling_beta.append(model.params['Nifty_Return'])
            dates.append(subset.index[-1])
        except:
            rolling_beta.append(np.nan)
            dates.append(subset.index[-1])
    return pd.Series(rolling_beta, index=dates)

def plot_rolling_beta(returns_df, windows, ticker):
    fig = go.Figure()
    colors = ['blue', 'green', 'orange']
    for idx, window in enumerate(windows):
        rolling_beta = calculate_rolling_beta(returns_df, window)
        fig.add_trace(go.Scatter(
            x=rolling_beta.index,
            y=rolling_beta,
            mode='lines',
            name=f'{window}-Day Rolling Beta',
            line=dict(width=2)
        ))
    fig.update_layout(
        title=f'{ticker} - Rolling Beta Analysis',
        xaxis_title='Date',
        yaxis_title='Beta',
        hovermode='x unified',
        height=500
    )
    return fig

def plot_price_history(stock_data, nifty_data, ticker):
    stock_norm = (stock_data['Close'] / stock_data['Close'].iloc[0]) * 100
    nifty_norm = (nifty_data['Close'] / nifty_data['Close'].iloc[0]) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_norm.index,
        y=stock_norm,
        mode='lines',
        name=ticker,
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=nifty_norm.index,
        y=nifty_norm,
        mode='lines',
        name='NIFTY',
        line=dict(width=2)
    ))
    fig.update_layout(
        title=f'{ticker} vs NIFTY - Normalized Price History (Base = 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        hovermode='x unified',
        height=500
    )
    return fig

# ----------------------
# Helper: Metric HTML block (with tooltip)
# ----------------------
def metric_html(title, value, tooltip_html):
    return f"""
    <div class="metric-card">
      <div class="metric-title">{title} {tooltip_html}</div>
      <div class="metric-value">{value}</div>
    </div>
    """

def tooltip_html(content):
    safe = content.replace("\n", "<br>")
    return f"""
    <span class="tooltip">❔
      <span class="tooltiptext">{safe}</span>
    </span>
    """

# ----------------------
# App Layout (single-file multipage via sidebar)
# ----------------------
page = st.sidebar.selectbox("🔹 Page", ["Calculator", "Formula Sheet"])

if page == "Calculator":
    # Main Header
    st.markdown('<p class="main-header">📊 NSE Stock Beta Calculator</p>', unsafe_allow_html=True)
    st.markdown("Calculate levered Beta for NSE-listed stocks using regression analysis with NIFTY 50 as the market benchmark.")
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    ticker = st.sidebar.text_input(
        "NSE Ticker Symbol",
        value="RELIANCE",
        help="Enter NSE ticker without .NS suffix (e.g., RELIANCE, TCS, HDFCBANK)"
    ).upper()
    
    frequency_map = {
        'Daily': '1d',
        'Weekly': '1wk',
        'Monthly': '1mo'
    }
    frequency_label = st.sidebar.selectbox(
        "Data Frequency",
        list(frequency_map.keys()),
        index=0,
        help="📌 Use Weekly/Monthly if you encounter rate limit errors"
    )
    frequency = frequency_map[frequency_label]
    
    st.sidebar.subheader("📅 Time Period")
    period_type = st.sidebar.radio("Select Period Type", ["Predefined", "Custom"])
    if period_type == "Predefined":
        period_options = {
            '1 Year': 365,
            '3 Years': 1095,
            '5 Years': 1825
        }
        period_label = st.sidebar.selectbox("Select Period", list(period_options.keys()))
        days = period_options[period_label]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=1095))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
    
    calculate_btn = st.sidebar.button("🚀 Calculate Beta", type="primary", use_container_width=True)
    
    with st.sidebar.expander("ℹ️ Troubleshooting Rate Limits"):
        st.markdown("""
        **If you see rate limit errors:**
        1. Wait 1-2 minutes before retrying  
        2. Use Weekly/Monthly frequency  
        3. Try a shorter date range  
        4. Clear cache from the menu (⋮) → Clear cache
        """)
    
    # If clicked
    if calculate_btn:
        if not ticker:
            st.error("⚠️ Please enter a valid NSE ticker symbol.")
        else:
            try:
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_placeholder.info(f'🔄 Fetching data for {ticker}... (This may take 10-15 seconds)')
                progress_bar.progress(15)
                
                stock_data, nifty_data, full_ticker = fetch_data_with_retry(
                    ticker, start_date, end_date, frequency
                )
                progress_bar.progress(45)
                
                status_placeholder.info('📊 Calculating returns...')
                returns_df = calculate_returns(stock_data, nifty_data)
                progress_bar.progress(65)
                
                status_placeholder.info('📈 Running regression analysis...')
                results, model = calculate_beta(returns_df)
                progress_bar.progress(100)
                
                status_placeholder.empty()
                progress_bar.empty()
                st.success(f'✅ Analysis completed for {full_ticker}')
                
                # Results header
                st.header(f"📈 Beta Analysis: {ticker}")
                
                # Tooltips text
                t_beta = tooltip_html("Beta Formula:\nβ = Cov(R_stock, R_market) / Var(R_market)\nMeasures sensitivity of stock returns to market returns.")
                t_alpha = tooltip_html("Alpha:\nIntercept from the regression (stock return unexplained by market).")
                t_r2 = tooltip_html("R²:\n1 - (SS_res / SS_tot). Fraction of variance explained by market.")
                t_obs = tooltip_html("Observations:\nNumber of paired return observations used in regression.")
                
                # display metrics using custom HTML cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(metric_html("Beta (β)", f"{results['beta']:.6f}", t_beta), unsafe_allow_html=True)
                with col2:
                    st.markdown(metric_html("Alpha (α)", f"{results['alpha']:.6f}%", t_alpha), unsafe_allow_html=True)
                with col3:
                    st.markdown(metric_html("R²", f"{results['r_squared']:.6f}", t_r2), unsafe_allow_html=True)
                with col4:
                    st.markdown(metric_html("Observations", f"{results['observations']}", t_obs), unsafe_allow_html=True)
                
                # Detailed stats table
                st.subheader("📊 Regression Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Beta', 'Alpha', 'R-Squared', 'Std Error', 'P-Value', 
                               '95% CI Lower', '95% CI Upper', 'Observations'],
                    'Value': [
                        f"{results['beta']:.6f}",
                        f"{results['alpha']:.6f}%",
                        f"{results['r_squared']:.6f}",
                        f"{results['std_error']:.6f}",
                        f"{results['p_value']:.6e}",
                        f"{results['conf_int_lower']:.6f}",
                        f"{results['conf_int_upper']:.6f}",
                        f"{results['observations']}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Interpretation
                st.subheader("💡 Interpretation")
                if results['beta'] > 1:
                    interpretation = f"**{ticker}** is more volatile than the market (Beta > 1). For every 1% move in NIFTY, {ticker} tends to move {results['beta']:.2f}%."
                elif 0 < results['beta'] < 1:
                    interpretation = f"**{ticker}** is less volatile than the market (0 < Beta < 1). For every 1% move in NIFTY, {ticker} tends to move {results['beta']:.2f}%."
                elif results['beta'] < 0:
                    interpretation = f"**{ticker}** moves inversely to the market (Beta < 0). It may serve as a hedge against market movements."
                else:
                    interpretation = f"**{ticker}** has minimal correlation with the market (Beta ≈ 0)."
                st.info(interpretation)
                
                if results['p_value'] < 0.05:
                    st.success(f"✅ Beta is statistically significant (p-value = {results['p_value']:.4e})")
                else:
                    st.warning(f"⚠️ Beta may not be statistically significant (p-value = {results['p_value']:.4e})")
                
                # Formula explanations (collapsible)
                st.subheader("📘 Formula Explanations")
                with st.expander("How Returns Are Calculated"):
                    st.markdown(r"""
                    **Return Formula (percentage)**  
                    
                    $$\text{Return}_t = \left(\frac{P_t - P_{t-1}}{P_{t-1}}\right) \times 100$$
                    
                    Where \(P_t\) = current period price, \(P_{t-1}\) = previous period price.
                    """)
                with st.expander("Beta (Regression-Based)"):
                    st.markdown(r"""
                    Regression equation used:  
                    
                    $$R_{stock} = \alpha + \beta R_{market} + \epsilon$$
                    
                    Beta can also be written as:  
                    $$\beta = \frac{\operatorname{Cov}(R_{stock}, R_{market})}{\operatorname{Var}(R_{market})}$$
                    """)
                with st.expander("Alpha and R-squared"):
                    st.markdown(r"""
                    **Alpha (α):** Intercept of the regression — portion of stock return unexplained by market.  
                    
                    **R²:** Goodness of fit; fraction of variance in stock returns explained by the market.
                    
                    $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
                    """)
                with st.expander("Rolling Beta"):
                    st.markdown(r"""
                    Rolling beta is computed by running OLS inside a sliding time window (e.g., 90 days):
                    
                    $$\beta^{(window)}_t = \text{OLS}\big(R_s(t-window:t),\; R_m(t-window:t)\big)$$
                    """)
                
                # Visualizations
                st.header("📊 Visualizations")
                st.subheader("Regression Analysis")
                fig_regression = plot_regression(returns_df, model, ticker)
                st.plotly_chart(fig_regression, use_container_width=True)
                
                st.subheader("Price History (Normalized)")
                fig_price = plot_price_history(stock_data, nifty_data, ticker)
                st.plotly_chart(fig_price, use_container_width=True)
                
                st.subheader("Rolling Beta Analysis")
                if len(returns_df) >= 180:
                    windows = [30, 90, 180]
                elif len(returns_df) >= 90:
                    windows = [30, 90]
                else:
                    windows = [30]
                if len(returns_df) >= 30:
                    fig_rolling = plot_rolling_beta(returns_df, windows, ticker)
                    st.plotly_chart(fig_rolling, use_container_width=True)
                else:
                    st.warning("⚠️ Insufficient data for rolling beta calculation (minimum 30 observations required).")
                
                # Download
                st.header("💾 Download Data")
                colx, coly = st.columns(2)
                with colx:
                    csv_returns = returns_df.to_csv()
                    st.download_button(
                        label="Download Returns Data (CSV)",
                        data=csv_returns,
                        file_name=f"{ticker}_returns.csv",
                        mime="text/csv"
                    )
                with coly:
                    stats_csv = stats_df.to_csv(index=False)
                    st.download_button(
                        label="Download Statistics (CSV)",
                        data=stats_csv,
                        file_name=f"{ticker}_beta_stats.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if 'rate limit' in str(e).lower() or 'too many requests' in str(e).lower():
                    st.warning("""
                    **💡 Rate Limit Solutions:**
                    - ⏱️ Wait 1-2 minutes before trying again
                    - 📅 Use **Weekly** or **Monthly** frequency instead of Daily
                    - 📉 Try a shorter date range
                    - 🔄 Clear the cache: Menu (⋮) → Settings → Clear cache
                    """)
                else:
                    st.info("💡 Please check the ticker symbol and try again. Make sure it's a valid NSE-listed stock.")
    else:
        # initial help screen
        st.info("👈 Enter a ticker symbol in the sidebar and click 'Calculate Beta' to begin analysis.")
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Enter NSE Ticker**: Input the stock symbol (e.g., RELIANCE, TCS, HDFCBANK)  
        2. **Select Frequency**: Choose Daily, Weekly, or Monthly  
        3. **Choose Time Period**: Predefined or custom dates  
        4. **Calculate**: Click the button to run the analysis
        """)
        st.subheader("📊 Example Tickers")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Large Cap**")
            st.code("RELIANCE\nTCS\nHDFCBANK\nINFY\nICICIBANK")
        with col2:
            st.markdown("**Mid Cap**")
            st.code("ADANIPORTS\nLT\nAXISBANK\nM&M\nTITAN")
        with col3:
            st.markdown("**IT Sector**")
            st.code("WIPRO\nHCLTECH\nTECHM\nLTIM\nCOFORGE")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with Streamlit | Data from Yahoo Finance | Regression via Statsmodels</p>
            <p style='font-size: 0.8rem;'>⚠️ For educational purposes only. Not financial advice.</p>
        </div>
    """, unsafe_allow_html=True)

# ----------------------
# Formula Sheet Page
# ----------------------
else:
    st.markdown('<p class="main-header">📘 Formula Sheet</p>', unsafe_allow_html=True)
    st.write("Complete list of formulas and explanations used in the Beta Calculator.")
    st.markdown("---")
    
    st.subheader("1. Returns Calculation")
    st.markdown(r"""
    **Return (percent)**  
    $$\text{Return}_t = \left(\frac{P_t - P_{t-1}}{P_{t-1}}\right) \times 100$$
    where \(P_t\) is price at time t and \(P_{t-1}\) is price at previous period.
    """)
    
    st.subheader("2. Regression Equation")
    st.markdown(r"""
    The basic regression model used to estimate beta:
    $$R_{stock} = \alpha + \beta R_{market} + \epsilon$$
    - \(R_{stock}\): stock returns  
    - \(R_{market}\): market returns (NIFTY)  
    - \(\alpha\): intercept (alpha)  
    - \(\beta\): slope (beta)  
    - \(\epsilon\): error term
    """)
    
    st.subheader("3. Beta (β)")
    st.markdown(r"""
    Beta measures sensitivity of stock returns to market returns:
    $$\beta = \frac{\operatorname{Cov}(R_{stock}, R_{market})}{\operatorname{Var}(R_{market})}$$
    Intuition:
    - \(\beta>1\): stock more volatile than market  
    - \(0<\beta<1\): stock less volatile than market  
    - \(\beta<0\): stock moves inversely to market
    """)
    
    st.subheader("4. Alpha (α)")
    st.markdown(r"""
    Alpha is the intercept of the regression and indicates the average return of the stock that is not explained by market movements.
    $$\alpha = \text{Intercept from OLS regression}$$
    """)
    
    st.subheader("5. R-Squared (R²)")
    st.markdown(r"""
    R-squared measures the proportion of variance in stock returns explained by the market:
    $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
    - Closer to 1: market explains most variation  
    - Closer to 0: market explains little variation
    """)
    
    st.subheader("6. Standard Error & Confidence Interval")
    st.markdown(r"""
    Standard error of beta (conceptual):
    $$SE_\beta = \sqrt{\frac{\sigma^2}{\sum (R_m - \bar{R}_m)^2 }}$$
    Confidence interval (approx):
    $$\beta \pm t_{(n-2, \; \alpha/2)} \times SE_\beta$$
    """)
    
    st.subheader("7. Rolling Beta")
    st.markdown(r"""
    Rolling beta is computed by running the same OLS regression inside a sliding window:
    $$\beta^{(window)}_t = \text{OLS}\big(R_s(t-window:t),\; R_m(t-window:t)\big)$$
    """)
    
    st.subheader("8. Normalized Price (for charts)")
    st.markdown(r"""
    To compare price series on the same axis we normalize to base 100:
    $$\text{Normalized Price}_t = \frac{P_t}{P_0} \times 100$$
    where \(P_0\) is the first available price in the selected period.
    """)
    
    st.markdown("---")
    st.markdown("<div class='small-muted'>Tip: Use the 'Calculator' page to run live calculations and hover the ❔ icons next to the metrics to quickly see short formula explanations.</div>", unsafe_allow_html=True)
