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

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Functions
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data_with_retry(ticker, start_date, end_date, frequency='1d', max_retries=3):
    """
    Fetch stock and NIFTY data with retry logic and exponential backoff
    
    Parameters:
    - ticker: NSE stock ticker (without .NS)
    - start_date: Start date for data
    - end_date: End date for data
    - frequency: '1d', '1wk', or '1mo'
    - max_retries: Maximum number of retry attempts
    
    Returns:
    - stock_data: DataFrame with stock prices
    - nifty_data: DataFrame with NIFTY prices
    """
    stock_ticker = f"{ticker}.NS"
    nifty_ticker = "^NSEI"
    
    for attempt in range(max_retries):
        try:
            # Add delay between attempts to avoid rate limiting
            if attempt > 0:
                wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                time.sleep(wait_time)
            
            # Download stock data
            stock_data = yf.download(
                stock_ticker,
                start=start_date,
                end=end_date,
                interval=frequency,
                progress=False
            )
            
            # Small delay between requests
            time.sleep(0.5)
            
            # Download NIFTY data
            nifty_data = yf.download(
                nifty_ticker,
                start=start_date,
                end=end_date,
                interval=frequency,
                progress=False
            )
            
            # Validate data
            if stock_data.empty:
                raise ValueError(f"No data found for ticker {stock_ticker}. Please verify the ticker symbol is correct.")
            
            if nifty_data.empty:
                raise ValueError("Unable to fetch NIFTY index data. Please try again later.")
            
            # Handle multi-level columns if present
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(1)
            if isinstance(nifty_data.columns, pd.MultiIndex):
                nifty_data.columns = nifty_data.columns.droplevel(1)
            
            return stock_data, nifty_data, stock_ticker
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if 'rate limit' in error_msg or 'too many requests' in error_msg or '429' in error_msg:
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    raise Exception(
                        "Yahoo Finance rate limit exceeded. Please try one of these solutions:\n"
                        "1. Wait 1-2 minutes and try again\n"
                        "2. Use a longer time period (Weekly/Monthly instead of Daily)\n"
                        "3. Reduce the date range\n"
                        "4. Try a different stock ticker first"
                    )
            else:
                # For other errors, raise immediately
                raise Exception(f"Error fetching data: {str(e)}")
    
    raise Exception("Failed to fetch data after multiple attempts. Please try again later.")

def calculate_returns(stock_data, nifty_data):
    """
    Calculate percentage returns and align data
    
    Parameters:
    - stock_data: DataFrame with stock prices
    - nifty_data: DataFrame with NIFTY prices
    
    Returns:
    - combined_df: DataFrame with aligned returns
    """
    try:
        # Calculate percentage returns
        stock_returns = stock_data['Close'].pct_change() * 100
        nifty_returns = nifty_data['Close'].pct_change() * 100
        
        # Create DataFrame
        returns_df = pd.DataFrame({
            'Stock_Return': stock_returns,
            'Nifty_Return': nifty_returns
        })
        
        # Remove missing values
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:
            raise ValueError("Insufficient data points for regression. Need at least 30 observations.")
        
        return returns_df
        
    except Exception as e:
        raise Exception(f"Error calculating returns: {str(e)}")

def calculate_beta(returns_df):
    """
    Calculate Beta using OLS regression
    
    Parameters:
    - returns_df: DataFrame with stock and nifty returns
    
    Returns:
    - results_dict: Dictionary with regression statistics
    - model: Fitted OLS model
    """
    try:
        # Prepare data for regression
        y = returns_df['Stock_Return']
        X = add_constant(returns_df['Nifty_Return'])
        
        # Fit OLS model
        model = OLS(y, X).fit()
        
        # Extract statistics
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
    """
    Create scatter plot with regression line
    """
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=returns_df['Nifty_Return'],
        y=returns_df['Stock_Return'],
        mode='markers',
        name='Returns',
        marker=dict(size=5, color='blue', opacity=0.5)
    ))
    
    # Regression line
    x_range = np.linspace(returns_df['Nifty_Return'].min(), 
                          returns_df['Nifty_Return'].max(), 100)
    y_pred = model.params['const'] + model.params['Nifty_Return'] * x_range
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='Regression Line',
        line=dict(color='red', width=2)
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
    """
    Calculate rolling beta
    
    Parameters:
    - returns_df: DataFrame with returns
    - window: Rolling window size
    
    Returns:
    - rolling_beta: Series with rolling beta values
    """
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
    """
    Plot rolling beta for multiple windows
    """
    fig = go.Figure()
    
    colors = ['blue', 'green', 'orange']
    
    for idx, window in enumerate(windows):
        rolling_beta = calculate_rolling_beta(returns_df, window)
        fig.add_trace(go.Scatter(
            x=rolling_beta.index,
            y=rolling_beta,
            mode='lines',
            name=f'{window}-Day Rolling Beta',
            line=dict(color=colors[idx], width=2)
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
    """
    Plot normalized price history
    """
    # Normalize prices to 100
    stock_norm = (stock_data['Close'] / stock_data['Close'].iloc[0]) * 100
    nifty_norm = (nifty_data['Close'] / nifty_data['Close'].iloc[0]) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=stock_norm.index,
        y=stock_norm,
        mode='lines',
        name=ticker,
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=nifty_norm.index,
        y=nifty_norm,
        mode='lines',
        name='NIFTY',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f'{ticker} vs NIFTY - Normalized Price History (Base = 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        hovermode='x unified',
        height=500
    )
    
    return fig

# Main App
def main():
    st.markdown('<p class="main-header">📊 NSE Stock Beta Calculator</p>', unsafe_allow_html=True)
    st.markdown("Calculate levered Beta for NSE-listed stocks using regression analysis with NIFTY 50 as the market benchmark.")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Ticker Input
    ticker = st.sidebar.text_input(
        "NSE Ticker Symbol",
        value="RELIANCE",
        help="Enter NSE ticker without .NS suffix (e.g., RELIANCE, TCS, HDFCBANK)"
    ).upper()
    
    # Frequency Selection
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
    
    # Time Period Selection
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
    
    # Calculate Button
    calculate_btn = st.sidebar.button("🚀 Calculate Beta", type="primary", use_container_width=True)
    
    # Info box about rate limits
    with st.sidebar.expander("ℹ️ Troubleshooting Rate Limits"):
        st.markdown("""
        **If you see rate limit errors:**
        1. Wait 1-2 minutes before retrying
        2. Use Weekly/Monthly frequency
        3. Try a shorter date range
        4. Clear cache from the menu (⋮) → Clear cache
        """)
    
    # Main Content
    if calculate_btn:
        if not ticker:
            st.error("⚠️ Please enter a valid NSE ticker symbol.")
            return
        
        try:
            # Progress with status updates
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            status_placeholder.info(f'🔄 Fetching data for {ticker}... (This may take 10-15 seconds)')
            progress_bar.progress(20)
            
            stock_data, nifty_data, full_ticker = fetch_data_with_retry(
                ticker, start_date, end_date, frequency
            )
            progress_bar.progress(50)
            
            status_placeholder.info('📊 Calculating returns...')
            returns_df = calculate_returns(stock_data, nifty_data)
            progress_bar.progress(70)
            
            status_placeholder.info('📈 Running regression analysis...')
            results, model = calculate_beta(returns_df)
            progress_bar.progress(100)
            
            status_placeholder.empty()
            progress_bar.empty()
            
            st.success(f'✅ Analysis completed for {full_ticker}')
            
            # Display Results
            st.header(f"📈 Beta Analysis: {ticker}")
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Beta (β)", f"{results['beta']:.4f}")
            
            with col2:
                st.metric("Alpha (α)", f"{results['alpha']:.4f}%")
            
            with col3:
                st.metric("R²", f"{results['r_squared']:.4f}")
            
            with col4:
                st.metric("Observations", f"{results['observations']}")
            
            # Detailed Statistics
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
            elif results['beta'] < 1 and results['beta'] > 0:
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
            
            # Visualizations
            st.header("📊 Visualizations")
            
            # Regression Plot
            st.subheader("Regression Analysis")
            fig_regression = plot_regression(returns_df, model, ticker)
            st.plotly_chart(fig_regression, use_container_width=True)
            
            # Price History
            st.subheader("Price History")
            fig_price = plot_price_history(stock_data, nifty_data, ticker)
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Rolling Beta
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
            
            # Download Data
            st.header("💾 Download Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_returns = returns_df.to_csv()
                st.download_button(
                    label="Download Returns Data (CSV)",
                    data=csv_returns,
                    file_name=f"{ticker}_returns.csv",
                    mime="text/csv"
                )
            
            with col2:
                stats_csv = stats_df.to_csv(index=False)
                st.download_button(
                    label="Download Statistics (CSV)",
                    data=stats_csv,
                    file_name=f"{ticker}_beta_stats.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            
            # Provide helpful suggestions
            if 'rate limit' in str(e).lower() or 'too many requests' in str(e).lower():
                st.warning("""
                **💡 Rate Limit Solutions:**
                - ⏱️ Wait 1-2 minutes before trying again
                - 📅 Use **Weekly** or **Monthly** frequency instead of Daily
                - 📉 Try a shorter date range (e.g., 1 year instead of 5 years)
                - 🔄 Clear the cache: Menu (⋮) → Settings → Clear cache
                - 🔁 Try a different stock ticker first
                """)
            else:
                st.info("💡 Please check the ticker symbol and try again. Make sure it's a valid NSE-listed stock.")
    
    else:
        # Welcome Message
        st.info("👈 Enter a ticker symbol in the sidebar and click 'Calculate Beta' to begin analysis.")
        
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Enter NSE Ticker**: Input the stock symbol (e.g., RELIANCE, TCS, HDFCBANK)
        2. **Select Frequency**: Choose between Daily, Weekly, or Monthly data
        3. **Choose Time Period**: Select a predefined period or set custom dates
        4. **Calculate**: Click the button to run the analysis
        
        **Beta Interpretation:**
        - **β > 1**: Stock is more volatile than the market
        - **β = 1**: Stock moves in line with the market
        - **β < 1**: Stock is less volatile than the market
        - **β < 0**: Stock moves inversely to the market
        
        **⚠️ Important:** If you encounter rate limit errors, use Weekly/Monthly frequency or wait 1-2 minutes between requests.
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

if __name__ == "__main__":
    main()
