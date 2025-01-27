import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
from datetime import datetime, timedelta
from openai import OpenAI



# App Owner Details
APP_OWNER = "Kshitij Kutumbe"
APP_OWNER_EMAIL = "kshitijkutumbe@gmail.com"
APP_OWNER_MEDIUM = "https://kshitijkutumbe.medium.com/"
APP_OWNER_LINKEDIN = "https://www.linkedin.com/in/kshitijkutumbe/"
LOGO_PATH = "logo.png"  # Ensure you have a logo.png file in the same directory

# Comprehensive list of major Indian companies with NSE symbols
INDIAN_STOCKS = {
    'Reliance Industries': 'RELIANCE',
    'TCS': 'TCS',
    'HDFC Bank': 'HDFCBANK',
    'Infosys': 'INFY',
    'ICICI Bank': 'ICICIBANK',
    'HUL': 'HINDUNILVR',
    'ITC': 'ITC',
    'SBI': 'SBIN',
    'Bharti Airtel': 'BHARTIARTL',
    'Larsen & Toubro': 'LT',
    'Kotak Mahindra Bank': 'KOTAKBANK',
    'Asian Paints': 'ASIANPAINT',
    'HCL Technologies': 'HCLTECH',
    'Maruti Suzuki': 'MARUTI',
    'Axis Bank': 'AXISBANK',
    'Sun Pharma': 'SUNPHARMA',
    'Tata Steel': 'TATASTEEL',
    'NTPC': 'NTPC',
    'ONGC': 'ONGC',
    'Wipro': 'WIPRO'
}

# Function to fetch stock data with error handling
def fetch_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        data = stock.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Enhanced fundamental analysis
def fundamental_analysis(ticker):
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        info = stock.info
        
        fundamentals = {
            'Current Price': info.get('currentPrice', 'N/A'),
            'Market Cap (Cr)': round(info.get('marketCap', 0)/1e7, 2) if info.get('marketCap') else 'N/A',
            'P/E Ratio': round(info.get('trailingPE'), 2) if info.get('trailingPE') else 'N/A',
            'P/B Ratio': round(info.get('priceToBook'), 2) if info.get('priceToBook') else 'N/A',
            'Dividend Yield (%)': round(info.get('dividendYield')*100, 2) if info.get('dividendYield') else 'N/A',
            'ROE (%)': round(info.get('returnOnEquity')*100, 2) if info.get('returnOnEquity') else 'N/A',
            'Debt/Equity': round(info.get('debtToEquity'), 2) if info.get('debtToEquity') else 'N/A',
            'EPS (TTM)': round(info.get('trailingEps'), 2) if info.get('trailingEps') else 'N/A',
            '52W High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52W Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'Beta': round(info.get('beta'), 2) if info.get('beta') else 'N/A'
        }
        return fundamentals
    except Exception as e:
        st.error(f"Fundamental analysis error: {str(e)}")
        return None

# Advanced technical analysis
def technical_analysis(data):
    try:
        # Trend Indicators
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
        data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
        
        # Momentum Indicators
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        data['Stochastic_%K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14)
        
        # Volatility Indicators
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Lower'] = bollinger.bollinger_lband()
        
        # Volume Indicators
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'])
        
        return data.dropna()
    except Exception as e:
        st.error(f"Technical analysis error: {str(e)}")
        return None

# Updated GPT-4 recommendation function using new API format
def get_gpt4_recommendation(analysis_text,api_key):
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a senior financial analyst specializing in Indian equities. 
                Provide professional recommendations considering both fundamental and technical factors. 
                Include price targets, risk assessment, and time horizon."""},
                {"role": "user", "content": analysis_text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Recommendation unavailable: {str(e)}"

# Professional Streamlit UI
def main():
    st.set_page_config(
        page_title="IntelliStock: Advanced Equity Analysis",
        page_icon="📈",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    
    .metric-box {padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .footer {text-align: center; padding: 10px; margin-top: 20px; font-size: 0.9em; color: #666;}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar with logo and owner details
    with st.sidebar:

        st.markdown("---")
        
        st.subheader("🔍 Analysis Parameters")
        api_key = st.text_input("Enter OpenAI API key")
        selected_company = st.selectbox(
            "Select Company",
            options=list(INDIAN_STOCKS.keys()),
            index=0,
            help="Select from Nifty 50 constituents"
        )
        
        analysis_period = st.selectbox(
            "Analysis Period",
            options=['3mo', '6mo', '1y', '2y', '5y'],
            index=2
        )
        
        st.markdown("---")
        st.markdown("**Disclaimer:** This is not investment advice. Always do your own research.")

    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Running comprehensive analysis..."):
            ticker = INDIAN_STOCKS[selected_company]
            data = fetch_stock_data(ticker, analysis_period)
            
            if data is None or data.empty:
                st.error("Failed to fetch data for selected company")
                return

            # Layout Organization
            st.title(f"📊 {selected_company} Analysis Report")
            st.subheader(f"({ticker}.NS) - {datetime.now().strftime('%d %b %Y')}")
            
            # Fundamental Analysis Section
            st.header("📈 Fundamental Analysis")
            fundamentals = fundamental_analysis(ticker)
            
            if fundamentals:
                cols = st.columns(4)
                metric_counter = 0
                for metric, value in fundamentals.items():
                    with cols[metric_counter % 4]:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h5 style="color: #4a4a4a; margin-bottom: 5px;">{metric}</h5>
                            <h3 style="color: #2c82c9;">{value}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        metric_counter += 1

            # Technical Analysis Section
            st.header("📉 Technical Analysis")
            data = technical_analysis(data)
            
            # Interactive Price Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.1, 
                              row_heights=[0.7, 0.3])
            
            # Price and Moving Averages
            fig.add_trace(go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'],
                                        name='Price'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], 
                                   line=dict(color='orange', width=1.5),
                                   name='50 SMA'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], 
                                   line=dict(color='green', width=1.5),
                                   name='200 SMA'), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'],
                                   line=dict(color='purple', width=1),
                                   name='RSI'), row=2, col=1)
            
            fig.update_layout(height=800, showlegend=True,
                             xaxis_rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)

            # Technical Indicators Table
            st.subheader("Latest Technical Indicators")
            latest_tech = data[['RSI', 'MACD', 'Stochastic_%K', 'BB_Upper', 'BB_Lower', 'VWAP']].tail().round(2)
            st.dataframe(latest_tech.style.background_gradient(cmap='Blues'), use_container_width=True)

            # AI Recommendation Section
            st.header("🤖 AI-Powered Recommendation")
            analysis_text = f"""
            Company: {selected_company} ({ticker})
            Fundamental Analysis: {fundamentals}
            Technical Analysis Summary:
            - Last Close: {data['Close'].iloc[-1]:.2f}
            - 50 SMA vs 200 SMA: {data['SMA_50'].iloc[-1]:.2f} vs {data['SMA_200'].iloc[-1]:.2f}
            - RSI: {data['RSI'].iloc[-1]:.2f}
            - MACD: {data['MACD'].iloc[-1]:.2f}
            - Bollinger Bands: {data['BB_Upper'].iloc[-1]:.2f} | {data['BB_Lower'].iloc[-1]:.2f}
            """
            
            recommendation = get_gpt4_recommendation(analysis_text,api_key)
            st.markdown(f"""
            <div style="padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #2c82c9;">
                {recommendation}
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.sidebar.markdown(f"### Developed by **{APP_OWNER}**")
    st.sidebar.markdown(f"📧 **Email:** [{APP_OWNER_EMAIL}](mailto:{APP_OWNER_EMAIL})")
    st.sidebar.markdown(f"📝 **Medium:** [Read my articles]({APP_OWNER_MEDIUM})")
    st.sidebar.markdown(f"🔗 **LinkedIn:** [Connect with me]({APP_OWNER_LINKEDIN})")
    st.markdown(f"""
    <div class="footer">
        Developed with ❤️ by <a href="{APP_OWNER_LINKEDIN}" target="_blank">{APP_OWNER}</a> | 
        <a href="mailto:{APP_OWNER_EMAIL}">Contact Me</a> | 
        <a href="{APP_OWNER_MEDIUM}" target="_blank">Read My Articles</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()