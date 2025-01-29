import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from openai import OpenAI
from duckduckgo_search import DDGS
from serpapi import GoogleSearch

# App Owner Details
APP_OWNER = "Kshitij Kutumbe"
APP_OWNER_EMAIL = "kshitijkutumbe@gmail.com"
APP_OWNER_MEDIUM = "https://kshitijkutumbe.medium.com/"
APP_OWNER_LINKEDIN = "https://www.linkedin.com/in/kshitijkutumbe/"
LOGO_PATH = "logo.png"

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

def fetch_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        data = stock.history(period=period)
        return data.dropna() if not data.empty else None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def fundamental_analysis(ticker):
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        info = stock.info
        
        return {
            'Current Price': info.get('currentPrice', np.nan),
            'Market Cap (Cr)': round(info.get('marketCap', 0)/1e7, 2),
            'P/E Ratio': round(info.get('trailingPE', np.nan), 2),
            'P/B Ratio': round(info.get('priceToBook', np.nan), 2),
            'Dividend Yield (%)': round(info.get('dividendYield', 0)*100, 2),
            'ROE (%)': round(info.get('returnOnEquity', 0)*100, 2),
            'Debt/Equity': round(info.get('debtToEquity', np.nan), 2),
            'EPS (TTM)': round(info.get('trailingEps', np.nan), 2),
            '52W High': info.get('fiftyTwoWeekHigh', np.nan),
            '52W Low': info.get('fiftyTwoWeekLow', np.nan),
            'Beta': round(info.get('beta', np.nan), 2)
        }
    except Exception as e:
        st.error(f"Fundamental analysis failed: {str(e)}")
        return None

def technical_analysis(data):
    try:
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], 50)
        data['SMA_200'] = ta.trend.sma_indicator(data['Close'], 200)
        data['RSI'] = ta.momentum.rsi(data['Close'], 14)
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Lower'] = bollinger.bollinger_lband()
        return data.dropna()
    except Exception as e:
        st.error(f"Technical analysis failed: {str(e)}")
        return None

def get_news(query):
    """Fetch news using DuckDuckGo's official API"""
    try:
        news_items = []
        with DDGS() as ddgs:
            for result in ddgs.news(
                keywords=f"{query} stock", 
                region="in-en", 
                safesearch="moderate", 
                timelimit="d",
                max_results=5
            ):
                news_items.append({
                    "title": result.get('title', 'No title available'),
                    "url": result.get('url', '#'),
                    "snippet": result.get('body', 'No description available')
                })
        return news_items
    except Exception as e:
        st.error(f"News fetch failed: {str(e)}")
        return []
    
def get_news(query, api_key):
    try:
        params = {
            "q": query,
            "tbm": "nws",
            "api_key": api_key,
            "hl": "en",
            "gl": "in"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get('news_results', [])
    except Exception as e:
        st.error(f"News fetch error: {str(e)}")
        return []

def get_gpt4_recommendation(fundamental, technical, news_items, api_key):
    """Generate GPT-4 recommendation with integrated news analysis"""
    news_context = "\n".join(
        [f"Headline: {item['title']}\nSummary: {item['snippet']}\n" 
         for item in news_items]
    ) if news_items else "No recent news available"

    prompt = f"""
    **Fundamental Analysis**
    {fundamental}

    **Technical Analysis**
    {technical}

    **Market News**
    {news_context}

    Provide detailed recommendation covering:
    1. Current valuation vs historical averages
    2. Technical trend analysis
    3. News sentiment impact
    4. Risk assessment
    5. Price targets (short-term 1-3 months, long-term 6-12 months)
    6. Suggested portfolio allocation percentage
    7. Stop-loss levels
    """

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a senior financial analyst at a top investment bank."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI analysis failed: {str(e)}"

def main():
    st.set_page_config(
        page_title="IntelliStock Pro",
        page_icon="📈",
        layout="wide"
    )

    # Custom CSS
    st.markdown(f"""
    <style>
    .main {{
        background-image: url({LOGO_PATH});
        background-size: 100px;
        background-repeat: no-repeat;
        background-position: 98% 2%;
    }}
    .metric-box {{
        padding: 15px; 
        background: white; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .footer {{
        text-align: center; 
        padding: 10px; 
        margin-top: 20px; 
        font-size: 0.9em; 
        color: #666;
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image(LOGO_PATH, width=100)
        st.markdown(f"### Developed by **{APP_OWNER}**")
        st.markdown(f"📧 **Email:** [{APP_OWNER_EMAIL}](mailto:{APP_OWNER_EMAIL})")
        st.markdown(f"📝 **Medium:** [Read my articles]({APP_OWNER_MEDIUM})")
        st.markdown(f"🔗 **LinkedIn:** [Connect with me]({APP_OWNER_LINKEDIN})")
        st.markdown("---")
        
        st.subheader("Analysis Parameters")
        api_key = st.text_input("Enter OpenAI API key", type="password")
        serpapi_key = st.sidebar.text_input("SerpAPI Key", type="password")
        selected_company = st.selectbox("Select Company", list(INDIAN_STOCKS.keys()))
        analysis_period = st.selectbox("Analysis Period", ['3mo', '6mo', '1y', '2y', '5y'], index=2)
        
        if st.button("Run Comprehensive Analysis", type="primary"):
            st.session_state.run_analysis = True

    if 'run_analysis' in st.session_state:
        with st.spinner("Running institutional-grade analysis..."):
            ticker = INDIAN_STOCKS[selected_company]
            data = fetch_stock_data(ticker, analysis_period)
            
            if data is None:
                st.error("Failed to fetch stock data")
                return

            # Fundamental Analysis
            st.header("💰 Fundamental Analysis")
            fundamentals = fundamental_analysis(ticker)
            
            if fundamentals:
                cols = st.columns(4)
                metrics = list(fundamentals.items())
                for i in range(0, len(metrics), 4):
                    current_metrics = metrics[i:i+4]
                    for j, (k, v) in enumerate(current_metrics):
                        with cols[j]:
                            st.markdown(f"""
                            <div class="metric-box">
                                <h5>{k}</h5>
                                <h3>{v}</h3>
                            </div>
                            """, unsafe_allow_html=True)

            # Technical Analysis
            st.header("📈 Technical Analysis")
            data = technical_analysis(data)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.1, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'],
                                       high=data['High'], low=data['Low'],
                                       close=data['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'],
                                   line=dict(color='orange', width=2),
                                   name='50 SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'],
                                   line=dict(color='green', width=2),
                                   name='200 SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'],
                                   line=dict(color='purple', width=2),
                                   name='RSI'), row=2, col=1)
            fig.update_layout(height=700, showlegend=False,
                             xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # News Analysis
            st.header("📰 Market News Digest")
            news_items = get_news(selected_company,serpapi_key)
            
            if news_items:
                for item in news_items:
                    with st.expander(f"{item['title']}"):
                        st.write(item['snippet'])
                        st.markdown(f"[Read full article]({item['link']})")
            else:
                st.warning("Could not fetch recent news updates")

            # AI Recommendation
            if api_key:
                st.header("🤖 Institutional Recommendation")
                fund_text = "\n".join([f"{k}: {v}" for k,v in fundamentals.items()])
                tech_text = f"""
                Price: ₹{data['Close'].iloc[-1]:.2f}
                50 SMA: ₹{data['SMA_50'].iloc[-1]:.2f}
                200 SMA: ₹{data['SMA_200'].iloc[-1]:.2f}
                RSI: {data['RSI'].iloc[-1]:.1f}
                MACD: {data['MACD'].iloc[-1]:.2f}
                Bollinger Bands: ₹{data['BB_Upper'].iloc[-1]:.2f}/₹{data['BB_Lower'].iloc[-1]:.2f}
                """
                
                recommendation = get_gpt4_recommendation(fund_text, tech_text, news_items, api_key)
                st.markdown(f"""
                <div style="padding:20px; background:#f8f9fa; border-radius:10px; border-left:4px solid #2c82c9;">
                    {recommendation}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Enter OpenAI API key for professional recommendations")

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        Developed with ❤️ by <a href="{APP_OWNER_LINKEDIN}" target="_blank">{APP_OWNER}</a> | 
        <a href="mailto:{APP_OWNER_EMAIL}">Contact</a> | 
        <a href="{APP_OWNER_MEDIUM}" target="_blank">Market Insights</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()