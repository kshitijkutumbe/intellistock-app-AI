import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from openai import OpenAI
from serpapi import GoogleSearch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------
# App Owner & UI Configurations
# ------------------------------
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

# ------------------------------
# Utility & Analysis Functions
# ------------------------------
def fetch_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        data = stock.history(period=period)
        return data.dropna() if not data.empty else None
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def fundamental_analysis(ticker):
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        info = stock.info
        fundamentals = {
            'Current Price': info.get('currentPrice', np.nan),
            'Market Cap (Cr)': round(info.get('marketCap', 0) / 1e7, 2),
            'P/E Ratio': round(info.get('trailingPE', np.nan), 2),
            'P/B Ratio': round(info.get('priceToBook', np.nan), 2),
            'Dividend Yield (%)': round(info.get('dividendYield', 0) * 100, 2),
            'ROE (%)': round(info.get('returnOnEquity', 0) * 100, 2),
            'Debt/Equity': round(info.get('debtToEquity', np.nan), 2),
            'EPS (TTM)': round(info.get('trailingEps', np.nan), 2),
            '52W High': info.get('fiftyTwoWeekHigh', np.nan),
            '52W Low': info.get('fiftyTwoWeekLow', np.nan),
            'Beta': round(info.get('beta', np.nan), 2)
        }
        return fundamentals
    except Exception as e:
        st.error(f"Fundamental analysis failed for {ticker}: {str(e)}")
        return None

def technical_analysis(data):
    try:
        data = data.copy()
        # Moving Averages
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], 50)
        data['SMA_200'] = ta.trend.sma_indicator(data['Close'], 200)
        # Momentum Indicators
        data['RSI'] = ta.momentum.rsi(data['Close'], 14)
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(data['Close'])
        data['BB_Upper'] = bb_indicator.bollinger_hband()
        data['BB_Lower'] = bb_indicator.bollinger_lband()
        # Additional Trend Indicator: ADX
        data['ADX'] = ta.trend.adx(high=data['High'], low=data['Low'], close=data['Close'], window=14)
        # Volume Moving Average
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        return data.dropna()
    except Exception as e:
        st.error(f"Technical analysis failed: {str(e)}")
        return None

def get_news(query, serpapi_key):
    try:
        params = {
            "q": query,
            "tbm": "nws",
            "api_key": serpapi_key,
            "hl": "en",
            "gl": "in"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get('news_results', [])
    except Exception as e:
        st.error(f"News fetch error: {str(e)}")
        return []

def analyze_news_sentiment(news_items):
    analyzer = SentimentIntensityAnalyzer()
    for item in news_items:
        snippet = item.get('snippet', '')
        sentiment = analyzer.polarity_scores(snippet)
        item['sentiment'] = sentiment  # Contains 'neg', 'neu', 'pos', 'compound'
    return news_items

def get_gpt4_recommendation(fundamental, technical, news_items, api_key):
    news_context = "\n".join(
        [f"Headline: {item['title']}\nSentiment: {item['sentiment']['compound']:.2f}\nSummary: {item['snippet']}\n"
         for item in news_items]
    ) if news_items else "No recent news available"

    prompt = f"""
**Fundamental Analysis**
{fundamental}

**Technical Analysis**
{technical}

**Market News**
{news_context}

Provide a detailed recommendation covering:
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

def explain_fundamentals(fundamentals, api_key):
    prompt = f"""
I have obtained the following fundamental metrics for a company:
{fundamentals}

Please provide a detailed explanation of what each metric means, how these metrics can be interpreted, and what they imply about the company's financial health, valuation, and performance.
"""
    try:
        with st.spinner("Generating fundamental explanation..."):
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an experienced financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"Fundamental explanation generation failed: {str(e)}"

def explain_technical(tech_data, api_key):
    summary = {
        "Latest Close": tech_data['Close'].iloc[-1],
        "50 SMA": tech_data['SMA_50'].iloc[-1],
        "200 SMA": tech_data['SMA_200'].iloc[-1],
        "RSI": tech_data['RSI'].iloc[-1],
        "MACD": tech_data['MACD'].iloc[-1],
        "ADX": tech_data['ADX'].iloc[-1],
        "Bollinger Upper": tech_data['BB_Upper'].iloc[-1],
        "Bollinger Lower": tech_data['BB_Lower'].iloc[-1]
    }
    prompt = f"""
I have performed a technical analysis on a stock and obtained the following key indicators:
{summary}

Additionally, there is a candlestick chart with the 50-day and 200-day simple moving averages.
Please explain what each of these technical indicators means and what they imply about the stock's current price trend, momentum, volatility, and overall technical outlook.
"""
    try:
        with st.spinner("Generating technical explanation..."):
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert technical analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"Technical explanation generation failed: {str(e)}"

def plot_stock_chart(data, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'],
                                 high=data['High'], low=data['Low'],
                                 close=data['Close'],
                                 name=f"{ticker} Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'],
                             line=dict(color='orange', width=2),
                             name='50 SMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'],
                             line=dict(color='green', width=2),
                             name='200 SMA'), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'],
                         marker_color='lightblue',
                         name='Volume'), row=2, col=1)
    fig.update_layout(height=700, showlegend=True,
                      xaxis_rangeslider_visible=False,
                      template="plotly_white")
    return fig

def plot_comparison_chart(stocks_data):
    fig = go.Figure()
    for name, data in stocks_data.items():
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=name))
    fig.update_layout(title="Multi-Stock Closing Price Comparison",
                      xaxis_title="Date",
                      yaxis_title="Close Price",
                      template="plotly_white")
    return fig

# ------------------------------
# Main Application
# ------------------------------
def main():
    st.set_page_config(
        page_title="IntelliStock Pro",
        page_icon="📈",
        layout="wide"
    )
    
    # Custom CSS for improved visuals
    st.markdown(
        """
    <style>
    .metric-box {
        padding: 15px; 
        background: #fff; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .footer {
        text-align: center; 
        padding: 10px; 
        margin-top: 20px; 
        font-size: 0.9em; 
        color: #666;
    }
    </style>
    """,
        unsafe_allow_html=True
    )
    
    # Sidebar: App Info & Analysis Parameters
    with st.sidebar:
        st.image(LOGO_PATH, width=120)
        st.markdown(f"### Developed by **{APP_OWNER}**")
        st.markdown(f"📧 [Email Me](mailto:{APP_OWNER_EMAIL})")
        st.markdown(f"📝 [Read my Articles]({APP_OWNER_MEDIUM})")
        st.markdown(f"🔗 [Connect on LinkedIn]({APP_OWNER_LINKEDIN})")
        st.markdown("---")
        st.subheader("Analysis Settings")
        
        analysis_mode = st.radio("Mode", ("Single Stock", "Multi-Stock Comparison"))
        openai_key = st.text_input("Enter OpenAI API key", type="password")
        serpapi_key = st.text_input("Enter SerpAPI Key", type="password")
        analysis_period = st.selectbox("Analysis Period", ['3mo', '6mo', '1y', '2y', '5y'], index=2)
        
        if analysis_mode == "Single Stock":
            selected_companies = [st.selectbox("Select Company", list(INDIAN_STOCKS.keys()))]
        else:
            selected_companies = st.multiselect("Select Companies", list(INDIAN_STOCKS.keys()),
                                                  default=["Reliance Industries", "TCS"])
        
        # When "Run Analysis" is pressed, mark the analysis as active in session state.
        if st.button("Run Analysis", key="run_analysis_btn"):
            st.session_state.run_analysis = True
            st.session_state.analysis_mode = analysis_mode
            st.session_state.selected_companies = selected_companies
            st.session_state.openai_key = openai_key
            st.session_state.serpapi_key = serpapi_key
            st.session_state.analysis_period = analysis_period

    # Only perform and display analysis if run_analysis is True in session state.
    if "run_analysis" in st.session_state and st.session_state.run_analysis:
        # For Single Stock Mode:
        if st.session_state.analysis_mode == "Single Stock":
            ticker = INDIAN_STOCKS[st.session_state.selected_companies[0]]
            data = fetch_stock_data(ticker, st.session_state.analysis_period)
            if data is None:
                st.error("Failed to fetch stock data.")
                return
            fundamentals = fundamental_analysis(ticker)
            tech_data = technical_analysis(data)
            news_items = get_news(st.session_state.selected_companies[0], st.session_state.serpapi_key)
            news_items = analyze_news_sentiment(news_items)
            
            # Store analysis objects in session state (to persist through re-runs)
            st.session_state.ticker = ticker
            st.session_state.fundamentals = fundamentals
            st.session_state.tech_data = tech_data
            st.session_state.news_items = news_items
            
            # Create tabs for the analysis sections
            tabs = st.tabs(["💰 Fundamentals", "📈 Technical", "📰 News", "🤖 AI Recommendation", "📥 Data Export"])
            
            # --- Fundamentals Tab ---
            with tabs[0]:
                st.header("Fundamental Analysis")
                if fundamentals:
                    cols = st.columns(3)
                    for idx, (key, value) in enumerate(fundamentals.items()):
                        cols[idx % 3].markdown(
                            f"""
                            <div class="metric-box">
                                <h5>{key}</h5>
                                <h3>{value}</h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    if st.session_state.openai_key:
                        with st.expander("Explain these metrics"):
                            explanation = explain_fundamentals(fundamentals, st.session_state.openai_key)
                            st.markdown(explanation)
                    else:
                        st.info("Enter your OpenAI API key to see a detailed explanation of these fundamentals.")
                else:
                    st.warning("No fundamental data available.")
            
            # --- Technical Tab ---
            with tabs[1]:
                st.header("Technical Analysis")
                st.plotly_chart(plot_stock_chart(tech_data, ticker), use_container_width=True)
                st.markdown("**Key Indicators:**")
                st.write(f"RSI: {tech_data['RSI'].iloc[-1]:.2f} | MACD Diff: {tech_data['MACD'].iloc[-1]:.2f} | ADX: {tech_data['ADX'].iloc[-1]:.2f}")
                if st.session_state.openai_key:
                    with st.expander("Explain the technical indicators"):
                        tech_explanation = explain_technical(tech_data, st.session_state.openai_key)
                        st.markdown(tech_explanation)
                else:
                    st.info("Enter your OpenAI API key to see a detailed explanation of the technical analysis.")
            
            # --- News Tab ---
            with tabs[2]:
                st.header("Market News & Sentiment")
                if news_items:
                    for item in news_items:
                        sentiment = item.get('sentiment', {}).get('compound', 0)
                        sentiment_str = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
                        with st.expander(f"{item['title']}  | Sentiment: {sentiment_str} ({sentiment:.2f})"):
                            st.write(item['snippet'])
                            st.markdown(f"[Read full article]({item.get('link', '#')})")
                else:
                    st.warning("No recent news available.")
            
            # --- AI Recommendation Tab ---
            with tabs[3]:
                st.header("Institutional Recommendation")
                if st.session_state.openai_key:
                    fund_text = "\n".join([f"{k}: {v}" for k, v in fundamentals.items()])
                    tech_text = (f"Price: ₹{tech_data['Close'].iloc[-1]:.2f}\n"
                                 f"50 SMA: ₹{tech_data['SMA_50'].iloc[-1]:.2f}\n"
                                 f"200 SMA: ₹{tech_data['SMA_200'].iloc[-1]:.2f}\n"
                                 f"RSI: {tech_data['RSI'].iloc[-1]:.2f}\n"
                                 f"MACD: {tech_data['MACD'].iloc[-1]:.2f}\n"
                                 f"ADX: {tech_data['ADX'].iloc[-1]:.2f}\n"
                                 f"Bollinger Bands: Upper ₹{tech_data['BB_Upper'].iloc[-1]:.2f} / Lower ₹{tech_data['BB_Lower'].iloc[-1]:.2f}")
                    recommendation = get_gpt4_recommendation(fund_text, tech_text, news_items, st.session_state.openai_key)
                    st.markdown(
                        f"""
                        <div style="padding:20px; background:#f8f9fa; border-radius:10px; border-left:4px solid #2c82c9;">
                            {recommendation}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Enter your OpenAI API key for professional recommendations.")
            
            # --- Data Export Tab ---
            with tabs[4]:
                st.header("Download Data")
                csv = tech_data.to_csv().encode('utf-8')
                # Wrap the download button in a spinner so the user sees progress.
                with st.spinner("Preparing your download..."):
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f'{ticker}_{st.session_state.analysis_period}.csv',
                        mime='text/csv',
                        key="download-csv"
                    )
        
        # For Multi-Stock Comparison Mode:
        else:
            st.header("Multi-Stock Comparison")
            stocks_data = {}
            fundamentals_data = {}
            for company in st.session_state.selected_companies:
                ticker = INDIAN_STOCKS[company]
                data = fetch_stock_data(ticker, st.session_state.analysis_period)
                if data is not None:
                    stocks_data[company] = data
                    fundamentals_data[company] = fundamental_analysis(ticker)
            if stocks_data:
                st.plotly_chart(plot_comparison_chart(stocks_data), use_container_width=True)
                st.subheader("Fundamental Metrics Comparison")
                for company, metrics in fundamentals_data.items():
                    with st.expander(company):
                        st.write(metrics)
            else:
                st.error("No data available for the selected stocks.")
    
    # Footer remains constant
    st.markdown("---")
    st.markdown(
        f"""
    <div class="footer">
        Developed with ❤️ by <a href="{APP_OWNER_LINKEDIN}" target="_blank">{APP_OWNER}</a> | 
        <a href="mailto:{APP_OWNER_EMAIL}">Contact</a> | 
        <a href="{APP_OWNER_MEDIUM}" target="_blank">Market Insights</a>
    </div>
    """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
