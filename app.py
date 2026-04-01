import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")

st.title("Stock Sentiment Analyzer")
st.caption("Predicts next day price direction using live news and machine learning")

ticker = st.text_input("Enter NSE stock ticker", value="RELIANCE.NS").strip().upper()

if st.button("Run Analysis"):

    with st.spinner("Fetching stock data..."):
        stock = yf.download(ticker, period="1y", interval="1d", progress=False)
        stock.columns = stock.columns.get_level_values(0)
        stock = stock[["Close"]].copy()
        stock["Target"] = (stock["Close"].shift(-1) > stock["Close"]).astype(int)

    with st.spinner("Fetching live headlines..."):
        analyzer = SentimentIntensityAnalyzer()
        query = ticker.replace(".NS", "").replace(".BO", "")
        rss_url = f"https://news.google.com/rss/search?q={query}+stock+India&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        headlines = [entry.title for entry in feed.entries[:10]]
        if not headlines:
            headlines = [f"{query} stock steady amid market activity"]

    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    avg_sentiment = sum(scores) / len(scores)

    stock["Price_Change"] = stock["Close"].pct_change()
    stock["MA_5"] = stock["Close"].rolling(window=5).mean()
    stock["MA_20"] = stock["Close"].rolling(window=20).mean()
    stock["Sentiment"] = avg_sentiment
    stock = stock.dropna()
    stock = stock[:-1]

    features = ["Price_Change", "MA_5", "MA_20", "Sentiment"]
    X = stock[features]
    y = stock["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    latest = stock[features].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest)[0]
    confidence = model.predict_proba(latest)[0][prediction] * 100
    direction = "UP" if prediction == 1 else "DOWN"

    # Results 
    st.subheader(f"Prediction for {ticker}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Direction", direction)
    col2.metric("Confidence", f"{confidence:.1f}%")
    col3.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    # Headlines 
    st.subheader("Live News Headlines")
    for h, s in zip(headlines, scores):
        color = "green" if s > 0 else "red" if s < 0 else "gray"
        st.markdown(f":{color}[{s:+.3f}] {h}")

    st.metric("Average Sentiment Score", f"{avg_sentiment:.3f}")

    # Price chart 
    st.subheader("1 Year Price Chart")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(stock.index, stock["Close"], label="Close Price", color="#1D9E75", linewidth=1.5)
    ax1.plot(stock.index, stock["MA_5"], label="5-Day MA", color="#EF9F27", linewidth=1, linestyle="--")
    ax1.plot(stock.index, stock["MA_20"], label="20-Day MA", color="#7F77DD", linewidth=1, linestyle="--")
    ax1.set_title(f"{ticker} — 1 Year Price + Prediction: {direction} ({confidence:.1f}% confidence)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (INR)")
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig1)

    # Backtest 
    st.subheader("Backtest Simulator")
    test_dates = stock.iloc[len(stock) - len(y_test):].index
    test_prices = stock.loc[test_dates, "Close"].values

    capital = 100000
    holding = False
    buy_price = 0
    portfolio = []

    for i in range(len(predictions)):
        if predictions[i] == 1 and not holding:
            buy_price = test_prices[i]
            holding = True
        elif predictions[i] == 0 and holding:
            capital += test_prices[i] - buy_price
            holding = False
        portfolio.append(capital)

    if holding:
        capital += test_prices[-1] - buy_price
        portfolio[-1] = capital

    total_return = ((capital - 100000) / 100000) * 100

    col4, col5 = st.columns(2)
    col4.metric("Final Capital", f"Rs {capital:,.2f}")
    col5.metric("Total Return", f"{total_return:+.2f}%")

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(test_dates, portfolio, color="#1D9E75", linewidth=2, label="Portfolio Value")
    ax2.axhline(y=100000, color="#EF9F27", linewidth=1, linestyle="--", label="Starting Capital")
    ax2.fill_between(test_dates, 100000, portfolio,
                     where=[p >= 100000 for p in portfolio],
                     color="#1D9E75", alpha=0.15)
    ax2.fill_between(test_dates, 100000, portfolio,
                     where=[p < 100000 for p in portfolio],
                     color="#D85A30", alpha=0.15)
    ax2.set_title(f"{ticker} Backtest — Starting Rs 1,00,000 | Return: {total_return:+.2f}%")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Portfolio Value (INR)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rs {x:,.0f}"))
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)