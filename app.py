import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# -------------------------- App Config --------------------------
st.set_page_config(page_title="🔮 Crypto Forecast", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📊 Crypto Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the next 30 days of crypto prices using ARIMAX, Prophet, or LSTM models, with optional sentiment control.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    coin = st.selectbox("🪙 Select Cryptocurrency", ['BTC', 'ETH', 'BNB'])
    model_type = st.radio("🧠 Choose Model", ['ARIMAX', 'Prophet', 'LSTM'])
    if model_type == "LSTM":
        sentiment_input = None  # not used
        st.markdown("💬 *Sentiment input is handled automatically per day for LSTM.*")
    else:
        sentiment_input = st.slider("💬 News Sentiment (Optional)", -1.0, 1.0, 0.0, 0.1)

    st.markdown("---")
    st.markdown("💡 *Sentiment score affects prediction. Default is neutral (0.0).*")

# -------------------------- Paths --------------------------
model_paths = {
    'BTC': {'arimax': 'Models/BTC/iarimax_btc_model.pkl', 'prophet': 'Models/BTC/prophet_btc_model.pkl'},
    'ETH': {'arimax': 'Models/ETH/iarimax_eth_model.pkl', 'prophet': 'Models/ETH/prophet_eth_model.pkl'},
    'BNB': {'arimax': 'Models/BNB/iarimax_bnb_model.pkl', 'prophet': 'Models/BNB/prophet_bnb_model.pkl'},
}

lstm_paths = {
    'BTC': {'model': 'Models/BTC/lstm_btc_7in7out_model.h5', 'scaler': 'Models/BTC/btc_price_scaler.pkl'},
    'ETH': {'model': 'Models/ETH/lstm_eth_7in7out_model.h5', 'scaler': 'Models/ETH/eth_price_scaler.pkl'},
    'BNB': {'model': 'Models/BNB/lstm_bnb_7in7out_model.h5', 'scaler': 'Models/BNB/bnb_price_scaler.pkl'},
}

# -------------------------- Load Data --------------------------
df = pd.read_csv(f"Dataset/{coin}_final.csv", parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
last_price_data = df['close'][-60:]
df['sentiment'] = df['sentiment'].fillna(0)

# -------------------------- LSTM --------------------------
def get_sentiment(text):
    if not text:
        return 0.0
    return analyzer.polarity_scores(text)['compound']

if model_type == "LSTM":
    st.markdown("### 🤖 LSTM 7-Day Forecast (Custom Inputs)")

    # Ask user for the last known date
    st.markdown("#### 📆 Select the Date of the Last Entry")
    user_last_date = st.date_input("Last Known Date", value=pd.to_datetime("today").date())
    last_date = pd.to_datetime(user_last_date)

    st.markdown("#### 📅 Enter Last 7 Days of Data (Oldest to Newest)")
    user_input = []
    for i in range(7):
        col1, col2 = st.columns([1, 3])
        with col1:
            close = col1.number_input(f"Day {i+1} Close Price", min_value=0.0, key=f"close_{i}")
        with col2:
            news = col2.text_input(f"Day {i+1} News Headline (Optional)", key=f"news_{i}")
        sentiment = get_sentiment(news)
        user_input.append((close, sentiment))

    if st.button("🔮 Predict 7-Day Forecast"):
        # Load model & scaler
        model = load_model(lstm_paths[coin]['model'])
        scaler = joblib.load(lstm_paths[coin]['scaler'])

        # Prepare input
        close_prices = np.array([x[0] for x in user_input]).reshape(-1, 1)
        scaled_close = scaler.transform(close_prices)
        sentiments = np.array([x[1] for x in user_input]).reshape(-1, 1)
        lstm_input = np.hstack((scaled_close, sentiments))       # shape (7, 2)
        lstm_input = np.expand_dims(lstm_input, axis=0)          # shape (1, 7, 2)

        # Predict
        future_scaled = model.predict(lstm_input)[0]
        future_pred = scaler.inverse_transform(future_scaled.reshape(-1, 1)).flatten()

        # Generate future dates
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7)

        # Output
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📈 Forecast Chart")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(future_dates, future_pred, label="🔮 Forecast (7 Days)", linestyle='--', color="#FF6F61", marker='o')
            ax.set_title(f"{coin} Forecast - LSTM")
            ax.set_xlabel("Date")
            ax.set_ylabel("Predicted Close Price")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.markdown("### 🧾 Forecast Table")
            lstm_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Close Price": np.round(future_pred, 2)
            }).set_index("Date")
            st.dataframe(lstm_df)

        # Optional message
        st.markdown(
            f"🕐 You are forecasting from **{future_dates[0].date()}** to **{future_dates[-1].date()}** based on last known date: {last_date.date()}."
        )

# -------------------------- ARIMAX --------------------------
elif model_type == "ARIMAX":
    model = joblib.load(model_paths[coin]['arimax'])
    future_sentiment = pd.DataFrame({'sentiment': [sentiment_input] * 30})
    forecast = model.forecast(steps=30, exog=future_sentiment)
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=30)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📈 Forecast Chart")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(last_price_data.index, last_price_data, label="📘 Historical", color="#4B8BBE")
        ax.plot(future_dates, forecast, label="🔮 Forecast", linestyle='--', color="#FF6F61")
        ax.set_title(f"{coin} Forecast - ARIMAX")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.markdown("### 🧾 Forecast Table")
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close Price": forecast.values
        }).set_index("Date")
        forecast_df["Predicted Close Price"] = forecast_df["Predicted Close Price"].round(2)
        st.dataframe(forecast_df)

# -------------------------- Prophet --------------------------
elif model_type == "Prophet":
    model = joblib.load(model_paths[coin]['prophet'])
    future = model.make_future_dataframe(periods=30)
    full_sentiment = list(df['sentiment']) + [sentiment_input] * 30
    future['sentiment'] = full_sentiment
    forecast = model.predict(future)
    next_30 = forecast[['ds', 'yhat']].tail(30)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📈 Forecast Chart")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index[-60:], df['close'].tail(60), label='📘 Historical', color="#4B8BBE")
        ax.plot(next_30['ds'], next_30['yhat'], label='🔮 Forecast', linestyle='--', color="#FF6F61")
        ax.set_title(f"{coin} Forecast - Prophet")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.markdown("### 🧾 Forecast Table")
        prophet_df = next_30.rename(columns={'ds': 'Date', 'yhat': 'Predicted Close Price'}).set_index("Date")
        prophet_df['Predicted Close Price'] = prophet_df['Predicted Close Price'].round(2)
        st.dataframe(prophet_df)

# -------------------------- Footer --------------------------
st.markdown("---")
st.markdown("<div style='text-align: center;'>Made by Khizer · Streamlit Forecast App</div>", unsafe_allow_html=True)
