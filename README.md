# 🔮 Crypto Price Prediction App

A Streamlit-based web application that predicts the next 7 days of **BTC**, **ETH**, and **BNB** prices using advanced forecasting models like **ARIMA**, **ARIMAX**, **Prophet**, and **LSTM**. The app also integrates news sentiment to improve prediction accuracy.

---

## 🚀 Features

- 📈 Predict next prices for **Bitcoin (BTC)**, **Ethereum (ETH)**, and **Binance Coin (BNB)**
- 🧠 Choose from 4 powerful models:
  - ARIMAX (with news sentiment)
  - Facebook Prophet
  - LSTM (Deep Learning)
- 📰 Input recent news manually to evaluate impact on price
- 📊 Visual charts and metrics (MSE, MAE, R²) for evaluation

---

## 🛠️ Installation & Setup

Follow the steps below to get the app running locally:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/crypto_price_prediction.git
cd crypto_price_prediction
```

### 2. Create a virtual environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

✅ **Important**: Ensure your environment uses compatible versions of:
- `tensorflow==2.11.0`
- `numpy` (must match the version used during model saving)
- `streamlit<2.0` if using `pandas<2.0`

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
crypto_price_prediction/
├── app.py
├── requirements.txt
├── resave_models.py
├── Models/
│   ├── BTC/
│   │   ├── arimax_btc_model.pkl
│   │   ├── prophet_btc_model.pkl
│   │   ├── lstm_btc_model.h5
│   │   └── scaler.pkl
│   └── ...
├── data/
│   └── historical_price_data.csv
├── utils/
│   └── sentiment_analysis.py
```

---

## 📌 Notes

- If you encounter `ModuleNotFoundError: No module named 'numpy._core'`, re-save your models using the same numpy version as your environment.
- ARIMAX and LSTM models use **sentiment scores** alongside price for better forecasting.
- Supports deployment to **Streamlit Cloud**.

---

