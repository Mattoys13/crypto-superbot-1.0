import requests
import time
import telebot
import pandas as pd
import numpy as np
import threading
from flask import Flask, request, jsonify, render_template_string
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import openai

# === TELEGRAM ===
API_KEY = "8330502624:AAEr5TliWy66wQm9EX02OUuGeWoslYjWeUY"
CHAT_ID = "7743162708"
bot = telebot.TeleBot(API_KEY)

# === OPENAI GPT z Railway Variables ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# === KONFIGURACJA CZĘSTOTLIWOŚCI ===
PUMP_PERCENT = 5
PUMP_INTERVAL = 5
SCAN_INTERVAL = 300        # Pump Detector co 5 minut (zmień według potrzeb)
AI_PREDICT_INTERVAL = 4*3600  # AI Prediction co 4 godziny (zmień według potrzeb)
DAILY_REPORT_HOUR = 20     # Dzienny raport o 20:00

KRAKEN_PAIRS = ["XBTUSDT", "ETHUSDT", "SOLUSDT"]
COINBASE_PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
AI_PAIRS = ["BTCUSDT", "ETHUSDT"]

signals_list = []

# ============== MODUŁ 1: Pump Detector =====================
def fetch_kraken_ticker(pair):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=1"
    response = requests.get(url).json()
    if "result" not in response:
        raise Exception(f"Błąd API Kraken dla pary {pair}: {response}")
    result = list(response['result'].values())[0]
    df = pd.DataFrame(result, columns=["time","open","high","low","close","vwap","volume","count"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

def fetch_coinbase_ticker(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=60"
    response = requests.get(url).json()
    if not isinstance(response, list):
        raise Exception(f"Błąd API Coinbase dla pary {symbol}: {response}")
    df = pd.DataFrame(response, columns=["time","low","high","open","close","volume"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df.sort_values("time")

def check_pump(df, symbol):
    recent = df.tail(PUMP_INTERVAL)
    price_change = ((recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]) * 100
    avg_volume = recent["volume"].iloc[:-1].mean()
    vol_spike = recent["volume"].iloc[-1] > avg_volume * 3

    if price_change >= PUMP_PERCENT or vol_spike:
        msg = f"🚨 *Pump Alert!* {symbol}\n" \
              f"💰 Cena: ${recent['close'].iloc[-1]:.2f}\n" \
              f"📈 Zmiana: {price_change:.2f}% (ostatnie {PUMP_INTERVAL} min)\n"
        if vol_spike:
            msg += "🔥 Wzrost wolumenu!\n"
        msg += f"🔗 [Wykres TradingView](https://www.tradingview.com/chart/?symbol={symbol.replace('-', '')})"
        send_signal("Pump Detector", msg)

def pump_detector_thread():
    print("🟢 Pump Detector start")
    bot.send_message(CHAT_ID, "✅ Pump Detector uruchomiony!")
    while True:
        try:
            for pair in KRAKEN_PAIRS:
                try:
                    kraken_df = fetch_kraken_ticker(pair)
                    check_pump(kraken_df, f"Kraken: {pair}")
                except Exception as e:
                    print(f"❌ Błąd Kraken ({pair}): {e}")

            for pair in COINBASE_PAIRS:
                try:
                    coinbase_df = fetch_coinbase_ticker(pair)
                    check_pump(coinbase_df, f"Coinbase: {pair}")
                except Exception as e:
                    print(f"❌ Błąd Coinbase ({pair}): {e}")

        except Exception as e:
            print(f"❌ Błąd główny PUMP: {e}")
        time.sleep(SCAN_INTERVAL)

# ============== MODUŁ 2: AI/ML Predykcja + GPT ================
def fetch_binance_ohlc(symbol="BTCUSDT", interval="1h", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=["time","open","high","low","close","volume","c1","c2","c3","c4","c5","c6"])
    df["close"] = df["close"].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def predict_price(df, steps_ahead=1):
    X = np.arange(len(df)).reshape(-1,1)
    y = df["close"].values
    model = LinearRegression()
    model.fit(X, y)
    next_time = np.array([[len(df) + steps_ahead - 1]])
    prediction = model.predict(next_time)
    return prediction[0]

def gpt_comment(coin, current, pred):
    if not openai.api_key:
        return "(Brak klucza OPENAI_API_KEY)"
    prompt = (
        f"Predykcja AI dla {coin}:\n"
        f"Aktualna cena: ${current:.2f}, prognoza za 1h: ${pred:.2f}. "
        f"Opisz sentyment/trend rynku i możliwe powody zmiany ceny w 2-3 zdaniach (krótko, rzeczowo, po polsku):"
    )
    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Błąd GPT: {e})"

def ai_prediction_thread():
    print("🟢 AI/ML Predykcja start")
    while True:
        try:
            for coin in AI_PAIRS:
                df = fetch_binance_ohlc(coin)
                pred = predict_price(df)
                current = df["close"].iloc[-1]
                dt = datetime.now().strftime("%Y-%m-%d %H:%M")
                komentarz = gpt_comment(coin, current, pred)
                msg = f"🤖 *AI Predykcja* {coin}\n\nCzas: {dt}\nAktualna cena: ${current:.2f}\nPredykcja za 1h: ${pred:.2f}\n\n💡 _{komentarz}_"
                send_signal("AI Prediction", msg)
        except Exception as e:
            print("❌ Błąd AI bota:", e)
        time.sleep(AI_PREDICT_INTERVAL)

# ============== MODUŁ 3: Sentyment ================
def fetch_fear_greed():
    try:
        url = "https://api.alternative.me/fng/"
        data = requests.get(url, timeout=10).json()
        index = data["data"][0]["value"]
        classification = data["data"][0]["value_classification"]
        return index, classification
    except:
        return "?", "Brak danych"

def gpt_sentyment(fear_value, fear_class):
    if not openai.api_key:
        return "(Brak klucza OPENAI_API_KEY)"
    prompt = (
        f"Indeks Fear & Greed wynosi {fear_value} ({fear_class}). "
        f"Oceń krótko nastroje rynku kryptowalut i czy to sprzyja kupnie/sprzedaży. Odpowiedz po polsku."
    )
    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Błąd GPT: {e})"

# ============== MODUŁ 4: Raport dzienny ================
def daily_report_thread():
    print("🟢 Dzienny raport start")
    last_report = None
    while True:
        now = datetime.now()
        if last_report != now.date() and now.hour == DAILY_REPORT_HOUR:
            try:
                # BTC/ETH
                df_btc = fetch_binance_ohlc("BTCUSDT")
                df_eth = fetch_binance_ohlc("ETHUSDT")
                btc = df_btc["close"].iloc[-1]
                eth = df_eth["close"].iloc[-1]

                # Fear&Greed
                fg_val, fg_class = fetch_fear_greed()
                fg_gpt = gpt_sentyment(fg_val, fg_class)

                # AI prediction & GPT komentarz
                pred_btc = predict_price(df_btc)
                ai_kom = gpt_comment("BTCUSDT", btc, pred_btc)

                # Największe gainery/losery
                tickers = requests.get("https://api.binance.com/api/v3/ticker/24hr").json()
                usdt = [x for x in tickers if x["symbol"].endswith("USDT")]
                g = sorted(usdt, key=lambda x: float(x["priceChangePercent"]), reverse=True)
                gainers = "\n".join([f"{x['symbol']}: {x['priceChangePercent']}%" for x in g[:3]])
                losers = "\n".join([f"{x['symbol']}: {x['priceChangePercent']}%" for x in g[-3:]])

                dt = now.strftime('%Y-%m-%d')
                msg = f"""📅 *Raport dzienny – {dt}*

*BTC:* ${btc:.2f}
*ETH:* ${eth:.2f}

😨 *Fear&Greed:* {fg_val} ({fg_class})
💬 {fg_gpt}

🤖 *AI Predykcja BTC*: ${pred_btc:.2f}
💡 {ai_kom}

🚀 *TOP Gainers:*
{gainers}

🔻 *TOP Losers:*
{losers}
"""
                send_signal("Dzienny raport", msg)
                last_report = now.date()
            except Exception as e:
                print("❌ Błąd raportu dziennego:", e)
        time.sleep(60)

# ============== MODUŁ 5: TradingView Webhook + Dashboard ===========
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Crypto SuperBot Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background: #121212; color: #fff; text-align: center; }
        h1 { color: #00e676; }
        table { margin: auto; border-collapse: collapse; width: 90%; background: #1e1e1e; }
        th, td { border: 1px solid #333; padding: 10px; }
        th { background: #00e676; color: black; }
        tr:nth-child(even) { background: #2a2a2a; }
        a { color: #00e676; text-decoration: none; }
    </style>
</head>
<body>
    <h1>🤖 Crypto SuperBot Dashboard</h1>
    <table>
        <tr>
            <th>Czas</th>
            <th>Moduł</th>
            <th>Wiadomość</th>
        </tr>
        {% for s in signals %}
        <tr>
            <td>{{ s.time }}</td>
            <td>{{ s.title }}</td>
            <td>{{ s.message|safe }}</td>
        </tr>
        {% endfor %}
    </table>
    <p>🔄 Auto-refresh co 60s</p>
    <script>setTimeout(() => location.reload(), 60000);</script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE, signals=signals_list[-50:])

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        print("📩 Odebrano sygnał z TradingView:", data)
        symbol = data.get("symbol", "N/A")
        action = data.get("action", "N/A")
        price = data.get("price", "N/A")
        rsi = data.get("rsi", "N/A")
        message = (
            f"📢 *Sygnał z TradingView*\n\n"
            f"💎 Symbol: {symbol}\n"
            f"📈 Akcja: {action}\n"
            f"💰 Cena: {price}\n"
            f"📊 RSI: {rsi}\n"
            f"🔗 [Wykres TradingView](https://www.tradingview.com/chart/?symbol={symbol})"
        )
        send_signal("TradingView", message)
        return jsonify({"status": "ok", "message": "Alert wysłany do Telegrama"}), 200
    except Exception as e:
        print("❌ Błąd webhooka:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

def run_flask():
    print("🟢 Flask Dashboard/Webhook start")
    app.run(host="0.0.0.0", port=8080)

# ============== FUNKCJA WYSYŁANIA ALERTÓW ==============
def send_signal(title, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signals_list.append({"time": timestamp, "title": title, "message": message})
    if len(signals_list) > 100:
        signals_list.pop(0)
    bot.send_message(CHAT_ID, f"🔔 *{title}*\n\n{message}", parse_mode="Markdown")

# ============== START WĄTKÓW ==============
if __name__ == "__main__":
    threading.Thread(target=pump_detector_thread, daemon=True).start()
    threading.Thread(target=ai_prediction_thread, daemon=True).start()
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=daily_report_thread, daemon=True).start()
    print("🚀 Crypto SuperBot Ultimate uruchomiony!")
    while True:
        time.sleep(60)

