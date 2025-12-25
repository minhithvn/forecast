import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import time
import hashlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Phân Tích Cổ Phiếu", layout="wide", initial_sidebar_state="expanded")

# CSS Enhancement
st.markdown("""
<style>
.big-font {font-size:20px !important; font-weight:bold;}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white;}
.buy-signal {background: linear-gradient(135deg, #00c853 0%, #00e676 100%); color: white; padding: 15px; border-radius: 10px; font-weight: bold; text-align: center; box-shadow: 0 4px 15px rgba(0,200,83,0.4);}
.sell-signal {background: linear-gradient(135deg, #d32f2f 0%, #f44336 100%); color: white; padding: 15px; border-radius: 10px; font-weight: bold; text-align: center; box-shadow: 0 4px 15px rgba(211,47,47,0.4);}
.hold-signal {background: linear-gradient(135deg, #ff9800 0%, #ffc107 100%); color: white; padding: 15px; border-radius: 10px; font-weight: bold; text-align: center; box-shadow: 0 4px 15px rgba(255,152,0,0.4);}
.prediction-box {background: linear-gradient(135deg, #1e88e5 0%, #42a5f5 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;}
.footer-stats {
    position: fixed;
    bottom: 10px;
    right: 10px;
    background: rgba(0,0,0,0.9);
    color: white;
    padding: 15px;
    border-radius: 10px;
    font-size: 12px;
    z-index: 999;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.indicator-positive {color: #00c853; font-weight: bold;}
.indicator-negative {color: #d32f2f; font-weight: bold;}
.indicator-neutral {color: #ff9800; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# 500+ mã cổ phiếu Việt Nam (mở rộng)
VN_STOCKS = {
    'Ngân hàng': ['ACB', 'BAB', 'BID', 'CTG', 'EIB', 'HDB', 'KLB', 'LPB', 'MBB', 'MSB', 'NAB', 'NVB', 'OCB', 'PGB',
                  'SCB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VAB', 'VBB', 'VCB', 'VIB', 'VPB'],
    'Chứng khoán': ['AGR', 'APS', 'ART', 'BSI', 'BVS', 'CTS', 'EVS', 'FTS', 'HCM', 'IVS', 'MBS', 'ORS', 'PSI', 'SHS',
                    'SSI', 'TVB', 'VCI', 'VDS', 'VIG', 'VIX', 'VND'],
    'Bất động sản': ['ASM', 'BCI', 'BCM', 'CEO', 'CIG', 'DIG', 'DRH', 'DXG', 'DXS', 'FLC', 'HAG', 'HDC', 'HDG', 'HQC',
                     'IDC', 'ITA', 'KBC', 'KDH', 'LDG', 'LHG', 'NLG', 'NTL', 'NVL', 'PDR', 'PPI', 'QCG', 'SCR', 'SIP',
                     'SJS', 'SZC', 'TDC', 'TDH', 'VHM', 'VIC', 'VPI', 'VRE'],
    'Xây dựng': ['C4G', 'CC1', 'CII', 'CTD', 'CTI', 'CVT', 'DPG', 'FCN', 'HBC', 'HT1', 'HTN', 'LCG', 'PC1', 'PCC',
                 'PXI', 'REE', 'SC5', 'SCG', 'SZL', 'TCO', 'THG', 'VC3', 'VCG', 'VE1', 'VE3', 'VE4', 'VE8', 'VE9'],
    'Thép': ['DTL', 'DXV', 'GVR', 'HMC', 'HPG', 'HSG', 'KSB', 'NKG', 'POM', 'SMC', 'TLH', 'TVN', 'VGS'],
    'Dầu khí': ['ASP', 'BSR', 'CNG', 'DVP', 'GAS', 'HFC', 'OIL', 'PGC', 'PGD', 'PGI', 'PGS', 'PLC', 'PLX', 'POS', 'POW',
                'PSH', 'PVB', 'PVC', 'PVD', 'PVG', 'PVS', 'PVT', 'PXS', 'PXT'],
    'Điện lực': ['GEG', 'GEX', 'HND', 'NT2', 'POW', 'QTP', 'REE', 'SBA', 'TBC', 'VSH'],
    'Bán lẻ': ['ABA', 'ABT', 'AST', 'BBC', 'DGW', 'FRT', 'MWG', 'PAN', 'PET', 'PNJ', 'SAM', 'SFI', 'VGC', 'VHC'],
    'Thực phẩm': ['ABT', 'ACL', 'AGF', 'BAF', 'BHS', 'CAN', 'HNG', 'KDC', 'LAF', 'MCH', 'MML', 'MSN', 'NHS', 'ORN',
                  'QNS', 'SAB', 'SAV', 'SBT', 'SGT', 'TAC', 'TLG', 'TS4', 'VHC', 'VIF', 'VNM', 'VSN'],
    'Dược phẩm': ['ADP', 'AGP', 'AMV', 'DBD', 'DCL', 'DHG', 'DHT', 'DMC', 'DP1', 'DP2', 'DP3', 'DVN', 'IMP', 'PME',
                  'PPP', 'TRA', 'VMD'],
    'Công nghệ': ['BMI', 'CMG', 'CMT', 'CMX', 'CNT', 'CTR', 'DAG', 'DGT', 'ELC', 'FPT', 'ICT', 'ITD', 'MFS', 'SAM',
                  'SGD', 'SGN', 'SGR', 'ST8', 'SVT', 'TDG', 'VGI', 'VNR', 'VNT'],
    'Vận tải': ['ACV', 'ATA', 'CAV', 'CLW', 'GMD', 'GSP', 'HAH', 'HTV', 'HVN', 'IDV', 'PAN', 'PJT', 'PVT', 'SCS', 'STG',
                'TCL', 'TMS', 'VFC', 'VJC', 'VOS', 'VSC', 'VTO'],
    'Vật liệu xây dựng': ['BCC', 'BMP', 'BTS', 'C32', 'DHA', 'DPR', 'DCM', 'HOM', 'HT1', 'KSB', 'NNC', 'PAN', 'PC1',
                          'SCG', 'TLH', 'VCM', 'VCS', 'VGC'],
    'Hóa chất': ['AAA', 'BFC', 'BTC', 'CSV', 'DAG', 'DGC', 'DPM', 'DRC', 'GVR', 'LAS', 'NCS', 'PAC', 'PLC', 'PMB',
                 'PTB', 'SFG', 'TNC', 'VFG'],
    'Cao su': ['BRC', 'CSM', 'DPR', 'DRC', 'GVR', 'HRC', 'PHR', 'TNC', 'TRC', 'VHG'],
    'Thủy sản': ['AAM', 'ABT', 'ACL', 'AGF', 'ANV', 'BLF', 'CMX', 'FMC', 'IDI', 'MPC', 'SJ1', 'TS4', 'VHC'],
    'Điện tử': ['CMG', 'DGW', 'FPT', 'ITD', 'SAM', 'ST8'],
    'Du lịch': ['CDO', 'DAH', 'DLG', 'HOT', 'OCH', 'PDN', 'PGT', 'PNG', 'SHN', 'TCH', 'VNG'],
    'Dệt may': ['ACL', 'AGM', 'GIL', 'HMC', 'MSH', 'NPS', 'PHT', 'STK', 'TNG', 'VGT'],
    'Giấy': ['AAA', 'BMP', 'DHC', 'GDT', 'MCV', 'SFC', 'TPC', 'VPG'],
    'Khoáng sản': ['BMW', 'BXH', 'CLC', 'DHM', 'DIC', 'DQC', 'KSH', 'MBG', 'NBC', 'PLC', 'THT', 'TMX'],
}

ALL_STOCKS = sorted(list(set([s for stocks in VN_STOCKS.values() for s in stocks])))

# Session state initialization
if 'visit_count' not in st.session_state:
    st.session_state.visit_count = 0
if 'online_users' not in st.session_state:
    st.session_state.online_users = set()
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()
    st.session_state.visit_count += 1

st.session_state.online_users.add(st.session_state.session_id)


# Cache functions
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(symbol, period='1y'):
    try:
        ticker = yf.Ticker(f"{symbol}.VN")
        df = ticker.history(period=period)
        info = ticker.info
        if df.empty:
            return None, None
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        return df, info
    except:
        return None, None


# Advanced Technical Indicators
def calculate_advanced_indicators(df):
    if df is None or df.empty:
        return df

    # Basic Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA100'] = df['close'].rolling(window=100).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()

    # EMA
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # Williams %R
    df['Williams_R'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))

    # ADX (Average Directional Index)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(window=14).mean()
    df['DI_plus'] = plus_di
    df['DI_minus'] = minus_di

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Volume indicators
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['volume'] / df['Volume_MA']

    # ATR (Average True Range)
    df['ATR'] = atr

    # CCI (Commodity Channel Index)
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

    # MFI (Money Flow Index)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # Rate of Change (ROC)
    df['ROC'] = ((df['close'] - df['close'].shift(12)) / df['close'].shift(12)) * 100

    return df


# Enhanced Signal Generation
def generate_advanced_signal(df, info=None):
    if df is None or df.empty or len(df) < 50:
        return "N/A", 50, "Không đủ dữ liệu", "N/A", {}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 50
    reasons = []
    details = {}

    # Trend Analysis (30 points)
    trend_score = 0
    if pd.notna(latest['MA5']) and pd.notna(latest['MA20']) and pd.notna(latest['MA50']):
        if latest['MA5'] > latest['MA20'] > latest['MA50']:
            trend_score += 15
            reasons.append("✅ Xu hướng tăng mạnh (MA5>MA20>MA50)")
        elif latest['MA5'] > latest['MA20']:
            trend_score += 10
            reasons.append("📈 Xu hướng tăng ngắn hạn")
        elif latest['MA5'] < latest['MA20'] < latest['MA50']:
            trend_score -= 10
            reasons.append("⚠️ Xu hướng giảm")

        if latest['close'] > latest['MA20']:
            trend_score += 5
        if latest['close'] > latest['MA50']:
            trend_score += 5
        if latest['close'] > latest['MA200']:
            trend_score += 5

    score += trend_score
    details['trend_score'] = trend_score

    # RSI Analysis (20 points)
    rsi_score = 0
    if pd.notna(latest['RSI']):
        rsi = latest['RSI']
        if 45 <= rsi <= 55:
            rsi_score += 15
            reasons.append(f"✅ RSI trung lập ({rsi:.1f})")
        elif 30 <= rsi < 45:
            rsi_score += 10
            reasons.append(f"💰 RSI thấp ({rsi:.1f} - cơ hội mua)")
        elif rsi < 30:
            rsi_score += 8
            reasons.append(f"💰 RSI quá bán ({rsi:.1f} - tín hiệu mua mạnh)")
        elif 55 < rsi <= 70:
            rsi_score += 5
            reasons.append(f"📈 RSI tích cực ({rsi:.1f})")
        elif rsi > 70:
            rsi_score -= 10
            reasons.append(f"⚠️ RSI quá mua ({rsi:.1f})")

    score += rsi_score
    details['rsi_score'] = rsi_score

    # MACD Analysis (15 points)
    macd_score = 0
    if pd.notna(latest['MACD']) and pd.notna(latest['MACD_signal']):
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            macd_score += 15
            reasons.append("✅ MACD Golden Cross")
        elif latest['MACD'] > latest['MACD_signal']:
            macd_score += 10
            reasons.append("✅ MACD tích cực")
        elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            macd_score -= 15
            reasons.append("⚠️ MACD Death Cross")
        else:
            macd_score -= 5

    score += macd_score
    details['macd_score'] = macd_score

    # Stochastic & Williams %R (10 points)
    momentum_score = 0
    if pd.notna(latest['Stoch_K']):
        if latest['Stoch_K'] < 20:
            momentum_score += 5
            reasons.append(f"💰 Stochastic quá bán ({latest['Stoch_K']:.1f})")
        elif latest['Stoch_K'] > 80:
            momentum_score -= 5
            reasons.append(f"⚠️ Stochastic quá mua ({latest['Stoch_K']:.1f})")

    if pd.notna(latest['Williams_R']):
        if latest['Williams_R'] < -80:
            momentum_score += 5
        elif latest['Williams_R'] > -20:
            momentum_score -= 5

    score += momentum_score
    details['momentum_score'] = momentum_score

    # ADX Trend Strength (10 points)
    adx_score = 0
    if pd.notna(latest['ADX']):
        if latest['ADX'] > 25:
            adx_score += 10
            reasons.append(f"✅ Xu hướng mạnh (ADX: {latest['ADX']:.1f})")
        elif latest['ADX'] < 20:
            adx_score -= 5
            reasons.append(f"⚠️ Xu hướng yếu (ADX: {latest['ADX']:.1f})")

    score += adx_score
    details['adx_score'] = adx_score

    # Volume Analysis (10 points)
    volume_score = 0
    if pd.notna(latest['Volume_ratio']):
        if latest['Volume_ratio'] > 2:
            volume_score += 10
            reasons.append("✅ Khối lượng tăng đột biến")
        elif latest['Volume_ratio'] > 1.5:
            volume_score += 5
            reasons.append("✅ Khối lượng tăng cao")
        elif latest['Volume_ratio'] < 0.5:
            volume_score -= 5
            reasons.append("⚠️ Khối lượng yếu")

    score += volume_score
    details['volume_score'] = volume_score

    # Bollinger Bands (5 points)
    bb_score = 0
    if pd.notna(latest['BB_upper']) and pd.notna(latest['BB_lower']):
        if latest['close'] < latest['BB_lower']:
            bb_score += 5
            reasons.append("💰 Giá dưới BB lower")
        elif latest['close'] > latest['BB_upper']:
            bb_score -= 5
            reasons.append("⚠️ Giá trên BB upper")

    score += bb_score
    details['bb_score'] = bb_score

    score = max(0, min(100, score))

    # Determine signal
    if score >= 80:
        signal = "MUA MẠNH"
        term = "Ngắn & Dài hạn"
    elif score >= 70:
        signal = "MUA"
        term = "Ngắn hạn"
    elif score >= 55:
        signal = "MUA (thận trọng)"
        term = "Ngắn hạn"
    elif score >= 45:
        signal = "GIỮ"
        term = "Theo dõi"
    elif score >= 30:
        signal = "BÁN (thận trọng)"
        term = "Cân nhắc bán"
    else:
        signal = "BÁN"
        term = "Nên bán"

    reason_text = "\n".join(reasons[:7])
    return signal, score, reason_text, term, details


# Simple LSTM-inspired prediction (using statistical methods)
def predict_future_price(df, days=7):
    if df is None or df.empty or len(df) < 30:
        return None

    try:
        # Simple trend-based prediction
        recent_data = df['close'].tail(30).values

        # Calculate trend
        x = np.arange(len(recent_data))
        z = np.polyfit(x, recent_data, 2)
        p = np.poly1d(z)

        # Predict future
        future_x = np.arange(len(recent_data), len(recent_data) + days)
        predictions = p(future_x)

        # Add some volatility based on historical
        volatility = df['close'].tail(30).std()
        predictions = predictions + np.random.normal(0, volatility * 0.3, days)

        return predictions
    except:
        return None


# Random Forest for trend classification
def predict_trend_ml(df):
    if df is None or df.empty or len(df) < 100:
        return "N/A", 0.5

    try:
        # Prepare features
        features = ['RSI', 'MACD', 'Stoch_K', 'ADX', 'Volume_ratio']
        df_clean = df[features].dropna()

        if len(df_clean) < 50:
            return "N/A", 0.5

        # Create target (1 if price goes up next day, 0 otherwise)
        df_clean['target'] = (df.loc[df_clean.index, 'close'].shift(-1) > df.loc[df_clean.index, 'close']).astype(int)
        df_clean = df_clean.dropna()

        if len(df_clean) < 50:
            return "N/A", 0.5

        # Split data
        split = int(len(df_clean) * 0.8)
        X_train = df_clean[features].iloc[:split]
        y_train = df_clean['target'].iloc[:split]
        X_test = df_clean[features].iloc[split:]

        # Train model
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        # Predict
        latest_features = df_clean[features].iloc[-1:].values
        prediction = rf.predict(latest_features)[0]
        probability = rf.predict_proba(latest_features)[0]

        trend = "TĂNG" if prediction == 1 else "GIẢM"
        confidence = max(probability)

        return trend, confidence
    except:
        return "N/A", 0.5


# Backtesting
def simple_backtest(df, initial_capital=100000000):
    if df is None or df.empty or len(df) < 100:
        return None

    capital = initial_capital
    shares = 0
    trades = []

    for i in range(50, len(df)):
        current_data = df.iloc[:i + 1]
        current_data = calculate_advanced_indicators(current_data)
        signal, score, _, _, _ = generate_advanced_signal(current_data)

        current_price = df.iloc[i]['close']

        # Buy signal
        if signal.startswith("MUA") and shares == 0 and capital > current_price:
            shares = int(capital * 0.9 / current_price)
            cost = shares * current_price
            capital -= cost
            trades.append({
                'date': df.index[i],
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'value': cost
            })

        # Sell signal
        elif signal.startswith("BÁN") and shares > 0:
            revenue = shares * current_price
            capital += revenue
            trades.append({
                'date': df.index[i],
                'action': 'SELL',
                'price': current_price,
                'shares': shares,
                'value': revenue
            })
            shares = 0

    # Close position
    if shares > 0:
        final_price = df.iloc[-1]['close']
        capital += shares * final_price
        shares = 0

    final_value = capital
    roi = ((final_value - initial_capital) / initial_capital) * 100

    return {
        'trades': trades,
        'final_value': final_value,
        'roi': roi,
        'num_trades': len(trades)
    }


# Plotting functions
def plot_advanced_chart(df, symbol, predictions=None):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00c853',
        decreasing_line_color='#d32f2f'
    ))

    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5', line=dict(color='#2196F3', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='#FF9800', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='#9C27B0', width=2)))

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                             line=dict(color='rgba(250,128,114,0.3)', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                             line=dict(color='rgba(250,128,114,0.3)', dash='dash'),
                             fill='tonexty', fillcolor='rgba(250,128,114,0.1)'))

    # Predictions
    if predictions is not None and len(predictions) > 0:
        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=len(predictions))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='Dự đoán',
                                 line=dict(color='#FFC107', width=3, dash='dot'),
                                 mode='lines+markers'))

    fig.update_layout(
        title=f'{symbol} - Biểu Đồ Kỹ Thuật Nâng Cao',
        xaxis_title='Ngày',
        yaxis_title='Giá (VND)',
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    return fig


def plot_multi_indicators(df):
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('RSI', 'MACD', 'Stochastic', 'ADX'),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#2196F3')), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#2196F3')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='#FF9800')), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram',
                         marker_color=df['MACD_hist'].apply(lambda x: '#00c853' if x > 0 else '#d32f2f')), row=2, col=1)

    # Stochastic
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name='%K', line=dict(color='#2196F3')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name='%D', line=dict(color='#FF9800')), row=3, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)

    # ADX
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX', line=dict(color='#9C27B0')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DI_plus'], name='+DI', line=dict(color='#00c853')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DI_minus'], name='-DI', line=dict(color='#d32f2f')), row=4, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="white", row=4, col=1)

    fig.update_layout(height=900, template='plotly_dark', showlegend=True)
    return fig


# Main App Title
st.title(" DỰ ĐOÁN CỔ PHIẾU ")
st.markdown(f"### {len(ALL_STOCKS)}+ mã cổ phiếu | 15+ chỉ báo | Dự đoán bằng AI/ML Prediction")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ ĐIỀU KHIỂN")

    mode = st.radio("Chế độ", [
        "🔍 Phân tích chi tiết",
        "🚀 Quét nhanh",
        "📊 So sánh",
        "🤖 AI Prediction/Dự đoán bằng AI",
        "📈 Backtesting/Chạy thử chiến thuật"
    ])

    if mode == "🔍 Phân tích chi tiết":
        search_term = st.text_input("🔎 Tìm mã nhanh", "")
        if search_term:
            filtered_stocks = [s for s in ALL_STOCKS if search_term.upper() in s]
        else:
            sector = st.selectbox("Chọn ngành", ['Tất cả'] + list(VN_STOCKS.keys()))
            filtered_stocks = ALL_STOCKS if sector == 'Tất cả' else VN_STOCKS[sector]

        symbol = st.selectbox("Chọn mã", filtered_stocks)
        period = st.selectbox("Thời gian", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

        st.markdown("---")
        show_prediction = st.checkbox("🔮 Dự đoán giá", value=True)
        if show_prediction:
            pred_days = st.slider("Số ngày dự đoán", 3, 30, 7)

        show_ml_trend = st.checkbox("🤖 ML Trend Analysis", value=True)

    elif mode == "🚀 Quét nhanh":
        scan_mode = st.radio("Quét", ["Theo ngành", "Toàn bộ", "Top 100"])
        if scan_mode == "Theo ngành":
            sector = st.selectbox("Chọn ngành quét", list(VN_STOCKS.keys()))
            stocks_to_scan = VN_STOCKS[sector]
        elif scan_mode == "Top 100":
            stocks_to_scan = ALL_STOCKS[:100]
        else:
            stocks_to_scan = ALL_STOCKS

        min_score = st.slider("Điểm tối thiểu", 50, 95, 70)
        max_results = st.slider("Số kết quả tối đa", 10, 100, 20)

    elif mode == "📊 So sánh":
        compare_symbols = st.multiselect("Chọn mã (tối đa 5)", ALL_STOCKS, max_selections=5)
        period = st.selectbox("Thời gian", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

    elif mode == "🤖 AI Prediction":
        sector = st.selectbox("Chọn ngành", list(VN_STOCKS.keys()))
        symbol = st.selectbox("Chọn mã", VN_STOCKS[sector])
        pred_days = st.slider("Số ngày dự đoán", 7, 30, 14)

    else:  # Backtesting
        sector = st.selectbox("Chọn ngành", list(VN_STOCKS.keys()))
        symbol = st.selectbox("Chọn mã", VN_STOCKS[sector])
        initial_capital = st.number_input("Vốn ban đầu (VND)", min_value=10000000, value=100000000, step=10000000)

    st.markdown("---")
    st.info(
        "💡 **Điểm số:**\n- ≥80: MUA MẠNH\n- 70-79: MUA\n- 55-69: MUA thận trọng\n- 45-54: GIỮ\n- 30-44: BÁN thận trọng\n- <30: BÁN")

    st.markdown("---")
    st.markdown(f"**📊 Tổng mã:** {len(ALL_STOCKS)}")
    st.markdown(f"**🏢 Ngành:** {len(VN_STOCKS)}")

# MAIN CONTENT
if mode == "🔍 Phân tích chi tiết":
    st.header(f"📊 PHÂN TÍCH CHI TIẾT: {symbol}")

    with st.spinner(f"Đang tải dữ liệu {symbol}..."):
        df, info = get_stock_data(symbol, period=period)

    if df is not None and not df.empty:
        df = calculate_advanced_indicators(df)
        signal, score, reason, term, details = generate_advanced_signal(df, info)
        latest = df.iloc[-1]

        # Price prediction
        predictions = None
        if show_prediction:
            predictions = predict_future_price(df, pred_days)

        # ML trend prediction
        ml_trend = "N/A"
        ml_confidence = 0
        if show_ml_trend:
            ml_trend, ml_confidence = predict_trend_ml(df)

        # Metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        price_change = ((latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100) if len(df) > 1 else 0

        col1.metric("💰 Giá hiện tại", f"{latest['close']:,.0f}", f"{price_change:+.2f}%")
        col2.metric("📊 Khối lượng", f"{latest['volume'] / 1000:.0f}K")
        col3.metric("⭐ Điểm AI", f"{score}/100",
                    f"{score - 50:+.0f}" if score >= 50 else f"{score - 50:.0f}")
        col4.metric("🎯 Khuyến nghị", term)
        col5.metric("📈 RSI", f"{latest['RSI']:.1f}")
        col6.metric("💪 ADX", f"{latest['ADX']:.1f}")

        # Signal display
        if signal.startswith("MUA"):
            st.markdown(f'<div class="buy-signal">🟢 {signal} - Điểm: {score}/100</div>', unsafe_allow_html=True)
        elif signal.startswith("BÁN"):
            st.markdown(f'<div class="sell-signal">🔴 {signal} - Điểm: {score}/100</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="hold-signal">🟡 {signal} - Điểm: {score}/100</div>', unsafe_allow_html=True)

        # ML Prediction box
        if ml_trend != "N/A":
            trend_color = "#00c853" if ml_trend == "TĂNG" else "#d32f2f"
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🤖 Dự đoán bằng AI/Machine Learning</h3>
                <p style="font-size:24px; margin:10px 0;">
                    Xu hướng: <span style="color:{trend_color}; font-weight:bold;">{ml_trend}</span>
                    | Độ tin cậy: <b>{ml_confidence * 100:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Price prediction
        if predictions is not None and len(predictions) > 0:
            pred_change = ((predictions[-1] - latest['close']) / latest['close']) * 100
            pred_color = "#00c853" if pred_change > 0 else "#d32f2f"
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🔮 Dự đoán giá {pred_days} ngày tới</h3>
                <p style="font-size:20px; margin:10px 0;">
                    Giá dự kiến: <b>{predictions[-1]:,.0f} VND</b>
                    | Thay đổi: <span style="color:{pred_color}; font-weight:bold;">{pred_change:+.2f}%</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Biểu đồ giá",
            "📊 Chỉ báo kỹ thuật",
            "📝 Phân tích chi tiết",
            "💼 Fundamental",
            "🎯 Mức giá quan trọng"
        ])

        with tab1:
            st.plotly_chart(plot_advanced_chart(df, symbol, predictions), use_container_width=True)

            # Volume chart
            fig_vol = go.Figure()
            colors = ['#00c853' if df['close'].iloc[i] >= df['open'].iloc[i] else '#d32f2f'
                      for i in range(len(df))]
            fig_vol.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors))
            fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volume_MA'], name='Volume MA',
                                         line=dict(color='#FFC107', width=2)))
            fig_vol.update_layout(title='Khối lượng giao dịch', height=250, template='plotly_dark')
            st.plotly_chart(fig_vol, use_container_width=True)

        with tab2:
            st.plotly_chart(plot_multi_indicators(df), use_container_width=True)

            # Additional indicators table
            st.subheader("📋 Bảng chỉ báo chi tiết")
            col1, col2 = st.columns(2)

            with col1:
                indicators_df = pd.DataFrame({
                    'Chỉ báo': ['RSI', 'MACD', 'Stochastic %K', 'Stochastic %D', 'Williams %R',
                                'ADX', 'CCI', 'MFI', 'ROC'],
                    'Giá trị': [
                        f"{latest['RSI']:.2f}",
                        f"{latest['MACD']:.2f}",
                        f"{latest['Stoch_K']:.2f}",
                        f"{latest['Stoch_D']:.2f}",
                        f"{latest['Williams_R']:.2f}",
                        f"{latest['ADX']:.2f}",
                        f"{latest['CCI']:.2f}",
                        f"{latest['MFI']:.2f}",
                        f"{latest['ROC']:.2f}%"
                    ]
                })
                st.dataframe(indicators_df, hide_index=True, use_container_width=True)

            with col2:
                ma_df = pd.DataFrame({
                    'Moving Average': ['MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200'],
                    'Giá trị': [
                        f"{latest['MA5']:,.0f}" if pd.notna(latest['MA5']) else "N/A",
                        f"{latest['MA10']:,.0f}" if pd.notna(latest['MA10']) else "N/A",
                        f"{latest['MA20']:,.0f}" if pd.notna(latest['MA20']) else "N/A",
                        f"{latest['MA50']:,.0f}" if pd.notna(latest['MA50']) else "N/A",
                        f"{latest['MA100']:,.0f}" if pd.notna(latest['MA100']) else "N/A",
                        f"{latest['MA200']:,.0f}" if pd.notna(latest['MA200']) else "N/A"
                    ],
                    'So với giá': [
                        "🟢 Trên" if latest['close'] > latest['MA5'] else "🔴 Dưới" if pd.notna(latest['MA5']) else "N/A",
                        "🟢 Trên" if latest['close'] > latest['MA10'] else "🔴 Dưới" if pd.notna(
                            latest['MA10']) else "N/A",
                        "🟢 Trên" if latest['close'] > latest['MA20'] else "🔴 Dưới" if pd.notna(
                            latest['MA20']) else "N/A",
                        "🟢 Trên" if latest['close'] > latest['MA50'] else "🔴 Dưới" if pd.notna(
                            latest['MA50']) else "N/A",
                        "🟢 Trên" if latest['close'] > latest['MA100'] else "🔴 Dưới" if pd.notna(
                            latest['MA100']) else "N/A",
                        "🟢 Trên" if latest['close'] > latest['MA200'] else "🔴 Dưới" if pd.notna(
                            latest['MA200']) else "N/A"
                    ]
                })
                st.dataframe(ma_df, hide_index=True, use_container_width=True)

        with tab3:
            st.subheader("📝 Phân Tích Kỹ Thuật Chi Tiết")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🎯 Tín hiệu tổng hợp: {signal}**")
                st.markdown(f"**⭐ Điểm số: {score}/100**")
                st.markdown(f"**⏰ Khuyến nghị: {term}**")

                st.markdown("---")
                st.markdown("**💡 Chi tiết điểm số:**")
                st.markdown(f"- Xu hướng: {details.get('trend_score', 0)}/30 điểm")
                st.markdown(f"- RSI: {details.get('rsi_score', 0)}/20 điểm")
                st.markdown(f"- MACD: {details.get('macd_score', 0)}/15 điểm")
                st.markdown(f"- Momentum: {details.get('momentum_score', 0)}/10 điểm")
                st.markdown(f"- ADX: {details.get('adx_score', 0)}/10 điểm")
                st.markdown(f"- Volume: {details.get('volume_score', 0)}/10 điểm")
                st.markdown(f"- Bollinger: {details.get('bb_score', 0)}/5 điểm")

            with col2:
                st.markdown("**🔍 Lý do chi tiết:**")
                st.text(reason)

        with tab4:
            st.subheader("💼 Phân Tích cơ bản Fundamental")
            if info and isinstance(info, dict) and len(info) > 0:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Market Cap",
                              f"{info.get('marketCap', 0) / 1e9:.2f}B VND" if info.get('marketCap') else "N/A")
                    st.metric("P/E Ratio",
                              f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A")
                    st.metric("PEG Ratio",
                              f"{info.get('pegRatio', 0):.2f}" if info.get('pegRatio') else "N/A")

                with col2:
                    st.metric("EPS",
                              f"{info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else "N/A")
                    st.metric("ROE",
                              f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A")
                    st.metric("ROA",
                              f"{info.get('returnOnAssets', 0) * 100:.2f}%" if info.get('returnOnAssets') else "N/A")

                with col3:
                    st.metric("Profit Margin",
                              f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else "N/A")
                    st.metric("Dividend Yield",
                              f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A")
                    st.metric("Debt/Equity",
                              f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else "N/A")

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Ratio",
                              f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else "N/A")
                    st.metric("Quick Ratio",
                              f"{info.get('quickRatio', 0):.2f}" if info.get('quickRatio') else "N/A")

                with col2:
                    st.metric("Revenue Growth",
                              f"{info.get('revenueGrowth', 0) * 100:.2f}%" if info.get('revenueGrowth') else "N/A")
                    st.metric("Earnings Growth",
                              f"{info.get('earningsGrowth', 0) * 100:.2f}%" if info.get('earningsGrowth') else "N/A")
            else:
                st.warning("⚠️ Không có dữ liệu fundamental chi tiết")

        with tab5:
            st.subheader("🎯 Mức Giá Quan Trọng")

            current_price = latest['close']
            high_52w = df['high'].tail(252).max() if len(df) >= 252 else df['high'].max()
            low_52w = df['low'].tail(252).min() if len(df) >= 252 else df['low'].min()

            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Kháng cự 1", f"{latest['BB_upper']:,.0f}" if pd.notna(latest['BB_upper']) else "N/A")
            col2.metric("💰 Giá hiện tại", f"{current_price:,.0f}")
            col3.metric("🟢 Hỗ trợ 1", f"{latest['BB_lower']:,.0f}" if pd.notna(latest['BB_lower']) else "N/A")

            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("📈 Cao nhất 52 tuần", f"{high_52w:,.0f}")
            col2.metric("📉 Thấp nhất 52 tuần", f"{low_52w:,.0f}")

            # Fibonacci levels
            st.markdown("---")
            st.markdown("**📐 Fibonacci Retracement (52 tuần)**")
            fib_range = high_52w - low_52w
            fib_levels = {
                '0%': high_52w,
                '23.6%': high_52w - (fib_range * 0.236),
                '38.2%': high_52w - (fib_range * 0.382),
                '50%': high_52w - (fib_range * 0.5),
                '61.8%': high_52w - (fib_range * 0.618),
                '100%': low_52w
            }

            fib_df = pd.DataFrame({
                'Mức': list(fib_levels.keys()),
                'Giá': [f"{v:,.0f}" for v in fib_levels.values()],
                'So với giá hiện tại': [
                    f"{((v - current_price) / current_price * 100):+.2f}%"
                    for v in fib_levels.values()
                ]
            })
            st.dataframe(fib_df, hide_index=True, use_container_width=True)

    else:
        st.error(f"❌ Không thể tải dữ liệu cho {symbol}")

elif mode == "🚀 Quét nhanh":
    st.header("🚀 QUÉT NHANH CỔ PHIẾU TIỀM NĂNG")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📊 Sẽ quét {len(stocks_to_scan)} mã cổ phiếu | Điểm tối thiểu: {min_score}")
    with col2:
        scan_button = st.button("🔍 BẮT ĐẦU QUÉT", type="primary", use_container_width=True)

    if scan_button:
        results = []
        progress = st.progress(0)
        status = st.empty()

        for idx, sym in enumerate(stocks_to_scan):
            status.text(f"Đang quét {sym}... ({idx + 1}/{len(stocks_to_scan)})")

            df, info = get_stock_data(sym, period='6mo')
            if df is not None and not df.empty:
                df = calculate_advanced_indicators(df)
                signal, score, reason, term, _ = generate_advanced_signal(df, info)

                if score >= min_score:
                    latest = df.iloc[-1]
                    ml_trend, ml_conf = predict_trend_ml(df)

                    results.append({
                        'Mã': sym,
                        'Tín hiệu': signal,
                        'Điểm': score,
                        'Giá': latest['close'],
                        'RSI': latest['RSI'],
                        'ADX': latest['ADX'],
                        'ML Trend': ml_trend,
                        'ML Confidence': f"{ml_conf * 100:.0f}%",
                        'Khuyến nghị': term
                    })

                    if len(results) >= max_results:
                        break

            progress.progress((idx + 1) / len(stocks_to_scan))
            time.sleep(0.05)

        progress.empty()
        status.empty()

        if results:
            result_df = pd.DataFrame(results).sort_values('Điểm', ascending=False)

            st.success(f"✅ Tìm thấy {len(result_df)} cổ phiếu tiềm năng!")

            # Display with styling
            st.dataframe(
                result_df.style.format({
                    'Giá': '{:,.0f}',
                    'RSI': '{:.1f}',
                    'ADX': '{:.1f}'
                }).background_gradient(subset=['Điểm'], cmap='RdYlGn'),
                use_container_width=True,
                height=600
            )

            # Download button
            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 Tải xuống CSV",
                csv,
                f"co_phieu_tiem_nang_{datetime.now():%Y%m%d_%H%M}.csv",
                "text/csv",
                use_container_width=True
            )

            # Summary stats
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Điểm TB", f"{result_df['Điểm'].mean():.1f}")
            col2.metric("⭐ Điểm cao nhất", f"{result_df['Điểm'].max():.0f}")
            col3.metric("📊 MUA signals", len(result_df[result_df['Tín hiệu'].str.contains('MUA')]))
            col4.metric("🤖 ML TĂNG", len(result_df[result_df['ML Trend'] == 'TĂNG']))
        else:
            st.warning("⚠️ Không tìm thấy cổ phiếu nào đạt tiêu chí")

elif mode == "📊 So sánh":
    st.header("📊 SO SÁNH CỔ PHIẾU")

    if compare_symbols and len(compare_symbols) >= 2:
        comparison_data = []
        all_dfs = {}

        for sym in compare_symbols:
            with st.spinner(f"Đang tải {sym}..."):
                df, info = get_stock_data(sym, period=period)
                if df is not None and not df.empty:
                    df = calculate_advanced_indicators(df)
                    signal, score, _, term, _ = generate_advanced_signal(df, info)
                    latest = df.iloc[-1]
                    ml_trend, ml_conf = predict_trend_ml(df)

                    all_dfs[sym] = df

                    comparison_data.append({
                        'Mã': sym,
                        'Giá': latest['close'],
                        'RSI': latest['RSI'],
                        'MACD': latest['MACD'],
                        'ADX': latest['ADX'],
                        'Điểm': score,
                        'Tín hiệu': signal,
                        'ML Trend': ml_trend,
                        'ML Conf': f"{ml_conf * 100:.0f}%",
                        'Khuyến nghị': term
                    })

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(
                comp_df.style.format({
                    'Giá': '{:,.0f}',
                    'RSI': '{:.1f}',
                    'MACD': '{:.2f}',
                    'ADX': '{:.1f}'
                }).background_gradient(subset=['Điểm'], cmap='RdYlGn'),
                use_container_width=True
            )

            # Price comparison chart (normalized)
            st.subheader("📈 So sánh biến động giá (chuẩn hóa)")
            fig_compare = go.Figure()
            for sym in compare_symbols:
                if sym in all_dfs:
                    df = all_dfs[sym]
                    normalized = (df['close'] / df['close'].iloc[0]) * 100
                    fig_compare.add_trace(go.Scatter(
                        x=df.index,
                        y=normalized,
                        name=sym,
                        mode='lines',
                        line=dict(width=2)
                    ))

            fig_compare.update_layout(
                title='So Sánh Giá (Chuẩn hóa = 100)',
                yaxis_title='Giá trị (%)',
                template='plotly_dark',
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # Volume comparison
            st.subheader("📊 So sánh khối lượng")
            fig_vol = go.Figure()
            for sym in compare_symbols:
                if sym in all_dfs:
                    df = all_dfs[sym]
                    fig_vol.add_trace(go.Bar(x=df.index, y=df['volume'], name=sym))

            fig_vol.update_layout(
                title='So sánh khối lượng giao dịch',
                yaxis_title='Volume',
                template='plotly_dark',
                height=400,
                barmode='group'
            )
            st.plotly_chart(fig_vol, use_container_width=True)

            # Correlation matrix
            if len(compare_symbols) >= 3:
                st.subheader("🔗 Ma trận tương quan")
                close_prices = pd.DataFrame({sym: all_dfs[sym]['close'] for sym in compare_symbols if sym in all_dfs})
                corr = close_prices.corr()

                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=corr.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 12}
                ))
                fig_corr.update_layout(
                    title='Correlation Matrix',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("ℹ️ Vui lòng chọn ít nhất 2 mã để so sánh")

elif mode == "🤖 AI Prediction":
    st.header(f"🤖 AI PREDICTION: {symbol}")

    with st.spinner(f"Đang tải và phân tích {symbol}..."):
        df, info = get_stock_data(symbol, period='1y')

    if df is not None and not df.empty:
        df = calculate_advanced_indicators(df)
        signal, score, reason, term, _ = generate_advanced_signal(df, info)

        # Current info
        latest = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Giá hiện tại", f"{latest['close']:,.0f}")
        col2.metric("⭐ Điểm AI", f"{score}/100")
        col3.metric("📈 RSI", f"{latest['RSI']:.1f}")
        col4.metric("💪 ADX", f"{latest['ADX']:.1f}")

        st.markdown("---")

        # ML Trend prediction
        ml_trend, ml_conf = predict_trend_ml(df)
        trend_color = "#00c853" if ml_trend == "TĂNG" else "#d32f2f"

        st.markdown(f"""
        <div class="prediction-box">
            <h2>🤖 Machine Learning Analysis</h2>
            <p style="font-size:28px; margin:15px 0;">
                Xu hướng dự đoán: <span style="color:{trend_color}; font-weight:bold;">{ml_trend}</span>
            </p>
            <p style="font-size:20px;">
                Độ tin cậy: <b>{ml_conf * 100:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Price prediction
        predictions = predict_future_price(df, pred_days)

        if predictions is not None:
            pred_change = ((predictions[-1] - latest['close']) / latest['close']) * 100
            pred_color = "#00c853" if pred_change > 0 else "#d32f2f"

            st.markdown(f"""
            <div class="prediction-box">
                <h2>🔮 Dự đoán giá {pred_days} ngày tới</h2>
                <p style="font-size:24px; margin:15px 0;">
                    Giá hiện tại: <b>{latest['close']:,.0f} VND</b>
                </p>
                <p style="font-size:28px; margin:15px 0;">
                    Giá dự kiến: <b>{predictions[-1]:,.0f} VND</b>
                </p>
                <p style="font-size:24px;">
                    Thay đổi: <span style="color:{pred_color}; font-weight:bold;">{pred_change:+.2f}%</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Prediction chart
            st.subheader("📈 Biểu đồ dự đoán")
            fig = plot_advanced_chart(df.tail(90), symbol, predictions)
            st.plotly_chart(fig, use_container_width=True)

            # Prediction table
            st.subheader("📊 Bảng dự đoán chi tiết")
            future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=pred_days)
            pred_df = pd.DataFrame({
                'Ngày': future_dates.strftime('%Y-%m-%d'),
                'Giá dự đoán': predictions,
                'Thay đổi vs hôm nay': [f"{((p - latest['close']) / latest['close'] * 100):+.2f}%" for p in
                                        predictions],
                'Thay đổi vs ngày trước': [
                    f"{((predictions[i] - (predictions[i - 1] if i > 0 else latest['close'])) / (predictions[i - 1] if i > 0 else latest['close']) * 100):+.2f}%"
                    for i in range(len(predictions))]
            })
            pred_df['Giá dự đoán'] = pred_df['Giá dự đoán'].apply(lambda x: f"{x:,.0f}")
            st.dataframe(pred_df, hide_index=True, use_container_width=True)

            # Risk assessment
            st.markdown("---")
            st.subheader("⚠️ Đánh giá rủi ro")
            col1, col2, col3 = st.columns(3)

            volatility = df['close'].tail(30).std() / df['close'].tail(30).mean() * 100
            max_drawdown = ((df['close'].tail(90).min() - df['close'].tail(90).max()) / df['close'].tail(
                90).max()) * 100

            col1.metric("📊 Độ biến động (30 ngày)", f"{volatility:.2f}%")
            col2.metric("📉 Max Drawdown (90 ngày)", f"{max_drawdown:.2f}%")
            col3.metric("🎲 Confidence Level", f"{ml_conf * 100:.0f}%")

            risk_level = "CAO" if volatility > 5 or abs(max_drawdown) > 20 else "TRUNG BÌNH" if volatility > 3 or abs(
                max_drawdown) > 10 else "THẤP"
            risk_color = "#d32f2f" if risk_level == "CAO" else "#ff9800" if risk_level == "TRUNG BÌNH" else "#00c853"

            st.markdown(f"""
            <div style="background:{risk_color}; padding:15px; border-radius:10px; color:white; text-align:center; margin:20px 0;">
                <h3>Mức độ rủi ro: {risk_level}</h3>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error(f"❌ Không thể tải dữ liệu cho {symbol}")

else:  # Backtesting
    st.header(f"📈 BACKTESTING: {symbol}")

    with st.spinner(f"Đang chạy backtest cho {symbol}..."):
        df, info = get_stock_data(symbol, period='2y')

    if df is not None and not df.empty:
        df = calculate_advanced_indicators(df)

        # Run backtest
        backtest_results = simple_backtest(df, initial_capital)

        if backtest_results:
            # Summary metrics
            st.subheader("📊 Kết quả tổng quan")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("💰 Vốn ban đầu", f"{initial_capital:,.0f} VND")
            col2.metric("💵 Giá trị cuối", f"{backtest_results['final_value']:,.0f} VND")
            col3.metric("📈 ROI", f"{backtest_results['roi']:.2f}%",
                        f"{backtest_results['roi']:+.2f}%")
            col4.metric("🔄 Số giao dịch", backtest_results['num_trades'])

            # Performance comparison
            buy_hold_return = ((df.iloc[-1]['close'] - df.iloc[50]['close']) / df.iloc[50]['close']) * 100

            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("🤖 Chiến lược AI", f"{backtest_results['roi']:.2f}%")
            col2.metric("🎯 Buy & Hold", f"{buy_hold_return:.2f}%")

            if backtest_results['roi'] > buy_hold_return:
                st.success(f"✅ Chiến lược AI tốt hơn Buy & Hold: +{(backtest_results['roi'] - buy_hold_return):.2f}%")
            else:
                st.warning(f"⚠️ Buy & Hold tốt hơn: +{(buy_hold_return - backtest_results['roi']):.2f}%")

            # Trades table
            st.markdown("---")
            st.subheader("📋 Lịch sử giao dịch")
            if backtest_results['trades']:
                trades_df = pd.DataFrame(backtest_results['trades'])
                trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
                trades_df['price'] = trades_df['price'].apply(lambda x: f"{x:,.0f}")
                trades_df['value'] = trades_df['value'].apply(lambda x: f"{x:,.0f}")

                st.dataframe(trades_df, hide_index=True, use_container_width=True, height=400)

            # Equity curve
            st.markdown("---")
            st.subheader("📈 Đường vốn (Equity Curve)")

            equity_data = []
            capital = initial_capital
            shares = 0

            for i in range(50, len(df)):
                current_data = df.iloc[:i + 1]
                current_data = calculate_advanced_indicators(current_data)
                signal, score, _, _, _ = generate_advanced_signal(current_data)

                current_price = df.iloc[i]['close']

                if signal.startswith("MUA") and shares == 0 and capital > current_price:
                    shares = int(capital * 0.9 / current_price)
                    capital -= shares * current_price
                elif signal.startswith("BÁN") and shares > 0:
                    capital += shares * current_price
                    shares = 0

                total_value = capital + (shares * current_price)
                equity_data.append({
                    'date': df.index[i],
                    'value': total_value
                })

            equity_df = pd.DataFrame(equity_data)

            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=equity_df['date'],
                y=equity_df['value'],
                name='Equity',
                line=dict(color='#00c853', width=3),
                fill='tonexty',
                fillcolor='rgba(0,200,83,0.1)'
            ))
            fig_equity.add_hline(y=initial_capital, line_dash="dash", line_color="white",
                                 annotation_text="Vốn ban đầu")

            fig_equity.update_layout(
                title='Biến động vốn theo thời gian',
                yaxis_title='Giá trị (VND)',
                template='plotly_dark',
                height=500
            )
            st.plotly_chart(fig_equity, use_container_width=True)

            # Trade analysis
            st.markdown("---")
            st.subheader("🎯 Phân tích giao dịch")

            if len(backtest_results['trades']) >= 2:
                buy_trades = [t for t in backtest_results['trades'] if t['action'] == 'BUY']
                sell_trades = [t for t in backtest_results['trades'] if t['action'] == 'SELL']

                wins = 0
                losses = 0
                for i in range(min(len(buy_trades), len(sell_trades))):
                    if sell_trades[i]['price'] > buy_trades[i]['price']:
                        wins += 1
                    else:
                        losses += 1

                win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

                col1, col2, col3 = st.columns(3)
                col1.metric("✅ Thắng", wins)
                col2.metric("❌ Thua", losses)
                col3.metric("📊 Tỷ lệ thắng", f"{win_rate:.1f}%")

                # Win rate visualization
                fig_winrate = go.Figure(data=[go.Pie(
                    labels=['Thắng', 'Thua'],
                    values=[wins, losses],
                    marker_colors=['#00c853', '#d32f2f'],
                    hole=0.4
                )])
                fig_winrate.update_layout(
                    title='Tỷ lệ thắng/thua',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_winrate, use_container_width=True)
        else:
            st.warning("⚠️ Không đủ dữ liệu để chạy backtest")
    else:
        st.error(f"❌ Không thể tải dữ liệu cho {symbol}")

# Footer with real-time stats
st.markdown("---")
st.markdown(f"""
<div class="footer-stats">
    <div style="font-size:14px; font-weight:bold; margin-bottom:8px;">📊 THỐNG KÊ</div>
    <div>👥 Online: <b>{len(st.session_state.online_users)}</b></div>
    <div>📈 Lượt truy cập: <b>{st.session_state.visit_count}</b></div>
    <div>💹 Tổng mã: <b>{len(ALL_STOCKS)}</b></div>
    <div style="margin-top:8px; font-size:10px; opacity:0.7;">
        ⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
    </div>
    <div style="margin-top:5px; font-size:9px; opacity:0.6;">
        Dữ liệu được thống kê tự động bằng AI
    </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("---")
st.warning("""
⚠️ **LƯU Ý QUAN TRỌNG:**
- Đây là công cụ hỗ trợ phân tích, KHÔNG phải lời khuyên đầu tư
- Kết quả dự đoán chỉ mang tính tham khảo
- Luôn tự nghiên cứu và chịu trách nhiệm với quyết định của mình
- Thị trường chứng khoán có rủi ro cao, có thể mất vốn
- Nên tham khảo ý kiến chuyên gia tài chính trước khi đầu tư
""")

st.info("""
💡 **MẸO SỬ DỤNG:**
- Kết hợp nhiều chỉ báo để có quyết định tốt nhất
- Điểm số ≥80: Tín hiệu mua mạnh nhưng vẫn cần kiểm tra fundamental
- Chú ý đến khối lượng giao dịch và xu hướng thị trường chung
- Sử dụng Stop Loss để bảo vệ vốn
- Đa dạng hóa danh mục để giảm rủi ro
""")