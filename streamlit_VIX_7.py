"""
VIX Risk Desk Trading Dashboard - Professional Edition v2.0
Enhanced with GARCH, LSTM, Dynamic Optimization, Real-time Greeks, and Trade Signals
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix, 
                             log_loss, brier_score_loss, roc_curve)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import norm
from scipy.interpolate import CubicSpline, Rbf, griddata
from hmmlearn import hmm
from arch import arch_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="VIX Risk Desk Dashboard v2.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        border-bottom: 2px solid #00D9FF;
    }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2e3340;
        padding: 15px;
        border-radius: 5px;
    }
    h1, h2, h3 { color: #ffffff; font-weight: 700; }
    .risk-high { color: #FF6B6B; font-weight: bold; font-size: 18px; }
    .risk-medium { color: #FFA500; font-weight: bold; font-size: 18px; }
    .risk-low { color: #4CAF50; font-weight: bold; font-size: 18px; }
    .trade-signal { 
        background-color: #1e2130; 
        padding: 20px; 
        border-radius: 10px; 
        border: 2px solid #00D9FF;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== VIX OPTIONS & IV SURFACE ====================
@st.cache_data(ttl=1800)
def fetch_vix_options():
    """Fetch VIX options chain for IV surface construction"""
    try:
        vix = yf.Ticker("^VIX")
        expirations = vix.options[:8]
        
        all_options = []
        for exp in expirations:
            try:
                opt_chain = vix.option_chain(exp)
                calls = opt_chain.calls
                puts = opt_chain.puts
                
                calls['type'] = 'call'
                puts['type'] = 'put'
                calls['expiration'] = exp
                puts['expiration'] = exp
                
                all_options.append(calls)
                all_options.append(puts)
            except:
                continue
        
        if all_options:
            df_options = pd.concat(all_options, ignore_index=True)
            df_options['expiration'] = pd.to_datetime(df_options['expiration'])
            df_options['dte'] = (df_options['expiration'] - pd.Timestamp.now()).dt.days
            return df_options
        return None
    except:
        return None

def black_scholes_iv(option_price, S, K, T, r, option_type='call'):
    """Calculate implied volatility using Newton-Raphson"""
    if T <= 0 or option_price <= 0:
        return np.nan
    
    def bs_price(sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    def vega(sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S*norm.pdf(d1)*np.sqrt(T)
    
    sigma = 0.5
    for _ in range(100):
        price = bs_price(sigma)
        v = vega(sigma)
        
        if abs(v) < 1e-10:
            break
            
        diff = option_price - price
        if abs(diff) < 1e-6:
            return sigma
        
        sigma = sigma + diff/v
        
        if sigma <= 0 or sigma > 5:
            return np.nan
    
    return sigma if 0 < sigma < 5 else np.nan

def calculate_iv_surface(df_options, current_vix):
    """Calculate IV surface from options data"""
    if df_options is None or len(df_options) == 0:
        return None
    
    df_options = df_options[
        (df_options['dte'] > 0) & 
        (df_options['volume'] > 0) &
        (df_options['bid'] > 0) &
        (df_options['ask'] > 0)
    ].copy()
    
    if len(df_options) == 0:
        return None
    
    df_options['mid_price'] = (df_options['bid'] + df_options['ask']) / 2
    df_options['moneyness'] = df_options['strike'] / current_vix
    df_options['T'] = df_options['dte'] / 365.0
    
    risk_free_rate = 0.05
    
    ivs = []
    for _, row in df_options.iterrows():
        iv = black_scholes_iv(
            row['mid_price'],
            current_vix,
            row['strike'],
            row['T'],
            risk_free_rate,
            row['type']
        )
        ivs.append(iv)
    
    df_options['iv'] = ivs
    df_options = df_options[df_options['iv'].notna() & (df_options['iv'] > 0) & (df_options['iv'] < 3)]
    
    return df_options

def plot_iv_surface_3d(df_iv):
    """Create 3D IV surface plot"""
    if df_iv is None or len(df_iv) == 0:
        return None
    
    calls = df_iv[df_iv['type'] == 'call']
    puts = df_iv[df_iv['type'] == 'put']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Call IV Surface', 'Put IV Surface'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )
    
    for idx, (data, title) in enumerate([(calls, 'Calls'), (puts, 'Puts')], 1):
        if len(data) == 0:
            continue
        
        moneyness = data['moneyness'].values
        dte = data['dte'].values
        iv = data['iv'].values
        
        moneyness_grid = np.linspace(moneyness.min(), moneyness.max(), 30)
        dte_grid = np.linspace(dte.min(), dte.max(), 30)
        moneyness_mesh, dte_mesh = np.meshgrid(moneyness_grid, dte_grid)
        
        try:
            from scipy.interpolate import Rbf
            rbf = Rbf(moneyness, dte, iv, function='multiquadric', smooth=0.1)
            iv_mesh = rbf(moneyness_mesh, dte_mesh)
            iv_mesh = np.clip(iv_mesh, 0, 3)
            
            fig.add_trace(
                go.Surface(
                    x=moneyness_mesh,
                    y=dte_mesh,
                    z=iv_mesh,
                    colorscale='Viridis',
                    name=title,
                    showscale=(idx==1),
                    colorbar=dict(title="IV", x=0.45 if idx==1 else 1.02)
                ),
                row=1, col=idx
            )
        except:
            fig.add_trace(
                go.Scatter3d(
                    x=moneyness,
                    y=dte,
                    z=iv,
                    mode='markers',
                    marker=dict(size=3, color=iv, colorscale='Viridis'),
                    name=title
                ),
                row=1, col=idx
            )
    
    fig.update_layout(
        title='VIX Options Implied Volatility Surface',
        scene=dict(
            xaxis_title='Moneyness (K/S)',
            yaxis_title='Days to Expiry',
            zaxis_title='Implied Volatility',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        scene2=dict(
            xaxis_title='Moneyness (K/S)',
            yaxis_title='Days to Expiry',
            zaxis_title='Implied Volatility',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        template='plotly_dark',
        height=700,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117'
    )
    
    return fig

def plot_iv_term_structure(df_iv):
    """Plot IV term structure by moneyness buckets"""
    if df_iv is None or len(df_iv) == 0:
        return None
    
    df_iv['moneyness_bucket'] = pd.cut(
        df_iv['moneyness'],
        bins=[0, 0.9, 0.95, 1.0, 1.05, 1.1, 2.0],
        labels=['<0.9', '0.9-0.95', '0.95-1.0', '1.0-1.05', '1.05-1.1', '>1.1']
    )
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Calls', 'Puts'))
    
    colors = ['#FF6B6B', '#FFA500', '#FFD700', '#4CAF50', '#00D9FF', '#9C27B0']
    
    for opt_type, col in [('call', 1), ('put', 2)]:
        data = df_iv[df_iv['type'] == opt_type]
        
        for bucket, color in zip(['<0.9', '0.9-0.95', '0.95-1.0', '1.0-1.05', '1.05-1.1', '>1.1'], colors):
            bucket_data = data[data['moneyness_bucket'] == bucket]
            if len(bucket_data) > 0:
                term_structure = bucket_data.groupby('dte')['iv'].mean().sort_index()
                
                fig.add_trace(
                    go.Scatter(
                        x=term_structure.index,
                        y=term_structure.values,
                        mode='lines+markers',
                        name=f'K/S {bucket}',
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                        showlegend=(col==1)
                    ),
                    row=1, col=col
                )
    
    fig.update_xaxes(title_text="Days to Expiry", row=1, col=1)
    fig.update_xaxes(title_text="Days to Expiry", row=1, col=2)
    fig.update_yaxes(title_text="Implied Volatility", row=1, col=1)
    fig.update_yaxes(title_text="Implied Volatility", row=1, col=2)
    
    fig.update_layout(
        title='IV Term Structure by Moneyness',
        template='plotly_dark',
        height=600,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117'
    )
    
    return fig

# ==================== DATA FETCHING ====================
@st.cache_data(ttl=3600)
def fetch_cross_asset_data(start_date, end_date):
    """Fetch VIX and cross-asset volatility data"""
    tickers = {
        'VIX': '^VIX', 'VVIX': '^VVIX', 'VXV': '^VXV',
        'SPY': 'SPY', 'TLT': 'TLT', 'GLD': 'GLD',
        'DXY': 'DX-Y.NYB', 'HYG': 'HYG',
        'VXX': 'VXX', 'UVXY': 'UVXY'
    }
    
    data_dict = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if 'Close' in df.columns:
                data_dict[name] = df['Close']
            if name in ['VXX', 'UVXY', 'SPY'] and 'Volume' in df.columns:
                data_dict[f'{name}_Volume'] = df['Volume']
        except:
            continue
    
    df = pd.DataFrame(data_dict)
    df = df.fillna(method='ffill').dropna()
    return df

@st.cache_data(ttl=1800)
def fetch_vix_futures():
    """Fetch VIX futures for term structure"""
    try:
        vix_futures = {}
        for month in range(1, 8):
            ticker = f'^VIX{month}'
            try:
                data = yf.download(ticker, period='5d', progress=False)
                if not data.empty and 'Close' in data.columns:
                    vix_futures[f'F{month}'] = data['Close'].iloc[-1]
            except:
                continue
        return vix_futures
    except:
        return {}

# ==================== GARCH MODELING ====================
def fit_garch_model(returns, p=1, q=1):
    """Fit GARCH model to returns"""
    try:
        model = arch_model(returns * 100, vol='Garch', p=p, q=q, rescale=False)
        fitted = model.fit(disp='off', show_warning=False)
        return fitted
    except:
        return None

def extract_garch_features(vix_series, lookback=252):
    """Extract GARCH volatility forecasts as features"""
    features = pd.DataFrame(index=vix_series.index)
    returns = vix_series.pct_change().dropna()
    
    garch_vol = []
    for i in range(lookback, len(returns)):
        window_returns = returns.iloc[i-lookback:i]
        fitted = fit_garch_model(window_returns)
        
        if fitted is not None:
            forecast = fitted.forecast(horizon=1)
            garch_vol.append(np.sqrt(forecast.variance.values[-1, 0]))
        else:
            garch_vol.append(np.nan)
    
    features['GARCH_vol'] = pd.Series(garch_vol, index=returns.index[lookback:])
    features['GARCH_vol'] = features['GARCH_vol'].fillna(method='ffill')
    
    return features

# ==================== LSTM MODEL ====================
def build_lstm_model(input_shape, units=64):
    """Build LSTM model"""
    model = keras.Sequential([
        layers.LSTM(units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(units // 2, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def prepare_lstm_data(X, y, lookback=20):
    """Prepare 3D data for LSTM"""
    X_lstm, y_lstm = [], []
    
    for i in range(lookback, len(X)):
        X_lstm.append(X.iloc[i-lookback:i].values)
        y_lstm.append(y.iloc[i])
    
    return np.array(X_lstm), np.array(y_lstm)

def train_lstm_model(X_train, y_train, X_test, y_test, lookback=20, epochs=50):
    """Train LSTM model"""
    X_train_lstm, y_train_lstm = prepare_lstm_data(X_train, y_train, lookback)
    X_test_lstm, y_test_lstm = prepare_lstm_data(X_test, y_test, lookback)
    
    if len(X_train_lstm) == 0 or len(X_test_lstm) == 0:
        return None, None, None
    
    model = build_lstm_model((lookback, X_train.shape[1]))
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train_lstm, y_train_lstm, 
              validation_data=(X_test_lstm, y_test_lstm),
              epochs=epochs, batch_size=32, verbose=0, callbacks=[early_stop])
    
    train_probs = model.predict(X_train_lstm, verbose=0).flatten()
    test_probs = model.predict(X_test_lstm, verbose=0).flatten()
    
    return model, train_probs, test_probs

# ==================== FEATURE ENGINEERING ====================
def build_microstructure_features(df):
    """Build microstructure features"""
    features = pd.DataFrame(index=df.index)
    
    if 'VXV' in df.columns:
        features['VIX_VXV_spread'] = df['VIX'] - df['VXV']
        features['VIX_VXV_ratio'] = df['VIX'] / df['VXV']
    
    if 'VXX' in df.columns and 'VIX' in df.columns:
        features['VXX_premium'] = df['VXX'].pct_change(5) - df['VIX'].pct_change(5)
    
    if 'UVXY' in df.columns and 'VIX' in df.columns:
        features['UVXY_premium'] = df['UVXY'].pct_change(5) - (df['VIX'].pct_change(5) * 2)
    
    if 'VXX_Volume' in df.columns:
        features['VXX_volume_ma_ratio'] = df['VXX_Volume'] / df['VXX_Volume'].rolling(20).mean()
    
    if 'SPY' in df.columns and 'VIX' in df.columns:
        spy_realized_vol = df['SPY'].pct_change().rolling(21).std() * np.sqrt(252) * 100
        features['RV_IV_spread'] = spy_realized_vol - df['VIX']
    
    return features.dropna()

def build_advanced_features(df, feature_config):
    """Build comprehensive feature set with polynomial features"""
    features = pd.DataFrame(index=df.index)
    
    if feature_config.get('vix_lag_5'): features['VIX_lag5'] = df['VIX'].shift(5)
    if feature_config.get('vix_lag_22'): features['VIX_lag22'] = df['VIX'].shift(22)
    if feature_config.get('vix_momentum_5'): features['VIX_momentum_5d'] = df['VIX'].pct_change(5)
    if feature_config.get('vix_momentum_22'): features['VIX_momentum_22d'] = df['VIX'].pct_change(22)
    if feature_config.get('vix_zscore'):
        features['VIX_zscore'] = (df['VIX'] - df['VIX'].rolling(252, min_periods=60).mean()) / \
                                 df['VIX'].rolling(252, min_periods=60).std()
    if feature_config.get('vix_realized_vol'):
        features['VIX_realized_vol'] = df['VIX'].pct_change().rolling(22).std() * np.sqrt(252)
    if feature_config.get('vix_ma_ratio'): features['VIX_MA_ratio'] = df['VIX'] / df['VIX'].rolling(20).mean()
    if 'VXV' in df.columns and feature_config.get('term_structure'): features['VIX_term_structure'] = df['VIX'] / df['VXV']
    if 'VVIX' in df.columns and feature_config.get('vvix'):
        features['VVIX'] = df['VVIX']
        features['VVIX_momentum'] = df['VVIX'].pct_change(5)
    if 'SPY' in df.columns and feature_config.get('spy_return'):
        features['SPY_return_5d'] = df['SPY'].pct_change(5)
        features['SPY_return_22d'] = df['SPY'].pct_change(22)
    if 'SPY' in df.columns and feature_config.get('spy_vol'):
        features['SPY_realized_vol'] = df['SPY'].pct_change().rolling(22).std() * np.sqrt(252)
    if 'HYG' in df.columns and feature_config.get('credit'): features['HYG_return_22d'] = df['HYG'].pct_change(22)
    if 'DXY' in df.columns and feature_config.get('dollar'): features['DXY_change_22d'] = df['DXY'].pct_change(22)
    if 'GLD' in df.columns and feature_config.get('gold'): features['GLD_return_22d'] = df['GLD'].pct_change(22)
    if feature_config.get('vix_regime'): features['VIX_high_regime'] = (df['VIX'].shift(1) > 20).astype(int)
    if 'TLT' in df.columns and feature_config.get('tlt_change'): features['TLT_change_22d'] = df['TLT'].pct_change(22)
    
    if feature_config.get('garch'):
        garch_features = extract_garch_features(df['VIX'])
        features = features.join(garch_features, how='left')
    
    if feature_config.get('microstructure'):
        micro_features = build_microstructure_features(df)
        features = features.join(micro_features, how='left')
    
    if feature_config.get('polynomial'):
        base_cols = ['VIX_lag5', 'VIX_momentum_5d', 'VIX_zscore']
        base_cols = [c for c in base_cols if c in features.columns]
        
        if len(base_cols) >= 2:
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            poly_features = poly.fit_transform(features[base_cols].fillna(0))
            poly_names = poly.get_feature_names_out(base_cols)
            
            for i, name in enumerate(poly_names):
                if name not in base_cols:
                    features[name] = poly_features[:, i]
    
    return features.dropna()

def remove_multicollinearity(X, vif_threshold=10):
    """Remove multicollinear features"""
    X_clean = X.copy()
    corr = X_clean.corr().abs()
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    X_clean = X_clean.drop(columns=to_drop)
    
    for _ in range(15):
        if X_clean.shape[1] <= 2: break
        try:
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X_clean.columns
            vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i) for i in range(X_clean.shape[1])]
            max_vif_idx = vif_data["VIF"].replace([np.inf, -np.inf], np.nan).idxmax()
            max_vif = vif_data.loc[max_vif_idx, "VIF"]
            if max_vif <= vif_threshold or pd.isna(max_vif): break
            X_clean = X_clean.drop(columns=[vif_data.loc[max_vif_idx, "Feature"]])
        except:
            break
    return X_clean

# ==================== MODEL TRAINING ====================
def fit_multiple_models(X_train, y_train, random_state=42):
    """Fit multiple models"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    models = {}
    
    log_reg = LogisticRegression(penalty='l2', C=0.1, solver='liblinear', class_weight='balanced', 
                                  max_iter=1000, random_state=random_state)
    log_reg.fit(X_train_scaled, y_train)
    models['Logistic'] = log_reg
    
    scale_pos_weight = max(1, len(y_train[y_train==0])/max(1, len(y_train[y_train==1])))
    xgb_clf = xgb.XGBClassifier(max_depth=4, n_estimators=200, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                                eval_metric="logloss", random_state=random_state)
    xgb_clf.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb_clf
    
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=10,
                                   class_weight='balanced', random_state=random_state, n_jobs=-1)
    rf_clf.fit(X_train_scaled, y_train)
    models['RandomForest'] = rf_clf
    
    calibrated_models = {}
    for name, model in models.items():
        try:
            cal_clf = CalibratedClassifierCV(model, method='isotonic', cv=3)
            cal_clf.fit(X_train_scaled, y_train)
            calibrated_models[f'{name}_Calibrated'] = cal_clf
        except:
            pass
    models.update(calibrated_models)
    
    return models, scaler

# ==================== REGIME DETECTION ====================
def detect_regimes_hmm(vix_data, n_regimes=3):
    """Detect volatility regimes"""
    features = np.column_stack([
        vix_data.values,
        vix_data.pct_change().fillna(0).values,
        vix_data.rolling(20).std().fillna(0).values
    ])
    
    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(features)
    regimes = model.predict(features)
    
    regime_means = [vix_data[regimes == i].mean() for i in range(n_regimes)]
    regime_labels = np.argsort(regime_means)
    regime_mapping = {old: new for new, old in enumerate(regime_labels)}
    regimes_labeled = np.array([regime_mapping[r] for r in regimes])
    
    regime_names = ['Low Vol', 'Medium Vol', 'High Vol'][:n_regimes]
    return pd.Series(regimes_labeled, index=vix_data.index), regime_names

# ==================== METRICS & EVALUATION ====================
def calculate_model_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate comprehensive metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Log Loss': log_loss(y_true, y_pred_proba),
        'Brier Score': brier_score_loss(y_true, y_pred_proba),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

def rolling_window_validation(X, y, model_template, scaler_template, window_size=252, step_size=21):
    """Rolling window validation"""
    results = []
    for i in range(window_size, len(X), step_size):
        X_train = X.iloc[i-window_size:i]
        y_train = y.iloc[i-window_size:i]
        X_test = X.iloc[i:min(i+step_size, len(X))]
        y_test = y.iloc[i:min(i+step_size, len(y))]
        
        if len(y_test) == 0 or len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
            continue
        
        from sklearn.base import clone
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        try:
            model = clone(model_template)
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            continue
        
        results.append({'Date': y_test.index[-1], 'AUC': auc})
    
    return pd.DataFrame(results)

# ==================== THRESHOLD OPTIMIZATION ====================
def optimize_threshold(y_true, y_pred_proba, fp_cost=1, fn_cost=5):
    """Find optimal threshold based on cost function"""
    thresholds = np.linspace(0.1, 0.9, 81)
    costs = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fp * fp_cost) + (fn * fn_cost)
        costs.append({
            'threshold': thresh,
            'cost': total_cost,
            'fp': fp,
            'fn': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        })
    
    df_costs = pd.DataFrame(costs)
    optimal_row = df_costs.loc[df_costs['cost'].idxmin()]
    
    return optimal_row['threshold'], df_costs

# ==================== GREEKS & PORTFOLIO ====================
def calculate_greeks(vix_current, spike_prob, portfolio_delta, portfolio_vega, portfolio_gamma=0):
    """Calculate portfolio Greeks and P&L impacts"""
    vix_spike_target = 30
    expected_vix_change = spike_prob * (vix_spike_target - vix_current)
    
    delta_impact = portfolio_delta * (-0.02) * spike_prob
    gamma_impact = 0.5 * portfolio_gamma * ((-0.02) ** 2) * spike_prob
    vega_impact = portfolio_vega * expected_vix_change
    
    total_pnl = delta_impact + gamma_impact + vega_impact
    var_95 = total_pnl * 1.645
    cvar_95 = total_pnl * 2.0
    
    return {
        'Expected_VIX_Change': expected_vix_change,
        'Delta_Impact': delta_impact,
        'Gamma_Impact': gamma_impact,
        'Vega_Impact': vega_impact,
        'Total_PnL_Impact': total_pnl,
        'VaR_95': var_95,
        'CVaR_95': cvar_95
    }

def calculate_portfolio_stress(spike_prob, portfolio_delta, portfolio_vega, vix_current):
    """Calculate portfolio stress metrics"""
    return calculate_greeks(vix_current, spike_prob, portfolio_delta, portfolio_vega)

# ==================== BACKTEST STRATEGY ====================
def backtest_vix_strategy(vix_series, probabilities, threshold_high=0.6, threshold_low=0.3, 
                         initial_capital=100000, position_size=0.10):
    """
    Improved VIX trading strategy backtest
    """
    portfolio_value = initial_capital
    portfolio_history = [initial_capital]
    cash = initial_capital
    position = None
    entry_vix = None
    entry_date = None
    trades = []
    dates_history = []  # ADD THIS
    
    # Align indices
    common_idx = vix_series.index.intersection(probabilities.index)
    vix_series = vix_series.loc[common_idx]
    probabilities = probabilities.loc[common_idx]
    
    dates_history.append(common_idx[0])  # ADD THIS
    
    for i in range(1, len(common_idx)):
        date = common_idx[i]
        prob = probabilities.iloc[i]
        vix = vix_series.iloc[i]
        prev_vix = vix_series.iloc[i-1]
        
        # Position management
        if position is not None:
            if position['type'] == 'LONG':
                pnl = position['notional'] * ((vix - entry_vix) / entry_vix)
                return_pct = (vix - entry_vix) / entry_vix
            else:
                pnl = position['notional'] * ((entry_vix - vix) / entry_vix)
                return_pct = (entry_vix - vix) / entry_vix
            
            current_value = position['notional'] + pnl
            
            should_exit = False
            exit_reason = ""
            
            if position['type'] == 'LONG':
                if prob < threshold_low:
                    should_exit = True
                    exit_reason = "Probability dropped"
                elif return_pct > 0.25:
                    should_exit = True
                    exit_reason = "Profit target hit"
                elif return_pct < -0.50:
                    should_exit = True
                    exit_reason = "Stop loss"
            else:
                if prob > threshold_high:
                    should_exit = True
                    exit_reason = "Probability increased"
                elif return_pct > 0.20:
                    should_exit = True
                    exit_reason = "Profit target hit"
                elif return_pct < -0.50:
                    should_exit = True
                    exit_reason = "Stop loss"
            
            if should_exit:
                cash += current_value
                portfolio_value = cash
                
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': date,
                    'Type': position['type'],
                    'Entry VIX': entry_vix,
                    'Exit VIX': vix,
                    'Entry Prob': position['entry_prob'],
                    'Exit Prob': prob,
                    'Notional': position['notional'],
                    'P&L': pnl,
                    'Return': return_pct,
                    'Exit Reason': exit_reason
                })
                
                position = None
                entry_vix = None
                entry_date = None
            else:
                portfolio_value = cash + current_value - position['notional']
        
        if position is None:
            capital_at_risk = cash * position_size
            
            if prob > threshold_high and vix < 45:
                position = {
                    'type': 'LONG',
                    'notional': capital_at_risk,
                    'entry_prob': prob
                }
                entry_vix = vix
                entry_date = date
                
            elif prob < threshold_low and vix > 10:
                position = {
                    'type': 'SHORT',
                    'notional': capital_at_risk,
                    'entry_prob': prob
                }
                entry_vix = vix
                entry_date = date
        
        portfolio_history.append(portfolio_value)
        dates_history.append(date)  # ADD THIS
    
    # Close remaining position
    if position is not None:
        final_vix = vix_series.iloc[-1]
        final_prob = probabilities.iloc[-1]
        
        if position['type'] == 'LONG':
            pnl = position['notional'] * ((final_vix - entry_vix) / entry_vix)
            return_pct = (final_vix - entry_vix) / entry_vix
        else:
            pnl = position['notional'] * ((entry_vix - final_vix) / entry_vix)
            return_pct = (entry_vix - final_vix) / entry_vix
        
        cash += position['notional'] + pnl
        portfolio_value = cash
        
        trades.append({
            'Entry Date': entry_date,
            'Exit Date': common_idx[-1],
            'Type': position['type'],
            'Entry VIX': entry_vix,
            'Exit VIX': final_vix,
            'Entry Prob': position['entry_prob'],
            'Exit Prob': final_prob,
            'Notional': position['notional'],
            'P&L': pnl,
            'Return': return_pct,
            'Exit Reason': 'End of period'
        })
        
        portfolio_history.append(portfolio_value)
        dates_history.append(common_idx[-1])  # ADD THIS
    
    df_trades = pd.DataFrame(trades)
    portfolio_series = pd.Series(portfolio_history, index=dates_history)  # CHANGE THIS LINE
    
    # Calculate metrics
    if len(df_trades) > 0:
        total_return = (portfolio_value - initial_capital) / initial_capital
        win_trades = df_trades[df_trades['P&L'] > 0]
        loss_trades = df_trades[df_trades['P&L'] <= 0]
        
        returns = portfolio_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        cummax = portfolio_series.expanding().max()
        drawdown = (portfolio_series - cummax) / cummax
        max_drawdown = drawdown.min()
        
        metrics = {
            'Total Return': total_return,
            'Final Portfolio': portfolio_value,
            'Num Trades': len(df_trades),
            'Win Rate': len(win_trades) / len(df_trades) if len(df_trades) > 0 else 0,
            'Avg Win': win_trades['P&L'].mean() if len(win_trades) > 0 else 0,
            'Avg Loss': loss_trades['P&L'].mean() if len(loss_trades) > 0 else 0,
            'Profit Factor': abs(win_trades['P&L'].sum() / loss_trades['P&L'].sum()) if len(loss_trades) > 0 and loss_trades['P&L'].sum() != 0 else np.inf,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Avg Return per Trade': df_trades['Return'].mean(),
            'Win/Loss Ratio': len(win_trades) / max(len(loss_trades), 1)
        }
    else:
        metrics = None
    
    return df_trades, portfolio_series, metrics

# ==================== TRADE SIGNALS ====================
def generate_trade_signals(spike_prob, vix_current, threshold_high=0.6, threshold_low=0.3):
    """Generate actionable trade signals"""
    signals = []
    
    if spike_prob > threshold_high:
        signals.append({
            'Action': 'BUY',
            'Instrument': 'VIX Calls',
            'Strikes': f'{vix_current + 5:.0f} - {vix_current + 10:.0f}',
            'Expiry': '30-45 DTE',
            'Rationale': f'High spike probability ({spike_prob:.1%})',
            'Position_Size': 'Defensive hedge: 10-15% of portfolio vega',
            'Stop_Loss': f'50% premium loss or VIX < {vix_current * 0.9:.1f}'
        })
        
        signals.append({
            'Action': 'BUY',
            'Instrument': 'SPX Put Spread',
            'Strikes': f'5-10% OTM',
            'Expiry': '30-60 DTE',
            'Rationale': 'Portfolio insurance against equity drawdown',
            'Position_Size': '2-3% of portfolio notional',
            'Stop_Loss': 'Time decay if VIX drops below 15'
        })
        
        signals.append({
            'Action': 'REDUCE',
            'Instrument': 'Net Delta Exposure',
            'Strikes': 'N/A',
            'Expiry': 'N/A',
            'Rationale': 'Decrease directional risk',
            'Position_Size': 'Reduce delta by 20-30%',
            'Stop_Loss': 'Re-evaluate if spike prob < 0.4'
        })
        
    elif spike_prob > threshold_low:
        signals.append({
            'Action': 'BUY',
            'Instrument': 'VIX Call Spreads',
            'Strikes': f'{vix_current + 3:.0f}/{vix_current + 8:.0f}',
            'Expiry': '30-45 DTE',
            'Rationale': f'Moderate spike risk ({spike_prob:.1%}), limited cost',
            'Position_Size': '5-8% of portfolio vega',
            'Stop_Loss': f'75% premium loss'
        })
        
        signals.append({
            'Action': 'MONITOR',
            'Instrument': 'VIX Term Structure',
            'Strikes': 'N/A',
            'Expiry': 'N/A',
            'Rationale': 'Watch for backwardation signals',
            'Position_Size': 'N/A',
            'Stop_Loss': 'N/A'
        })
        
    else:
        signals.append({
            'Action': 'SELL',
            'Instrument': 'VIX Call Credit Spreads',
            'Strikes': f'{vix_current + 10:.0f}/{vix_current + 15:.0f}',
            'Expiry': '20-30 DTE',
            'Rationale': f'Low spike probability ({spike_prob:.1%}), collect premium',
            'Position_Size': 'Max 5% of portfolio risk',
            'Stop_Loss': f'Close if VIX > {vix_current + 5:.1f}'
        })
        
        signals.append({
            'Action': 'BUY',
            'Instrument': 'Equity Dips',
            'Strikes': 'N/A',
            'Expiry': 'N/A',
            'Rationale': 'Low volatility environment favors longs',
            'Position_Size': 'Normal allocation',
            'Stop_Loss': 'Re-evaluate if spike prob > 0.4'
        })
    
    return pd.DataFrame(signals)

def calculate_kelly_criterion(prob, win_amount, loss_amount):
    """Calculate Kelly optimal position size"""
    if loss_amount == 0:
        return 0
    b = win_amount / loss_amount
    q = 1 - prob
    kelly = (prob * b - q) / b
    
    return max(0, min(kelly * 0.25, 0.25))

# ==================== SCENARIO SIMULATION ====================
def simulate_scenario(base_features, perturbations, model, scaler):
    """Simulate what-if scenarios"""
    scenario_features = base_features.copy()
    for feature, change in perturbations.items():
        if feature in scenario_features.index:
            scenario_features[feature] += change
    
    X_scaled = scaler.transform(scenario_features.values.reshape(1, -1))
    prob = model.predict_proba(X_scaled)[0, 1] if hasattr(model, 'predict_proba') else model.predict(X_scaled)[0, 1]
    return prob

# ==================== VISUALIZATION FUNCTIONS ====================
def plot_regime_analysis(vix_data, regimes, regime_names, probabilities):
    """Plot VIX with regime overlay"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                       subplot_titles=('VIX with Volatility Regimes', 'Spike Probability by Regime'),
                       row_heights=[0.6, 0.4])
    
    colors = ['#4CAF50', '#FFA500', '#FF6B6B']
    for i, regime_name in enumerate(regime_names):
        regime_mask = regimes == i
        regime_dates = vix_data.index[regime_mask]
        fig.add_trace(go.Scatter(x=regime_dates, y=vix_data[regime_mask], mode='markers',
                                name=regime_name, marker=dict(color=colors[i], size=3)), row=1, col=1)
    
    for i, regime_name in enumerate(regime_names):
        regime_mask = regimes == i
        regime_probs = probabilities[regime_mask]
        if len(regime_probs) > 0:
            fig.add_trace(go.Box(y=regime_probs, name=regime_name, marker=dict(color=colors[i]),
                                showlegend=False), row=2, col=1)
    
    fig.update_layout(template='plotly_dark', height=700, showlegend=True, hovermode='x unified',
                     paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2e3340')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2e3340')
    return fig

def plot_volatility_surface(vix_futures):
    """Plot VIX futures term structure"""
    if not vix_futures:
        return None
    
    months = list(range(1, len(vix_futures) + 1))
    futures_prices = list(vix_futures.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=futures_prices, mode='lines+markers',
                            name='VIX Futures', line=dict(color='#00D9FF', width=3),
                            marker=dict(size=10)))
    
    if len(futures_prices) >= 2:
        slope = futures_prices[1] - futures_prices[0]
        structure = "Contango" if slope > 0 else "Backwardation"
        color = '#4CAF50' if slope > 0 else '#FF6B6B'
        
        fig.add_annotation(x=len(months)/2, y=max(futures_prices),
                          text=f"{structure}: {slope:+.2f} pts",
                          showarrow=False, font=dict(size=16, color=color))
    
    fig.update_layout(title='VIX Futures Term Structure', xaxis_title='Contract Month',
                     yaxis_title='Futures Price', template='plotly_dark', height=500,
                     paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    return fig

def plot_2d_heatmap(X, y_prob, feature1, feature2, bins=20):
    """Plot 2D probability heatmap"""
    f1_bins = pd.qcut(X[feature1], bins, duplicates='drop')
    f2_bins = pd.qcut(X[feature2], bins, duplicates='drop')
    df_temp = pd.DataFrame({'f1_bin': f1_bins, 'f2_bin': f2_bins, 'probability': y_prob})
    pivot = df_temp.groupby(['f1_bin', 'f2_bin'])['probability'].mean().unstack()
    
    f1_centers = [interval.mid for interval in pivot.index]
    f2_centers = [interval.mid for interval in pivot.columns]
    
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=f2_centers, y=f1_centers,
                                    colorscale='RdYlGn_r', zmid=0.5, colorbar=dict(title="Spike Probability")))
    fig.update_layout(title=f'Spike Probability: {feature1} vs {feature2}', xaxis_title=feature2,
                     yaxis_title=feature1, template='plotly_dark', height=600,
                     paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    return fig

def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """Plot calibration curve"""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration',
                            line=dict(color='white', dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Model Calibration',
                            line=dict(color='#00D9FF', width=3), marker=dict(size=10)))
    
    fig.update_layout(title='Probability Calibration', xaxis_title='Predicted Probability',
                     yaxis_title='Observed Frequency', template='plotly_dark', height=500,
                     paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2e3340')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2e3340')
    return fig

def plot_cross_asset_correlation(df, window=60):
    """Plot rolling correlations"""
    vol_indices = [col for col in df.columns if col in ['VIX', 'VVIX', 'VXV']]
    if len(vol_indices) < 2: return None
    
    fig = go.Figure()
    colors = ['#00D9FF', '#FF6B6B', '#4CAF50', '#FFA500', '#9C27B0']
    
    for i, col in enumerate(vol_indices[1:]):
        corr = df['VIX'].rolling(window).corr(df[col])
        fig.add_trace(go.Scatter(x=corr.index, y=corr.values, mode='lines',
                                name=f'VIX vs {col}', line=dict(color=colors[i % len(colors)], width=2)))
    
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.update_layout(title=f'Rolling {window}-Day Correlations', xaxis_title='Date', yaxis_title='Correlation',
                     template='plotly_dark', height=500, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2e3340')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2e3340')
    return fig

def plot_threshold_optimization(df_costs):
    """Plot cost curve for threshold optimization"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=('Total Cost by Threshold', 'Precision vs Recall'))
    
    fig.add_trace(go.Scatter(x=df_costs['threshold'], y=df_costs['cost'],
                            mode='lines', name='Total Cost', line=dict(color='#FF6B6B', width=3)),
                 row=1, col=1)
    
    optimal_thresh = df_costs.loc[df_costs['cost'].idxmin(), 'threshold']
    optimal_cost = df_costs['cost'].min()
    fig.add_vline(x=optimal_thresh, line_dash="dash", line_color="#00D9FF", 
                 annotation_text=f"Optimal: {optimal_thresh:.2f}", row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_costs['threshold'], y=df_costs['precision'],
                            mode='lines', name='Precision', line=dict(color='#4CAF50', width=2)),
                 row=2, col=1)
    fig.add_trace(go.Scatter(x=df_costs['threshold'], y=df_costs['recall'],
                            mode='lines', name='Recall', line=dict(color='#FFA500', width=2)),
                 row=2, col=1)
    
    fig.update_layout(template='plotly_dark', height=700, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    fig.update_xaxes(title_text="Threshold", row=2, col=1)
    fig.update_yaxes(title_text="Cost", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    return fig

def plot_interactive_scenario(X, model, scaler, feature_impacts):
    """Plot real-time scenario analysis"""
    fig = go.Figure()
    
    features = list(feature_impacts.keys())
    impacts = list(feature_impacts.values())
    colors = ['#FF6B6B' if imp > 0 else '#4CAF50' for imp in impacts]
    
    fig.add_trace(go.Bar(x=impacts, y=features, orientation='h',
                        marker=dict(color=colors)))
    
    fig.update_layout(title='Feature Impact on Probability',
                     xaxis_title='Probability Change (%)',
                     yaxis_title='Feature',
                     template='plotly_dark', height=600,
                     paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    return fig

def plot_backtest_results(portfolio_values, trades_df):
    """Plot backtest equity curve and trade markers"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                       subplot_titles=('Portfolio Value', 'Trade Returns'),
                       row_heights=[0.7, 0.3])
    
    # Portfolio value
    fig.add_trace(go.Scatter(x=portfolio_values.index, y=portfolio_values.values,
                            mode='lines', name='Portfolio', 
                            line=dict(color='#00D9FF', width=2)),
                 row=1, col=1)
    
    # Mark trades
    if len(trades_df) > 0:
        long_trades = trades_df[trades_df['Type'] == 'LONG']
        short_trades = trades_df[trades_df['Type'] == 'SHORT']
        
        for _, trade in long_trades.iterrows():
            fig.add_vline(x=trade['Entry Date'], line_dash="dot", line_color="green", 
                         opacity=0.5, row=1, col=1)
        
        for _, trade in short_trades.iterrows():
            fig.add_vline(x=trade['Entry Date'], line_dash="dot", line_color="red", 
                         opacity=0.5, row=1, col=1)
        
        # Trade returns
        fig.add_trace(go.Bar(x=trades_df['Exit Date'], y=trades_df['Return'],
                            marker_color=['green' if r > 0 else 'red' for r in trades_df['Return']],
                            name='Trade Returns'),
                     row=2, col=1)
    
    fig.update_layout(template='plotly_dark', height=800, showlegend=True,
                     paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2e3340')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2e3340')
    
    return fig

def main():
    st.title("VIX Risk Desk Trading Dashboard v2.0")
    st.markdown("### Advanced Volatility Analysis with GARCH, LSTM & Trade Signals")
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Data Parameters")
        start_date = st.date_input("Start Date", value=datetime(2010, 1, 1))
        end_date = st.date_input("End Date", value=datetime.today())
        
        st.subheader("Spike Threshold Analysis")
        vix_thresholds = st.multiselect("Test Multiple Thresholds", 
                                       [15.0, 17.5, 20.0, 22.5, 25.0, 30.0],
                                       default=[17.5, 20.0, 25.0])
        primary_threshold = st.selectbox("Primary Threshold", vix_thresholds, index=0 if vix_thresholds else 0)
        
        st.subheader("Feature Selection")
        with st.expander("Core Features", expanded=False):
            feature_config = {
                'vix_lag_5': st.checkbox("VIX Lag 5", True),
                'vix_lag_22': st.checkbox("VIX Lag 22", True),
                'vix_momentum_5': st.checkbox("VIX Momentum 5D", True),
                'vix_momentum_22': st.checkbox("VIX Momentum 22D", True),
                'vix_zscore': st.checkbox("VIX Z-Score", True),
                'vix_realized_vol': st.checkbox("VIX Realized Vol", True),
                'vix_ma_ratio': st.checkbox("VIX MA Ratio", True),
                'vix_regime': st.checkbox("VIX Regime", True),
            }
        
        with st.expander("Advanced Features", expanded=False):
            feature_config.update({
                'term_structure': st.checkbox("Term Structure", True),
                'vvix': st.checkbox("VVIX", True),
                'spy_return': st.checkbox("SPY Returns", True),
                'spy_vol': st.checkbox("SPY Vol", True),
                'credit': st.checkbox("Credit (HYG)", True),
                'dollar': st.checkbox("Dollar Index", False),
                'gold': st.checkbox("Gold", False),
                'tlt_change': st.checkbox("TLT", True),
            })
        
        with st.expander("Enhanced Features", expanded=True):
            feature_config.update({
                'garch': st.checkbox("GARCH Volatility", True),
                'microstructure': st.checkbox("Microstructure", True),
                'polynomial': st.checkbox("Polynomial Features", True),
            })
        
        st.subheader("Model Parameters")
        selected_models = st.multiselect("Models to Compare", 
                                        ['Logistic', 'XGBoost', 'RandomForest', 
                                         'Logistic_Calibrated', 'XGBoost_Calibrated'],
                                        default=['Logistic', 'XGBoost_Calibrated'])
        
        enable_lstm = st.checkbox("Enable LSTM Model", True)
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        vif_threshold = st.slider("VIF Threshold", 5, 20, 10)
        
        st.subheader("Portfolio Parameters")
        portfolio_delta = st.number_input("Portfolio Delta ($)", value=1000000, step=100000)
        portfolio_vega = st.number_input("Portfolio Vega ($)", value=50000, step=10000)
        portfolio_gamma = st.number_input("Portfolio Gamma ($)", value=10000, step=5000)
        
        st.subheader("Backtest Parameters")
        backtest_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
        backtest_position_size = st.slider("Position Size (%)", 5, 30, 10) / 100
        threshold_high = st.slider("Buy Threshold", 0.4, 0.9, 0.6, 0.05)
        threshold_low = st.slider("Sell Threshold", 0.1, 0.5, 0.3, 0.05)
        
        st.subheader("Cost Parameters")
        fp_cost = st.number_input("False Positive Cost (units)", value=1.0, step=0.5)
        fn_cost = st.number_input("False Negative Cost (units)", value=5.0, step=0.5)
        
        st.subheader("Analysis Options")
        enable_regimes = st.checkbox("Enable Regime Detection", True)
        enable_rolling = st.checkbox("Enable Rolling Validation", True)
        enable_iv_surface = st.checkbox("Enable IV Surface Analysis", True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)
        with col2:
            if st.session_state.analysis_complete:
                rerun_backtest = st.button("Update Backtest", use_container_width=True)
            else:
                rerun_backtest = False
    
    # Full analysis run
    if run_analysis:
        st.session_state.analysis_complete = False
        
        with st.spinner("Loading market data..."):
            df = fetch_cross_asset_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            vix_futures = fetch_vix_futures()
        st.success(f"✓ Data loaded: {len(df)} trading days")
        
        with st.spinner("Building features..."):
            X = build_advanced_features(df, feature_config)
            X = remove_multicollinearity(X, vif_threshold)
        st.success(f"✓ Features: {len(X.columns)} after multicollinearity removal")
        
        targets = {}
        for threshold in vix_thresholds:
            targets[threshold] = (df['VIX'].loc[X.index] > threshold).astype(int)
        y = targets[primary_threshold]
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        with st.spinner("Training models..."):
            all_models, scaler = fit_multiple_models(X_train, y_train)
            models = {k: v for k, v in all_models.items() if k in selected_models}
        st.success(f"✓ Models trained: {', '.join(models.keys())}")
        
        lstm_model = None
        if enable_lstm:
            with st.spinner("Training LSTM model..."):
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                lstm_model, lstm_train_probs, lstm_test_probs = train_lstm_model(
                    pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
                    y_train,
                    pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index),
                    y_test,
                    lookback=20,
                    epochs=30
                )
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model_probs = {}
        for name, model in models.items():
            train_probs = model.predict_proba(X_train_scaled)[:, 1]
            test_probs = model.predict_proba(X_test_scaled)[:, 1]
            model_probs[name] = {
                'train': train_probs, 
                'test': test_probs,
                'all': np.concatenate([train_probs, test_probs])
            }
        
        if enable_lstm and lstm_model is not None:
            lookback = 20
            all_probs = np.concatenate([lstm_train_probs, lstm_test_probs])
            all_idx = X.index[lookback + split_idx - len(X_train):]
            
            model_probs['LSTM'] = {
                'train': lstm_train_probs,
                'test': lstm_test_probs,
                'all': all_probs,
                'index': all_idx
            }
        
        if enable_regimes:
            with st.spinner("Detecting volatility regimes..."):
                regimes, regime_names = detect_regimes_hmm(df['VIX'].loc[X.index])
                st.session_state.regimes = regimes
                st.session_state.regime_names = regime_names
        
        primary_model = list(models.keys())[0]
        optimal_threshold, df_costs = optimize_threshold(y_test, model_probs[primary_model]['test'], 
                                                        fp_cost, fn_cost)
        
        with st.spinner("Running strategy backtest..."):
            probs_series = pd.Series(model_probs[primary_model]['all'], index=X.index)
            vix_series = df['VIX'].loc[X.index]
            trades_df, portfolio_values, backtest_metrics = backtest_vix_strategy(
                vix_series, probs_series, 
                threshold_high=threshold_high, 
                threshold_low=threshold_low,
                initial_capital=backtest_capital,
                position_size=backtest_position_size
            )
        
        # Store in session state
        st.session_state.df = df
        st.session_state.vix_futures = vix_futures
        st.session_state.X = X
        st.session_state.models = models
        st.session_state.scaler = scaler
        st.session_state.model_probs = model_probs
        st.session_state.targets = targets
        st.session_state.y = y
        st.session_state.y_test = y_test
        st.session_state.split_idx = split_idx
        st.session_state.primary_threshold = primary_threshold
        st.session_state.optimal_threshold = optimal_threshold
        st.session_state.df_costs = df_costs
        st.session_state.primary_model = primary_model
        st.session_state.trades_df = trades_df
        st.session_state.portfolio_values = portfolio_values
        st.session_state.backtest_metrics = backtest_metrics
        st.session_state.enable_regimes = enable_regimes
        st.session_state.portfolio_delta = portfolio_delta
        st.session_state.portfolio_vega = portfolio_vega
        st.session_state.portfolio_gamma = portfolio_gamma
        st.session_state.threshold_high = threshold_high
        st.session_state.threshold_low = threshold_low
        st.session_state.analysis_complete = True
        
        st.success("✓ Analysis complete!")
        st.rerun()
    
    # Backtest update only
    elif rerun_backtest and st.session_state.analysis_complete:
        with st.spinner("Updating backtest..."):
            df = st.session_state.df
            X = st.session_state.X
            primary_model = st.session_state.primary_model
            model_probs = st.session_state.model_probs
            
            probs_series = pd.Series(model_probs[primary_model]['all'], index=X.index)
            vix_series = df['VIX'].loc[X.index]
            
            trades_df, portfolio_values, backtest_metrics = backtest_vix_strategy(
                vix_series, probs_series,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                initial_capital=backtest_capital,
                position_size=backtest_position_size
            )
            
            st.session_state.trades_df = trades_df
            st.session_state.portfolio_values = portfolio_values
            st.session_state.backtest_metrics = backtest_metrics
            st.session_state.threshold_high = threshold_high
            st.session_state.threshold_low = threshold_low
        
        st.success("✓ Backtest updated!")
    
    # Display results
    if st.session_state.analysis_complete:
        df = st.session_state.df
        vix_futures = st.session_state.vix_futures
        X = st.session_state.X
        models = st.session_state.models
        scaler = st.session_state.scaler
        model_probs = st.session_state.model_probs
        targets = st.session_state.targets
        y = st.session_state.y
        y_test = st.session_state.y_test
        split_idx = st.session_state.split_idx
        primary_threshold = st.session_state.primary_threshold
        optimal_threshold = st.session_state.optimal_threshold
        df_costs = st.session_state.df_costs
        primary_model = st.session_state.primary_model
        trades_df = st.session_state.trades_df
        portfolio_values = st.session_state.portfolio_values
        backtest_metrics = st.session_state.backtest_metrics
        enable_regimes = st.session_state.enable_regimes
        
        tabs = st.tabs([
            "Overview", 
            "Threshold Optimization",
            "Regime Analysis", 
            "Model Comparison",
            "Feature Analysis", 
            "Interactive Simulator", 
            "Portfolio & Greeks",
            "Trade Signals",
            "Cross-Asset",
            "Volatility Surface",
            "VIX Options IV Surface"
        ])
        
        with tabs[0]:
            st.header("Market Overview & Real-Time Greeks")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            current_vix = df['VIX'].iloc[-1]
            prev_vix = df['VIX'].iloc[-2]
            vix_change = ((current_vix - prev_vix) / prev_vix) * 100
            current_prob = model_probs[primary_model]['all'][-1]
            
            with col1:
                st.metric("Current VIX", f"{current_vix:.2f}", f"{vix_change:+.2f}%")
            with col2:
                st.metric("Spike Probability", f"{current_prob:.1%}", 
                         f"Threshold: {primary_threshold:.1f}")
            with col3:
                st.metric("Optimal Threshold", f"{optimal_threshold:.2f}")
            with col4:
                if enable_regimes and 'regimes' in st.session_state:
                    current_regime = st.session_state.regimes.iloc[-1]
                    st.metric("Current Regime", st.session_state.regime_names[current_regime])
            with col5:
                test_auc = roc_auc_score(y_test, model_probs[primary_model]['test'])
                st.metric("Model AUC", f"{test_auc:.3f}")
            
            st.markdown("---")
            st.subheader("Real-Time Portfolio Greeks & P&L")
            
            greeks = calculate_greeks(current_vix, current_prob, 
                                     st.session_state.portfolio_delta, 
                                     st.session_state.portfolio_vega, 
                                     st.session_state.portfolio_gamma)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Expected VIX Change", f"{greeks['Expected_VIX_Change']:+.2f} pts")
                st.metric("Delta Impact", f"${greeks['Delta_Impact']:,.0f}")
            with col2:
                st.metric("Vega Impact", f"${greeks['Vega_Impact']:,.0f}")
                st.metric("Gamma Impact", f"${greeks['Gamma_Impact']:,.0f}")
            with col3:
                pnl_color = "normal" if greeks['Total_PnL_Impact'] >= 0 else "inverse"
                st.metric("Total P&L Impact", f"${greeks['Total_PnL_Impact']:,.0f}",
                         delta_color=pnl_color)
                st.metric("VaR 95%", f"${greeks['VaR_95']:,.0f}")
            with col4:
                st.metric("CVaR 95%", f"${greeks['CVaR_95']:,.0f}")
                kelly_size = calculate_kelly_criterion(current_prob, 
                                                       abs(greeks['Vega_Impact']), 
                                                       abs(greeks['Delta_Impact']))
                st.metric("Kelly Position Size", f"{kelly_size:.1%}")
            
            st.markdown("---")
            st.subheader("Model Performance Summary")
            
            perf_data = []
            for name, probs in model_probs.items():
                if name == 'LSTM':
                    y_test_lstm = y_test.iloc[20:]
                    if len(probs['test']) == len(y_test_lstm):
                        metrics = calculate_model_metrics(y_test_lstm, probs['test'])
                    else:
                        continue
                else:
                    metrics = calculate_model_metrics(y_test, probs['test'])
                
                perf_data.append({
                    'Model': name,
                    'AUC': f"{metrics['AUC']:.3f}",
                    'F1': f"{metrics['F1']:.3f}",
                    'Precision': f"{metrics['Precision']:.3f}",
                    'Recall': f"{metrics['Recall']:.3f}",
                    'Brier': f"{metrics['Brier Score']:.3f}",
                    'MCC': f"{metrics['MCC']:.3f}"
                })
            
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
        
        with tabs[1]:
            st.header("Dynamic Threshold Optimization")
            
            st.markdown(f"""
            **Cost Function Analysis**: Optimizing decision threshold based on economic costs
            - False Positive Cost: **{fp_cost}** units
            - False Negative Cost: **{fn_cost}** units
            - **Optimal Threshold: {optimal_threshold:.3f}**
            """)
            
            fig = plot_threshold_optimization(df_costs)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Threshold Sensitivity Analysis")
            
            threshold_results = []
            for threshold in vix_thresholds:
                y_thresh = targets[threshold]
                y_test_thresh = y_thresh.iloc[split_idx:]
                
                for name, probs in model_probs.items():
                    if name == 'LSTM':
                        continue
                    metrics = calculate_model_metrics(y_test_thresh, probs['test'])
                    threshold_results.append({
                        'Threshold': threshold,
                        'Model': name,
                        **metrics
                    })
            
            df_thresh = pd.DataFrame(threshold_results)
            
            fig = go.Figure()
            for model_name in df_thresh['Model'].unique():
                model_data = df_thresh[df_thresh['Model'] == model_name]
                fig.add_trace(go.Scatter(x=model_data['Threshold'], y=model_data['AUC'],
                                        mode='lines+markers', name=model_name,
                                        line=dict(width=3), marker=dict(size=10)))
            
            fig.update_layout(title='Model AUC Across Thresholds', xaxis_title='VIX Threshold',
                             yaxis_title='AUC', template='plotly_dark', height=500,
                             paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Comprehensive Metrics by Threshold")
            st.dataframe(df_thresh.round(3), use_container_width=True)
        
        with tabs[2]:
            if enable_regimes and 'regimes' in st.session_state:
                st.header("Volatility Regime Analysis")
                
                regimes = st.session_state.regimes
                regime_names = st.session_state.regime_names
                
                primary_probs = pd.Series(model_probs[primary_model]['all'], index=X.index)
                fig = plot_regime_analysis(df['VIX'].loc[X.index], regimes, regime_names, primary_probs)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Regime Transition Matrix")
                transition_matrix = pd.crosstab(regimes.shift(1), regimes, normalize='index')
                transition_matrix.index = regime_names
                transition_matrix.columns = regime_names
                
                fig = go.Figure(data=go.Heatmap(z=transition_matrix.values,
                                                x=regime_names, y=regime_names,
                                                colorscale='Viridis', text=transition_matrix.values,
                                                texttemplate='%{text:.2%}', textfont=dict(size=14)))
                fig.update_layout(title='Regime Transition Probabilities', template='plotly_dark',
                                 height=500, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Enable regime detection in sidebar to see analysis")
        
        with tabs[3]:
            st.header("Model Comparison & Calibration")
            
            st.subheader("ROC Curves")
            fig = go.Figure()
            
            for name, probs in model_probs.items():
                if name == 'LSTM':
                    y_test_lstm = y_test.iloc[20:]
                    if len(probs['test']) == len(y_test_lstm):
                        fpr, tpr, _ = roc_curve(y_test_lstm, probs['test'])
                        auc = roc_auc_score(y_test_lstm, probs['test'])
                    else:
                        continue
                else:
                    fpr, tpr, _ = roc_curve(y_test, probs['test'])
                    auc = roc_auc_score(y_test, probs['test'])
                
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc:.3f})',
                                        line=dict(width=2)))
            
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                    line=dict(color='white', width=1, dash='dash')))
            
            fig.update_layout(title='ROC Curves - Model Comparison', xaxis_title='FPR', yaxis_title='TPR',
                             template='plotly_dark', height=600, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
            st.plotly_chart(fig, use_container_width=True)
            
            if enable_rolling:
                st.subheader("Rolling Window Validation")
                from sklearn.base import clone
                rolling_results = rolling_window_validation(X, y, 
                                                           clone(models[primary_model]),
                                                           StandardScaler())
                
                if len(rolling_results) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=rolling_results['Date'], y=rolling_results['AUC'],
                                            mode='lines', name='Rolling AUC', 
                                            line=dict(color='#00D9FF', width=2)))
                    fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.3)
                    
                    fig.update_layout(title='Rolling Window AUC (252-day window)', xaxis_title='Date',
                                     yaxis_title='AUC', template='plotly_dark', height=500,
                                     paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
                    st.plotly_chart(fig, use_container_width=True)
        
        with tabs[4]:
            st.header("Feature Attribution Analysis")
            
            st.subheader("Feature Importance by Model")
            
            importance_data = []
            for name, model in models.items():
                if 'Calibrated' in name:
                    continue
                
                if hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])
                elif hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                else:
                    continue
                
                for feat, imp in zip(X.columns, importance):
                    importance_data.append({'Model': name, 'Feature': feat, 'Importance': imp})
            
            df_imp = pd.DataFrame(importance_data)
            
            if len(df_imp) > 0:
                top_features = df_imp.groupby('Feature')['Importance'].mean().nlargest(15).index
                df_imp_filtered = df_imp[df_imp['Feature'].isin(top_features)]
                
                fig = px.bar(df_imp_filtered, x='Importance', y='Feature', color='Model',
                            barmode='group', orientation='h')
                fig.update_layout(title='Top 15 Features by Importance', template='plotly_dark',
                                 height=600, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("2D Probability Surfaces")
            
            if len(X.columns) >= 2:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    feature1 = st.selectbox("Feature 1 (Y-axis)", X.columns, index=0, key='f1_heat')
                    feature2 = st.selectbox("Feature 2 (X-axis)", X.columns, 
                                          index=min(1, len(X.columns)-1), key='f2_heat')
                    bins = st.slider("Number of Bins", 10, 30, 20, key='bins_heat')
                
                with col2:
                    primary_probs_series = pd.Series(model_probs[primary_model]['all'], index=X.index)
                    fig = plot_2d_heatmap(X, primary_probs_series, feature1, feature2, bins)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tabs[5]:
            st.header("Interactive Scenario Simulation Engine")
            
            col1, col2 = st.columns([1, 1])
            
            current_features = X.iloc[-1]
            base_prob = model_probs[primary_model]['all'][-1]
            
            with col1:
                st.subheader("Current Market State")
                for feat in X.columns[:8]:
                    st.metric(feat, f"{current_features[feat]:.4f}")
            
            with col2:
                st.subheader("Scenario Builder")
                perturbations = {}
                feature_impacts = {}
                
                for feat in X.columns[:8]:
                    current_val = current_features[feat]
                    std_val = X[feat].std()
                    
                    change = st.slider(
                        f"{feat}",
                        min_value=-3.0 * std_val,
                        max_value=3.0 * std_val,
                        value=0.0,
                        step=0.1 * std_val,
                        format="%.4f",
                        key=f'sim_{feat}'
                    )
                    
                    if change != 0:
                        perturbations[feat] = change
                        test_perturb = {feat: change}
                        marginal_prob = simulate_scenario(current_features, test_perturb, 
                                                         models[primary_model], scaler)
                        feature_impacts[feat] = (marginal_prob - base_prob) * 100
            
            st.markdown("---")
            
            col_a, col_b, col_c = st.columns(3)
            
            if len(perturbations) > 0:
                scenario_prob = simulate_scenario(current_features, perturbations, 
                                                 models[primary_model], scaler)
            else:
                scenario_prob = base_prob
            
            with col_a:
                st.metric("Base Probability", f"{base_prob:.2%}")
            with col_b:
                st.metric("Scenario Probability", f"{scenario_prob:.2%}")
            with col_c:
                change = scenario_prob - base_prob
                st.metric("Change", f"{change:+.2%}")
            
            if len(feature_impacts) > 0:
                st.subheader("Feature Impact Analysis")
                fig = plot_interactive_scenario(X, models[primary_model], scaler, feature_impacts)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[6]:
            st.header("Portfolio Stress Testing & Greeks Dashboard")
            
            current_prob = model_probs[primary_model]['all'][-1]
            current_vix = df['VIX'].iloc[-1]
            
            st.subheader("Current Portfolio Exposure")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Portfolio Delta", f"${st.session_state.portfolio_delta:,.0f}")
            with col2:
                st.metric("Portfolio Vega", f"${st.session_state.portfolio_vega:,.0f}")
            with col3:
                st.metric("Portfolio Gamma", f"${st.session_state.portfolio_gamma:,.0f}")
            with col4:
                net_exposure = abs(st.session_state.portfolio_delta) + abs(st.session_state.portfolio_vega)
                st.metric("Net Exposure", f"${net_exposure:,.0f}")
            
            st.markdown("---")
            st.subheader("Stress Scenarios")
            
            scenarios = {
                'Severe Crash (-30% SPY)': {'prob': 0.95, 'spy_move': -0.30, 'vix_target': 50},
                'Market Crash (-20% SPY)': {'prob': 0.85, 'spy_move': -0.20, 'vix_target': 40},
                'High Volatility': {'prob': 0.65, 'spy_move': -0.10, 'vix_target': 32},
                'Elevated Risk': {'prob': 0.45, 'spy_move': -0.05, 'vix_target': 25},
                'Normal': {'prob': 0.25, 'spy_move': 0.0, 'vix_target': 18},
                'Low Volatility': {'prob': 0.10, 'spy_move': 0.05, 'vix_target': 12}
            }
            
            scenario_results = []
            for scenario_name, params in scenarios.items():
                spike_prob = params['prob']
                vix_change = params['vix_target'] - current_vix
                
                delta_impact = st.session_state.portfolio_delta * params['spy_move']
                gamma_impact = 0.5 * st.session_state.portfolio_gamma * (params['spy_move'] ** 2)
                vega_impact = st.session_state.portfolio_vega * vix_change * spike_prob
                total_pnl = delta_impact + gamma_impact + vega_impact
                
                scenario_results.append({
                    'Scenario': scenario_name,
                    'Spike Prob': f"{spike_prob:.0%}",
                    'SPY Move': f"{params['spy_move']:+.1%}",
                    'VIX Target': f"{params['vix_target']:.0f}",
                    'Delta P&L': f"${delta_impact:,.0f}",
                    'Gamma P&L': f"${gamma_impact:,.0f}",
                    'Vega P&L': f"${vega_impact:,.0f}",
                    'Total P&L': f"${total_pnl:,.0f}",
                    'VaR 95%': f"${total_pnl * 1.645:,.0f}"
                })
            
            df_scenarios = pd.DataFrame(scenario_results)
            st.dataframe(df_scenarios, use_container_width=True)
        
        with tabs[7]:
            st.header("Automated Trade Signal Generator & Backtest")
            
            current_prob = model_probs[primary_model]['all'][-1]
            current_vix = df['VIX'].iloc[-1]
            
            st.subheader("Current Trading Signals")
            
            signals_df = generate_trade_signals(current_prob, current_vix, 
                                               threshold_high=st.session_state.threshold_high, 
                                               threshold_low=st.session_state.threshold_low)
            
            for _, signal in signals_df.iterrows():
                color = '#FF6B6B' if signal['Action'] == 'BUY' else '#4CAF50' if signal['Action'] == 'SELL' else '#FFA500'
                
                st.markdown(f"""
                <div class='trade-signal'>
                <h3 style='color: {color};'>{signal['Action']} {signal['Instrument']}</h3>
                <p><strong>Strikes:</strong> {signal['Strikes']}</p>
                <p><strong>Expiry:</strong> {signal['Expiry']}</p>
                <p><strong>Rationale:</strong> {signal['Rationale']}</p>
                <p><strong>Position Size:</strong> {signal['Position_Size']}</p>
                <p><strong>Stop Loss:</strong> {signal['Stop_Loss']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Strategy Backtest Results")
            
            if backtest_metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{backtest_metrics['Total Return']:.2%}")
                    st.metric("Final Portfolio", f"${backtest_metrics['Final Portfolio']:,.0f}")
                with col2:
                    st.metric("Number of Trades", f"{backtest_metrics['Num Trades']:.0f}")
                    st.metric("Win Rate", f"{backtest_metrics['Win Rate']:.1%}")
                with col3:
                    st.metric("Avg Win", f"${backtest_metrics['Avg Win']:,.0f}")
                    st.metric("Avg Loss", f"${backtest_metrics['Avg Loss']:,.0f}")
                with col4:
                    st.metric("Sharpe Ratio", f"{backtest_metrics['Sharpe Ratio']:.2f}")
                    st.metric("Max Drawdown", f"{backtest_metrics['Max Drawdown']:.1%}")
                
                st.markdown("---")
                st.subheader("Equity Curve & Trades")
                fig = plot_backtest_results(portfolio_values, trades_df)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Trade Log")
                if len(trades_df) > 0:
                    display_trades = trades_df.copy()
                    display_trades['Entry VIX'] = display_trades['Entry VIX'].round(2)
                    display_trades['Exit VIX'] = display_trades['Exit VIX'].round(2)
                    display_trades['P&L'] = display_trades['P&L'].round(0)
                    display_trades['Return'] = (display_trades['Return'] * 100).round(2)
                    st.dataframe(display_trades, use_container_width=True)
            else:
                st.info("No trades executed in backtest period")
        
        with tabs[8]:
            st.header("Cross-Asset Volatility Analysis")
            
            corr_cols = [col for col in df.columns if col in ['VIX', 'VVIX', 'VXV', 'SPY', 'TLT', 'HYG', 'GLD']]
            if len(corr_cols) > 1:
                recent_data = df[corr_cols].iloc[-252:]
                corr_matrix = recent_data.corr()
                
                fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns,
                                                y=corr_matrix.columns, colorscale='RdBu', zmid=0,
                                                text=corr_matrix.values, texttemplate='%{text:.2f}',
                                                textfont=dict(size=12)))
                fig.update_layout(title='1-Year Rolling Correlation Matrix', template='plotly_dark',
                                 height=600, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
                st.plotly_chart(fig, use_container_width=True)
            
            fig = plot_cross_asset_correlation(df.loc[X.index], window=60)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[9]:
            st.header("VIX Futures Term Structure")
            
            if vix_futures and len(vix_futures) > 0:
                fig = plot_volatility_surface(vix_futures)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("VIX futures data unavailable")
        
        with tabs[10]:
            st.header("VIX Options Implied Volatility Surface")
            
            if enable_iv_surface:
                with st.spinner("Fetching VIX options data..."):
                    df_options = fetch_vix_options()
                
                if df_options is not None and len(df_options) > 0:
                    with st.spinner("Calculating IV surface..."):
                        df_iv = calculate_iv_surface(df_options, current_vix)
                    
                    if df_iv is not None and len(df_iv) > 0:
                        fig_3d = plot_iv_surface_3d(df_iv)
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True)
                        
                        fig_term = plot_iv_term_structure(df_iv)
                        if fig_term:
                            st.plotly_chart(fig_term, use_container_width=True)
                    else:
                        st.warning("Unable to calculate IV")
                else:
                    st.warning("No VIX options data available")
            else:
                st.info("Enable IV Surface Analysis in sidebar")
    
    else:
        st.info("Configure parameters and click 'Run Analysis' to begin")

if __name__ == '__main__':
    main()