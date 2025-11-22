"""
VIX Professional Trading Dashboard
Advanced volatility analysis with Heston model, Markov Chains, and econometric forecasting
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import RBFInterpolator
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm
from arch import arch_model
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="VIX Professional Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Dark Theme
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
    .signal-box {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00D9FF;
        margin: 10px 0;
    }
    .model-header {
        font-size: 14px;
        font-weight: 600;
        color: #00D9FF;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HESTON MODEL IMPLEMENTATION ====================

class HestonModel:
    """
    Heston Stochastic Volatility Model for VIX options pricing

    dS_t = μS_t dt + √v_t S_t dW_t^S
    dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v

    Parameters:
    - v0: Initial variance
    - kappa: Mean reversion speed
    - theta: Long-term variance
    - sigma: Volatility of volatility
    - rho: Correlation between asset and variance
    """

    def __init__(self, S0, v0, kappa, theta, sigma, rho, r=0.05):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r

    def characteristic_function(self, u, T):
        """Heston characteristic function"""
        kappa, theta, sigma, rho, v0, r = self.kappa, self.theta, self.sigma, self.rho, self.v0, self.r

        # Parameters
        d = np.sqrt((rho * sigma * u * 1j - kappa)**2 + sigma**2 * (u * 1j + u**2))
        g = (kappa - rho * sigma * u * 1j - d) / (kappa - rho * sigma * u * 1j + d)

        # Characteristic function
        C = (r * u * 1j * T + (kappa * theta) / (sigma**2) *
             ((kappa - rho * sigma * u * 1j - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))

        D = ((kappa - rho * sigma * u * 1j - d) / sigma**2 *
             ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))))

        return np.exp(C + D * v0 + 1j * u * np.log(self.S0))

    def option_price(self, K, T, option_type='call'):
        """Price European option using Heston model via FFT"""
        # Integration bounds
        N = 2**12
        dv = 0.01
        v_max = N * dv

        # Grid
        v = np.arange(0, v_max, dv)
        k = np.log(K)

        # Integrand for P1
        integrand1 = np.real(np.exp(-1j * v * k) * self.characteristic_function(v - 1j, T) /
                            (1j * v * self.characteristic_function(-1j, T)))

        # Integrand for P2
        integrand2 = np.real(np.exp(-1j * v * k) * self.characteristic_function(v, T) / (1j * v))

        # Integration
        P1 = 0.5 + (1/np.pi) * np.sum(integrand1) * dv
        P2 = 0.5 + (1/np.pi) * np.sum(integrand2) * dv

        # Option price
        if option_type == 'call':
            price = self.S0 * P1 - K * np.exp(-self.r * T) * P2
        else:
            price = K * np.exp(-self.r * T) * (1 - P2) - self.S0 * (1 - P1)

        return max(price, 0)

    def simulate_paths(self, T, n_steps, n_paths):
        """Monte Carlo simulation of Heston model paths"""
        dt = T / n_steps

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = self.S0
        v[:, 0] = self.v0

        for t in range(1, n_steps + 1):
            Z1 = np.random.randn(n_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.randn(n_paths)

            v[:, t] = np.maximum(v[:, t-1] + self.kappa * (self.theta - v[:, t-1]) * dt +
                                 self.sigma * np.sqrt(np.maximum(v[:, t-1], 0)) * np.sqrt(dt) * Z2, 0)

            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * v[:, t-1]) * dt +
                                         np.sqrt(np.maximum(v[:, t-1], 0)) * np.sqrt(dt) * Z1)

        return S, v

    @staticmethod
    def calibrate_to_surface(market_prices, S0, strikes, maturities, r=0.05):
        """Calibrate Heston parameters to market option prices"""

        def objective(params):
            v0, kappa, theta, sigma, rho = params

            # Parameter constraints
            if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
                return 1e10

            # Feller condition
            if 2 * kappa * theta < sigma**2:
                return 1e10

            heston = HestonModel(S0, v0, kappa, theta, sigma, rho, r)

            total_error = 0
            for (K, T, market_price) in market_prices:
                try:
                    model_price = heston.option_price(K, T, 'call')
                    total_error += (model_price - market_price)**2
                except:
                    total_error += 1e8

            return total_error

        # Initial guess
        bounds = [(0.01, 1), (0.1, 10), (0.01, 1), (0.1, 2), (-0.9, 0.9)]

        result = differential_evolution(objective, bounds, seed=42, maxiter=100, workers=1)

        return result.x

# ==================== EXOTIC OPTIONS PRICING ====================

class ExoticOptions:
    """Exotic VIX options pricing using numpy"""

    @staticmethod
    def asian_option_price(S_paths, K, option_type='call'):
        """Asian option - average price"""
        avg_prices = np.mean(S_paths, axis=1)

        if option_type == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)

        return np.mean(payoffs)

    @staticmethod
    def barrier_option_price(S_paths, K, barrier, option_type='call', barrier_type='up-and-out'):
        """Barrier option pricing"""
        n_paths = S_paths.shape[0]
        payoffs = np.zeros(n_paths)

        for i in range(n_paths):
            path = S_paths[i]

            if barrier_type == 'up-and-out':
                knocked_out = np.any(path >= barrier)
            elif barrier_type == 'down-and-out':
                knocked_out = np.any(path <= barrier)
            elif barrier_type == 'up-and-in':
                knocked_out = not np.any(path >= barrier)
            else:  # down-and-in
                knocked_out = not np.any(path <= barrier)

            if not knocked_out:
                if option_type == 'call':
                    payoffs[i] = max(path[-1] - K, 0)
                else:
                    payoffs[i] = max(K - path[-1], 0)

        return np.mean(payoffs)

    @staticmethod
    def lookback_option_price(S_paths, option_type='call'):
        """Lookback option - best historical price"""
        if option_type == 'call':
            payoffs = S_paths[:, -1] - np.min(S_paths, axis=1)
        else:
            payoffs = np.max(S_paths, axis=1) - S_paths[:, -1]

        return np.mean(payoffs)

    @staticmethod
    def digital_option_price(S_paths, K, payout=1.0, option_type='call'):
        """Digital/Binary option"""
        final_prices = S_paths[:, -1]

        if option_type == 'call':
            payoffs = np.where(final_prices > K, payout, 0)
        else:
            payoffs = np.where(final_prices < K, payout, 0)

        return np.mean(payoffs)

# ==================== S&P500 LAG TO VIX FORECASTING ====================

class SPYtoVIXModel:
    """Vector Autoregression model for S&P500 lags to VIX prediction"""

    def __init__(self, spy_data, vix_data, max_lags=10):
        self.spy_data = spy_data
        self.vix_data = vix_data
        self.max_lags = max_lags
        self.model = None
        self.model_fit = None

    def prepare_data(self):
        """Prepare data for VAR model"""
        # Align indices
        common_idx = self.spy_data.index.intersection(self.vix_data.index)
        spy = self.spy_data.loc[common_idx]
        vix = self.vix_data.loc[common_idx]

        # Create dataframe
        df = pd.DataFrame({
            'SPY_return': spy.pct_change(),
            'VIX': vix,
            'VIX_change': vix.diff()
        }).dropna()

        return df

    def fit(self, lag_order=None):
        """Fit VAR model"""
        df = self.prepare_data()

        # Determine optimal lag order if not specified
        if lag_order is None:
            model_temp = VAR(df)
            lag_order_results = model_temp.select_order(maxlags=self.max_lags)
            lag_order = lag_order_results.aic

        # Fit model
        self.model = VAR(df)
        self.model_fit = self.model.fit(lag_order)

        return self.model_fit

    def forecast(self, steps=10):
        """Forecast VIX using SPY lags"""
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.model_fit.forecast(self.model_fit.endog[-self.model_fit.k_ar:], steps=steps)

        return pd.DataFrame(forecast, columns=self.model_fit.names)

    def granger_causality(self):
        """Test Granger causality from SPY to VIX"""
        df = self.prepare_data()

        # Test if SPY returns Granger-cause VIX
        results = grangercausalitytests(df[['VIX', 'SPY_return']], maxlag=self.max_lags, verbose=False)

        p_values = {}
        for lag in range(1, self.max_lags + 1):
            p_values[lag] = results[lag][0]['ssr_ftest'][1]

        return p_values

    def impulse_response(self, periods=20):
        """Calculate impulse response functions"""
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")

        irf = self.model_fit.irf(periods)
        return irf

# ==================== MARKOV CHAIN REGIME DETECTION ====================

class MarkovRegimeModel:
    """Enhanced Markov Chain regime detection with transition probabilities"""

    def __init__(self, n_states=3):
        self.n_states = n_states
        self.model = None
        self.states = None
        self.state_names = None

    def fit(self, vix_series):
        """Fit Hidden Markov Model to VIX data"""
        # Prepare features
        vix_level = vix_series.values.reshape(-1, 1)
        vix_change = vix_series.pct_change().fillna(0).values.reshape(-1, 1)
        vix_vol = vix_series.rolling(20).std().fillna(method='bfill').values.reshape(-1, 1)

        X = np.column_stack([vix_level, vix_change, vix_vol])

        # Fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        self.model.fit(X)

        # Predict states
        self.states = self.model.predict(X)

        # Name states based on VIX levels
        state_means = [vix_series[self.states == i].mean() for i in range(self.n_states)]
        sorted_indices = np.argsort(state_means)

        state_mapping = {old: new for new, old in enumerate(sorted_indices)}
        self.states = np.array([state_mapping[s] for s in self.states])

        self.state_names = ['Low Volatility', 'Medium Volatility', 'High Volatility'][:self.n_states]

        return self

    def get_transition_matrix(self):
        """Calculate empirical transition probability matrix"""
        n = len(self.states)
        transitions = np.zeros((self.n_states, self.n_states))

        for i in range(n - 1):
            transitions[self.states[i], self.states[i + 1]] += 1

        # Normalize to get probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transitions, row_sums, where=row_sums != 0)

        return transition_probs

    def get_steady_state(self):
        """Calculate steady-state distribution"""
        P = self.get_transition_matrix()

        # Find eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        steady_state = np.real(eigenvectors[:, idx])
        steady_state = steady_state / steady_state.sum()

        return steady_state

    def get_expected_duration(self):
        """Calculate expected duration in each state"""
        P = self.get_transition_matrix()
        durations = 1 / (1 - np.diag(P))

        return durations

    def forecast_state(self, current_state, n_steps=10):
        """Forecast future state probabilities"""
        P = self.get_transition_matrix()

        # Initial distribution
        prob = np.zeros(self.n_states)
        prob[current_state] = 1.0

        forecasts = [prob.copy()]

        for _ in range(n_steps):
            prob = prob @ P
            forecasts.append(prob.copy())

        return np.array(forecasts)

# ==================== DATA FETCHING ====================

@st.cache_data(ttl=3600)
def fetch_market_data(start_date, end_date):
    """Fetch comprehensive market data"""
    tickers = {
        'VIX': '^VIX',
        'VVIX': '^VVIX',
        'VXV': '^VXV',
        'SPY': 'SPY',
        'TLT': 'TLT',
        'GLD': 'GLD',
        'HYG': 'HYG',
        'VXX': 'VXX'
    }

    data_dict = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if 'Close' in df.columns:
                data_dict[name] = df['Close']
        except:
            continue

    df = pd.DataFrame(data_dict)
    df = df.fillna(method='ffill').dropna()

    return df

@st.cache_data(ttl=1800)
def fetch_vix_options():
    """Fetch VIX options for Heston calibration"""
    try:
        vix = yf.Ticker("^VIX")
        expirations = vix.options[:6]

        all_options = []
        for exp in expirations:
            try:
                opt_chain = vix.option_chain(exp)
                calls = opt_chain.calls

                calls['expiration'] = exp
                calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
                all_options.append(calls)
            except:
                continue

        if all_options:
            df_options = pd.concat(all_options, ignore_index=True)
            df_options['expiration'] = pd.to_datetime(df_options['expiration'])
            df_options['dte'] = (df_options['expiration'] - pd.Timestamp.now()).dt.days
            df_options['T'] = df_options['dte'] / 365.0

            # Filter valid options
            df_options = df_options[
                (df_options['dte'] > 0) &
                (df_options['volume'] > 0) &
                (df_options['mid_price'] > 0.1)
            ]

            return df_options
    except:
        pass

    return None

# ==================== FEATURE ENGINEERING ====================

def build_features(df, include_lags=True):
    """Build comprehensive feature set"""
    features = pd.DataFrame(index=df.index)

    # VIX features
    if include_lags:
        for lag in [1, 5, 10, 22]:
            features[f'VIX_lag_{lag}'] = df['VIX'].shift(lag)

    features['VIX_momentum_5d'] = df['VIX'].pct_change(5)
    features['VIX_momentum_22d'] = df['VIX'].pct_change(22)
    features['VIX_zscore'] = (df['VIX'] - df['VIX'].rolling(252).mean()) / df['VIX'].rolling(252).std()
    features['VIX_realized_vol'] = df['VIX'].pct_change().rolling(22).std() * np.sqrt(252)
    features['VIX_MA_ratio'] = df['VIX'] / df['VIX'].rolling(20).mean()

    # Term structure
    if 'VXV' in df.columns:
        features['VIX_term_structure'] = df['VIX'] / df['VXV']
        features['term_spread'] = df['VIX'] - df['VXV']

    # VVIX
    if 'VVIX' in df.columns:
        features['VVIX'] = df['VVIX']
        features['VVIX_change'] = df['VVIX'].pct_change(5)

    # SPY features
    if 'SPY' in df.columns:
        for lag in [1, 5, 22]:
            features[f'SPY_return_{lag}d'] = df['SPY'].pct_change(lag)

        features['SPY_realized_vol'] = df['SPY'].pct_change().rolling(22).std() * np.sqrt(252)

        # Correlation
        spy_returns = df['SPY'].pct_change()
        vix_changes = df['VIX'].diff()
        features['SPY_VIX_corr_60d'] = spy_returns.rolling(60).corr(vix_changes)

    # Credit
    if 'HYG' in df.columns:
        features['HYG_return_22d'] = df['HYG'].pct_change(22)

    # Gold
    if 'GLD' in df.columns:
        features['GLD_return_22d'] = df['GLD'].pct_change(22)

    # Bonds
    if 'TLT' in df.columns:
        features['TLT_return_22d'] = df['TLT'].pct_change(22)

    return features.dropna()

def remove_multicollinearity(X, threshold=10):
    """Remove highly collinear features using VIF"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_clean = X.copy()

    # Remove perfect correlations
    corr_matrix = X_clean.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    X_clean = X_clean.drop(columns=to_drop)

    # Iterative VIF removal
    max_iter = 15
    for iteration in range(max_iter):
        if X_clean.shape[1] <= 3:
            break

        try:
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X_clean.columns
            vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i)
                              for i in range(X_clean.shape[1])]

            max_vif = vif_data["VIF"].replace([np.inf, -np.inf], np.nan).max()

            if pd.isna(max_vif) or max_vif <= threshold:
                break

            max_vif_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
            X_clean = X_clean.drop(columns=[max_vif_feature])
        except:
            break

    return X_clean

# ==================== MODEL TRAINING ====================

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple classification models"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    # Logistic Regression
    log_reg = LogisticRegression(
        penalty='l2',
        C=0.1,
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    log_reg.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = log_reg

    # XGBoost
    scale_pos_weight = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))
    xgb_clf = xgb.XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )
    xgb_clf.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb_clf

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf_clf

    # Evaluate models
    results = {}
    for name, model in models.items():
        train_probs = model.predict_proba(X_train_scaled)[:, 1]
        test_probs = model.predict_proba(X_test_scaled)[:, 1]

        results[name] = {
            'model': model,
            'train_probs': train_probs,
            'test_probs': test_probs,
            'train_auc': roc_auc_score(y_train, train_probs),
            'test_auc': roc_auc_score(y_test, test_probs)
        }

    return results, scaler

# ==================== VISUALIZATION ====================

def plot_heston_surface(heston_model, strikes, maturities):
    """Plot Heston implied volatility surface"""
    K_grid = np.linspace(min(strikes), max(strikes), 30)
    T_grid = np.linspace(min(maturities), max(maturities), 30)

    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
    IV_mesh = np.zeros_like(K_mesh)

    for i in range(len(T_grid)):
        for j in range(len(K_grid)):
            try:
                price = heston_model.option_price(K_mesh[i, j], T_mesh[i, j], 'call')

                # Implied vol approximation
                IV_mesh[i, j] = np.sqrt(heston_model.v0)
            except:
                IV_mesh[i, j] = np.nan

    fig = go.Figure(data=[go.Surface(
        x=K_mesh,
        y=T_mesh,
        z=IV_mesh,
        colorscale='Viridis',
        colorbar=dict(title="Implied Volatility")
    )])

    fig.update_layout(
        title='Heston Model Implied Volatility Surface',
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='Time to Maturity',
            zaxis_title='Implied Volatility'
        ),
        template='plotly_dark',
        height=700,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117'
    )

    return fig

def plot_var_forecast(forecast_df, historical_df):
    """Plot VAR model forecasts"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('VIX Forecast', 'SPY Return Forecast'),
        vertical_spacing=0.1
    )

    # VIX
    fig.add_trace(
        go.Scatter(
            x=historical_df.index[-60:],
            y=historical_df['VIX'].values[-60:],
            mode='lines',
            name='Historical VIX',
            line=dict(color='#00D9FF', width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=pd.date_range(start=historical_df.index[-1], periods=len(forecast_df)+1)[1:],
            y=forecast_df['VIX'].values,
            mode='lines+markers',
            name='Forecast VIX',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ),
        row=1, col=1
    )

    # SPY Return
    fig.add_trace(
        go.Scatter(
            x=historical_df.index[-60:],
            y=historical_df['SPY_return'].values[-60:],
            mode='lines',
            name='Historical SPY Return',
            line=dict(color='#4CAF50', width=2)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=pd.date_range(start=historical_df.index[-1], periods=len(forecast_df)+1)[1:],
            y=forecast_df['SPY_return'].values,
            mode='lines+markers',
            name='Forecast SPY Return',
            line=dict(color='#FFA500', width=2, dash='dash')
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        height=800,
        showlegend=True,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117'
    )

    return fig

def plot_markov_regimes(vix_series, states, state_names):
    """Plot VIX with Markov regime overlay"""
    colors = ['#4CAF50', '#FFA500', '#FF6B6B']

    fig = go.Figure()

    for i, name in enumerate(state_names):
        mask = states == i
        dates = vix_series.index[mask]
        values = vix_series.values[mask]

        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='markers',
            name=name,
            marker=dict(color=colors[i], size=4)
        ))

    fig.update_layout(
        title='VIX with Markov Regime Classification',
        xaxis_title='Date',
        yaxis_title='VIX Level',
        template='plotly_dark',
        height=600,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117'
    )

    return fig

def plot_transition_matrix(transition_matrix, state_names):
    """Plot transition probability matrix"""
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=state_names,
        y=state_names,
        colorscale='Viridis',
        text=transition_matrix,
        texttemplate='%{text:.2%}',
        textfont=dict(size=14),
        colorbar=dict(title="Probability")
    ))

    fig.update_layout(
        title='Markov Chain Transition Probability Matrix',
        xaxis_title='To State',
        yaxis_title='From State',
        template='plotly_dark',
        height=500,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117'
    )

    return fig

def plot_model_comparison(results, y_test):
    """Plot ROC curves for model comparison"""
    fig = go.Figure()

    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['test_probs'])
        auc = result['test_auc']

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{name} (AUC={auc:.3f})',
            line=dict(width=2)
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='white', width=1, dash='dash')
    ))

    fig.update_layout(
        title='Model Comparison - ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark',
        height=600,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117'
    )

    return fig

# ==================== MAIN APPLICATION ====================

def main():
    st.title("VIX Professional Trading Dashboard")
    st.markdown("### Advanced Volatility Analysis with Heston Model, Markov Chains, and Econometric Forecasting")

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Data Parameters")
        start_date = st.date_input("Start Date", value=datetime(2015, 1, 1))
        end_date = st.date_input("End Date", value=datetime.today())

        st.subheader("Model Parameters")
        vix_threshold = st.slider("VIX Spike Threshold", 15.0, 35.0, 25.0, 0.5)
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100

        st.subheader("VAR Model")
        max_lags = st.slider("Maximum Lags for VAR", 1, 22, 10)
        forecast_steps = st.slider("Forecast Steps", 5, 30, 10)

        st.subheader("Markov Chain")
        n_states = st.selectbox("Number of States", [2, 3, 4], index=1)

        st.subheader("Heston Model")
        enable_heston = st.checkbox("Enable Heston Calibration", value=True)
        n_heston_paths = st.slider("Monte Carlo Paths", 1000, 10000, 5000, 1000)

        st.markdown("---")
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

    # Run analysis
    if run_analysis:
        st.session_state.analysis_complete = False

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Fetch data
        status_text.text("Fetching market data...")
        progress_bar.progress(10)
        df = fetch_market_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        # Build features
        status_text.text("Building features...")
        progress_bar.progress(20)
        X = build_features(df, include_lags=True)
        X = remove_multicollinearity(X, threshold=10)

        # Prepare target
        y = (df['VIX'].loc[X.index] > vix_threshold).astype(int)

        # Train-test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train models
        status_text.text("Training classification models...")
        progress_bar.progress(40)
        model_results, scaler = train_models(X_train, y_train, X_test, y_test)

        # VAR Model
        status_text.text("Fitting VAR model...")
        progress_bar.progress(60)
        var_model = SPYtoVIXModel(df['SPY'], df['VIX'], max_lags=max_lags)
        var_fit = var_model.fit()
        var_forecast = var_model.forecast(steps=forecast_steps)
        granger_results = var_model.granger_causality()

        # Markov Chain
        status_text.text("Detecting volatility regimes...")
        progress_bar.progress(70)
        markov_model = MarkovRegimeModel(n_states=n_states)
        markov_model.fit(df['VIX'].loc[X.index])
        transition_matrix = markov_model.get_transition_matrix()
        steady_state = markov_model.get_steady_state()
        expected_durations = markov_model.get_expected_duration()

        # Heston Model
        heston_model = None
        heston_params = None
        if enable_heston:
            status_text.text("Calibrating Heston model...")
            progress_bar.progress(85)

            df_options = fetch_vix_options()
            if df_options is not None and len(df_options) > 20:
                current_vix = df['VIX'].iloc[-1]

                # Prepare market data for calibration
                market_data = []
                for _, row in df_options.head(20).iterrows():
                    market_data.append((row['strike'], row['T'], row['mid_price']))

                try:
                    heston_params = HestonModel.calibrate_to_surface(
                        market_data,
                        current_vix,
                        df_options['strike'].values[:20],
                        df_options['T'].values[:20]
                    )

                    v0, kappa, theta, sigma, rho = heston_params
                    heston_model = HestonModel(current_vix, v0, kappa, theta, sigma, rho)
                except:
                    st.warning("Heston calibration failed. Using default parameters.")

        # Store in session state
        progress_bar.progress(100)
        status_text.text("Analysis complete!")

        st.session_state.df = df
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.model_results = model_results
        st.session_state.scaler = scaler
        st.session_state.var_model = var_model
        st.session_state.var_fit = var_fit
        st.session_state.var_forecast = var_forecast
        st.session_state.granger_results = granger_results
        st.session_state.markov_model = markov_model
        st.session_state.transition_matrix = transition_matrix
        st.session_state.steady_state = steady_state
        st.session_state.expected_durations = expected_durations
        st.session_state.heston_model = heston_model
        st.session_state.heston_params = heston_params
        st.session_state.vix_threshold = vix_threshold
        st.session_state.n_heston_paths = n_heston_paths
        st.session_state.analysis_complete = True

        st.success("Analysis completed successfully!")
        st.rerun()

    # Display results
    if st.session_state.analysis_complete:
        df = st.session_state.df
        X = st.session_state.X
        y = st.session_state.y
        y_test = st.session_state.y_test
        model_results = st.session_state.model_results
        var_forecast = st.session_state.var_forecast
        granger_results = st.session_state.granger_results
        markov_model = st.session_state.markov_model
        transition_matrix = st.session_state.transition_matrix
        steady_state = st.session_state.steady_state
        expected_durations = st.session_state.expected_durations
        heston_model = st.session_state.heston_model
        heston_params = st.session_state.heston_params

        # Tabs
        tabs = st.tabs([
            "Overview",
            "Model Performance",
            "S&P500 to VIX Forecasting",
            "Markov Chain Regimes",
            "Heston Model & Exotics",
            "Feature Analysis"
        ])

        # Tab 1: Overview
        with tabs[0]:
            st.header("Market Overview")

            col1, col2, col3, col4, col5 = st.columns(5)

            current_vix = df['VIX'].iloc[-1]
            prev_vix = df['VIX'].iloc[-2]
            vix_change = ((current_vix - prev_vix) / prev_vix) * 100

            best_model = max(model_results.items(), key=lambda x: x[1]['test_auc'])
            current_prob = best_model[1]['test_probs'][-1]

            current_state = markov_model.states[-1]
            state_name = markov_model.state_names[current_state]

            with col1:
                st.metric("Current VIX", f"{current_vix:.2f}", f"{vix_change:+.2f}%")
            with col2:
                st.metric("Spike Probability", f"{current_prob:.1%}")
            with col3:
                st.metric("Best Model AUC", f"{best_model[1]['test_auc']:.3f}")
            with col4:
                st.metric("Current Regime", state_name)
            with col5:
                if 'VXV' in df.columns:
                    term_structure = df['VIX'].iloc[-1] / df['VXV'].iloc[-1]
                    st.metric("VIX/VXV Ratio", f"{term_structure:.3f}")

            st.markdown("---")
            st.subheader("VIX Historical Performance")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index[-252:],
                y=df['VIX'].values[-252:],
                mode='lines',
                name='VIX',
                line=dict(color='#00D9FF', width=2)
            ))

            fig.add_hline(
                y=st.session_state.vix_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {st.session_state.vix_threshold}"
            )

            fig.update_layout(
                title='VIX - Last 12 Months',
                xaxis_title='Date',
                yaxis_title='VIX Level',
                template='plotly_dark',
                height=500,
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Model Performance
        with tabs[1]:
            st.header("Classification Model Performance")

            # Performance metrics
            perf_data = []
            for name, result in model_results.items():
                perf_data.append({
                    'Model': name,
                    'Train AUC': f"{result['train_auc']:.4f}",
                    'Test AUC': f"{result['test_auc']:.4f}",
                    'Overfitting': f"{(result['train_auc'] - result['test_auc']):.4f}"
                })

            st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

            st.markdown("---")
            st.subheader("ROC Curve Comparison")

            fig = plot_model_comparison(model_results, y_test)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Confusion Matrices")

            cols = st.columns(len(model_results))
            for idx, (name, result) in enumerate(model_results.items()):
                with cols[idx]:
                    y_pred = (result['test_probs'] >= 0.5).astype(int)
                    cm = confusion_matrix(y_test, y_pred)

                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted 0', 'Predicted 1'],
                        y=['Actual 0', 'Actual 1'],
                        colorscale='Blues',
                        text=cm,
                        texttemplate='%{text}',
                        textfont=dict(size=16)
                    ))

                    fig.update_layout(
                        title=name,
                        template='plotly_dark',
                        height=400,
                        paper_bgcolor='#0e1117',
                        plot_bgcolor='#0e1117'
                    )

                    st.plotly_chart(fig, use_container_width=True)

        # Tab 3: S&P500 to VIX Forecasting
        with tabs[2]:
            st.header("Vector Autoregression: S&P500 Lags to VIX Prediction")

            st.subheader("VAR Model Summary")
            st.text(st.session_state.var_fit.summary())

            st.markdown("---")
            st.subheader("Granger Causality Test Results")

            st.markdown("""
            **Null Hypothesis:** S&P500 returns do NOT Granger-cause VIX changes

            P-values below 0.05 indicate rejection of null hypothesis (causality exists)
            """)

            granger_df = pd.DataFrame([
                {'Lag': lag, 'P-Value': pval, 'Significant': 'Yes' if pval < 0.05 else 'No'}
                for lag, pval in granger_results.items()
            ])

            st.dataframe(granger_df, use_container_width=True)

            st.markdown("---")
            st.subheader("VAR Forecast")

            hist_data = st.session_state.var_model.prepare_data()
            fig = plot_var_forecast(var_forecast, hist_data)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Impulse Response Functions")

            irf = st.session_state.var_model.impulse_response(periods=20)

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('VIX Response to SPY Shock', 'SPY Response to VIX Shock')
            )

            periods = list(range(20))

            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=irf.irfs[:, 1, 0],  # VIX response to SPY shock
                    mode='lines+markers',
                    name='VIX <- SPY',
                    line=dict(color='#FF6B6B', width=2)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=irf.irfs[:, 0, 1],  # SPY response to VIX shock
                    mode='lines+markers',
                    name='SPY <- VIX',
                    line=dict(color='#4CAF50', width=2)
                ),
                row=1, col=2
            )

            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)

            fig.update_layout(
                template='plotly_dark',
                height=500,
                showlegend=False,
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Tab 4: Markov Chain Regimes
        with tabs[3]:
            st.header("Markov Chain Volatility Regime Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Steady-State Distribution")
                steady_df = pd.DataFrame({
                    'State': markov_model.state_names,
                    'Probability': steady_state
                })
                st.dataframe(steady_df, use_container_width=True)

            with col2:
                st.subheader("Expected Duration (Days)")
                duration_df = pd.DataFrame({
                    'State': markov_model.state_names,
                    'Duration': expected_durations
                })
                st.dataframe(duration_df, use_container_width=True)

            with col3:
                st.subheader("Current State Analysis")
                current_state = markov_model.states[-1]
                st.metric("Current State", markov_model.state_names[current_state])
                st.metric("Expected Duration", f"{expected_durations[current_state]:.1f} days")
                st.metric("Steady-State Prob", f"{steady_state[current_state]:.1%}")

            st.markdown("---")
            st.subheader("Regime Classification Over Time")

            fig = plot_markov_regimes(df['VIX'].loc[X.index], markov_model.states, markov_model.state_names)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Transition Probability Matrix")

            fig = plot_transition_matrix(transition_matrix, markov_model.state_names)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("State Forecast (10 Days Ahead)")

            current_state = markov_model.states[-1]
            state_forecasts = markov_model.forecast_state(current_state, n_steps=10)

            fig = go.Figure()

            for i, state_name in enumerate(markov_model.state_names):
                fig.add_trace(go.Scatter(
                    x=list(range(11)),
                    y=state_forecasts[:, i],
                    mode='lines+markers',
                    name=state_name,
                    line=dict(width=2)
                ))

            fig.update_layout(
                title='State Probability Evolution',
                xaxis_title='Days Ahead',
                yaxis_title='Probability',
                template='plotly_dark',
                height=500,
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Tab 5: Heston Model & Exotics
        with tabs[4]:
            st.header("Heston Stochastic Volatility Model & Exotic Options")

            if heston_model is not None and heston_params is not None:
                v0, kappa, theta, sigma, rho = heston_params

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Initial Variance (v0)", f"{v0:.4f}")
                with col2:
                    st.metric("Mean Reversion (κ)", f"{kappa:.4f}")
                with col3:
                    st.metric("Long-term Var (θ)", f"{theta:.4f}")
                with col4:
                    st.metric("Vol of Vol (σ)", f"{sigma:.4f}")
                with col5:
                    st.metric("Correlation (ρ)", f"{rho:.4f}")

                st.markdown("---")
                st.subheader("Heston Model Paths Simulation")

                current_vix = df['VIX'].iloc[-1]
                S_paths, v_paths = heston_model.simulate_paths(
                    T=1.0,
                    n_steps=252,
                    n_paths=st.session_state.n_heston_paths
                )

                # Plot sample paths
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('VIX Price Paths', 'Variance Paths')
                )

                # Sample 100 paths for visualization
                sample_indices = np.random.choice(S_paths.shape[0], min(100, S_paths.shape[0]), replace=False)

                for idx in sample_indices:
                    fig.add_trace(
                        go.Scatter(
                            y=S_paths[idx],
                            mode='lines',
                            line=dict(width=0.5),
                            showlegend=False,
                            opacity=0.3
                        ),
                        row=1, col=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            y=v_paths[idx],
                            mode='lines',
                            line=dict(width=0.5),
                            showlegend=False,
                            opacity=0.3
                        ),
                        row=1, col=2
                    )

                fig.update_layout(
                    template='plotly_dark',
                    height=600,
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#0e1117'
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("Exotic Options Pricing")

                exotic_opts = ExoticOptions()

                col1, col2 = st.columns(2)

                with col1:
                    strike_pct = st.slider("Strike (% of current VIX)", 80, 120, 100, 5)
                    strike = current_vix * (strike_pct / 100)

                    st.markdown(f"**Strike Level:** {strike:.2f}")

                with col2:
                    barrier_pct = st.slider("Barrier (% of current VIX)", 90, 150, 120, 5)
                    barrier = current_vix * (barrier_pct / 100)

                    st.markdown(f"**Barrier Level:** {barrier:.2f}")

                # Calculate exotic prices
                asian_call = exotic_opts.asian_option_price(S_paths, strike, 'call')
                asian_put = exotic_opts.asian_option_price(S_paths, strike, 'put')

                barrier_uao = exotic_opts.barrier_option_price(S_paths, strike, barrier, 'call', 'up-and-out')
                barrier_dao = exotic_opts.barrier_option_price(S_paths, strike, barrier, 'call', 'down-and-out')

                lookback_call = exotic_opts.lookback_option_price(S_paths, 'call')
                lookback_put = exotic_opts.lookback_option_price(S_paths, 'put')

                digital_call = exotic_opts.digital_option_price(S_paths, strike, 1.0, 'call')
                digital_put = exotic_opts.digital_option_price(S_paths, strike, 1.0, 'put')

                # Display results
                exotic_df = pd.DataFrame([
                    {'Option Type': 'Asian Call', 'Price': f"${asian_call:.4f}"},
                    {'Option Type': 'Asian Put', 'Price': f"${asian_put:.4f}"},
                    {'Option Type': 'Up-and-Out Barrier Call', 'Price': f"${barrier_uao:.4f}"},
                    {'Option Type': 'Down-and-Out Barrier Call', 'Price': f"${barrier_dao:.4f}"},
                    {'Option Type': 'Lookback Call', 'Price': f"${lookback_call:.4f}"},
                    {'Option Type': 'Lookback Put', 'Price': f"${lookback_put:.4f}"},
                    {'Option Type': 'Digital Call', 'Price': f"${digital_call:.4f}"},
                    {'Option Type': 'Digital Put', 'Price': f"${digital_put:.4f}"}
                ])

                st.dataframe(exotic_df, use_container_width=True)

                st.info(f"Prices based on {st.session_state.n_heston_paths:,} Monte Carlo simulations")

            else:
                st.warning("Heston model calibration not available. Enable in sidebar and ensure sufficient options data exists.")

        # Tab 6: Feature Analysis
        with tabs[5]:
            st.header("Feature Importance Analysis")

            importance_data = []
            for name, result in model_results.items():
                model = result['model']

                if hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])
                elif hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                else:
                    continue

                for feat, imp in zip(X.columns, importance):
                    importance_data.append({
                        'Model': name,
                        'Feature': feat,
                        'Importance': imp
                    })

            df_imp = pd.DataFrame(importance_data)

            if len(df_imp) > 0:
                # Top features
                top_features = df_imp.groupby('Feature')['Importance'].mean().nlargest(15).index
                df_imp_filtered = df_imp[df_imp['Feature'].isin(top_features)]

                fig = px.bar(
                    df_imp_filtered,
                    x='Importance',
                    y='Feature',
                    color='Model',
                    barmode='group',
                    orientation='h'
                )

                fig.update_layout(
                    title='Top 15 Features by Average Importance',
                    template='plotly_dark',
                    height=700,
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#0e1117'
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("Feature Correlation with Target")

                correlations = []
                for col in X.columns:
                    corr = X[col].corr(y)
                    correlations.append({'Feature': col, 'Correlation': corr, 'Abs Correlation': abs(corr)})

                corr_df = pd.DataFrame(correlations).sort_values('Abs Correlation', ascending=False).head(15)

                fig = go.Figure()

                colors = ['#FF6B6B' if c > 0 else '#4CAF50' for c in corr_df['Correlation']]

                fig.add_trace(go.Bar(
                    x=corr_df['Correlation'],
                    y=corr_df['Feature'],
                    orientation='h',
                    marker=dict(color=colors)
                ))

                fig.update_layout(
                    title='Top 15 Features by Correlation with Spike Target',
                    xaxis_title='Correlation',
                    yaxis_title='Feature',
                    template='plotly_dark',
                    height=600,
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#0e1117'
                )

                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Configure parameters in the sidebar and click 'Run Analysis' to begin the comprehensive VIX analysis.")

if __name__ == '__main__':
    main()
