"""
VIX Spike Probability Model - Enhanced Robust Version
Combines sentiment analysis capabilities with robust model fitting and presentation
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import requests
from pathlib import Path

# Parameters
VIX_SPIKE_THRESHOLD = 30.0
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Optional: FRED API
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except:
    FRED_AVAILABLE = False

# Directories
OUT = Path('./vix_output')
OUT.mkdir(exist_ok=True)

# API Keys (set these if available)
FRED_API_KEY = "48f048d09b8cface96982149dcf2c538"  # Set  your FRED API key here
NEWSAPI_KEY = None   # Set your NewsAPI key here

def fetch_market_data():
    """Fetch VIX and related market data"""
    print("\n📊 Fetching market data...")
    
    # Fetch VIX
    vix = yf.download("^VIX", start="2000-01-01", end=datetime.today().strftime("%Y-%m-%d"), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix.rename(columns={"Close": "VIX"})
    vix = vix[["VIX"]].dropna()
    
    # Fetch SPY as proxy for S&P 500
    spy = yf.download("SPY", start="2000-01-01", end=datetime.today().strftime("%Y-%m-%d"), progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy.rename(columns={"Close": "SPY"})
    spy = spy[["SPY"]].dropna()
    
    # Fetch TLT for bond yields proxy
    tlt = yf.download("TLT", start="2002-07-26", end=datetime.today().strftime("%Y-%m-%d"), progress=False)
    if isinstance(tlt.columns, pd.MultiIndex):
        tlt.columns = tlt.columns.get_level_values(0)
    tlt = tlt.rename(columns={"Close": "TLT"})
    tlt = tlt[["TLT"]].dropna()
    
    # Combine data
    df = vix.join(spy, how='outer').join(tlt, how='outer')
    df = df.fillna(method='ffill').dropna()
    
    print(f"✓ Data loaded: {len(df)} trading days")
    print(f"  • Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    return df

def fetch_gdelt_sentiment(start_date=None, end_date=None):
    """
    Fetch GDELT sentiment data - simplified version that works with current API
    """
    print("\n📰 Fetching GDELT sentiment data...")
    try:
        # Use GDELT DOC API for recent sentiment
        gdelt_url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            "?query=(market OR stocks OR VIX OR volatility OR S%26P)"
            "&mode=ToneChart"
            "&maxrecords=250"
            "&format=CSV"
            "&sort=DateDesc"
        )
        
        response = requests.get(gdelt_url, timeout=30)
        if response.status_code == 200:
            from io import StringIO
            news_df = pd.read_csv(StringIO(response.text))
            
            # Process the tone data
            if 'Date' in news_df.columns:
                news_df['Date'] = pd.to_datetime(news_df['Date'], format='%Y%m%d%H%M%S', errors='coerce')
                news_df = news_df.dropna(subset=['Date'])
                news_df = news_df.set_index('Date')
                
                # Find tone column (might be named differently)
                tone_cols = [col for col in news_df.columns if 'tone' in col.lower() or 'avg' in col.lower()]
                if tone_cols:
                    tone_col = tone_cols[0]
                else:
                    # If no tone column found, use first numeric column
                    numeric_cols = news_df.select_dtypes(include=[np.number]).columns
                    tone_col = numeric_cols[0] if len(numeric_cols) > 0 else news_df.columns[0]
                
                # Resample to daily
                daily_sentiment = news_df[[tone_col]].resample('D').mean()
                daily_sentiment.columns = ['gdelt_tone']
                
                print(f"  ✓ GDELT sentiment retrieved: {len(daily_sentiment)} days")
                return daily_sentiment
            else:
                print("  ⚠️ GDELT data format unexpected, using neutral sentiment")
                return None
        else:
            print(f"  ⚠️ GDELT API error (status {response.status_code}), continuing without sentiment")
            return None
    except Exception as e:
        print(f"  ⚠️ GDELT fetch failed: {e}")
        return None

def fetch_fred_data(series_ids=None):
    """Fetch FRED data if available"""
    if not FRED_AVAILABLE or not FRED_API_KEY:
        print("  ℹ️ FRED data not available (no API key or library)")
        return None
    
    print("\n📈 Fetching FRED economic data...")
    
    if series_ids is None:
        series_ids = {
            'T10Y2Y': 'Term Spread',
            'DGS10': '10Y Treasury',
            'FEDFUNDS': 'Fed Funds Rate',
            'BAA10Y': 'Credit Spread'
        }
    
    try:
        fred = Fred(api_key=FRED_API_KEY)
        fred_df = pd.DataFrame()
        
        for series_id, name in series_ids.items():
            try:
                series = fred.get_series(series_id, observation_start='2000-01-01')
                series.name = name
                fred_df = pd.concat([fred_df, series], axis=1)
                print(f"  ✓ {name} loaded")
            except Exception as e:
                print(f"  ⚠️ Could not fetch {name}: {e}")
        
        if not fred_df.empty:
            fred_df.index = pd.to_datetime(fred_df.index)
            return fred_df
    except Exception as e:
        print(f"  ⚠️ FRED fetch error: {e}")
    
    return None

def build_robust_features(df, sentiment_df=None, fred_df=None, threshold=30):
    """Build features carefully avoiding multicollinearity"""
    print("\n🔧 Building robust features (avoiding collinearity)...")
    
    features = pd.DataFrame(index=df.index)
    
    # GROUP 1: VIX Level Features (pick selective lags)
    features['VIX_lag5'] = df['VIX'].shift(5)
    features['VIX_lag22'] = df['VIX'].shift(22)
    
    # GROUP 2: VIX Momentum (non-overlapping periods)
    features['VIX_momentum_5d'] = df['VIX'].pct_change(5)
    features['VIX_momentum_22d'] = df['VIX'].pct_change(22)
    
    # GROUP 3: VIX Mean Reversion (z-score)
    features['VIX_zscore'] = (df['VIX'] - df['VIX'].rolling(252, min_periods=60).mean()) / \
                             df['VIX'].rolling(252, min_periods=60).std()
    
    # GROUP 4: VIX Volatility
    features['VIX_realized_vol'] = df['VIX'].pct_change().rolling(22).std() * np.sqrt(252)
    
    # GROUP 5: Technical Indicators
    features['VIX_MA_ratio'] = df['VIX'] / df['VIX'].rolling(20).mean()
    
    # GROUP 6: Market Features
    if 'SPY' in df.columns:
        features['SPY_return_22d'] = df['SPY'].pct_change(22)
        features['SPY_vol_22d'] = df['SPY'].pct_change().rolling(22).std() * np.sqrt(252)
    
    # GROUP 7: Bond Market (flight to quality)
    if 'TLT' in df.columns:
        features['TLT_change_22d'] = df['TLT'].pct_change(22)
    
    # GROUP 8: Sentiment Features
    if sentiment_df is not None and not sentiment_df.empty:
        # Align indices
        sentiment_aligned = sentiment_df.reindex(df.index, method='ffill')
        if 'gdelt_tone' in sentiment_aligned.columns:
            features['sentiment'] = sentiment_aligned['gdelt_tone']
            features['sentiment_lag5'] = features['sentiment'].shift(5)
            features['sentiment_ma'] = features['sentiment'].rolling(10, min_periods=1).mean()
            print("  ✓ Sentiment features added")
    
    # GROUP 9: FRED Economic Features
    if fred_df is not None and not fred_df.empty:
        fred_aligned = fred_df.reindex(df.index, method='ffill')
        
        # Add term spread if available
        if 'Term Spread' in fred_aligned.columns:
            features['term_spread'] = fred_aligned['Term Spread']
            features['term_spread_change'] = features['term_spread'].diff(22)
        
        # Add credit spread if available
        if 'Credit Spread' in fred_aligned.columns:
            features['credit_spread'] = fred_aligned['Credit Spread']
        
        print("  ✓ FRED economic features added")
    
    # GROUP 10: Regime Indicators
    features['VIX_high_regime'] = (df['VIX'].shift(1) > 20).astype(int)
    features['VIX_extreme_regime'] = (df['VIX'].shift(1) > 30).astype(int)
    
    # Target variable
    target = (df['VIX'] > threshold).astype(int)
    
    # Remove NaN
    features = features.dropna()
    target = target.loc[features.index]
    
    print(f"✓ Features created: {len(features.columns)} features")
    print(f"✓ Observations: {len(features)} days")
    print(f"✓ Spike rate: {target.mean():.1%}")
    
    # List features by category
    print("\n📋 Feature Categories:")
    feature_groups = {
        'VIX Levels': [f for f in features.columns if 'lag' in f],
        'Momentum': [f for f in features.columns if 'momentum' in f],
        'Mean Reversion': [f for f in features.columns if 'zscore' in f or 'MA_ratio' in f],
        'Volatility': [f for f in features.columns if 'vol' in f],
        'Market': [f for f in features.columns if 'SPY' in f or 'TLT' in f],
        'Sentiment': [f for f in features.columns if 'sentiment' in f],
        'Economic': [f for f in features.columns if 'spread' in f],
        'Regime': [f for f in features.columns if 'regime' in f]
    }
    
    for group, feats in feature_groups.items():
        if feats:
            print(f"  • {group}: {', '.join(feats)}")
    
    return features, target

def aggressive_multicollinearity_removal(X):
    """More aggressive removal of multicollinear features"""
    print("\n🔍 Aggressive multicollinearity removal...")
    
    X_clean = X.copy()
    
    # Step 1: Remove perfect correlations (|r| > 0.95)
    corr = X_clean.corr().abs()
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper_tri.columns:
        if col in to_drop:
            continue
        high_corr = upper_tri[col][upper_tri[col] > 0.95]
        if not high_corr.empty:
            print(f"  • Removing {col} (corr > 0.95 with {list(high_corr.index)})")
            to_drop.add(col)
    
    X_clean = X_clean.drop(columns=to_drop)
    
    # Step 2: Calculate VIF and iteratively remove high VIF features
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        max_iterations = 10
        for iteration in range(max_iterations):
            if X_clean.shape[1] <= 2:
                break
                
            # Calculate VIF for all features
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X_clean.columns
            vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i) 
                              for i in range(X_clean.shape[1])]
            
            # Find feature with highest VIF
            max_vif_idx = vif_data["VIF"].replace([np.inf, -np.inf], np.nan).idxmax()
            if pd.isna(max_vif_idx):
                break
            
            max_vif = vif_data.loc[max_vif_idx, "VIF"]
            
            # If max VIF is acceptable, stop
            if max_vif <= 10 or pd.isna(max_vif):
                break
            
            # Remove feature with highest VIF
            feature_to_remove = vif_data.loc[max_vif_idx, "Feature"]
            print(f"  • Iteration {iteration+1}: Removing {feature_to_remove} (VIF={max_vif:.1f})")
            X_clean = X_clean.drop(columns=[feature_to_remove])
        
        # Final VIF report
        print("\n📊 Final VIF values:")
        vif_final = pd.DataFrame()
        vif_final["Feature"] = X_clean.columns
        vif_final["VIF"] = [variance_inflation_factor(X_clean.values, i) 
                           for i in range(X_clean.shape[1])]
        
        for _, row in vif_final.sort_values('VIF', ascending=False).iterrows():
            status = "✓" if row['VIF'] <= 10 else "⚠️"
            print(f"  {status} {row['Feature']:25s}: {row['VIF']:6.2f}")
    
    except Exception as e:
        print(f"  ⚠️ VIF calculation issue: {e}")
        print("  Falling back to correlation-based removal only")
    
    print(f"\n✓ Final features retained: {len(X_clean.columns)} features")
    return X_clean

def fit_robust_logit(X_train, y_train, X_test, y_test):
    """Fit logistic regression with multiple fallback options"""
    print("\n📊 Fitting Robust Logistic Regression...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = None
    method_used = None
    
    # Try statsmodels first
    try:
        X_train_const = sm.add_constant(X_train_scaled, has_constant='add')
        X_test_const = sm.add_constant(X_test_scaled, has_constant='add')
        
        for method in ['newton', 'bfgs', 'lbfgs']:
            try:
                logit = sm.Logit(y_train.values, X_train_const)
                result = logit.fit(method=method, maxiter=1000, disp=False, warn_convergence=False)
                
                if result.mle_retvals['converged']:
                    model = result
                    method_used = f"statsmodels ({method})"
                    break
            except:
                continue
        
        if model is not None:
            print(f"✅ {method_used} fitted successfully!")
            y_pred_prob = model.predict(X_test_const)
            auc = roc_auc_score(y_test, y_pred_prob)
            print(f"   ROC-AUC: {auc:.4f}")
            
            # Show top features
            try:
                feature_importance = pd.DataFrame({
                    'Feature': ['const'] + list(X_train.columns),
                    'Coefficient': model.params,
                    'P-value': model.pvalues
                })
                
                print("\n📊 Top Features (by |coefficient|):")
                feature_importance['abs_coef'] = feature_importance['Coefficient'].abs()
                for _, row in feature_importance.nlargest(5, 'abs_coef').iterrows():
                    if row['Feature'] != 'const':
                        sig = "***" if row['P-value'] < 0.01 else "**" if row['P-value'] < 0.05 else "*" if row['P-value'] < 0.1 else ""
                        print(f"  • {row['Feature']:25s}: coef={row['Coefficient']:7.3f} {sig}")
            except:
                pass
            
            return model, scaler, auc
    
    except Exception as e:
        print(f"  ⚠️ Statsmodels failed: {e}")
    
    # Fallback to sklearn
    print("  Falling back to sklearn LogisticRegression...")
    
    log_reg = LogisticRegression(
        penalty='l2',
        C=0.1,
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    
    log_reg.fit(X_train_scaled, y_train)
    y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"✅ Sklearn Logit fitted successfully!")
    print(f"   ROC-AUC: {auc:.4f}")
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': log_reg.coef_[0]
    })
    
    print("\n📊 Top Features (by |coefficient|):")
    feature_importance['abs_coef'] = feature_importance['Coefficient'].abs()
    for _, row in feature_importance.nlargest(5, 'abs_coef').iterrows():
        print(f"  • {row['Feature']:25s}: coef={row['Coefficient']:7.3f}")
    
    return log_reg, scaler, auc

def fit_xgboost(X_train, y_train, X_test, y_test):
    """Fit XGBoost with proper handling of class imbalance"""
    print("\n🌲 Fitting XGBoost...")
    
    try:
        # Calculate class weight
        scale_pos_weight = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))
        
        xgb_clf = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=RANDOM_SEED
        )
        
        # Fit with early stopping
        xgb_clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Get predictions
        xgb_prob = xgb_clf.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_prob)
        
        print(f"✅ XGBoost fitted successfully!")
        print(f"   ROC-AUC: {xgb_auc:.4f}")
        
        # Feature importance
        if len(X_train.columns) > 0:
            feature_imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': xgb_clf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Top Features (XGBoost Importance):")
            for _, row in feature_imp.head(5).iterrows():
                print(f"  • {row['feature']:25s}: {row['importance']:.3f}")
        
        return xgb_clf, xgb_auc
    
    except Exception as e:
        print(f"  ⚠️ XGBoost failed: {e}")
        return None, None

def forecast_spike_probability(model, scaler, last_features, n_days=10):
    """Generate VIX spike probability forecasts with scenario analysis"""
    forecasts = []
    
    # Create scenarios
    scenarios = {
        'baseline': {'momentum_decay': 0.9, 'vol_decay': 0.95, 'sentiment_shift': 0},
        'risk_off': {'momentum_decay': 1.1, 'vol_decay': 1.05, 'sentiment_shift': -0.5},
        'risk_on': {'momentum_decay': 0.8, 'vol_decay': 0.85, 'sentiment_shift': 0.3}
    }
    
    for scenario_name, params in scenarios.items():
        scenario_forecasts = []
        current_features = last_features.copy()
        
        for day in range(1, n_days + 1):
            # Apply scenario-specific dynamics
            if 'VIX_momentum_5d' in current_features.index:
                current_features['VIX_momentum_5d'] *= params['momentum_decay']
            if 'VIX_momentum_22d' in current_features.index:
                current_features['VIX_momentum_22d'] *= (params['momentum_decay'] ** 0.5)
            if 'VIX_realized_vol' in current_features.index:
                current_features['VIX_realized_vol'] *= params['vol_decay']
            if 'sentiment' in current_features.index:
                current_features['sentiment'] += params['sentiment_shift'] / day
            
            # Scale features
            X_scaled = scaler.transform(current_features.values.reshape(1, -1))
            
            # Get probability
            if hasattr(model, 'predict'):  # statsmodels
                try:
                    X_const = sm.add_constant(X_scaled, has_constant='add')
                    prob = model.predict(X_const)[0]
                except:
                    prob = 0.5
            else:  # sklearn
                prob = model.predict_proba(X_scaled)[0, 1]
            
            scenario_forecasts.append(prob)
            
            # Update features for next iteration with some mean reversion
            current_features = current_features * 0.98 + np.random.randn(len(current_features)) * 0.001
        
        forecasts.append({
            'scenario': scenario_name,
            'probs': scenario_forecasts
        })
    
    return forecasts

def display_forecasts(forecasts, threshold):
    """Display forecasts with nice visualization"""
    print("\n" + "="*70)
    print("🔮 VIX SPIKE PROBABILITY FORECAST (10 Days)")
    print("="*70)
    
    # Find baseline scenario
    baseline = next(f for f in forecasts if f['scenario'] == 'baseline')
    
    print(f"\n📊 Baseline Scenario (VIX > {threshold})")
    print("-" * 60)
    
    for day, prob in enumerate(baseline['probs'], 1):
        bar_length = int(prob * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        
        # Risk emoji
        if prob > 0.7:
            emoji = "🔴"
            risk = "HIGH"
        elif prob > 0.3:
            emoji = "🟡"
            risk = "MODERATE"
        else:
            emoji = "🟢"
            risk = "LOW"
        
        print(f"Day {day:2d}: [{bar}] {prob:6.2%} {emoji} {risk}")
    
    # Scenario comparison
    print("\n📈 Scenario Analysis:")
    print("-" * 60)
    
    for forecast in forecasts:
        avg_prob = np.mean(forecast['probs'])
        max_prob = np.max(forecast['probs'])
        
        scenario_name = forecast['scenario'].replace('_', ' ').title()
        print(f"{scenario_name:12s}: Avg={avg_prob:6.2%}, Max={max_prob:6.2%}")
    
    # Calculate ensemble forecast
    all_probs = [f['probs'] for f in forecasts]
    ensemble_probs = np.mean(all_probs, axis=0)
    
    avg_ensemble = np.mean(ensemble_probs)
    max_ensemble = np.max(ensemble_probs)
    
    print(f"\n🎯 Ensemble Forecast:")
    print(f"  • Average spike probability: {avg_ensemble:.2%}")
    print(f"  • Maximum spike probability: {max_ensemble:.2%}")
    print(f"  • High risk days: {sum(p > 0.7 for p in ensemble_probs)}/10")
    
    # Trading recommendations
    print("\n💼 TRADING RECOMMENDATIONS:")
    if avg_ensemble > 0.5:
        print("  ⚠️ HIGH VOLATILITY EXPECTED")
        print("  • Consider: VIX calls, SPX puts, reduce equity exposure")
        print("  • Increase cash allocation")
        print("  • Review stop-losses on long positions")
        print("  • Consider volatility arbitrage strategies")
    elif avg_ensemble > 0.2:
        print("  🟡 MODERATE VOLATILITY RISK")
        print("  • Monitor positions closely")
        print("  • Consider partial hedges")
        print("  • Maintain diversification")
        print("  • Watch for entry opportunities on spikes")
    else:
        print("  🟢 LOW VOLATILITY ENVIRONMENT")
        print("  • Consider: Selling volatility (with proper risk management)")
        print("  • Maintain current positions")
        print("  • Look for entry opportunities on dips")
        print("  • Consider carry trades")

def main():
    print("="*70)
    print("VIX SPIKE PROBABILITY MODEL - ENHANCED ROBUST VERSION")
    print("="*70)
    
    # 1. Fetch market data
    df = fetch_market_data()
    
    # 2. Fetch sentiment data
    sentiment_df = fetch_gdelt_sentiment()
    
    # 3. Fetch FRED data (if available)
    fred_df = fetch_fred_data() if FRED_API_KEY else None
    
    # 4. Build features
    X, y = build_robust_features(df, sentiment_df, fred_df, threshold=VIX_SPIKE_THRESHOLD)
    
    # 5. Remove multicollinearity
    X = aggressive_multicollinearity_removal(X)
    
    # 6. Train-test split
    split_date = df.index[-int(len(df)*0.2)]
    X_train = X[:split_date]
    X_test = X[split_date:]
    y_train = y[:split_date]
    y_test = y[split_date:]
    
    print(f"\n📊 Data Split:")
    print(f"  • Train: {len(X_train)} days ({y_train.mean():.1%} spike rate)")
    print(f"  • Test:  {len(X_test)} days ({y_test.mean():.1%} spike rate)")
    
    # 7. Fit models
    logit_model, scaler, logit_auc = fit_robust_logit(X_train, y_train, X_test, y_test)
    xgb_model, xgb_auc = fit_xgboost(X_train, y_train, X_test, y_test)
    
    # 8. Select best model
    if xgb_model and xgb_auc and xgb_auc > logit_auc:
        print(f"\n✅ Using XGBoost (AUC: {xgb_auc:.4f})")
        best_model = xgb_model
        best_auc = xgb_auc
    else:
        print(f"\n✅ Using Logistic Regression (AUC: {logit_auc:.4f})")
        best_model = logit_model
        best_auc = logit_auc
    
    # 9. Current market status
    current_vix = df['VIX'].iloc[-1]
    print(f"\n📍 Current Market Status:")
    print(f"  • VIX Level: {current_vix:.2f}")
    print(f"  • Threshold: {VIX_SPIKE_THRESHOLD:.0f}")
    print(f"  • Status: {'⚠️ ABOVE THRESHOLD' if current_vix > VIX_SPIKE_THRESHOLD else '✓ Below threshold'}")
    
    # 10. Generate forecasts
    last_obs = X.iloc[-1]
    forecasts = forecast_spike_probability(best_model, scaler, last_obs, n_days=10)
    
    # 11. Display results
    display_forecasts(forecasts, VIX_SPIKE_THRESHOLD)
    
    # 12. Model confidence
    print("\n🎯 Model Performance:")
    print(f"  • Best Model ROC-AUC: {best_auc:.4f}")
    if best_auc > 0.85:
        print("  • Model confidence: HIGH ✓")
    elif best_auc > 0.75:
        print("  • Model confidence: GOOD")
    else:
        print("  • Model confidence: MODERATE (use with additional analysis)")
    
    # 13. Save model (optional)
    try:
        import joblib
        joblib.dump(best_model, OUT / 'best_vix_model.pkl')
        joblib.dump(scaler, OUT / 'scaler.pkl')
        print(f"\n💾 Model saved to {OUT}")
    except:
        pass
    
    print("\n" + "="*70)
    print("✅ Analysis Complete! Enhanced model with sentiment analysis.")
    print("="*70)

if __name__ == '__main__':
    main()