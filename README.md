# VIX Professional Trading Dashboard

A comprehensive, professional-grade Streamlit application for advanced volatility analysis, featuring Heston stochastic volatility models, Markov Chain regime detection, and econometric forecasting.

## Features

### 1. Heston Stochastic Volatility Model
- Full implementation of the Heston model for VIX options pricing
- Calibration to market option prices using differential evolution
- Monte Carlo path simulation
- Implied volatility surface generation

### 2. Exotic Options Pricing
All exotic options priced using numpy-based Monte Carlo simulations:
- Asian Options (average price)
- Barrier Options (knock-in/knock-out)
- Lookback Options
- Digital/Binary Options

### 3. Vector Autoregression (VAR) Forecasting
- Linear relationship modeling between S&P500 lags and VIX
- Granger causality testing
- Impulse response functions
- Multi-step ahead forecasting

### 4. Markov Chain Regime Detection
- Hidden Markov Model for volatility regime classification
- Transition probability matrix estimation
- Steady-state distribution calculation
- Expected duration in each regime
- State probability forecasting

### 5. Machine Learning Classification
- Logistic Regression
- XGBoost
- Random Forest
- Comprehensive model comparison with ROC curves
- Feature importance analysis

### 6. Professional UI/UX
- Clean, emoji-free professional interface
- Dark theme optimized for trading desks
- Interactive visualizations using Plotly
- Real-time data fetching from Yahoo Finance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run vix_professional_dashboard.py
```

## Configuration

The sidebar provides comprehensive configuration options:

- **Data Parameters**: Select date range for historical analysis
- **Model Parameters**: VIX spike threshold and test/train split
- **VAR Model**: Configure lags and forecast horizon
- **Markov Chain**: Select number of volatility regimes
- **Heston Model**: Enable calibration and set simulation parameters

## Model Details

### Heston Model

The Heston model describes the evolution of asset price S and variance v:

```
dS_t = μS_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
```

Parameters:
- v0: Initial variance
- κ (kappa): Mean reversion speed
- θ (theta): Long-term variance
- σ (sigma): Volatility of volatility
- ρ (rho): Correlation between price and variance innovations

### VAR Model

Vector Autoregression modeling the relationship:
- S&P500 returns → VIX changes
- VIX changes → S&P500 returns
- Includes Granger causality testing to validate predictive relationships

### Markov Chain

Hidden Markov Model for regime detection:
- States: Low/Medium/High volatility regimes
- Features: VIX level, VIX changes, VIX volatility
- Transition probabilities estimated from historical data

## Data Sources

- **VIX**: CBOE Volatility Index
- **VVIX**: VIX of VIX
- **VXV**: 3-Month VIX futures
- **SPY**: S&P 500 ETF
- **TLT**: Treasury bonds
- **GLD**: Gold
- **HYG**: High-yield corporate bonds
- **VXX**: VIX short-term futures ETN

## Output

The dashboard provides six main analysis tabs:

1. **Overview**: Current market state and historical VIX performance
2. **Model Performance**: Classification model comparison and evaluation
3. **S&P500 to VIX Forecasting**: VAR model results and impulse responses
4. **Markov Chain Regimes**: State classification and transition analysis
5. **Heston Model & Exotics**: Stochastic volatility modeling and exotic options pricing
6. **Feature Analysis**: Feature importance and correlation analysis

## Technical Notes

- All models are production-ready with proper error handling
- No emojis or unprofessional elements in the UI
- Optimized for performance with Streamlit caching
- Robust to missing data and API failures
- Professional dark theme optimized for extended use

## Performance

- Heston calibration: ~30-60 seconds (depends on options data availability)
- VAR model fitting: ~5-10 seconds
- Markov Chain regime detection: ~10-15 seconds
- Classification models: ~10-20 seconds
- Monte Carlo simulations: ~5-30 seconds (depending on number of paths)

## License

MIT License

## Author

Professional VIX Trading Desk Analytics

## Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified financial professionals before making trading decisions.
