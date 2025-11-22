# Architecture & Code Quality Improvements

## Critical Improvements for Production Readiness

### 1. Error Handling & Resilience

**Current Issue**: Basic try-catch blocks, unclear error messages
**Improvement**: Structured error handling with custom exceptions

```python
# errors.py - Custom exception hierarchy

class VIXDashboardError(Exception):
    """Base exception for VIX dashboard"""
    pass

class DataFetchError(VIXDashboardError):
    """Raised when data fetching fails"""
    def __init__(self, ticker, reason):
        self.ticker = ticker
        self.reason = reason
        super().__init__(f"Failed to fetch {ticker}: {reason}")

class ModelCalibrationError(VIXDashboardError):
    """Raised when model calibration fails"""
    def __init__(self, model_name, params):
        self.model_name = model_name
        self.params = params
        super().__init__(f"{model_name} calibration failed with params: {params}")

class InsufficientDataError(VIXDashboardError):
    """Raised when insufficient data for analysis"""
    def __init__(self, required, available):
        self.required = required
        self.available = available
        super().__init__(f"Need {required} data points, only {available} available")


# Usage in code

def fetch_market_data(start_date, end_date):
    """Fetch market data with proper error handling"""
    try:
        df = yf.download('^VIX', start=start_date, end=end_date)

        if len(df) < 252:  # Less than 1 year of data
            raise InsufficientDataError(required=252, available=len(df))

        return df

    except yf.errors.YFinanceError as e:
        raise DataFetchError(ticker='^VIX', reason=str(e))

    except Exception as e:
        logger.error(f"Unexpected error in fetch_market_data: {e}", exc_info=True)
        raise VIXDashboardError(f"Data fetch failed: {e}")


# In Streamlit app

try:
    df = fetch_market_data(start_date, end_date)
except InsufficientDataError as e:
    st.error(f"Insufficient data: Need {e.required} days, got {e.available}")
    st.info("Please select an earlier start date")
except DataFetchError as e:
    st.error(f"Unable to fetch {e.ticker}: {e.reason}")
    st.info("Check your internet connection or try again later")
except VIXDashboardError as e:
    st.error(f"Application error: {e}")
    logger.error(f"Dashboard error", exc_info=True)
```

---

### 2. Configuration Management

**Current Issue**: Hardcoded parameters throughout code
**Improvement**: Centralized configuration with validation

```python
# config/settings.py

from pydantic import BaseSettings, validator
from typing import List, Optional

class ModelConfig(BaseSettings):
    """Model configuration with validation"""

    # XGBoost parameters
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 200

    # Heston parameters (initial guesses)
    heston_kappa_min: float = 0.1
    heston_kappa_max: float = 10.0
    heston_theta_min: float = 0.01
    heston_theta_max: float = 1.0

    # VAR parameters
    var_max_lags: int = 10
    var_ic_criterion: str = 'aic'

    @validator('xgb_max_depth')
    def validate_max_depth(cls, v):
        if not 1 <= v <= 20:
            raise ValueError('max_depth must be between 1 and 20')
        return v

    @validator('xgb_learning_rate')
    def validate_learning_rate(cls, v):
        if not 0.001 <= v <= 1.0:
            raise ValueError('learning_rate must be between 0.001 and 1.0')
        return v

    class Config:
        env_file = '.env'
        env_prefix = 'VIX_'


class DataConfig(BaseSettings):
    """Data configuration"""

    # Tickers to fetch
    tickers: List[str] = ['^VIX', '^VVIX', '^VXV', 'SPY', 'TLT']

    # Date ranges
    default_lookback_days: int = 1825  # 5 years

    # Caching
    cache_ttl_seconds: int = 3600

    # Data quality
    min_observations: int = 252
    max_missing_pct: float = 0.05

    class Config:
        env_file = '.env'
        env_prefix = 'DATA_'


class BacktestConfig(BaseSettings):
    """Backtesting configuration"""

    initial_capital: float = 100000
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    position_size_pct: float = 10.0

    @validator('position_size_pct')
    def validate_position_size(cls, v):
        if not 1 <= v <= 50:
            raise ValueError('position_size_pct must be between 1 and 50')
        return v


# config/__init__.py

from .settings import ModelConfig, DataConfig, BacktestConfig

# Global config instances
model_config = ModelConfig()
data_config = DataConfig()
backtest_config = BacktestConfig()


# Usage in code

from config import model_config

xgb_clf = xgb.XGBClassifier(
    max_depth=model_config.xgb_max_depth,
    learning_rate=model_config.xgb_learning_rate,
    n_estimators=model_config.xgb_n_estimators
)
```

**Benefits**:
- Environment-specific configs (.env files)
- Type checking and validation
- Documentation of all parameters
- Easy testing with different configs

---

### 3. Dependency Injection

**Current Issue**: Tight coupling between components
**Improvement**: Dependency injection for testability

```python
# services/data_service.py

from abc import ABC, abstractmethod

class DataService(ABC):
    """Abstract data service interface"""

    @abstractmethod
    def fetch_ticker(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_options_chain(self, ticker: str) -> pd.DataFrame:
        pass


class YahooFinanceService(DataService):
    """Yahoo Finance implementation"""

    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager

    def fetch_ticker(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        # Check cache first
        if self.cache_manager:
            cached = self.cache_manager.get(f"{ticker}_{start}_{end}")
            if cached is not None:
                return cached

        # Fetch from API
        df = yf.download(ticker, start=start, end=end)

        # Cache result
        if self.cache_manager:
            self.cache_manager.set(f"{ticker}_{start}_{end}", df)

        return df

    def get_options_chain(self, ticker: str) -> pd.DataFrame:
        # Implementation
        pass


class MockDataService(DataService):
    """Mock service for testing"""

    def __init__(self, mock_data: dict):
        self.mock_data = mock_data

    def fetch_ticker(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return self.mock_data.get(ticker, pd.DataFrame())

    def get_options_chain(self, ticker: str) -> pd.DataFrame:
        return pd.DataFrame()


# services/model_service.py

class ModelService:
    """Model training and prediction service"""

    def __init__(self, data_service: DataService, config: ModelConfig):
        self.data_service = data_service
        self.config = config
        self.models = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all models"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use config for parameters
        xgb_clf = xgb.XGBClassifier(
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate
        )
        xgb_clf.fit(X_scaled, y)

        self.models['xgboost'] = xgb_clf

        return {'models': self.models, 'scaler': scaler}


# Usage in tests

def test_model_training():
    # Create mock data
    mock_data = {
        '^VIX': pd.DataFrame({'Close': [20, 21, 19, 22]}, index=pd.date_range('2024-01-01', periods=4))
    }

    # Inject mock service
    data_service = MockDataService(mock_data)
    model_service = ModelService(data_service, ModelConfig())

    # Test without hitting real API
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))

    result = model_service.train(X, y)

    assert 'xgboost' in result['models']
```

---

### 4. Service Layer Architecture

**Current Issue**: Business logic mixed with UI code
**Improvement**: Clean separation of concerns

```python
# Architecture layers:
#
# Presentation Layer (Streamlit UI)
#     ↓
# Service Layer (Business logic)
#     ↓
# Repository Layer (Data access)
#     ↓
# External APIs (Yahoo Finance, FRED, etc.)


# repositories/vix_repository.py

class VIXRepository:
    """Data access layer for VIX data"""

    def __init__(self, data_service: DataService):
        self.data_service = data_service

    def get_historical_data(self, start: str, end: str) -> pd.DataFrame:
        """Get historical VIX data"""
        return self.data_service.fetch_ticker('^VIX', start, end)

    def get_cross_asset_data(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """Get multiple tickers"""
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = self.data_service.fetch_ticker(ticker, start, end)['Close']
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")

        return pd.DataFrame(data)


# services/analytics_service.py

class AnalyticsService:
    """Analytics business logic"""

    def __init__(self, repository: VIXRepository, config: ModelConfig):
        self.repository = repository
        self.config = config

    def run_complete_analysis(self, start: str, end: str, threshold: float):
        """Run full analysis pipeline"""

        # 1. Fetch data
        df = self.repository.get_historical_data(start, end)

        # 2. Build features
        X = self.build_features(df)

        # 3. Train models
        y = (df['VIX'] > threshold).astype(int)
        models = self.train_models(X, y)

        # 4. Generate forecasts
        forecasts = self.generate_forecasts(models, X)

        # 5. Calculate metrics
        metrics = self.calculate_metrics(models, X, y)

        return {
            'data': df,
            'features': X,
            'models': models,
            'forecasts': forecasts,
            'metrics': metrics
        }

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering"""
        # Implementation
        pass


# app.py (Streamlit UI - presentation layer only)

def main():
    st.title("VIX Dashboard")

    # Initialize services
    data_service = YahooFinanceService()
    repository = VIXRepository(data_service)
    analytics_service = AnalyticsService(repository, model_config)

    # UI configuration
    with st.sidebar:
        start_date = st.date_input("Start")
        end_date = st.date_input("End")
        threshold = st.slider("Threshold", 15.0, 35.0, 25.0)

        if st.button("Run Analysis"):
            with st.spinner("Running analysis..."):
                # Call service layer
                results = analytics_service.run_complete_analysis(
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    threshold
                )

                # Store in session state
                st.session_state.results = results

    # Display results (UI only)
    if 'results' in st.session_state:
        display_results(st.session_state.results)
```

**Benefits**:
- Testable without UI
- Reusable business logic
- Clear responsibilities
- Easy to mock dependencies

---

### 5. Async Data Fetching

**Current Issue**: Sequential data fetching is slow
**Improvement**: Parallel async fetching

```python
# utils/async_data.py

import asyncio
import aiohttp
from typing import List, Dict

class AsyncDataFetcher:
    """Async data fetching for better performance"""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_ticker_async(self, session, ticker: str, start: str, end: str):
        """Fetch single ticker asynchronously"""
        async with self.semaphore:
            try:
                # Use async HTTP client
                url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
                params = {
                    'period1': self._date_to_timestamp(start),
                    'period2': self._date_to_timestamp(end),
                    'interval': '1d',
                    'events': 'history'
                }

                async with session.get(url, params=params) as response:
                    content = await response.text()
                    df = pd.read_csv(io.StringIO(content))
                    return ticker, df

            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                return ticker, None

    async def fetch_multiple_tickers(self, tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Fetch multiple tickers in parallel"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_ticker_async(session, ticker, start, end)
                for ticker in tickers
            ]

            results = await asyncio.gather(*tasks)

        # Convert to dict, filtering out failures
        return {ticker: df for ticker, df in results if df is not None}

    def _date_to_timestamp(self, date_str: str) -> int:
        """Convert date string to Unix timestamp"""
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())


# Usage in Streamlit

@st.cache_data(ttl=3600)
def fetch_all_market_data(start: str, end: str):
    """Fetch all market data in parallel"""
    tickers = ['^VIX', '^VVIX', '^VXV', 'SPY', 'TLT', 'GLD', 'HYG']

    fetcher = AsyncDataFetcher(max_concurrent=5)

    # Run async code in Streamlit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data = loop.run_until_complete(
        fetcher.fetch_multiple_tickers(tickers, start, end)
    )
    loop.close()

    return pd.DataFrame({k: v['Close'] for k, v in data.items()})
```

**Performance Gain**: 5-10x faster for multiple tickers

---

### 6. Database Integration

**Current Issue**: All data in memory, no persistence
**Improvement**: PostgreSQL for historical data

```python
# repositories/database.py

from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class MarketData(Base):
    """Market data table"""
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), index=True)
    date = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


class ModelPrediction(Base):
    """Store model predictions for tracking"""
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    model_name = Column(String(50))
    prediction_date = Column(DateTime, index=True)
    target_date = Column(DateTime)
    probability = Column(Float)
    actual_outcome = Column(Integer, nullable=True)


class DatabaseRepository:
    """Database access layer"""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def get_market_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Get data from DB if available, else fetch and store"""
        from sqlalchemy import and_

        # Query database
        query = self.session.query(MarketData).filter(
            and_(
                MarketData.ticker == ticker,
                MarketData.date >= start,
                MarketData.date <= end
            )
        )

        df = pd.read_sql(query.statement, self.engine)

        # If empty, fetch from API and store
        if df.empty:
            df = self.fetch_and_store(ticker, start, end)

        return df

    def fetch_and_store(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch from API and store in database"""
        # Fetch from Yahoo Finance
        df = yf.download(ticker, start=start, end=end)

        # Store in database
        for date, row in df.iterrows():
            market_data = MarketData(
                ticker=ticker,
                date=date,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume']
            )
            self.session.add(market_data)

        self.session.commit()

        return df

    def store_prediction(self, model_name: str, prediction_date: datetime,
                        target_date: datetime, probability: float):
        """Store model prediction"""
        pred = ModelPrediction(
            model_name=model_name,
            prediction_date=prediction_date,
            target_date=target_date,
            probability=probability
        )
        self.session.add(pred)
        self.session.commit()

    def update_actual_outcome(self, prediction_id: int, outcome: int):
        """Update with actual outcome for validation"""
        pred = self.session.query(ModelPrediction).get(prediction_id)
        pred.actual_outcome = outcome
        self.session.commit()


# Usage

from config import db_connection_string

db_repo = DatabaseRepository(db_connection_string)

# Get data (from DB if available)
vix_data = db_repo.get_market_data('^VIX', '2020-01-01', '2024-01-01')

# Store predictions
db_repo.store_prediction(
    model_name='XGBoost',
    prediction_date=datetime.now(),
    target_date=datetime.now() + timedelta(days=5),
    probability=0.65
)
```

**Benefits**:
- Persistent storage
- Faster repeated queries
- Track prediction history
- Validate model over time

---

### 7. Model Registry

**Current Issue**: No versioning of models
**Improvement**: MLflow for model management

```python
# services/model_registry.py

import mlflow
import mlflow.sklearn
from typing import Dict, Any

class ModelRegistry:
    """Manage model versions and experiments"""

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def log_model(self, model: Any, model_name: str, metrics: Dict[str, float],
                  params: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series):
        """Log model with MLflow"""

        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name
            )

            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                mlflow.log_table(importance_df, "feature_importance.json")

            # Log test predictions
            predictions = model.predict_proba(X_test)[:, 1]
            mlflow.log_artifact(self._save_predictions(predictions, y_test))

    def load_best_model(self, model_name: str, metric: str = "test_auc"):
        """Load best performing model version"""
        # Get all versions
        versions = self.client.search_model_versions(f"name='{model_name}'")

        # Find best based on metric
        best_version = None
        best_metric = -float('inf')

        for version in versions:
            run = self.client.get_run(version.run_id)
            metric_value = run.data.metrics.get(metric, -float('inf'))

            if metric_value > best_metric:
                best_metric = metric_value
                best_version = version

        if best_version:
            return mlflow.sklearn.load_model(f"models:/{model_name}/{best_version.version}")

        return None

    def compare_models(self, experiment_name: str):
        """Compare all models in an experiment"""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        return runs.sort_values('metrics.test_auc', ascending=False)


# Usage

registry = ModelRegistry()

# Train and log model
xgb_model = xgb.XGBClassifier(**params)
xgb_model.fit(X_train, y_train)

metrics = {
    'train_auc': roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:, 1]),
    'test_auc': roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
}

registry.log_model(
    model=xgb_model,
    model_name='VIX_Spike_XGBoost',
    metrics=metrics,
    params=params,
    X_test=X_test,
    y_test=y_test
)

# Load best model
best_model = registry.load_best_model('VIX_Spike_XGBoost')

# Compare all XGBoost runs
comparison = registry.compare_models('VIX_Spike_Prediction')
```

---

### 8. API Endpoints (FastAPI)

**Current Issue**: Streamlit only, no API access
**Improvement**: Add REST API for programmatic access

```python
# api/main.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="VIX Analytics API", version="1.0.0")

class PredictionRequest(BaseModel):
    """Request for VIX prediction"""
    date: Optional[str] = None
    features: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    """Prediction response"""
    probability: float
    model_name: str
    timestamp: datetime
    confidence_interval: List[float]

class BacktestRequest(BaseModel):
    """Backtest request"""
    start_date: str
    end_date: str
    strategy_params: Dict[str, Any]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/predict", response_model=PredictionResponse)
async def predict_vix_spike(request: PredictionRequest):
    """Get VIX spike probability"""
    try:
        # Load model
        model = registry.load_best_model('VIX_Spike_XGBoost')

        # If features provided, use them
        if request.features:
            features_df = pd.DataFrame([request.features])
        else:
            # Fetch latest features
            features_df = analytics_service.get_latest_features()

        # Predict
        probability = model.predict_proba(features_df)[0, 1]

        # Calculate confidence interval (bootstrap)
        ci = calculate_confidence_interval(model, features_df)

        return PredictionResponse(
            probability=probability,
            model_name='XGBoost',
            timestamp=datetime.now(),
            confidence_interval=ci
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-data/{ticker}")
async def get_market_data(ticker: str, start: str, end: str):
    """Get historical market data"""
    try:
        data = db_repo.get_market_data(ticker, start, end)
        return data.to_dict(orient='records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Run strategy backtest"""
    try:
        results = backtesting_service.run_backtest(
            start=request.start_date,
            end=request.end_date,
            **request.strategy_params
        )

        return {
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe'],
            'max_drawdown': results['max_dd'],
            'num_trades': results['num_trades']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn api.main:app --reload
```

**Usage**:
```bash
# Get prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-01-15"}'

# Get market data
curl "http://localhost:8000/market-data/^VIX?start=2024-01-01&end=2024-01-15"

# Run backtest
curl -X POST "http://localhost:8000/backtest" \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2023-01-01", "end_date": "2024-01-01", "strategy_params": {"threshold": 0.6}}'
```

---

## Summary: Architecture Improvements Priority

### High Priority (Implement First):
1. ✓ **Error Handling** - Prevents crashes, better UX
2. ✓ **Configuration Management** - Essential for flexibility
3. ✓ **Service Layer** - Clean code, testability

### Medium Priority:
4. ✓ **Dependency Injection** - Better testing
5. ✓ **Database Integration** - Data persistence
6. ✓ **Async Data Fetching** - Performance boost

### Lower Priority (Nice to Have):
7. ✓ **Model Registry** - MLOps best practice
8. ✓ **API Endpoints** - Programmatic access

**Implementation Order**: 1 → 2 → 3 → 4 → 6 → 5 → 7 → 8

**Estimated Time**: 3-4 weeks for items 1-6, additional 2-3 weeks for items 7-8

These architectural improvements will make the codebase:
- More maintainable
- Easier to test
- More scalable
- Production-ready
- Team-friendly
