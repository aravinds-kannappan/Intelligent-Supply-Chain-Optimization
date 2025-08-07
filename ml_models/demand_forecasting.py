"""
Advanced demand forecasting using XGBoost and time series analysis
Demonstrates ML algorithms for supply chain optimization
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemandForecastingModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.model_metrics = {}
        
    def create_features(self, df):
        """Create advanced features for demand forecasting"""
        df = df.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_holiday_season'] = ((df['month'] == 11) | (df['month'] == 12)).astype(int)
        df['is_new_year'] = ((df['month'] == 1) | (df['month'] == 2)).astype(int)
        
        # Cyclical encoding for seasonal patterns
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Lag features (previous demand patterns)
        df = df.sort_values(['product_id', 'facility_id', 'date'])
        
        for lag in [1, 7, 30, 90]:
            df[f'demand_lag_{lag}'] = df.groupby(['product_id', 'facility_id'])['actual_demand'].shift(lag)
        
        # Rolling statistics
        for window in [7, 30, 90]:
            df[f'demand_rolling_mean_{window}'] = df.groupby(['product_id', 'facility_id'])['actual_demand'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'demand_rolling_std_{window}'] = df.groupby(['product_id', 'facility_id'])['actual_demand'].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
        
        # Demand volatility
        df['demand_volatility'] = df.groupby(['product_id', 'facility_id'])['actual_demand'].rolling(window=30, min_periods=1).std().reset_index(0, drop=True)
        
        # Growth rate
        df['demand_growth_rate'] = df.groupby(['product_id', 'facility_id'])['actual_demand'].pct_change(periods=30).fillna(0)
        
        # Forecast error features
        df['forecast_error'] = df['forecasted_demand'] - df['actual_demand']
        df['forecast_error_abs'] = np.abs(df['forecast_error'])
        df['forecast_bias'] = df.groupby(['product_id', 'facility_id'])['forecast_error'].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
        
        return df
    
    def prepare_data(self, demand_df, products_df, facilities_df):
        """Prepare and merge data for training"""
        # Merge with product and facility information
        df = demand_df.merge(products_df, on='product_id', how='left')
        df = df.merge(facilities_df, on='facility_id', how='left')
        
        # Create features
        df = self.create_features(df)
        
        # Encode categorical variables
        categorical_cols = ['product_id', 'facility_id', 'category', 'lifecycle_stage', 
                           'type', 'region', 'demand_driver']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def select_features(self, df):
        """Select relevant features for the model"""
        feature_cols = [
            # Time features
            'year', 'month', 'quarter', 'day_of_year', 'week_of_year',
            'is_holiday_season', 'is_new_year',
            'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
            
            # Lag features
            'demand_lag_1', 'demand_lag_7', 'demand_lag_30', 'demand_lag_90',
            
            # Rolling statistics
            'demand_rolling_mean_7', 'demand_rolling_mean_30', 'demand_rolling_mean_90',
            'demand_rolling_std_7', 'demand_rolling_std_30', 'demand_rolling_std_90',
            
            # Volatility and growth
            'demand_volatility', 'demand_growth_rate',
            
            # Forecast features
            'forecasted_demand', 'forecast_accuracy', 'confidence_interval',
            'forecast_bias',
            
            # Product features
            'base_cost', 'complexity_score', 'weight_kg', 'environmental_impact',
            
            # Facility features
            'capacity', 'utilization_rate', 'operational_cost_per_unit',
            'energy_efficiency', 'automation_level',
            
            # Encoded categorical features
            'product_id_encoded', 'facility_id_encoded', 'category_encoded',
            'lifecycle_stage_encoded', 'type_encoded', 'region_encoded',
            'demand_driver_encoded'
        ]
        
        # Filter only existing columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        return df[available_features].fillna(0)
    
    def train_model(self, X, y):
        """Train XGBoost model with hyperparameter optimization"""
        logger.info("Training demand forecasting model...")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # XGBoost parameters optimized for demand forecasting
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = xgb.XGBRegressor(**params)
        
        # Cross-validation scores
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train_cv, y_train_cv)
            y_pred_cv = self.model.predict(X_val_cv)
            cv_scores.append(mean_absolute_error(y_val_cv, y_pred_cv))
        
        logger.info(f"Cross-validation MAE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
        
        # Train final model on all data
        self.model.fit(X, y)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Model training completed")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        self.model_metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        logger.info(f"Model Performance:")
        logger.info(f"MAE: {self.model_metrics['mae']:.2f}")
        logger.info(f"RMSE: {self.model_metrics['rmse']:.2f}")
        logger.info(f"R²: {self.model_metrics['r2']:.4f}")
        logger.info(f"MAPE: {self.model_metrics['mape']:.2f}%")
        
        return self.model_metrics
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save trained model and encoders"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'model_metrics': self.model_metrics
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and encoders"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data.get('feature_importance')
        self.model_metrics = model_data.get('model_metrics', {})
        
        logger.info(f"Model loaded from {filepath}")

def train_demand_forecasting_model():
    """Main function to train the demand forecasting model"""
    # Load data
    demand_df = pd.read_csv('data/demand_forecast.csv')
    products_df = pd.read_csv('data/products.csv')
    facilities_df = pd.read_csv('data/facilities.csv')
    
    # Initialize model
    model = DemandForecastingModel()
    
    # Prepare data
    df = model.prepare_data(demand_df, products_df, facilities_df)
    
    # Select features and target
    X = model.select_features(df)
    y = df['actual_demand']
    
    # Remove rows with missing target values
    mask = ~y.isna()
    X, y = X[mask], y[mask]
    
    # Split data (time-aware split)
    df_sorted = df[mask].sort_values('date')
    split_date = df_sorted['date'].quantile(0.8)
    
    train_mask = df_sorted['date'] <= split_date
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Train model
    model.train_model(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate_model(X_test, y_test)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_model('models/demand_forecasting_model.pkl')
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = train_demand_forecasting_model()
    print("Demand forecasting model training completed!")
    print(f"Top 10 most important features:")
    print(model.feature_importance.head(10))