#!/usr/bin/env python3
"""
Train and save models for Streamlit Cloud deployment.
Run this script in an environment with Python 3.9-3.11 and numpy<2.0.

Example:
    pip install numpy scikit-learn xgboost pandas joblib
    python train_models.py
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

RF_MODEL_PATH = "rf_model.pkl"
XGB_MODEL_PATH = "xgb_model.pkl"

def prepare_ml_data(df):
    df_cleaned = df.copy()
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    df_cleaned['Year'] = df_cleaned['Date'].dt.year
    df_cleaned['Month'] = df_cleaned['Date'].dt.month
    cols_to_drop = ['Date', 'CCME_WQI', 'Area']
    df_cleaned = df_cleaned.drop(columns=[col for col in cols_to_drop if col in df_cleaned.columns])
    df_cleaned = df_cleaned.dropna(subset=['CCME_Values'])
    return df_cleaned

def train_rf_model(df):
    print("Preparing data...")
    df_cleaned = prepare_ml_data(df)
    df_rf = df_cleaned.sample(n=min(20000, len(df_cleaned)), random_state=42)

    target_col = 'CCME_Values'
    X = df_rf.drop(columns=[target_col])
    y = df_rf[target_col]

    cat_features = ['Country', 'Waterbody Type']
    num_features = [col for col in X.columns if col not in cat_features]

    print("Building Random Forest pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_pipeline.fit(X_train, y_train)

    print("Saving Random Forest model...")
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(rf_pipeline, f)
    print(f"✅ Random Forest model saved to {RF_MODEL_PATH}")

    return rf_pipeline

def train_xgb_model(df):
    print("Preparing data...")
    df_cleaned = prepare_ml_data(df)
    df_xgb = df_cleaned.sample(n=min(20000, len(df_cleaned)), random_state=42)

    target_col = 'CCME_Values'
    X = df_xgb.drop(columns=[target_col])
    y = df_xgb[target_col]

    cat_features = ['Country', 'Waterbody Type']
    num_features = [col for col in X.columns if col not in cat_features]

    print("Building XGBoost pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0))
    ])

    print("Training XGBoost model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_pipeline.fit(X_train, y_train)

    print("Saving XGBoost model...")
    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_pipeline, f)
    print(f"✅ XGBoost model saved to {XGB_MODEL_PATH}")

    return xgb_pipeline

if __name__ == "__main__":
    import os
    import urllib.request

    DATA_URL = "https://raw.githubusercontent.com/24236510-ui/wqd7012_groupwork/main/River_Water_Quality.csv"
    DATA_PATH = "River_Water_Quality.csv"

    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("✅ Dataset downloaded")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(df)} records")

    train_rf_model(df)
    train_xgb_model(df)

    print("\n🎉 Training complete! Model files created:")
    print(f"  - {RF_MODEL_PATH}")
    print(f"  - {XGB_MODEL_PATH}")
    print("\nUpload these files to your GitHub repository.")
