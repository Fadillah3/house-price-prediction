import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os

# Load dataset
df = pd.read_csv('data/housing.csv')
print(f"loaded: {len(df)} samples")

#feature engineering
features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 
            'waterfront', 'view', 'condition', 'sqft_above', 
            'sqft_basement', 'yr_built']
X = df[features]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#mLflow tracking
mlflow.set_experiment("house_price_prediction")

with mlflow.start_run() as run:
    # log parameters
    mlflow.log_params({
        "model": "RandomForestRegressor",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "test_size": 0.2
    })

    # Train model
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

#prediction
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

#calculate metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_mrse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

# log metrics
mlflow.log_metrics({
        'train_mae': train_mae,
        'test_mae': test_mae,
        'test_mrse': test_mrse,
        'test_r2': test_r2
    })

# log model
mlflow.sklearn.log_model(model, "model")

# save model locally
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/house_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print(f"Model training and logging completed.")
print(f"Test MAE: {test_mae:,.2f}")
print(f"Test R²: {test_r2:,.2f}")

# save run ID for later
with open('models/latest_run_id.txt', 'w') as f:
    f.write(run.info.run_id)

print("\nTo view MLflow UI, run:")
print("mlflow ui --port 5000")
