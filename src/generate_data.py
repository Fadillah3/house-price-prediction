import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
    'sqft_living': np.random.randint(500, 5000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.uniform(1, 4, n_samples).round(1),
    'floors': np.random.choice([1, 2, 3], n_samples),
    'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'view': np.random.randint(0, 4, n_samples),
    'condition': np.random.randint(1, 6, n_samples),
    'sqft_above': np.random.normal(1500, 400, n_samples).astype(int),
    'sqft_basement': np.random.normal(500, 200, n_samples).astype(int),
    'yr_built': np.random.randint(1900, 2024, n_samples),
    'price': np.random.normal(500000, 150000, n_samples).round(2)
}

df = pd.DataFrame(data)
df.to_csv('housing.csv', index=False)
print(f"dataset saved:{len(df)} samples")

