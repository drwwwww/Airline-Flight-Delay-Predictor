
import json
from pathlib import Path
import os

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

import pandas as pd

# Load your full dataset
df = pd.read_csv("data/flight_delays_full.csv")

# Random 1% sample (adjust to desired size)
sample_df = df.sample(frac=0.01, random_state=42)

# Save to a new file
sample_df.to_csv("data/sample_flight_delays_full.csv", index=False)

print("✅ Sample created:", sample_df.shape)


