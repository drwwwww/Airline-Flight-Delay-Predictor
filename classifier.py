
from pathlib import Path
import os

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("mode.copy_on_write", True)

if os.path.exists("data/flight_delays_full.csv"):
    df = pd.read_csv("data/flight_delays_full.csv")
else:
    df = pd.read_csv("data/flight_delays_sample.csv")
    
# Datacleaning

date_cols = ["ScheduledDeparture", "ActualDeparture", "ScheduledArrival", "ActualArrival"]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

df["Delayed"] = ((df["DelayMinutes"] > 15) | (df["Cancelled"] == True)).astype(int)


X = df.drop(columns=["ActualDeparture", "ActualArrival", "DelayMinutes", "DelayReason", "Cancelled", "Diverted", "FlightID", "FlightNumber", "TailNumber"])
y = df["Delayed"]


# Pipeline Set-Up

num_cols = [
"Distance", 
"EstimatedDuration_hr", "WeatherSeverityIndex_Origin", "WeatherSeverityIndex_Dest",
"Wx_Origin_TempC", "Wx_Origin_WindKts", "Wx_Origin_VisibilityKm", "Wx_Origin_Precip", "Wx_Dest_TempC",
"Wx_Dest_WindKts", "Wx_Dest_VisibilityKm", "Wx_Dest_Precip", "Traffic_Origin_Score", "Traffic_Dest_Score",
"ATC_Restriction_Origin", "ATC_Restriction_Dest", "GateCongestion_Origin", "GateCongestion_Dest",
"TaxiOutMinutes", "TaxiInMinutes", "InboundDelayMinutes"
]

cat_cols = [
"Airline", "Origin", "Destination", "ScheduledDeparture", "ScheduledArrival", "AircraftType", "AgentQuality"
]

# Pipeline Creation

num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy="mean")),
        ('scaler',StandardScaler())
    ])

cat_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="most_frequent")),
    ('encoder',OneHotEncoder(handle_unknown="ignore"))
    ])

# Pipeline Joining

preprocessor = ColumnTransformer([
    ('num',num_pipeline,num_cols),
    ('cat',cat_pipeline,cat_cols)
])

pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',LogisticRegression(solver="saga", max_iter=2000, n_jobs=-1, class_weight="balanced"))
])

# Train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipeline.fit(X_train,y_train)

prediction = pipeline.predict(X_test)

# Test

accuracy = accuracy_score(y_test, prediction)

matrix = confusion_matrix(y_test, prediction)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Matrix: {matrix}')


plt.figure(figsize=(6,5))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted On-Time","Predicted Delayed"],
    yticklabels=["Actual On-Time","Actual Delayed"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Flight Delay Classifier — Confusion Matrix")
plt.tight_layout()
plt.show()