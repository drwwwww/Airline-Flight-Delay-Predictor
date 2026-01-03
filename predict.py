from pathlib import Path
import joblib
import pandas as pd

model_path = Path(__file__).with_name("flight_delay_pipeline.joblib")

date_cols = ["ScheduledDeparture", "ScheduledArrival"]

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

feat_cols = cat_cols + num_cols

def load_pipeline():
    return joblib.load(model_path)

pipeline = load_pipeline()

def perpare_row(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in feat_cols:
        if col not in df.columns:
            df[col] = pd.NA
    
    df = df[feat_cols]

    return df

def predict_delay_from_full(data: dict) -> dict:
    
    row = perpare_row(data)

    y_pred = pipeline.predict(row)[0]
    proba = pipeline.predict_proba(row)[0, 1]

    return {
        "delayed_label": int(y_pred),        
        "delayed_bool": bool(y_pred),          
        "delayed_probability": float(proba),   
    }

if __name__ == "__main__":
    example = {
        "Airline": "DL",
        "Origin": "ATL",
        "Destination": "JFK",
        "ScheduledDeparture": "2025-12-25T15:30:00Z",
        "ScheduledArrival": "2025-12-25T18:10:00Z",
        "AircraftType": "A321",
        "AgentQuality": "MEDIUM",
        "Distance": 760.0,
        "EstimatedDuration_hr": 2.67,
        "WeatherSeverityIndex_Origin": 1.0,
        "WeatherSeverityIndex_Dest": 1.0,
        "Wx_Origin_TempC": 23.0,
        "Wx_Origin_WindKts": 8.0,
        "Wx_Origin_VisibilityKm": 10.0,
        "Wx_Origin_Precip": 0.0,
        "Wx_Dest_TempC": 20.0,
        "Wx_Dest_WindKts": 5.0,
        "Wx_Dest_VisibilityKm": 10.0,
        "Wx_Dest_Precip": 0.0,
        "Traffic_Origin_Score": 0.5,
        "Traffic_Dest_Score": 0.6,
        "ATC_Restriction_Origin": 0.0,
        "ATC_Restriction_Dest": 0.0,
        "GateCongestion_Origin": 0.4,
        "GateCongestion_Dest": 0.3,
        "TaxiOutMinutes": 15.0,
        "TaxiInMinutes": 7.0,
        "InboundDelayMinutes": 0.0,
    }

    result = predict_delay_from_full(example)
    print(result)