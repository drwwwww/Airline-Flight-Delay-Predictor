from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from feature_enrich import (
    predict_delay_from_minimal_input,
    get_flight_schedules_from_adb,
    build_full_feature_row_from_minimal_input,
    predict_delay_from_full,
    try_get_schedules_with_variants,
    pick_closest_flight_by_datetime,
    process_raw_flight_from_adb,
)

app = FastAPI(title="Flight Delay Predictor API")

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    airline: str = Field(..., examples=["DL"])
    flight_number: str = Field(..., alias="flightNumber", examples=["123", "1564"])
    origin: str = Field(..., alias="departureAirport", examples=["ATL"])
    destination: str = Field(..., alias="arrivalAirport", examples=["JFK"])
    departure_datetime: Optional[str] = Field(None, examples=["2025-12-25T15:30:00Z"])
    departure_date: Optional[str] = Field(None, examples=["2025-12-25"])

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if not (req.departure_datetime or req.departure_date):
            raise HTTPException(status_code=400, detail="Provide either departure_datetime or departure_date")

        flight_num = req.flight_number.strip().upper()
        airline_code = req.airline.strip().upper()
        if flight_num.startswith(airline_code):
            flight_num = flight_num[len(airline_code):].lstrip()

        origin = req.origin.strip().upper()
        destination = req.destination.strip().upper()

        if req.departure_datetime:
            target_dt = None
            try:
                from feature_enrich import parse_iso
                target_dt = parse_iso(req.departure_datetime)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid departure_datetime format")

            departure_date = target_dt.strftime("%Y-%m-%d")
            flights, matched_code, tried = try_get_schedules_with_variants(req.airline, flight_num, departure_date)

            if not flights:
                detail = {
                    "message": f"No flights found for {req.airline}{flight_num} on {departure_date}",
                    "tried": tried,
                }
                raise HTTPException(status_code=404, detail=str(detail))

            if len(flights) > 1:
                from feature_enrich import pick_closest_flight_by_datetime
                chosen = pick_closest_flight_by_datetime(flights, target_dt)
                if chosen is None:
                    candidates = []
                    for i, f in enumerate(flights):
                        dep = f.get("departure", {})
                        arr = f.get("arrival", {})
                        scheduled = (
                            dep.get("scheduledTimeUtc")
                            or dep.get("scheduledTime", {}).get("utc")
                            or dep.get("scheduledTime", {}).get("local")
                        )
                        candidates.append({
                            "id": i,
                            "scheduled": scheduled,
                            "origin": dep.get("airport", {}).get("iata"),
                            "destination": arr.get("airport", {}).get("iata"),
                            "airline_code": matched_code,
                        })
                    return {"candidates": candidates}
                processed_flight = process_raw_flight_from_adb(chosen, matched_code or req.airline)
                full_features = build_full_feature_row_from_minimal_input(
                    airline=matched_code or req.airline,
                    flight_number=flight_num,
                    origin=origin,
                    destination=destination,
                    flight_info=processed_flight,
                )
                result = predict_delay_from_full(full_features)
                return result

            processed_flight = process_raw_flight_from_adb(flights[0], matched_code or req.airline)
            full_features = build_full_feature_row_from_minimal_input(
                airline=matched_code or req.airline,
                flight_number=flight_num,
                origin=origin,
                destination=destination,
                flight_info=processed_flight,
            )
            result = predict_delay_from_full(full_features)
            return result

        flights, matched_code, tried = try_get_schedules_with_variants(req.airline, flight_num, req.departure_date)

        if not flights:
            detail = {"message": f"No flights found for {req.airline}{flight_num} on {req.departure_date}", "tried": tried}
            raise HTTPException(status_code=404, detail=str(detail))

        if len(flights) > 1:
            candidates = []
            for i, f in enumerate(flights):
                dep = f.get("departure", {})
                arr = f.get("arrival", {})
                scheduled = (
                    dep.get("scheduledTimeUtc")
                    or dep.get("scheduledTime", {}).get("utc")
                    or dep.get("scheduledTime", {}).get("local")
                )
                candidates.append({
                    "id": i,
                    "scheduled": scheduled,
                    "origin": dep.get("airport", {}).get("iata"),
                    "destination": arr.get("airport", {}).get("iata"),
                    "airline_code": matched_code,
                })
            return {"candidates": candidates}

        processed_flight = process_raw_flight_from_adb(flights[0], matched_code or req.airline)
        full_features = build_full_feature_row_from_minimal_input(
            airline=matched_code or req.airline,
            flight_number=flight_num,
            origin=origin,
            destination=destination,
            flight_info=processed_flight,
        )
        result = predict_delay_from_full(full_features)
        return result

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))
