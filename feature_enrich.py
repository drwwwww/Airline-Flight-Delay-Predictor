import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Tuple

import requests

from predict import predict_delay_from_full

load_dotenv()

adb_api = os.getenv("ADBAPI")
weather_api = os.getenv("WEATHERAPI")

adb_url = "https://prod.api.market/api/v1"

adb_apimarket = os.getenv("ADB_APIMARKET_PREFIX", "/aedbx/aerodatabox")

adb_base_url = f"{adb_url.rstrip('/')}/{adb_apimarket.strip('/')}"

AIRLINE_ALIASES = {
    "DL": "DAL",  # Delta
    "AA": "AAL",  # American
    "UA": "UAL",  # United
    "WN": "SWA",  # Southwest
    "AS": "ASA",  # Alaska
    "NK": "NKS",  # Spirit
    "B6": "JBU",  # JetBlue
    "F9": "FFT",  # Frontier
    "HA": "HAL",  # Hawaiian
    "G4": "AAY",  # Allegiant
}

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    
    from math import radians, sin, cos, asin, sqrt

    R = 6371.0 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def parse_iso(dt_str: str) -> datetime:
    """
    Parse ISO-like datetime strings and normalize common variants returned by ADB.
    Handles:
      - non-zero-padded dates like '2025-1-7T15:30:00Z'
      - space-separated date/time like '2026-01-02 22:30-05:00' or '2026-01-03 03:30Z'
      - times without seconds like '22:30' (adds :00)
      - trailing 'Z' (converted to +00:00)
    Returns a timezone-aware datetime when possible.
    """
    s = dt_str.strip()

    s = s.replace("Z", "+00:00")

    if "T" not in s and " " in s:
        s = s.replace(" ", "T", 1)

    if "T" in s:
        date_part, time_part = s.split("T", 1)
        parts = date_part.split("-")
        if len(parts) == 3:
            y, m, d = parts
            m = m.zfill(2)
            d = d.zfill(2)
        else:
            raise ValueError(f"Invalid date format: {date_part!r}")

        tz_pos = None
        for sep in ("+", "-"):
            pos = time_part.find(sep)
            if pos != -1:
                tz_pos = pos
                break

        if tz_pos is not None:
            core = time_part[:tz_pos]
            tz = time_part[tz_pos:]
        else:
            core = time_part
            tz = ""

        if core.count(":") == 1:
            core = core + ":00"

        s = f"{y}-{m}-{d}T{core}{tz}"

    return datetime.fromisoformat(s)


def round_to_nearest_half_hour(dt: datetime) -> datetime:
    """Round a datetime to the nearest half hour.

    Examples:
    - 07:43 -> 07:30
    - 07:46 -> 08:00
    Preserves tzinfo if present.
    """
    minute = dt.minute
    second = dt.second + dt.microsecond / 1_000_000
    total_minutes = minute + second / 60.0

    rem = total_minutes % 30
    if rem < 15:
        delta_minutes = -rem
    else:
        delta_minutes = 30 - rem

    from datetime import timedelta

    rounded = dt + timedelta(minutes=delta_minutes)
    rounded = rounded.replace(second=0, microsecond=0)
    return rounded


def get_flight_schedule_from_adb(
    airline: str,
    flight_number: str,
    departure_date: str,
) -> dict:
    
    if not adb_api:
        raise RuntimeError("ADB_API_KEY environment variable is not set.")

    headers = {
        "x-api-market-key": adb_api,
        "accept": "application/json",
}
    url = f"{adb_base_url}/flights/0/{airline}{flight_number}/{departure_date}"


    params = {
        "dateLocalRole": "Departure",
        "withLocation": "true",
        "withAircraftImage": "false",
}


    response = requests.get(url, headers=headers, params=params, timeout=10)
    print("STATUS:", response.status_code)
    print("URL:", response.url)
    print("CONTENT-TYPE:", response.headers.get("Content-Type"))
    print("BODY (first 500 chars):", response.text[:500])

    if response.status_code == 204 or not response.content:
        raise ValueError(
            f"No flight data returned from AeroDataBox (status={response.status_code}) "
            f"for {airline}{flight_number} on {departure_date}"
        )

    response.raise_for_status()

    ct = (response.headers.get("Content-Type") or "").lower()
    if "application/json" not in ct:
        raise RuntimeError(
        f"ADB returned non-JSON. status={response.status_code} content_type={ct} "
        f"body_start={response.text[:200]!r}"
    )

    data = response.json()

    if not data:
        raise ValueError("No flight data returned from AeroDataBox.")

    flight = data[0]

    dep = flight["departure"]
    arr = flight["arrival"]

    scheduled_departure = (
        dep.get("scheduledTimeUtc")
        or dep.get("scheduledTime", {}).get("utc")
        or dep.get("scheduledTime", {}).get("local")
    )
    scheduled_arrival = (
        arr.get("scheduledTimeUtc")
        or arr.get("scheduledTime", {}).get("utc")
        or arr.get("scheduledTime", {}).get("local")
    )

    if not scheduled_departure or not scheduled_arrival:
        raise ValueError(
            f"Unexpected ADB response format: missing scheduled times. dep keys={list(dep.keys())} arr keys={list(arr.keys())}"
        )

    origin_iata = dep["airport"]["iata"]
    dest_iata = arr["airport"]["iata"]

    origin_lat = dep["airport"]["location"]["lat"]
    origin_lon = dep["airport"]["location"]["lon"]
    dest_lat = arr["airport"]["location"]["lat"]
    dest_lon = arr["airport"]["location"]["lon"]

    aircraft = flight.get("aircraft")
    if isinstance(aircraft, dict):
        model = aircraft.get("model")
        if isinstance(model, dict):
            aircraft_type = model.get("icao") or model.get("name") or aircraft.get("icao") or "UNKNOWN"
        else:
            aircraft_type = model or aircraft.get("icao") or "UNKNOWN"
    elif isinstance(aircraft, str):
        aircraft_type = aircraft
    else:
        aircraft_type = "UNKNOWN"

    distance_km = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    est_duration_hours = (
        parse_iso(scheduled_arrival) - parse_iso(scheduled_departure)
    ).total_seconds() / 3600.0

    return {
        "Airline": airline,
        "Origin": origin_iata,
        "Destination": dest_iata,
        "ScheduledDeparture": scheduled_departure,
        "ScheduledArrival": scheduled_arrival,
        "AircraftType": aircraft_type,
        "Distance": distance_km,
        "EstimatedDuration_hr": est_duration_hours,
        "origin_coords": (origin_lat, origin_lon),
        "dest_coords": (dest_lat, dest_lon),
    }

def get_flight_schedules_from_adb(
    airline: str,
    flight_number: str,
    departure_date: str,
) -> list:
    """Return the raw list of flight objects from ADB for the given date.

    This function is tolerant of 204 / empty responses and will return an empty
    list instead of raising to allow callers to handle 'no data' cases.
    """
    if not adb_api:
        raise RuntimeError("ADB_API_KEY environment variable is not set.")

    headers = {
        "x-api-market-key": adb_api,
        "accept": "application/json",
    }
    url = f"{adb_base_url}/flights/0/{airline}{flight_number}/{departure_date}"

    params = {
        "dateLocalRole": "Departure",
        "withLocation": "true",
        "withAircraftImage": "false",
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    if response.status_code == 204 or not response.content:
        return []

    response.raise_for_status()

    ct = (response.headers.get("Content-Type") or "").lower()
    if "application/json" not in ct:
        raise RuntimeError(
            f"ADB returned non-JSON. status={response.status_code} content_type={ct} "
            f"body_start={response.text[:200]!r}"
        )

    data = response.json()
    if not data:
        return []

    return data


def try_get_schedules_with_variants(airline: str, flight_number: str, departure_date: str):
    """Try the provided airline and common ICAO variants, return (flights, matched_code, tried).

    'tried' is a list of tuples (code, status, info) for diagnostics.
    """
    codes = []
    provided = (airline or "").strip().upper()
    if provided:
        codes.append(provided)

    alias = AIRLINE_ALIASES.get(provided)
    if alias and alias not in codes:
        codes.append(alias)

    if len(provided) == 2:
        derived = provided + "AL"
        if derived not in codes:
            codes.append(derived)

    tried = []
    for code in codes:
        try:
            flights = get_flight_schedules_from_adb(code, flight_number, departure_date)
        except Exception as e:
            tried.append((code, "error", str(e)))
            continue
        if flights:
            return flights, code, tried
        tried.append((code, "empty", None))

    return [], None, tried


def pick_closest_flight_by_datetime(flights: list, target_dt: datetime):
    """Pick the flight whose scheduled departure is closest to target_dt."""
    best = None
    best_delta = None
    for f in flights:
        dep = f.get("departure", {})
        scheduled = (
            dep.get("scheduledTimeUtc")
            or dep.get("scheduledTime", {}).get("utc")
            or dep.get("scheduledTime", {}).get("local")
        )
        if not scheduled:
            continue
        try:
            sdt = parse_iso(scheduled)
        except Exception:
            continue
        delta = abs((sdt - target_dt).total_seconds())
        if best is None or delta < best_delta:
            best = f
            best_delta = delta
    return best



def get_airport_delay_index_from_adb(airport_iata: str) -> dict:

    if not adb_api:
        raise RuntimeError("ADB_API_KEY environment variable is not set.")

    headers = {
        "x-api-market-key": adb_api,
        "accept": "application/json",
    }

    url = f"{adb_base_url}/airports/iata/{airport_iata}/delays"

    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 404:
        return {
            "Traffic_Score": 0.5,
            "ATC_Restriction": 0.0,
            "GateCongestion": 0.5,
        }

    response.raise_for_status()
    data = response.json()

    delay_index = data.get("delayIndex", 2.5)

    traffic_score = max(0.0, min(delay_index / 5.0, 1.0))

    return {
        "Traffic_Score": traffic_score,
        "ATC_Restriction": 1.0 if delay_index > 3.5 else 0.0,
        "GateCongestion": 0.3 + 0.7 * traffic_score,
    }


def get_weather_features_from_weatherapi(lat: float, lon: float, when_iso: str) -> dict:

    if not weather_api:
        return {
            "temp_c": 20.0,
            "wind_kts": 10.0,
            "visibility_km": 10.0,
            "precip_mm": 0.0,
        }

    base_url = "https://api.weatherapi.com/v1/forecast.json"

    dt = parse_iso(when_iso)
    date_str = dt.strftime("%Y-%m-%d")
    hour = dt.hour

    params = {
        "key": weather_api,
        "q": f"{lat},{lon}",
        "dt": date_str,
        "aqi": "no",
        "alerts": "no",
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        forecastdays = data.get("forecast", {}).get("forecastday", [])
        if not forecastdays:
            print(f"Warning: No forecast data from WeatherAPI for {date_str}")
            return {
                "temp_c": 20.0,
                "wind_kts": 10.0,
                "visibility_km": 10.0,
                "precip_mm": 0.0,
            }
        
        hours = forecastdays[0]["hour"]

        target_epoch = int(dt.timestamp())
        best_hour = min(hours, key=lambda h: abs(h["time_epoch"] - target_epoch))

        temp_c = best_hour["temp_c"]
        wind_kph = best_hour["wind_kph"]
        vis_km = best_hour["vis_km"]
        precip_mm = best_hour["precip_mm"]

        wind_kts = wind_kph * 0.539957

        return {
            "temp_c": temp_c,
            "wind_kts": wind_kts,
            "visibility_km": vis_km,
            "precip_mm": precip_mm,
        }
    except Exception as e:
        print(f"Warning: WeatherAPI request failed: {e}. Using default weather values.")
        return {
            "temp_c": 20.0,
            "wind_kts": 10.0,
            "visibility_km": 10.0,
            "precip_mm": 0.0,
        }



def compute_weather_severity(
    temp_c: float,
    wind_kts: float,
    visibility_km: float,
    precip_mm: float,
) -> float:

    score = 0.0

    if wind_kts > 20:
        score += 1.0
    if visibility_km < 5:
        score += 1.0
    if precip_mm > 0.0:
        score += 1.0
    if temp_c < -5 or temp_c > 35:
        score += 1.0

    return float(min(score, 5.0))


def process_raw_flight_from_adb(flight: dict, airline: str) -> dict:
    """Convert raw ADB flight object to internal format with origin_coords and dest_coords.
    
    This processes the raw response from ADB API into a normalized format that
    build_full_feature_row_from_minimal_input expects.
    """
    dep = flight["departure"]
    arr = flight["arrival"]

    scheduled_departure = (
        dep.get("scheduledTimeUtc")
        or dep.get("scheduledTime", {}).get("utc")
        or dep.get("scheduledTime", {}).get("local")
    )
    scheduled_arrival = (
        arr.get("scheduledTimeUtc")
        or arr.get("scheduledTime", {}).get("utc")
        or arr.get("scheduledTime", {}).get("local")
    )

    if not scheduled_departure or not scheduled_arrival:
        raise ValueError(
            f"Unexpected ADB response format: missing scheduled times. dep keys={list(dep.keys())} arr keys={list(arr.keys())}"
        )

    origin_iata = dep["airport"]["iata"]
    dest_iata = arr["airport"]["iata"]

    origin_lat = dep["airport"]["location"]["lat"]
    origin_lon = dep["airport"]["location"]["lon"]
    dest_lat = arr["airport"]["location"]["lat"]
    dest_lon = arr["airport"]["location"]["lon"]

    aircraft = flight.get("aircraft")
    if isinstance(aircraft, dict):
        model = aircraft.get("model")
        if isinstance(model, dict):
            aircraft_type = model.get("icao") or model.get("name") or aircraft.get("icao") or "UNKNOWN"
        else:
            aircraft_type = model or aircraft.get("icao") or "UNKNOWN"
    elif isinstance(aircraft, str):
        aircraft_type = aircraft
    else:
        aircraft_type = "UNKNOWN"

    distance_km = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    est_duration_hours = (
        parse_iso(scheduled_arrival) - parse_iso(scheduled_departure)
    ).total_seconds() / 3600.0

    return {
        "Airline": airline,
        "Origin": origin_iata,
        "Destination": dest_iata,
        "ScheduledDeparture": scheduled_departure,
        "ScheduledArrival": scheduled_arrival,
        "AircraftType": aircraft_type,
        "Distance": distance_km,
        "EstimatedDuration_hr": est_duration_hours,
        "origin_coords": (origin_lat, origin_lon),
        "dest_coords": (dest_lat, dest_lon),
    }

def build_full_feature_row_from_minimal_input(
    airline: str,
    flight_number: str,
    origin: str,
    destination: str,
    departure_datetime: str = None,
    departure_date: str = None,
    flight_info: dict = None,
) -> dict:

    if flight_info is None:
        if departure_datetime:
            dt = parse_iso(departure_datetime)
            rounded_dt = round_to_nearest_half_hour(dt)
            print(f"Input departure datetime: {dt.isoformat()}, rounded to: {rounded_dt.isoformat()}")
            departure_date = rounded_dt.strftime("%Y-%m-%d")
        elif departure_date:
            departure_date = departure_date
        else:
            raise ValueError("Either departure_datetime, departure_date, or flight_info must be provided")

        flight_info = get_flight_schedule_from_adb(
            airline=airline,
            flight_number=flight_number,
            departure_date=departure_date,
        )


    origin_lat, origin_lon = flight_info["origin_coords"]
    dest_lat, dest_lon = flight_info["dest_coords"]

    origin_weather = get_weather_features_from_weatherapi(
        origin_lat, origin_lon, flight_info["ScheduledDeparture"]
    )
    dest_weather = get_weather_features_from_weatherapi(
        dest_lat, dest_lon, flight_info["ScheduledArrival"]
    )

    origin_severity = compute_weather_severity(
        origin_weather["temp_c"],
        origin_weather["wind_kts"],
        origin_weather["visibility_km"],
        origin_weather["precip_mm"],
    )
    dest_severity = compute_weather_severity(
        dest_weather["temp_c"],
        dest_weather["wind_kts"],
        dest_weather["visibility_km"],
        dest_weather["precip_mm"],
    )

    origin_delay = get_airport_delay_index_from_adb(flight_info["Origin"])
    dest_delay = get_airport_delay_index_from_adb(flight_info["Destination"])

    agent_quality = "MEDIUM"
    taxi_out_minutes = 15.0
    taxi_in_minutes = 7.0
    inbound_delay_minutes = 0.0

    full_features = {
        "Airline": flight_info["Airline"],
        "Origin": flight_info["Origin"],
        "Destination": flight_info["Destination"],
        "ScheduledDeparture": flight_info["ScheduledDeparture"],
        "ScheduledArrival": flight_info["ScheduledArrival"],
        "AircraftType": flight_info["AircraftType"],
        "AgentQuality": agent_quality,

        "Distance": flight_info["Distance"],
        "EstimatedDuration_hr": flight_info["EstimatedDuration_hr"],

        "WeatherSeverityIndex_Origin": origin_severity,
        "WeatherSeverityIndex_Dest": dest_severity,

        "Wx_Origin_TempC": origin_weather["temp_c"],
        "Wx_Origin_WindKts": origin_weather["wind_kts"],
        "Wx_Origin_VisibilityKm": origin_weather["visibility_km"],
        "Wx_Origin_Precip": origin_weather["precip_mm"],

        "Wx_Dest_TempC": dest_weather["temp_c"],
        "Wx_Dest_WindKts": dest_weather["wind_kts"],
        "Wx_Dest_VisibilityKm": dest_weather["visibility_km"],
        "Wx_Dest_Precip": dest_weather["precip_mm"],

        "Traffic_Origin_Score": origin_delay["Traffic_Score"],
        "Traffic_Dest_Score": dest_delay["Traffic_Score"],
        "ATC_Restriction_Origin": origin_delay["ATC_Restriction"],
        "ATC_Restriction_Dest": dest_delay["ATC_Restriction"],
        "GateCongestion_Origin": origin_delay["GateCongestion"],
        "GateCongestion_Dest": dest_delay["GateCongestion"],

        "TaxiOutMinutes": taxi_out_minutes,
        "TaxiInMinutes": taxi_in_minutes,
        "InboundDelayMinutes": inbound_delay_minutes,
    }

    return full_features


def predict_delay_from_minimal_input(
    airline: str,
    flight_number: str,
    origin: str,
    destination: str,
    departure_datetime: str,
) -> dict:
    
    full_features = build_full_feature_row_from_minimal_input(
        airline=airline,
        flight_number=flight_number,
        origin=origin,
        destination=destination,
        departure_datetime=departure_datetime,
    )

    result = predict_delay_from_full(full_features)
    return result


if __name__ == "__main__":
    try:
        test_result = predict_delay_from_minimal_input(
            airline="DAL",
            flight_number="865",
            origin="ATL",
            destination="BOS",
            departure_datetime="2026-01-02T22:30:00Z",
        )
        print(test_result)
    except Exception as e:
        print(f"Live enrichment failed: {e}")
        raise


