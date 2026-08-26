import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.solar.locations import get_city, list_cities
from src.solar.pv_model import simulate_pv, is_city_solar_cached, DEFAULT_SYSTEM
from app import app


def test_api_cities_endpoint():
    """Test GET /api/cities returns all supported Indian cities."""
    client = TestClient(app)
    resp = client.get("/api/cities")
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert data["count"] >= 50
    assert "cities" in data

    names = [c["name"] for c in data["cities"]]
    assert "Bengaluru" in names
    assert "Chennai" in names
    assert "Hyderabad" in names
    assert "Mysuru" in names
    assert "Mumbai" in names
    assert "Delhi" in names


def test_simulate_pv_with_city_coords():
    """Test pvlib simulation with Bengaluru coordinates using mock hourly weather."""
    city = get_city("Bengaluru")
    assert city is not None

    # Synthetic hourly weather for testing
    timestamps = pd.date_range("2021-01-01", periods=48, freq="h")
    mock_weather = pd.DataFrame({
        "timestamp": timestamps,
        "ghi": [0.0 if (t.hour < 6 or t.hour > 18) else 600.0 for t in timestamps],
        "dni": [0.0 if (t.hour < 6 or t.hour > 18) else 700.0 for t in timestamps],
        "dhi": [0.0 if (t.hour < 6 or t.hour > 18) else 150.0 for t in timestamps],
        "temp_air": [25.0 for _ in timestamps],
        "wind_speed": [2.5 for _ in timestamps],
    })

    sim_df = simulate_pv(
        weather_hourly=mock_weather,
        lat=city.lat,
        lon=city.lon,
        altitude_m=city.altitude_m,
        tilt_deg=city.optimal_tilt,
    )

    assert len(sim_df) == 48 * 4  # 15-min rows
    assert "solar_power_kw" in sim_df.columns
    assert "solar_energy_kwh" in sim_df.columns
    assert sim_df["solar_power_kw"].max() > 0.0


def test_api_size_with_indian_city():
    """Test POST /api/size accepts new Indian cities."""
    client = TestClient(app)
    payload = {
        "monthly_bill_inr": 2500,
        "city": "Bengaluru",
        "subsidy_eligible": True,
        "depth": "quick",
    }
    resp = client.post("/api/size", json=payload)
    # Status should be 200 OK
    assert resp.status_code == 200
    data = resp.json()
    assert "payback" in data
    assert "npv" in data
    assert "backup" in data
    assert data["payback"]["panel"] > 0
