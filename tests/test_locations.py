import pytest
from src.solar.locations import (
    get_city,
    list_cities,
    get_city_names,
    get_city_coords,
    get_all_city_coords_dict,
    CityInfo,
)


def test_major_cities_exist():
    """Verify all major requested cities exist in the registry."""
    required = [
        "Bengaluru",
        "Chennai",
        "Hyderabad",
        "Mysuru",
        "Mumbai",
        "Delhi",
        "Kolkata",
        "Pune",
        "Ahmedabad",
        "Jaipur",
        "Lucknow",
        "Kochi",
        "Visakhapatnam",
        "Mathura",
        "Bareilly",
    ]
    city_names = get_city_names()
    for name in required:
        assert name in city_names, f"Expected {name} to be in registered cities"


def test_alias_resolution():
    """Verify common spellings and aliases resolve to the canonical CityInfo."""
    test_cases = [
        ("bangalore", "Bengaluru"),
        ("bengaluru", "Bengaluru"),
        ("BLR", "Bengaluru"),
        ("mysore", "Mysuru"),
        ("mysuru", "Mysuru"),
        ("chennai", "Chennai"),
        ("madras", "Chennai"),
        ("bombay", "Mumbai"),
        ("mumbai", "Mumbai"),
        ("calcutta", "Kolkata"),
        ("kolkata", "Kolkata"),
        ("new delhi", "Delhi"),
        ("delhi", "Delhi"),
        ("poona", "Pune"),
        ("gurgaon", "Gurugram"),
        ("allahabad", "Prayagraj"),
        ("aurangabad", "Chhatrapati Sambhajinagar"),
        ("cochin", "Kochi"),
        ("trivandrum", "Thiruvananthapuram"),
        ("vizag", "Visakhapatnam"),
        ("hubli", "Hubballi"),
    ]
    for alias, expected_name in test_cases:
        info = get_city(alias)
        assert info is not None, f"Failed to resolve alias '{alias}'"
        assert info.name == expected_name, f"Alias '{alias}' resolved to '{info.name}', expected '{expected_name}'"


def test_city_state_string_resolution():
    """Verify 'City, State' strings resolve properly."""
    info = get_city("Bengaluru, Karnataka")
    assert info is not None
    assert info.name == "Bengaluru"

    info = get_city("Chennai, Tamil Nadu")
    assert info is not None
    assert info.name == "Chennai"

    info = get_city("Mathura, Uttar Pradesh")
    assert info is not None
    assert info.name == "Mathura"


def test_city_coordinates_validity():
    """Verify all Indian cities have coordinates within India's geographical bounds."""
    all_cities = list_cities()
    assert len(all_cities) >= 50

    for c in all_cities:
        # Latitude range for India: ~8.0N to ~37.0N
        assert 8.0 <= c.lat <= 37.0, f"{c.name} latitude {c.lat} out of bounds"
        # Longitude range for India: ~68.0E to ~97.0E
        assert 68.0 <= c.lon <= 97.5, f"{c.name} longitude {c.lon} out of bounds"
        # Altitude should be non-negative and reasonable (< 3000m for cities)
        assert 0.0 <= c.altitude_m <= 3000.0, f"{c.name} altitude {c.altitude_m} out of bounds"
        # Optimal tilt should roughly match latitude
        assert 5.0 <= c.optimal_tilt <= 40.0, f"{c.name} optimal_tilt {c.optimal_tilt} out of bounds"


def test_get_city_coords_helper():
    """Test get_city_coords function returns dict with lat, lon, altitude_m."""
    coords = get_city_coords("Bengaluru")
    assert "lat" in coords and "lon" in coords and "altitude_m" in coords
    assert coords["lat"] == pytest.approx(12.9716, 0.001)

    with pytest.raises(ValueError):
        get_city_coords("NonExistentCity12345")
