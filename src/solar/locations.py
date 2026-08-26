"""
IEMS – Indian Cities & Solar Location Registry
===============================================
Centralized database of major Indian metropolitan and tier-1/tier-2 cities
with precise geographical coordinates (lat/lon), altitudes, state associations,
and optimal solar panel tilt angles for NASA POWER API data fetching.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass(frozen=True)
class CityInfo:
    name: str
    state: str
    lat: float
    lon: float
    altitude_m: float
    optimal_tilt: float
    aliases: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return f"{self.name}, {self.state}"


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive Major Indian Cities Database
# ─────────────────────────────────────────────────────────────────────────────
_INDIAN_CITIES_RAW: List[CityInfo] = [
    # ── South India ───────────────────────────────────────────────────────────
    CityInfo(
        name="Bengaluru",
        state="Karnataka",
        lat=12.9716,
        lon=77.5946,
        altitude_m=920.0,
        optimal_tilt=13.0,
        aliases=("bangalore", "bengaluru", "blr"),
    ),
    CityInfo(
        name="Chennai",
        state="Tamil Nadu",
        lat=13.0827,
        lon=80.2707,
        altitude_m=6.7,
        optimal_tilt=13.0,
        aliases=("chennai", "madras", "maa"),
    ),
    CityInfo(
        name="Hyderabad",
        state="Telangana",
        lat=17.3850,
        lon=78.4867,
        altitude_m=542.0,
        optimal_tilt=17.0,
        aliases=("hyderabad", "hyd", "secunderabad"),
    ),
    CityInfo(
        name="Mysuru",
        state="Karnataka",
        lat=12.2958,
        lon=76.6394,
        altitude_m=763.0,
        optimal_tilt=12.0,
        aliases=("mysore", "mysuru", "myq"),
    ),
    CityInfo(
        name="Kochi",
        state="Kerala",
        lat=9.9312,
        lon=76.2673,
        altitude_m=4.0,
        optimal_tilt=10.0,
        aliases=("kochi", "cochin", "cok"),
    ),
    CityInfo(
        name="Thiruvananthapuram",
        state="Kerala",
        lat=8.5241,
        lon=76.9366,
        altitude_m=10.0,
        optimal_tilt=9.0,
        aliases=("thiruvananthapuram", "trivandrum", "trv"),
    ),
    CityInfo(
        name="Visakhapatnam",
        state="Andhra Pradesh",
        lat=17.6868,
        lon=83.2185,
        altitude_m=45.0,
        optimal_tilt=18.0,
        aliases=("visakhapatnam", "vizag", "vtz"),
    ),
    CityInfo(
        name="Coimbatore",
        state="Tamil Nadu",
        lat=11.0168,
        lon=76.9558,
        altitude_m=411.0,
        optimal_tilt=11.0,
        aliases=("coimbatore", "kovai", "cjb"),
    ),
    CityInfo(
        name="Madurai",
        state="Tamil Nadu",
        lat=9.9252,
        lon=78.1198,
        altitude_m=101.0,
        optimal_tilt=10.0,
        aliases=("madurai", "ixm"),
    ),
    CityInfo(
        name="Vijayawada",
        state="Andhra Pradesh",
        lat=16.5062,
        lon=80.6480,
        altitude_m=11.0,
        optimal_tilt=16.0,
        aliases=("vijayawada", "bezawada", "vga"),
    ),
    CityInfo(
        name="Mangaluru",
        state="Karnataka",
        lat=12.9141,
        lon=74.8560,
        altitude_m=22.0,
        optimal_tilt=13.0,
        aliases=("mangaluru", "mangalore", "ixe"),
    ),
    CityInfo(
        name="Hubballi",
        state="Karnataka",
        lat=15.3647,
        lon=75.1240,
        altitude_m=671.0,
        optimal_tilt=15.0,
        aliases=("hubballi", "hubli", "hubli-dharwad", "hbxb"),
    ),
    CityInfo(
        name="Kozhikode",
        state="Kerala",
        lat=11.2588,
        lon=75.7804,
        altitude_m=1.0,
        optimal_tilt=11.0,
        aliases=("kozhikode", "calicut", "ccj"),
    ),

    # ── North India ───────────────────────────────────────────────────────────
    CityInfo(
        name="Delhi",
        state="Delhi (NCR)",
        lat=28.6139,
        lon=77.2090,
        altitude_m=216.0,
        optimal_tilt=28.0,
        aliases=("delhi", "new delhi", "del", "ncr"),
    ),
    CityInfo(
        name="Mathura",
        state="Uttar Pradesh",
        lat=27.4924,
        lon=77.6737,
        altitude_m=182.0,
        optimal_tilt=27.0,
        aliases=("mathura",),
    ),
    CityInfo(
        name="Bareilly",
        state="Uttar Pradesh",
        lat=28.3670,
        lon=79.4304,
        altitude_m=172.0,
        optimal_tilt=28.0,
        aliases=("bareilly", "bareli"),
    ),
    CityInfo(
        name="Lucknow",
        state="Uttar Pradesh",
        lat=26.8467,
        lon=80.9462,
        altitude_m=123.0,
        optimal_tilt=27.0,
        aliases=("lucknow", "lko"),
    ),
    CityInfo(
        name="Kanpur",
        state="Uttar Pradesh",
        lat=26.4499,
        lon=80.3319,
        altitude_m=126.0,
        optimal_tilt=26.0,
        aliases=("kanpur", "kbp"),
    ),
    CityInfo(
        name="Varanasi",
        state="Uttar Pradesh",
        lat=25.3176,
        lon=82.9739,
        altitude_m=81.0,
        optimal_tilt=25.0,
        aliases=("varanasi", "banaras", "kashi", "vns"),
    ),
    CityInfo(
        name="Agra",
        state="Uttar Pradesh",
        lat=27.1767,
        lon=78.0081,
        altitude_m=171.0,
        optimal_tilt=27.0,
        aliases=("agra", "agr"),
    ),
    CityInfo(
        name="Noida",
        state="Uttar Pradesh",
        lat=28.5355,
        lon=77.3910,
        altitude_m=200.0,
        optimal_tilt=28.0,
        aliases=("noida", "greater noida"),
    ),
    CityInfo(
        name="Gurugram",
        state="Haryana",
        lat=28.4595,
        lon=77.0266,
        altitude_m=217.0,
        optimal_tilt=28.0,
        aliases=("gurugram", "gurgaon"),
    ),
    CityInfo(
        name="Chandigarh",
        state="Chandigarh (UT)",
        lat=30.7333,
        lon=76.7794,
        altitude_m=321.0,
        optimal_tilt=30.0,
        aliases=("chandigarh", "ixc"),
    ),
    CityInfo(
        name="Jaipur",
        state="Rajasthan",
        lat=26.9124,
        lon=75.7873,
        altitude_m=431.0,
        optimal_tilt=27.0,
        aliases=("jaipur", "jai", "pink city"),
    ),
    CityInfo(
        name="Jodhpur",
        state="Rajasthan",
        lat=26.2389,
        lon=73.0243,
        altitude_m=231.0,
        optimal_tilt=26.0,
        aliases=("jodhpur", "jdh"),
    ),
    CityInfo(
        name="Udaipur",
        state="Rajasthan",
        lat=24.5854,
        lon=73.7125,
        altitude_m=598.0,
        optimal_tilt=25.0,
        aliases=("udaipur", "udr"),
    ),
    CityInfo(
        name="Amritsar",
        state="Punjab",
        lat=31.6340,
        lon=74.8723,
        altitude_m=234.0,
        optimal_tilt=31.0,
        aliases=("amritsar", "atq"),
    ),
    CityInfo(
        name="Ludhiana",
        state="Punjab",
        lat=30.9010,
        lon=75.8573,
        altitude_m=244.0,
        optimal_tilt=31.0,
        aliases=("ludhiana", "ldh"),
    ),
    CityInfo(
        name="Dehradun",
        state="Uttarakhand",
        lat=30.3165,
        lon=78.0322,
        altitude_m=435.0,
        optimal_tilt=30.0,
        aliases=("dehradun", "ded"),
    ),
    CityInfo(
        name="Prayagraj",
        state="Uttar Pradesh",
        lat=25.4358,
        lon=81.8463,
        altitude_m=98.0,
        optimal_tilt=25.0,
        aliases=("prayagraj", "allahabad", "ixd"),
    ),
    CityInfo(
        name="Srinagar",
        state="Jammu & Kashmir",
        lat=34.0837,
        lon=74.7973,
        altitude_m=1585.0,
        optimal_tilt=34.0,
        aliases=("srinagar", "svr"),
    ),

    # ── West India ────────────────────────────────────────────────────────────
    CityInfo(
        name="Mumbai",
        state="Maharashtra",
        lat=19.0760,
        lon=72.8777,
        altitude_m=14.0,
        optimal_tilt=19.0,
        aliases=("mumbai", "bombay", "bom"),
    ),
    CityInfo(
        name="Pune",
        state="Maharashtra",
        lat=18.5204,
        lon=73.8567,
        altitude_m=560.0,
        optimal_tilt=18.0,
        aliases=("pune", "poona", "pnq"),
    ),
    CityInfo(
        name="Ahmedabad",
        state="Gujarat",
        lat=23.0225,
        lon=72.5714,
        altitude_m=53.0,
        optimal_tilt=23.0,
        aliases=("ahmedabad", "amdavad", "amd"),
    ),
    CityInfo(
        name="Surat",
        state="Gujarat",
        lat=21.1702,
        lon=72.8311,
        altitude_m=13.0,
        optimal_tilt=21.0,
        aliases=("surat", "stv"),
    ),
    CityInfo(
        name="Vadodara",
        state="Gujarat",
        lat=22.3072,
        lon=73.1812,
        altitude_m=39.0,
        optimal_tilt=22.0,
        aliases=("vadodara", "baroda", "bdq"),
    ),
    CityInfo(
        name="Nagpur",
        state="Maharashtra",
        lat=21.1458,
        lon=79.0882,
        altitude_m=310.0,
        optimal_tilt=21.0,
        aliases=("nagpur", "nag"),
    ),
    CityInfo(
        name="Nashik",
        state="Maharashtra",
        lat=19.9975,
        lon=73.7898,
        altitude_m=600.0,
        optimal_tilt=20.0,
        aliases=("nashik", "nasik", "isk"),
    ),
    CityInfo(
        name="Rajkot",
        state="Gujarat",
        lat=22.3039,
        lon=70.8022,
        altitude_m=128.0,
        optimal_tilt=22.0,
        aliases=("rajkot", "raj"),
    ),
    CityInfo(
        name="Panaji",
        state="Goa",
        lat=15.4909,
        lon=73.8278,
        altitude_m=7.0,
        optimal_tilt=15.0,
        aliases=("panaji", "panjim", "goa", "goi"),
    ),
    CityInfo(
        name="Chhatrapati Sambhajinagar",
        state="Maharashtra",
        lat=19.8762,
        lon=75.3433,
        altitude_m=569.0,
        optimal_tilt=20.0,
        aliases=("chhatrapati sambhajinagar", "aurangabad", "ixu"),
    ),

    # ── East & Central India ──────────────────────────────────────────────────
    CityInfo(
        name="Kolkata",
        state="West Bengal",
        lat=22.5726,
        lon=88.3639,
        altitude_m=9.0,
        optimal_tilt=22.0,
        aliases=("kolkata", "calcutta", "ccu"),
    ),
    CityInfo(
        name="Bhopal",
        state="Madhya Pradesh",
        lat=23.2599,
        lon=77.4126,
        altitude_m=500.0,
        optimal_tilt=23.0,
        aliases=("bhopal", "bho"),
    ),
    CityInfo(
        name="Indore",
        state="Madhya Pradesh",
        lat=22.7196,
        lon=75.8577,
        altitude_m=553.0,
        optimal_tilt=23.0,
        aliases=("indore", "idr"),
    ),
    CityInfo(
        name="Patna",
        state="Bihar",
        lat=25.5941,
        lon=85.1376,
        altitude_m=53.0,
        optimal_tilt=25.0,
        aliases=("patna", "pat"),
    ),
    CityInfo(
        name="Bhubaneswar",
        state="Odisha",
        lat=20.2961,
        lon=85.8245,
        altitude_m=45.0,
        optimal_tilt=20.0,
        aliases=("bhubaneswar", "bbi"),
    ),
    CityInfo(
        name="Ranchi",
        state="Jharkhand",
        lat=23.3441,
        lon=85.3096,
        altitude_m=651.0,
        optimal_tilt=23.0,
        aliases=("ranchi", "ixr"),
    ),
    CityInfo(
        name="Guwahati",
        state="Assam",
        lat=26.1445,
        lon=91.7362,
        altitude_m=55.0,
        optimal_tilt=26.0,
        aliases=("guwahati", "gauhati", "gau"),
    ),
    CityInfo(
        name="Raipur",
        state="Chhattisgarh",
        lat=21.2514,
        lon=81.6296,
        altitude_m=298.0,
        optimal_tilt=21.0,
        aliases=("raipur", "rpr"),
    ),
    CityInfo(
        name="Gwalior",
        state="Madhya Pradesh",
        lat=26.2183,
        lon=78.1828,
        altitude_m=211.0,
        optimal_tilt=26.0,
        aliases=("gwalior", "gwl"),
    ),
    CityInfo(
        name="Jabalpur",
        state="Madhya Pradesh",
        lat=23.1815,
        lon=79.9864,
        altitude_m=411.0,
        optimal_tilt=23.0,
        aliases=("jabalpur", "jlr"),
    ),
]


# Lookup dictionary by standardized lowercase names and aliases
CITIES_BY_KEY: Dict[str, CityInfo] = {}
for city in _INDIAN_CITIES_RAW:
    # Standard name key
    CITIES_BY_KEY[city.name.lower()] = city
    # Strip spaces / punctuation
    clean_name = city.name.lower().replace(" ", "").replace("-", "")
    CITIES_BY_KEY[clean_name] = city
    # Register all aliases
    for alias in city.aliases:
        CITIES_BY_KEY[alias.lower()] = city
        CITIES_BY_KEY[alias.lower().replace(" ", "").replace("-", "")] = city


# Standard dictionary keyed by official City Name
CITIES_BY_NAME: Dict[str, CityInfo] = {c.name: c for c in _INDIAN_CITIES_RAW}


# ─────────────────────────────────────────────────────────────────────────────
# Public Lookup Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_city(name_or_alias: str) -> Optional[CityInfo]:
    """
    Look up a city by name, alias, or fuzzy search string.
    Returns CityInfo or None if not found.

    Examples:
        get_city("bengaluru")  -> CityInfo(name='Bengaluru', ...)
        get_city("Bangalore")  -> CityInfo(name='Bengaluru', ...)
        get_city("Mysore")     -> CityInfo(name='Mysuru', ...)
        get_city("Chennai, Tamil Nadu") -> CityInfo(name='Chennai', ...)
    """
    if not name_or_alias:
        return None

    query = name_or_alias.strip().lower()

    # If format is "City, State", split on comma first
    if "," in query:
        query_part = query.split(",")[0].strip()
        if query_part in CITIES_BY_KEY:
            return CITIES_BY_KEY[query_part]

    # Direct match
    if query in CITIES_BY_KEY:
        return CITIES_BY_KEY[query]

    # Clean query (no spaces or hyphens)
    clean_q = query.replace(" ", "").replace("-", "")
    if clean_q in CITIES_BY_KEY:
        return CITIES_BY_KEY[clean_q]

    # Substring search match
    for key, city in CITIES_BY_KEY.items():
        if query in key or key in query:
            return city

    return None


def get_city_coords(name_or_alias: str) -> Dict[str, float]:
    """
    Returns dict(lat=..., lon=..., altitude_m=...) for the given city.
    Raises ValueError if city is not found.
    """
    city = get_city(name_or_alias)
    if city is None:
        raise ValueError(
            f"City '{name_or_alias}' not found in registry. "
            f"Supported cities include: {', '.join(sorted(CITIES_BY_NAME.keys())[:10])}..."
        )
    return {
        "lat": city.lat,
        "lon": city.lon,
        "altitude_m": city.altitude_m,
    }


def list_cities() -> List[CityInfo]:
    """Return all supported cities sorted by name."""
    return sorted(_INDIAN_CITIES_RAW, key=lambda c: c.name)


def get_city_names() -> List[str]:
    """Return list of all official city names."""
    return [c.name for c in list_cities()]


def get_all_city_coords_dict() -> Dict[str, Dict[str, float]]:
    """Return a dictionary of {city_name: {lat, lon, altitude_m}} for all cities."""
    return {c.name: {"lat": c.lat, "lon": c.lon, "altitude_m": c.altitude_m} for c in _INDIAN_CITIES_RAW}
