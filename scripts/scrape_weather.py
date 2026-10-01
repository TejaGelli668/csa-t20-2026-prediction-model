"""Rain history and forecasts for every CSA T20 venue from Open-Meteo.

Rain has already wiped out several 2026 matches, so washout risk is modelled
per venue and date instead of with one league-wide rate.

Outputs (data/weather/):
  venues.csv     venue -> coordinates
  history.csv    daily precipitation per venue, Sep 2011 -> yesterday
  forecast.csv   16-day daily forecast per venue (precip sum, hours, probability)
"""
import csv, json, os, sys, time
from datetime import date, timedelta

import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "weather")
AS_OF = date(2026, 9, 30)

# key -> (display name, city, lat, lon). Keys are matched as substrings of
# venue names from Cricbuzz, Cricsheet and ESPN.
VENUES = {
    "supersport": ("SuperSport Park", "Centurion", -25.8601, 28.1795),
    "wanderers": ("The Wanderers Stadium", "Johannesburg", -26.1317, 28.0573),
    "kingsmead": ("Kingsmead", "Durban", -29.8510, 31.0293),
    "newlands": ("Newlands", "Cape Town", -33.9746, 18.4689),
    "mangaung": ("Mangaung Oval", "Bloemfontein", -29.1170, 26.2085),
    "stgeorge": ("St George's Park", "Gqeberha", -33.9640, 25.6118),
    "bolandpark": ("Boland Park", "Paarl", -33.7365, 18.9662),
    "senwes": ("Senwes Park", "Potchefstroom", -26.7134, 27.0832),
    "buffalo": ("Buffalo Park", "East London", -33.0016, 27.9007),
    "cityoval": ("City Oval", "Pietermaritzburg", -29.6006, 30.3794),
    "willowmoore": ("Willowmoore Park", "Benoni", -26.1858, 28.3210),
    "diamondoval": ("Diamond Oval", "Kimberley", -28.7406, 24.7601),
    "recreationground": ("Recreation Ground", "Oudtshoorn", -33.5930, 22.2010),
    "polokwane": ("Polokwane Cricket Club", "Polokwane", -23.9045, 29.4689),
    "uplands": ("Uplands College", "White River", -25.3318, 31.0108),
    "devilliers": ("LC de Villiers Oval", "Pretoria", -25.7522, 28.2470),
    "nwcoval": ("NWC Oval", "Potchefstroom", -26.7134, 27.0832),
}
ALIASES = {  # older names of the same grounds
    "kimberleyoval": "diamondoval", "chevroletpark": "mangaung", "outsuranceoval": "mangaung", "goodyearpark": "mangaung",
    "bloemfontein": "mangaung", "portelizabeth": "stgeorge", "centurion": "supersport", "paarl": "bolandpark",
    "potchefstroom": "senwes", "benoni": "willowmoore", "eastlondon": "buffalo", "pietermaritzburg": "cityoval",
    "oudtshoorn": "recreationground", "whiteriver": "uplands", "tuks": "devilliers", "durban": "kingsmead",
    "capetown": "newlands", "johannesburg": "wanderers", "kimberley": "diamondoval",
}


def venue_key(name):
    s = "".join(ch for ch in str(name).lower() if ch.isalpha())
    for k in VENUES:
        if k in s:
            return k
    for a, k in ALIASES.items():
        if a in s:
            return k
    return None


def get_cached(url, params, cache, refresh=False):
    """GET with an on-disk cache and back-off on Open-Meteo's rate limit."""
    path = os.path.join(OUT, "raw", cache)
    if os.path.exists(path) and not refresh:
        return json.load(open(path))
    for attempt in range(6):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code in (429, 502, 503):
            time.sleep(20 * (attempt + 1))
            continue
        r.raise_for_status()
        json.dump(r.json(), open(path, "w"))
        time.sleep(2)
        return r.json()
    return None


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote data/weather/{name} ({len(rows)} rows)")


def main():
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    write("venues.csv", [dict(key=k, venue=v[0], city=v[1], lat=v[2], lon=v[3]) for k, v in VENUES.items()])
    hist, fc = [], []
    for k, (vn, city, lat, lon) in VENUES.items():
        h = get_cached("https://archive-api.open-meteo.com/v1/archive", dict(
            latitude=lat, longitude=lon, start_date="2011-09-01", end_date=str(AS_OF - timedelta(days=1)),
            daily="precipitation_sum,precipitation_hours", timezone="Africa/Johannesburg"), f"history_{k}.json")
        d = h["daily"]
        n_hist = len(d["time"])
        hist += [dict(key=k, date=t, precip_mm=p, precip_hours=hr)
                 for t, p, hr in zip(d["time"], d["precipitation_sum"], d["precipitation_hours"])]
        f = get_cached("https://api.open-meteo.com/v1/forecast", dict(
            latitude=lat, longitude=lon, forecast_days=16, timezone="Africa/Johannesburg",
            daily="precipitation_sum,precipitation_hours,precipitation_probability_max"),
            f"forecast_{k}_{AS_OF}.json", refresh="--refresh" in sys.argv)
        if f:
            d = f["daily"]
            fc += [dict(key=k, date=t, precip_mm=p, precip_hours=hr, precip_prob=pp) for t, p, hr, pp in
                   zip(d["time"], d["precipitation_sum"], d["precipitation_hours"], d["precipitation_probability_max"])]
        print(f"  {vn:28s} history {n_hist} days, forecast {'ok' if f else 'unavailable'}")
    write("history.csv", hist)
    if fc:
        write("forecast.csv", fc)


if __name__ == "__main__":
    main()
