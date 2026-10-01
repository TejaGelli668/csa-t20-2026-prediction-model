# CSA T20 Challenge 2026 — Winner Prediction Model

Statistical / ML model that predicts the winner of the 2026 CSA T20 Challenge
(16 teams, two pools of 8 → Super Eights → semi-finals → final on 14 Nov 2026),
with a React dashboard that explains the numbers.

## Layout

```
data/
  matches.csv, batting.csv, bowling.csv, points_table_2026.csv   Cricbuzz
  cricsheet/      Cricsheet ball-by-ball (2011-12 → 2024-25) + player register
  espn/           ESPN / ESPNcricinfo API: tosses, XIs, points, standings, player profiles
  squads/         CricTracker official 2026 squads
  weather/        Open-Meteo rain history + 16-day forecast for 17 venues
  unified/        merged dataset + validation.json (built by build_dataset.py)
  player_availability.json   curated national-duty / injury windows
scripts/
  scrape_cricbuzz.py  scrape_cricsheet.py  scrape_espn.py  scrape_squads.py  scrape_weather.py
  build_dataset.py    merge sources, link player ids, rebuild standings, cross-check
  model.py            features, ML models, back-test, washout model, simulation
  refresh_all.py      run everything in order
web/                  React (Vite) dashboard; reads web/src/data/model_output.json
```

## Data sources (Step 2)

| Source | What it adds |
|---|---|
| Cricbuzz | Fixtures, results and scores 2018-19 → 2026, official points table, scorecards from 2022-23 |
| Cricsheet | Ball-by-ball for 314 CSA T20 Challenge matches 2011-12 → 2024-25 (lineups, tosses, phase splits) |
| ESPN / ESPNcricinfo API | 2025-26 & 2026 tosses, playing XIs with Cricinfo ids, points awarded, official standings, 452 player profiles |
| CricTracker | Official 2026 squads with roles |
| Open-Meteo | Daily rain at every venue since 2011 + 16-day forecast |

espncricinfo.com blocks scripts (HTTP 403) and Sofascore does too; ESPN's public
API serves the same Cricinfo data. `data/unified/validation.json` records how
the sources agree. Winners agree in every overlapping match. Standings rebuilt
from results match ESPNcricinfo's table, while Cricbuzz's table differs for two
teams.

## Models (Step 3)

* 542 matches (2011-12 → 2026) and ~10k batting lines after merging sources.
* Features (pre-match, leakage-free): Elo, home ground, batting and bowling
  strength of the XI (from decayed, shrunk player ratings), plus form, run-rate
  margin, head-to-head and venue record, which are kept as context only
  because they don't reliably improve out-of-sample accuracy.
* Models: Elo, Logistic Regression, Random Forest, Gradient Boosting, blended
  by back-test skill.
* Back-test: train on earlier seasons and predict each season from 2015-16 to
  2026 (330 matches). The ensemble scores log loss 0.679 vs 0.693 for a coin
  flip, with 58.5% accuracy and AUC 0.60.
* Washout model: logistic regression of abandonment on venue rainfall, applied
  to the forecast (up to 16 days ahead) or to venue climatology.
* Monte Carlo: 20,000 simulations of the rest of the tournament, with
  washouts, bonus points, NRR, date-specific player availability and
  team-strength uncertainty.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/refresh_all.py                 # scrape all sources, merge, model
cd web && npm install && npm run dev                    # http://localhost:5173
```
