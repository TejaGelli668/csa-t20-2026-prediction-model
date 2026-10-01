"""Official 2026 squads from CricTracker (its public GraphQL gateway).

Output: data/squads/crictracker_2026.csv  (team, player, role)
"""
import csv, json, os, re, sys

import requests

sys.path.insert(0, os.path.dirname(__file__))
from scrape_cricbuzz import canon

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "squads")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
PAGE = "https://www.crictracker.com/t20/csa-t20-challenge/squads/"
GQL = "https://gateway.crictracker.com/graphql"
QUERY = "query($input: listSeriesSquadInput){ listSeriesSquad(input:$input){ oPlayer { sFullName sPlayingRole sCountry } } }"
ROLES = {"bat": "Batter", "bowl": "Bowler", "all": "All-rounder", "wk": "Wicketkeeper", "wkbat": "Wicketkeeper"}


def main():
    os.makedirs(OUT, exist_ok=True)
    html = requests.get(PAGE, headers={"User-Agent": UA}, timeout=30).text
    open(os.path.join(OUT, "crictracker_squads_page.html"), "w").write(html)
    nd = json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S).group(1))
    st = nd["props"]["pageProps"]["category"]["seriesTeamData"]["listSeriesTeams"]
    rows = []
    for t in st["aTeams"]:
        r = requests.post(GQL, json={"query": QUERY, "variables": {"input": {"iSeriesId": st["iSeriesId"], "iTeamId": t["_id"]}}},
                          headers={"User-Agent": UA, "Origin": "https://www.crictracker.com", "Referer": PAGE}, timeout=30)
        r.raise_for_status()
        squad = r.json()["data"]["listSeriesSquad"] or []
        for p in squad:
            pl = p["oPlayer"]
            rows.append(dict(team=canon(t["sTitle"]), team_name=t["sTitle"], player=pl["sFullName"],
                             role=ROLES.get(pl.get("sPlayingRole"), pl.get("sPlayingRole") or "")))
        print(f"  {t['sTitle']:32s} {len(squad)} players")
    with open(os.path.join(OUT, "crictracker_2026.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote data/squads/crictracker_2026.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
