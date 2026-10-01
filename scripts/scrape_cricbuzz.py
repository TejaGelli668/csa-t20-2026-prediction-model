"""Scrape CSA T20 Challenge data (historic + 2026) from Cricbuzz.

ESPNcricinfo blocks scripted requests (HTTP 403), so Cricbuzz is the source.
Every page is cached under data/raw/ so re-runs are cheap; pass --refresh to
re-download the current season.

Outputs (data/):
  matches.csv            every match of every season (results + fixtures)
  points_table_2026.csv  official current standings
  batting.csv, bowling.csv  per-player scorecard rows for completed matches
  squads_2026.json       squad list (teams, ids) for the current season
"""
import csv, json, os, re, sys
from datetime import datetime, timezone
from cbparse import fetch, flight, objects_for_key

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
REFRESH = "--refresh" in sys.argv

SERIES = [
    # (cricbuzz id, slug, season label, competition)
    (2753, "csa-t20-challenge-2019", "2018-19", "T20 Challenge"),
    (3315, "csa-t20-challenge-2021", "2020-21", "T20 Challenge"),
    (4006, "csa-t20-challenge-2022", "2021-22", "T20 Challenge"),
    (4744, "csa-t20-challenge-2022-23", "2022-23", "T20 Challenge"),
    (7663, "csa-t20-challenge-2024", "2023-24", "T20 Challenge"),
    (8643, "csa-t20-challenge-2024", "2024-25", "T20 Challenge"),
    (8746, "csa-t20-knock-out-competition-2024", "2024-25", "T20 Knock-Out"),
    (11032, "csa-t20-challenge-2025", "2025-26", "T20 Challenge"),
    (11110, "csa-t20-knock-out-competition-2025", "2025-26", "T20 Knock-Out"),
    (13180, "csa-t20-challenge-2026", "2026", "T20 Challenge"),
]
CURRENT = 13180

# Canonical team codes. Franchises have been renamed by sponsors many times,
# so match on keywords in the full name.
TEAM_KEYS = [
    ("cobras", "WPR"), ("western province", "WPR"),
    ("titans", "TIT"), ("lions", "LIONS"), ("dolphins", "DOL"),
    ("knights", "KNG"), ("warriors", "WAR"), ("boland", "BOL"), ("rocks", "BOL"),
    ("eastern cape", "BOR"), ("iinyathi", "BOR"), ("garden route", "SWD"), ("badgers", "SWD"), ("ecape", "BOR"), ("tusks", "KZNIN"),
    ("north west", "NWEST"), ("northwest", "NWEST"), ("dragons", "NWEST"),
    ("border", "BOR"), ("kwazulu", "KZNIN"), ("tuskers", "KZNIN"),
    ("eastern storm", "ESTORM"), ("easterns", "ESTORM"),
    ("northern cape", "NCAPE"), ("heat", "NCAPE"),
    ("south western", "SWD"), ("swd", "SWD"),
    ("limpopo", "LIMPO"), ("impala", "LIMPO"),
    ("mpumalanga", "MPR"), ("rhinos", "MPR"),
    ("emerging", "SAEP"), ("impi", "IMPI"), ("free state", "FS"), ("gauteng", "GAU"),
    ("eastern province", "EP"), ("north west", "NWEST"),
]


def canon(name):
    n = (name or "").lower()
    for k, code in TEAM_KEYS:
        if k in n:
            return code
    return name or "TBC"


def iso(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def parse_winner(status, t1, t2):
    s = status or ""
    m = re.search(r"\((.+?) won the super over\)", s, re.I)
    if m:
        return canon(m.group(1))
    m = re.match(r"(.+?) won by", s)
    if m:
        return canon(m.group(1))
    m = re.match(r"(.+?) won", s)  # e.g. "X won (Super Over)"
    if m:
        return canon(m.group(1))
    return ""


def score_winner(t1, r1, t2, r2):
    """Older archive rows sometimes lack a status string; fall back to runs."""
    if r1 is None or r2 is None or r1 == r2:
        return ""
    return t1 if r1 > r2 else t2


def bat_first(ms, t1, t2):
    for key, team in (("team1Score", t1), ("team2Score", t2)):
        if ((ms.get(key) or {}).get("inngs1") or {}).get("inningsId") == 1:
            return team
    return ""


def innings(score):
    if not score:
        return None, None, None
    i = score.get("inngs1") or {}
    return i.get("runs"), i.get("wickets"), i.get("overs")


def scrape_series(sid, slug, season, comp):
    h = fetch(f"https://www.cricbuzz.com/cricket-series/{sid}/{slug}/matches",
              f"cb_{sid}_matches.html", refresh=REFRESH and sid == CURRENT)
    f = flight(h)
    seen = {}
    for v in objects_for_key(f, "matchDetailsMap"):
        if not isinstance(v, dict):
            continue
        for m in v.get("match", []):
            mi = m.get("matchInfo", {})
            if mi.get("seriesId") == sid:
                seen[mi["matchId"]] = m
    rows = []
    for m in seen.values():
        mi, ms = m["matchInfo"], m.get("matchScore", {})
        t1, t2 = canon(mi["team1"]["teamName"]), canon(mi["team2"]["teamName"])
        r1, w1, o1 = innings(ms.get("team1Score"))
        r2, w2, o2 = innings(ms.get("team2Score"))
        status = mi.get("status", "")
        rows.append(dict(
            season=season, competition=comp, series_id=sid, match_id=mi["matchId"],
            date=iso(mi["startDate"]), stage=mi.get("matchDesc", ""), state=mi.get("state", ""),
            team1=t1, team2=t2, team1_name=mi["team1"]["teamName"], team2_name=mi["team2"]["teamName"],
            venue=(mi.get("venueInfo") or {}).get("ground", ""), city=(mi.get("venueInfo") or {}).get("city", ""),
            t1_runs=r1, t1_wkts=w1, t1_overs=o1, t2_runs=r2, t2_wkts=w2, t2_overs=o2,
            status=status, winner=parse_winner(status, t1, t2) or score_winner(t1, r1, t2, r2),
            bat_first=bat_first(ms, t1, t2),
            no_result=int(bool(re.search(r"abandon|no result", status, re.I))),
            tied=int("tie" in status.lower()),
            slug=f"{mi['team1']['teamSName'].lower()}-vs-{mi['team2']['teamSName'].lower()}-"
                 f"{mi.get('matchDesc','').lower().replace(' ','-')}-{slug}",
        ))
    rows.sort(key=lambda r: r["date"])
    print(f"  {season} {comp} ({sid}): {len(rows)} matches, "
          f"{sum(1 for r in rows if r['winner'])} with result")
    return rows


def scrape_scorecard(match):
    mid = match["match_id"]
    url = f"https://www.cricbuzz.com/live-cricket-scorecard/{mid}/{re.sub('[^a-z0-9-]', '', match['slug'])}"
    h = fetch(url, f"cb_sc_{mid}.html", refresh=REFRESH and match["series_id"] == CURRENT)
    f = flight(h)
    sc = next((v for v in objects_for_key(f, "scoreCard") if isinstance(v, list) and v), [])
    bat, bowl = [], []
    for inn in sc:
        bt, bw = inn.get("batTeamDetails", {}), inn.get("bowlTeamDetails", {})
        bteam, wteam = canon(bt.get("batTeamName")), canon(bw.get("bowlTeamName"))
        for i, p in enumerate((bt.get("batsmenData") or {}).values()):
            bat.append(dict(match_id=mid, season=match["season"], date=match["date"], innings=inn.get("inningsId"),
                            team=bteam, opponent=wteam, venue=match["venue"], player_id=p.get("batId"),
                            player=p.get("batName"), position=i + 1, runs=p.get("runs"), balls=p.get("balls"),
                            fours=p.get("fours"), sixes=p.get("sixes"), out=p.get("outDesc", ""),
                            is_captain=p.get("isCaptain"), is_keeper=p.get("isKeeper")))
        for p in (bw.get("bowlersData") or {}).values():
            bowl.append(dict(match_id=mid, season=match["season"], date=match["date"], innings=inn.get("inningsId"),
                             team=wteam, opponent=bteam, venue=match["venue"], player_id=p.get("bowlerId"),
                             player=p.get("bowlName"), overs=p.get("overs"), maidens=p.get("maidens"),
                             runs=p.get("runs"), wickets=p.get("wickets"), economy=p.get("economy")))
    return bat, bowl


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")


def main():
    os.makedirs(os.path.join(DATA, "raw"), exist_ok=True)
    print("Match lists")
    matches = []
    for s in SERIES:
        matches += scrape_series(*s)
    write_csv(os.path.join(DATA, "matches.csv"), matches)

    # Current standings
    h = fetch("https://www.cricbuzz.com/cricket-series/13180/csa-t20-challenge-2026/points-table",
              "cb_2026_points-table.html", refresh=REFRESH)
    pt = next(objects_for_key(flight(h), "pointsTable"))
    rows = []
    for g in pt:
        for t in g["pointsTableInfo"]:
            rows.append(dict(pool=g["groupName"], team=canon(t["teamFullName"]), team_name=t["teamFullName"],
                             played=t["matchesPlayed"], won=t["matchesWon"], lost=t["matchesLost"],
                             tied=t["matchesTied"], no_result=t["noRes"], nrr=float(t["nrr"]),
                             points=t["points"], form="".join(t.get("form", []))))
    write_csv(os.path.join(DATA, "points_table_2026.csv"), rows)

    h = fetch("https://www.cricbuzz.com/cricket-series/13180/csa-t20-challenge-2026/squads", "cb_2026_squads.html")
    sq = next(objects_for_key(flight(h), "squads"))
    squads = [dict(team=canon(s["squadType"]), name=s["squadType"], squad_id=s["squadId"], team_id=s["teamId"])
              for s in sq if "squadId" in s]
    json.dump(squads, open(os.path.join(DATA, "squads_2026.json"), "w"), indent=1)

    print("Scorecards (2022-23 onwards)")
    bat, bowl = [], []
    recent = [m for m in matches if m["season"] >= "2022-23" and m["winner"] and not m["no_result"]]
    for i, m in enumerate(recent):
        try:
            b, w = scrape_scorecard(m)
            bat += b
            bowl += w
        except Exception as e:
            print("  scorecard failed", m["match_id"], e)
        if i % 25 == 0:
            print(f"  {i}/{len(recent)}")
    write_csv(os.path.join(DATA, "batting.csv"), bat)
    write_csv(os.path.join(DATA, "bowling.csv"), bowl)


if __name__ == "__main__":
    main()
