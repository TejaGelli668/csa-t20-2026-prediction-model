"""ESPN / ESPNcricinfo data via ESPN's public site API.

espncricinfo.com itself returns HTTP 403 to scripts, but ESPN's site API
serves the same Cricinfo database (match ids, player ids, scorecards).
League 8656 = CSA domestic T20; seasons 2025 (2025-26) and 2026 are exposed.

Outputs (data/espn/):
  espn_matches.csv   fixtures/results with toss and points awarded
  espn_xi.csv        playing XIs with ESPNcricinfo player ids
  athletes.csv       player profiles (full name, DOB, role, batting/bowling style)
                     for every player seen in ESPN or Cricsheet data
"""
import csv, json, os, re, sys, time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from scrape_cricbuzz import canon

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "espn")
RAW = os.path.join(OUT, "raw")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}
SITE = "https://site.web.api.espn.com/apis/site/v2/sports/cricket/8656"
REFRESH = "--refresh" in sys.argv


def get_json(url, cache, refresh=False):
    path = os.path.join(RAW, cache)
    if os.path.exists(path) and not refresh:
        return json.load(open(path))
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            json.dump(data, open(path, "w"))
            time.sleep(0.25)
            return data
        except Exception:
            time.sleep(1 + attempt)
    return None


def write(name, rows):
    if not rows:
        return
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote data/espn/{name} ({len(rows)} rows)")


def main():
    os.makedirs(RAW, exist_ok=True)
    matches, xi = [], []
    for season, label in ((2025, "2025-26"), (2026, "2026")):
        sb = get_json(f"{SITE}/scoreboard?season={season}", f"scoreboard_{season}.json", refresh=REFRESH and season == 2026)
        for ev in sb.get("events", []):
            comp = ev["competitions"][0]
            cs = sorted(comp["competitors"], key=lambda c: c.get("order", 0))
            if len(cs) < 2:
                continue
            state = ev.get("status", {}).get("type", {}).get("state") or comp.get("status", {}).get("type", {}).get("state", "")
            row = dict(espn_id=ev["id"], season=label, date=ev["date"][:16].replace("T", " "), stage=comp.get("description", ""),
                       team1=canon(cs[0]["team"]["displayName"]), team2=canon(cs[1]["team"]["displayName"]),
                       venue=comp.get("venue", {}).get("fullName", ""), state=state,
                       score1=cs[0].get("score", ""), score2=cs[1].get("score", ""),
                       winner=next((canon(c["team"]["displayName"]) for c in cs if str(c.get("winner")).lower() == "true"), ""),
                       toss_winner="", toss_decision="", points1="", points2="", result_note="")
            if state == "post":
                sm = get_json(f"{SITE}/summary?event={ev['id']}", f"summary_{ev['id']}.json")
                if sm:
                    for n in sm.get("notes", []):
                        if n.get("type") == "toss":
                            m = re.match(r"(.+?)\s*,\s*elected to (\w+)", n["text"])
                            if m:
                                row["toss_winner"], row["toss_decision"] = canon(m.group(1)), ("bat" if m.group(2) == "bat" else "field")
                        if n.get("type") == "points":
                            for part in n["text"].split(","):
                                m = re.match(r"\s*(.+?)\s+(\d+)\s*$", part)
                                if m:
                                    t = canon(m.group(1))
                                    if t == row["team1"]:
                                        row["points1"] = int(m.group(2))
                                    elif t == row["team2"]:
                                        row["points2"] = int(m.group(2))
                    hdr = (sm.get("header", {}).get("competitions") or [{}])[0]
                    row["result_note"] = hdr.get("status", {}).get("summary", "") or hdr.get("status", {}).get("type", {}).get("description", "")
                    for r in sm.get("rosters", []):
                        team = canon(r["team"]["displayName"])
                        for p in r.get("roster", []):
                            a = p["athlete"]
                            xi.append(dict(espn_id=ev["id"], season=label, date=row["date"], team=team,
                                           cricinfo_id=a["id"], player=a.get("displayName"), captain=p.get("captain"),
                                           starter=p.get("starter"),
                                           styles="; ".join(s.get("description", "") for s in a.get("style", []))))
            matches.append(row)
    write("espn_matches.csv", matches)

    st = get_json("https://site.web.api.espn.com/apis/v2/sports/cricket/8656/standings?season=2026",
                  "standings_2026.json", refresh=REFRESH)
    srows = []
    for g in (st or {}).get("children", []):
        for e in g["standings"]["entries"]:
            v = {x.get("name"): x.get("displayValue") for x in e.get("stats", [])}
            srows.append(dict(pool=g["name"], team=canon(e["team"]["displayName"]), played=v.get("matchesPlayed"),
                              won=v.get("matchesWon"), lost=v.get("matchesLost"), no_result=v.get("noresult"),
                              points=v.get("matchPoints"), nrr=v.get("netrr")))
    write("espn_standings_2026.csv", srows)
    write("espn_xi.csv", xi)

    # Player profiles for everyone in ESPN XIs and in Cricsheet's CSA data.
    ids = {str(r["cricinfo_id"]) for r in xi}
    cs_bat = os.path.join(ROOT, "data", "cricsheet", "cs_batting.csv")
    people = os.path.join(ROOT, "data", "cricsheet", "people.csv")
    if os.path.exists(cs_bat) and os.path.exists(people):
        reg = pd.read_csv(people, dtype=str).set_index("identifier")["key_cricinfo"]
        used = pd.read_csv(cs_bat, dtype=str)["cs_player"].dropna().unique()
        ids |= {str(reg[c]) for c in used if c in reg.index and pd.notna(reg[c])}
    print(f"fetching {len(ids)} athlete profiles")
    ath = []
    for i, pid in enumerate(sorted(ids)):
        a = get_json(f"http://core.espnuk.org/v2/sports/cricket/athletes/{pid}", f"athlete_{pid}.json")
        if not a:
            continue
        styles = [s.get("description", "") for s in a.get("style", [])]
        pos = a.get("position", {})
        ath.append(dict(cricinfo_id=pid, name=a.get("displayName") or a.get("name"), full_name=a.get("fullName"),
                        batting_name=a.get("battingName"), dob=(a.get("dateOfBirth") or "")[:10],
                        role=pos.get("name") if isinstance(pos, dict) else "",
                        bat_style=next((s for s in styles if "bat" in s.lower()), ""),
                        bowl_style=next((s for s in styles if "bat" not in s.lower()), "")))
        if i % 100 == 0:
            print(f"  {i}/{len(ids)}")
    write("athletes.csv", ath)


if __name__ == "__main__":
    main()
