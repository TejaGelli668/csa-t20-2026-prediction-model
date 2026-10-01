"""Cricsheet ball-by-ball data for the CSA T20 Challenge (2011-12 onwards).

Cricsheet publishes open, ball-by-ball JSON for every CSA T20 Challenge match
it has (code CTC). From it we derive per-match results and tosses, per-player
batting/bowling lines, and per-innings phase splits (powerplay, middle, death).

Outputs (data/cricsheet/):
  cs_matches.csv   one row per match
  cs_batting.csv   one row per player in the XI (did-not-bat rows included)
  cs_bowling.csv   one row per bowler per innings
  cs_phases.csv    runs/wickets per innings in overs 1-6, 7-15, 16-20
  people.csv       Cricsheet player register (ids -> ESPNcricinfo ids)
"""
import csv, glob, io, json, os, sys, zipfile

import requests

sys.path.insert(0, os.path.dirname(__file__))
from scrape_cricbuzz import canon  # shared team-name normalisation

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "cricsheet")
UA = {"User-Agent": "Mozilla/5.0 (research project; CSA T20 model)"}
NOT_BOWLER_WICKETS = {"run out", "retired hurt", "retired out", "obstructing the field", "retired not out"}


def download():
    os.makedirs(OUT, exist_ok=True)
    zpath = os.path.join(OUT, "ctc_json.zip")
    if "--refresh" in sys.argv or not os.path.exists(zpath):
        r = requests.get("https://cricsheet.org/downloads/ctc_json.zip", headers=UA, timeout=60)
        r.raise_for_status()
        open(zpath, "wb").write(r.content)
    zipfile.ZipFile(zpath).extractall(os.path.join(OUT, "ctc"))
    ppath = os.path.join(OUT, "people.csv")
    if "--refresh" in sys.argv or not os.path.exists(ppath):
        r = requests.get("https://cricsheet.org/register/people.csv", headers=UA, timeout=60)
        r.raise_for_status()
        open(ppath, "w", encoding="utf-8").write(r.text)


def season_label(s):
    s = str(s)
    return s.replace("/", "-") if "/" in s else s


def phase(over):
    return "powerplay" if over < 6 else ("middle" if over < 15 else "death")


def parse(path):
    d = json.load(open(path))
    info = d["info"]
    mid = os.path.basename(path).split(".")[0]
    reg = info["registry"]["people"]
    teams = info["teams"]
    t1, t2 = canon(teams[0]), canon(teams[1])
    out = info.get("outcome", {})
    winner = out.get("winner") or out.get("eliminator") or ""
    result = out.get("result", "")
    by = out.get("by", {})
    match = dict(
        cs_id=mid, date=info["dates"][0], season=season_label(info.get("season")),
        event=info.get("event", {}).get("name", ""),
        stage=info.get("event", {}).get("stage") or (f"Match {info['event']['match_number']}" if info.get("event", {}).get("match_number") else ""),
        venue=info.get("venue", ""), city=info.get("city", ""), team1=t1, team2=t2,
        team1_name=teams[0], team2_name=teams[1],
        toss_winner=canon(info.get("toss", {}).get("winner", "")), toss_decision=info.get("toss", {}).get("decision", ""),
        winner=canon(winner) if winner else "", result=result,
        no_result=int(result == "no result"), tied=int(result == "tie"),
        win_by_runs=by.get("runs"), win_by_wickets=by.get("wickets"), method=out.get("method", ""),
        player_of_match="; ".join(info.get("player_of_match", [])),
    )
    batting, bowling, phases = [], [], []
    xi = {canon(t): [(p, reg.get(p)) for p in pl] for t, pl in info.get("players", {}).items()}
    for n, inn in enumerate(d.get("innings", [])[:2], start=1):  # super overs excluded
        bt = canon(inn["team"])
        ft = t2 if bt == t1 else t1
        order, bat = [], {}
        bowl = {}
        ph = {k: [0, 0, 0] for k in ("powerplay", "middle", "death")}  # runs, wkts, balls
        tot_runs = tot_wk = legal = 0
        for ov in inn.get("overs", []):
            o = ov["over"]
            for dl in ov["deliveries"]:
                ex = dl.get("extras", {})
                wide, nb = "wides" in ex, "noballs" in ex
                b = dl["batter"]
                for who in (b, dl["non_striker"]):
                    if who not in bat:
                        bat[who] = dict(runs=0, balls=0, fours=0, sixes=0, out="not out")
                        order.append(who)
                br = dl["runs"]["batter"]
                bat[b]["runs"] += br
                if not wide:
                    bat[b]["balls"] += 1
                if br == 4 and not dl["runs"].get("non_boundary"):
                    bat[b]["fours"] += 1
                if br == 6:
                    bat[b]["sixes"] += 1
                bw = bowl.setdefault(dl["bowler"], dict(balls=0, runs=0, wickets=0))
                if not (wide or nb):
                    bw["balls"] += 1
                    legal += 1
                    ph[phase(o)][2] += 1
                bw["runs"] += br + ex.get("wides", 0) + ex.get("noballs", 0)
                ph[phase(o)][0] += dl["runs"]["total"]
                tot_runs += dl["runs"]["total"]
                for w in dl.get("wickets", []):
                    po = w["player_out"]
                    if po in bat:
                        bat[po]["out"] = w["kind"]
                    if w["kind"] not in NOT_BOWLER_WICKETS:
                        bw["wickets"] += 1
                    if w["kind"] not in ("retired hurt", "retired not out"):
                        ph[phase(o)][1] += 1
                        tot_wk += 1
        match[f"inn{n}_team"] = bt
        match[f"inn{n}_runs"] = tot_runs
        match[f"inn{n}_wkts"] = tot_wk
        match[f"inn{n}_balls"] = legal
        names_in_xi = [p for p, _ in xi.get(bt, [])] or order
        for pos, p in enumerate(order + [p for p in names_in_xi if p not in order], start=1):
            s = bat.get(p, dict(runs=0, balls=0, fours=0, sixes=0, out=""))
            batting.append(dict(cs_id=mid, date=match["date"], season=match["season"], innings=n, team=bt, opponent=ft,
                                venue=match["venue"], cs_player=reg.get(p), player=p, position=pos,
                                runs=s["runs"], balls=s["balls"], fours=s["fours"], sixes=s["sixes"],
                                out=s["out"] if p in bat else ""))
        for p, s in bowl.items():
            bowling.append(dict(cs_id=mid, date=match["date"], season=match["season"], innings=n, team=ft, opponent=bt,
                                venue=match["venue"], cs_player=reg.get(p), player=p, balls=s["balls"],
                                overs=f"{s['balls'] // 6}.{s['balls'] % 6}", runs=s["runs"], wickets=s["wickets"]))
        for k, (r, w, b) in ph.items():
            phases.append(dict(cs_id=mid, date=match["date"], season=match["season"], innings=n, team=bt, opponent=ft,
                               phase=k, runs=r, wickets=w, balls=b))
    return match, batting, bowling, phases


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote data/cricsheet/{name} ({len(rows)} rows)")


def main():
    download()
    M, B, W, P = [], [], [], []
    for f in sorted(glob.glob(os.path.join(OUT, "ctc", "*.json"))):
        m, b, w, p = parse(f)
        M.append(m)
        B += b
        W += w
        P += p
    keys = sorted({k for m in M for k in m}, key=lambda k: list(M[0]).index(k) if k in M[0] else 99)
    M = [{k: m.get(k, "") for k in keys} for m in sorted(M, key=lambda m: m["date"])]
    write("cs_matches.csv", M)
    write("cs_batting.csv", B)
    write("cs_bowling.csv", W)
    write("cs_phases.csv", P)


if __name__ == "__main__":
    main()
