"""Merge Cricbuzz, Cricsheet, ESPN/ESPNcricinfo, CricTracker and Open-Meteo
into one consistent dataset, and cross-check the sources against each other.

Outputs (data/unified/):
  matches.csv        one row per match, 2011-12 -> 2026, with the source(s) it came from
  batting.csv        player batting lines with a single player key
  bowling.csv        player bowling lines with a single player key
  players.csv        player crosswalk: Cricbuzz id <-> ESPNcricinfo id <-> Cricsheet id
  squads_2026.csv    official squads mapped onto player keys
  standings_2026.csv points table rebuilt from all sources' results
  validation.json    agreement statistics between sources
"""
import json, os, re, sys, unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scrape_weather import venue_key

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "unified")
os.makedirs(OUT, exist_ok=True)


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def overs_str(balls):
    balls = int(balls)
    return f"{balls // 6}.{balls % 6}"


def parse_espn_score(s):
    """'131/3 (15.3/20 ov, target 129)' -> (131, 3, '15.3'); '135' -> (135, 10, None)."""
    if not isinstance(s, str) or not s.strip():
        return None, None, None
    m = re.match(r"\s*(\d+)(?:/(\d+))?", s)
    if not m:
        return None, None, None
    ov = re.search(r"\(([\d.]+)(?:/\d+)? ov", s)
    return int(m.group(1)), int(m.group(2)) if m.group(2) else 10, ov.group(1) if ov else None


# ------------------------------------------------------------------ load
cb = pd.read_csv(os.path.join(D, "matches.csv"))
cb["day"] = pd.to_datetime(cb.date).dt.normalize()
cs = pd.read_csv(os.path.join(D, "cricsheet", "cs_matches.csv"), dtype={"cs_id": str})
cs["day"] = pd.to_datetime(cs.date).dt.normalize()
es = pd.read_csv(os.path.join(D, "espn", "espn_matches.csv"))
es["day"] = pd.to_datetime(es.date).dt.normalize()


def find(df, day, t1, t2, tol=1):
    m = df[(abs((df.day - day).dt.days) <= tol) &
           (((df.team1 == t1) & (df.team2 == t2)) | ((df.team1 == t2) & (df.team2 == t1)))]
    return m.iloc[0] if len(m) else None


# ---------------------------------------------------------- unified matches
rows = []
used_cs, used_es = set(), set()
validation = defaultdict(Counter)
disagreements = []
for _, m in cb.iterrows():
    r = m.to_dict()
    r.update(cb_id=m.match_id, cs_id=None, espn_id=None, toss_winner=None, toss_decision=None,
             points1=None, points2=None, result_source="cricbuzz" if m.winner == m.winner or m.no_result else None)
    src = ["cricbuzz"]
    if m.team1 != "TBC":
        c = find(cs, m.day, m.team1, m.team2)
        if c is not None:
            used_cs.add(c.cs_id)
            src.append("cricsheet")
            r.update(cs_id=c.cs_id, toss_winner=c.toss_winner, toss_decision=c.toss_decision)
            if isinstance(m.winner, str) and isinstance(c.winner, str):
                validation["cricbuzz_vs_cricsheet"]["winner_agree" if m.winner == c.winner else "winner_disagree"] += 1
                if m.winner != c.winner:
                    disagreements.append(dict(pair="cricbuzz/cricsheet", date=str(m.day.date()), match=f"{m.team1} v {m.team2}",
                                              a=m.winner, b=c.winner))
            # first-innings score check
            if pd.notna(m.t1_runs) and pd.notna(c.get("inn1_runs")):
                cb_first = m.t1_runs if m.bat_first == m.team1 else m.t2_runs
                validation["cricbuzz_vs_cricsheet"]["score_agree" if cb_first == c.inn1_runs else "score_disagree"] += 1
            if not isinstance(r.get("bat_first"), str) and isinstance(c.get("inn1_team"), str):
                r["bat_first"] = c.inn1_team
        e = find(es, m.day, m.team1, m.team2)
        if e is not None:
            used_es.add(e.espn_id)
            src.append("espn")
            r.update(espn_id=e.espn_id)
            if isinstance(e.toss_winner, str):
                if isinstance(r.get("toss_winner"), str):
                    validation["cricsheet_vs_espn"]["toss_agree" if r["toss_winner"] == e.toss_winner else "toss_disagree"] += 1
                r.update(toss_winner=e.toss_winner, toss_decision=e.toss_decision)
            same = e.team1 == m.team1
            r.update(points1=e.points1 if same else e.points2, points2=e.points2 if same else e.points1)
            if isinstance(m.winner, str) and isinstance(e.winner, str):
                validation["cricbuzz_vs_espn"]["winner_agree" if m.winner == e.winner else "winner_disagree"] += 1
                if m.winner != e.winner:
                    disagreements.append(dict(pair="cricbuzz/espn", date=str(m.day.date()), match=f"{m.team1} v {m.team2}",
                                              a=m.winner, b=e.winner))
            # ESPN already has a result Cricbuzz has not published yet
            if e.state == "post" and m.state not in ("Complete", "complete", "Abandon"):
                s1, s2 = (e.score1, e.score2) if same else (e.score2, e.score1)
                (r1, w1, o1), (r2, w2, o2) = parse_espn_score(s1), parse_espn_score(s2)
                abandoned = "abandon" in str(e.result_note).lower() or "no result" in str(e.result_note).lower()
                r.update(state="Abandon" if abandoned else "Complete", status=e.result_note,
                         winner=e.winner if isinstance(e.winner, str) else None, no_result=int(abandoned),
                         t1_runs=r1, t1_wkts=w1, t1_overs=o1 or (20 if r1 is not None else None),
                         t2_runs=r2, t2_wkts=w2, t2_overs=o2 or (20 if r2 is not None else None),
                         result_source="espn")
                if r1 is not None and r2 is not None and not isinstance(r.get("bat_first"), str):
                    # the side with a "target" in its score batted second
                    r["bat_first"] = r["team2"] if "target" in str(s1) else r["team1"]
                validation["gaps_filled"]["result_from_espn"] += 1
    r["sources"] = "|".join(src)
    rows.append(r)

for _, c in cs[~cs.cs_id.isin(used_cs)].iterrows():
    first_is_t1 = c.inn1_team == c.team1
    r1, w1, b1 = (c.inn1_runs, c.inn1_wkts, c.inn1_balls) if first_is_t1 else (c.inn2_runs, c.inn2_wkts, c.inn2_balls)
    r2, w2, b2 = (c.inn2_runs, c.inn2_wkts, c.inn2_balls) if first_is_t1 else (c.inn1_runs, c.inn1_wkts, c.inn1_balls)
    stage = str(c.stage) if isinstance(c.stage, str) else ""
    rows.append(dict(
        season=c.season, competition="T20 Challenge", series_id=None, match_id=f"cs{c.cs_id}", date=f"{c.date} 12:00",
        stage=stage, state="complete", team1=c.team1, team2=c.team2, team1_name=c.team1_name, team2_name=c.team2_name,
        venue=c.venue, city=c.city, t1_runs=r1, t1_wkts=w1, t1_overs=overs_str(b1) if pd.notna(b1) else None,
        t2_runs=r2, t2_wkts=w2, t2_overs=overs_str(b2) if pd.notna(b2) else None,
        status=(f"{c.winner} won by {int(c.win_by_runs)} runs" if pd.notna(c.win_by_runs) else
                f"{c.winner} won by {int(c.win_by_wickets)} wkts" if pd.notna(c.win_by_wickets) else
                f"{c.winner} won ({c.result or 'super over'})") if isinstance(c.winner, str) else c.result, winner=c.winner if isinstance(c.winner, str) else None,
        bat_first=c.inn1_team, no_result=c.no_result, tied=c.tied, slug="", day=c.day, cb_id=None, cs_id=c.cs_id,
        espn_id=None, toss_winner=c.toss_winner, toss_decision=c.toss_decision, points1=None, points2=None,
        result_source="cricsheet", sources="cricsheet"))
    validation["gaps_filled"]["match_only_in_cricsheet"] += 1

U = pd.DataFrame(rows)
U["match_key"] = U.apply(lambda r: str(int(r.cb_id)) if pd.notna(r.cb_id) else f"cs{r.cs_id}", axis=1)
U["vkey"] = U.venue.map(venue_key)
U = U.sort_values("date").reset_index(drop=True)
U = U.drop(columns=["day", "slug"])
U.to_csv(os.path.join(OUT, "matches.csv"), index=False)
print(f"unified matches: {len(U)} ({Counter(U.sources)})")

# ----------------------------------------------------------- players
ath = pd.read_csv(os.path.join(D, "espn", "athletes.csv"), dtype=str)
reg = pd.read_csv(os.path.join(D, "cricsheet", "people.csv"), dtype=str).set_index("identifier")["key_cricinfo"]
ath["n1"], ath["n2"] = ath.name.map(norm), ath.full_name.map(norm)
by_name = {}
for _, a in ath.iterrows():
    for n in {a.n1, a.n2}:
        by_name.setdefault(n, set()).add(a.cricinfo_id)
by_initial = defaultdict(set)
for _, a in ath.iterrows():
    parts = a.n1.split()
    if len(parts) >= 2:
        by_initial[(parts[0][0], parts[-1])].add(a.cricinfo_id)

cb_bat = pd.read_csv(os.path.join(D, "batting.csv"))
cb_bowl = pd.read_csv(os.path.join(D, "bowling.csv"))
cb_players = pd.concat([cb_bat[["player_id", "player"]], cb_bowl[["player_id", "player"]]]).drop_duplicates("player_id")


def match_name(name):
    n = norm(name)
    hits = by_name.get(n, set())
    if len(hits) == 1:
        return next(iter(hits)), "exact"
    parts = n.split()
    if len(parts) >= 2:
        hits = by_initial.get((parts[0][0], parts[-1]), set())
        if len(hits) == 1:
            return next(iter(hits)), "initial+surname"
    return None, None


cb2ci, how = {}, Counter()
for _, p in cb_players.iterrows():
    ci, h = match_name(p.player)
    if ci:
        cb2ci[int(p.player_id)] = ci
    how[h or "unmatched"] += 1
validation["player_crosswalk"] = dict(how, cricbuzz_players=len(cb_players))
print(f"Cricbuzz -> ESPNcricinfo player match: {dict(how)}")


def key_cb(pid):
    ci = cb2ci.get(int(pid))
    return f"ci:{ci}" if ci else f"cb:{int(pid)}"


def key_cs(ident):
    ci = reg.get(ident) if isinstance(ident, str) else None
    return f"ci:{ci}" if isinstance(ci, str) else f"cs:{ident}"


# ----------------------------------------------------- batting / bowling
cb_scored = set(cb_bat.match_id.astype(str))
_with_cs = U.dropna(subset=["cs_id"])
mk_by_cs = dict(zip(_with_cs.cs_id.astype(str), _with_cs.match_key))
cs_bat = pd.read_csv(os.path.join(D, "cricsheet", "cs_batting.csv"), dtype={"cs_id": str})
cs_bowl = pd.read_csv(os.path.join(D, "cricsheet", "cs_bowling.csv"), dtype={"cs_id": str})

b1 = cb_bat.assign(match_key=cb_bat.match_id.astype(str), player_key=cb_bat.player_id.map(key_cb), source="cricbuzz")
b2 = cs_bat.assign(match_key=cs_bat.cs_id.map(mk_by_cs), player_key=cs_bat.cs_player.map(key_cs), source="cricsheet")
b2 = b2[~b2.match_key.isin(cb_scored)]
cols = ["match_key", "season", "date", "innings", "team", "opponent", "venue", "player_key", "player", "position",
        "runs", "balls", "fours", "sixes", "out", "source"]
BAT = pd.concat([b1[cols], b2[cols]], ignore_index=True)
w1 = cb_bowl.assign(match_key=cb_bowl.match_id.astype(str), player_key=cb_bowl.player_id.map(key_cb), source="cricbuzz")
w2 = cs_bowl.assign(match_key=cs_bowl.cs_id.map(mk_by_cs), player_key=cs_bowl.cs_player.map(key_cs), source="cricsheet")
w2 = w2[~w2.match_key.isin(cb_scored)]
wcols = ["match_key", "season", "date", "innings", "team", "opponent", "venue", "player_key", "player", "overs", "runs", "wickets", "source"]
BOWL = pd.concat([w1[wcols], w2[wcols]], ignore_index=True)
BAT.to_csv(os.path.join(OUT, "batting.csv"), index=False)
BOWL.to_csv(os.path.join(OUT, "bowling.csv"), index=False)
print(f"unified batting rows {len(BAT)} ({dict(Counter(BAT.source))}), bowling rows {len(BOWL)}")

# player table: prefer full Cricbuzz/ESPN display names over Cricsheet initials
names = {}
for _, r in pd.concat([b2[["player_key", "player"]], w2[["player_key", "player"]]]).iterrows():
    names.setdefault(r.player_key, r.player)
ath_by_ci = ath.set_index("cricinfo_id")
for k in list(names):
    if k.startswith("ci:") and k[3:] in ath_by_ci.index:
        names[k] = ath_by_ci.loc[k[3:], "name"]
for _, r in pd.concat([b1[["player_key", "player"]], w1[["player_key", "player"]]]).iterrows():
    names[r.player_key] = r.player
ci2cb = {v: k for k, v in cb2ci.items()}
P = []
for k, n in names.items():
    ci = k[3:] if k.startswith("ci:") else None
    a = ath_by_ci.loc[ci] if ci in ath_by_ci.index else None
    P.append(dict(player_key=k, name=n, cricinfo_id=ci, cricbuzz_id=ci2cb.get(ci) if ci else (k[3:] if k.startswith("cb:") else None),
                  role=a.role if a is not None else None, bat_style=a.bat_style if a is not None else None,
                  bowl_style=a.bowl_style if a is not None else None, dob=a.dob if a is not None else None))
P = pd.DataFrame(P)
P.to_csv(os.path.join(OUT, "players.csv"), index=False)

# --------------------------------------------------------------- squads
sq = pd.read_csv(os.path.join(D, "squads", "crictracker_2026.csv"))
name_to_key = defaultdict(set)
for _, p in P.iterrows():
    name_to_key[norm(p["name"])].add(p.player_key)
for _, a in ath.iterrows():
    name_to_key[a.n2].add(f"ci:{a.cricinfo_id}")
recent_team = BAT[BAT.season >= "2024-25"].sort_values("date").groupby("player_key").team.last().to_dict()
init_idx = defaultdict(set)
for _, p in P.iterrows():
    parts = norm(p["name"]).split()
    if len(parts) >= 2:
        init_idx[(parts[0][0], parts[-1])].add(p.player_key)
sk, sq_how = [], Counter()
for _, s in sq.iterrows():
    hits = name_to_key.get(norm(s.player), set())
    if len(hits) != 1:
        parts = norm(s.player).split()
        cand = init_idx.get((parts[0][0], parts[-1]), set()) if len(parts) >= 2 else set()
        # break ties with the team the player last appeared for
        cand_t = {c for c in cand if recent_team.get(c) == s.team}
        hits = cand_t if len(cand_t) == 1 else (cand if len(cand) == 1 else set())
    k = next(iter(hits)) if len(hits) == 1 else None
    sq_how["matched" if k else "no history"] += 1
    sk.append(k)
sq["player_key"] = sk
sq.to_csv(os.path.join(OUT, "squads_2026.csv"), index=False)
validation["squads"] = dict(sq_how, players=len(sq), teams=int(sq.team.nunique()))
print(f"squad players linked to history: {dict(sq_how)}")

# ------------------------------------------------------- 2026 standings
cbpt = pd.read_csv(os.path.join(D, "points_table_2026.csv")).set_index("team")
cur = U[(U.season == "2026") & U.stage.str.startswith("Pool") & U.state.isin(["Complete", "Abandon"])]


def ov(o):
    if o is None or (isinstance(o, float) and np.isnan(o)):
        return np.nan
    w = int(float(o))
    return w + round((float(o) - w) * 10) / 6


agg = defaultdict(lambda: dict(played=0, won=0, lost=0, nr=0, pts=0.0, rf=0.0, of=0.0, ra=0.0, ob=0.0))
for _, m in cur.iterrows():
    for side, opp, pts in (("1", "2", m.points1), ("2", "1", m.points2)):
        t = m[f"team{side}"]
        a = agg[t]
        a["played"] += 1
        if m.no_result:
            a["nr"] += 1
            a["pts"] += pts if pd.notna(pts) else 2
            continue
        won = m.winner == t
        a["won" if won else "lost"] += 1
        a["pts"] += pts if pd.notna(pts) else (4 if won else 0)
        # NRR: a side bowled out is charged its full allocation (20 overs, or the reduced allocation)
        alloc = 20 if max(ov(m.t1_overs) or 0, ov(m.t2_overs) or 0) > 6.01 else 6
        rf, wf, of_ = m[f"t{side}_runs"], m[f"t{side}_wkts"], ov(m[f"t{side}_overs"])
        rA, wA, oA = m[f"t{opp}_runs"], m[f"t{opp}_wkts"], ov(m[f"t{opp}_overs"])
        if pd.notna(rf) and pd.notna(rA):
            a["rf"] += rf
            a["of"] += alloc if wf >= 10 else of_
            a["ra"] += rA
            a["ob"] += alloc if wA >= 10 else oA
st = []
for t, row in cbpt.iterrows():
    a = agg[t]
    nrr = (a["rf"] / a["of"] - a["ra"] / a["ob"]) if a["of"] and a["ob"] else 0.0
    cb_counted = int(row.played)
    st.append(dict(pool=row.pool, team=t, team_name=row.team_name, played=a["played"], won=a["won"], lost=a["lost"],
                   no_result=a["nr"], points_from_results=a["pts"], nrr_computed=round(nrr, 3),
                   cricbuzz_played=cb_counted, cricbuzz_points=int(row.points), cricbuzz_nrr=float(row.nrr)))
S = pd.DataFrame(st)
# Points come from the match results (ESPN's per-match points, which include
# bonus points). Both official tables are kept for comparison.
esp = pd.read_csv(os.path.join(D, "espn", "espn_standings_2026.csv")).set_index("team")
S["espn_points"] = [float(esp.points.get(t, np.nan)) for t in S.team]
S["espn_nrr"] = [float(esp.nrr.get(t, np.nan)) for t in S.team]
S["points"] = S.points_from_results
S["nrr"] = S.nrr_computed
S["form"] = [cbpt.loc[t, "form"] if isinstance(cbpt.loc[t, "form"], str) else "" for t in S.team]
S.to_csv(os.path.join(OUT, "standings_2026.csv"), index=False)
up_to_date = S[S.played == S.cricbuzz_played]
validation["standings"] = dict(
    teams=len(S),
    agree_with_espn_points=int((S.points == S.espn_points).sum()),
    agree_with_espn_nrr=int(((S.nrr - S.espn_nrr).abs() < 0.01).sum()),
    agree_with_cricbuzz_points=int((S.points == S.cricbuzz_points).sum()),
    cricbuzz_behind=[r.team for _, r in S.iterrows() if r.played > r.cricbuzz_played],
    cricbuzz_points_differ={r.team: dict(results=r.points, cricbuzz=r.cricbuzz_points, espn=r.espn_points)
                            for _, r in S.iterrows() if r.points != r.cricbuzz_points and r.played == r.cricbuzz_played})
print(S[["pool", "team", "played", "points", "nrr", "espn_points", "espn_nrr", "cricbuzz_points", "cricbuzz_nrr"]].to_string())

# ------------------------------------------------------------- report
validation = {k: dict(v) for k, v in validation.items()}
validation["disagreements"] = disagreements
validation["source_counts"] = {
    "cricbuzz_matches": int(len(cb)), "cricsheet_matches": int(len(cs)), "espn_matches": int(len(es)),
    "unified_matches": int(len(U)), "unified_batting_rows": int(len(BAT)), "unified_bowling_rows": int(len(BOWL)),
    "players": int(len(P)), "athlete_profiles": int(len(ath)), "squad_players": int(len(sq)),
    "seasons": sorted(U.season.astype(str).unique().tolist()),
}
json.dump(validation, open(os.path.join(OUT, "validation.json"), "w"), indent=1, default=int)
print(json.dumps({k: v for k, v in validation.items() if k != "disagreements"}, indent=1, default=int))
print("disagreements:", disagreements)
