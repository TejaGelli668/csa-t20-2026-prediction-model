"""CSA T20 Challenge 2026 winner model.

Pipeline
  1. Load the unified multi-source dataset (data/unified/, built by
     build_dataset.py from Cricbuzz + Cricsheet + ESPN/ESPNcricinfo +
     CricTracker squads), Open-Meteo rain data and curated availability.
  2. Build pre-match features for every historic match (no leakage: each
     feature only uses information from before that match).
       elo_diff   Elo rating gap (warm-started from 2011-12)
       home       +1 team A at home, -1 team B at home, 0 neutral
       form_diff  exponentially weighted win rate over last 8 games
       rrd_diff   exponentially weighted run-rate differential
       bat_diff   batting strength of the XI (player ratings, top 7)
       bowl_diff  bowling strength of the XI (player ratings, top 5)
       h2h        shrunk head-to-head win rate
       venue_diff shrunk win rate at this venue
  3. Train Elo / Logistic Regression / Random Forest / Gradient Boosting,
     back-test season by season, and blend them by out-of-sample log loss.
  4. Monte Carlo the rest of the tournament (pool → Super Eights → semis →
     final) and write web/src/data/model_output.json for the React app.
"""
import json, math, os, re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scrape_weather import venue_key

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "web", "src", "data", "model_output.json")
AS_OF = "2026-09-30"
SEED = 2026
N_SIMS = 20000
rng = np.random.default_rng(SEED)

TEAM_NAMES = {
    "WPR": "Western Province", "TIT": "Titans", "LIONS": "Lions", "DOL": "Dolphins", "KNG": "Knights",
    "WAR": "Warriors", "BOL": "Boland", "NWEST": "North West Dragons", "KZNIN": "KZN Inland Tuskers",
    "BOR": "Border", "ESTORM": "Eastern Storm", "NCAPE": "Northern Cape", "SWD": "South Western Districts",
    "LIMPO": "Limpopo", "MPR": "Mpumalanga Rhinos", "SAEP": "SA Emerging Players",
}
# Starting Elo reflects the division a team entered the data in. Div-2
# provinces only met franchise sides occasionally before 2026, so they start
# lower and the data moves them from there.
ELO_INIT = {t: 1500 for t in ["WPR", "TIT", "LIONS", "DOL", "KNG", "WAR"]}
ELO_INIT.update({t: 1460 for t in ["BOL", "NWEST", "KZNIN"]})
ELO_INIT.update({t: 1400 for t in ["BOR", "ESTORM", "NCAPE", "SWD", "LIMPO", "MPR"]})
ELO_INIT["SAEP"] = 1420
ELO_INIT["IMPI"] = 1450  # short-lived franchise in the Cricsheet era
ELO_K, ELO_HOME, ELO_REGRESS = 24, 35, 0.25
FEATURES = ["elo_diff", "home", "form_diff", "rrd_diff", "bat_diff", "bowl_diff", "h2h", "venue_diff"]
# Form, run-rate margin, head-to-head and venue record were tested (see the
# ablation below) and made out-of-sample log loss worse, so the models only use
# these four. The others are still reported as context.
MODEL_FEATURES = ["elo_diff", "home", "bat_diff", "bowl_diff"]
ABLATION_SETS = {
    "All 8 features": FEATURES,
    "Elo + home + batting + bowling (chosen)": MODEL_FEATURES,
    "Chosen + recent form + run-rate margin": MODEL_FEATURES + ["form_diff", "rrd_diff"],
    "Chosen + head-to-head + venue record": MODEL_FEATURES + ["h2h", "venue_diff"],
    "Elo + home only": ["elo_diff", "home"],
    "Batting + bowling + home only": ["bat_diff", "bowl_diff", "home"],
}
FEATURE_LABELS = {
    "elo_diff": "Elo rating gap", "home": "Home advantage", "form_diff": "Recent form (win rate)",
    "rrd_diff": "Recent run-rate margin", "bat_diff": "Batting strength of XI", "bowl_diff": "Bowling strength of XI",
    "h2h": "Head-to-head record", "venue_diff": "Record at this venue",
}


# ----------------------------------------------------------------- helpers
def overs_to_float(o):
    if o is None or (isinstance(o, float) and math.isnan(o)):
        return np.nan
    whole = int(float(o))
    balls = int(round((float(o) - whole) * 10))
    return whole + balls / 6


def vkey(v):
    """Venue key shared with the weather data (falls back to a slug)."""
    k = venue_key(v)
    return k or re.sub(r"[^a-z]", "", str(v).lower())


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -------------------------------------------------------------- load data
UNI = os.path.join(DATA, "unified")
matches = pd.read_csv(os.path.join(UNI, "matches.csv"), dtype={"match_key": str})
matches["season"] = matches["season"].astype(str)
matches["dt"] = pd.to_datetime(matches["date"], format="mixed")
matches["vkey"] = matches["venue"].map(vkey)
bat = pd.read_csv(os.path.join(UNI, "batting.csv"), dtype={"match_key": str, "season": str})
bowl = pd.read_csv(os.path.join(UNI, "bowling.csv"), dtype={"match_key": str, "season": str})
bat = bat.rename(columns={"player_key": "player_id"})
bowl = bowl.rename(columns={"player_key": "player_id"})
validation = json.load(open(os.path.join(UNI, "validation.json")))
squads = pd.read_csv(os.path.join(UNI, "squads_2026.csv"))
bat["dt"], bowl["dt"] = pd.to_datetime(bat["date"], format="mixed"), pd.to_datetime(bowl["date"], format="mixed")
bowl["ov"] = bowl["overs"].map(overs_to_float)
bat["inns"] = ((bat["balls"] > 0) | (~bat["out"].fillna("").isin(["", "not out"]))).astype(int)
points = pd.read_csv(os.path.join(UNI, "standings_2026.csv"))  # rebuilt from results, checked vs ESPN
avail_cfg = json.load(open(os.path.join(DATA, "player_availability.json")))

cur = matches[matches.season == "2026"].copy()
HOME = cur[cur.team1 != "TBC"].groupby("team1")["venue"].agg(lambda s: s.mode()[0]).to_dict()
HOME_V = {t: vkey(v) for t, v in HOME.items()}
POOL = dict(zip(points.team, points.pool))

for side in ("t1", "t2"):
    wk = matches[f"{side}_wkts"]
    ov = matches[f"{side}_overs"].map(overs_to_float)
    matches[f"{side}_rr"] = matches[f"{side}_runs"] / np.where(wk >= 10, 20, ov)

decided = matches[(matches.winner.notna()) & (matches.no_result == 0)].sort_values("dt").reset_index(drop=True)
LEAGUE_ECON = bowl["runs"].sum() / bowl["ov"].sum()


def margin_mult(status):
    s = str(status)
    if "super over" in s.lower() or "tie" in s.lower():
        return 0.6
    m = re.search(r"by (\d+) run", s)
    if m:
        return 1 + 0.5 * min(int(m.group(1)) / 20, 2)
    m = re.search(r"by (\d+) wkt", s)
    if m:
        return 1 + 0.5 * min(int(m.group(1)) / 5, 2)
    return 1.0


# ------------------------------------------------------- player ratings
def player_ratings(before, half_life=365):
    """Decay-weighted, shrunk player values using only data before `before`.

    Batting value: runs per innings plus half the runs above a 125 strike rate.
    Bowling value: runs saved per match vs league economy, 22 runs per wicket.
    """
    b = bat[bat.dt < before]
    w = bowl[bowl.dt < before]
    if b.empty:
        return {}, {}
    bw = 0.5 ** ((before - b.dt).dt.days / half_life)
    ww = 0.5 ** ((before - w.dt).dt.days / half_life)
    bg = pd.DataFrame({"pid": b.player_id, "runs": b.runs * bw, "balls": b.balls * bw, "inns": b.inns * bw})
    bg = bg.groupby("pid").sum()
    bat_v = (bg.runs + 0.5 * (bg.runs - 1.25 * bg.balls) + 4 * 10) / (bg.inns + 4)
    wg = pd.DataFrame({"pid": w.player_id, "val": (22 * w.wickets + LEAGUE_ECON * w.ov - w.runs) * ww, "n": ww})
    wg = wg.groupby("pid").sum()
    bowl_v = wg.val / (wg.n + 3)
    return bat_v.to_dict(), bowl_v.to_dict()


BAT_REPL, BOWL_REPL = 10.0, -3.0


def lineup_strength(pids, bat_v, bowl_v):
    bv = sorted((bat_v.get(p, BAT_REPL) for p in pids), reverse=True)[:7]
    wv = sorted((bowl_v.get(p, BOWL_REPL) for p in pids), reverse=True)[:5]
    bv += [BAT_REPL] * (7 - len(bv))
    wv += [BOWL_REPL] * (5 - len(wv))
    return sum(bv), sum(wv)


# ------------------------------------------------------ feature building
elo = dict(ELO_INIT)
elo_hist = defaultdict(list)
results = defaultdict(list)  # team -> [(dt, win, rrd, opp, vkey)]
last_season = None
rows = []
xi = bat.groupby(["match_key", "team"]).player_id.apply(list).to_dict()


def form_feats(team, upto=8, decay=0.85):
    r = results[team][-upto:]
    if not r:
        return 0.5, 0.0
    ws = np.array([decay ** (len(r) - 1 - i) for i in range(len(r))])
    win = (np.dot(ws, [x[1] for x in r]) + 2 * 0.5) / (ws.sum() + 2)
    rrd = np.dot(ws, [x[2] for x in r]) / (ws.sum() + 1)
    return win, rrd


def h2h_feat(a, b):
    g = [x for x in results[a] if x[3] == b]
    return (sum(x[1] for x in g) + 1) / (len(g) + 2) - 0.5


def venue_feat(team, vk):
    g = [x for x in results[team] if x[4] == vk]
    return (sum(x[1] for x in g) + 1) / (len(g) + 2) - 0.5


def home_flag(a, b, vk):
    return 1 if HOME_V.get(a) == vk else (-1 if HOME_V.get(b) == vk else 0)


rating_cache = {}
for _, m in decided.iterrows():
    if m.season != last_season:  # regress towards starting prior between seasons
        for t in elo:
            elo[t] = elo[t] + ELO_REGRESS * (ELO_INIT.get(t, 1450) - elo[t])
        last_season = m.season
    a, b = m.team1, m.team2
    for t in (a, b):
        elo.setdefault(t, ELO_INIT.get(t, 1450))
    y = 1 if m.winner == a else 0
    h = home_flag(a, b, m.vkey)
    fa, fb = form_feats(a), form_feats(b)
    has_sc = (m.match_key, a) in xi and (m.match_key, b) in xi
    feat = dict(match_id=m.match_key, season=m.season, date=m.date, team_a=a, team_b=b, y=y, venue=m.venue,
                elo_diff=elo[a] - elo[b], home=h, form_diff=fa[0] - fb[0], rrd_diff=fa[1] - fb[1],
                h2h=h2h_feat(a, b), venue_diff=venue_feat(a, m.vkey) - venue_feat(b, m.vkey), has_sc=has_sc)
    if has_sc:
        day = m["dt"].normalize()
        if day not in rating_cache:
            rating_cache[day] = player_ratings(day)
        bv, wv = rating_cache[day]
        sa, sb = lineup_strength(xi[(m.match_key, a)], bv, wv), lineup_strength(xi[(m.match_key, b)], bv, wv)
        feat.update(bat_diff=sa[0] - sb[0], bowl_diff=sa[1] - sb[1])
    rows.append(feat)

    # post-match updates
    exp = 1 / (1 + 10 ** (-(elo[a] - elo[b] + ELO_HOME * h) / 400))
    delta = ELO_K * margin_mult(m.status) * (y - exp)
    elo[a] += delta
    elo[b] -= delta
    for t in (a, b):
        elo_hist[t].append({"date": m.date[:10], "elo": round(elo[t], 1), "season": m.season})
    rr = (m.t1_rr - m.t2_rr) if not (np.isnan(m.t1_rr) or np.isnan(m.t2_rr)) else (1.0 if y else -1.0)
    results[a].append((m["dt"], y, rr, b, m.vkey))
    results[b].append((m["dt"], 1 - y, -rr, a, m.vkey))

feats = pd.DataFrame(rows)
# 2011-12 is used only to warm up Elo and player ratings.
train_df = feats[feats.has_sc & (feats.season >= "2012-13")].reset_index(drop=True)
print(f"Feature rows: {len(feats)}, with lineup data: {len(train_df)}")


def augment(df):
    flip = df.copy()
    for f in FEATURES:
        flip[f] = -flip[f]
    flip["y"] = 1 - flip["y"]
    flip["team_a"], flip["team_b"] = df["team_b"], df["team_a"]
    return pd.concat([df, flip], ignore_index=True)


def make_models():
    return {
        # ~200 matches is a small sample, so every model is heavily regularised.
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=600, max_depth=3, min_samples_leaf=15, max_features=3,
                                                random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=1, learning_rate=0.03,
                                                        subsample=0.7, min_samples_leaf=10, random_state=SEED),
    }


def elo_prob(df):
    return 1 / (1 + 10 ** (-(df["elo_diff"] + ELO_HOME * df["home"]) / 400))


def sym_predict(model, df):
    """Average p(A beats B) and 1 - p(B beats A) so team order can't matter."""
    flip = df[MODEL_FEATURES].copy() * -1
    return (model.predict_proba(df[MODEL_FEATURES])[:, 1] + 1 - model.predict_proba(flip)[:, 1]) / 2


# ---------------------------------------------------- season back-testing
test_seasons = sorted(s for s in train_df.season.unique() if s >= "2015-16")
oof = defaultdict(list)
fold_metrics = []
for s in test_seasons:
    tr, te = train_df[train_df.season < s], train_df[train_df.season == s]
    if len(tr) < 60 or te.empty:  # need a season+ of lineup data to train on
        continue
    preds = {"Elo": elo_prob(te).values}
    for name, mdl in make_models().items():
        mdl.fit(augment(tr)[MODEL_FEATURES], augment(tr)["y"])
        preds[name] = sym_predict(mdl, te)
    for name, p in preds.items():
        oof[name].append(pd.DataFrame({"season": s, "y": te.y.values, "p": p}))
        fold_metrics.append(dict(model=name, season=s, n=len(te), log_loss=log_loss(te.y, p, labels=[0, 1]),
                                 brier=brier_score_loss(te.y, p), accuracy=accuracy_score(te.y, p > 0.5)))

metrics = {}
for name, parts in oof.items():
    d = pd.concat(parts)
    metrics[name] = dict(log_loss=log_loss(d.y, d.p), brier=brier_score_loss(d.y, d.p),
                         accuracy=accuracy_score(d.y, d.p > 0.5), auc=roc_auc_score(d.y, d.p), n=len(d))
base = log_loss(pd.concat(oof["Elo"]).y, np.full(len(pd.concat(oof["Elo"])), 0.5))
# Blend weight: skill over a coin flip (in log loss), floored at a small value.
skill = {k: max(base - v["log_loss"], 0.002) for k, v in metrics.items()}
weights = {k: v / sum(skill.values()) for k, v in skill.items()}
ens = sum(weights[k] * pd.concat(oof[k]).p.values for k in weights)
yy = pd.concat(oof["Elo"]).y.values
metrics["Ensemble"] = dict(log_loss=log_loss(yy, ens), brier=brier_score_loss(yy, ens),
                           accuracy=accuracy_score(yy, ens > 0.5), auc=roc_auc_score(yy, ens), n=len(yy))
metrics["Coin flip"] = dict(log_loss=base, brier=0.25, accuracy=0.5, auc=0.5, n=len(yy))
for k, v in metrics.items():
    print(f"  {k:20s} logloss {v['log_loss']:.3f}  brier {v['brier']:.3f}  acc {v['accuracy']:.3f}  auc {v['auc']:.3f}")
print("  weights", {k: round(v, 3) for k, v in weights.items()})

ablation = []
for label, cols in ABLATION_SETS.items():
    ps, ys = [], []
    for s_ in test_seasons:
        tr, te = train_df[train_df.season < s_], train_df[train_df.season == s_]
        if len(tr) < 60 or te.empty:
            continue
        a = augment(tr)
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(a[cols], a["y"])
        ps += list((mdl.predict_proba(te[cols])[:, 1] + 1 - mdl.predict_proba(-te[cols])[:, 1]) / 2)
        ys += list(te.y)
    ablation.append(dict(features=label, log_loss=round(log_loss(ys, ps), 4),
                         accuracy=round(accuracy_score(ys, np.array(ps) > 0.5), 4), auc=round(roc_auc_score(ys, ps), 4)))
    print(f"  ablation {label:45s} {ablation[-1]}")
feature_auc = {FEATURE_LABELS[f]: round(float(roc_auc_score(train_df.y, train_df[f])), 3) for f in FEATURES}

# ------------------------------------------------------------ final fit
final = {}
aug = augment(train_df)
for name, mdl in make_models().items():
    mdl.fit(aug[MODEL_FEATURES], aug["y"])
    final[name] = mdl
lr = final["Logistic Regression"]
lr_coef = dict(zip(MODEL_FEATURES, lr[-1].coef_[0]))
importance = {
    "Logistic Regression": {f: float(abs(c)) for f, c in lr_coef.items()},
    "Random Forest": dict(zip(MODEL_FEATURES, map(float, final["Random Forest"].feature_importances_))),
    "Gradient Boosting": dict(zip(MODEL_FEATURES, map(float, final["Gradient Boosting"].feature_importances_))),
}
for k in importance:
    tot = sum(importance[k].values())
    importance[k] = {f: v / tot for f, v in importance[k].items()}


def predict_all(df):
    out = {"Elo": elo_prob(df).values}
    for name, mdl in final.items():
        out[name] = sym_predict(mdl, df)
    out["Ensemble"] = sum(weights[k] * out[k] for k in weights)
    return out


# ------------------------------------------------------- availability
def avail_p(player, date):
    for g in avail_cfg["groups"]:
        if player in g["players"]:
            for w in g["windows"]:
                if w["from"] <= date <= w["to"]:
                    return w["p"], w["reason"]
    return None, None


NOW = pd.Timestamp(AS_OF) + pd.Timedelta(days=1)
BAT_V, BOWL_V = player_ratings(NOW)
recent_rows = pd.concat([bat[["player_id", "player", "team", "season", "dt"]],
                         bowl[["player_id", "player", "team", "season", "dt"]]])
recent_rows = recent_rows[recent_rows.season >= "2024-25"].sort_values("dt")
last_team = recent_rows.groupby("player_id").last()
played_2026 = set(recent_rows[recent_rows.season == "2026"].apply(lambda r: (r.player_id, r.team), axis=1))
n_2026_games = recent_rows[recent_rows.season == "2026"].groupby("team").dt.nunique().to_dict()


PNAME = pd.read_csv(os.path.join(UNI, "players.csv")).set_index("player_key")["name"].to_dict()
SQUAD_KEYS = {t: set(g.player_key) for t, g in squads.dropna(subset=["player_key"]).groupby("team")}


def team_pool(team):
    """Candidates: the official 2026 squad (CricTracker) plus anyone whose
    latest team since 2024-25 is `team`, each with a base chance of playing."""
    sq = SQUAD_KEYS.get(team, set())
    ids = set(last_team[last_team.team == team].index) | sq
    out = []
    for pid in ids:
        lt = last_team.loc[pid] if pid in last_team.index else None
        if (pid, team) in played_2026:
            base, why = 0.95, "In 2026 XI"
        elif pid in sq:
            base, why = 0.6, "In official 2026 squad, yet to play"
        elif sq:
            base, why = 0.1, "Not in official 2026 squad"
        elif lt is not None and lt.season == "2025-26":
            base, why = 0.55, "Played 2025-26, not yet seen in 2026"
        else:
            base, why = 0.35, "Last played 2024-25"
        out.append(dict(pid=pid, player=PNAME.get(pid, lt.player if lt is not None else pid), base=base, why=why,
                        in_squad=pid in sq,
                        bat=float(BAT_V.get(pid, BAT_REPL)), bowl=float(BOWL_V.get(pid, BOWL_REPL))))
    return out


POOLS = {t: team_pool(t) for t in TEAM_NAMES}


def expected_strength(team, date):
    """Greedy expected XI: fill 7 batting / 5 bowling slots with probability mass."""
    players = []
    for p in POOLS[team]:
        a, _ = avail_p(p["player"], date)
        prob = p["base"] if a is None else a  # listed players' status is known
        players.append((p, prob))

    def fill(key, slots, repl):
        tot, mass = 0.0, 0.0
        for p, prob in sorted(players, key=lambda x: -x[0][key]):
            if mass >= slots:
                break
            take = min(prob, slots - mass)
            tot += take * p[key]
            mass += take
        return tot + (slots - mass) * repl

    return fill("bat", 7, BAT_REPL), fill("bowl", 5, BOWL_REPL)


strength_cache = {}


def strength(team, date):
    if (team, date) not in strength_cache:
        if team == "SAEP" or not POOLS.get(team):
            # No scorecard history: use the median provincial side.
            prov = [expected_strength(t, date) for t in ["BOR", "ESTORM", "NCAPE", "SWD", "LIMPO", "MPR"]]
            strength_cache[(team, date)] = (float(np.median([p[0] for p in prov])), float(np.median([p[1] for p in prov])))
        else:
            strength_cache[(team, date)] = expected_strength(team, date)
    return strength_cache[(team, date)]


def match_features(a, b, date, venue_key=None):
    fa, fb = form_feats(a), form_feats(b)
    sa, sb = strength(a, date), strength(b, date)
    vk = venue_key or ""
    return dict(elo_diff=elo[a] - elo[b], home=home_flag(a, b, vk) if vk else 0, form_diff=fa[0] - fb[0],
                rrd_diff=fa[1] - fb[1], bat_diff=sa[0] - sb[0], bowl_diff=sa[1] - sb[1], h2h=h2h_feat(a, b),
                venue_diff=(venue_feat(a, vk) - venue_feat(b, vk)) if vk else 0.0)


# ------------------------------------------------------ upcoming fixtures
upcoming = cur[cur.state.isin(["Upcoming", "Preview", "Toss", "In Progress", "Live"]) & (cur.team1 != "TBC")].sort_values("dt")
fx = pd.DataFrame([dict(match_id=r.match_id, date=r.date, a=r.team1, b=r.team2, venue=r.venue, pool=r.stage,
                        **match_features(r.team1, r.team2, r.date[:10], r.vkey)) for _, r in upcoming.iterrows()])
fx_pred = predict_all(fx)
std = lr[0]
contrib = (fx[MODEL_FEATURES].values - 0) / std.scale_ * lr[-1].coef_[0]  # logit contribution per feature
fixtures_out = []
for i, r in fx.iterrows():
    reasons = sorted(zip(MODEL_FEATURES, contrib[i]), key=lambda x: -abs(x[1]))[:3]
    fixtures_out.append(dict(
        match_id=int(r.match_id), date=r.date, team_a=r.a, team_b=r.b, venue=r.venue, stage=r.pool,
        p_a=round(float(fx_pred["Ensemble"][i]), 4),
        models={k: round(float(v[i]), 4) for k, v in fx_pred.items()},
        features={f: round(float(r[f]), 3) for f in FEATURES},
        reasons=[dict(feature=FEATURE_LABELS[f], favours=r.a if c > 0 else r.b, logit=round(float(c), 3)) for f, c in reasons],
    ))

# ------------------------------------------------------- washout risk
# Fit P(no result) on rain at the venue that day (Open-Meteo archive), using
# Cricbuzz-era matches because they record abandonments (Cricsheet omits them).
WH = pd.read_csv(os.path.join(DATA, "weather", "history.csv"))
WF = pd.read_csv(os.path.join(DATA, "weather", "forecast.csv"))
WH["md"] = WH.date.str[5:]
hist_ix = WH.set_index(["key", "date"])
fc_ix = WF.set_index(["key", "date"])
MAIN_VENUES = ["supersport", "wanderers", "kingsmead", "newlands", "mangaung", "stgeorge", "bolandpark", "senwes"]


def rain_x(mm, hours):
    return np.column_stack([np.log1p(np.asarray(mm, float)), np.asarray(hours, float)])


wx_rows = []
for _, m in matches[matches.sources.str.contains("cricbuzz") & matches.state.str.lower().isin(["complete", "abandon"])].iterrows():
    k = (m.vkey, m.date[:10])
    if k in hist_ix.index:
        h = hist_ix.loc[k]
        wx_rows.append((h.precip_mm, h.precip_hours, int(m.no_result)))
wx = np.array(wx_rows, float)
nr_model = LogisticRegression(C=1.0).fit(rain_x(wx[:, 0], wx[:, 1]), wx[:, 2])
print(f"  washout model on {len(wx)} matches ({int(wx[:, 2].sum())} no-results): coef {nr_model.coef_[0].round(3)}")
WH["p_nr"] = nr_model.predict_proba(rain_x(WH.precip_mm.fillna(0), WH.precip_hours.fillna(0)))[:, 1]


def climatology(vk, date):
    """Average modelled washout chance for this venue within +/-10 days of the date, 2011-2025."""
    d = pd.Timestamp(date)
    window = {(d + pd.Timedelta(days=i)).strftime("%m-%d") for i in range(-10, 11)}
    keys = [vk] if vk in set(WH.key) else MAIN_VENUES
    sub = WH[WH.key.isin(keys) & WH.md.isin(window)]
    return float(sub.p_nr.mean())


def washout(vk, date):
    """(p_no_result, source, forecast_mm) for a venue key and ISO date."""
    clim = climatology(vk, date)
    if vk and (vk, date) in fc_ix.index:
        f = fc_ix.loc[(vk, date)]
        mm, hrs = (0.0 if pd.isna(f.precip_mm) else f.precip_mm), (0.0 if pd.isna(f.precip_hours) else f.precip_hours)
        pf = float(nr_model.predict_proba(rain_x([mm], [hrs]))[0, 1])
        lead = (pd.Timestamp(date) - pd.Timestamp(AS_OF)).days
        w = 1.0 if lead <= 7 else 0.5  # longer-range forecasts lean on climatology
        prob = None if pd.isna(f.precip_prob) else float(f.precip_prob)
        return w * pf + (1 - w) * clim, "forecast" if w == 1 else "forecast+climate", float(mm), prob
    return clim, "climatology", None, None


for f in fixtures_out:
    vk = vkey(f["venue"])
    pnr, src, mm, prob = washout(vk if vk in set(WH.key) else None, f["date"][:10])
    f.update(p_no_result=round(pnr, 4), weather_source=src, forecast_mm=mm, forecast_rain_prob=prob)

WV = pd.read_csv(os.path.join(DATA, "weather", "venues.csv")).drop_duplicates("key").set_index("key")
venue_rain = [dict(key=k, venue=WV.loc[k, "venue"], city=WV.loc[k, "city"],
                   oct_washout=round(climatology(k, "2026-10-15"), 4), nov_washout=round(climatology(k, "2026-11-10"), 4),
                   home_team=next((t for t, hv in HOME_V.items() if hv == k), None))
              for k in WV.index if k != "nwcoval"]

# -------------------------------------------------------- simulation
recent = matches[(matches.season >= "2024-25") & matches.sources.str.contains("cricbuzz")]
p_nr = float(recent.no_result.sum() / len(recent[recent.state.str.lower().isin(["complete", "abandon"])]))
pts_known = matches[matches.points1.notna() & (matches.no_result == 0) & matches.winner.notna()]
win_pts = np.where(pts_known.winner == pts_known.team1, pts_known.points1, pts_known.points2)
bonus_observed = float((win_pts == 5).mean()) if len(win_pts) else None
dec = decided[decided.season >= "2022-23"].dropna(subset=["t1_rr", "t2_rr"])
win_rr = np.where(dec.winner == dec.team1, dec.t1_rr - dec.t2_rr, dec.t2_rr - dec.t1_rr)
lose_rr = np.where(dec.winner == dec.team1, dec.t2_rr, dec.t1_rr)
bonus_flag = (win_rr + lose_rr) >= 1.25 * lose_rr
margin_pool = np.column_stack([win_rr, bonus_flag]).astype(float)
margin_pool = margin_pool[np.isfinite(margin_pool[:, 0])]
print(f"  p(no result) league={p_nr:.3f}  p(bonus|win) from margins={margin_pool[:, 1].mean():.3f}, ESPN observed={bonus_observed}")

teams = list(TEAM_NAMES)
idx = {t: i for i, t in enumerate(teams)}
S8_DATES = sorted(cur[cur.stage == "Super Eights"].date.str[:10].tolist())
SF_DATE = cur[cur.stage.str.contains("Semi")].date.str[:10].min()
F_DATE = cur[cur.stage == "Final"].date.str[:10].min()


def prob_matrix(date):
    pairs = [(a, b) for a in teams for b in teams if a != b]
    df = pd.DataFrame([match_features(a, b, date) for a, b in pairs])
    p = predict_all(df)["Ensemble"]
    M = np.full((len(teams), len(teams)), 0.5)
    for (a, b), v in zip(pairs, p):
        M[idx[a], idx[b]] = v
    return M


print("  building neutral-venue probability matrices")
MATS = {d: prob_matrix(d) for d in sorted(set(S8_DATES + [SF_DATE, F_DATE]))}

pool_fx = [(idx[f["team_a"]], idx[f["team_b"]], f["p_a"], f["p_no_result"]) for f in fixtures_out if f["stage"].startswith("Pool")]
S8_NR = [washout(None, d)[0] for d in S8_DATES]
KO_NR = 0.3 * washout(None, SF_DATE)[0]  # knockouts have reserve days
base_pts = np.array([points.set_index("team").points.get(t, 0) for t in teams], float)
base_nrr = np.array([points.set_index("team").nrr.get(t, 0) for t in teams], float)
base_pl = np.array([points.set_index("team").played.get(t, 0) - points.set_index("team").no_result.get(t, 0) for t in teams], float)
pool_a = [idx[t] for t in points[points.pool == "Pool A"].team]
pool_b = [idx[t] for t in points[points.pool == "Pool B"].team]
UNC = 0.25  # per-simulation team strength uncertainty (logit sd)

reach = {k: np.zeros(len(teams)) for k in ["super8", "semi", "final", "champion", "pool_top"]}
pool_pts_sum = np.zeros(len(teams))
pool_rank_sum = np.zeros(len(teams))
finals_pairs = defaultdict(int)


def play(i, j, p, shock, pts, nrr, pl, p_wash):
    """Return winner index (or -1 for no result) and update table arrays."""
    if rng.random() < p_wash:
        pts[i] += 2
        pts[j] += 2
        return -1
    p = sigmoid(logit(p) + shock[i] - shock[j])
    w, l = (i, j) if rng.random() < p else (j, i)
    rrd, bonus = margin_pool[rng.integers(len(margin_pool))]
    pts[w] += 4 + bonus
    nrr[w] = (nrr[w] * pl[w] + rrd) / (pl[w] + 1)
    nrr[l] = (nrr[l] * pl[l] - rrd) / (pl[l] + 1)
    pl[w] += 1
    pl[l] += 1
    return w


def rank(group, pts, nrr):
    return sorted(group, key=lambda t: (-pts[t], -nrr[t], rng.random()))


for s in range(N_SIMS):
    shock = rng.normal(0, UNC, len(teams))
    pts, nrr, pl = base_pts.copy(), base_nrr.copy(), base_pl.copy()
    for i, j, p, pw in pool_fx:
        play(i, j, p, shock, pts, nrr, pl, pw)
    ra, rb = rank(pool_a, pts, nrr), rank(pool_b, pts, nrr)
    pool_pts_sum += pts
    for r, t in enumerate(ra):
        pool_rank_sum[t] += r + 1
    for r, t in enumerate(rb):
        pool_rank_sum[t] += r + 1
    reach["pool_top"][ra[0]] += 1
    reach["pool_top"][rb[0]] += 1
    g1, g2 = [ra[0], rb[1], ra[2], rb[3]], [rb[0], ra[1], rb[2], ra[3]]
    for t in g1 + g2:
        reach["super8"][t] += 1
    # Super Eights: fresh table, round robin inside each group of four.
    p8, n8, l8 = np.zeros(len(teams)), np.zeros(len(teams)), np.zeros(len(teams))
    slot = 0
    for gi, g in enumerate((g1, g2)):
        for x in range(4):
            for y in range(x + 1, 4):
                si = min(slot, len(S8_DATES) - 1)
                M = MATS[S8_DATES[si]]
                play(g[x], g[y], M[g[x], g[y]], shock, p8, n8, l8, S8_NR[si])
                slot += 1
    s1, s2 = rank(g1, p8, n8), rank(g2, p8, n8)
    semis = [(s1[0], s2[1]), (s2[0], s1[1])]
    finalists = []
    for hi, lo in semis:
        reach["semi"][hi] += 1
        reach["semi"][lo] += 1
        M = MATS[SF_DATE]
        if rng.random() < KO_NR:  # washout after reserve day: higher seed goes through
            finalists.append(hi)
        else:
            finalists.append(hi if rng.random() < sigmoid(logit(M[hi, lo]) + shock[hi] - shock[lo]) else lo)
    for t in finalists:
        reach["final"][t] += 1
    a, b = finalists
    M = MATS[F_DATE]
    champ = a if rng.random() < sigmoid(logit(M[a, b]) + shock[a] - shock[b]) else b
    reach["champion"][champ] += 1
    finals_pairs[tuple(sorted((teams[a], teams[b])))] += 1

# ---------------------------------------------------------------- output
def venue_stats():
    """Per venue since 2018-19: first-innings average, chase success and toss effects."""
    out = []
    d = decided[decided.season >= "2018-19"].dropna(subset=["t1_runs", "t2_runs"])
    d = d[d.bat_first.notna()]
    names = WV.venue.to_dict()
    for k, g in d.groupby("vkey"):
        if len(g) < 4:
            continue
        first = np.where(g.bat_first == g.team1, g.t1_runs, g.t2_runs)
        tg = g[g.toss_winner.notna()]
        out.append(dict(venue=names.get(k, g.venue.iloc[0]), matches=int(len(g)), avg_first_innings=round(float(np.mean(first)), 1),
                        chase_win_pct=round(float((g.winner != g.bat_first).mean()) * 100, 1),
                        toss_matches=int(len(tg)),
                        toss_winner_win_pct=round(float((tg.toss_winner == tg.winner).mean()) * 100, 1) if len(tg) else None,
                        field_first_pct=round(float((tg.toss_decision == "field").mean()) * 100, 1) if len(tg) else None,
                        home_team=next((t for t, hv in HOME_V.items() if hv == k), None)))
    return sorted(out, key=lambda x: -x["matches"])


def toss_stats():
    t = decided[decided.toss_winner.notna()]
    return dict(matches=int(len(t)), toss_winner_won=round(float((t.toss_winner == t.winner).mean()), 4),
                chose_field=round(float((t.toss_decision == "field").mean()), 4),
                field_first_won=round(float(((t.toss_decision == "field") & (t.toss_winner == t.winner)).sum() /
                                            max((t.toss_decision == "field").sum(), 1)), 4),
                chasing_side_won=round(float((decided.dropna(subset=["bat_first"]).winner !=
                                              decided.dropna(subset=["bat_first"]).bat_first).mean()), 4))


PH = pd.read_csv(os.path.join(DATA, "cricsheet", "cs_phases.csv"), dtype={"season": str})
PH = PH[PH.season >= "2021-22"]
PH_LEAGUE = (PH.groupby("phase").runs.sum() / PH.groupby("phase").balls.sum() * 6).to_dict()


def phase_profile(team):
    """Run rates by phase from Cricsheet ball-by-ball (2021-22 to 2024-25)."""
    bt, bw = PH[PH.team == team], PH[PH.opponent == team]
    if bt.empty:
        return None
    out = {}
    for ph in ("powerplay", "middle", "death"):
        a, b = bt[bt.phase == ph], bw[bw.phase == ph]
        out[ph] = dict(bat_rr=round(float(a.runs.sum() / a.balls.sum() * 6), 2) if a.balls.sum() else None,
                       bowl_rr=round(float(b.runs.sum() / b.balls.sum() * 6), 2) if b.balls.sum() else None,
                       league_rr=round(float(PH_LEAGUE[ph]), 2))
    out["innings"] = int(bt.drop_duplicates(["cs_id", "innings"]).shape[0])
    return out


def team_venue_record(team):
    rec = defaultdict(lambda: [0, 0])
    for r in results[team]:
        rec[r[4]][0 if r[1] == 1 else 1] += 1
    names = dict(zip(matches.vkey, matches.venue))
    return sorted([dict(venue=names.get(k, k), won=w, lost=l) for k, (w, l) in rec.items()],
                  key=lambda x: -(x["won"] + x["lost"]))[:6]


def champions():
    out = []
    for (season, comp), g in decided[decided.stage.str.contains("Final") & ~decided.stage.str.contains("Semi")].groupby(["season", "competition"]):
        r = g.iloc[-1]
        out.append(dict(season=season, competition=comp, winner=r.winner, runner_up=r.team2 if r.winner == r.team1 else r.team1,
                        result=r.status if isinstance(r.status, str) else ""))
    return out


team_out = []
for t in teams:
    i = idx[t]
    next_date = next((f["date"][:10] for f in fixtures_out if t in (f["team_a"], f["team_b"])), AS_OF)
    row = points.set_index("team").loc[t]
    fw, frr = form_feats(t)
    s_now = strength(t, next_date)
    s_s8 = strength(t, S8_DATES[0])
    key_players = sorted(POOLS.get(t, []), key=lambda p: -(p["bat"] + max(p["bowl"], 0)))[:10]
    kp = []
    for p in key_players:
        a, why = avail_p(p["player"], next_date)
        a8, why8 = avail_p(p["player"], S8_DATES[0])
        kp.append(dict(player=p["player"], bat=round(p["bat"], 1), bowl=round(p["bowl"], 1), status=p["why"], in_squad=p["in_squad"],
                       p_next=round(a if a is not None else p["base"], 2), note=why or "",
                       p_super8=round(a8 if a8 is not None else p["base"], 2), note_super8=why8 or ""))
    team_out.append(dict(
        code=t, name=TEAM_NAMES[t], pool=POOL[t], home=HOME.get(t),
        table=dict(played=int(row.played), won=int(row.won), lost=int(row.lost), nr=int(row.no_result),
                   points=float(row.points), nrr=float(row.nrr), form=row.form if isinstance(row.form, str) else ""),
        elo=round(elo[t], 1), elo_history=elo_hist[t][-40:],
        next_match=next_date,
        form=dict(win_rate=round(fw, 3), rr_margin=round(frr, 2),
                  last5="".join("W" if r[1] else "L" for r in results[t][-5:])),
        strength=dict(bat=round(s_now[0], 1), bowl=round(s_now[1], 1), bat_super8=round(s_s8[0], 1), bowl_super8=round(s_s8[1], 1)),
        key_players=kp, venue_record=team_venue_record(t), phases=phase_profile(t),
        squad=[dict(player=r.player, role=r.role, rated=isinstance(r.player_key, str),
                    bat=round(float(BAT_V.get(r.player_key, BAT_REPL)), 1) if isinstance(r.player_key, str) else None,
                    bowl=round(float(BOWL_V.get(r.player_key, BOWL_REPL)), 1) if isinstance(r.player_key, str) else None,
                    played_2026=(r.player_key, t) in played_2026)
               for _, r in squads[squads.team == t].iterrows()],
        sim=dict(super8=reach["super8"][i] / N_SIMS, semi=reach["semi"][i] / N_SIMS, final=reach["final"][i] / N_SIMS,
                 champion=reach["champion"][i] / N_SIMS, top_of_pool=reach["pool_top"][i] / N_SIMS,
                 exp_pool_points=pool_pts_sum[i] / N_SIMS, exp_pool_rank=pool_rank_sum[i] / N_SIMS),
    ))
team_out.sort(key=lambda x: -x["sim"]["champion"])

completed_2026 = cur[cur.state.isin(["Complete", "Abandon"])].sort_values("dt")
output = dict(
    meta=dict(as_of=AS_OF, generated=pd.Timestamp.now().isoformat(timespec="seconds"), n_sims=N_SIMS,
              n_matches=int(len(matches)), n_decided=int(len(decided)), n_train=int(len(train_df)),
              n_batting_rows=int(len(bat)), n_bowling_rows=int(len(bowl)),
              seasons=sorted(matches.season.unique().tolist()),
              p_no_result=round(p_nr, 3), p_no_result_s8=round(float(np.mean(S8_NR)), 4), p_bonus_given_win=round(float(margin_pool[:, 1].mean()), 3),
              strength_uncertainty=UNC, league_economy=round(float(LEAGUE_ECON), 2),
              p_bonus_observed=round(bonus_observed, 3) if bonus_observed is not None else None,
              washout_model=dict(n=int(len(wx)), no_results=int(wx[:, 2].sum()),
                                 coef_log_mm=round(float(nr_model.coef_[0][0]), 3), coef_hours=round(float(nr_model.coef_[0][1]), 3)),
              sources=[
                  dict(name="Cricbuzz", url="https://www.cricbuzz.com/cricket-series/13180/csa-t20-challenge-2026",
                       provides="Fixtures, results and scores 2018-19 → 2026, official points table, scorecards from 2022-23",
                       records=f"{validation['source_counts']['cricbuzz_matches']} matches"),
                  dict(name="Cricsheet (ball-by-ball)", url="https://cricsheet.org/downloads/ctc_json.zip",
                       provides="Every ball of 314 CSA T20 Challenge matches 2011-12 → 2024-25: lineups, tosses, phase splits",
                       records=f"{validation['source_counts']['cricsheet_matches']} matches"),
                  dict(name="ESPN / ESPNcricinfo API", url="https://www.espncricinfo.com/series/csa-t20-challenge-2026-27-1551607",
                       provides="2025-26 and 2026: tosses, playing XIs with Cricinfo ids, points awarded (bonus points), official standings, player profiles",
                       records=f"{validation['source_counts']['espn_matches']} matches, {validation['source_counts']['athlete_profiles']} player profiles"),
                  dict(name="CricTracker", url="https://www.crictracker.com/t20/csa-t20-challenge/squads/",
                       provides="Official 2026 squads with playing roles", records=f"{validation['source_counts']['squad_players']} players"),
                  dict(name="Open-Meteo", url="https://open-meteo.com/",
                       provides="Daily rain at 17 venues since 2011 and a 16-day forecast", records="93k venue-days"),
              ],
              references=[{"name": s["title"], "url": s["url"]} for s in avail_cfg["sources"]]),
    validation=validation,
    toss=toss_stats(),
    venue_rain=venue_rain,
    prediction=dict(winner=team_out[0]["code"], winner_name=team_out[0]["name"], p=team_out[0]["sim"]["champion"],
                    runner_up=team_out[1]["code"], most_likely_final=[list(k) + [v / N_SIMS] for k, v in
                                                                       sorted(finals_pairs.items(), key=lambda x: -x[1])[:8]]),
    teams=team_out,
    fixtures=fixtures_out,
    results_2026=[dict(date=r.date, stage=r.stage, team1=r.team1, team2=r.team2, venue=r.venue, status=r.status,
                       winner=r.winner if isinstance(r.winner, str) else None, result_source=r.result_source,
                       score1=f"{int(r.t1_runs)}/{int(r.t1_wkts)} ({r.t1_overs})" if pd.notna(r.t1_runs) else "",
                       score2=f"{int(r.t2_runs)}/{int(r.t2_wkts)} ({r.t2_overs})" if pd.notna(r.t2_runs) else "")
                  for _, r in completed_2026.iterrows()],
    models=dict(metrics={k: {m: round(float(x), 4) if m != "n" else int(x) for m, x in v.items()} for k, v in metrics.items()},
                folds=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()} for f in fold_metrics],
                weights={k: round(v, 4) for k, v in weights.items()},
                importance={k: {FEATURE_LABELS[f]: round(v, 4) for f, v in d.items()} for k, d in importance.items()},
                lr_coefficients={FEATURE_LABELS[f]: round(float(c), 4) for f, c in lr_coef.items()},
                features=[dict(key=f, label=FEATURE_LABELS[f], used=f in MODEL_FEATURES) for f in FEATURES],
                ablation=ablation, feature_auc=feature_auc, test_seasons=sorted({f["season"] for f in fold_metrics})),
    venues=venue_stats(),
    champions=champions(),
    availability=avail_cfg,
    assumptions=[
        "Super Eights groups assumed as A1, B2, A3, B4 and B1, A2, B3, A4, with points reset. CSA has not published the seeding.",
        "Semi-finals assumed to be Group 1 winner v Group 2 runner-up and Group 2 winner v Group 1 runner-up. A washed-out semi-final sends the higher seed through.",
        f"Points: win 4, bonus point for a win at 1.25x the opponent's run rate (happened in {margin_pool[:, 1].mean():.0%} of historic wins), no result 2.",
        f"Washout risk is set per match from rain at that venue: Open-Meteo forecast for the next 7 days, a forecast/climate blend for days 8-16, and 2011-2025 climatology after that (Super Eights average {np.mean(S8_NR):.1%}). Knockouts have reserve days, so their risk is cut by 70%.",
        "Super Eights and knockout venues are TBC, so those matches are treated as neutral (no home advantage).",
        "Current points and NRR are rebuilt from every source's match results and agree with ESPNcricinfo's official table for all 16 teams. Cricbuzz's table was behind on two results and 2 points short for Titans and Western Province.",
        "Player availability uses the official squads from CricTracker: squad members who haven't played yet are given a 60% chance per match, and non-squad players 10%.",
        "SA Emerging Players have no scorecard history, so their lineup strength is the median of the six provincial sides.",
        "Test-squad availability is an assumption because the squad had not been announced as of 30 Sep. It is listed in the Availability tab.",
        f"Each simulation gives every team a random strength shock (sd {UNC} in logit terms), so model uncertainty carries through the whole tournament.",
    ],
)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
def clean(o):
    """JSON has no NaN: turn NaN/inf (and numpy scalars) into plain values or null."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(output), open(OUT, "w"), indent=1, allow_nan=False)
print("\nTitle odds:")
for t in team_out:
    print(f"  {t['code']:7s} {t['sim']['champion']:.3f}  S8 {t['sim']['super8']:.2f}  SF {t['sim']['semi']:.2f}  F {t['sim']['final']:.2f}  elo {t['elo']}  bat {t['strength']['bat']} bowl {t['strength']['bowl']}")
print(f"wrote {OUT}")
