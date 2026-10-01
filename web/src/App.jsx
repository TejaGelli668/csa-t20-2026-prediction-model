import React, { useEffect, useMemo, useState } from "react";
import data from "./data/model_output.json";
import { DotPlot, GroupedBars, HBarChart, LineChart, TeamScatter, TipRows } from "./components/charts.jsx";

const TEAMS = Object.fromEntries(data.teams.map((t) => [t.code, t]));
const name = (c) => TEAMS[c]?.name ?? c;
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const fmtDate = (s) =>
  new Date(s.replace(" ", "T") + "Z").toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
const fmtTime = (s) => `${s.slice(11, 16)} GMT`;
const rankOf = (code, key) => {
  const sorted = [...data.teams].sort((a, b) => key(b) - key(a));
  return sorted.findIndex((t) => t.code === code) + 1;
};
const ord = (n) => n + (["th", "st", "nd", "rd"][((n % 100) - 20) % 10] || ["th", "st", "nd", "rd"][n % 100] || "th");

const V = data.validation;
const agree = {
  ok: (V.cricbuzz_vs_cricsheet?.winner_agree || 0) + (V.cricbuzz_vs_espn?.winner_agree || 0),
  total: ["cricbuzz_vs_cricsheet", "cricbuzz_vs_espn"].reduce((n, k) => n + (V[k]?.winner_agree || 0) + (V[k]?.winner_disagree || 0), 0),
};

const TABS = ["Overview", "Fixtures", "Teams", "Models", "Data & Method"];

export default function App() {
  const [tab, setTab] = useState(() => {
    try { return localStorage.getItem("tab") || "Overview"; } catch { return "Overview"; }
  });
  const [theme, setTheme] = useState(null);
  useEffect(() => {
    try { localStorage.setItem("tab", tab); } catch {}
    window.scrollTo({ top: 0 });
  }, [tab]);
  useEffect(() => {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  const toggleTheme = () => {
    const dark = theme ? theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(dark ? "light" : "dark");
  };

  return (
    <>
      <header className="top">
        <div className="wrap">
          <div className="brand">
            <h1>CSA T20 Challenge 2026</h1>
            <span>Winner prediction model · data as of {data.meta.as_of}</span>
          </div>
          <nav className="tabs" role="tablist">
            {TABS.map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>{t}</button>
            ))}
          </nav>
          <button className="theme-btn" onClick={toggleTheme} aria-label="Toggle colour theme">◐ Theme</button>
        </div>
      </header>
      <main className="wrap">
        {tab === "Overview" && <Overview go={setTab} />}
        {tab === "Fixtures" && <Fixtures />}
        {tab === "Teams" && <Teams />}
        {tab === "Models" && <Models />}
        {tab === "Data & Method" && <Method />}
        <footer>
          Built from {data.meta.n_matches} scraped matches ({data.meta.seasons[0]} → {data.meta.seasons.at(-1)}),{" "}
          {data.meta.n_batting_rows.toLocaleString()} batting and {data.meta.n_bowling_rows.toLocaleString()} bowling scorecard rows,{" "}
          {data.meta.n_sims.toLocaleString()} tournament simulations. Generated {data.meta.generated.replace("T", " ")}.
          Statistical estimates, not certainties.
        </footer>
      </main>
    </>
  );
}

/* ------------------------------------------------------------------ Overview */
function Overview({ go }) {
  const fav = data.teams[0];
  const second = data.teams[1];
  const ens = data.models.metrics.Ensemble;
  const coin = data.models.metrics["Coin flip"];
  const final = data.prediction.most_likely_final[0];
  const eloRank = rankOf(fav.code, (t) => t.elo);
  const batRank = rankOf(fav.code, (t) => t.strength.bat);
  const bowlRank = rankOf(fav.code, (t) => t.strength.bowl);
  const poolTeams = data.teams.filter((t) => t.pool === fav.pool).sort((a, b) => b.table.points - a.table.points || b.table.nrr - a.table.nrr);
  const poolPos = poolTeams.findIndex((t) => t.code === fav.code) + 1;
  const titles = data.champions.filter((c) => c.winner === fav.code && c.competition === "T20 Challenge").length;
  const unavailable = fav.key_players.filter((p) => p.p_super8 < 0.3);

  return (
    <>
      <section className="grid g-hero">
        <div className="card hero">
          <div className="eyebrow">Predicted champion</div>
          <div className="big">{fav.name}</div>
          <div className="pct">{pct(fav.sim.champion)} chance to win the title</div>
          <p className="sub" style={{ marginTop: 14 }}>
            {fav.name} are the most likely winner, but this is a flat field: {second.name} are close behind at{" "}
            {pct(second.sim.champion)}, and no team reaches 1 in 5. The most likely final is{" "}
            <b>{name(final[0])} v {name(final[1])}</b> ({pct(final[2])} of simulations).
          </p>
          <div className="prose">
            <p>
              <strong>Why {fav.name}?</strong> They have the {ord(eloRank)}-best Elo rating ({fav.elo}), the{" "}
              {ord(batRank)}-strongest batting and {ord(bowlRank)}-strongest bowling projected XI, and sit{" "}
              {ord(poolPos)} in {fav.pool} with {fav.table.points} points. They reach the Super Eights in {pct(fav.sim.super8, 0)} of
              simulations. History helps too: they have won {titles} of the last {data.champions.filter((c) => c.competition === "T20 Challenge").length} T20 Challenge titles in the data.
              {unavailable.length > 0 && (
                <> The main risk is availability: {unavailable.map((p) => p.player).join(", ")}{" "}
                  {unavailable.length === 1 ? "is" : "are"} likely to miss the Super Eights (
                  {unavailable.map((p) => `${p.player.split(" ").at(-1)}: ${(p.note_super8 || p.status).toLowerCase()}`).join("; ")}).</>
              )}
            </p>
          </div>
        </div>
        <div className="grid g2" style={{ alignContent: "start" }}>
          <Tile label="Simulations" value={data.meta.n_sims.toLocaleString()} note="full tournament Monte Carlo" />
          <Tile label="Back-test accuracy" value={pct(ens.accuracy, 0)} note={`${ens.n} held-out matches (coin flip 50%)`} />
          <Tile label="Back-test log loss" value={ens.log_loss.toFixed(3)} note={`coin flip ${coin.log_loss.toFixed(3)}; lower is better`} />
          <Tile label="Matches remaining" value={data.fixtures.length + 15} note={`${data.fixtures.length} pool + 12 Super Eights + 3 knockouts`} />
        </div>
      </section>

      <section className="card">
        <h3>Built from {data.meta.sources.length} independent sources</h3>
        <div className="grid g3" style={{ marginTop: 10 }}>
          {data.meta.sources.map((s) => (
            <div key={s.name}>
              <div style={{ fontWeight: 600 }}><a href={s.url} target="_blank" rel="noreferrer">{s.name}</a></div>
              <div className="small muted">{s.records} · {s.provides}</div>
            </div>
          ))}
          <div>
            <div style={{ fontWeight: 600 }}>Cross-checked</div>
            <div className="small muted">
              Winners agree in {agree.ok} of {agree.total} matches covered by more than one source. Standings rebuilt from
              results match ESPNcricinfo for {data.validation.standings.agree_with_espn_points} of 16 teams.
            </div>
          </div>
        </div>
      </section>

      <section className="grid g2">
        <div className="card">
          <h2>Title odds</h2>
          <p className="sub small">Share of {data.meta.n_sims.toLocaleString()} simulated tournaments each team won. Hover for each stage.</p>
          <HBarChart
            data={data.teams.map((t) => ({ key: t.code, label: t.name, value: t.sim.champion, t }))}
            valueFormat={(v) => pct(v, v < 0.1 && v > 0 ? 1 : 0)}
            highlight={[fav.code]}
            height={20}
            tooltip={(d) => (
              <TipRows title={d.t.name} rows={[
                ["Super Eights", pct(d.t.sim.super8)], ["Semi-final", pct(d.t.sim.semi)],
                ["Final", pct(d.t.sim.final)], ["Champion", pct(d.t.sim.champion)],
              ]} />
            )}
          />
        </div>
        <div className="card">
          <h2>Path to the title</h2>
          <p className="sub small">Probability of reaching each stage. Stronger shading means more likely.</p>
          <StageTable />
        </div>
      </section>

      <section>
        <h2>Standings and projections</h2>
        <p className="sub">
          Table rebuilt from every source's results as of {data.meta.as_of} (it matches ESPNcricinfo's official table), with the
          simulated average final pool points and the chance of finishing top four, which qualifies for the Super Eights.
        </p>
        <div className="grid g2">
          {["Pool A", "Pool B"].map((p) => <PoolTable key={p} pool={p} />)}
        </div>
      </section>

      <section className="callout">
        <b>How to read this.</b> Single T20 matches are close to coin flips. Even the best model here calls only about{" "}
        {pct(ens.accuracy, 0)} of held-out matches correctly. The model is useful for <i>relative</i> strength, not for
        certainty. See <a href="#" onClick={(e) => { e.preventDefault(); go("Models"); }}>Models</a> for the evidence and{" "}
        <a href="#" onClick={(e) => { e.preventDefault(); go("Data & Method"); }}>Data &amp; Method</a> for the assumptions.
      </section>
    </>
  );
}

function Tile({ label, value, note }) {
  return (
    <div className="card tile">
      <div className="label">{label}</div>
      <div className="value num">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

function heat(v) {
  const step = v <= 0.005 ? 0 : Math.min(6, 1 + Math.floor(v * 6.5));
  return { background: `var(--seq-${step})`, color: step >= 4 ? "var(--seq-ink-light)" : "var(--seq-ink-dark)" };
}

function StageTable() {
  const cols = [["top_of_pool", "Top pool"], ["super8", "Super 8"], ["semi", "Semi"], ["final", "Final"], ["champion", "Win"]];
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr><th>Team</th>{cols.map(([k, l]) => <th key={k} className="c">{l}</th>)}</tr>
        </thead>
        <tbody>
          {data.teams.map((t) => (
            <tr key={t.code}>
              <td style={{ whiteSpace: "nowrap" }}><span className="team-badge">{t.code}</span></td>
              {cols.map(([k]) => (
                <td key={k} style={{ padding: 3 }}>
                  <div className="heat" style={heat(t.sim[k])}>{(t.sim[k] * 100).toFixed(t.sim[k] < 0.1 ? 1 : 0)}</div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 8 }}>Values are percentages.</p>
    </div>
  );
}

function PoolTable({ pool }) {
  const rows = data.teams.filter((t) => t.pool === pool)
    .sort((a, b) => b.table.points - a.table.points || b.table.nrr - a.table.nrr);
  return (
    <div className="card">
      <h3>{pool}</h3>
      <div className="table-scroll">
        <table>
          <thead>
            <tr><th>Team</th><th className="r">P</th><th className="r">W</th><th className="r">L</th><th className="r">NR</th>
              <th className="r">NRR</th><th className="r">Pts</th><th className="r">Proj. pts</th><th className="r">Top 4</th></tr>
          </thead>
          <tbody>
            {rows.map((t, i) => (
              <tr key={t.code} className={i < 4 ? "qual" : ""}>
                <td><span className="team-chip"><span className="team-badge">{t.code}</span><span className="small" style={{ fontWeight: 500 }}>{t.name}</span></span></td>
                <td className="r">{t.table.played}</td><td className="r">{t.table.won}</td><td className="r">{t.table.lost}</td>
                <td className="r">{t.table.nr}</td><td className="r">{t.table.nrr.toFixed(2)}</td>
                <td className="r"><b>{t.table.points}</b></td>
                <td className="r">{t.sim.exp_pool_points.toFixed(1)}</td>
                <td className="r">{pct(t.sim.super8, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ Fixtures */
function Fixtures() {
  const [pool, setPool] = useState("All");
  const list = data.fixtures.filter((f) => pool === "All" || f.stage === pool);
  const byDate = useMemo(() => {
    const m = new Map();
    list.forEach((f) => { const d = f.date.slice(0, 10); m.set(d, [...(m.get(d) || []), f]); });
    return [...m.entries()];
  }, [list]);
  return (
    <>
      <section>
        <h2>Upcoming fixtures</h2>
        <p className="sub">
          Win probability for every remaining pool match from the ensemble (weighted blend of the Elo, logistic regression,
          random forest and gradient boosting models). Washout risk comes from the Open-Meteo forecast for the venue, with
          seasonal averages beyond 16 days. Open a match to see each model's estimate and the factors driving it.
        </p>
        <div className="team-picker">
          {["All", "Pool A", "Pool B"].map((p) => (
            <button key={p} aria-pressed={pool === p} onClick={() => setPool(p)}>{p}</button>
          ))}
        </div>
        <div className="legend">
          <span><i style={{ background: "var(--series-1)" }} />First-named team</span>
          <span><i style={{ background: "var(--series-2)" }} />Second-named team</span>
        </div>
        <div className="grid g2">
          {byDate.map(([d, fx]) => (
            <div className="card" key={d} style={{ paddingTop: 10, paddingBottom: 10 }}>
              {fx.map((f) => <FixtureRow key={f.match_id} f={f} />)}
            </div>
          ))}
        </div>
      </section>
      <section>
        <h2>2026 results so far</h2>
        <div className="card table-scroll">
          <table>
            <thead><tr><th>Date</th><th>Pool</th><th>Match</th><th>Venue</th><th>Result</th></tr></thead>
            <tbody>
              {data.results_2026.map((r, i) => (
                <tr key={i}>
                  <td style={{ whiteSpace: "nowrap" }}>{fmtDate(r.date)}</td><td>{r.stage}</td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <b>{r.team1}</b> <span className="muted small">{r.score1}</span> v <b>{r.team2}</b> <span className="muted small">{r.score2}</span>
                  </td>
                  <td className="small">{r.venue}</td>
                  <td className="small">{r.status}{r.result_source === "espn" && <span className="pill warn" style={{ marginLeft: 6 }}>from ESPN</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function FixtureRow({ f }) {
  const pa = f.p_a, pb = 1 - f.p_a;
  return (
    <div className="fixture">
      <div className="when">
        <div style={{ color: "var(--text)", fontWeight: 600 }}>{fmtDate(f.date)}</div>
        <div>{fmtTime(f.date)}</div>
        <div>{f.stage}</div>
      </div>
      <div>
        <div className="teams">
          <span>{name(f.team_a)} <span className="num" style={{ color: "var(--text-2)" }}>{pct(pa, 0)}</span></span>
          <span><span className="num" style={{ color: "var(--text-2)" }}>{pct(pb, 0)}</span> {name(f.team_b)}</span>
        </div>
        <div className="split" role="img" aria-label={`${name(f.team_a)} ${pct(pa, 0)}, ${name(f.team_b)} ${pct(pb, 0)}`}>
          <div style={{ width: `${pa * 100}%`, background: "var(--series-1)" }} />
          <div style={{ width: `${pb * 100}%`, background: "var(--series-2)" }} />
        </div>
        <div className="small muted" style={{ marginTop: 6, display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <span>{f.venue}</span>
          <Rain f={f} />
        </div>
        <details>
          <summary>Why?</summary>
          <ul className="plain" style={{ marginTop: 8 }}>
            {f.reasons.map((r) => (
              <li key={r.feature}><b>{r.feature}</b> favours {name(r.favours)} ({r.logit > 0 ? "+" : ""}{r.logit.toFixed(2)} log-odds)</li>
            ))}
          </ul>
          <table style={{ marginTop: 6 }}>
            <tbody>
              {Object.entries(f.models).map(([m, p]) => (
                <tr key={m}><td>{m}</td><td className="r">{name(f.team_a)} {pct(p, 0)}</td></tr>
              ))}
            </tbody>
          </table>
        </details>
      </div>
    </div>
  );
}

function Rain({ f }) {
  const p = f.p_no_result;
  const cls = p >= 0.2 ? "bad" : p >= 0.08 ? "warn" : "good";
  const detail = f.forecast_mm != null
    ? `${f.forecast_mm} mm forecast${f.forecast_rain_prob != null ? `, ${f.forecast_rain_prob}% chance of rain` : ""}`
    : "seasonal average";
  return <span className={`pill ${cls}`} title={f.weather_source}>☂ {pct(p, 0)} washout risk · {detail}</span>;
}

/* --------------------------------------------------------------------- Teams */
function Teams() {
  const [sel, setSel] = useState(data.teams[0].code);
  const t = TEAMS[sel];
  return (
    <>
      <section>
        <h2>Team profiles</h2>
        <p className="sub">Ratings, recent form, projected lineup strength and player availability for each side.</p>
        <div className="team-picker">
          {[...data.teams].sort((a, b) => a.name.localeCompare(b.name)).map((x) => (
            <button key={x.code} aria-pressed={sel === x.code} onClick={() => setSel(x.code)}>{x.name}</button>
          ))}
        </div>
        <div className="grid g4">
          <Tile label="Title chance" value={pct(t.sim.champion)} note={`Super Eights ${pct(t.sim.super8, 0)} · Final ${pct(t.sim.final, 0)}`} />
          <Tile label="Elo rating" value={t.elo} note={`${ord(rankOf(t.code, (x) => x.elo))} of 16 · league start 1500`} />
          <Tile label={`Projected XI (next match, ${t.next_match.slice(5)})`} value={`${t.strength.bat} / ${t.strength.bowl}`}
            note={`bat / bowl · Super Eights: ${t.strength.bat_super8} / ${t.strength.bowl_super8}`} />
          <Tile label="Recent form (oldest → latest)" value={t.form.last5 || "–"}
            note={`last 5 across seasons · weighted win rate ${pct(t.form.win_rate, 0)} · RR margin ${t.form.rr_margin > 0 ? "+" : ""}${t.form.rr_margin}`} />
        </div>
      </section>
      <section className="grid g2">
        <div className="card">
          <h3>Elo rating history</h3>
          <p className="sub small">Rating after each of the team's last {t.elo_history.length} matches (all competitions). The line at 1500 is the franchise starting rating.</p>
          <LineChart
            points={t.elo_history.map((h) => ({ y: h.elo, label: h.date, group: h.season }))}
            yLabel="Elo" reference={1500} refLabel="1500" format={(v) => v.toFixed(0)}
          />
        </div>
        <div className="card">
          <h3>Lineup strength, all teams</h3>
          <p className="sub small">Projected batting and bowling strength for the next match. Up and right is better. Click a dot to switch team.</p>
          <TeamScatter
            selected={sel} onSelect={setSel}
            teams={data.teams.map((x) => ({
              key: x.code, x: x.strength.bat, y: x.strength.bowl, pool: x.pool,
              tip: <TipRows title={x.name} rows={[["Batting", x.strength.bat], ["Bowling", x.strength.bowl], ["Title", pct(x.sim.champion)]]} />,
            }))}
          />
        </div>
      </section>
      <section className="grid g2">
        <div className="card">
          <h3>Key players and availability</h3>
          <p className="sub small">
            Top-rated players whose latest team is {t.name}. "Next" is the chance they play the next match;
            "S8" is for the Super Eights (from 26 Oct).
          </p>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Player</th><th className="r">Bat</th><th className="r">Bowl</th><th className="r">Next</th><th className="r">S8</th><th>Status</th></tr></thead>
              <tbody>
                {t.key_players.map((p) => (
                  <tr key={p.player}>
                    <td><b style={{ fontWeight: 600 }}>{p.player}</b></td>
                    <td className="r">{p.bat}</td><td className="r">{p.bowl}</td>
                    <td className="r"><Avail p={p.p_next} /></td>
                    <td className="r"><Avail p={p.p_super8} /></td>
                    <td className="small muted">{p.note_super8 || p.note || p.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small" style={{ marginTop: 8 }}>
            Bat = runs per innings plus half the runs scored above a 125 strike rate. Bowl = runs saved per match against
            league economy ({data.meta.league_economy}/over), counting a wicket as 22 runs. Both are shrunk toward replacement level
            and decay with a one-year half-life.
          </p>
        </div>
        <div className="card">
          <h3>Venue record</h3>
          <p className="sub small">Home ground: <b>{t.home}</b>. Wins and losses at the grounds this team plays most (all seasons in the data).</p>
          <table>
            <thead><tr><th>Venue</th><th className="r">W</th><th className="r">L</th><th className="r">Win %</th></tr></thead>
            <tbody>
              {t.venue_record.map((v) => (
                <tr key={v.venue}><td>{v.venue}</td><td className="r">{v.won}</td><td className="r">{v.lost}</td>
                  <td className="r">{pct(v.won / Math.max(v.won + v.lost, 1), 0)}</td></tr>
              ))}
            </tbody>
          </table>
          <h3 style={{ marginTop: 18 }}>Current season</h3>
          <p className="small muted" style={{ margin: 0 }}>
            {t.pool}: {t.table.played} played, {t.table.won} won, {t.table.lost} lost, {t.table.nr} no result,{" "}
            {t.table.points} pts, NRR {t.table.nrr.toFixed(3)}. Projected pool finish: {t.sim.exp_pool_rank.toFixed(1)} on average.
          </p>
        </div>
      </section>
      <section className="grid g2">
        <div className="card">
          <h3>Official 2026 squad</h3>
          <p className="sub small">
            From CricTracker. Ratings come from all three scorecard sources; "–" means no CSA T20 history yet, so the player
            counts at replacement level.
          </p>
          {t.squad.length ? (
            <div className="table-scroll">
              <table>
                <thead><tr><th>Player</th><th>Role</th><th className="r">Bat</th><th className="r">Bowl</th><th className="c">Played 2026</th></tr></thead>
                <tbody>
                  {[...t.squad].sort((a, b) => (b.bat ?? 0) + Math.max(b.bowl ?? 0, 0) - (a.bat ?? 0) - Math.max(a.bowl ?? 0, 0)).map((p) => (
                    <tr key={p.player}>
                      <td>{p.player}</td><td className="small muted">{p.role}</td>
                      <td className="r">{p.bat ?? "–"}</td><td className="r">{p.bowl ?? "–"}</td>
                      <td className="c">{p.played_2026 ? "✓" : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p className="muted small">No squad published for this team yet.</p>}
        </div>
        <div className="card">
          <h3>Phase profile (ball-by-ball)</h3>
          <p className="sub small">
            Run rates by phase from Cricsheet ball-by-ball data, 2021-22 to 2024-25. Batting above the league rate is good;
            bowling below it is good.
          </p>
          {t.phases ? (
            <>
              <table>
                <thead><tr><th>Phase</th><th className="r">Batting RR</th><th className="r">Bowling RR</th><th className="r">League RR</th></tr></thead>
                <tbody>
                  {[["powerplay", "Powerplay (1-6)"], ["middle", "Middle (7-15)"], ["death", "Death (16-20)"]].map(([k, l]) => {
                    const ph = t.phases[k];
                    return (
                      <tr key={k}><td>{l}</td>
                        <td className="r"><Delta v={ph.bat_rr} base={ph.league_rr} higherIsBetter /></td>
                        <td className="r"><Delta v={ph.bowl_rr} base={ph.league_rr} /></td>
                        <td className="r muted">{ph.league_rr}</td></tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="small muted" style={{ marginTop: 8 }}>Based on {t.phases.innings} innings. Context only; not a model input.</p>
            </>
          ) : <p className="muted small">No ball-by-ball data: this side wasn't in the top-flight T20 Challenge up to 2024-25.</p>}
        </div>
      </section>
    </>
  );
}

function Delta({ v, base, higherIsBetter }) {
  if (v == null) return "–";
  const good = higherIsBetter ? v > base : v < base;
  return <span className={`pill ${good ? "good" : "bad"}`}>{good ? "▲" : "▼"} {v.toFixed(2)}</span>;
}

function Avail({ p }) {
  const cls = p >= 0.7 ? "good" : p >= 0.3 ? "warn" : "bad";
  const icon = p >= 0.7 ? "●" : p >= 0.3 ? "◐" : "○";
  return <span className={`pill ${cls}`}>{icon} {Math.round(p * 100)}%</span>;
}

/* -------------------------------------------------------------------- Models */
function Models() {
  const M = data.models;
  const order = ["Ensemble", "Elo", "Logistic Regression", "Random Forest", "Gradient Boosting"];
  const coin = M.metrics["Coin flip"].log_loss;
  const lls = order.map((k) => M.metrics[k].log_loss);
  const lo = Math.floor(Math.min(...lls, coin) * 100 - 0.5) / 100, hi = Math.ceil(Math.max(...lls, coin) * 100 + 0.5) / 100;
  const used = M.features.filter((f) => f.used).map((f) => f.label);
  const seasons = M.test_seasons;
  return (
    <>
      <section>
        <h2>The models</h2>
        <p className="sub">
          Four models each estimate P(team A beats team B) from the same pre-match features. The ensemble averages them using
          weights based on back-test skill. Every model is trained only on data from before the season it is tested on.
        </p>
        <div className="grid g4">
          <ModelCard title="Elo rating" body="Ratings update after every match, scaled by the winning margin, with 35 points of home advantage and a 25% pull back to each team's starting level every new season. Run over every match since 2011-12 (Cricsheet + Cricbuzz + ESPN)." />
          <ModelCard title="Logistic regression" body="A linear model on standardised features (L2 regularised, C = 0.1). Easy to read: each coefficient is the change in log-odds per standard deviation of a feature." />
          <ModelCard title="Random forest" body={`600 shallow trees (depth 3, at least 15 matches per leaf) to catch non-linear effects without memorising ${data.meta.n_train} matches.`} />
          <ModelCard title="Gradient boosting" body="100 boosted decision stumps with a 0.03 learning rate and 70% subsampling. Additive and heavily damped." />
        </div>
      </section>

      <section className="grid g2">
        <div className="card">
          <h3>Out-of-sample log loss</h3>
          <p className="sub small">Seasons {seasons.join(", ")} predicted with models trained on earlier seasons only. Lower is better; the vertical line is a 50/50 guess.</p>
          <DotPlot
            data={order.map((k) => ({ key: k, label: k, value: M.metrics[k].log_loss, strong: k === "Ensemble", m: M.metrics[k] }))}
            domain={[lo, hi]} reference={coin} refLabel="coin flip"
            tooltip={(d) => <TipRows title={d.label} rows={[["Log loss", d.m.log_loss.toFixed(4)], ["Brier", d.m.brier.toFixed(4)], ["Accuracy", pct(d.m.accuracy)], ["AUC", d.m.auc.toFixed(3)]]} />}
          />
        </div>
        <div className="card">
          <h3>Back-test scorecard</h3>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Model</th><th className="r">Log loss</th><th className="r">Brier</th><th className="r">Accuracy</th><th className="r">AUC</th><th className="r">Weight</th></tr></thead>
              <tbody>
                {[...order, "Coin flip"].map((k) => (
                  <tr key={k}>
                    <td><b style={{ fontWeight: k === "Ensemble" ? 700 : 500 }}>{k}</b></td>
                    <td className="r">{M.metrics[k].log_loss.toFixed(3)}</td>
                    <td className="r">{M.metrics[k].brier.toFixed(3)}</td>
                    <td className="r">{pct(M.metrics[k].accuracy)}</td>
                    <td className="r">{M.metrics[k].auc.toFixed(3)}</td>
                    <td className="r">{M.weights[k] != null ? pct(M.weights[k], 0) : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>
            Weights are proportional to each model's log-loss improvement over a coin flip. The gaps between models are
            small, so the blend is mainly a hedge: no single model is clearly best on this much data.
          </p>
          <h3 style={{ marginTop: 14 }}>Accuracy by season</h3>
          <div className="table-scroll">
          <table>
            <thead><tr><th>Model</th>{seasons.map((s) => <th key={s} className="r">{s}</th>)}</tr></thead>
            <tbody>
              {["Elo", "Logistic Regression", "Random Forest", "Gradient Boosting"].map((m) => (
                <tr key={m}><td>{m}</td>
                  {seasons.map((s) => {
                    const f = M.folds.find((x) => x.model === m && x.season === s);
                    return <td key={s} className="r">{f ? pct(f.accuracy, 0) : "–"}</td>;
                  })}
                </tr>
              ))}
              <tr><td className="muted">Matches</td>{seasons.map((s) => <td key={s} className="r muted">{M.folds.find((x) => x.season === s)?.n}</td>)}</tr>
            </tbody>
          </table>
          </div>
        </div>
      </section>

      <section className="grid g2">
        <div className="card">
          <h3>What drives the predictions</h3>
          <p className="sub small">Normalised feature importance in each fitted model (logistic regression uses |coefficient| on standardised inputs).</p>
          <GroupedBars
            categories={used}
            series={["Logistic Regression", "Random Forest", "Gradient Boosting"].map((m) => ({ name: m, values: M.importance[m] }))}
            format={(v) => pct(v, 0)}
          />
          <div className="prose small" style={{ marginTop: 10 }}>
            <p>
              Logistic regression coefficients (log-odds per standard deviation):{" "}
              {Object.entries(M.lr_coefficients).map(([k, v], i) => (
                <span key={k}>{i ? "; " : ""}{k} <b>{v > 0 ? "+" : ""}{v.toFixed(2)}</b></span>
              ))}. All four point the expected way: the stronger, higher-rated, home team is favoured.
            </p>
          </div>
        </div>
        <div className="card">
          <h3>Feature selection (ablation)</h3>
          <p className="sub small">
            Eight features were built. Each set below was back-tested with the same logistic regression. The differences are
            tiny (a few thousandths of log loss, within noise). Recent form and run-rate margin never helped, so the compact
            four-feature set is kept for stability, and the rest are shown as context.
          </p>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Feature set</th><th className="r">Log loss</th><th className="r">Acc.</th><th className="r">AUC</th></tr></thead>
              <tbody>
                {M.ablation.map((a) => (
                  <tr key={a.features}>
                    <td style={{ fontWeight: a.features.includes("chosen") ? 650 : 400 }}>{a.features}</td>
                    <td className="r">{a.log_loss.toFixed(3)}</td><td className="r">{pct(a.accuracy, 0)}</td><td className="r">{a.auc.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>
            Elo is kept even where lineup features score similarly, because it is the only signal with long history for the
            provincial sides and for SA Emerging Players, whose scorecard data is thin.
          </p>
          <h3 style={{ marginTop: 14 }}>Single-feature AUC</h3>
          <p className="sub small">How well each feature on its own ranks winners over all {data.meta.n_train} training matches (0.5 = no signal).</p>
          <table>
            <tbody>
              {Object.entries(M.feature_auc).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                <tr key={k}><td>{k}{M.features.find((f) => f.label === k)?.used ? "" : <span className="muted small"> (context only)</span>}</td><td className="r">{v.toFixed(3)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card prose">
        <h3>From match probabilities to a champion</h3>
        <p>
          The ensemble gives a win probability for every remaining pool fixture (with home advantage) and for every possible
          pairing on later dates at a neutral venue. Projected lineups are recalculated for each date, so a team that loses
          players to the Test series is weaker in the Super Eights than in early October.
        </p>
        <p>
          Each of the <strong>{data.meta.n_sims.toLocaleString()}</strong> simulations then plays out the rest of the event.
          Each match has its own washout risk from the rain forecast or venue climatology (2 points each; Super Eights average{" "}
          <strong>{pct(data.meta.p_no_result_s8)}</strong>), and winners take 4 points
          plus a bonus point in <strong>{pct(data.meta.p_bonus_given_win, 0)}</strong> of wins (cross-checked against points ESPN
          actually awarded: {pct(data.meta.p_bonus_observed, 0)}). Net run rate updates from margins
          resampled from real matches. The top four in each pool go to two Super Eights groups, then the top two in each group
          go to the semi-finals and final. Each simulation also nudges every team's strength by a random amount (sd{" "}
          {data.meta.strength_uncertainty} log-odds), so the uncertainty in the ratings themselves feeds through to the title odds.
        </p>
      </section>
    </>
  );
}

function ModelCard({ title, body }) {
  return (
    <div className="card">
      <h3>{title}</h3>
      <p className="small muted" style={{ margin: 0 }}>{body}</p>
    </div>
  );
}

/* -------------------------------------------------------------------- Method */
function Method() {
  const m = data.meta;
  const st = V.standings;
  const cw = V.player_crosswalk;
  const tz = data.toss;
  return (
    <>
      <section className="card">
        <h2>Data sources</h2>
        <p className="sub small">
          Every source is scraped by its own script into <code>data/</code>. <code>build_dataset.py</code> then merges them into{" "}
          <code>data/unified/</code> and cross-checks them. ESPNcricinfo's website blocks scripts, but ESPN's public API serves
          the same Cricinfo database, so that is how its data comes in.
        </p>
        <div className="table-scroll">
          <table>
            <thead><tr><th>Source</th><th>Script</th><th>What it provides</th><th className="r">Volume</th></tr></thead>
            <tbody>
              {m.sources.map((s, i) => (
                <tr key={s.name}>
                  <td style={{ whiteSpace: "nowrap" }}><a href={s.url} target="_blank" rel="noreferrer">{s.name}</a></td>
                  <td><code>{["scrape_cricbuzz.py", "scrape_cricsheet.py", "scrape_espn.py", "scrape_squads.py", "scrape_weather.py"][i]}</code></td>
                  <td className="small">{s.provides}</td>
                  <td className="r small" style={{ whiteSpace: "nowrap" }}>{s.records}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="grid g2">
        <div className="card">
          <h2>Cross-source validation</h2>
          <table style={{ marginTop: 10 }}>
            <tbody>
              <tr><td>Cricbuzz v Cricsheet: same winner</td><td className="r"><b>{V.cricbuzz_vs_cricsheet.winner_agree}</b> / {V.cricbuzz_vs_cricsheet.winner_agree + (V.cricbuzz_vs_cricsheet.winner_disagree || 0)}</td></tr>
              <tr><td>Cricbuzz v Cricsheet: same first-innings score</td><td className="r"><b>{V.cricbuzz_vs_cricsheet.score_agree}</b> / {V.cricbuzz_vs_cricsheet.score_agree + (V.cricbuzz_vs_cricsheet.score_disagree || 0)}</td></tr>
              <tr><td>Cricbuzz v ESPN: same winner</td><td className="r"><b>{V.cricbuzz_vs_espn.winner_agree}</b> / {V.cricbuzz_vs_espn.winner_agree + (V.cricbuzz_vs_espn.winner_disagree || 0)}</td></tr>
              <tr><td>Rebuilt standings v ESPNcricinfo table (points)</td><td className="r"><b>{st.agree_with_espn_points}</b> / {st.teams}</td></tr>
              <tr><td>Rebuilt standings v ESPNcricinfo table (NRR)</td><td className="r"><b>{st.agree_with_espn_nrr}</b> / {st.teams}</td></tr>
              <tr><td>Rebuilt standings v Cricbuzz table (points)</td><td className="r"><b>{st.agree_with_cricbuzz_points}</b> / {st.teams}</td></tr>
              <tr><td>Matches only in Cricsheet (added history)</td><td className="r"><b>{V.gaps_filled.match_only_in_cricsheet}</b></td></tr>
              <tr><td>2026 results filled from ESPN before Cricbuzz</td><td className="r"><b>{V.gaps_filled.result_from_espn || 0}</b></td></tr>
            </tbody>
          </table>
          <div className="callout" style={{ marginTop: 14 }}>
            <b>Discrepancy found.</b> Cricbuzz's table was behind on {st.cricbuzz_behind.join(", ") || "no teams"}, and it shows{" "}
            {Object.entries(st.cricbuzz_points_differ).map(([t, v]) => `${name(t)} on ${v.cricbuzz} (results and ESPNcricinfo: ${v.espn})`).join("; ")}.
            The model uses the standings rebuilt from match results, which match ESPNcricinfo exactly.
          </div>
        </div>
        <div className="card">
          <h2>Player identity crosswalk</h2>
          <p className="sub small">
            Each source uses its own player ids. Cricsheet's register links its ids to ESPNcricinfo ids, and Cricbuzz names are
            matched to ESPNcricinfo profiles, so a player's whole career is rated as one person.
          </p>
          <table>
            <tbody>
              <tr><td>Cricbuzz players matched exactly by name</td><td className="r"><b>{cw.exact}</b></td></tr>
              <tr><td>Matched by initial + surname</td><td className="r"><b>{cw["initial+surname"]}</b></td></tr>
              <tr><td>No ESPNcricinfo profile (kept with a Cricbuzz id)</td><td className="r"><b>{cw.unmatched}</b></td></tr>
              <tr><td>Unified players</td><td className="r"><b>{V.source_counts.players}</b></td></tr>
              <tr><td>Official squad players linked to history</td><td className="r"><b>{V.squads.matched}</b> / {V.squads.players}</td></tr>
            </tbody>
          </table>
          <h3 style={{ marginTop: 16 }}>Toss (all sources, {tz.matches} matches)</h3>
          <table>
            <tbody>
              <tr><td>Toss winner won the match</td><td className="r"><b>{pct(tz.toss_winner_won)}</b></td></tr>
              <tr><td>Toss winners who chose to field</td><td className="r"><b>{pct(tz.chose_field)}</b></td></tr>
              <tr><td>Side batting second won</td><td className="r"><b>{pct(tz.chasing_side_won)}</b></td></tr>
            </tbody>
          </table>
          <p className="small muted" style={{ marginTop: 8 }}>
            Winning the toss gives no edge in this competition (it is slightly below 50%), and the toss isn't known before
            the match, so it is not a model feature.
          </p>
        </div>
      </section>

      <section className="grid g2">
        <div className="card">
          <h2>Pipeline</h2>
          <ol className="steps" style={{ marginTop: 12 }}>
            <li><b>Scrape</b> five sources: Cricbuzz, Cricsheet ball-by-ball, the ESPN/ESPNcricinfo API, CricTracker squads,
              and Open-Meteo weather.</li>
            <li><b>Merge and validate</b> (<code>build_dataset.py</code>): join matches across sources by date and teams, link
              player ids, fill gaps, rebuild the standings and report agreement.</li>
            <li><b>Features</b> (<code>model.py</code>): Elo since 2011-12, form, run-rate margin, player-rated XI strength
              (from {m.n_batting_rows.toLocaleString()} batting and {m.n_bowling_rows.toLocaleString()} bowling lines), home ground,
              head-to-head and venue record. Each is computed using only data from before the match.</li>
            <li><b>Train and back-test</b> four models season by season ({data.models.test_seasons[0]} → {data.models.test_seasons.at(-1)}),
              then blend them by skill.</li>
            <li><b>Washout model</b>: logistic regression of abandonment on venue rainfall ({m.washout_model.n} matches,{" "}
              {m.washout_model.no_results} no-results), applied to the forecast or to 2011-2025 climatology.</li>
            <li><b>Simulate</b> the rest of the tournament {m.n_sims.toLocaleString()} times.</li>
          </ol>
        </div>
        <div className="card">
          <h2>Washout risk by venue</h2>
          <p className="sub small">Modelled chance of a no-result from 15 years of daily rainfall at each ground.</p>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Venue</th><th>Home</th><th className="r">Mid-Oct</th><th className="r">Early Nov</th></tr></thead>
              <tbody>
                {[...data.venue_rain].sort((a, b) => b.oct_washout - a.oct_washout).map((v) => (
                  <tr key={v.key}><td>{v.venue} <span className="muted small">{v.city}</span></td><td>{v.home_team || <span className="muted">–</span>}</td>
                    <td className="r">{pct(v.oct_washout)}</td><td className="r">{pct(v.nov_washout)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="card">
        <h2>Assumptions</h2>
        <ul className="plain" style={{ marginTop: 10 }}>
          {data.assumptions.map((a) => <li key={a}>{a}</li>)}
        </ul>
      </section>

      <section className="card">
        <h2>Player availability inputs</h2>
        <p className="sub small">{data.availability.notes}</p>
        <div className="grid g2">
          {data.availability.groups.map((g) => (
            <div key={g.label}>
              <h3>{g.label} <span className={`pill ${g.confidence === "reported" ? "good" : "warn"}`}>{g.confidence}</span></h3>
              <p className="small" style={{ margin: "4px 0 6px" }}>{g.players.join(", ")}</p>
              <table>
                <tbody>
                  {g.windows.map((w) => (
                    <tr key={w.from}><td className="small" style={{ whiteSpace: "nowrap" }}>{w.from.slice(5)} → {w.to.slice(5)}</td>
                      <td className="r"><Avail p={w.p} /></td><td className="small muted">{w.reason}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      </section>

      <section className="grid g2">
        <div className="card">
          <h2>Venue statistics</h2>
          <p className="sub small">Completed matches since 2018-19 where the side batting first is known. Toss columns use Cricsheet and ESPN toss data.</p>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Venue</th><th className="r">Matches</th><th className="r">Avg 1st inns</th><th className="r">Chase win</th><th className="r">Toss winner won</th><th className="r">Chose field</th></tr></thead>
              <tbody>
                {data.venues.map((v) => (
                  <tr key={v.venue}><td>{v.venue}{v.home_team && <span className="muted small"> · {v.home_team}</span>}</td>
                    <td className="r">{v.matches}</td><td className="r">{v.avg_first_innings}</td><td className="r">{v.chase_win_pct}%</td>
                    <td className="r">{v.toss_winner_win_pct != null ? `${v.toss_winner_win_pct}%` : "–"}</td>
                    <td className="r">{v.field_first_pct != null ? `${v.field_first_pct}%` : "–"}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <h2>Past champions</h2>
          <div className="table-scroll">
            <table style={{ marginTop: 10 }}>
              <thead><tr><th>Season</th><th>Competition</th><th>Winner</th><th>Runner-up</th></tr></thead>
              <tbody>
                {data.champions.map((c) => (
                  <tr key={c.season + c.competition}><td>{c.season}</td><td className="small">{c.competition}</td>
                    <td><b>{name(c.winner)}</b></td><td>{name(c.runner_up)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <h2 style={{ marginTop: 20 }}>Refresh</h2>
          <p className="small muted">Re-scrape every source, rebuild and re-run after each match day:</p>
          <pre className="small" style={{ background: "var(--surface-2)", padding: 10, borderRadius: 8, overflowX: "auto" }}>
{`.venv/bin/python scripts/refresh_all.py
cd web && npm run dev`}
          </pre>
          <h3 style={{ marginTop: 14 }}>News references</h3>
          <ul className="plain small">
            {m.references.map((s) => <li key={s.url}><a href={s.url} target="_blank" rel="noreferrer">{s.name}</a></li>)}
          </ul>
        </div>
      </section>
    </>
  );
}
