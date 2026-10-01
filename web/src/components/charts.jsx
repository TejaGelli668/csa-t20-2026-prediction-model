import React, { useLayoutEffect, useRef, useState } from "react";

/* Small hand-rolled SVG charts. Each one keeps a hover tooltip, thin marks,
   rounded data ends and hairline grids, and pulls colour from CSS tokens. */

/* Tooltip state plus the container's pixel width, so SVGs draw at 1:1 and
   text never shrinks with the viewport. */
function useTip(fallback = 640) {
  const ref = useRef(null);
  const [tip, setTip] = useState(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  const show = (evt, content) => {
    const box = ref.current.getBoundingClientRect();
    setTip({ x: evt.clientX - box.left, y: evt.clientY - box.top, content });
  };
  const hide = () => setTip(null);
  const node = tip && (
    <div className="tooltip" style={{ left: tip.x, top: tip.y }}>
      {tip.content}
    </div>
  );
  return { ref, show, hide, node, width };
}

export function TipRows({ title, rows }) {
  return (
    <>
      <div className="t">{title}</div>
      {rows.map(([k, v]) => (
        <div className="row" key={k}>
          <span>{k}</span>
          <b>{v}</b>
        </div>
      ))}
    </>
  );
}

function niceTicks(max, count = 4) {
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

/* Horizontal bar chart, one series. `highlight` keys get the accent fill, the
   rest use the same hue at lower emphasis. */
export function HBarChart({ data, valueFormat = (v) => v, max, highlight = [], tooltip, height = 26 }) {
  const t = useTip();
  const W = t.width, L = Math.min(150, W * 0.32), R = 44, gap = 8;
  const H = data.length * (height + gap) + 24;
  const vmax = max ?? Math.max(...data.map((d) => d.value)) * 1.05;
  const ticks = niceTicks(vmax);
  const x = (v) => L + (v / ticks[ticks.length - 1]) * (W - L - R);
  return (
    <div className="chart" ref={t.ref} onMouseLeave={t.hide}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="bar chart">
        {ticks.map((v) => (
          <g key={v}>
            <line className="gridline" x1={x(v)} x2={x(v)} y1={0} y2={H - 20} />
            <text x={x(v)} y={H - 4} textAnchor="middle" fontSize="11">{valueFormat(v)}</text>
          </g>
        ))}
        {data.map((d, i) => {
          const y = i * (height + gap);
          const w = Math.max(x(d.value) - L, 2);
          const hi = highlight.includes(d.key);
          return (
            <g key={d.key}
              onMouseMove={(e) => tooltip && t.show(e, tooltip(d))}
              style={{ cursor: tooltip ? "default" : undefined }}>
              <rect x={0} y={y} width={W} height={height} fill="transparent" />
              <text x={L - 10} y={y + height / 2 + 4} textAnchor="end" style={{ fill: "var(--text)", fontWeight: hi ? 650 : 500 }}>
                {d.label}
              </text>
              {w > 8 ? (
                <path
                  d={`M${L},${y + 3} h${w - 4} a4,4 0 0 1 4,4 v${height - 14} a4,4 0 0 1 -4,4 h-${w - 4} z`}
                  fill="var(--series-1)" opacity={hi || !highlight.length ? 1 : 0.55}
                />
              ) : (
                <rect x={L} y={y + 3} width={w} height={height - 6} rx="1" fill="var(--series-1)" opacity={hi || !highlight.length ? 1 : 0.55} />
              )}
              <text x={L + w + 6} y={y + height / 2 + 4} style={{ fill: "var(--text)", fontWeight: 600 }}>
                {valueFormat(d.value)}
              </text>
            </g>
          );
        })}
      </svg>
      {t.node}
    </div>
  );
}

/* Single-series line chart with crosshair tooltip and optional reference line. */
export function LineChart({ points, yLabel, reference, refLabel, format = (v) => v }) {
  const t = useTip();
  const [hover, setHover] = useState(null);
  if (!points.length) return <p className="muted small">No history.</p>;
  const W = t.width, H = 240, L = 44, R = 16, T = 12, B = 28;
  const ys = points.map((p) => p.y).concat(reference ?? []);
  const lo = Math.floor((Math.min(...ys) - 15) / 25) * 25, hi = Math.ceil((Math.max(...ys) + 15) / 25) * 25;
  const x = (i) => L + (i / Math.max(points.length - 1, 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const ticks = [];
  for (let v = lo; v <= hi; v += (hi - lo) / 4) ticks.push(v);
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.y)}`).join("");
  const seasons = [];
  points.forEach((p, i) => {
    if (i && p.group === points[i - 1].group) return;
    // drop a season label that would collide with the previous one
    if (seasons.length && x(i) - x(seasons.at(-1).i) < 52) seasons.pop();
    seasons.push({ i, g: p.group });
  });
  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(points.length - 1, Math.round(((px - L) / (W - L - R)) * (points.length - 1))));
    setHover(i);
    const p = points[i];
    t.show(e, <TipRows title={p.label} rows={[[yLabel, format(p.y)], ...(p.extra || [])]} />);
  };
  return (
    <div className="chart" ref={t.ref} onMouseLeave={() => { t.hide(); setHover(null); }}>
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} role="img" aria-label={`${yLabel} over time`}>
        {ticks.map((v) => (
          <g key={v}>
            <line className="gridline" x1={L} x2={W - R} y1={y(v)} y2={y(v)} />
            <text x={L - 8} y={y(v) + 4} textAnchor="end" fontSize="11">{Math.round(v)}</text>
          </g>
        ))}
        {seasons.map((s) => (
          <text key={s.i} x={x(s.i)} y={H - 8} fontSize="11">{s.g}</text>
        ))}
        {reference != null && (
          <g>
            <line x1={L} x2={W - R} y1={y(reference)} y2={y(reference)} stroke="var(--text-3)" strokeWidth="1" />
            <text x={W - R} y={y(reference) - 5} textAnchor="end" fontSize="11">{refLabel}</text>
          </g>
        )}
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth="2" strokeLinejoin="round" />
        {hover != null && (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1={T} y2={H - B} stroke="var(--text-3)" strokeWidth="1" />
            <circle cx={x(hover)} cy={y(points[hover].y)} r="5" fill="var(--series-1)" stroke="var(--surface)" strokeWidth="2" />
          </g>
        )}
        <circle cx={x(points.length - 1)} cy={y(points[points.length - 1].y)} r="4" fill="var(--series-1)" stroke="var(--surface)" strokeWidth="2" />
      </svg>
      {t.node}
    </div>
  );
}

/* Dot plot on a zoomed axis (used for log loss, where differences are tiny). */
export function DotPlot({ data, domain, reference, refLabel, format = (v) => v.toFixed(3), tooltip }) {
  const t = useTip();
  const W = t.width, L = Math.min(150, W * 0.36), R = 30, rowH = 32;
  const H = data.length * rowH + 30;
  const [lo, hi] = domain;
  const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  const ticks = [];
  const step = (hi - lo) / 4;
  for (let v = lo; v <= hi + 1e-9; v += step) ticks.push(v);
  return (
    <div className="chart" ref={t.ref} onMouseLeave={t.hide}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="dot plot">
        {ticks.map((v) => (
          <g key={v}>
            <line className="gridline" x1={x(v)} x2={x(v)} y1={0} y2={H - 22} />
            <text x={x(v)} y={H - 6} textAnchor="middle" fontSize="11">{format(v)}</text>
          </g>
        ))}
        {reference != null && (
          <g>
            <line x1={x(reference)} x2={x(reference)} y1={0} y2={H - 22} stroke="var(--text-3)" strokeWidth="1.5" />
            <text x={x(reference) + 5} y={10} fontSize="11">{refLabel}</text>
          </g>
        )}
        {data.map((d, i) => {
          const cy = i * rowH + rowH / 2 + 4;
          return (
            <g key={d.key} onMouseMove={(e) => tooltip && t.show(e, tooltip(d))}>
              <rect x={0} y={cy - rowH / 2} width={W} height={rowH} fill="transparent" />
              <text x={L - 10} y={cy + 4} textAnchor="end" style={{ fill: "var(--text)", fontWeight: d.strong ? 650 : 500 }}>{d.label}</text>
              <line x1={x(Math.min(d.value, reference ?? d.value))} x2={x(Math.max(d.value, reference ?? d.value))} y1={cy} y2={cy}
                stroke="var(--series-1)" strokeWidth="2" opacity="0.35" />
              <circle cx={x(d.value)} cy={cy} r="6" fill="var(--series-1)" stroke="var(--surface)" strokeWidth="2" />
            </g>
          );
        })}
      </svg>
      {t.node}
    </div>
  );
}

/* Grouped horizontal bars: categories x series (max 3 series). */
export function GroupedBars({ categories, series, format = (v) => v }) {
  const t = useTip();
  const W = t.width, L = Math.min(170, W * 0.38), R = 40, barH = 9, inner = 2, groupGap = 16;
  const groupH = series.length * (barH + inner);
  const H = categories.length * (groupH + groupGap) + 24;
  const vmax = Math.max(...series.flatMap((s) => categories.map((c) => s.values[c] ?? 0)));
  const ticks = niceTicks(vmax * 1.05);
  const x = (v) => L + (v / ticks[ticks.length - 1]) * (W - L - R);
  const colors = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
  return (
    <div className="chart" ref={t.ref} onMouseLeave={t.hide}>
      <div className="legend">
        {series.map((s, i) => (
          <span key={s.name}><i style={{ background: colors[i] }} />{s.name}</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="grouped bar chart">
        {ticks.map((v) => (
          <g key={v}>
            <line className="gridline" x1={x(v)} x2={x(v)} y1={0} y2={H - 20} />
            <text x={x(v)} y={H - 4} textAnchor="middle" fontSize="11">{format(v)}</text>
          </g>
        ))}
        {categories.map((c, ci) => {
          const y0 = ci * (groupH + groupGap);
          return (
            <g key={c}
              onMouseMove={(e) => t.show(e, <TipRows title={c} rows={series.map((s) => [s.name, format(s.values[c] ?? 0)])} />)}>
              <rect x={0} y={y0 - 4} width={W} height={groupH + 8} fill="transparent" />
              <text x={L - 10} y={y0 + groupH / 2 + 4} textAnchor="end" style={{ fill: "var(--text)" }}>{c}</text>
              {series.map((s, si) => {
                const w = Math.max(x(s.values[c] ?? 0) - L, 1.5);
                const y = y0 + si * (barH + inner);
                return <rect key={s.name} x={L} y={y} width={w} height={barH} rx="3" fill={colors[si]} />;
              })}
            </g>
          );
        })}
      </svg>
      {t.node}
    </div>
  );
}

/* Scatter of all teams: one hue, pool encoded by shape (filled vs ring). */
export function TeamScatter({ teams, selected, onSelect }) {
  const t = useTip();
  const W = t.width, H = 340, L = 48, R = 20, T = 14, B = 40;
  const xs = teams.map((d) => d.x), ys = teams.map((d) => d.y);
  const pad = (a, b) => [a - (b - a) * 0.08, b + (b - a) * 0.08];
  const [x0, x1] = pad(Math.min(...xs), Math.max(...xs));
  const [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
  const x = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  return (
    <div className="chart" ref={t.ref} onMouseLeave={t.hide}>
      <div className="legend">
        <span><i style={{ background: "var(--series-1)", borderRadius: 99 }} />Pool A</span>
        <span><i style={{ border: "2px solid var(--series-1)", borderRadius: 99, width: 10, height: 10 }} />Pool B</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="batting vs bowling strength">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line className="gridline" x1={L} x2={W - R} y1={T + f * (H - T - B)} y2={T + f * (H - T - B)} />
            <text x={L - 8} y={T + f * (H - T - B) + 4} textAnchor="end" fontSize="11">{Math.round(y1 - f * (y1 - y0))}</text>
            <text x={L + f * (W - L - R)} y={H - 22} textAnchor="middle" fontSize="11">{Math.round(x0 + f * (x1 - x0))}</text>
          </g>
        ))}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" fontSize="11.5">Batting strength (runs-equivalent, top 7) →</text>
        <text x={12} y={(T + H - B) / 2} textAnchor="middle" fontSize="11.5" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>Bowling strength (runs saved, top 5) →</text>
        {(() => {
          // label greedily (selected first) and skip labels that would overlap
          const placed = [];
          const order = [...teams].sort((a, b) => (b.key === selected) - (a.key === selected));
          order.forEach((d) => {
            const lx = x(d.x) + 9, ly = y(d.y) - 7, w = d.key.length * 7 + 4;
            if (!placed.some((p) => Math.abs(p.lx - lx) < Math.max(p.w, w) && Math.abs(p.ly - ly) < 13)) placed.push({ key: d.key, lx, ly, w });
          });
          teams.forEach((d) => { d._label = placed.some((p) => p.key === d.key); });
        })()}
        {teams.map((d) => {
          const sel = d.key === selected;
          return (
            <g key={d.key} style={{ cursor: "pointer" }} onClick={() => onSelect?.(d.key)}
              onMouseMove={(e) => t.show(e, d.tip)}>
              <circle cx={x(d.x)} cy={y(d.y)} r="14" fill="transparent" />
              <circle cx={x(d.x)} cy={y(d.y)} r={sel ? 7 : 5.5}
                fill={d.pool === "Pool A" ? "var(--series-1)" : "var(--surface)"}
                stroke="var(--series-1)" strokeWidth="2" />
              {sel && <circle cx={x(d.x)} cy={y(d.y)} r="11" fill="none" stroke="var(--series-1)" strokeWidth="1" />}
              {d._label && <text x={x(d.x) + 9} y={y(d.y) - 7} fontSize="11" style={{ fill: sel ? "var(--text)" : "var(--text-2)", fontWeight: sel ? 700 : 500 }}>{d.key}</text>}
            </g>
          );
        })}
      </svg>
      {t.node}
    </div>
  );
}
