/**
 * Pure edge-routing geometry for the local data-flow pipeline.
 *
 * The renderer lays cards out in rows (see `layoutPipeline`), measures their boxes and asks
 * `routeEdges` for one SVG path per real edge of the local graph. Keeping this DOM-free lets
 * the routing rules be unit tested with synthetic rectangles.
 */
import type { EdgeData } from "./model-graph";

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type EdgeRouteKind = "data" | "skip" | "buffer";

export interface EdgeRoute {
  from: string;
  to: string;
  kind: EdgeRouteKind;
  /** SVG path data, in the coordinate space of the measured rectangles. */
  d: string;
  /** True when the edge leaves the straight corridor and runs along a side channel. */
  detour: boolean;
  /** Skip lane strip drawn under a straight skip edge. */
  lane?: Rect;
}

export interface FlowGeometry {
  /** Card box per flow-node id. */
  cards: Map<string, Rect>;
  /** Residual block box keyed by the block's merge (Add) target id. */
  blocks: Map<string, Rect>;
  /** Box enclosing all rows; detoured edges run in a channel to its right. */
  bounds: Rect;
}

/** Spacing between parallel side channels. */
const CHANNEL_STEP = 8;
/** Distance of the first side channel from the rows. */
const CHANNEL_OFFSET = 14;
/** Offset of the skip lane center from the left edge of its residual block. */
const SKIP_GUTTER = 11;
/** Width of the skip lane strip. */
const LANE_WIDTH = 8;
/** Vertical clearance used when an edge turns into a side channel. */
const TURN = 9;
/** Corner radius of orthogonal paths. */
const RADIUS = 6;

export function edgeRouteKind(e: EdgeData): EdgeRouteKind {
  if (e.is_skip) return "skip";
  return e.kind === "buffer" ? "buffer" : "data";
}

function fmt(n: number): string {
  return String(Math.round(n * 10) / 10);
}

/** Orthogonal polyline with rounded corners. */
export function orthoPath(points: [number, number][], radius = RADIUS): string {
  const pts = points.filter((p, i) => i === 0 || p[0] !== points[i - 1][0] || p[1] !== points[i - 1][1]);
  if (pts.length === 0) return "";
  let d = `M ${fmt(pts[0][0])} ${fmt(pts[0][1])}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i - 1];
    const [cx, cy] = pts[i];
    const [nx, ny] = pts[i + 1];
    const inLen = Math.hypot(cx - px, cy - py);
    const outLen = Math.hypot(nx - cx, ny - cy);
    const r = Math.min(radius, inLen / 2, outLen / 2);
    const ax = cx - ((cx - px) / (inLen || 1)) * r;
    const ay = cy - ((cy - py) / (inLen || 1)) * r;
    const bx = cx + ((nx - cx) / (outLen || 1)) * r;
    const by = cy + ((ny - cy) / (outLen || 1)) * r;
    d += ` L ${fmt(ax)} ${fmt(ay)} Q ${fmt(cx)} ${fmt(cy)} ${fmt(bx)} ${fmt(by)}`;
  }
  const last = pts[pts.length - 1];
  d += ` L ${fmt(last[0])} ${fmt(last[1])}`;
  return d;
}

function inside(p: [number, number], r: Rect, pad = 2): boolean {
  return p[0] > r.x + pad && p[0] < r.x + r.w - pad && p[1] > r.y + pad && p[1] < r.y + r.h - pad;
}

/** True when the polyline passes through any card other than its endpoints. */
function polylineHitsCard(points: [number, number][], cards: Map<string, Rect>, skip: Set<string>): boolean {
  for (let i = 1; i < points.length; i++) {
    const [ax, ay] = points[i - 1];
    const [bx, by] = points[i];
    const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 4));
    for (let k = 0; k <= steps; k++) {
      const p: [number, number] = [ax + ((bx - ax) * k) / steps, ay + ((by - ay) * k) / steps];
      for (const [id, r] of cards) {
        if (!skip.has(id) && inside(p, r)) return true;
      }
    }
  }
  return false;
}

function spread(r: Rect, i: number, n: number): number {
  return r.x + (r.w * (i + 1)) / (n + 1);
}

/** Minimum horizontal distance between two anchors on the same side of a card. */
const MIN_ANCHOR_GAP = 14;
/** Anchors keep this distance from the card corners. */
const ANCHOR_MARGIN = 10;

/**
 * Route every edge whose endpoints both have a card.
 *
 * - Data / buffer edges leave the bottom of the source and enter the top of the target. Whenever
 *   the two cards overlap horizontally the edge is a straight vertical line, placed at the center
 *   of the narrower card (clamped into the overlap), so chains, fan-out and fan-in all use
 *   straight drops. Otherwise, or when straight anchors on one card side would collide, that side
 *   spreads its anchors evenly (ordered by the x of the other end) and the edge becomes an
 *   orthogonal elbow that turns in the gap just above the target. If the path would pass through
 *   another card, the edge runs in its own channel to the right of the rows instead.
 * - Skip edges run straight down the left lane of their residual block, from the bottom of the
 *   source to the top of the merge node, over a lane strip. If the source or the merge node does
 *   not reach over the lane, the edge leaves the source bottom, runs down the lane and enters the
 *   merge node from the left.
 */
export function routeEdges(edges: EdgeData[], geo: FlowGeometry): EdgeRoute[] {
  const drawable = edges.filter((e) => geo.cards.has(e.from) && geo.cards.has(e.to) && e.from !== e.to);
  const card = (id: string) => geo.cards.get(id)!;
  const center = (id: string) => card(id).x + card(id).w / 2;
  const spans = (r: Rect, x: number) => x > r.x + 2 && x < r.x + r.w - 2;

  // A skip edge is drawn as a straight lane when its block's lane x lies under both the source
  // and the merge node (the renderer aligns them); such edges take no anchor slot.
  const laneX = new Map<EdgeData, number>();
  for (const e of drawable) {
    const block = e.is_skip ? geo.blocks.get(e.to) : undefined;
    if (!block) continue;
    const x = block.x + SKIP_GUTTER;
    const s = card(e.from);
    const t = card(e.to);
    if (spans(s, x) && spans(t, x) && t.y > s.y + s.h) laneX.set(e, x);
  }
  const lanesAt = (id: string) => [...laneX].filter(([e]) => e.from === id || e.to === id).map(([, x]) => x);

  // Straight candidate: shared x inside the horizontal overlap of both cards.
  const straightX = new Map<EdgeData, number>();
  for (const e of drawable) {
    if (e.is_skip) continue;
    const s = card(e.from);
    const t = card(e.to);
    const lo = Math.max(s.x, t.x) + ANCHOR_MARGIN;
    const hi = Math.min(s.x + s.w, t.x + t.w) - ANCHOR_MARGIN;
    if (lo > hi) continue;
    const pref = s.w <= t.w ? center(e.from) : center(e.to);
    straightX.set(e, Math.min(hi, Math.max(lo, pref)));
  }

  /** Anchor x per edge on one card side: straight candidates if they fit, else an even spread. */
  const assignSide = (id: string, list: EdgeData[], order: (a: EdgeData, b: EdgeData) => number): Map<EdgeData, number> => {
    const out = new Map<EdgeData, number>();
    const xs = list.map((e) => straightX.get(e));
    const taken = [...lanesAt(id)];
    let fits = xs.every((x) => x !== undefined);
    if (fits) {
      for (const x of xs as number[]) {
        if (taken.some((y) => Math.abs(y - x) < MIN_ANCHOR_GAP)) {
          fits = false;
          break;
        }
        taken.push(x);
      }
    }
    if (fits) {
      list.forEach((e, i) => out.set(e, xs[i]!));
      return out;
    }
    const sorted = [...list].sort(order);
    sorted.forEach((e, i) => out.set(e, spread(card(id), i, sorted.length)));
    return out;
  };

  // Outgoing anchors (bottom side): fallback skips first (they turn left), then by target x.
  const outX = new Map<EdgeData, number>();
  const bySource = new Map<string, EdgeData[]>();
  for (const e of drawable) {
    if (laneX.has(e)) continue;
    if (!bySource.has(e.from)) bySource.set(e.from, []);
    bySource.get(e.from)!.push(e);
  }
  for (const [src, list] of bySource) {
    const side = assignSide(src, list, (a, b) => (a.is_skip === b.is_skip ? center(a.to) - center(b.to) : a.is_skip ? -1 : 1));
    side.forEach((x, e) => outX.set(e, x));
  }

  // Incoming anchors (top side) for non-skip edges, ordered by the x of their outgoing anchor.
  const inX = new Map<EdgeData, number>();
  const byTarget = new Map<string, EdgeData[]>();
  for (const e of drawable) {
    if (e.is_skip) continue;
    if (!byTarget.has(e.to)) byTarget.set(e.to, []);
    byTarget.get(e.to)!.push(e);
  }
  for (const [dst, list] of byTarget) {
    const side = assignSide(dst, list, (a, b) => outX.get(a)! - outX.get(b)!);
    side.forEach((x, e) => inX.set(e, x));
  }

  const routes: EdgeRoute[] = [];
  let channel = 0;
  const rightChannelBase = geo.bounds.x + geo.bounds.w + CHANNEL_OFFSET;

  for (const e of drawable) {
    const kind = edgeRouteKind(e);
    const src = card(e.from);
    const dst = card(e.to);
    const sy = src.y + src.h;
    const lx = laneX.get(e);
    if (lx !== undefined) {
      routes.push({
        from: e.from,
        to: e.to,
        kind,
        d: `M ${fmt(lx)} ${fmt(sy)} L ${fmt(lx)} ${fmt(dst.y)}`,
        detour: false,
        lane: { x: lx - LANE_WIDTH / 2, y: sy, w: LANE_WIDTH, h: dst.y - sy },
      });
      continue;
    }
    const s: [number, number] = [outX.get(e)!, sy];
    if (e.is_skip) {
      const ty = dst.y + dst.h / 2;
      const block = geo.blocks.get(e.to);
      const gx = block ? block.x + SKIP_GUTTER : Math.min(src.x, dst.x) - CHANNEL_OFFSET;
      const turnY = block ? Math.max(s[1] + TURN / 2, Math.min(block.y - TURN, s[1] + TURN * 2)) : s[1] + TURN;
      routes.push({
        from: e.from,
        to: e.to,
        kind,
        d: orthoPath([s, [s[0], turnY], [gx, turnY], [gx, ty], [dst.x, ty]]),
        detour: true,
      });
      continue;
    }
    const t: [number, number] = [inX.get(e)!, dst.y];
    const ends = new Set([e.from, e.to]);
    if (t[1] > s[1]) {
      const straight = Math.abs(s[0] - t[0]) < 0.5;
      const gap = t[1] - s[1];
      const elbowY = gap > 3 * TURN ? t[1] - 1.5 * TURN : s[1] + gap / 2;
      const points: [number, number][] = straight ? [s, [s[0], t[1]]] : [s, [s[0], elbowY], [t[0], elbowY], t];
      if (!polylineHitsCard(points, geo.cards, ends)) {
        routes.push({ from: e.from, to: e.to, kind, d: straight ? `M ${fmt(s[0])} ${fmt(s[1])} L ${fmt(t[0])} ${fmt(t[1])}` : orthoPath(points), detour: false });
        continue;
      }
    }
    const gx = rightChannelBase + channel * CHANNEL_STEP;
    channel += 1;
    routes.push({
      from: e.from,
      to: e.to,
      kind,
      d: orthoPath([s, [s[0], s[1] + TURN], [gx, s[1] + TURN], [gx, t[1] - TURN], [t[0], t[1] - TURN], t]),
      detour: true,
    });
  }
  return routes;
}
