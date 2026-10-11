import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sample from "../../../examples/sample_model_graph.json";
import { orthoPath, routeEdges, type FlowGeometry, type Rect } from "./flow-edges";
import {
  CANONICAL_MODEL_NAMES,
  buildLocalGraph,
  findModuleByPath,
  hasLocalGraph,
  layoutPipeline,
  listModules,
  type EdgeData,
  type ModelHierarchyGraph,
  type PipelineRow,
} from "./model-graph";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = path.resolve(__dirname, "../../../examples/models");

const edge = (from: string, to: string, extra: Partial<EdgeData> = {}): EdgeData => ({ from, to, shape: "[1]", is_skip: false, kind: "data", ...extra });
const rect = (x: number, y: number, w = 100, h = 40): Rect => ({ x, y, w, h });
const bounds = rect(0, 0, 300, 300);

function endpoints(d: string): { start: [number, number]; end: [number, number] } {
  const nums = d.match(/-?\d+(\.\d+)?/g)!.map(Number);
  return { start: [nums[0], nums[1]], end: [nums[nums.length - 2], nums[nums.length - 1]] };
}

describe("flow-edges routing", () => {
  it("draws parallel edges as separate connectors between their own cards", () => {
    const cards = new Map([
      ["a", rect(0, 0)],
      ["b", rect(150, 0)],
      ["c", rect(0, 80)],
      ["d", rect(150, 80)],
    ]);
    const routes = routeEdges([edge("a", "c"), edge("b", "d")], { cards, blocks: new Map(), bounds });
    expect(routes).toHaveLength(2);
    const ac = endpoints(routes.find((r) => r.from === "a")!.d);
    const bd = endpoints(routes.find((r) => r.from === "b")!.d);
    expect(ac).toEqual({ start: [50, 40], end: [50, 80] });
    expect(bd).toEqual({ start: [200, 40], end: [200, 80] });
    expect(routes.find((r) => r.from === "a")!.d).toBe("M 50 40 L 50 80");
    expect(routes.every((r) => !r.detour)).toBe(true);
  });

  it("uses straight drops for fan-in when the sources sit over the target", () => {
    const cards = new Map([
      ["a", rect(0, 0)],
      ["b", rect(150, 0)],
      ["sum", rect(0, 80, 250)],
    ]);
    const routes = routeEdges([edge("b", "sum"), edge("a", "sum")], { cards, blocks: new Map(), bounds });
    const fromA = endpoints(routes.find((r) => r.from === "a")!.d).end;
    const fromB = endpoints(routes.find((r) => r.from === "b")!.d).end;
    expect(fromA).toEqual([50, 80]);
    expect(fromB).toEqual([200, 80]);
    expect(routes.every((r) => /^M [\d.]+ [\d.]+ L [\d.]+ [\d.]+$/.test(r.d))).toBe(true);
  });

  it("uses straight drops for fan-out and keeps chains aligned across different widths", () => {
    const cards = new Map([
      ["in", rect(0, 0, 300)],
      ["q", rect(0, 80, 90)],
      ["k", rect(105, 80, 90)],
      ["v", rect(210, 80, 90)],
      ["wide", rect(0, 160, 300)],
      ["inner", rect(28, 240, 260)],
    ]);
    const routes = routeEdges([edge("in", "q"), edge("in", "k"), edge("in", "v"), edge("k", "wide"), edge("wide", "inner")], { cards, blocks: new Map(), bounds });
    for (const [to, x] of [["q", 45], ["k", 150], ["v", 255]] as const) {
      expect(routes.find((r) => r.to === to)!.d).toBe(`M ${x} 40 L ${x} 80`);
    }
    // The narrower card decides the column, so a chain into a padded inner card stays straight.
    expect(routes.find((r) => r.to === "inner")!.d).toBe("M 158 200 L 158 240");
  });

  it("spreads colliding anchors and turns in the gap above the target", () => {
    const cards = new Map([
      ["b", rect(0, 80, 100)],
      ["c", rect(110, 80, 100)],
      ["sum", rect(40, 160, 40)],
    ]);
    const routes = routeEdges([edge("b", "sum"), edge("c", "sum")], { cards, blocks: new Map(), bounds });
    const fromB = routes.find((r) => r.from === "b")!;
    const fromC = routes.find((r) => r.from === "c")!;
    expect(fromB.detour).toBe(false);
    expect(fromC.d).toContain("Q");
    expect(endpoints(fromB.d).end[0]).toBeLessThan(endpoints(fromC.d).end[0]);
  });

  it("detours an edge that would pass through another card", () => {
    const cards = new Map([
      ["a", rect(0, 0)],
      ["m", rect(0, 80)],
      ["t", rect(0, 160)],
    ]);
    const routes = routeEdges([edge("a", "m"), edge("m", "t"), edge("a", "t", { kind: "buffer" })], { cards, blocks: new Map(), bounds });
    const long = routes.find((r) => r.from === "a" && r.to === "t")!;
    expect(long.kind).toBe("buffer");
    expect(long.detour).toBe(true);
    expect(long.d).toContain(`${bounds.w + 14}`);
    expect(endpoints(long.d).end[1]).toBe(160);
    expect(routes.find((r) => r.to === "m")!.detour).toBe(false);
  });

  it("draws a skip edge as a straight lane from the source into the merge node", () => {
    const cards = new Map([
      ["x", rect(0, 0, 200)],
      ["f", rect(40, 90)],
      ["add", rect(0, 240, 200)],
    ]);
    const blocks = new Map([["add", rect(0, 70, 200, 140)]]);
    const routes = routeEdges([edge("x", "f"), edge("f", "add"), edge("x", "add", { is_skip: true })], { cards, blocks, bounds });
    const skip = routes.find((r) => r.kind === "skip")!;
    expect(skip.d).toBe("M 11 40 L 11 240");
    expect(skip.lane).toEqual({ x: 7, y: 40, w: 8, h: 200 });
    // The skip takes no anchor slot: x -> f drops straight into the middle of f.
    expect(endpoints(routes.find((r) => r.to === "f")!.d).start).toEqual([90, 40]);
  });

  it("falls back to a gutter path into the side of the merge node when the lane is not covered", () => {
    const cards = new Map([
      ["x", rect(40, 0)],
      ["f", rect(40, 90)],
      ["add", rect(40, 170)],
    ]);
    const blocks = new Map([["add", rect(10, 70, 160, 150)]]);
    const routes = routeEdges([edge("x", "f"), edge("f", "add"), edge("x", "add", { is_skip: true })], { cards, blocks, bounds });
    const skip = routes.find((r) => r.kind === "skip")!;
    expect(skip.lane).toBeUndefined();
    expect(skip.d).toContain("21");
    expect(endpoints(skip.d).end).toEqual([40, 190]);
  });

  it("rounds corners of orthogonal paths", () => {
    expect(orthoPath([[0, 0], [0, 20], [20, 20]])).toBe("M 0 0 L 0 14 Q 0 20 6 20 L 20 20");
  });

  it("gives every edge of the gpt scope its own connector", () => {
    const graph = sample as unknown as ModelHierarchyGraph;
    const g = buildLocalGraph(findModuleByPath(graph.root, "gpt")!);
    const cards = new Map<string, Rect>();
    layoutPipeline(g).forEach((row, i) =>
      row.units.forEach((u, j) => {
        if (u.type === "node") cards.set(u.id, rect(j * 160, i * 80));
      }),
    );
    const routes = routeEdges(g.edges, { cards, blocks: new Map(), bounds: rect(0, 0, 320, 480) });
    expect(routes).toHaveLength(g.edges.length);
    const tokens = routes.find((r) => r.from === "@in0" && r.to === "token_embedding")!;
    const positions = routes.find((r) => r.from === "pos_indices" && r.to === "position_embedding")!;
    expect(endpoints(tokens.d).start[0]).toBe(50);
    expect(endpoints(positions.d).start[0]).toBe(210);
    expect(positions.kind).toBe("buffer");
  });
});

function layoutSimulatedGeometry(rows: PipelineRow[], containerWidth = 600): FlowGeometry {
  const cards = new Map<string, Rect>();
  const blocks = new Map<string, Rect>();
  let curY = 0;

  function placeRows(rList: PipelineRow[], startX: number, width: number) {
    for (const row of rList) {
      const n = row.units.length;
      const colW = width / Math.max(n, 1);
      for (let j = 0; j < n; j++) {
        const u = row.units[j];
        const x = startX + j * colW + 10;
        const w = Math.max(colW - 20, 60);
        if (u.type === "node") {
          cards.set(u.id, { x, y: curY, w, h: 40 });
        } else {
          const blockY = curY;
          curY += 20;
          placeRows(u.branch, x + 20, w - 20);
          curY += 20;
          const blockH = Math.max(curY - blockY, 40);
          blocks.set(u.target, { x, y: blockY, w, h: blockH });
          cards.set(u.target, { x, y: curY, w, h: 40 });
          curY += 50;
        }
      }
      curY += 60;
    }
  }

  placeRows(rows, 0, containerWidth);
  const bounds: Rect = { x: 0, y: 0, w: containerWidth + 100, h: Math.max(curY, 100) };
  return { cards, blocks, bounds };
}

describe("canonical 18 models pipeline layout and edge routing", () => {
  it.each(CANONICAL_MODEL_NAMES)(
    "routes edges without NaN coordinates for all scopes in %s",
    (name) => {
      const raw = fs.readFileSync(path.join(MODELS_DIR, `${name}.json`), "utf8");
      const model = JSON.parse(raw) as ModelHierarchyGraph;
      const modules = listModules(model.root);

      for (const m of modules) {
        if (!hasLocalGraph(m)) continue;
        const g = buildLocalGraph(m);
        const rows = layoutPipeline(g);
        const geo = layoutSimulatedGeometry(rows);

        // Verify bounding boxes have no NaN or non-finite values
        for (const [, r] of geo.cards) {
          expect(Number.isFinite(r.x)).toBe(true);
          expect(Number.isFinite(r.y)).toBe(true);
          expect(Number.isFinite(r.w)).toBe(true);
          expect(Number.isFinite(r.h)).toBe(true);
          expect(Number.isNaN(r.x)).toBe(false);
          expect(Number.isNaN(r.y)).toBe(false);
          expect(Number.isNaN(r.w)).toBe(false);
          expect(Number.isNaN(r.h)).toBe(false);
        }
        for (const [, b] of geo.blocks) {
          expect(Number.isFinite(b.x)).toBe(true);
          expect(Number.isFinite(b.y)).toBe(true);
          expect(Number.isFinite(b.w)).toBe(true);
          expect(Number.isFinite(b.h)).toBe(true);
          expect(Number.isNaN(b.x)).toBe(false);
          expect(Number.isNaN(b.y)).toBe(false);
          expect(Number.isNaN(b.w)).toBe(false);
          expect(Number.isNaN(b.h)).toBe(false);
        }
        expect(Number.isFinite(geo.bounds.w)).toBe(true);
        expect(Number.isFinite(geo.bounds.h)).toBe(true);
        expect(Number.isNaN(geo.bounds.w)).toBe(false);
        expect(Number.isNaN(geo.bounds.h)).toBe(false);

        const routes = routeEdges(g.edges, geo);
        // Every edge where both endpoints are cards gets routed
        const expectedCount = g.edges.filter((e) => geo.cards.has(e.from) && geo.cards.has(e.to) && e.from !== e.to).length;
        expect(routes.length).toBe(expectedCount);

        for (const route of routes) {
          // Zero NaN in SVG edge paths
          expect(route.d).not.toContain("NaN");
          expect(route.d).not.toContain("undefined");
          expect(route.d).toMatch(/^M\s+[0-9.-]+\s+[0-9.-]+/);

          if (route.lane) {
            expect(Number.isFinite(route.lane.x)).toBe(true);
            expect(Number.isFinite(route.lane.y)).toBe(true);
            expect(Number.isFinite(route.lane.w)).toBe(true);
            expect(Number.isFinite(route.lane.h)).toBe(true);
            expect(Number.isNaN(route.lane.x)).toBe(false);
            expect(Number.isNaN(route.lane.y)).toBe(false);
            expect(Number.isNaN(route.lane.w)).toBe(false);
            expect(Number.isNaN(route.lane.h)).toBe(false);
          }
        }
      }
    }
  );
});

