import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sample from "../../../examples/sample_model_graph.json";
import {
  CANONICAL_MODEL_NAMES,
  buildLocalGraph,
  buildMermaidLocalGraph,
  collectSubtree,
  edgeShapeLabel,
  findModuleByPath,
  hasLocalGraph,
  isAddNode,
  layoutPipeline,
  listModules,
  PIPELINE_CARD_MIN_REM,
  PIPELINE_GAP_REM,
  PIPELINE_RESIDUAL_CHROME_REM,
  pipelineMinWidthRem,
  residualBranch,
  resolveCanonical,
  type ModelHierarchyGraph,
  type ModuleNode,
  type PipelineRow,
} from "./model-graph";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = path.resolve(__dirname, "../../../examples/models");

const graph = sample as unknown as ModelHierarchyGraph;

// Module paths in the GPT sample (znn names modules by their struct field paths).
const BLOCK = "gpt.decoder.h.0";
const NEXT_BLOCK = "gpt.decoder.h.1";

function mod(path: string): ModuleNode {
  const m = findModuleByPath(graph.root, path);
  if (!m) throw new Error(`module ${path} not found`);
  return m;
}

function rowIds(rows: PipelineRow[]): string[][] {
  return rows.map((r) =>
    r.units.map((u) => (u.type === "node" ? u.id : `residual(${u.skip.from}->${u.target}: ${u.branchIds.join(",")})`)),
  );
}

describe("model-graph schema 2.0", () => {
  it("exposes default_scope instead of top-level edges", () => {
    expect(graph.version).toBe("2.0");
    expect(graph.default_scope).toBe("gpt");
    expect((graph as unknown as Record<string, unknown>).edges).toBeUndefined();
    expect(graph.summary?.buffer_nodes).toBe(1);
  });

  it("finds modules by path, treating root and '' alike", () => {
    expect(findModuleByPath(graph.root, "")).toBe(graph.root);
    expect(findModuleByPath(graph.root, "root")).toBe(graph.root);
    expect(findModuleByPath(graph.root, `${BLOCK}.attn.core`)?.module_type).toBe("ScaledDotProductAttention");
    expect(findModuleByPath(graph.root, "gpt.decoder.h.9")).toBeNull();
  });

  it("resolves ports and children to canonical paths", () => {
    const block = mod(BLOCK);
    expect(resolveCanonical(block, "@in0")).toBe("gpt.embeddings_sum");
    expect(resolveCanonical(block, "@out0")).toBe(NEXT_BLOCK);
    expect(resolveCanonical(block, "attn")).toBe(`${BLOCK}.attn`);
    expect(resolveCanonical(block, "residual_mlp")).toBe(`${BLOCK}.residual_mlp`);
    expect(resolveCanonical(mod("gpt"), "pos_indices")).toBe("gpt.pos_indices");
    expect(resolveCanonical(graph.root, "@in0")).toBe("inputs.token_ids");
  });

  it("marks only Add ops as merge nodes", () => {
    const block = buildLocalGraph(mod(BLOCK));
    expect(isAddNode(block.nodes.get("residual_attn"))).toBe(true);
    const attn = buildLocalGraph(mod(`${BLOCK}.attn`));
    expect(isAddNode(attn.nodes.get("core"))).toBe(false);
    expect(attn.incoming.get("core")?.length).toBe(3);
  });

  it("derives residual branches from reachability", () => {
    const g = buildLocalGraph(mod(BLOCK));
    const skips = g.edges.filter((e) => e.is_skip);
    expect(skips.map((e) => `${e.from}->${e.to}`)).toEqual(["@in0->residual_attn", "residual_attn->residual_mlp"]);
    expect(residualBranch(g, skips[0])).toEqual(["ln_1", "attn"]);
    expect(residualBranch(g, skips[1])).toEqual(["ln_2", "mlp"]);
  });

  it("lays out a transformer block as two residual units", () => {
    const rows = layoutPipeline(buildLocalGraph(mod(BLOCK)));
    expect(rowIds(rows)).toEqual([
      ["@in0"],
      ["residual(@in0->residual_attn: ln_1,attn)"],
      ["residual(residual_attn->residual_mlp: ln_2,mlp)"],
      ["@out0"],
    ]);
  });

  it("draws a shared input once and fans out to q/k/v", () => {
    const rows = layoutPipeline(buildLocalGraph(mod(`${BLOCK}.attn`)));
    expect(rowIds(rows)).toEqual([["@in0"], ["q_attn", "k_attn", "v_attn"], ["core"], ["c_proj"], ["@out0"]]);
  });

  it("keeps the embedding merge as parallel sources into one Add", () => {
    const rows = layoutPipeline(buildLocalGraph(mod("gpt")));
    expect(rowIds(rows)).toEqual([
      ["@in0", "pos_indices"],
      ["token_embedding", "position_embedding"],
      ["embeddings_sum"],
      ["decoder"],
      ["lm_head"],
      ["@out0"],
    ]);
  });

  it("places sources directly above their first consumer", () => {
    const rows = layoutPipeline(buildLocalGraph(mod(`${BLOCK}.attn.core`)));
    expect(rowIds(rows)).toEqual([
      ["@in0", "@in1", "@in2"],
      ["act_Attention_22"],
      ["@out0"],
    ]);
  });

  it("orders each row by the barycenter of its predecessors", () => {
    const synthetic: ModuleNode = {
      name: "m",
      path: "m",
      flow_nodes: ["a", "b", "y", "x"].map((id) => ({ id, kind: "op", ref: `m.${id}` })),
      edges: [
        { from: "a", to: "x", shape: "[1]", is_skip: false, kind: "data" },
        { from: "b", to: "y", shape: "[1]", is_skip: false, kind: "data" },
      ],
    };
    expect(rowIds(layoutPipeline(buildLocalGraph(synthetic)))).toEqual([
      ["a", "b"],
      ["x", "y"],
    ]);
  });

  it("labels edges with shape transitions", () => {
    const g = buildLocalGraph(mod("gpt"));
    const e = g.edges.find((x) => x.from === "decoder" && x.to === "lm_head")!;
    expect(edgeShapeLabel(e)).toBe("[2, 16, 64] → [32, 64]");
    expect(edgeShapeLabel(g.edges.find((x) => x.from === "token_embedding")!)).toBe("[2, 16, 64]");
  });

  it("generates Mermaid for a local graph with dashed skips and dotted buffers", () => {
    const { code } = buildMermaidLocalGraph(mod(BLOCK), (n) => n.ref);
    expect(code).toContain('(("⊕ residual_attn"))');
    expect(code).toContain("skip · [2, 16, 64]");
    expect(code).toContain("stroke-dasharray:8 4");
    expect(code).toContain(`call onMermaidNodeClick("${BLOCK}.attn")`);
    expect(code).not.toContain(`${BLOCK}.output`);

    const gptGraph = buildMermaidLocalGraph(mod("gpt")).code;
    expect(gptGraph).toContain('[("🧱 pos_indices")]');
    expect(gptGraph).toContain("stroke-dasharray:2 3");

    const core = buildMermaidLocalGraph(mod(`${BLOCK}.attn.core`)).code;
    expect(core).toContain('("⚡ act_Attention_22 : Attention")');
  });

  it("collects subtree nodes by module rather than by name prefix", () => {
    const { nodes, params, ops } = collectSubtree(mod(`${BLOCK}.mlp`));
    expect(params.map((p) => p.name).sort()).toEqual([
      `${BLOCK}.mlp.c_fc.bias`,
      `${BLOCK}.mlp.c_fc.weight`,
      `${BLOCK}.mlp.c_proj.bias`,
      `${BLOCK}.mlp.c_proj.weight`,
    ]);
    expect(nodes.some((n) => n.name === `${BLOCK}.mlp.gelu`)).toBe(true);
    expect(ops.some((o) => o.op_type === "Gelu" && o.module === `${BLOCK}.mlp`)).toBe(true);
  });
});

// Recurrent cells unroll their time steps over shared weight modules, so their local graphs
// legitimately contain back edges (h_t feeds the same w_hh module again at step t + 1).
const RECURRENT_CELL_TYPES = new Set(["RNNCell", "LSTMCell", "GRUCell"]);
const isRecurrentCell = (m: ModuleNode) => RECURRENT_CELL_TYPES.has(m.module_type ?? "");

describe("canonical 18 models DAG and navigation validation", () => {
  it.each(CANONICAL_MODEL_NAMES)(
    "succeeds in topological sorting without cycles for all scopes in %s",
    (name) => {
      const raw = fs.readFileSync(path.join(MODELS_DIR, `${name}.json`), "utf8");
      const model = JSON.parse(raw) as ModelHierarchyGraph;
      const modules = listModules(model.root);

      for (const m of modules) {
        if (!hasLocalGraph(m)) continue;
        const g = buildLocalGraph(m);

        // Topological sorting order must contain every node exactly once
        expect(g.order.length).toBe(g.nodes.size);
        expect(new Set(g.order).size).toBe(g.nodes.size);

        // For all DAG scopes (root, model container, and all scopes of feed-forward architectures),
        // topological sorting succeeds strictly without cycles: every edge points strictly forward
        if (!isRecurrentCell(m)) {
          const orderMap = new Map<string, number>(g.order.map((id, idx) => [id, idx]));
          for (const e of g.edges) {
            const fromIdx = orderMap.get(e.from);
            const toIdx = orderMap.get(e.to);
            expect(fromIdx).toBeDefined();
            expect(toIdx).toBeDefined();
            expect(fromIdx!).toBeLessThan(toIdx!);
          }
        }
      }
    }
  );

  it.each(CANONICAL_MODEL_NAMES)(
    "ensures submodule navigation and breadcrumb paths are well-formed without dangling references in %s",
    (name) => {
      const raw = fs.readFileSync(path.join(MODELS_DIR, `${name}.json`), "utf8");
      const model = JSON.parse(raw) as ModelHierarchyGraph;
      const modules = listModules(model.root);

      // Verify default_scope if specified
      if (model.default_scope) {
        const defMod = findModuleByPath(model.root, model.default_scope);
        expect(defMod).not.toBeNull();
      }

      for (const m of modules) {
        // Module can be found by its own path
        const found = findModuleByPath(model.root, m.path);
        expect(found).not.toBeNull();
        expect(found?.path || "").toBe(m.path || "");

        // Breadcrumb path navigation: every sub-path prefix must resolve to a valid module
        const segments = m.path ? m.path.split(".") : [];
        let prefix = "";
        for (const seg of segments) {
          prefix = prefix ? `${prefix}.${seg}` : seg;
          const parentMod = findModuleByPath(model.root, prefix);
          expect(parentMod).not.toBeNull();
        }

        // Child flow nodes referencing modules must point to existing modules
        for (const fn of m.flow_nodes || []) {
          if (fn.kind === "module") {
            const childMod = findModuleByPath(model.root, fn.ref);
            expect(childMod).not.toBeNull();
            expect(childMod?.path).toBe(fn.ref);
          }
          // Canonical resolution must be a non-empty string
          const canonical = resolveCanonical(m, fn.id);
          expect(canonical).toBeTruthy();
        }
      }
    }
  );

  it.each(CANONICAL_MODEL_NAMES)(
    "lays out pipeline rows without null or unmapped units in %s",
    (name) => {
      const raw = fs.readFileSync(path.join(MODELS_DIR, `${name}.json`), "utf8");
      const model = JSON.parse(raw) as ModelHierarchyGraph;
      const modules = listModules(model.root);

      for (const m of modules) {
        if (!hasLocalGraph(m)) continue;
        const g = buildLocalGraph(m);
        const rows = layoutPipeline(g);
        expect(rows.length).toBeGreaterThan(0);

        function verifyRows(rList: PipelineRow[]) {
          for (const row of rList) {
            expect(row.units.length).toBeGreaterThan(0);
            for (const u of row.units) {
              if (u.type === "node") {
                expect(g.nodes.has(u.id)).toBe(true);
              } else if (u.type === "residual") {
                expect(g.nodes.has(u.target)).toBe(true);
                expect(u.branchIds.length).toBeGreaterThan(0);
                verifyRows(u.branch);
              }
            }
          }
        }
        verifyRows(rows);
      }
    }
  );

  it("safely handles undefined, null, or non-string paths in findModuleByPath", () => {
    expect(findModuleByPath(graph.root, undefined as unknown as string)).toBeNull();
    expect(findModuleByPath(graph.root, null as unknown as string)).toBeNull();
    expect(findModuleByPath(graph.root, 123 as unknown as string)).toBeNull();
  });

  const ALL_TEST_MODELS = [
    ...CANONICAL_MODEL_NAMES.map((name) => ({ name, file: path.join(MODELS_DIR, `${name}.json`) })),
    { name: "sample", file: path.resolve(__dirname, "../../../examples/sample_model_graph.json") },
    { name: "minimal", file: path.resolve(__dirname, "../../../examples/minimal_model_graph.json") },
  ];

  it.each(ALL_TEST_MODELS)(
    "resolving the initial scope via visualizer logic yields a module with hasLocalGraph in $name",
    ({ file }) => {
      const raw = fs.readFileSync(file, "utf8");
      const model = JSON.parse(raw) as ModelHierarchyGraph;

      const rawScope = model.default_scope ?? "";
      const rawMod = findModuleByPath(model.root, rawScope);
      const initialScope = rawMod && hasLocalGraph(rawMod) ? rawScope : "";
      const resolvedMod = findModuleByPath(model.root, initialScope);

      expect(resolvedMod).not.toBeNull();
      expect(hasLocalGraph(resolvedMod!)).toBe(true);
      // In the regenerated backend JSONs, default_scope itself points directly to a module with local graph
      const defaultMod = findModuleByPath(model.root, rawScope);
      expect(defaultMod).not.toBeNull();
      expect(hasLocalGraph(defaultMod!)).toBe(true);
    }
  );
});

describe("pipeline minimum width", () => {
  it("is one card wide for a single-column pipeline", () => {
    const rows = layoutPipeline(buildLocalGraph(mod(`${BLOCK}.mlp`)));
    expect(rows.every((r) => r.units.length === 1 && r.units[0].type === "node")).toBe(true);
    expect(pipelineMinWidthRem(rows)).toBe(PIPELINE_CARD_MIN_REM);
  });

  it("sums cards and gaps of the widest row", () => {
    const rows = layoutPipeline(buildLocalGraph(mod(`${BLOCK}.attn`)));
    expect(pipelineMinWidthRem(rows)).toBe(3 * PIPELINE_CARD_MIN_REM + 2 * PIPELINE_GAP_REM);
  });

  it("adds the residual frame chrome around a branch", () => {
    const rows = layoutPipeline(buildLocalGraph(mod(BLOCK)));
    expect(pipelineMinWidthRem(rows)).toBe(PIPELINE_CARD_MIN_REM + PIPELINE_RESIDUAL_CHROME_REM);
  });

  it.each(CANONICAL_MODEL_NAMES)("fits the default view of %s within the 7xl page", (name) => {
    const model = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, `${name}.json`), "utf8")) as ModelHierarchyGraph;
    const scope = findModuleByPath(model.root, model.default_scope ?? "")!;
    expect(pipelineMinWidthRem(layoutPipeline(buildLocalGraph(scope)))).toBeLessThanOrEqual(70);
  });
});
