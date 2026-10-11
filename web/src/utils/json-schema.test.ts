import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import schemaJson from "../../../src/nn/model_graph.schema.json";
import sample from "../../../examples/sample_model_graph.json";
import minimal from "../../../examples/minimal_model_graph.json";
import { describeType, schemaSections, validateJson, type JsonSchema } from "./json-schema";
import {
  CANONICAL_MODEL_NAMES,
  EDGE_KINDS,
  FLOW_NODE_KINDS,
  NODE_KINDS,
  NODE_STATUSES,
  type ModelHierarchyGraph,
  type ModuleNode,
  type NodeData,
} from "./model-graph";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = path.resolve(__dirname, "../../../examples/models");

const schema = schemaJson as JsonSchema;
const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;
const enumOf = (def: string, field: string) => schema.$defs?.[def]?.properties?.[field]?.enum;

describe("enum types", () => {
  it("match the enum lists of the schema one to one", () => {
    expect(enumOf("TensorNode", "kind")).toEqual([...NODE_KINDS]);
    expect(enumOf("TensorNode", "status")).toEqual([...NODE_STATUSES]);
    expect(enumOf("FlowNode", "kind")).toEqual([...FLOW_NODE_KINDS]);
    expect(enumOf("Edge", "kind")).toEqual([...EDGE_KINDS]);
    expect(enumOf("ParamEntry", "status")).toEqual(["CUSTOM_INIT", "AUTO_GRAPH"]);
  });
});

describe("model graph JSON schema", () => {
  it("accepts the bundled sample and the minimal template", () => {
    expect(validateJson(schema, sample)).toEqual([]);
    expect(validateJson(schema, minimal)).toEqual([]);
  });

  it("documents every property and every definition", () => {
    const undocumented: string[] = [];
    for (const section of schemaSections(schema)) {
      if (!section.description) undocumented.push(section.name);
      for (const f of section.fields) if (!f.description) undocumented.push(`${section.name}.${f.name}`);
    }
    expect(undocumented).toEqual([]);
  });

  it("closes every object definition", () => {
    const open = Object.entries(schema.$defs || {})
      .filter(([, def]) => def.type === "object" && def.additionalProperties !== false)
      .map(([name]) => name);
    expect(open).toEqual([]);
    expect(schema.additionalProperties).toBe(false);
  });

  it("reports unknown fields, missing fields, wrong enums and wrong versions with paths", () => {
    const doc = clone(minimal) as Record<string, any>;
    doc.version = "1.0";
    doc.model_name = "x";
    delete doc.root.children[0].edges;
    doc.root.flow_nodes[0].kind = "input";
    doc.summary.total_nodes = -1;
    const errors = validateJson(schema, doc);
    const byPath = Object.fromEntries(errors.map((e) => [e.path, e.message]));
    expect(byPath["$.version"]).toMatch(/expected "2.0"/);
    expect(byPath["$.model_name"]).toBe("unknown field");
    expect(byPath["$.root.children[0].edges"]).toBe("missing required field");
    expect(byPath["$.root.flow_nodes[0].kind"]).toMatch(/expected one of/);
    expect(byPath["$.summary.total_nodes"]).toMatch(/>= 0/);
  });

  it("accepts null formulas but rejects other types", () => {
    const doc = clone(minimal) as Record<string, any>;
    doc.root.formula = null;
    expect(validateJson(schema, doc)).toEqual([]);
    doc.root.formula = 3;
    expect(validateJson(schema, doc)).toEqual([{ path: "$.root.formula", message: "expected string | null, got integer" }]);
  });

  it("caps the number of reported errors", () => {
    const doc = clone(sample) as Record<string, any>;
    const strip = (m: Record<string, any>) => {
      delete m.kind;
      m.children.forEach(strip);
    };
    strip(doc.root);
    expect(validateJson(schema, doc, 5)).toHaveLength(5);
  });
});

describe("schema documentation helpers", () => {
  it("renders references, arrays, enums, consts and unions", () => {
    expect(describeType({ $ref: "#/$defs/Summary" })).toEqual({ type: "Summary", ref: "Summary" });
    expect(describeType({ type: "array", items: { $ref: "#/$defs/ModuleNode" } })).toEqual({ type: "ModuleNode[]", ref: "ModuleNode" });
    expect(describeType({ enum: ["data", "buffer"] })).toEqual({ type: '"data" | "buffer"' });
    expect(describeType({ const: "2.0" })).toEqual({ type: '"2.0"' });
    expect(describeType({ type: ["string", "null"] })).toEqual({ type: "string | null" });
    expect(describeType({ type: "array", items: { type: "string" } })).toEqual({ type: "string[]", ref: undefined });
  });

  it("lists the top level first and then every definition with its required flags", () => {
    const sections = schemaSections(schema);
    expect(sections.map((s) => s.name)).toEqual([
      "ModelHierarchyGraph",
      "Summary",
      "ModuleNode",
      "ParamEntry",
      "OpEntry",
      "Ports",
      "PortEntry",
      "FlowNode",
      "Edge",
      "TensorNode",
    ]);
    const edge = sections.find((s) => s.name === "Edge")!;
    expect(edge.fields.find((f) => f.name === "dst_shape")!.required).toBe(false);
    expect(edge.fields.find((f) => f.name === "is_skip")!.required).toBe(true);
  });
});

describe("canonical 18 models schema and port verification", () => {
  it("includes all 18 canonical models in public/tools/visualizer/models/", () => {
    const files = fs.readdirSync(MODELS_DIR).filter((f) => f.endsWith(".json")).sort();
    expect(files).toEqual(CANONICAL_MODEL_NAMES.map((name) => `${name}.json`));
  });

  it.each(CANONICAL_MODEL_NAMES)("validates %s.json against model_graph.schema.json with zero errors", (name) => {
    const raw = fs.readFileSync(path.join(MODELS_DIR, `${name}.json`), "utf8");
    const doc = JSON.parse(raw);
    const errors = validateJson(schema, doc);
    expect(errors).toEqual([]);
  });

  it.each(CANONICAL_MODEL_NAMES)("ensures %s has >=1 input port in root.ports.inputs connected to downstream nodes", (name) => {
    const raw = fs.readFileSync(path.join(MODELS_DIR, `${name}.json`), "utf8");
    const doc = JSON.parse(raw) as ModelHierarchyGraph;
    const inputs = doc.root?.ports?.inputs ?? [];
    expect(inputs.length).toBeGreaterThanOrEqual(1);

    const edges = doc.root?.edges ?? [];
    for (const port of inputs) {
      expect(port.id).toMatch(/^@in\d+/);
      const connections = edges.filter((e) => e.from === port.id);
      expect(connections.length).toBeGreaterThanOrEqual(1);
    }
  });

  it.each(CANONICAL_MODEL_NAMES)("ensures %s has no input tensor misclassified as buffer", (name) => {
    const raw = fs.readFileSync(path.join(MODELS_DIR, `${name}.json`), "utf8");
    const doc = JSON.parse(raw) as ModelHierarchyGraph;

    function checkModule(m: ModuleNode) {
      const allNodes: NodeData[] = [...(m.nodes || []), ...(m.parameters || [])];
      for (const node of allNodes) {
        if (node.kind === "Input") {
          expect(node.status).not.toBe("BUFFER");
        }
        if (node.status === "INPUT") {
          expect(node.kind).not.toBe("Buffer");
        }
        if (node.name && (node.name.startsWith("inputs.") || node.name === "inputs")) {
          expect(node.kind).not.toBe("Buffer");
          expect(node.status).not.toBe("BUFFER");
        }
      }
      for (const fn of m.flow_nodes || []) {
        if (fn.id.startsWith("@in") || fn.kind === "port_in") {
          expect(fn.kind).not.toBe("buffer");
        }
      }
      for (const child of m.children || []) {
        checkModule(child);
      }
    }

    checkModule(doc.root);
  });
});

