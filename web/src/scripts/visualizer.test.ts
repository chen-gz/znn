import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { loadGraphData } from "./visualizer";
import type { ModelHierarchyGraph } from "../utils/model-graph";

/** Element stub; getElementById hands out one per id, so tests can read what loadGraphData wrote. */
function elementStub() {
  return {
    innerHTML: "",
    textContent: "",
    appendChild: vi.fn(),
    querySelector: vi.fn().mockReturnValue(null),
    querySelectorAll: vi.fn().mockReturnValue([]),
    classList: {
      add: vi.fn(),
      remove: vi.fn(),
      contains: vi.fn(),
      toggle: vi.fn(),
    },
    addEventListener: vi.fn(),
    style: {},
    options: [],
  };
}

describe("loadGraphData", () => {
  let elements: Map<string, ReturnType<typeof elementStub>>;

  beforeEach(() => {
    elements = new Map();
    vi.stubGlobal("document", {
      getElementById: vi.fn((id: string) => {
        if (!elements.has(id)) elements.set(id, elementStub());
        return elements.get(id);
      }),
      querySelectorAll: vi.fn().mockReturnValue([]),
      createElement: vi.fn().mockReturnValue({
        appendChild: vi.fn(),
        setAttribute: vi.fn(),
        classList: { add: vi.fn(), remove: vi.fn(), toggle: vi.fn() },
        style: {},
        textContent: "",
        innerHTML: "",
        addEventListener: vi.fn(),
      }),
    });
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      requestAnimationFrame: vi.fn((cb: FrameRequestCallback) => cb(0)),
      mermaid: { initialize: vi.fn(), render: vi.fn() },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("throws error if graphObj is missing or has no root", () => {
    expect(() => loadGraphData(null as unknown as ModelHierarchyGraph)).toThrow(
      "Invalid graph data: missing root node",
    );
    expect(() => loadGraphData({ version: "2.0" } as unknown as ModelHierarchyGraph)).toThrow(
      "Invalid graph data: missing root node",
    );
  });

  it("successfully loads a valid graph and initializes state", () => {
    const validGraphObj: ModelHierarchyGraph = {
      version: "2.0",
      default_scope: "",
      root: {
        name: "root",
        path: "",
        kind: "module",
        module_type: "TestModule",
        formula: "y=x+1",
        total_params: 10,
        total_bytes: 100,
        ops: [{ op_type: "Add", name: "op1", module: "", formula: "x+1" }],
        nodes: [
          {
            name: "node1",
            kind: "Buffer",
            module: "",
            shape: "[10]",
            elements: 10,
            bytes: 100,
            status: "BUFFER",
          },
        ],
      },
      summary: {
        total_params: 10,
        total_bytes: 100,
        param_nodes: 1,
      },
    };

    loadGraphData(validGraphObj, "test-source");

    expect(elements.get("loaded-model-info")?.textContent).toBe("root (TestModule) · schema 2.0 · test-source");
    expect(document.getElementById).toHaveBeenCalledWith(
      "details-select-module",
    );
    expect(document.getElementById).toHaveBeenCalledWith("kpi-total-params");
  });

  it("uses default source name if none provided", () => {
    loadGraphData({
      version: "2.0",
      root: { name: "root", path: "", kind: "module", module_type: "TestModule" },
    });
    expect(elements.get("loaded-model-info")?.textContent).toBe("root (TestModule) · schema 2.0 · bundled sample");
  });
});
