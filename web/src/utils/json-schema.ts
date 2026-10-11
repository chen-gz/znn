/**
 * src/utils/json-schema.ts
 *
 * Minimal JSON Schema (draft 2020-12) support for the ZNN model visualizer.
 * Covers exactly the keyword subset used by `public/tools/visualizer/model_graph.schema.json`
 * (`$ref` into `$defs`, `type`, `const`, `enum`, `minimum`, `required`, `properties`,
 * `additionalProperties: false`, `items`). The same schema drives upload validation and
 * the field tables of the JSON Format Guide.
 */

export type JsonType = "string" | "integer" | "number" | "boolean" | "null" | "array" | "object";

export interface JsonSchema {
  $schema?: string;
  $id?: string;
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  title?: string;
  description?: string;
  type?: JsonType | JsonType[];
  const?: string;
  enum?: string[];
  minimum?: number;
  required?: string[];
  properties?: Record<string, JsonSchema>;
  additionalProperties?: boolean;
  items?: JsonSchema;
}

export interface SchemaError {
  /** JSON path of the offending value, e.g. `$.root.children[0].edges[2].kind`. */
  path: string;
  message: string;
}

const DEFS_PREFIX = "#/$defs/";

function resolveRef(root: JsonSchema, schema: JsonSchema): JsonSchema {
  if (!schema.$ref) return schema;
  if (!schema.$ref.startsWith(DEFS_PREFIX)) throw new Error(`unsupported $ref ${schema.$ref}`);
  const target = root.$defs?.[schema.$ref.slice(DEFS_PREFIX.length)];
  if (!target) throw new Error(`unresolved $ref ${schema.$ref}`);
  return target;
}

function jsonTypeOf(value: unknown): JsonType {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
  return typeof value as JsonType;
}

function typeMatches(expected: JsonType, actual: JsonType): boolean {
  return expected === actual || (expected === "number" && actual === "integer");
}

/** Validates `value` against `schema`; returns at most `maxErrors` violations. */
export function validateJson(schema: JsonSchema, value: unknown, maxErrors = 20): SchemaError[] {
  const errors: SchemaError[] = [];

  const visit = (node: JsonSchema, v: unknown, path: string): void => {
    if (errors.length >= maxErrors) return;
    const s = resolveRef(schema, node);
    const actual = jsonTypeOf(v);
    const report = (message: string) => errors.push({ path, message });

    if (s.type !== undefined) {
      const expected = Array.isArray(s.type) ? s.type : [s.type];
      if (!expected.some((t) => typeMatches(t, actual))) {
        report(`expected ${expected.join(" | ")}, got ${actual}`);
        return;
      }
    }
    if (s.const !== undefined && v !== s.const) {
      report(`expected ${JSON.stringify(s.const)}, got ${JSON.stringify(v)}`);
      return;
    }
    if (s.enum !== undefined && !s.enum.includes(v as string)) {
      report(`expected one of ${s.enum.map((e) => JSON.stringify(e)).join(", ")}, got ${JSON.stringify(v)}`);
      return;
    }
    if (s.minimum !== undefined && typeof v === "number" && v < s.minimum) {
      report(`must be >= ${s.minimum}`);
    }

    if (actual === "object") {
      const obj = v as Record<string, unknown>;
      for (const key of s.required || []) {
        if (!(key in obj)) errors.push({ path: `${path}.${key}`, message: "missing required field" });
      }
      for (const [key, child] of Object.entries(obj)) {
        const sub = s.properties?.[key];
        if (sub) visit(sub, child, `${path}.${key}`);
        else if (s.additionalProperties === false) errors.push({ path: `${path}.${key}`, message: "unknown field" });
        if (errors.length >= maxErrors) return;
      }
    } else if (actual === "array" && s.items) {
      (v as unknown[]).forEach((item, i) => visit(s.items as JsonSchema, item, `${path}[${i}]`));
    }
  };

  visit(schema, value, "$");
  return errors.slice(0, maxErrors);
}

// ---------------------------------------------------------------------------
// Documentation helpers (JSON Format Guide)
// ---------------------------------------------------------------------------

export interface SchemaFieldRow {
  name: string;
  /** Human-readable type, e.g. `ModuleNode[]`, `string | null`, `"data" | "buffer"`. */
  type: string;
  /** `$defs` name the type refers to (for in-page links), if any. */
  ref?: string;
  required: boolean;
  description: string;
}

export interface SchemaSection {
  /** `$defs` key, or the schema title for the top level. */
  name: string;
  description: string;
  fields: SchemaFieldRow[];
}

function refName(ref: string | undefined): string | undefined {
  return ref && ref.startsWith(DEFS_PREFIX) ? ref.slice(DEFS_PREFIX.length) : undefined;
}

/** Renders the type of a property schema as a short TypeScript-like string. */
export function describeType(prop: JsonSchema): { type: string; ref?: string } {
  const ref = refName(prop.$ref);
  if (ref) return { type: ref, ref };
  if (prop.const !== undefined) return { type: JSON.stringify(prop.const) };
  if (prop.enum) return { type: prop.enum.map((e) => JSON.stringify(e)).join(" | ") };
  if (prop.type === "array" && prop.items) {
    const inner = describeType(prop.items);
    return { type: `${inner.type}[]`, ref: inner.ref };
  }
  if (Array.isArray(prop.type)) return { type: prop.type.join(" | ") };
  return { type: prop.type || "any" };
}

function sectionOf(name: string, s: JsonSchema): SchemaSection {
  const required = new Set(s.required || []);
  return {
    name,
    description: s.description || "",
    fields: Object.entries(s.properties || {}).map(([field, prop]) => ({
      name: field,
      ...describeType(prop),
      required: required.has(field),
      description: prop.description || "",
    })),
  };
}

/** Top-level object first, then every `$defs` entry in declaration order. */
export function schemaSections(schema: JsonSchema): SchemaSection[] {
  return [
    sectionOf(schema.title || "Document", schema),
    ...Object.entries(schema.$defs || {}).map(([name, def]) => sectionOf(name, def)),
  ];
}
