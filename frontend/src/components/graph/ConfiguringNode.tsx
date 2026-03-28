import { memo, useEffect, useState } from "react";
import { type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { ComponentInfo } from "@/lib/types";
import { fetchOptions, type Option } from "@/lib/api";
import { Dropdown } from "@/components/ui/Dropdown";

interface ConfiguringNodeData {
  componentInfo: ComponentInfo;
  mode?: "create" | "edit";
  submitLabel?: string;
  initialValues?: Record<string, unknown>;
  onConfirm: (config: Record<string, unknown>) => void;
  onCancel: () => void;
}

type SchemaObj = {
  type?: string;
  properties?: Record<string, SchemaObj>;
  title?: string;
  default?: unknown;
  $ref?: string;
  $defs?: Record<string, SchemaObj>;
  anyOf?: SchemaObj[];
  description?: string;
  enum?: unknown[];
  format?: string;
  options?: Record<string, object>;
};

type ResolvedSchema = { properties: Record<string, SchemaObj>; title?: string; options?: Record<string, object> };

function hasProps(s: SchemaObj): s is SchemaObj & { properties: Record<string, SchemaObj> } {
  return s.type === "object" && s.properties !== undefined;
}

function resolveSchema(schema: SchemaObj): ResolvedSchema | null {
  if (hasProps(schema)) return schema;

  if (schema.anyOf) {
    for (const branch of schema.anyOf) {
      if (branch.$ref && schema.$defs) {
        const refName = branch.$ref.split("/").pop()!;
        const resolved = schema.$defs[refName];
        if (resolved && hasProps(resolved)) return resolved;
      }
      if (hasProps(branch)) return branch;
    }
  }

  if (schema.$ref && schema.$defs) {
    const refName = schema.$ref.split("/").pop()!;
    const resolved = schema.$defs[refName];
    if (resolved && hasProps(resolved)) return resolved;
  }

  return null;
}

function getDefaultValue(prop: SchemaObj): unknown {
  if (prop.default !== undefined) return prop.default;
  if (prop.type === "boolean") return false;
  return "";
}

/** Collect all fields from every init parameter's schema. */
function collectFields(init: Record<string, unknown>): {
  fields: Record<string, SchemaObj>;
  optionFields: Set<string>;
} {
  const fields: Record<string, SchemaObj> = {};
  const optionFields = new Set<string>();
  for (const [paramName, rawSchema] of Object.entries(init)) {
    if (!rawSchema || typeof rawSchema !== "object") continue;
    const schema = rawSchema as SchemaObj;
    const resolved = resolveSchema(schema);
    if (resolved) {
      // Detect options from the parent schema
      const coKeys = schema.options ?? resolved.options;
      // Object schema — each property becomes a field
      for (const [prop, propSchema] of Object.entries(resolved.properties)) {
        const key = `${paramName}.${prop}`;
        fields[key] = propSchema;
        if (coKeys && prop in coKeys) {
          optionFields.add(key);
        }
      }
    } else {
      // Simple type — the param itself is a field
      fields[paramName] = schema;
      // Also check for config options on simple fields
      optionFields.add(paramName);
    }
  }
  return { fields, optionFields };
}

/** Rebuild the nested init values dict from flat "param.prop" keys. */
function buildInitValues(
  fields: Record<string, SchemaObj>,
  values: Record<string, unknown>,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const key of Object.keys(fields)) {
    const dotIdx = key.indexOf(".");
    if (dotIdx !== -1) {
      const param = key.slice(0, dotIdx);
      const prop = key.slice(dotIdx + 1);
      if (!result[param] || typeof result[param] !== "object") {
        result[param] = {};
      }
      (result[param] as Record<string, unknown>)[prop] = values[key];
    } else {
      result[key] = values[key];
    }
  }
  return result;
}

function flattenInitValues(initValues: Record<string, unknown>): Record<string, unknown> {
  const flat: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(initValues)) {
    if (v && typeof v === "object" && !Array.isArray(v)) {
      for (const [subK, subV] of Object.entries(v as Record<string, unknown>)) {
        flat[`${k}.${subK}`] = subV;
      }
    } else {
      flat[k] = v;
    }
  }
  return flat;
}

function normalizeFetchedOptions(raw: Record<string, unknown>): Record<string, Option[]> {
  const flat: Record<string, Option[]> = {};
  for (const [key, val] of Object.entries(raw)) {
    if (Array.isArray(val)) {
      flat[key] = val as Option[];
    } else if (val && typeof val === "object") {
      for (const [subKey, subVal] of Object.entries(val as Record<string, unknown>)) {
        if (Array.isArray(subVal)) {
          flat[`${key}.${subKey}`] = subVal as Option[];
        }
      }
    }
  }
  return flat;
}

function applyOptionDefaults(
  prev: Record<string, unknown>,
  options: Record<string, Option[]>,
): Record<string, unknown> {
  const updated = { ...prev };
  for (const [fieldKey, opts] of Object.entries(options)) {
    if (opts.length > 0 && (updated[fieldKey] === "" || updated[fieldKey] === undefined)) {
      updated[fieldKey] = opts[0]!.value;
    }
  }
  return updated;
}

function resolveFieldFormat(prop: SchemaObj): string | undefined {
  return prop.format ?? prop.anyOf?.find((b) => b.format)?.format;
}

function resolveEnumValues(prop: SchemaObj): unknown[] | undefined {
  return prop.enum ?? prop.anyOf?.find((b) => b.enum)?.enum;
}

function parseFieldValue(propType: string, raw: string): string | number {
  if (propType === "number" || propType === "integer") {
    return raw === "" ? "" : Number(raw);
  }
  return raw;
}

function resolveSubmitLabel(submitLabel: string | undefined, mode: "create" | "edit"): string {
  return submitLabel ?? (mode === "edit" ? "Save" : "Create");
}

export const __test__ = {
  hasProps,
  resolveSchema,
  getDefaultValue,
  collectFields,
  buildInitValues,
  flattenInitValues,
  normalizeFetchedOptions,
  applyOptionDefaults,
  resolveFieldFormat,
  resolveEnumValues,
  parseFieldValue,
  resolveSubmitLabel,
};

function ConfiguringNodeComponent({ data }: NodeProps) {
  const d = data as unknown as ConfiguringNodeData;
  const { componentInfo, onConfirm, onCancel, initialValues, mode = "create" } = d;

  const { fields, optionFields } = collectFields(componentInfo.init);

  const [values, setValues] = useState<Record<string, unknown>>(() => {
    const flatInitial = initialValues ? flattenInitValues(initialValues) : {};
    const initial: Record<string, unknown> = {};
    for (const [key, prop] of Object.entries(fields)) {
      initial[key] = flatInitial[key] ?? getDefaultValue(prop);
    }
    return initial;
  });

  const [options, setOptions] = useState<Record<string, Option[]>>({});

  useEffect(() => {
    fetchOptions(componentInfo.type_)
      .then((raw) => {
        const flat = normalizeFetchedOptions(raw);
        setOptions(flat);
        setValues((prev) => applyOptionDefaults(prev, flat));
      })
      .catch(() => {});
  }, [componentInfo.type_]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onConfirm(buildInitValues(fields, values));
  };


  return (
    <div
      className={cn(
        "rounded-2xl border border-conduit/40 px-6 py-5 min-w-[320px] max-w-[400px]",
        "bg-gradient-to-b from-conduit/10 to-conduit/5",
        "backdrop-blur-xs bg-glass backdrop-saturate-150",
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-3 pb-4 border-b border-white/[0.06]">
        <span
          className="font-bold text-lg truncate"
          style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
        >
          Configure {componentInfo.type_}
        </span>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="nodrag nopan pt-4 flex flex-col gap-3">
        {Object.entries(fields).map(([key, prop]) => {
          const propType = prop.type ?? "string";

          if (propType === "boolean") {
            return (
              <label key={key} className="flex items-center gap-2.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={Boolean(values[key])}
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [key]: e.target.checked }))
                  }
                  className="w-4 h-4 rounded accent-conduit"
                />
                <span className="text-[13px] font-mono text-white/80">{key}</span>
              </label>
            );
          }

          // Dynamic config options dropdown
          if (options[key]) {
            const opts = options[key];
            return (
              <label key={key} className="flex flex-col gap-1">
                <span className="text-[12px] font-mono text-white/60">{key}</span>
                <Dropdown
                  value={String(values[key])}
                  options={opts}
                  onChange={(v) => setValues((prev) => ({ ...prev, [key]: v }))}
                />
              </label>
            );
          }

          // File path field — show filename + browse button
          const format = resolveFieldFormat(prop);
          if (format === "path") {
            const currentPath = String(values[key]);
            const displayName = currentPath ? currentPath.split(/[/\\]/).pop() : "No file selected";
            return (
              <div key={key} className="flex flex-col gap-1">
                <span className="text-[12px] font-mono text-white/60">{key}</span>
                <div className="flex items-center gap-2">
                  <span className="flex-1 truncate rounded-md border border-white/10 bg-white/5 px-2 py-1 text-sm text-white/40">
                    {displayName}
                  </span>
                  <button
                    type="button"
                    className="nodrag nopan shrink-0 cursor-pointer rounded-md border border-white/10 bg-white/5 px-3 py-1 text-sm text-white/60 hover:bg-white/10"
                    onClick={async () => {
                      if (window.__TAURI__) {
                        const { open } = await import("@tauri-apps/plugin-dialog");
                        const selected = await open({ multiple: false });
                        if (selected) {
                          setValues((v) => ({ ...v, [key]: selected }));
                        }
                      } else {
                        const input = document.createElement("input");
                        input.type = "file";
                        input.onchange = async () => {
                          const file = input.files?.[0];
                          if (!file) return;
                          const form = new FormData();
                          form.append("file", file);
                          const res = await fetch("/upload", { method: "POST", body: form });
                          if (res.ok) {
                            const { path } = await res.json();
                            setValues((v) => ({ ...v, [key]: path }));
                          }
                        };
                        input.click();
                      }
                    }}
                  >
                    Browse
                  </button>
                </div>
              </div>
            );
          }

          // Unwrap enum from anyOf (e.g. Literal[...] | None → anyOf with enum branch)
          const enumValues = resolveEnumValues(prop);
          if (enumValues) {
            return (
              <div key={key} className="flex flex-col gap-1">
                <span className="text-[12px] font-mono text-white/60">{key}</span>
                <Dropdown
                  value={String(values[key])}
                  options={enumValues.map((val: unknown) => ({ value: String(val), label: String(val) }))}
                  onChange={(v) => setValues((prev) => ({ ...prev, [key]: v }))}
                />
              </div>
            );
          }

          return (
            <label key={key} className="flex flex-col gap-1">
              <span className="text-[12px] font-mono text-white/60">{key}</span>
              <input
                type={propType === "number" || propType === "integer" ? "number" : "text"}
                step={propType === "number" ? "any" : undefined}
                value={String(values[key])}
                onChange={(e) => {
                  const raw = e.target.value;
                  const parsed = parseFieldValue(propType, raw);
                  setValues((v) => ({ ...v, [key]: parsed }));
                }}
                className={cn(
                  "rounded-lg px-3 py-1.5 text-[13px] font-mono",
                  "bg-white/[0.06] border border-white/[0.08] text-white/90",
                  "focus:outline-none focus:border-conduit/60",
                )}
              />
            </label>
          );
        })}

        {/* Buttons */}
        <div className="flex gap-2 pt-2">
          <button
            type="submit"
            className={cn(
              "flex-1 rounded-lg px-4 py-2 text-[13px] font-semibold",
              "bg-conduit/20 text-conduit hover:bg-conduit/30",
              "transition-colors cursor-pointer",
            )}
          >
            {resolveSubmitLabel(d.submitLabel, mode)}
          </button>
          <button
            type="button"
            onClick={onCancel}
            className={cn(
              "flex-1 rounded-lg px-4 py-2 text-[13px] font-semibold",
              "bg-white/[0.06] text-white/60 hover:bg-white/[0.10]",
              "transition-colors cursor-pointer",
            )}
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}

export const ConfiguringNode = memo(ConfiguringNodeComponent);
