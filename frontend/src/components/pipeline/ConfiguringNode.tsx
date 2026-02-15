import { memo, useState } from "react";
import { type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { ComponentInfo } from "@/lib/types";

interface ConfiguringNodeData {
  componentInfo: ComponentInfo;
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
};

type ResolvedSchema = { properties: Record<string, SchemaObj>; title?: string };

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
function collectFields(init: Record<string, unknown>): Record<string, SchemaObj> {
  const fields: Record<string, SchemaObj> = {};
  for (const [paramName, rawSchema] of Object.entries(init)) {
    if (!rawSchema || typeof rawSchema !== "object") continue;
    const schema = rawSchema as SchemaObj;
    const resolved = resolveSchema(schema);
    if (resolved) {
      // Object schema — each property becomes a field
      for (const [prop, propSchema] of Object.entries(resolved.properties)) {
        fields[`${paramName}.${prop}`] = propSchema;
      }
    } else {
      // Simple type — the param itself is a field
      fields[paramName] = schema;
    }
  }
  return fields;
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

function ConfiguringNodeComponent({ data }: NodeProps) {
  const d = data as unknown as ConfiguringNodeData;
  const { componentInfo, onConfirm, onCancel } = d;

  const fields = collectFields(componentInfo.init);

  const [values, setValues] = useState<Record<string, unknown>>(() => {
    const initial: Record<string, unknown> = {};
    for (const [key, prop] of Object.entries(fields)) {
      initial[key] = getDefaultValue(prop);
    }
    return initial;
  });

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
          Configure {componentInfo.name}
        </span>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="pt-4 flex flex-col gap-3">
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

          if (prop.enum) {
            return (
              <label key={key} className="flex flex-col gap-1">
                <span className="text-[12px] font-mono text-white/60">{key}</span>
                <select
                  value={String(values[key] ?? "")}
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [key]: e.target.value }))
                  }
                  className={cn(
                    "rounded-lg px-3 py-1.5 text-[13px] font-mono",
                    "bg-white/[0.06] border border-white/[0.08] text-white/90",
                    "focus:outline-none focus:border-conduit/60",
                  )}
                >
                  {prop.enum.map((val) => (
                    <option key={String(val)} value={String(val)}>
                      {String(val)}
                    </option>
                  ))}
                </select>
              </label>
            );
          }

          return (
            <label key={key} className="flex flex-col gap-1">
              <span className="text-[12px] font-mono text-white/60">{key}</span>
              <input
                type={propType === "number" || propType === "integer" ? "number" : "text"}
                step={propType === "number" ? "any" : undefined}
                value={String(values[key] ?? "")}
                onChange={(e) => {
                  const raw = e.target.value;
                  const parsed =
                    propType === "number" || propType === "integer"
                      ? raw === "" ? "" : Number(raw)
                      : raw;
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
            Create
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
