import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { __test__, ConfiguringNode } from "@/components/graph/ConfiguringNode";
import * as api from "@/lib/api";

vi.mock("@/components/ui/Dropdown", () => ({
  Dropdown: ({
    value,
    options,
    onChange,
  }: {
    value: string;
    options: { value: string; label: string }[];
    onChange: (value: string) => void;
  }) => (
    <select
      data-testid={`dropdown-${options.map((option) => option.value).join("-")}`}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(async () => "C:/temp/model.bin"),
}));

describe("ConfiguringNode", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders fields, applies dynamic options, submits nested config, and cancels", async () => {
    vi.spyOn(api, "fetchOptions").mockResolvedValue({
      settings: {
        mode: [{ value: "slow", label: "Slow" }],
      },
      simple: [{ value: "opt", label: "Option" }],
      refSettings: {
        mode: [{ value: "safe", label: "Safe" }],
      },
      unionSettings: {
        level: [{ value: "high", label: "High" }],
      },
    });
    const onConfirm = vi.fn();
    const onCancel = vi.fn();

    render(
      <ConfiguringNode
        id="cfg"
        data={{
          componentInfo: {
            type_: "Builder",
            tags: { io: ["conduit"], functionality: ["misc"], gpu: ["cpu"] },
            init: {
              settings: {
                type: "object",
                options: { mode: {} },
                properties: {
                  enabled: { type: "boolean", default: true },
                  mode: { type: "string" },
                  threshold: { type: "number", default: 1.5 },
                },
              },
              refSettings: {
                $ref: "#/$defs/RefConfig",
                $defs: {
                  RefConfig: {
                    type: "object",
                    properties: {
                      mode: { type: "string" },
                    },
                  },
                },
              },
              unionSettings: {
                anyOf: [{ $ref: "#/$defs/UnionConfig" }],
                $defs: {
                  UnionConfig: {
                    type: "object",
                    properties: {
                      level: { type: "string" },
                    },
                  },
                },
              },
              simple: { type: "string" },
              choice: { anyOf: [{ enum: ["a", "b"] }] },
            },
            inputs: {},
            outputs: {},
            ui_inputs: {},
            ui_outputs: {},
          },
          initialValues: {
            settings: { threshold: 2 },
            choice: "a",
          },
          submitLabel: "Save",
          onConfirm,
          onCancel,
        }}
        selected={false}
        dragging={false}
        zIndex={1}
        type="configuring"
        isConnectable
        xPos={0}
        yPos={0}
      />,
    );

    await waitFor(() => expect(api.fetchOptions).toHaveBeenCalledWith("Builder"));

    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toBeChecked();
    fireEvent.click(checkbox);

    fireEvent.change(screen.getByDisplayValue("2"), { target: { value: "2.75" } });
    fireEvent.change(screen.getByTestId("dropdown-slow"), { target: { value: "slow" } });
    fireEvent.change(screen.getByTestId("dropdown-safe"), { target: { value: "safe" } });
    fireEvent.change(screen.getByTestId("dropdown-high"), { target: { value: "high" } });
    fireEvent.change(screen.getByTestId("dropdown-opt"), { target: { value: "opt" } });
    fireEvent.change(screen.getByTestId("dropdown-a-b"), { target: { value: "b" } });

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onConfirm).toHaveBeenCalledWith({
      settings: {
        enabled: false,
        mode: "slow",
        threshold: 2.75,
      },
      refSettings: {
        mode: "safe",
      },
      unionSettings: {
        level: "high",
      },
      simple: "opt",
      choice: "b",
    });

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalled();
  });

  it("handles tauri and browser file selection for path fields", async () => {
    vi.spyOn(api, "fetchOptions").mockResolvedValue({});

    const tauriWindow = window as Window & { __TAURI__?: unknown };
    tauriWindow.__TAURI__ = {};
    const { rerender } = render(
      <ConfiguringNode
        id="cfg"
        data={{
          componentInfo: {
            type_: "Uploader",
            tags: { io: ["source"], functionality: ["other"], gpu: ["cpu"] },
            init: {
              file: { type: "string", format: "path" },
            },
            inputs: {},
            outputs: {},
            ui_inputs: {},
            ui_outputs: {},
          },
          onConfirm: vi.fn(),
          onCancel: vi.fn(),
        }}
        selected={false}
        dragging={false}
        zIndex={1}
        type="configuring"
        isConnectable
        xPos={0}
        yPos={0}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Browse" }));
    await waitFor(() => expect(screen.getByText("model.bin")).toBeInTheDocument());

    delete tauriWindow.__TAURI__;
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ path: "C:/uploaded/demo.txt" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const realCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation(((tagName: string) => {
      const element = realCreateElement(tagName);
      if (tagName === "input") {
        const input = element as HTMLInputElement;
        Object.defineProperty(input, "files", {
          configurable: true,
          value: [new File(["demo"], "demo.txt")],
        });
        input.click = () => {
          input.onchange?.(new Event("change"));
        };
        return input;
      }
      return element;
    }) as typeof document.createElement);

    rerender(
      <ConfiguringNode
        id="cfg"
        data={{
          componentInfo: {
            type_: "Uploader",
            tags: { io: ["source"], functionality: ["other"], gpu: ["cpu"] },
            init: {
              file: { type: "string", format: "path" },
            },
            inputs: {},
            outputs: {},
            ui_inputs: {},
            ui_outputs: {},
          },
          onConfirm: vi.fn(),
          onCancel: vi.fn(),
        }}
        selected={false}
        dragging={false}
        zIndex={1}
        type="configuring"
        isConnectable
        xPos={0}
        yPos={0}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Browse" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/upload", expect.any(Object)));
    await waitFor(() => expect(screen.getByText("demo.txt")).toBeInTheDocument());
  });

  it("covers helper branches and default create mode behavior", async () => {
    expect(__test__.hasProps({ type: "object", properties: { a: { type: "string" } } })).toBe(true);
    expect(__test__.resolveSchema({
      anyOf: [{ type: "string" }, { type: "object", properties: { a: { type: "string" } } }],
    })).toEqual({ type: "object", properties: { a: { type: "string" } } });
    expect(__test__.resolveSchema({
      anyOf: [{ $ref: "#/$defs/Missing" }],
      $defs: {},
    })).toBeNull();
    expect(__test__.resolveSchema({
      $ref: "#/$defs/Missing",
      $defs: {},
    })).toBeNull();
    expect(__test__.getDefaultValue({ type: "boolean" })).toBe(false);
    expect(__test__.getDefaultValue({ type: "string" })).toBe("");
    expect(__test__.collectFields({
      skip: null,
      simple: { type: "string" },
      nested: { type: "object", properties: { name: { type: "string" } }, options: { name: {} } },
    })).toEqual({
      fields: {
        simple: { type: "string" },
        "nested.name": { type: "string" },
      },
      optionFields: new Set(["simple", "nested.name"]),
    });
    expect(__test__.buildInitValues(
      { plain: { type: "string" }, "nested.value": { type: "integer" } },
      { plain: "x", "nested.value": 2 },
    )).toEqual({ plain: "x", nested: { value: 2 } });
    expect(__test__.flattenInitValues({ nested: { value: 2 }, plain: "x" })).toEqual({
      "nested.value": 2,
      plain: "x",
    });
    expect(__test__.normalizeFetchedOptions({
      simple: [{ value: "a", label: "A" }],
      nested: {
        mode: [{ value: "b", label: "B" }],
        ignored: { nope: true },
      },
      primitive: "skip",
    })).toEqual({
      simple: [{ value: "a", label: "A" }],
      "nested.mode": [{ value: "b", label: "B" }],
    });
    expect(__test__.applyOptionDefaults(
      { "nested.mode": "", simple: "keep", empty: undefined },
      {
        "nested.mode": [{ value: "fast", label: "Fast" }],
        simple: [{ value: "override", label: "Override" }],
        empty: [{ value: "filled", label: "Filled" }],
      },
    )).toEqual({
      "nested.mode": "fast",
      simple: "keep",
      empty: "filled",
    });
    expect(__test__.resolveFieldFormat({ anyOf: [{ format: "path" }] })).toBe("path");
    expect(__test__.resolveEnumValues({ anyOf: [{ enum: ["x", "y"] }] })).toEqual(["x", "y"]);
    expect(__test__.parseFieldValue("integer", "")).toBe("");
    expect(__test__.parseFieldValue("number", "2.5")).toBe(2.5);
    expect(__test__.parseFieldValue("string", "raw")).toBe("raw");
    expect(__test__.resolveSubmitLabel(undefined, "edit")).toBe("Save");
    expect(__test__.resolveSubmitLabel(undefined, "create")).toBe("Create");

    vi.spyOn(api, "fetchOptions").mockRejectedValue(new Error("no options"));
    const fetchMock = vi.fn().mockResolvedValue({ ok: false });
    vi.stubGlobal("fetch", fetchMock);

    const realCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation(((tagName: string) => {
      const element = realCreateElement(tagName);
      if (tagName === "input") {
        const input = element as HTMLInputElement;
        let clicked = false;
        Object.defineProperty(input, "files", {
          configurable: true,
          get: () => (clicked ? [] : null),
        });
        input.click = () => {
          clicked = true;
          input.onchange?.(new Event("change"));
        };
        return input;
      }
      return element;
    }) as typeof document.createElement);

    render(
      <ConfiguringNode
        id="cfg"
        data={{
          componentInfo: {
            type_: "FallbackBuilder",
            tags: { io: ["conduit"], functionality: ["misc"], gpu: ["cpu"] },
            init: {
              integerValue: { type: "integer" },
              plainText: { type: "string" },
              pathFromAnyOf: { anyOf: [{ format: "path" }] },
              enumDirect: { enum: ["x", "y"] },
            },
            inputs: {},
            outputs: {},
            ui_inputs: {},
            ui_outputs: {},
          },
          onConfirm: vi.fn(),
          onCancel: vi.fn(),
        }}
        selected={false}
        dragging={false}
        zIndex={1}
        type="configuring"
        isConnectable
        xPos={0}
        yPos={0}
      />,
    );

    await waitFor(() => expect(api.fetchOptions).toHaveBeenCalledWith("FallbackBuilder"));
    expect(screen.getByRole("button", { name: "Create" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "typed" } });
    fireEvent.change(screen.getByDisplayValue(""), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Browse" }));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("covers resolved refs, explicit defaults, ignored option payloads, and failed uploads", async () => {
    expect(__test__.resolveSchema({
      $ref: "#/$defs/RealConfig",
      $defs: {
        RealConfig: {
          type: "object",
          properties: {
            mode: { type: "string" },
          },
        },
      },
    })).toEqual({
      type: "object",
      properties: {
        mode: { type: "string" },
      },
    });
    expect(__test__.getDefaultValue({ type: "string", default: "preset" })).toBe("preset");
    expect(__test__.flattenInitValues({ list: [1, 2], plain: "x" })).toEqual({
      list: [1, 2],
      plain: "x",
    });

    vi.spyOn(api, "fetchOptions").mockResolvedValue({
      settings: {
        mode: [{ value: "fast", label: "Fast" }],
        ignored: { nested: true } as never,
      },
      simple: [{ value: "override", label: "Override" }],
    });

    const fetchMock = vi.fn().mockResolvedValue({ ok: false });
    vi.stubGlobal("fetch", fetchMock);

    const realCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation(((tagName: string) => {
      const element = realCreateElement(tagName);
      if (tagName === "input") {
        const input = element as HTMLInputElement;
        Object.defineProperty(input, "files", {
          configurable: true,
          value: [new File(["demo"], "demo.bin")],
        });
        input.click = () => {
          input.onchange?.(new Event("change"));
        };
        return input;
      }
      return element;
    }) as typeof document.createElement);

    const onConfirm = vi.fn();
    render(
      <ConfiguringNode
        id="cfg"
        data={{
          componentInfo: {
            type_: "ConfiguredUploader",
            tags: { io: ["conduit"], functionality: ["misc"], gpu: ["cpu"] },
            init: {
              settings: {
                $ref: "#/$defs/RealConfig",
                $defs: {
                  RealConfig: {
                    type: "object",
                    properties: {
                      mode: { type: "string", default: "manual" },
                    },
                  },
                },
              },
              simple: { type: "string", default: "keep" },
              choice: { anyOf: [{ type: "null" }, { enum: ["left", "right"] }] },
              integerValue: { type: "integer" },
              file: { type: "string", format: "path" },
            },
            inputs: {},
            outputs: {},
            ui_inputs: {},
            ui_outputs: {},
          },
          initialValues: {
            settings: { mode: "manual" },
            simple: "keep",
          },
          onConfirm,
          onCancel: vi.fn(),
        }}
        selected={false}
        dragging={false}
        zIndex={1}
        type="configuring"
        isConnectable
        xPos={0}
        yPos={0}
      />,
    );

    await waitFor(() => expect(api.fetchOptions).toHaveBeenCalledWith("ConfiguredUploader"));
    fireEvent.click(screen.getByRole("button", { name: "Browse" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/upload", expect.any(Object)));
    expect(screen.getByText("No file selected")).toBeInTheDocument();

    fireEvent.change(screen.getByTestId("dropdown-left-right"), { target: { value: "right" } });
    fireEvent.change(screen.getByDisplayValue(""), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Create" }));

    expect(onConfirm).toHaveBeenCalledWith({
      settings: { mode: "manual" },
      simple: "keep",
      choice: "right",
      integerValue: "",
      file: "",
    });
  });

  it("keeps path empty when tauri dialog is canceled", async () => {
    vi.spyOn(api, "fetchOptions").mockResolvedValue({});

    const dialog = await import("@tauri-apps/plugin-dialog");
    vi.mocked(dialog.open).mockResolvedValueOnce(null);

    const tauriWindow = window as Window & { __TAURI__?: unknown };
    tauriWindow.__TAURI__ = {};

    render(
      <ConfiguringNode
        id="cfg"
        data={{
          componentInfo: {
            type_: "Uploader",
            tags: { io: ["source"], functionality: ["other"], gpu: ["cpu"] },
            init: {
              file: { type: "string", format: "path" },
            },
            inputs: {},
            outputs: {},
            ui_inputs: {},
            ui_outputs: {},
          },
          onConfirm: vi.fn(),
          onCancel: vi.fn(),
        }}
        selected={false}
        dragging={false}
        zIndex={1}
        type="configuring"
        isConnectable
        xPos={0}
        yPos={0}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Browse" }));
    await waitFor(() => expect(dialog.open).toHaveBeenCalled());
    expect(screen.getByText("No file selected")).toBeInTheDocument();

    delete tauriWindow.__TAURI__;
  });
});


