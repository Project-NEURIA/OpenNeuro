import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConfiguringNode } from "@/components/graph/ConfiguringNode";
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

  it("renders nested and simple fields, applies fetched options, and submits nested config", async () => {
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
                options: {
                  mode: {},
                },
                properties: {
                  enabled: { type: "boolean", default: true },
                  mode: { type: "string" },
                  threshold: { type: "number", default: 1.5 },
                  notes: { type: "string", default: "hello" },
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
    expect(screen.getByRole("checkbox")).toBeChecked();

    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.change(screen.getByDisplayValue("2"), { target: { value: "2.75" } });
    fireEvent.blur(screen.getByDisplayValue("2.75"));
    fireEvent.change(screen.getByTestId("dropdown-slow"), { target: { value: "slow" } });
    fireEvent.change(screen.getByTestId("dropdown-safe"), { target: { value: "safe" } });
    fireEvent.change(screen.getByTestId("dropdown-high"), { target: { value: "high" } });
    fireEvent.change(screen.getByTestId("dropdown-opt"), { target: { value: "opt" } });
    fireEvent.change(screen.getByTestId("dropdown-a-b"), { target: { value: "b" } });
    const notesField = screen.getByDisplayValue("hello");
    Object.defineProperty(notesField, "scrollHeight", { configurable: true, value: 48 });
    fireEvent.change(notesField, { target: { value: "updated notes" } });

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onConfirm).toHaveBeenCalledWith({
      settings: {
        enabled: false,
        mode: "slow",
        threshold: 2.75,
        notes: "updated notes",
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

  it("handles tauri and browser path selection flows", async () => {
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

  it("leaves path values unchanged when tauri or browser file selection is cancelled", async () => {
    vi.spyOn(api, "fetchOptions").mockResolvedValue({});
    const tauriWindow = window as Window & { __TAURI__?: unknown };
    tauriWindow.__TAURI__ = {};

    const dialog = await import("@tauri-apps/plugin-dialog");
    vi.mocked(dialog.open).mockResolvedValueOnce(null);

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
    expect(screen.getByText("No file selected")).toBeInTheDocument();

    delete tauriWindow.__TAURI__;
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const realCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation(((tagName: string) => {
      const element = realCreateElement(tagName);
      if (tagName === "input") {
        const input = element as HTMLInputElement;
        Object.defineProperty(input, "files", {
          configurable: true,
          value: [],
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
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.getByText("No file selected")).toBeInTheDocument();
  });

  it("keeps defaults when options fail to load and leaves failed uploads unchanged", async () => {
    vi.spyOn(api, "fetchOptions").mockRejectedValue(new Error("no options"));
    const onConfirm = vi.fn();
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

    render(
      <ConfiguringNode
        id="cfg"
        data={{
          componentInfo: {
            type_: "ConfiguredUploader",
            tags: { io: ["conduit"], functionality: ["misc"], gpu: ["cpu"] },
            init: {
              integerValue: { type: "integer" },
              plainText: { type: "string", default: "keep" },
              choice: { anyOf: [{ type: "null" }, { enum: ["left", "right"] }] },
              file: { type: "string", format: "path" },
            },
            inputs: {},
            outputs: {},
            ui_inputs: {},
            ui_outputs: {},
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

    fireEvent.change(screen.getByTestId("dropdown-left-right"), { target: { value: "right" } });
    fireEvent.click(screen.getByRole("button", { name: "Browse" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/upload", expect.any(Object)));
    expect(screen.getByText("No file selected")).toBeInTheDocument();

    fireEvent.change(screen.getByDisplayValue(""), { target: { value: "12" } });
    fireEvent.blur(screen.getByDisplayValue("12"));
    fireEvent.click(screen.getByRole("button", { name: "Create" }));

    expect(onConfirm).toHaveBeenCalledWith({
      integerValue: 12,
      plainText: "keep",
      choice: "right",
      file: "",
    });
  });

  it("covers edit-mode defaults, direct anyOf objects, and ignored invalid numeric input", async () => {
    vi.spyOn(api, "fetchOptions").mockResolvedValue({
      nestedChoice: {
        value: [{ value: "preset", label: "Preset" }],
      },
      plainChoice: [],
      ignoredObject: {
        notAnArray: { label: "skip" },
      },
    });
    const onConfirm = vi.fn();

    render(
      <ConfiguringNode
        id="cfg"
        data={{
          componentInfo: {
            type_: "Editor",
            tags: { io: ["conduit"], functionality: ["misc"], gpu: ["cpu"] },
            init: {
              nestedChoice: {
                anyOf: [
                  {
                    type: "object",
                    properties: {
                      value: { type: "string" },
                    },
                  },
                ],
              },
              refChoice: {
                $ref: "#/$defs/RefChoice",
                $defs: {
                  RefChoice: {
                    type: "object",
                    properties: {
                      path: { type: "string" },
                    },
                  },
                },
              },
              integerValue: { type: "integer" },
              booleanValue: { type: "boolean" },
              nullableField: null as never,
            },
            inputs: {},
            outputs: {},
            ui_inputs: {},
            ui_outputs: {},
          },
          mode: "edit",
          initialValues: {
            refChoice: { path: "existing" },
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

    await waitFor(() => expect(api.fetchOptions).toHaveBeenCalledWith("Editor"));
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.change(screen.getByDisplayValue(""), { target: { value: "12x" } });
    expect(screen.getByDisplayValue("")).toBeInTheDocument();
    fireEvent.change(screen.getByDisplayValue(""), { target: { value: "-" } });
    fireEvent.blur(screen.getByDisplayValue("-"));
    fireEvent.change(screen.getByTestId("dropdown-preset"), { target: { value: "preset" } });

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onConfirm).toHaveBeenCalledWith({
      nestedChoice: { value: "preset" },
      refChoice: { path: "existing" },
      integerValue: "-",
      booleanValue: true,
    });
  });
});
