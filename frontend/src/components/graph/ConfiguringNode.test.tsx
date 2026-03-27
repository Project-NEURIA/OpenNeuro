import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConfiguringNode } from "./ConfiguringNode";
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
                properties: {
                  enabled: { type: "boolean", default: true },
                  mode: { type: "string" },
                  threshold: { type: "number", default: 1.5 },
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
    fireEvent.change(screen.getByTestId("dropdown-opt"), { target: { value: "opt" } });
    fireEvent.change(screen.getByTestId("dropdown-a-b"), { target: { value: "b" } });

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onConfirm).toHaveBeenCalledWith({
      settings: {
        enabled: false,
        mode: "slow",
        threshold: 2.75,
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
    expect(screen.getByText("demo.txt")).toBeInTheDocument();
  });
});

