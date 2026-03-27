import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EnvEditor } from "./EnvEditor";
import * as api from "@/lib/api";

describe("EnvEditor", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("loads env entries, updates values, toggles visibility, removes rows, saves, and closes", async () => {
    vi.spyOn(api, "fetchEnv").mockResolvedValue("# comment\nALPHA=1\nBROKEN\nBETA = 2");
    const putEnv = vi.spyOn(api, "putEnv").mockResolvedValue();
    const onClose = vi.fn();

    const { container } = render(<EnvEditor onClose={onClose} />);

    await act(async () => {});

    const keyInputs = screen.getAllByPlaceholderText("KEY") as HTMLInputElement[];
    const valueInputs = screen.getAllByPlaceholderText("value") as HTMLInputElement[];
    expect(keyInputs.map((input) => input.value)).toEqual(["ALPHA", "BROKEN", "BETA"]);
    expect(valueInputs.map((input) => input.value)).toEqual(["1", "", "2"]);

    const firstRow = keyInputs[0]!.parentElement!;
    fireEvent.click(within(firstRow).getAllByRole("button")[0]!);
    expect((screen.getAllByPlaceholderText("value")[0] as HTMLInputElement).type).toBe("text");

    fireEvent.change(valueInputs[1]!, { target: { value: "filled" } });
    fireEvent.click(screen.getByRole("button", { name: /add/i }));
    expect(screen.getAllByPlaceholderText("KEY")).toHaveLength(4);

    const secondRow = (screen.getAllByPlaceholderText("KEY")[1] as HTMLInputElement).parentElement!;
    fireEvent.click(within(secondRow).getAllByRole("button")[1]!);
    expect(screen.getAllByPlaceholderText("KEY")).toHaveLength(3);

    fireEvent.click(screen.getByRole("button", { name: /save/i }));
    await waitFor(() =>
      expect(putEnv).toHaveBeenCalledWith("ALPHA=1\nBETA=2"),
    );
    expect(screen.getByText("Saved")).toBeInTheDocument();

    fireEvent.click(container.querySelector("button")!);
    expect(onClose).toHaveBeenCalled();
  });

  it("falls back to one blank row when env is empty", async () => {
    vi.spyOn(api, "fetchEnv").mockResolvedValue("");
    vi.spyOn(api, "putEnv").mockResolvedValue();

    render(<EnvEditor onClose={() => {}} />);

    await act(async () => {});
    expect((screen.getByPlaceholderText("KEY") as HTMLInputElement).value).toBe("");
  });
});
