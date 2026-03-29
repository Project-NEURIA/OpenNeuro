import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Dropdown } from "@/components/ui/Dropdown";

describe("Dropdown", () => {
  it("opens, selects an option, and closes on outside click", () => {
    const onChange = vi.fn();
    render(
      <Dropdown
        value=""
        options={[
          { value: "one", label: "One" },
          { value: "two", label: "Two" },
        ]}
        onChange={onChange}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /select/i }));
    fireEvent.click(screen.getByRole("button", { name: "Two" }));
    expect(onChange).toHaveBeenCalledWith("two");

    fireEvent.click(screen.getByRole("button", { name: /select/i }));
    expect(screen.getByRole("button", { name: "One" })).toBeInTheDocument();
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole("button", { name: "One" })).not.toBeInTheDocument();
  });

  it("shows the selected label", () => {
    const onChange = vi.fn();
    render(
      <Dropdown
        value="one"
        options={[{ value: "one", label: "One" }]}
        onChange={onChange}
        placeholder="Pick one"
      />,
    );
    expect(screen.getByRole("button", { name: /one/i })).toHaveTextContent("One");
  });

  it("stops pointerdown propagation on trigger and option buttons", () => {
    const onChange = vi.fn();
    const onPointerDown = vi.fn();
    render(
      <div onPointerDown={onPointerDown}>
        <Dropdown
          value=""
          options={[
            { value: "one", label: "One" },
            { value: "two", label: "Two" },
          ]}
          onChange={onChange}
        />
      </div>,
    );

    fireEvent.pointerDown(screen.getByRole("button", { name: /select/i }));
    expect(onPointerDown).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /select/i }));
    fireEvent.pointerDown(screen.getByRole("button", { name: "One" }));
    expect(onPointerDown).not.toHaveBeenCalled();
  });

  it("keeps the menu open for inside mousedown and marks the selected option", () => {
    const onChange = vi.fn();
    render(
      <Dropdown
        value="one"
        options={[
          { value: "one", label: "One" },
          { value: "two", label: "Two" },
        ]}
        onChange={onChange}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /one/i }));
    const selectedOption = screen.getAllByRole("button", { name: "One" })[1]!;
    fireEvent.mouseDown(selectedOption);
    expect(selectedOption).toHaveClass("bg-conduit/20");
  });
});

