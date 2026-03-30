import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SplashScreen } from "@/components/SplashScreen";

describe("SplashScreen", () => {
  it("renders the loading status and draws svg letters once", () => {
    const { container, rerender } = render(<SplashScreen status="Connecting..." />);

    expect(screen.getByText("Connecting...")).toBeInTheDocument();
    const svg = container.querySelector("svg")!;
    expect(svg.querySelectorAll("text")).toHaveLength("OpenNeuro".length * 2);

    rerender(<SplashScreen status="Still connecting" />);
    expect(screen.getByText("Still connecting")).toBeInTheDocument();
    expect(svg.querySelectorAll("text")).toHaveLength("OpenNeuro".length * 2);
  });

  it("renders the drag region and progress bar chrome", () => {
    const { container } = render(<SplashScreen status="Booting..." />);

    expect(screen.getByText("Booting...")).toBeInTheDocument();
    expect(container.querySelector("[data-tauri-drag-region='true']")).toBeTruthy();
    expect(container.querySelector("svg#logo")).toBeTruthy();
    expect(container.querySelector(".animate-pulse")).toBeTruthy();
  });

  it("does not duplicate the svg lettering after rerenders", () => {
    const { container, rerender } = render(<SplashScreen status="Loading..." />);

    const svg = container.querySelector("svg")!;
    expect(svg.querySelectorAll("text")).toHaveLength("OpenNeuro".length * 2);

    rerender(<SplashScreen status="Still loading..." />);
    rerender(<SplashScreen status="Almost there..." />);

    expect(svg.querySelectorAll("text")).toHaveLength("OpenNeuro".length * 2);
    expect(screen.getByText("Almost there...")).toBeInTheDocument();
  });
});


