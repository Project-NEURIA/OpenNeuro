import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SplashScreen } from "./SplashScreen";

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
});

