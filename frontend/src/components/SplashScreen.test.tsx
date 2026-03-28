import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { __test__, SplashScreen } from "./SplashScreen";

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

  it("short-circuits splash animation when svg is missing or already animated", () => {
    const animated = { current: false };
    expect(__test__.animateSplashSvg(null, animated)).toBeUndefined();
    expect(animated.current).toBe(false);

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    animated.current = true;
    __test__.animateSplashSvg(svg, animated);
    expect(svg.querySelectorAll("text")).toHaveLength(0);
  });
});

