import { describe, expect, it, vi } from "vitest";
import { formatBytes, formatCount, formatUptime } from "./format";

describe("format", () => {
  it("formats counts across thresholds", () => {
    expect(formatCount(12)).toBe("12");
    expect(formatCount(1_500)).toBe("1.5k");
    expect(formatCount(2_500_000)).toBe("2.5M");
  });

  it("formats bytes across thresholds", () => {
    expect(formatBytes(12)).toBe("12 B");
    expect(formatBytes(2_048)).toBe("2.0 KB");
    expect(formatBytes(2_097_152)).toBe("2.0 MB");
    expect(formatBytes(2_147_483_648)).toBe("2.0 GB");
  });

  it("formats uptime for missing, seconds, minutes, and hours", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-27T12:00:00Z"));

    expect(formatUptime(null)).toBe("--");
    expect(formatUptime(Math.floor(Date.now() / 1000) - 30)).toBe("30s");
    expect(formatUptime(Math.floor(Date.now() / 1000) - 90)).toBe("1m 30s");
    expect(formatUptime(Math.floor(Date.now() / 1000) - 3_900)).toBe("1h 5m");

    vi.useRealTimers();
  });
});

