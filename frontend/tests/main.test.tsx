import { describe, expect, it, vi } from "vitest";

describe("main", () => {
  it("mounts App into the root element", async () => {
    const render = vi.fn();
    vi.doMock("react-dom/client", () => ({
      createRoot: vi.fn(() => ({ render })),
    }));
    vi.doMock("@/App", () => ({
      default: () => <div>App</div>,
    }));

    document.body.innerHTML = '<div id="root"></div>';
    await import("@/main");

    expect(render).toHaveBeenCalledTimes(1);
  });
});

