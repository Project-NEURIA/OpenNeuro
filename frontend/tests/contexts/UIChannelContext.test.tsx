import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { UIChannelProvider, useUIChannel } from "@/contexts/UIChannelContext";
import * as channelHook from "@/hooks/useUIChannel";

function Consumer() {
  const value = useUIChannel();
  return <div>{String(value.connected)}</div>;
}

describe("UIChannelContext", () => {
  it("provides the channel manager", () => {
    vi.spyOn(channelHook, "useUIChannelManager").mockReturnValue({
      connected: true,
      sendUIInput: vi.fn(),
      subscribe: vi.fn(() => vi.fn()),
    });

    render(
      <UIChannelProvider>
        <Consumer />
      </UIChannelProvider>,
    );

    expect(screen.getByText("true")).toBeInTheDocument();
  });

  it("throws when used outside the provider", () => {
    expect(() => render(<Consumer />)).toThrow(
      "useUIChannel must be used within UIChannelProvider",
    );
  });
});


