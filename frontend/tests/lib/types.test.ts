import { describe, expect, it } from "vitest";
import { categoryFromTags, parseSlotType } from "@/lib/types";

describe("types helpers", () => {
  it("derives category from the first io tag or defaults to conduit", () => {
    expect(
      categoryFromTags({
        io: ["source"],
        functionality: ["audio"],
        gpu: ["cpu"],
      }),
    ).toBe("source");

    expect(
      categoryFromTags({
        io: [],
        functionality: ["other"],
        gpu: ["cpu"],
      }),
    ).toBe("conduit");
  });

  it("parses optional sender and receiver slot types", () => {
    expect(parseSlotType("Sender[AudioFrame]")).toEqual({
      name: "AudioFrame",
      optional: false,
    });

    expect(parseSlotType("Receiver[Union[AudioFrame, NoneType]]")).toEqual({
      name: "AudioFrame",
      optional: true,
    });

    expect(parseSlotType("Union[A, B]")).toEqual({
      name: "A | B",
      optional: false,
    });

    expect(parseSlotType("Frame | None")).toEqual({
      name: "Frame",
      optional: true,
    });
  });
});


