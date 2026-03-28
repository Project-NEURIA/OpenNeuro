import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  closeProject,
  createEdge,
  createNode,
  createProject,
  createSubgraph,
  deleteEdge,
  deleteNode,
  deleteProject,
  fetchComponentLogs,
  fetchComponents,
  fetchCurrentProject,
  fetchEdges,
  fetchEnv,
  fetchIsSubtype,
  fetchIsType,
  fetchNodes,
  fetchOptions,
  fetchProjects,
  projectThumbnailUrl,
  putEnv,
  saveGraph,
  startAll,
  startProject,
  stopAll,
  ungroupNode,
  updateNode,
  updateNodeInitArgs,
} from "./api";

type MockResponseInit = {
  ok?: boolean;
  status?: number;
  json?: unknown;
  text?: string;
  headers?: Record<string, string>;
};

function mockResponse(init: MockResponseInit = {}) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    json: vi.fn(async () => init.json),
    text: vi.fn(async () => init.text ?? ""),
    headers: {
      get: vi.fn((name: string) => init.headers?.[name.toLowerCase()] ?? null),
    },
  } as unknown as Response;
}

describe("api", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.stubGlobal("fetch", vi.fn());
  });

  it("covers component endpoints", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock
      .mockResolvedValueOnce(mockResponse({ json: [{ type_: "Mic" }] }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 500 }))
      .mockResolvedValueOnce(mockResponse({ json: true }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 400 }))
      .mockResolvedValueOnce(mockResponse({ json: false }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 404 }));

    await expect(fetchComponents()).resolves.toEqual([{ type_: "Mic" }]);
    await expect(fetchComponents()).rejects.toThrow("Fetch components failed: 500");
    await expect(fetchIsType("Dog")).resolves.toBe(true);
    await expect(fetchIsType("Dog")).rejects.toThrow("is-type failed: 400");
    await expect(fetchIsSubtype("Dog", "Animal")).resolves.toBe(false);
    await expect(fetchIsSubtype("Dog", "Animal")).rejects.toThrow("is-subtype failed: 404");

    expect(fetchMock).toHaveBeenNthCalledWith(2, "/component");
    expect(fetchMock).toHaveBeenNthCalledWith(3, "/component/is-type?name=Dog");
    expect(fetchMock).toHaveBeenNthCalledWith(5, "/component/is-subtype?sub=Dog&sup=Animal");
  });

  it("covers graph endpoints", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock
      .mockResolvedValueOnce(mockResponse({ json: [{ id: "n1" }] }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 502 }))
      .mockResolvedValueOnce(mockResponse({ json: [{ source_node: "a" }] }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 503 }))
      .mockResolvedValueOnce(mockResponse({ json: { id: "new-node" } }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 500 }))
      .mockResolvedValueOnce(mockResponse())
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 409 }))
      .mockResolvedValueOnce(mockResponse({ json: { id: "patched" } }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 422 }))
      .mockResolvedValueOnce(mockResponse())
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 404 }))
      .mockResolvedValueOnce(mockResponse({ json: { source_node: "a" } }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 400 }))
      .mockResolvedValueOnce(mockResponse())
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 401 }))
      .mockResolvedValueOnce(mockResponse({ json: { id: "group" } }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 400 }))
      .mockResolvedValueOnce(mockResponse({ json: [{ id: "ungrouped" }] }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 500 }))
      .mockResolvedValueOnce(mockResponse())
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 503 }))
      .mockResolvedValueOnce(mockResponse())
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 504 }))
      .mockResolvedValueOnce(mockResponse())
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 500 }));

    await expect(fetchNodes()).resolves.toEqual([{ id: "n1" }]);
    await expect(fetchNodes()).rejects.toThrow("Fetch nodes failed: 502");
    await expect(fetchEdges()).resolves.toEqual([{ source_node: "a" }]);
    await expect(fetchEdges()).rejects.toThrow("Fetch edges failed: 503");
    await expect(createNode("Mic", { gain: 2 })).resolves.toEqual({ id: "new-node" });
    await expect(createNode("Mic")).rejects.toThrow("Create node failed: 500");
    await expect(updateNode("n1", { x: 1, y: 2 })).resolves.toBeUndefined();
    await expect(updateNode("n1", { x: 1 })).rejects.toThrow("Update node failed: 409");
    await expect(updateNodeInitArgs("n1", { gain: 1 })).resolves.toEqual({ id: "patched" });
    await expect(updateNodeInitArgs("n1", { gain: 2 })).rejects.toThrow("Update node init args failed: 422");
    await expect(deleteNode("n1")).resolves.toBeUndefined();
    await expect(deleteNode("n1")).rejects.toThrow("Delete node failed: 404");
    await expect(createEdge("a", "out", "b", "in")).resolves.toEqual({ source_node: "a" });
    await expect(createEdge("a", "out", "b", "in")).rejects.toThrow("Create edge failed: 400");
    await expect(deleteEdge("a", "out", "b", "in")).resolves.toBeUndefined();
    await expect(deleteEdge("a", "out", "b", "in")).rejects.toThrow("Delete edge failed: 401");
    await expect(createSubgraph(["a", "b"], "Combo")).resolves.toEqual({ id: "group" });
    await expect(createSubgraph(["a"])).rejects.toThrow("Create subgraph failed: 400");
    await expect(ungroupNode("group")).resolves.toEqual([{ id: "ungrouped" }]);
    await expect(ungroupNode("group")).rejects.toThrow("Ungroup failed: 500");
    await expect(startAll()).resolves.toBeUndefined();
    await expect(startAll()).rejects.toThrow("Start failed: 503");
    await expect(stopAll()).resolves.toBeUndefined();
    await expect(stopAll()).rejects.toThrow("Stop failed: 504");
    await expect(saveGraph()).resolves.toBeUndefined();
    await expect(saveGraph()).rejects.toThrow("Save graph failed: 500");

    expect(fetchMock).toHaveBeenCalledWith("/graph/save", { method: "POST" });
  });

  it("covers project and env endpoints", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock
      .mockResolvedValueOnce(mockResponse({ json: [{ name: "Alpha", has_thumbnail: true }] }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 500 }))
      .mockResolvedValueOnce(mockResponse())
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 400 }))
      .mockResolvedValueOnce(mockResponse())
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 404 }))
      .mockResolvedValueOnce(mockResponse({ json: { current_project: "Alpha" } }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 409 }))
      .mockResolvedValueOnce(mockResponse())
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 503 }))
      .mockResolvedValueOnce(mockResponse({ json: { current_project: "Alpha" } }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 500 }))
      .mockResolvedValueOnce(mockResponse({ json: { content: "KEY=VALUE" } }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 404 }))
      .mockResolvedValueOnce(mockResponse())
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 400 }))
      .mockResolvedValueOnce(mockResponse({ json: { field: [{ value: "one", label: "One" }] } }))
      .mockResolvedValueOnce(mockResponse({ ok: false, status: 500 }));

    await expect(fetchProjects()).resolves.toEqual([{ name: "Alpha", has_thumbnail: true }]);
    await expect(fetchProjects()).rejects.toThrow("Fetch projects failed: 500");
    await expect(createProject("Alpha")).resolves.toBeUndefined();
    await expect(createProject("Alpha")).rejects.toThrow("Create project failed: 400");
    await expect(deleteProject("Alpha")).resolves.toBeUndefined();
    await expect(deleteProject("Alpha")).rejects.toThrow("Delete project failed: 404");
    await expect(startProject("Alpha")).resolves.toEqual({ current_project: "Alpha" });
    await expect(startProject("Alpha")).rejects.toThrow("Start project failed: 409");
    await expect(closeProject()).resolves.toBeUndefined();
    await expect(closeProject()).rejects.toThrow("Close project failed: 503");
    await expect(fetchCurrentProject()).resolves.toEqual({ current_project: "Alpha" });
    await expect(fetchCurrentProject()).rejects.toThrow("Fetch current project failed: 500");
    await expect(fetchEnv()).resolves.toBe("KEY=VALUE");
    await expect(fetchEnv()).rejects.toThrow("Fetch env failed: 404");
    await expect(putEnv("A=B")).resolves.toBeUndefined();
    await expect(putEnv("A=B")).rejects.toThrow("Put env failed: 400");
    await expect(fetchOptions("Comp", { level: 2 })).resolves.toEqual({
      field: [{ value: "one", label: "One" }],
    });
    await expect(fetchOptions("Comp")).rejects.toThrow("Fetch config options failed: 500");

    expect(projectThumbnailUrl("Alpha One")).toBe("/projects/Alpha%20One/thumbnail");
  });

  it("covers component logs happy path and content-type failures", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock
      .mockResolvedValueOnce(
        mockResponse({
          json: { node_id: "node-1", entries: [] },
          headers: { "content-type": "application/json; charset=utf-8" },
        }),
      )
      .mockResolvedValueOnce(
        mockResponse({
          ok: false,
          status: 500,
          headers: { "content-type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        mockResponse({
          text: "<html>oops</html>",
          headers: { "content-type": "text/html" },
        }),
      )
      .mockResolvedValueOnce(
        mockResponse({
          text: "plain text",
        }),
      );

    await expect(fetchComponentLogs("node 1", 2, 10)).resolves.toEqual({
      node_id: "node-1",
      entries: [],
    });
    await expect(fetchComponentLogs("node 1")).rejects.toThrow("Fetch component logs failed: 500");
    await expect(fetchComponentLogs("node 1")).rejects.toThrow(
      "Fetch component logs returned non-JSON (text/html): <html>oops</html>",
    );
    await expect(fetchComponentLogs("node 1")).rejects.toThrow(
      "Fetch component logs returned non-JSON (unknown): plain text",
    );

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/logs/node%201?after=2&limit=10",
    );
  });
});
