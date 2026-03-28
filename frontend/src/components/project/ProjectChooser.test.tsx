import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProjectChooser } from "./ProjectChooser";
import * as api from "@/lib/api";

describe("ProjectChooser", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("loads projects, opens, deletes, creates, and cancels", async () => {
    vi.spyOn(api, "fetchProjects").mockResolvedValue([
      { name: "Alpha", has_thumbnail: true },
      { name: "Beta", has_thumbnail: false },
    ]);
    const startProject = vi.spyOn(api, "startProject").mockResolvedValue({
      current_project: "Alpha",
    });
    const createProject = vi.spyOn(api, "createProject").mockResolvedValue();
    const deleteProject = vi.spyOn(api, "deleteProject").mockResolvedValue();
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const onOpen = vi.fn();
    const onCancel = vi.fn();

    const { container } = render(
      <ProjectChooser hadProject onOpen={onOpen} onCancel={onCancel} />,
    );

    await waitFor(() => expect(screen.getByRole("button", { name: /alpha/i })).toBeInTheDocument());
    expect(errorSpy).not.toHaveBeenCalledWith(
      expect.stringContaining("<button> cannot be a descendant of <button>"),
    );
    expect(screen.getByRole("img", { name: "Alpha" })).toHaveAttribute(
      "src",
      "/projects/Alpha/thumbnail",
    );

    fireEvent.click(screen.getAllByRole("button")[0]!);
    expect(onCancel).toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /alpha/i }));
    await waitFor(() => expect(startProject).toHaveBeenCalledWith("Alpha"));
    expect(onOpen).toHaveBeenCalledWith("Alpha");

    const betaCard = screen.getByRole("button", { name: /beta/i });
    fireEvent.click(betaCard.parentElement?.querySelector("button.absolute") as HTMLButtonElement);
    await waitFor(() => expect(deleteProject).toHaveBeenCalledWith("Beta"));

    fireEvent.click(screen.getByRole("button", { name: /new project/i }));
    const input = screen.getByPlaceholderText("Project name");
    fireEvent.change(input, { target: { value: "Gamma" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => expect(createProject).toHaveBeenCalledWith("Gamma"));
    expect(startProject).toHaveBeenCalledWith("Gamma");
    expect(onOpen).toHaveBeenCalledWith("Gamma");

    fireEvent.click(screen.getByRole("button", { name: /new project/i }));
    fireEvent.click(container.querySelector(".fixed.inset-0.z-60")!);
    expect(screen.queryByPlaceholderText("Project name")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /new project/i }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByPlaceholderText("Project name")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /new project/i }));
    const blankInput = screen.getByPlaceholderText("Project name");
    fireEvent.change(blankInput, { target: { value: "   " } });
    fireEvent.keyDown(blankInput, { key: "Enter" });
    expect(createProject).toHaveBeenCalledTimes(1);
  });

  it("hides cancel without previous project and resets creating state on create failure", async () => {
    vi.spyOn(api, "fetchProjects").mockResolvedValue([]);
    vi.spyOn(api, "createProject").mockRejectedValue(new Error("boom"));
    vi.spyOn(api, "startProject").mockResolvedValue({ current_project: null });
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(<ProjectChooser hadProject={false} onOpen={vi.fn()} onCancel={vi.fn()} />);

    await waitFor(() => expect(screen.getByRole("button", { name: /new project/i })).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /new project/i }));
    const input = screen.getByPlaceholderText("Project name");
    fireEvent.change(input, { target: { value: "Failing" } });
    fireEvent.click(screen.getByRole("button", { name: /create/i }));

    await waitFor(() => expect(errorSpy).toHaveBeenCalled());
    expect(screen.getByRole("button", { name: /create/i })).toBeEnabled();

    fireEvent.keyDown(input, { key: "Escape" });
    expect(screen.queryByPlaceholderText("Project name")).not.toBeInTheDocument();
  });
});

