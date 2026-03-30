from __future__ import annotations

import argparse
import html
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoverageResult:
    name: str
    badge_label: str
    covered: int
    total: int

    @property
    def percent(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.covered / self.total) * 100

    @property
    def percent_text(self) -> str:
        return f"{self.percent:.2f}%"

    @property
    def lines_text(self) -> str:
        return f"{self.covered} / {self.total}"


def parse_backend_coverage(path: Path) -> CoverageResult:
    root = ET.parse(path).getroot()
    covered = int(root.attrib["lines-covered"])
    total = int(root.attrib["lines-valid"])
    return CoverageResult(
        name="Backend",
        badge_label="backend coverage",
        covered=covered,
        total=total,
    )


def parse_frontend_coverage(path: Path) -> CoverageResult:
    covered = 0
    total = 0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("LH:"):
            covered += int(line[3:])
        elif line.startswith("LF:"):
            total += int(line[3:])

    return CoverageResult(
        name="Frontend",
        badge_label="frontend coverage",
        covered=covered,
        total=total,
    )


def badge_color(percent: float) -> str:
    if percent >= 90:
        return "#4c1"
    if percent >= 80:
        return "#97CA00"
    if percent >= 70:
        return "#A4A61D"
    if percent >= 60:
        return "#dfb317"
    if percent >= 50:
        return "#fe7d37"
    return "#e05d44"


def text_width(text: str) -> int:
    return max(36, 10 + (len(text) * 6))


def render_badge(result: CoverageResult) -> str:
    label = result.badge_label
    value = result.percent_text
    label_width = text_width(label)
    value_width = text_width(value)
    total_width = label_width + value_width
    label_center = label_width / 2
    value_center = label_width + (value_width / 2)
    color = badge_color(result.percent)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{html.escape(label)}: {html.escape(value)}">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h{label_width}v20H0z"/>
        <path fill="{color}" d="M{label_width} 0h{value_width}v20H{label_width}z"/>
        <path fill="url(#b)" d="M0 0h{total_width}v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="{label_center}" y="15" fill="#010101" fill-opacity=".3">{html.escape(label)}</text>
        <text x="{label_center}" y="14">{html.escape(label)}</text>
        <text x="{value_center}" y="15" fill="#010101" fill-opacity=".3">{html.escape(value)}</text>
        <text x="{value_center}" y="14">{html.escape(value)}</text>
    </g>
</svg>
"""


def write_badge(path: Path, result: CoverageResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_badge(result), encoding="utf-8")


def write_comment(path: Path, results: list[CoverageResult]) -> None:
    lines = [
        "## Test Coverage",
        "",
        "| Target | Coverage | Covered Lines |",
        "| --- | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            f"| {result.name} | {result.percent_text} | {result.lines_text} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-xml", type=Path)
    parser.add_argument("--backend-svg", type=Path)
    parser.add_argument("--frontend-lcov", type=Path)
    parser.add_argument("--frontend-svg", type=Path)
    parser.add_argument("--comment", type=Path)
    args = parser.parse_args()

    results: list[CoverageResult] = []

    if args.backend_xml or args.backend_svg:
        if not args.backend_xml or not args.backend_svg:
            raise SystemExit("backend badge generation requires --backend-xml and --backend-svg")
        backend_result = parse_backend_coverage(args.backend_xml)
        write_badge(args.backend_svg, backend_result)
        results.append(backend_result)

    if args.frontend_lcov or args.frontend_svg:
        if not args.frontend_lcov or not args.frontend_svg:
            raise SystemExit("frontend badge generation requires --frontend-lcov and --frontend-svg")
        frontend_result = parse_frontend_coverage(args.frontend_lcov)
        write_badge(args.frontend_svg, frontend_result)
        results.append(frontend_result)

    if args.comment:
        if not results:
            raise SystemExit("comment generation requires at least one coverage input")
        write_comment(args.comment, results)

    if not results and not args.comment:
        raise SystemExit("no work requested")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
