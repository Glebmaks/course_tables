"""A transparent, dependency-free baseline for flattening HTML tables.

Scope: one regular table, top header rows, left-side row headers, finite
rowspan/colspan. No LLM or inference of missing semantics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Optional


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Cell:
    tag: str
    text: str
    rowspan: int
    colspan: int
    scope: str
    section: str
    source_row: int
    source_cell: int

    def location(self) -> dict:
        return {"html_row": self.source_row, "html_cell": self.source_cell}


@dataclass
class Row:
    section: str
    cells: list[Cell] = field(default_factory=list)


@dataclass
class Table:
    rows: list[Row] = field(default_factory=list)
    caption: str = ""


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[Table] = []
        self.depth = 0
        self.table: Optional[Table] = None
        self.row: Optional[Row] = None
        self.cell_tag: Optional[str] = None
        self.cell_attrs: dict = {}
        self.cell_text: list[str] = []
        self.section = "body"
        self.in_caption = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.table = Table()
                self.tables.append(self.table)
                self.section = "body"
        if self.depth != 1:
            return
        if tag in ("thead", "tbody", "tfoot"):
            self.section = tag
        elif tag == "caption":
            self.in_caption = True
            self.cell_text = []
        elif tag == "tr":
            self.row = Row(self.section)
            assert self.table is not None
            self.table.rows.append(self.row)
        elif tag in ("th", "td") and self.row is not None:
            self.cell_tag = tag
            self.cell_attrs = attrs
            self.cell_text = []
        elif tag == "br" and (self.cell_tag or self.in_caption):
            self.cell_text.append(" ")

    def handle_data(self, data):
        if self.depth == 1 and (self.cell_tag or self.in_caption):
            self.cell_text.append(data)

    def handle_endtag(self, tag):
        if self.depth == 1:
            if tag in ("th", "td") and self.cell_tag == tag and self.row:
                def span(key):
                    try:
                        value = int(self.cell_attrs.get(key, 1))
                        if value < 1 or value > 10000:
                            raise ValueError
                        return value
                    except ValueError as exc:
                        raise ValueError(f"Unsupported {key}={self.cell_attrs.get(key)!r}") from exc

                self.row.cells.append(Cell(
                    tag, clean("".join(self.cell_text)), span("rowspan"),
                    span("colspan"), self.cell_attrs.get("scope", ""),
                    self.row.section, len(self.table.rows) - 1, len(self.row.cells)
                ))
                self.cell_tag = None
                self.cell_text = []
            elif tag == "caption" and self.in_caption:
                self.table.caption = clean("".join(self.cell_text))
                self.in_caption = False
                self.cell_text = []
            elif tag == "tr":
                self.row = None
            elif tag in ("thead", "tbody", "tfoot"):
                self.section = "body"
        if tag == "table" and self.depth > 0:
            self.depth -= 1
            if self.depth == 0:
                self.table = None


def grid_for(table: Table) -> list[list[Optional[Cell]]]:
    grid: list[list[Optional[Cell]]] = [[] for _ in table.rows]
    for r, row in enumerate(table.rows):
        col = 0
        for cell in row.cells:
            while col < len(grid[r]) and grid[r][col] is not None:
                col += 1
            if r + cell.rowspan > len(table.rows):
                raise ValueError(f"rowspan passes end of table in row {r}")
            for rr in range(r, r + cell.rowspan):
                line = grid[rr]
                if len(line) < col + cell.colspan:
                    line.extend([None] * (col + cell.colspan - len(line)))
                for cc in range(col, col + cell.colspan):
                    if line[cc] is not None:
                        raise ValueError(f"Overlapping spans in row {r}, col {cc}")
                    line[cc] = cell
            col += cell.colspan
    width = max((len(row) for row in grid), default=0)
    if not width:
        raise ValueError("Empty table")
    for r, line in enumerate(grid):
        if len(line) != width or any(cell is None for cell in line):
            raise ValueError(f"Ragged table at row {r}; cannot flatten safely")
    return grid


def flatten(html: str, table_index: int = 0) -> dict:
    parser = TableParser()
    parser.feed(html)
    if table_index < 0 or table_index >= len(parser.tables):
        raise ValueError(f"Table {table_index} not found ({len(parser.tables)} tables)")
    table = parser.tables[table_index]
    grid = grid_for(table)
    width = len(grid[0])
    header_rows = [i for i, row in enumerate(table.rows) if row.section == "thead"]
    if not header_rows:
        for i, row in enumerate(table.rows):
            if row.cells and all(cell.tag == "th" for cell in row.cells):
                header_rows.append(i)
            else:
                break
    if not header_rows or header_rows != list(range(len(header_rows))):
        raise ValueError("Expected one top, contiguous header block")
    body = [i for i in range(len(header_rows), len(grid)) if table.rows[i].section != "tfoot"]
    if not body:
        raise ValueError("No data rows")

    # A left stub is consistently a <th> in data rows. Vertical spans still
    # point to the same <th>, so the parent label is repeated correctly.
    stub_count = 0
    for col in range(width):
        if all(grid[r][col].tag == "th" for r in body):
            stub_count += 1
        else:
            break
    if stub_count == 0 or stub_count == width:
        raise ValueError("Expected row headers (<th>) on the left and data (<td>) on the right")
    for r in body:
        for col in range(stub_count, width):
            if grid[r][col].tag != "td":
                raise ValueError(f"Group or subtotal row {r} requires separate handling")
            if grid[r][col].colspan > 1 or grid[r][col].rowspan > 1:
                raise ValueError(f"Merged data value at row {r}, col {col}: cannot safely duplicate it")
        for col in range(stub_count):
            if not grid[r][col].text:
                raise ValueError(f"Unlabelled row header at row {r}, col {col}")

    def path(col):
        seen = set()
        parts = []
        locations = []
        for r in header_rows:
            cell = grid[r][col]
            key = (cell.source_row, cell.source_cell)
            if key not in seen:
                seen.add(key)
                if cell.text:
                    parts.append(cell.text)
                    locations.append(cell.location())
        return parts, locations

    headers = []
    header_sources = []
    for col in range(width):
        parts, locations = path(col)
        if not parts:
            raise ValueError(f"Column {col} has no header")
        name = " / ".join(parts)
        if name in headers:
            raise ValueError(f"Ambiguous duplicate column name: {name}")
        headers.append(name)
        header_sources.append(locations)

    values = []
    provenance = []
    for r in body:
        values.append([grid[r][col].text for col in range(width)])
        provenance.append([grid[r][col].location() for col in range(width)])
    return {
        "headers": headers,
        "rows": values,
        "metadata": {
            "caption": table.caption, "table_index": table_index,
            "stub_columns": stub_count, "header_sources": header_sources,
            "cell_sources": provenance,
            "note": "Coordinates are zero-based HTML <tr> and cell positions; caption/footnotes are not inferred into column names."
        }
    }


def write_csv(path: Path, result: dict) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(result["headers"])
        writer.writerows(result["rows"])


def main():
    cli = argparse.ArgumentParser(description="Flatten a hierarchical HTML table")
    cli.add_argument("input", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--table-index", type=int, default=0)
    cli.add_argument("--metadata", type=Path, help="Optional JSON with source coordinates")
    args = cli.parse_args()
    result = flatten(args.input.read_text(encoding="utf-8"), args.table_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, result)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(result["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(result['rows'])} rows, {len(result['headers'])} columns → {args.output}")


if __name__ == "__main__":
    main()
