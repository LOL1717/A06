"""Lightweight LangGraph pipeline to test fast PDF parsing in constrained environments.

Design goals for Codespaces-like limits:
- Favor text extraction first and avoid OCR when text is available.
- Limit work to small PDFs and first N pages for speed.
- Keep table extraction as a first-class output.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, TypedDict

import requests
from duckduckgo_search import DDGS
from docling.document_converter import DocumentConverter
from langgraph.graph import END, StateGraph
from pypdf import PdfReader, PdfWriter

# Constrained-environment defaults (speed > stability > completeness)
MAX_PDF_URLS = 2
MAX_PDF_SIZE_MB = 20
MAX_PDF_PAGES = 50
MAX_PROCESS_PAGES = 8
DOWNLOAD_TIMEOUT_SEC = 20
PARSE_TIMEOUT_SEC = 45


class DoclingTestState(TypedDict):
    """State for the 3-node Docling test graph."""

    topic: str
    search_queries: list[str]
    pdf_urls: list[str]
    extracted_data: list[dict[str, Any]]


def generate_queries_node(state: DoclingTestState) -> DoclingTestState:
    """Create two deterministic PDF-only search queries from the topic."""

    topic = state["topic"].strip()
    state["search_queries"] = [
        f"{topic} filetype:pdf",
        f"{topic} clinical trial filetype:pdf",
    ]
    return state


def find_pdfs_node(state: DoclingTestState) -> DoclingTestState:
    """Search DuckDuckGo and keep at most two direct .pdf links."""

    found: list[str] = []
    seen: set[str] = set()

    with DDGS() as ddgs:
        for query in state.get("search_queries", []):
            try:
                results = ddgs.text(query, max_results=12)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Search failed for query={query!r}: {exc}")
                continue

            for result in results or []:
                url = (result or {}).get("href") or ""
                normalized = url.split("?", 1)[0].lower()
                if normalized.endswith(".pdf") and url not in seen:
                    seen.add(url)
                    found.append(url)
                if len(found) >= MAX_PDF_URLS:
                    break

            if len(found) >= MAX_PDF_URLS:
                break

    state["pdf_urls"] = found
    return state


def _is_pdf_too_large(url: str) -> bool:
    """Fast size gate using HEAD before download when content-length is available."""

    try:
        response = requests.head(url, allow_redirects=True, timeout=DOWNLOAD_TIMEOUT_SEC)
        content_length = int(response.headers.get("Content-Length", "0") or 0)
    except Exception:
        return False

    return content_length > MAX_PDF_SIZE_MB * 1024 * 1024


def _download_pdf(url: str, temp_dir: str) -> Path:
    """Download PDF with streaming and hard size limit."""

    local_path = Path(temp_dir) / "source.pdf"
    max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
    downloaded = 0

    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
        response.raise_for_status()
        with local_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError(f"PDF exceeds {MAX_PDF_SIZE_MB}MB limit during download")
                handle.write(chunk)

    return local_path


def _truncate_pdf_pages(source_pdf: Path, temp_dir: str) -> tuple[Path, int]:
    """Skip huge PDFs; otherwise keep only first MAX_PROCESS_PAGES pages for parsing."""

    reader = PdfReader(str(source_pdf))
    total_pages = len(reader.pages)

    if total_pages > MAX_PDF_PAGES:
        raise ValueError(
            f"Skipping PDF with {total_pages} pages (> {MAX_PDF_PAGES} page limit)"
        )

    pages_to_keep = min(total_pages, MAX_PROCESS_PAGES)
    trimmed_path = Path(temp_dir) / "trimmed.pdf"

    writer = PdfWriter()
    for page_idx in range(pages_to_keep):
        writer.add_page(reader.pages[page_idx])

    with trimmed_path.open("wb") as handle:
        writer.write(handle)

    return trimmed_path, total_pages


def _extract_text_fast(pdf_path: Path) -> str:
    """Native text extraction from first MAX_PROCESS_PAGES pages (no OCR)."""

    reader = PdfReader(str(pdf_path))
    text_chunks: list[str] = []

    for page in reader.pages[:MAX_PROCESS_PAGES]:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_chunks.append(page_text)

    return "\n\n".join(text_chunks).strip()


def _parse_with_timeout(converter: DocumentConverter, path: Path, timeout_sec: int):
    """Run Docling conversion with an upper-bound timeout."""

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(converter.convert, str(path))
        return future.result(timeout=timeout_sec)


def _table_to_rows(table_obj: Any) -> list[list[Any]]:
    """Best-effort conversion of a Docling table object to row/column structure."""

    if hasattr(table_obj, "export_to_dataframe"):
        try:
            dataframe = table_obj.export_to_dataframe()
            if dataframe is not None:
                rows = [list(dataframe.columns)]
                rows.extend(dataframe.fillna("").values.tolist())
                return rows
        except Exception:
            pass

    if hasattr(table_obj, "data"):
        data = getattr(table_obj, "data")
        if isinstance(data, list):
            return data

    if hasattr(table_obj, "cells"):
        cells = getattr(table_obj, "cells")
        if isinstance(cells, list):
            return cells

    return [[str(table_obj)]]


def _extract_structured_tables(parsed_document: Any) -> list[list[list[Any]]]:
    """Extract tables as nested row structures from common Docling shapes."""

    table_objects: list[Any] = []

    if hasattr(parsed_document, "tables"):
        maybe_tables = getattr(parsed_document, "tables")
        if maybe_tables is not None:
            try:
                table_objects.extend(list(maybe_tables))
            except TypeError:
                table_objects.append(maybe_tables)

    if not table_objects and hasattr(parsed_document, "items"):
        for item in getattr(parsed_document, "items") or []:
            if getattr(item, "label", "").lower() == "table":
                table_objects.append(item)

    return [_table_to_rows(table) for table in table_objects]


def _parse_single_pdf(url: str) -> dict[str, Any] | None:
    """Process one PDF with fast guards and OCR-minimizing strategy."""

    if _is_pdf_too_large(url):
        print(f"[warn] Skipping oversized PDF by HEAD check: {url}")
        return None

    with tempfile.TemporaryDirectory(prefix="docling_test_") as temp_dir:
        source_pdf = _download_pdf(url, temp_dir)
        trimmed_pdf, total_pages = _truncate_pdf_pages(source_pdf, temp_dir)

        native_text = _extract_text_fast(trimmed_pdf)
        method = "text" if native_text else "ocr"

        converter = DocumentConverter()
        conversion_result = _parse_with_timeout(converter, trimmed_pdf, PARSE_TIMEOUT_SEC)
        document = getattr(conversion_result, "document", conversion_result)

        markdown = ""
        if hasattr(document, "export_to_markdown"):
            markdown = document.export_to_markdown() or ""

        text_output = native_text if native_text else markdown
        tables = _extract_structured_tables(document)

        return {
            "url": url,
            "text": text_output,
            "tables": tables,
            "method": method,
            "total_pages_detected": total_pages,
            "processed_pages": min(total_pages, MAX_PROCESS_PAGES),
            "table_count": len(tables),
        }


def parse_pdfs_node(state: DoclingTestState) -> DoclingTestState:
    """Parse candidate PDFs while keeping runtime bounded and robust."""

    extracted: list[dict[str, Any]] = []

    for url in state.get("pdf_urls", []):
        try:
            item = _parse_single_pdf(url)
            if item is not None:
                extracted.append(item)
        except FuturesTimeoutError:
            print(f"[warn] Timed out parsing PDF (>{PARSE_TIMEOUT_SEC}s): {url}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to parse PDF {url}: {exc}")

    state["extracted_data"] = extracted
    return state


def build_graph():
    """Create and compile the 3-node LangGraph pipeline."""

    graph = StateGraph(DoclingTestState)
    graph.add_node("generate_queries_node", generate_queries_node)
    graph.add_node("find_pdfs_node", find_pdfs_node)
    graph.add_node("parse_pdfs_node", parse_pdfs_node)

    graph.set_entry_point("generate_queries_node")
    graph.add_edge("generate_queries_node", "find_pdfs_node")
    graph.add_edge("find_pdfs_node", "parse_pdfs_node")
    graph.add_edge("parse_pdfs_node", END)

    return graph.compile()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    app = build_graph()

    initial_state: DoclingTestState = {
        "topic": "Antibody-Drug Conjugates ADCs",
        "search_queries": [],
        "pdf_urls": [],
        "extracted_data": [],
    }

    result = app.invoke(initial_state)

    print("\n=== Docling Pipeline Summary ===")
    print(f"Topic: {result.get('topic', '')}")
    print(f"Generated queries: {result.get('search_queries', [])}")
    print(f"PDF URLs found: {len(result.get('pdf_urls', []))}")

    for i, item in enumerate(result.get("extracted_data", []), start=1):
        print(f"\n[{i}] URL: {item.get('url', '')}")
        print(f"    Method: {item.get('method', '')}")
        print(f"    Text length: {len(item.get('text', ''))} chars")
        print(f"    Tables found: {len(item.get('tables', []))}")
        print(
            "    Pages: "
            f"processed={item.get('processed_pages', 0)}/"
            f"detected={item.get('total_pages_detected', 0)}"
        )
