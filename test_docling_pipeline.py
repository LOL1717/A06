"""Lightweight LangGraph pipeline to test Docling PDF parsing.

Pipeline:
1) Build simple PDF-focused search queries from a topic.
2) Find direct PDF URLs with DuckDuckGo.
3) Parse discovered PDFs with Docling and collect markdown + tables.
"""

from __future__ import annotations

from typing import TypedDict

from duckduckgo_search import DDGS
from docling.document_converter import DocumentConverter
from langgraph.graph import END, StateGraph


class DoclingTestState(TypedDict):
    """State for the 3-node Docling test graph."""

    topic: str
    search_queries: list[str]
    pdf_urls: list[str]
    extracted_data: list[dict]


def generate_queries_node(state: DoclingTestState) -> DoclingTestState:
    """Create two deterministic PDF-only search queries from the topic."""

    topic = state["topic"].strip()
    queries = [
        f"{topic} filetype:pdf",
        f"{topic} clinical trial filetype:pdf",
    ]
    state["search_queries"] = queries
    return state


def find_pdfs_node(state: DoclingTestState) -> DoclingTestState:
    """Search DuckDuckGo and keep at most two direct .pdf links."""

    found: list[str] = []
    seen: set[str] = set()

    with DDGS() as ddgs:
        for query in state.get("search_queries", []):
            try:
                results = ddgs.text(query, max_results=10)
            except Exception as exc:  # noqa: BLE001 - test utility should continue
                print(f"[warn] Search failed for query={query!r}: {exc}")
                continue

            for result in results or []:
                url = (result or {}).get("href") or ""
                if not url:
                    continue

                normalized = url.split("?", 1)[0].lower()
                if normalized.endswith(".pdf") and url not in seen:
                    seen.add(url)
                    found.append(url)

                if len(found) >= 2:
                    break

            if len(found) >= 2:
                break

    state["pdf_urls"] = found
    return state


def _extract_tables(parsed_document: object) -> list:
    """Best-effort extraction for Docling table structures."""

    tables = []
    if parsed_document is None:
        return tables

    # Most common shape in recent Docling versions.
    if hasattr(parsed_document, "tables"):
        maybe_tables = getattr(parsed_document, "tables")
        if maybe_tables:
            try:
                tables.extend(list(maybe_tables))
            except TypeError:
                tables.append(maybe_tables)

    # Fallback shape where structured content can hold typed items.
    if not tables and hasattr(parsed_document, "items"):
        for item in getattr(parsed_document, "items") or []:
            if getattr(item, "label", "").lower() == "table":
                tables.append(item)

    return tables


def parse_pdfs_node(state: DoclingTestState) -> DoclingTestState:
    """Parse PDFs with Docling and collect markdown + tables."""

    converter = DocumentConverter()
    extracted: list[dict] = []

    for url in state.get("pdf_urls", []):
        try:
            conversion_result = converter.convert(url)

            # Docling commonly exposes parsed content as result.document.
            document = getattr(conversion_result, "document", conversion_result)

            markdown = ""
            if hasattr(document, "export_to_markdown"):
                markdown = document.export_to_markdown() or ""

            tables = _extract_tables(document)

            extracted.append(
                {
                    "url": url,
                    "markdown": markdown,
                    "tables": tables,
                }
            )
        except Exception as exc:  # noqa: BLE001 - intentional for robust test runs
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
        url = item.get("url", "")
        markdown = item.get("markdown", "")
        tables = item.get("tables", [])
        print(f"\n[{i}] URL: {url}")
        print(f"    Markdown length: {len(markdown)} chars")
        print(f"    Tables found: {len(tables)}")
