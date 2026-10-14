"""Verified local NBA documents and stable, location-aware chunks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader


@dataclass(frozen=True)
class SourceUnit:
    text: str
    metadata: dict


def load_corpus(manifest_path: Path) -> list[SourceUnit]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not manifest.get("sources"):
        raise ValueError("manifest must contain version 1 and sources")
    root = (manifest_path.parent / manifest.get("source_root", ".")).resolve()
    units: list[SourceUnit] = []
    seen: set[str] = set()
    for source in manifest["sources"]:
        source_id = source["id"]
        if source_id in seen:
            raise ValueError(f"duplicate source id: {source_id}")
        seen.add(source_id)
        relative = Path(source["path"])
        path = (root / relative).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"source path escapes manifest directory: {relative}")
        raw = path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        if sha256 != source.get("sha256"):
            raise ValueError(f"sha256 mismatch for {source_id}")
        metadata = {key: source[key] for key in ("id", "source_url", "version", "retrieved_at", "license")}
        metadata["sha256"] = sha256
        metadata["path"] = relative.as_posix()
        if source["format"] == "html":
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            root_tag = soup.find("main") or soup.body or soup
            articles = root_tag.find_all("article")
            heading = "Untitled"
            anchor = None
            parts: list[str] = []

            def flush():
                if parts:
                    units.append(SourceUnit("\n".join(parts), {**metadata, "heading": heading, "anchor": anchor, "page": None}))
                    parts.clear()

            for tag in root_tag.find_all(re.compile(r"^(h[1-6]|p|li|dt|dd)$")):
                if articles and tag.find_parent("article") is None:
                    continue
                if tag.find_parent(re.compile(r"^(p|li)$")):
                    continue
                if tag.name in {"dt", "dd"}:
                    value = " ".join(
                        str(child).strip() if isinstance(child, NavigableString) else child.get_text(" ", strip=True)
                        for child in tag.contents if not getattr(child, "name", None) in {"dt", "dd"}
                    ).strip()
                else:
                    value = tag.get_text(" ", strip=True)
                if not value:
                    continue
                if re.fullmatch(r"h[1-6]", tag.name):
                    flush()
                    heading = value
                    article = tag.find_parent("article")
                    anchor = tag.get("id") or (article.get("id") if article else None)
                else:
                    parts.append(value)
            flush()
        elif source["format"] == "pdf":
            for page_number, page in enumerate(PdfReader(path).pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    units.append(SourceUnit(text, {**metadata, "heading": None, "anchor": None, "page": page_number}))
            if not any(unit.metadata["id"] == source_id for unit in units):
                raise ValueError(f"PDF has no extractable text: {source_id}")
        else:
            raise ValueError(f"unsupported source format: {source['format']}")
    return units


def split_corpus(units: list[SourceUnit], *, chunk_size: int = 500, overlap: int = 50):
    if chunk_size < 64 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("invalid chunk parameters")
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    nodes = []
    for unit_index, unit in enumerate(units):
        document = Document(text=unit.text, metadata=unit.metadata)
        for ordinal, node in enumerate(splitter.get_nodes_from_documents([document])):
            location = f"p{unit.metadata['page']}" if unit.metadata["page"] else unit.metadata["heading"]
            digest = hashlib.sha256(f"{unit.metadata['sha256']}:{location}:{unit_index}:{ordinal}:{node.text}".encode()).hexdigest()[:16]
            citation_id = f"{unit.metadata['id']}:{digest}"
            node.id_ = citation_id
            node.metadata["citation_id"] = citation_id
            nodes.append(node)
    return nodes
