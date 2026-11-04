"""Small, traceable PubMed research service used by Muse.

Only generalized biomedical/engineering terms are accepted. No report text, identifiers,
or raw patient data are sent. Results retain the abstract passage and publication metadata;
they are reference material, never application instructions.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDENTIFIER = re.compile(r"\b(?:mrn|dob|date of birth|patient|name|address|phone|email)\b", re.I)


def status():
    return {"available": True, "provider": "NCBI PubMed E-utilities", "full_text": "abstracts; PMC links when available",
            "privacy": "generalized query terms only; patient details are rejected"}


def _get(path, params):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "ACLScaffoldMuse/1.0 (research; contact via repository)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def _text(node, path, default=""):
    x = node.find(path)
    return " ".join("".join(x.itertext()).split()) if x is not None else default


def search(query, limit=5):
    query = " ".join(str(query or "").split())[:350]
    if not query:
        raise ValueError("research query is empty")
    if IDENTIFIER.search(query) or re.search(r"\b\d{2}[/-]\d{2}[/-]\d{2,4}\b", query):
        raise ValueError("patient-identifying terms are not allowed in external research queries")
    limit = max(1, min(int(limit or 5), 8))
    raw = _get("/esearch.fcgi", {"db": "pubmed", "term": query, "retmode": "json", "retmax": limit, "sort": "relevance"})
    ids = json.loads(raw.decode("utf-8"))["esearchresult"].get("idlist", [])
    if not ids:
        return {"query": query, "provider": "NCBI PubMed", "sources": []}
    root = ET.fromstring(_get("/efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}))
    out = []
    for article in root.findall(".//PubmedArticle"):
        citation = article.find(".//MedlineCitation")
        pmid = _text(citation, "PMID")
        title = _text(citation, "Article/ArticleTitle", "Untitled")
        abstract = " ".join(_text(x, ".") for x in citation.findall("Article/Abstract/AbstractText")).strip()
        journal = _text(citation, "Article/Journal/Title")
        year = (_text(citation, "Article/Journal/JournalIssue/PubDate/Year") or
                _text(citation, "Article/Journal/JournalIssue/PubDate/MedlineDate")[:4])
        authors = []
        for a in citation.findall("Article/AuthorList/Author")[:8]:
            name = " ".join(filter(None, [_text(a, "ForeName"), _text(a, "LastName")]))
            if name:
                authors.append(name)
        doi = ""
        pmc = ""
        for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if aid.attrib.get("IdType") == "doi": doi = (aid.text or "").strip()
            if aid.attrib.get("IdType") == "pmc": pmc = (aid.text or "").strip()
        out.append({"title": title, "authors": authors, "year": year, "journal": journal, "pmid": pmid,
                    "doi": doi or None, "url": "https://pubmed.ncbi.nlm.nih.gov/%s/" % pmid,
                    "full_text_url": "https://pmc.ncbi.nlm.nih.gov/articles/%s/" % pmc if pmc else None,
                    "access": "abstract" if abstract else "metadata only", "passage": abstract[:1800],
                    "context": "Publication type and applicability must be assessed against the current scaffold material, architecture, and loading."})
    return {"query": query, "provider": "NCBI PubMed", "sources": out}
