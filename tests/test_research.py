"""Privacy and traceability checks for Muse's PubMed connector."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import research as R  # noqa: E402


def test_rejects_patient_identifiers_before_network(monkeypatch):
    monkeypatch.setattr(R, "_get", lambda *a: (_ for _ in ()).throw(AssertionError("network called")))
    for query in ("patient Jane ACL", "DOB 03/02/1990 ACL"):
        try:
            R.search(query)
            assert False, "identifier should be rejected"
        except ValueError as e:
            assert "patient-identifying" in str(e)


def test_pubmed_metadata_and_passage_are_traceable(monkeypatch):
    search = json.dumps({"esearchresult": {"idlist": ["123"]}}).encode()
    xml = b'''<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
      <ArticleTitle>ACL scaffold study</ArticleTitle><Abstract><AbstractText Label="RESULTS">A measured result.</AbstractText></Abstract>
      <Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
      <AuthorList><Author><ForeName>Ada</ForeName><LastName>Lovelace</LastName></Author></AuthorList>
      </Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/example</ArticleId>
      <ArticleId IdType="pmc">PMC123</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>'''
    monkeypatch.setattr(R, "_get", lambda path, params: search if "esearch" in path else xml)
    out = R.search("ACL scaffold", 3)
    src = out["sources"][0]
    assert src["pmid"] == "123" and src["doi"] == "10.1/example"
    assert src["url"].endswith("/123/") and src["full_text_url"].endswith("/PMC123/")
    assert src["passage"] == "A measured result."
