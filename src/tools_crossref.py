"""Fetch Crossref metadata for DOIs and print a compact, complete author list."""
import json
import sys
import requests

out = []
for doi in sys.argv[1:]:
    m = requests.get(f"https://api.crossref.org/works/{doi}", timeout=30).json()["message"]
    au = [f"{a.get('family', '')} {''.join(x[0] for x in a.get('given', '').replace('-', ' ').split())}".strip()
          for a in m.get("author", [])]
    rec = {"doi": doi, "authors": au, "title": (m.get("title") or [""])[0],
           "journal": (m.get("container-title") or [""])[0], "year": m.get("issued", {}).get("date-parts", [[None]])[0][0],
           "volume": m.get("volume"), "issue": m.get("issue"), "pages": m.get("page") or m.get("article-number"),
           "abstract": (m.get("abstract") or "")[:600]}
    out.append(rec)
    print(json.dumps(rec, ensure_ascii=False))
