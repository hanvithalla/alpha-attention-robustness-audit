"""Step 3 (literature review): verify each intended citation against OpenAlex
and emit workspace/refs.bib plus workspace/citation_pool.json.

No Semantic Scholar or Exa key is available in this environment, so the skill's
parallel discovery pipeline is not run. Instead every reference the outline's
citation_hints call for is looked up by title against OpenAlex (no key
required), the returned title is fuzzy-matched against the query to reject
wrong hits, and only verified records are written. Unverified entries are
reported and kept out of refs.bib rather than being invented.

    python build_refs.py
"""

import difflib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OPENALEX = "https://api.openalex.org/works"
MATCH_THRESHOLD = 0.72

# (bibtex_key, query_title, fallback_entry_type)
WANTED = [
    ("hu2018squeeze", "Squeeze-and-Excitation Networks", "inproceedings"),
    ("park2018bam", "BAM: Bottleneck Attention Module", "inproceedings"),
    ("woo2018cbam", "CBAM: Convolutional Block Attention Module", "inproceedings"),
    ("hendrycks2019benchmarking",
     "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations",
     "inproceedings"),
    ("he2016deep", "Deep Residual Learning for Image Recognition", "inproceedings"),
    ("ioffe2015batch",
     "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
     "inproceedings"),
    ("loshchilov2017sgdr", "SGDR: Stochastic Gradient Descent with Warm Restarts", "inproceedings"),
    ("micikevicius2018mixed", "Mixed Precision Training", "inproceedings"),
    ("guo2022attention", "Attention mechanisms in computer vision: A survey", "article"),
    ("wang2020eca",
     "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks",
     "inproceedings"),
    ("hendrycks2020augmix",
     "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty",
     "inproceedings"),
    ("krizhevsky2009learning",
     "Learning Multiple Layers of Features from Tiny Images", "techreport"),
    ("ar2repair2025",
     "AR2: Attention-Guided Repair for the Robustness of CNNs Against Common Corruptions",
     "inproceedings"),
    ("pushpull2025",
     "Multi-Scale Unrectified Push-Pull with Channel Attention for Enhanced Corruption Robustness",
     "inproceedings"),
    ("geirhos2019imagenet",
     "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness",
     "inproceedings"),
]

# OpenAlex reliably verifies that a work exists and that its title matches, but
# it is not reliable for canonical venue/year: it often returns the preprint
# record, and for the CIFAR-10 report it returns a re-dated 2024 entry.
# Existence stays machine-verified; these canonical fields are applied on top
# and logged in citation_pool.json as overrides, so every substitution is
# auditable rather than silent.
CANONICAL = {
    "hu2018squeeze": ("inproceedings", "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)", 2018),
    "park2018bam": ("inproceedings", "British Machine Vision Conference (BMVC)", 2018),
    "woo2018cbam": ("inproceedings", "European Conference on Computer Vision (ECCV)", 2018),
    "hendrycks2019benchmarking": ("inproceedings", "International Conference on Learning Representations (ICLR)", 2019),
    "he2016deep": ("inproceedings", "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)", 2016),
    "ioffe2015batch": ("inproceedings", "International Conference on Machine Learning (ICML)", 2015),
    "loshchilov2017sgdr": ("inproceedings", "International Conference on Learning Representations (ICLR)", 2017),
    "micikevicius2018mixed": ("inproceedings", "International Conference on Learning Representations (ICLR)", 2018),
    "wang2020eca": ("inproceedings", "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 2020),
    "hendrycks2020augmix": ("inproceedings", "International Conference on Learning Representations (ICLR)", 2020),
    "geirhos2019imagenet": ("inproceedings", "International Conference on Learning Representations (ICLR)", 2019),
    "krizhevsky2009learning": ("techreport", "University of Toronto", 2009),
}

# OpenAlex returns U+2010 in hyphenated author names; T1 font encoding cannot
# typeset it, so normalise to ASCII before writing LaTeX.
UNICODE_FIXES = {
    "‐": "-", "‑": "-", "‒": "-", "–": "--", "—": "---",
    "‘": "`", "’": "'", "“": "``", "”": "''", " ": " ",
}


def fetch(title):
    url = f"{OPENALEX}?{urllib.parse.urlencode({'search': title, 'per_page': 3})}"
    req = urllib.request.Request(url, headers={"User-Agent": "paper-orchestra-litreview/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("results", [])


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def best_match(query, results):
    best, score = None, 0.0
    for w in results:
        s = difflib.SequenceMatcher(None, norm(query), norm(w.get("title"))).ratio()
        if s > score:
            best, score = w, s
    return best, score


def authors_of(work, limit=12):
    names = [a["author"]["display_name"] for a in work.get("authorships", [])[:limit]
             if a.get("author", {}).get("display_name")]
    return " and ".join(names)


def venue_of(work):
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name") or ""


def escape(s):
    s = s or ""
    for bad, good in UNICODE_FIXES.items():
        s = s.replace(bad, good)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main():
    ws = Path("workspace")
    ws.mkdir(exist_ok=True)
    entries, pool, failed = [], [], []

    for key, title, etype in WANTED:
        try:
            results = fetch(title)
        except Exception as e:
            failed.append((key, title, f"request failed: {e}"))
            time.sleep(0.5)
            continue

        work, score = best_match(title, results)
        if not work or score < MATCH_THRESHOLD:
            failed.append((key, title, f"no confident match (best ratio {score:.2f})"))
            time.sleep(0.5)
            continue

        year = work.get("publication_year")
        venue = venue_of(work)
        override = None
        if key in CANONICAL:
            etype, cvenue, cyear = CANONICAL[key]
            override = {"from": {"year": year, "venue": venue},
                        "to": {"year": cyear, "venue": cvenue}}
            year, venue = cyear, cvenue

        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        rec = {
            "key": key, "title": work.get("title"), "year": year, "venue": venue,
            "doi": doi, "openalex_id": work.get("id"),
            "cited_by_count": work.get("cited_by_count"),
            "match_ratio": round(score, 3),
            "authors": authors_of(work),
            "canonical_override": override,
        }
        pool.append(rec)

        kind = etype if venue and "arxiv" not in venue.lower() else "misc"
        fields = [f"  title        = {{{escape(work.get('title'))}}}",
                  f"  author       = {{{escape(rec['authors'])}}}",
                  f"  year         = {{{year}}}"]
        if kind == "misc":
            fields.append(f"  howpublished = {{{escape(venue) or 'Preprint'}}}")
        elif kind == "techreport":
            fields.append(f"  institution  = {{{escape(venue)}}}")
        elif kind == "article":
            fields.append(f"  journal      = {{{escape(venue)}}}")
        else:
            fields.append(f"  booktitle    = {{{escape(venue)}}}")
        if doi:
            fields.append(f"  doi          = {{{doi}}}")
        entries.append(f"@{kind}{{{key},\n" + ",\n".join(fields) + "\n}\n")
        print(f"  [ok {score:.2f}] {key}: {(work.get('title') or '')[:60]} ({year})")
        time.sleep(0.35)

    (ws / "refs.bib").write_text("\n".join(entries), encoding="utf-8")
    (ws / "citation_pool.json").write_text(
        json.dumps({"source": "openalex", "verified": pool, "unverified": failed}, indent=2),
        encoding="utf-8")

    print(f"\nVerified {len(pool)} / {len(WANTED)} references -> workspace/refs.bib")
    for key, title, why in failed:
        print(f"  [UNVERIFIED] {key}: {title[:55]} -- {why}")
    if failed:
        print("Unverified entries were NOT written to refs.bib.")


if __name__ == "__main__":
    main()
