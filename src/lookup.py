import re
import difflib
import pandas as pd
from src.data_loader import load_data

_title_year = re.compile(r"^(.*?)(?:\s*\((\d{4})\))?$")

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def _parse_title_year(q: str):
    m = _title_year.match(q.strip())
    if not m: return q.strip(), None
    t, y = m.group(1), m.group(2)
    return t.strip(), (int(y) if y else None)

def lookup_titles(query: str, topk: int = 10):
    """Return list of (movieId, title, year) best matches for a title query."""
    ratings, movies, *_ = load_data()
    movies = movies.copy()

    # split out year if provided in query
    q_title, q_year = _parse_title_year(query)
    q_norm = _normalize(q_title)

    mask = movies["title"].str.contains(q_title, case=False, na=False)
    candidates = movies[mask].copy()

    # if year provided in query, prefer those rows
    if q_year is not None:
        candidates["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
        year_hits = candidates[candidates["year"] == q_year]
        if not year_hits.empty:
            candidates = year_hits

    # If too few matches, do fuzzy across all titles
    if len(candidates) < min(5, topk):
        all_titles = movies["title"].fillna("").tolist()
        close = difflib.get_close_matches(query, all_titles, n=topk, cutoff=0.6)
        fuzz = movies[movies["title"].isin(close)]
        candidates = pd.concat([candidates, fuzz]).drop_duplicates()

    # Score by normalized string similarity (simple ratio)
    def score_row(t):
        return difflib.SequenceMatcher(a=_normalize(t), b=q_norm).ratio()

    candidates["__score"] = candidates["title"].map(score_row)
    out = (candidates
           .assign(year=candidates["title"].str.extract(r"\((\d{4})\)"))
           .sort_values(["__score", "title"], ascending=[False, True])
           .head(topk)[["movieId", "title", "year"]])

    return [(int(r.movieId), r.title, int(r.year) if pd.notna(r.year) else None)
            for r in out.itertuples(index=False)]
