import argparse
import ast
import os
import re
from typing import List

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "event_registry_sentivent.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build event_registry-style CSV from SENTiVENT TSV files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to SENTiVENT TSV (e.g., dataset_event_subtype.tsv or dataset_event_type.tsv).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="x_language value for generated queries.",
    )
    parser.add_argument(
        "--default-date",
        default="",
        help="Fallback YYYY-MM-DD for all rows when source has no publication date column.",
    )
    return parser.parse_args()


def _safe_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _slugify(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def _parse_listish(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    return []


def _is_none_label(label: str) -> bool:
    t = str(label).strip().lower()
    return t == "none" or t.endswith(".none")


def _headline_from_title(document_title: str) -> str:
    t = str(document_title or "").strip()
    if not t:
        return ""
    t = re.sub(r"\.txt$", "", t, flags=re.IGNORECASE)
    if "_" in t:
        t = t.split("_", 1)[1]
    return _safe_text(t.replace("-", " "))


def _extract_ticker(document_id: str) -> str:
    m = re.match(r"([a-z]+)\d+", str(document_id).strip().lower())
    if not m:
        return ""
    ticker = m.group(1).upper()
    return ticker if 1 < len(ticker) <= 5 else ""


def _company_phrase_from_headline(headline: str) -> str:
    stop = {
        "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at", "by", "with",
        "is", "are", "was", "were", "be", "as", "its", "it", "this", "that", "after", "before",
        "from", "into", "up", "down", "over", "under", "new", "nt", "why", "how", "look", "reasons",
    }
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", headline)]
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    if not tokens:
        return ""
    # Use first 2-3 informative words as a broad company/topic phrase.
    phrase = " ".join(tokens[:3])
    return phrase.strip()


def _event_keywords(event_subtype: str, event_type: str) -> List[str]:
    base = str(event_type or "").lower()
    subtype = str(event_subtype or "").lower()
    words = set(re.findall(r"[a-z]+", subtype.replace("/", " ")))
    keywords = set()

    type_map = {
        "securityvalue": {"stock", "shares", "price"},
        "financialreport": {"earnings", "results", "guidance"},
        "revenue": {"revenue", "sales"},
        "salesvolume": {"demand", "sales", "volume"},
        "profit/loss": {"profit", "margin", "loss"},
        "deal": {"deal", "contract"},
        "merger/acquisition": {"merger", "acquisition", "buyout"},
        "product/service": {"product", "launch", "trial"},
        "facility": {"plant", "facility", "expansion"},
        "employment": {"hiring", "layoffs", "workforce"},
        "legal": {"lawsuit", "court", "settlement"},
        "rating": {"upgrade", "downgrade", "rating"},
        "dividend": {"dividend", "payout"},
        "expense": {"costs", "expense"},
        "financing": {"debt", "financing"},
        "investment": {"investment", "capex"},
        "macroeconomics": {"economy", "market"},
        "csr/brand": {"brand", "reputation"},
    }
    keywords |= type_map.get(base, set())

    if "increase" in words:
        keywords |= {"surge", "rise", "up"}
    if "decrease" in words:
        keywords |= {"drop", "fall", "down"}
    if "beat" in words:
        keywords |= {"beat", "above"}
    if "miss" in words:
        keywords |= {"miss", "below"}
    if "launch" in words:
        keywords |= {"launch", "announce"}
    if "trial" in words:
        keywords |= {"trial", "study"}
    if "upgrade" in words:
        keywords |= {"upgrade", "bullish"}
    if "downgrade" in words:
        keywords |= {"downgrade", "bearish"}

    # Keep query compact.
    return sorted(k for k in keywords if k)[:4]


def _build_query(document_id: str, headline: str, event_subtype: str, event_type: str) -> str:
    terms = []
    ticker = _extract_ticker(document_id)
    if ticker:
        terms.append(f"${ticker}")
        terms.append(ticker)

    company_phrase = _company_phrase_from_headline(headline)
    if company_phrase:
        terms.append(f"\"{company_phrase}\"")

    left = " OR ".join(dict.fromkeys(terms))
    kws = _event_keywords(event_subtype, event_type)
    right = " OR ".join(dict.fromkeys(kws))

    if left and right:
        return f"({left}) ({right})"
    if left:
        return left
    if right:
        return right
    return ""


def _choose_date_col(df: pd.DataFrame) -> str:
    candidates = ["event_date", "date", "article_date", "published_at", "publish_date", "timestamp"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return ""


def main() -> None:
    args = _parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input, sep="\t")
    required = {"document_id", "document_title", "text"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            "Expected SENTiVENT sentence-level schema with columns: "
            "document_id, document_title, text."
        )

    date_col = _choose_date_col(df)
    if date_col:
        df["_event_date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    elif args.default_date:
        df["_event_date"] = pd.to_datetime(args.default_date, errors="coerce", utc=True)
    else:
        df["_event_date"] = pd.NaT

    rows = []
    for (doc_id, doc_title), g in df.groupby(["document_id", "document_title"]):
        headline = _headline_from_title(doc_title)
        type_counter = {}
        subtype_counter = {}
        for _, gr in g.iterrows():
            for t in _parse_listish(gr.get("types_event_unq", gr.get("types_event", ""))):
                if not _is_none_label(t):
                    type_counter[t] = type_counter.get(t, 0) + 1
            for st in _parse_listish(gr.get("subtypes_event_unq", gr.get("subtypes_event", ""))):
                if not _is_none_label(st):
                    subtype_counter[st] = subtype_counter.get(st, 0) + 1

        event_type = max(type_counter, key=type_counter.get) if type_counter else ""
        event_subtype = max(subtype_counter, key=subtype_counter.get) if subtype_counter else ""
        query = _build_query(str(doc_id), headline, event_subtype, event_type)
        if not query:
            continue

        doc_date = g["_event_date"].dropna().min()
        event_date = doc_date.strftime("%Y-%m-%d") if pd.notna(doc_date) else ""
        label = " | ".join([p for p in [headline, event_subtype or event_type] if p]) or str(doc_id)
        event_slug = _slugify(f"{doc_id}_{event_subtype or event_type}_{event_date or 'missing_date'}")

        rows.append(
            {
                "event": event_slug,
                "event_date": event_date,
                "event_label": label,
                "event_type": event_type or "sentivent",
                "x_query": query,
                "x_language": args.language,
                "wsj_file": "",
                "official_file": "",
                "analysis_note": f"Derived from SENTiVENT doc {doc_id} (sentences={len(g)}).",
            }
        )

    if not rows:
        raise ValueError("No events generated from input.")

    out = pd.DataFrame(rows).sort_values(["event_date", "event_label"], na_position="last")
    out.to_csv(args.output, index=False)
    missing_dates = int(out["event_date"].astype(str).str.strip().eq("").sum())
    print(f"Generated {len(out)} events -> {args.output}")
    if missing_dates:
        print(
            f"WARNING: {missing_dates} events are missing event_date. "
            "Fill event_date before scraping, or rerun with --default-date."
        )


if __name__ == "__main__":
    main()
