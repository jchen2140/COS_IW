import argparse
import html
import os
import re
from typing import List

import pandas as pd

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_FILE = os.path.join(BASE_DIR, "x_financial_narratives_master.csv")
DEFAULT_SAMPLE_FILE = os.path.join(BASE_DIR, "x_data_sample.csv")
DEFAULT_OUTPUT_FILE = os.path.join(BASE_DIR, "x_data_cleaned.csv")
DEFAULT_EVENT_COUNTS_FILE = os.path.join(BASE_DIR, "x_event_counts.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean collected X posts for narrative analysis.")
    parser.add_argument(
        "--input",
        default="",
        help="Input CSV path. If omitted, falls back to x_financial_narratives_master.csv then x_data_sample.csv.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Output CSV path for cleaned rows.",
    )
    parser.add_argument(
        "--event-counts-output",
        default=DEFAULT_EVENT_COUNTS_FILE,
        help="Output CSV path for per-event counts after cleaning.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="Minimum word count for keeping a post.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate rows by id (if available) and clean_body.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Unescape HTML (e.g., convert "&amp;" to "&")
    text = html.unescape(text)

    # 2. Remove URLs (http/https)
    text = re.sub(r"http\S+", "", text)

    # 3. Remove markdown link formatting (e.g., [label](url))
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)

    # 4. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_input(input_path: str) -> str:
    if input_path:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return input_path

    if os.path.exists(DEFAULT_INPUT_FILE):
        return DEFAULT_INPUT_FILE
    if os.path.exists(DEFAULT_SAMPLE_FILE):
        return DEFAULT_SAMPLE_FILE
    raise FileNotFoundError(
        "No source file found. Expected one of: "
        f"{DEFAULT_INPUT_FILE}, {DEFAULT_SAMPLE_FILE}."
    )


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def main() -> None:
    args = parse_args()
    input_file = resolve_input(args.input)
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    initial_count = len(df)
    print(f"Original row count: {initial_count}")

    if "body" not in df.columns:
        raise ValueError(f"Input file must include a 'body' column: {input_file}")

    # 1) Remove empty/deleted rows.
    df = df.dropna(subset=["body"])
    df = df[~df["body"].isin(["[deleted]", "[removed]"])]
    print(f"Rows after removing empty/deleted: {len(df)}")

    # 2) Clean text.
    print("Cleaning text (removing URLs/artifacts)...")
    df["clean_body"] = df["body"].apply(clean_text)
    df = df[df["clean_body"].str.len() > 0]

    # 3) Min length filter.
    df["word_count"] = df["clean_body"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= max(1, args.min_words)]
    print(f"Rows after length filter (min {max(1, args.min_words)} words): {len(df)}")

    # 4) Optional de-dup.
    if args.dedupe:
        before = len(df)
        subset = []
        if "id" in df.columns:
            subset.append("id")
        subset.append("clean_body")
        df = df.drop_duplicates(subset=subset, keep="first")
        print(f"Rows after de-dup: {len(df)} (removed {before - len(df)})")

    if "event" not in df.columns:
        df["event"] = "sample_event"

    # 5) Save cleaned output.
    final_columns = ["event", "id", "author", "date", "score", "clean_body", "record_type", "query_used"]
    out_cols = ensure_columns(df, final_columns)
    df[out_cols].to_csv(args.output, index=False)

    # 6) Save event counts (drops zero-tweet events naturally).
    event_counts = (
        df.groupby("event", as_index=False)
        .size()
        .rename(columns={"size": "tweet_count"})
        .sort_values("tweet_count", ascending=False)
    )
    event_counts.to_csv(args.event_counts_output, index=False)

    print(f"\nSUCCESS! Cleaned data saved to: {args.output}")
    print(f"Per-event counts saved to: {args.event_counts_output}")
    print(f"Total rows removed: {initial_count - len(df)}")
    print(f"Events with >=1 tweet: {len(event_counts)}")


if __name__ == "__main__":
    main()