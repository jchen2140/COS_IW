import os
import time
import argparse
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from x_api_config import get_x_token


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVENT_REGISTRY_FILE = os.path.join(BASE_DIR, "event_registry_sentivent.csv")
WSJ_DIR = os.path.join(BASE_DIR, "../wsj_articles")
OFFICIAL_DIR = os.path.join(BASE_DIR, "../official_docs")
COMMENTS_OUTPUT_FILE = os.path.join(BASE_DIR, "x_financial_narratives_master.csv")
THREAD_REPORT_FILE = os.path.join(BASE_DIR, "x_thread_selection_report.csv")

REQUEST_TIMEOUT = 30
REQUEST_PAUSE_SECONDS = 1.0
SEARCH_WINDOW_BEFORE_DAYS = 1
SEARCH_WINDOW_AFTER_DAYS = 7
MAX_RESULTS_PER_PAGE = 100
MAX_PAGES_PER_EVENT = 10
MAX_CONVERSATIONS_FOR_REPLIES = 40
MAX_REPLY_PAGES_PER_CONVERSATION = 2


def _require_bearer_token() -> str:
    token = get_x_token()
    if not token:
        raise EnvironmentError(
            "Missing X token. Run `python3 configure_x_api.py` "
            "or export X_BEARER_TOKEN='YOUR_TOKEN_HERE'."
        )
    return token


def load_events_registry(
    event_registry_path: str,
    require_source_docs: bool = True,
    allow_missing_event_dates: bool = False,
) -> List[Dict]:
    if not os.path.exists(event_registry_path):
        raise FileNotFoundError(f"Event registry not found: {event_registry_path}")

    events_df = pd.read_csv(event_registry_path)
    required = {"event", "event_date"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Event registry missing required columns: {sorted(missing)}")

    if "x_query" not in events_df.columns:
        raise ValueError("Event registry must include 'x_query'.")

    if "x_query" not in events_df.columns:
        events_df["x_query"] = ""
    if "x_language" not in events_df.columns:
        events_df["x_language"] = "en"

    events_df["event_date"] = pd.to_datetime(events_df["event_date"], errors="coerce", utc=True)
    if not allow_missing_event_dates:
        events_df = events_df.dropna(subset=["event_date"])

    def _resolve_query(row: pd.Series) -> str:
        x_q = str(row.get("x_query", "")).strip()
        return x_q

    events_df["resolved_query"] = events_df.apply(_resolve_query, axis=1)
    events_df = events_df[events_df["resolved_query"].str.strip() != ""]

    # Optional filtering for legacy workflow where each event must map to source docs.
    if require_source_docs:
        def _has_existing_source_docs(row: pd.Series) -> bool:
            wsj_file = str(row.get("wsj_file", "") or "").strip()
            official_file = str(row.get("official_file", "") or "").strip()
            wsj_ok = bool(wsj_file) and os.path.exists(os.path.join(WSJ_DIR, wsj_file))
            official_ok = bool(official_file) and os.path.exists(os.path.join(OFFICIAL_DIR, official_file))
            return wsj_ok or official_ok

        events_df = events_df[events_df.apply(_has_existing_source_docs, axis=1)]

    return events_df.to_dict("records")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect X posts/replies for events in an event registry."
    )
    parser.add_argument(
        "--event-registry",
        default=EVENT_REGISTRY_FILE,
        help="Path to CSV with columns: event,event_date,x_query (plus optional metadata).",
    )
    parser.add_argument(
        "--output",
        default=COMMENTS_OUTPUT_FILE,
        help="Output CSV path for collected X rows.",
    )
    parser.add_argument(
        "--report-output",
        default=THREAD_REPORT_FILE,
        help="Output CSV path for collection report.",
    )
    parser.add_argument(
        "--allow-missing-source-docs",
        action="store_true",
        help="Do not require wsj_file/official_file to exist (recommended for SENTiVENT workflows).",
    )
    parser.add_argument(
        "--allow-missing-event-dates",
        action="store_true",
        help="Allow rows with blank event_date and run query-only collection windows.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Optional cap on number of events to process (0 means all).",
    )
    parser.add_argument(
        "--search-endpoint",
        choices=["auto", "recent", "all"],
        default="auto",
        help="Endpoint mode: auto (default), recent, or all (full archive).",
    )
    parser.add_argument(
        "--global-start-date",
        default="",
        help="Optional YYYY-MM-DD start date used when event_date is missing.",
    )
    parser.add_argument(
        "--global-end-date",
        default="",
        help="Optional YYYY-MM-DD end date used when event_date is missing.",
    )
    parser.add_argument(
        "--max-pages-per-event",
        type=int,
        default=MAX_PAGES_PER_EVENT,
        help="Max pagination pages for the base search per event.",
    )
    parser.add_argument(
        "--max-conversations-for-replies",
        type=int,
        default=MAX_CONVERSATIONS_FOR_REPLIES,
        help="How many top conversations to expand for replies per event.",
    )
    parser.add_argument(
        "--max-reply-pages-per-conversation",
        type=int,
        default=MAX_REPLY_PAGES_PER_CONVERSATION,
        help="Max pagination pages when collecting replies for one conversation.",
    )
    parser.add_argument(
        "--skip-replies",
        action="store_true",
        help="Skip reply expansion entirely (most API-credit friendly).",
    )
    parser.add_argument(
        "--max-results-per-page",
        type=int,
        default=MAX_RESULTS_PER_PAGE,
        help="API page size for tweet search (10-100).",
    )
    parser.add_argument(
        "--max-tweets-per-event",
        type=int,
        default=0,
        help="Hard cap on total tweets retained per event after de-dup (0 means no cap).",
    )
    parser.add_argument(
        "--request-pause-seconds",
        type=float,
        default=REQUEST_PAUSE_SECONDS,
        help="Pause between API requests to reduce rate limiting.",
    )
    return parser.parse_args()


def _build_query(raw_query: str, lang: str) -> str:
    q = raw_query.strip()
    if "-is:retweet" not in q:
        q = f"{q} -is:retweet"
    if lang and f"lang:{lang}" not in q:
        q = f"{q} lang:{lang}"
    return q


def _iso(dt: pd.Timestamp) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def x_get(url: str, token: str, params: Optional[Dict] = None) -> Optional[Dict]:
    response = requests.get(
        url,
        headers=_headers(token),
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 402:
        print("    API error 402 (Payment Required): account/app lacks required X API access credits.")
        return None
    if response.status_code != 200:
        print(f"    API error {response.status_code}: {response.text[:180]}")
        return None
    try:
        return response.json()
    except Exception:
        return None


def _choose_endpoint(event_date: pd.Timestamp) -> str:
    # For old events, recent search will fail due to the 7-day limit.
    now_utc = pd.Timestamp.now(tz="UTC")
    if event_date.tzinfo is None:
        event_date = event_date.tz_localize("UTC")
    age_days = (now_utc - event_date).days
    if age_days > 7:
        return "https://api.x.com/2/tweets/search/all"
    return "https://api.x.com/2/tweets/search/recent"


def _parse_optional_date(value: str) -> Optional[pd.Timestamp]:
    text = str(value or "").strip()
    if not text:
        return None
    dt = pd.to_datetime(text, errors="coerce", utc=True)
    return None if pd.isna(dt) else dt


def _search_tweets(
    endpoint: str,
    token: str,
    query: str,
    window_start: Optional[pd.Timestamp],
    window_end: Optional[pd.Timestamp],
    max_pages: int,
    max_results_per_page: int = MAX_RESULTS_PER_PAGE,
    max_tweets_cap: int = 0,
    request_pause_seconds: float = REQUEST_PAUSE_SECONDS,
) -> List[Dict]:
    rows: List[Dict] = []
    next_token = None
    page = 0

    page_size = max(10, min(100, int(max_results_per_page)))

    while page < max_pages:
        params = {
            "query": query,
            "max_results": page_size,
            "tweet.fields": "author_id,created_at,conversation_id,public_metrics,lang,referenced_tweets",
            "user.fields": "username,name",
            "expansions": "author_id",
        }
        if window_start is not None:
            params["start_time"] = _iso(window_start)
        if window_end is not None:
            params["end_time"] = _iso(window_end)
        if next_token:
            params["next_token"] = next_token

        payload = x_get(endpoint, token, params=params)
        time.sleep(max(0.0, float(request_pause_seconds)))
        if not payload:
            break

        users_by_id = {
            u.get("id"): u for u in payload.get("includes", {}).get("users", []) if u.get("id")
        }

        tweets = payload.get("data", [])
        for tweet in tweets:
            author_id = tweet.get("author_id")
            user = users_by_id.get(author_id, {})
            created_at = pd.to_datetime(tweet.get("created_at"), errors="coerce")
            if pd.isna(created_at):
                continue

            metrics = tweet.get("public_metrics", {}) or {}
            like_count = int(metrics.get("like_count", 0) or 0)
            reply_count = int(metrics.get("reply_count", 0) or 0)
            repost_count = int(metrics.get("retweet_count", 0) or 0)
            quote_count = int(metrics.get("quote_count", 0) or 0)
            score = like_count + reply_count + repost_count + quote_count

            tweet_id = str(tweet.get("id", "")).strip()
            convo_id = str(tweet.get("conversation_id", tweet_id)).strip()
            thread_url = f"https://x.com/i/web/status/{convo_id}" if convo_id else ""
            referenced = tweet.get("referenced_tweets", []) or []
            is_reply = any(str(x.get("type", "")).lower() == "replied_to" for x in referenced)

            rows.append(
                {
                    "id": tweet_id,
                    "parent_id": convo_id,
                    "author": user.get("username", author_id),
                    "body": tweet.get("text", ""),
                    "score": score,
                    "date": created_at.isoformat(),
                    "thread_url": thread_url,
                    "lang": tweet.get("lang"),
                    "reply_count": reply_count,
                    "record_type": "reply" if is_reply else "post",
                }
            )

        meta = payload.get("meta", {}) or {}
        next_token = meta.get("next_token")
        page += 1
        if max_tweets_cap and len(rows) >= max_tweets_cap:
            break
        if not next_token:
            break

    return rows


def _search_replies_for_conversation(
    endpoint: str,
    token: str,
    conversation_id: str,
    lang: str,
    window_start: Optional[pd.Timestamp],
    window_end: Optional[pd.Timestamp],
    max_reply_pages_per_conversation: int,
    max_results_per_page: int,
    max_tweets_cap: int = 0,
    request_pause_seconds: float = REQUEST_PAUSE_SECONDS,
) -> List[Dict]:
    reply_query = f"conversation_id:{conversation_id} -is:retweet lang:{lang}"
    rows = _search_tweets(
        endpoint=endpoint,
        token=token,
        query=reply_query,
        window_start=window_start,
        window_end=window_end,
        max_pages=max_reply_pages_per_conversation,
        max_results_per_page=max_results_per_page,
        max_tweets_cap=max_tweets_cap,
        request_pause_seconds=request_pause_seconds,
    )
    # Remove the original root post if returned.
    return [r for r in rows if str(r.get("id")) != str(conversation_id)]


def search_tweets_for_event(
    event: Dict,
    token: str,
    endpoint_mode: str = "auto",
    global_start_date: Optional[pd.Timestamp] = None,
    global_end_date: Optional[pd.Timestamp] = None,
    max_pages_per_event: int = MAX_PAGES_PER_EVENT,
    max_conversations_for_replies: int = MAX_CONVERSATIONS_FOR_REPLIES,
    max_reply_pages_per_conversation: int = MAX_REPLY_PAGES_PER_CONVERSATION,
    skip_replies: bool = False,
    max_results_per_page: int = MAX_RESULTS_PER_PAGE,
    max_tweets_per_event: int = 0,
    request_pause_seconds: float = REQUEST_PAUSE_SECONDS,
) -> Tuple[List[Dict], str]:
    event_name = str(event["event"])
    event_label = str(event.get("event_label", event_name))
    event_date = event.get("event_date")
    lang = str(event.get("x_language", "en")).strip() or "en"
    query = _build_query(str(event["resolved_query"]), lang)

    if pd.notna(event_date):
        window_start = event_date - timedelta(days=SEARCH_WINDOW_BEFORE_DAYS)
        window_end = event_date + timedelta(days=SEARCH_WINDOW_AFTER_DAYS)
        auto_endpoint = _choose_endpoint(event_date)
    else:
        # Date-free mode: use optional global window when provided.
        window_start = global_start_date
        window_end = global_end_date
        auto_endpoint = "https://api.x.com/2/tweets/search/recent"

    if endpoint_mode == "all":
        endpoint = "https://api.x.com/2/tweets/search/all"
    elif endpoint_mode == "recent":
        endpoint = "https://api.x.com/2/tweets/search/recent"
    else:
        endpoint = auto_endpoint
    endpoint_name = "search/all" if endpoint.endswith("/all") else "search/recent"

    base_rows = _search_tweets(
        endpoint=endpoint,
        token=token,
        query=query,
        window_start=window_start,
        window_end=window_end,
        max_pages=max_pages_per_event,
        max_results_per_page=max_results_per_page,
        max_tweets_cap=max_tweets_per_event,
        request_pause_seconds=request_pause_seconds,
    )

    reply_rows: List[Dict] = []
    if not skip_replies:
        # Pull replies from top conversations so output contains post + comment-like rows.
        post_rows = [r for r in base_rows if r.get("record_type") == "post"]
        post_rows_sorted = sorted(post_rows, key=lambda r: int(r.get("reply_count", 0) or 0), reverse=True)
        top_conversations = []
        seen: Set[str] = set()
        for row in post_rows_sorted:
            convo = str(row.get("parent_id", "")).strip()
            if not convo or convo in seen:
                continue
            seen.add(convo)
            top_conversations.append(convo)
            if len(top_conversations) >= max_conversations_for_replies:
                break

        for convo in top_conversations:
            remaining = 0
            if max_tweets_per_event:
                remaining = max_tweets_per_event - len(base_rows) - len(reply_rows)
                if remaining <= 0:
                    break
            reply_rows.extend(
                _search_replies_for_conversation(
                    endpoint=endpoint,
                    token=token,
                    conversation_id=convo,
                    lang=lang,
                    window_start=window_start,
                    window_end=window_end,
                    max_reply_pages_per_conversation=max_reply_pages_per_conversation,
                    max_results_per_page=max_results_per_page,
                    max_tweets_cap=remaining if remaining > 0 else 0,
                    request_pause_seconds=request_pause_seconds,
                )
            )

    # Merge + de-dup by tweet id.
    merged: Dict[str, Dict] = {}
    for row in base_rows + reply_rows:
        rid = str(row.get("id", "")).strip()
        if not rid:
            continue
        merged[rid] = row

    all_rows: List[Dict] = []
    for row in merged.values():
        all_rows.append(
            {
                "event": event_name,
                "event_label": event_label,
                "event_date": event_date.strftime("%Y-%m-%d") if pd.notna(event_date) else "",
                "thread_source": "x",
                "thread_url": row.get("thread_url", ""),
                "id": row.get("id", ""),
                "parent_id": row.get("parent_id", ""),
                "author": row.get("author", ""),
                "body": row.get("body", ""),
                "score": row.get("score", 0),
                "date": row.get("date", ""),
                "query_used": query,
                "lang": row.get("lang", ""),
                "record_type": row.get("record_type", "post"),
            }
        )
    if max_tweets_per_event and len(all_rows) > max_tweets_per_event:
        # Keep highest-engagement rows under the per-event cap.
        all_rows = sorted(all_rows, key=lambda r: int(r.get("score", 0) or 0), reverse=True)[:max_tweets_per_event]

    return all_rows, endpoint_name


def main():
    args = _parse_args()
    token = _require_bearer_token()
    global_start_date = _parse_optional_date(args.global_start_date)
    global_end_date = _parse_optional_date(args.global_end_date)
    events = load_events_registry(
        event_registry_path=args.event_registry,
        require_source_docs=not args.allow_missing_source_docs,
        allow_missing_event_dates=args.allow_missing_event_dates,
    )
    if args.max_events and args.max_events > 0:
        events = events[: args.max_events]
    print(f"Starting X collection for {len(events)} events from {args.event_registry}...")

    all_data: List[Dict] = []
    report_rows: List[Dict] = []

    for event in events:
        event_name = str(event["event"])
        event_label = str(event.get("event_label", event_name))
        event_date = event["event_date"]
        resolved_query = str(event["resolved_query"])

        date_display = event_date.date() if pd.notna(event_date) else "no-date"
        print(f"\nSearching X for {event_name} ({date_display}) with query: {resolved_query}")
        rows, endpoint_name = search_tweets_for_event(
            event,
            token,
            endpoint_mode=args.search_endpoint,
            global_start_date=global_start_date,
            global_end_date=global_end_date,
            max_pages_per_event=max(1, int(args.max_pages_per_event)),
            max_conversations_for_replies=max(0, int(args.max_conversations_for_replies)),
            max_reply_pages_per_conversation=max(1, int(args.max_reply_pages_per_conversation)),
            skip_replies=bool(args.skip_replies),
            max_results_per_page=max(10, min(100, int(args.max_results_per_page))),
            max_tweets_per_event=max(0, int(args.max_tweets_per_event)),
            request_pause_seconds=max(0.0, float(args.request_pause_seconds)),
        )
        all_data.extend(rows)

        unique_threads = len({r.get("thread_url") for r in rows if r.get("thread_url")})
        report_rows.append(
            {
                "event": event_name,
                "event_label": event_label,
                "event_date": event_date.strftime("%Y-%m-%d") if pd.notna(event_date) else "",
                "status": "collected" if rows else "no_results",
                "selected_source": "x",
                "selected_title": "",
                "selected_url": "",
                "selected_ranking_score": "",
                "candidate_count": len(rows),
                "search_endpoint": endpoint_name,
                "query_used": _build_query(resolved_query, str(event.get("x_language", "en")).strip() or "en"),
                "unique_threads": unique_threads,
            }
        )

        print(f"    Collected {len(rows)} tweets ({unique_threads} conversations).")

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(args.output, index=False)
        print(f"\nDONE! Saved {len(df)} total rows to {args.output}")
    else:
        print("\nNo tweet data collected. Check API tier permissions and query windows.")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(args.report_output, index=False)
    print(f"Saved collection report to {args.report_output}")


if __name__ == "__main__":
    main()
