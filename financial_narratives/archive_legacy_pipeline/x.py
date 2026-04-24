import requests
from x_api_config import get_x_token


API_URL = "https://api.x.com/2/tweets/search/recent"


def main():
    bearer = get_x_token()
    if not bearer:
        raise EnvironmentError(
            "No X token found. Run `python3 configure_x_api.py` or set X_BEARER_TOKEN."
        )

    query = "Federal Reserve lang:en -is:retweet"
    params = {
        "query": query,
        "max_results": 10,
        "tweet.fields": "created_at,author_id,public_metrics",
    }

    response = requests.get(
        API_URL,
        headers={"Authorization": f"Bearer {bearer}"},
        params=params,
        timeout=30,
    )
    if response.status_code == 402:
        print("X API returned 402 Payment Required.")
        print("Your X developer project likely needs paid credits/access for this endpoint.")
        return
    response.raise_for_status()
    payload = response.json()

    tweets = payload.get("data", [])
    print(f"Fetched {len(tweets)} tweets.")
    for t in tweets[:5]:
        text = (t.get("text") or "").replace("\n", " ")
        print(f"- {t.get('id')}: {text[:120]}")


if __name__ == "__main__":
    main()
