import requests
import re
from .extraction import extract_structured

HEADERS = {"User-Agent": "authentic-food-assistant/0.1"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

FOOD_TERMS = {
    "restaurant",
    "restaurants",
    "food",
    "eat",
    "eating",
    "dinner",
    "lunch",
    "breakfast",
    "brunch",
    "cafe",
    "spot",
    "spots",
    "dish",
    "dishes",
    "ramen",
    "pho",
    "noodles",
    "pizza",
    "burger",
    "sushi",
    "bbq",
    "dumplings",
    "authentic",
    "best",
    "recommend",
}

NEGATIVE_CONTEXT_TERMS = {
    "subaru",
    "toyota",
    "dealership",
    "dealerships",
    "office",
    "administrator",
    "employees",
    "march",
    "protest",
    "return to office",
    "parking",
    "childcare",
}

LOCATION_STOPWORDS = {
    "best",
    "authentic",
    "near",
    "downtown",
    "local",
    "restaurant",
    "restaurants",
    "food",
    "foods",
    "ramen",
    "pho",
    "pizza",
    "sushi",
    "bbq",
    "burger",
    "brunch",
    "in",
}

CITY_SUBREDDIT_HINTS = {
    "houston": {"houstonfood", "houston", "askhouston"},
    "ann": {"annarbor", "annarbor"},
    "arbor": {"annarbor"},
    "seattle": {"seattle", "seattlewa"},
    "austin": {"austinfood", "austin"},
}


def _extract_query_locations(query: str):
    tokens = re.findall(r"[a-zA-Z]+", query.lower())
    return [token for token in tokens if len(token) > 2 and token not in LOCATION_STOPWORDS]


def _location_relevance_score(query: str, title: str, comments, subreddit: str):
    query_locations = _extract_query_locations(query)
    if not query_locations:
        return 0

    lowered = f"{title}\n{subreddit}\n" + "\n".join(comments[:3])
    lowered = lowered.lower()
    subreddit_lower = subreddit.lower()

    score = 0
    for location in query_locations:
        if location in lowered:
            score += 2
        if location in subreddit_lower:
            score += 3

    return score


def _subreddit_matches_query_location(query: str, subreddit: str):
    query_locations = _extract_query_locations(query)
    if not query_locations:
        return True

    subreddit_lower = subreddit.lower()
    hinted_subreddits = set()
    for location in query_locations:
        hinted_subreddits.update(CITY_SUBREDDIT_HINTS.get(location, set()))

    if hinted_subreddits:
        return subreddit_lower in hinted_subreddits

    return any(location in subreddit_lower for location in query_locations)
def _food_intent_score(query: str, title: str, comments):
    text = f"{title}\n" + "\n".join(comments[:3])
    lowered = text.lower()
    tokens = set(re.findall(r"[a-zA-Z']+", lowered))
    query_tokens = set(re.findall(r"[a-zA-Z']+", query.lower()))

    score = 0
    score += sum(1 for term in FOOD_TERMS if term in tokens)
    score += sum(2 for token in query_tokens if token in tokens and len(token) > 2)
    score -= sum(2 for term in NEGATIVE_CONTEXT_TERMS if term in lowered)
    if "ramen" in query.lower() and "ramen" in lowered:
        score += 3
    return score


def _is_food_relevant(query: str, title: str, comments):
    score = _food_intent_score(query, title, comments)
    return score >= 2


def _merge_restaurant_results(posts):
    restaurant_map = {}

    for post in posts:
        structured = post.get("structured", {})
        for candidate in structured.get("restaurant_candidates", []):
            key = candidate.get("normalized_name")
            if not key:
                continue

            existing = restaurant_map.get(key)
            if not existing:
                restaurant_map[key] = {
                    "name": candidate.get("name", ""),
                    "normalized_name": key,
                    "score": round(
                        candidate.get("confidence", 0.0) + min(post.get("score", 0), 200) / 200.0,
                        2,
                    ),
                    "mention_count": candidate.get("mention_count", 0),
                    "post_count": 1,
                    "subreddits": [post.get("subreddit", "")] if post.get("subreddit") else [],
                    "dishes": list(candidate.get("dishes", [])),
                    "location_hints": list(candidate.get("location_hints", [])),
                    "evidence": candidate.get("evidence", [])[:2],
                    "source_urls": [post.get("url")] if post.get("url") else [],
                    "sentiment_counts": {candidate.get("sentiment", "neutral"): 1},
                }
                continue

            existing["mention_count"] += candidate.get("mention_count", 0)
            existing["post_count"] += 1
            existing["score"] = round(
                existing["score"] + candidate.get("confidence", 0.0) + min(post.get("score", 0), 200) / 250.0,
                2,
            )

            subreddit = post.get("subreddit")
            if subreddit and subreddit not in existing["subreddits"]:
                existing["subreddits"].append(subreddit)

            for dish in candidate.get("dishes", []):
                if dish not in existing["dishes"]:
                    existing["dishes"].append(dish)

            for location_hint in candidate.get("location_hints", []):
                if location_hint not in existing["location_hints"]:
                    existing["location_hints"].append(location_hint)

            for snippet in candidate.get("evidence", []):
                if snippet not in existing["evidence"]:
                    existing["evidence"].append(snippet)

            source_url = post.get("url")
            if source_url and source_url not in existing["source_urls"]:
                existing["source_urls"].append(source_url)

            sentiment = candidate.get("sentiment", "neutral")
            existing["sentiment_counts"][sentiment] = existing["sentiment_counts"].get(sentiment, 0) + 1

    restaurants = []
    for restaurant in restaurant_map.values():
        sentiment = max(
            restaurant["sentiment_counts"],
            key=lambda key: (restaurant["sentiment_counts"][key], key == "positive"),
        )
        restaurants.append(
            {
                "name": restaurant["name"],
                "score": restaurant["score"],
                "mention_count": restaurant["mention_count"],
                "post_count": restaurant["post_count"],
                "sentiment": sentiment,
                "dishes": restaurant["dishes"][:5],
                "location_hints": restaurant["location_hints"][:3],
                "subreddits": restaurant["subreddits"],
                "evidence": restaurant["evidence"][:3],
                "source_urls": restaurant["source_urls"][:5],
            }
        )

    restaurants.sort(
        key=lambda item: (item["score"], item["mention_count"], item["post_count"]),
        reverse=True,
    )
    return restaurants[:10]

def get_comments(permalink: str, limit: int = 5):
    post_json_url = f"https://www.reddit.com{permalink}.json"

    try:
        response = SESSION.get(post_json_url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    comments = []
    try:
        for comment in data[1]["data"]["children"]:
            if comment.get("kind") == "t1":
                body = comment.get("data", {}).get("body", "")
                if body:
                    comments.append(body)
            if len(comments) >= limit:
                break
    except Exception:
        return []

    return comments

def search_reddit(query: str, limit: int = 15, debug: bool = False):
    url = "https://www.reddit.com/search.json"
    params = {"q": query, "limit": limit, "sort": "relevance", "t": "year"}

    try:
        response = SESSION.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        result = {"posts": [], "restaurants": []}
        if debug:
            result["debug_filters"] = [{"stage": "reddit_request", "reason": "request_failed"}]
        return result

    posts = []
    debug_filters = []
    for post in data.get("data", {}).get("children", []):
        p = post.get("data", {})
        permalink = p.get("permalink")
        if not permalink:
            if debug:
                debug_filters.append(
                    {
                        "title": p.get("title", ""),
                        "subreddit": p.get("subreddit", ""),
                        "reason": "missing_permalink",
                    }
                )
            continue

        comments = get_comments(permalink, limit=5)
        title = p.get("title", "")
        subreddit = p.get("subreddit", "")
        relevance_score = _food_intent_score(query, title, comments)
        location_score = _location_relevance_score(query, title, comments, subreddit)
        location_tokens = _extract_query_locations(query)
        subreddit_match = _subreddit_matches_query_location(query, subreddit)

        if not _is_food_relevant(query, title, comments):
            if debug:
                debug_filters.append(
                    {
                        "title": title,
                        "subreddit": subreddit,
                        "reason": "low_food_relevance",
                        "food_relevance_score": relevance_score,
                        "location_score": location_score,
                    }
                )
            continue
        if location_tokens and location_score <= 0:
            if debug:
                debug_filters.append(
                    {
                        "title": title,
                        "subreddit": subreddit,
                        "reason": "low_location_relevance",
                        "food_relevance_score": relevance_score,
                        "location_score": location_score,
                        "query_locations": location_tokens,
                    }
                )
            continue
        if location_tokens and location_score < 2 and not subreddit_match:
            if debug:
                debug_filters.append(
                    {
                        "title": title,
                        "subreddit": subreddit,
                        "reason": "subreddit_location_mismatch",
                        "food_relevance_score": relevance_score,
                        "location_score": location_score,
                        "query_locations": location_tokens,
                    }
                )
            continue

        structured = extract_structured(title, comments)
        if not structured.get("restaurant_candidates"):
            if debug:
                debug_filters.append(
                    {
                        "title": title,
                        "subreddit": subreddit,
                        "reason": "no_restaurant_candidates",
                        "food_relevance_score": relevance_score,
                        "location_score": location_score,
                    }
                )
            continue

        posts.append({
            "title": title,
            "subreddit": subreddit,
            "score": p.get("score", 0),
            "relevance_score": relevance_score,
            "location_score": location_score,
            "subreddit_match": subreddit_match,
            "comments": comments,
            "url": f"https://www.reddit.com{permalink}",
            "structured": structured
        })

    posts.sort(
        key=lambda post: (
            post.get("subreddit_match", False),
            post.get("location_score", 0),
            post.get("relevance_score", 0),
            post.get("score", 0),
        ),
        reverse=True,
    )
    result = {
        "posts": posts[:limit],
        "restaurants": _merge_restaurant_results(posts),
    }
    if debug:
        result["debug_filters"] = debug_filters
    return result
