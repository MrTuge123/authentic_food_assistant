import re
from functools import lru_cache
from typing import Dict, List

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - exercised when nltk is not installed
    SentimentIntensityAnalyzer = None

DISH_KEYWORDS = [
    "ramen",
    "pizza",
    "sushi",
    "dumplings",
    "burger",
    "tacos",
    "bbq",
    "noodles",
    "pho",
    "hotpot",
    "brunch",
    "fried chicken",
    "kebab",
]

DISH_PHRASES = [
    "red curry ramen",
    "shoyu ramen",
    "tonkotsu ramen",
    "hand pulled noodles",
]

LOWERCASE_RESTAURANT_PREFIXES = {
    "pho",
    "mama",
    "taqueria",
    "pizzeria",
    "bbq",
}

POSITIVE_WORDS = {
    "great",
    "amazing",
    "authentic",
    "best",
    "love",
    "fantastic",
    "delicious",
    "good",
    "solid",
    "favorite",
    "legit",
    "excellent",
    "worth",
    "stellar",
    "impressed",
    "friendly",
    "fresh",
    "top",
}

NEGATIVE_WORDS = {
    "bad",
    "overrated",
    "bland",
    "expensive",
    "worst",
    "avoid",
    "disappointing",
    "mediocre",
    "meh",
    "terrible",
    "awful",
    "weak",
}

RECOMMENDATION_CUES = {
    "recommend",
    "recommended",
    "try",
    "favorite",
    "love",
    "worth",
    "go",
    "order",
    "get",
    "solid",
    "best",
    "answer",
    "impressed",
    "favorite",
    "closest",
    "alternative",
}

GENERIC_NAME_WORDS = {
    "best",
    "downtown",
    "uptown",
    "chinatown",
    "capitol",
    "hill",
    "district",
    "heights",
    "midtown",
    "downtown",
    "broadway",
    "boone",
    "side",
    "backside",
    "seattle",
    "portland",
    "vancouver",
    "houston",
    "ann",
    "arbor",
    "austin",
    "bellevue",
    "ypsi",
    "milwaukee",
    "chicago",
    "nyc",
    "ramen",
    "sushi",
    "pizza",
    "pho",
    "bbq",
    "cocktails",
    "drinks",
    "broth",
    "tendon",
    "option",
    "options",
    "vibe",
    "atmosphere",
    "allergen",
    "friendly",
    "vegetarian",
    "vegan",
    "formal",
    "romantic",
    "date",
    "reddit",
    "authentic",
    "restaurant",
    "restaurants",
    "spot",
    "place",
    "places",
    "area",
    "town",
    "neighborhood",
    "highly",
    "personal",
    "midnight",
    "their",
    "theyre",
    "more",
    "pto",
    "rto",
    "grand",
    "rapids",
    "subaru",
    "toyota",
    "dealership",
    "dealerships",
    "nasa",
    "jsc",
    "university",
    "museum",
    "farmersmarket",
    "market",
    "gift",
    "shop",
    "repair",
}

NON_RESTAURANT_TERMS = {
    "subaru",
    "toyota",
    "dealership",
    "dealerships",
    "office",
    "administrator",
    "employees",
    "meeting",
    "meetings",
    "parking",
    "childcare",
    "pto",
    "rto",
    "adobe",
    "illustrator",
    "qgis",
    "gis",
    "layout",
    "labeling",
    "museum",
    "office",
    "university",
    "market",
    "maps",
    "county",
    "shapefiles",
    "festival",
    "festivals",
    "cartography",
    "census",
    "usda",
    "immigration",
    "dealership",
    "dealerships",
}

BAD_START_TOKENS = {
    "highly",
    "if",
    "i'd",
    "ill",
    "i'll",
    "their",
    "theyre",
    "they'd",
    "theyll",
    "the",
    "more",
    "personal",
    "both",
    "one",
    "what",
    "every",
    "incredible",
    "absolutely",
    "recently",
    "spent",
    "while",
    "edit",
    "also",
    "another",
    "totally",
    "sorry",
    "now",
    "then",
    "super",
}

TITLE_STOPWORDS = {
    "best",
    "southern",
    "authentic",
    "regional",
    "food",
    "foods",
    "traditions",
    "mapping",
    "year",
    "work",
    "america",
    "vietnam",
    "vietnamese",
    "refugees",
    "houston",
    "heights",
    "town",
    "tx",
    "seattle",
    "bellevue",
    "annarbor",
    "portland",
    "trip",
    "report",
    "visit",
    "iconic",
    "local",
    "private",
    "equity",
}

BUSINESS_SUFFIXES = {
    "pho",
    "ramen",
    "cafe",
    "kitchen",
    "house",
    "grill",
    "bbq",
    "deli",
    "bistro",
    "bar",
    "turtle",
    "sato",
    "sushi",
    "taqueria",
    "pizzeria",
    "cantina",
    "izakaya",
}

LEADING_TRIMMERS = (
    "the ",
    "a ",
    "an ",
    "at ",
    "to ",
)

TITLE_CASE_NAME = r"[A-Z0-9][\w&'’\-.]*(?:\s+[A-Z0-9][\w&'’\-.]*){0,4}"

RECOMMENDATION_PATTERNS = [
    re.compile(
        rf"(?i:(?:try|go to|recommend(?:ed)?|love|liked?|favorite(?: spot)?(?: is)?|worth checking out is))\s+(?P<name>{TITLE_CASE_NAME})(?=\s+(?:in|near|around|for)\b|[,.!?]|$)"
    ),
    re.compile(
        rf"(?:for\s+[\w\s'-]+,\s*)?(?P<name>{TITLE_CASE_NAME})\s+(?i:(?:is|was|has|serves)\s+(?:really\s+|super\s+)?(?:great|amazing|authentic|good|solid|excellent|delicious|worth it|legit))"
    ),
    re.compile(
        rf"(?P<name>{TITLE_CASE_NAME})\s+(?i:(?:for|if you want)\s+(?:the\s+)?[\w\s'-]+)"
    ),
    re.compile(
        rf"(?P<name>{TITLE_CASE_NAME})\s+(?i:is the answer(?: here)?)"
    ),
]

LOCATION_PATTERN = re.compile(
    r"(?:in|near|around|by|close to|downtown|uptown)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _extract_dishes(text: str) -> List[str]:
    lower = text.lower()
    dishes = [dish for dish in DISH_PHRASES if dish in lower]
    for dish in DISH_KEYWORDS:
        if dish in lower and dish not in dishes:
            dishes.append(dish)
    return dishes[:5]


def _extract_location_hints(text: str) -> List[str]:
    hits = LOCATION_PATTERN.findall(text)
    seen = set()
    output = []
    for hit in hits:
        cleaned = hit.strip()
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output[:3]


def _valid_location_hint(hint: str) -> bool:
    tokens = hint.lower().split()
    if not tokens:
        return False
    if any(token in {"pho", "town"} for token in tokens):
        return False
    return True


@lru_cache(maxsize=1)
def _get_sentiment_analyzer():
    if SentimentIntensityAnalyzer is None:
        return None

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        # Allow the app to keep working if vader_lexicon has not been downloaded yet.
        return None


def _fallback_sentiment_score(text: str) -> int:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    pos = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    return pos - neg


def _sentiment_score(text: str) -> float:
    analyzer = _get_sentiment_analyzer()
    if analyzer is not None:
        return analyzer.polarity_scores(text)["compound"]
    return float(_fallback_sentiment_score(text))


def _sentiment_label(text: str) -> str:
    analyzer = _get_sentiment_analyzer()
    score = _sentiment_score(text)
    if analyzer is not None:
        if score >= 0.2:
            return "positive"
        if score <= -0.2:
            return "negative"
        return "neutral"

    if score >= 2:
        return "positive"
    if score <= -2:
        return "negative"
    return "neutral"


def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\s&'-]", "", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -'")
    lower = cleaned.lower()
    for prefix in LEADING_TRIMMERS:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lower = cleaned.lower()
            break
    return cleaned


def _to_display_name(name: str) -> str:
    normalized = _normalize_name(name)
    if not normalized:
        return normalized

    words = []
    for token in normalized.split():
        if token.lower() in {"bbq"}:
            words.append(token.upper())
        elif token and token[0].isalpha():
            words.append(token[0].upper() + token[1:])
        else:
            words.append(token)
    return " ".join(words)


def _canonical_name(name: str) -> str:
    normalized = _normalize_name(name)
    return re.sub(r"[^a-z0-9]+", "", normalized.lower())


def _looks_plausible_name(name: str) -> bool:
    normalized = _normalize_name(name)
    if len(normalized) < 3:
        return False
    if normalized[0].isdigit():
        return False

    tokens = normalized.split()
    if len(tokens) > 5:
        return False

    lowered_tokens = [token.lower() for token in tokens]
    if lowered_tokens[0] in BAD_START_TOKENS:
        return False

    if all(token in GENERIC_NAME_WORDS for token in lowered_tokens):
        return False

    if lowered_tokens[0] in {"i", "we", "they", "he", "she", "it"}:
        return False

    if normalized.lower() in GENERIC_NAME_WORDS:
        return False

    if any(token in NON_RESTAURANT_TERMS for token in lowered_tokens):
        return False

    if len(tokens) == 1 and normalized.isupper():
        return False

    if len(tokens) == 1 and lowered_tokens[0] in TITLE_STOPWORDS:
        return False

    uppercase_tokens = sum(1 for token in tokens if token.isupper() and len(token) > 1)
    has_title_or_number_token = any(
        any(char.islower() for char in token) or any(char.isdigit() for char in token) for token in tokens
    )
    if uppercase_tokens >= 1 and not has_title_or_number_token:
        return False

    return any(char.isalpha() for char in normalized)


def _candidate_payload(name: str, sentence: str, confidence: float) -> Dict:
    display_name = _to_display_name(name)
    return {
        "name": display_name,
        "normalized_name": _canonical_name(name),
        "confidence": confidence,
        "evidence": sentence,
        "dishes": _extract_dishes(sentence),
        "location_hints": [
            hint for hint in _extract_location_hints(sentence) if _valid_location_hint(hint)
        ],
        "sentiment": _sentiment_label(sentence),
    }


def _strip_trailing_dish_phrase(name: str) -> str:
    cleaned = _normalize_name(name)
    lower = cleaned.lower()
    for phrase in sorted(DISH_PHRASES, key=len, reverse=True):
        if lower.endswith(phrase):
            trimmed = cleaned[: -len(phrase)].strip(" -")
            if trimmed:
                return trimmed

    for dish in sorted(DISH_KEYWORDS, key=len, reverse=True):
        if lower.endswith(dish):
            trimmed = cleaned[: -len(dish)].strip(" -")
            if trimmed:
                return trimmed

    return cleaned


def _sentence_has_recommendation_cue(sentence: str) -> bool:
    words = set(re.findall(r"[a-zA-Z']+", sentence.lower()))
    return any(word in RECOMMENDATION_CUES for word in words)


def _is_low_quality_sentence_start_candidate(name: str, sentence: str, start_index: int) -> bool:
    if start_index != 0:
        return False

    normalized = _normalize_name(name)
    tokens = normalized.split()
    if any(char.isdigit() for char in normalized):
        return False
    if len(tokens) != 1:
        return False

    lowered = tokens[0].lower()
    if lowered in BAD_START_TOKENS or lowered in TITLE_STOPWORDS:
        return True

    remaining = sentence[len(name) :].lstrip(" ,.-:;!?")
    if not remaining:
        return False

    next_word_match = re.match(r"[A-Za-z']+", remaining)
    if not next_word_match:
        return False

    next_word = next_word_match.group(0).lower()
    if next_word in DISH_KEYWORDS or next_word in {"cocktails", "drinks", "atmosphere", "vibe", "broth"}:
        return True

    return False


def _is_title_candidate(candidate: str) -> bool:
    normalized = _normalize_name(candidate)
    if not _looks_plausible_name(normalized):
        return False

    tokens = normalized.split()
    lowered = [token.lower() for token in tokens]
    if len(tokens) > 4:
        return False

    if all(token in TITLE_STOPWORDS for token in lowered):
        return False

    if any(token in {"adobe", "illustrator", "america", "vietnam", "refugees"} for token in lowered):
        return False

    return lowered[0] in BUSINESS_SUFFIXES or lowered[-1] in BUSINESS_SUFFIXES or "pho" in lowered


def _extract_title_candidates(title: str) -> List[Dict]:
    candidates = []
    seen = set()
    clean_title = _normalize_name(title)

    patterns = [
        re.compile(rf"^(?P<name>{TITLE_CASE_NAME})(?=\s+(?:in|at)\b|$)"),
        re.compile(rf"(?i:best\s+)(?P<name>{TITLE_CASE_NAME})(?=\s+\(|\s+pho\b|\s+ramen\b|$)"),
    ]

    for pattern in patterns:
        for match in pattern.finditer(clean_title):
            name = _strip_trailing_dish_phrase(match.group("name"))
            key = _canonical_name(name)
            if not key or key in seen or not _is_title_candidate(name):
                continue

            seen.add(key)
            candidates.append(
                {
                    "name": name,
                    "normalized_name": key,
                    "confidence": 0.8,
                    "evidence": title,
                    "dishes": _extract_dishes(title),
                    "location_hints": [
                        hint for hint in _extract_location_hints(title) if _valid_location_hint(hint)
                    ],
                    "sentiment": _sentiment_label(title),
                }
            )

    return candidates


def _score_candidate(name: str, sentence: str) -> float:
    score = 0.35
    lower_sentence = sentence.lower()

    if _sentence_has_recommendation_cue(sentence):
        score += 0.25

    if any(word in lower_sentence for word in POSITIVE_WORDS):
        score += 0.15

    if any(word in lower_sentence for word in NEGATIVE_WORDS):
        score -= 0.2

    if _extract_dishes(sentence):
        score += 0.1

    if _extract_location_hints(sentence):
        score += 0.05

    if len(name.split()) >= 2:
        score += 0.05

    return max(0.0, min(score, 1.0))


def _extract_list_candidates(sentence: str) -> List[str]:
    lowered = sentence.lower()
    matches = []

    like_match = re.search(r"(?i)\bi like\s+(.+)", sentence)
    if like_match:
        tail = like_match.group(1)
        tail = re.split(r"[.!?]", tail)[0]
        for part in re.split(r",| and ", tail):
            candidate = _normalize_name(part)
            if candidate:
                matches.append(candidate)

    short_line = _normalize_name(sentence)
    if short_line and len(short_line.split()) <= 3:
        first = short_line.split()[0].lower()
        if first in LOWERCASE_RESTAURANT_PREFIXES or sentence.strip() == short_line:
            matches.append(short_line)

    output = []
    seen = set()
    for match in matches:
        key = _canonical_name(match)
        if not key or key in seen or not _looks_plausible_name(match):
            continue
        seen.add(key)
        output.append(match)

    return output


def _find_candidate_mentions(sentence: str) -> List[Dict]:
    candidates = []
    seen = set()
    if any(term in sentence.lower() for term in NON_RESTAURANT_TERMS):
        return []

    sentence_location_hints = {
        hint.lower() for hint in _extract_location_hints(sentence) if _valid_location_hint(hint)
    }

    for pattern in RECOMMENDATION_PATTERNS:
        for match in pattern.finditer(sentence):
            name = _strip_trailing_dish_phrase(match.group("name"))
            key = _canonical_name(name)
            if (
                not key
                or key in seen
                or not _looks_plausible_name(name)
                or name.lower() in sentence_location_hints
            ):
                continue

            seen.add(key)
            candidates.append(_candidate_payload(name, sentence, _score_candidate(name, sentence)))

    for name in _extract_list_candidates(sentence):
        key = _canonical_name(name)
        if (
            not key
            or key in seen
            or not _looks_plausible_name(name)
            or name.lower() in sentence_location_hints
            or _is_low_quality_sentence_start_candidate(name, sentence, 0)
        ):
            continue

        seen.add(key)
        candidates.append(_candidate_payload(name, sentence, max(0.4, _score_candidate(name, sentence) - 0.05)))

    # Fallback: a recommendation sentence with a capitalized phrase often still names a place.
    if not candidates and _sentence_has_recommendation_cue(sentence):
        for match in re.finditer(TITLE_CASE_NAME, sentence):
            name = _strip_trailing_dish_phrase(match.group(0))
            key = _canonical_name(name)
            if (
                not key
                or key in seen
                or not _looks_plausible_name(name)
                or name.lower() in sentence_location_hints
                or _is_low_quality_sentence_start_candidate(name, sentence, match.start())
            ):
                continue

            seen.add(key)
            candidates.append(
                _candidate_payload(name, sentence, max(0.3, _score_candidate(name, sentence) - 0.1))
            )

    return candidates


def extract_structured(title: str, comments: List[str]) -> Dict:
    title_sentences = _split_sentences(title)
    comment_sentences = []
    for comment in comments:
        comment_sentences.extend(_split_sentences(comment))

    candidate_map: Dict[str, Dict] = {}

    for candidate in _extract_title_candidates(title):
        key = candidate["normalized_name"]
        candidate_map[key] = {
            "name": candidate["name"],
            "normalized_name": key,
            "confidence": candidate["confidence"],
            "evidence": [candidate["evidence"]],
            "dishes": list(candidate["dishes"]),
            "location_hints": list(candidate["location_hints"]),
            "sentiment_counts": {candidate["sentiment"]: 1},
            "mention_count": 1,
        }

    for sentence in comment_sentences:
        for candidate in _find_candidate_mentions(sentence):
            key = candidate["normalized_name"]
            existing = candidate_map.get(key)
            if not existing:
                candidate_map[key] = {
                    "name": candidate["name"],
                    "normalized_name": key,
                    "confidence": candidate["confidence"],
                    "evidence": [candidate["evidence"]],
                    "dishes": list(candidate["dishes"]),
                    "location_hints": list(candidate["location_hints"]),
                    "sentiment_counts": {candidate["sentiment"]: 1},
                    "mention_count": 1,
                }
                continue

            existing["mention_count"] += 1
            existing["confidence"] = max(existing["confidence"], candidate["confidence"])
            if candidate["evidence"] not in existing["evidence"]:
                existing["evidence"].append(candidate["evidence"])

            for dish in candidate["dishes"]:
                if dish not in existing["dishes"]:
                    existing["dishes"].append(dish)

            for location_hint in candidate["location_hints"]:
                if location_hint not in existing["location_hints"]:
                    existing["location_hints"].append(location_hint)

            sentiment = candidate["sentiment"]
            existing["sentiment_counts"][sentiment] = existing["sentiment_counts"].get(sentiment, 0) + 1

    restaurant_candidates = []
    for candidate in candidate_map.values():
        sentiment_counts = candidate["sentiment_counts"]
        dominant_sentiment = max(
            sentiment_counts,
            key=lambda key: (sentiment_counts[key], key == "positive"),
        )
        restaurant_candidates.append(
            {
                "name": candidate["name"],
                "normalized_name": candidate["normalized_name"],
                "confidence": round(candidate["confidence"], 2),
                "mention_count": candidate["mention_count"],
                "dishes": candidate["dishes"][:5],
                "location_hints": candidate["location_hints"][:3],
                "sentiment": dominant_sentiment,
                "evidence": candidate["evidence"][:3],
            }
        )

    restaurant_candidates.sort(
        key=lambda item: (item["mention_count"], item["confidence"]),
        reverse=True,
    )

    joined = f"{title}\n" + "\n".join(comments)
    return {
        "restaurant_name": [candidate["name"] for candidate in restaurant_candidates[:5]],
        "restaurant_candidates": restaurant_candidates[:10],
        "dish": _extract_dishes(joined),
        "location_hint": [hint for hint in _extract_location_hints(joined) if _valid_location_hint(hint)],
        "sentiment": _sentiment_label(joined),
    }
