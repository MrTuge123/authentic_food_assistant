import json
import os
from typing import Dict, List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised when openai is not installed
    OpenAI = None


MODEL_NAME = "gpt-4.1-mini"


def _default_result() -> Dict:
    return {
        "restaurant_name": [],
        "restaurant_candidates": [],
        "dish": [],
        "location_hint": [],
        "sentiment": "neutral",
    }


def _normalize_result(payload: Dict) -> Dict:
    result = _default_result()
    result.update({key: payload.get(key, result[key]) for key in result})

    candidates = []
    for candidate in result.get("restaurant_candidates", []):
        name = candidate.get("name", "").strip()
        if not name:
            continue

        candidates.append(
            {
                "name": name,
                "normalized_name": candidate.get("normalized_name", "").strip() or "".join(
                    ch for ch in name.lower() if ch.isalnum()
                ),
                "confidence": float(candidate.get("confidence", 0.5)),
                "mention_count": int(candidate.get("mention_count", 1)),
                "dishes": list(candidate.get("dishes", []))[:5],
                "location_hints": list(candidate.get("location_hints", []))[:3],
                "sentiment": candidate.get("sentiment", "neutral"),
                "evidence": list(candidate.get("evidence", []))[:3],
            }
        )

    result["restaurant_candidates"] = candidates
    result["restaurant_name"] = result.get("restaurant_name") or [candidate["name"] for candidate in candidates[:5]]
    result["dish"] = list(result.get("dish", []))[:5]
    result["location_hint"] = list(result.get("location_hint", []))[:5]
    result["sentiment"] = result.get("sentiment", "neutral")
    return result


def extract_structured_with_llm(title: str, comments: List[str]) -> Dict:
    if OpenAI is None:
        raise ImportError("openai is not installed. Run `pip install -r requirements.txt` first.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    comment_block = "\n\n".join(f"- {comment}" for comment in comments)
    user_input = f"""Title:
{title}

Comments:
{comment_block if comment_block else "- No comments"}
"""

    instructions = """You extract restaurant recommendations from community discussion text.
Return only valid JSON with this exact top-level structure:
{
  "restaurant_name": ["..."],
  "restaurant_candidates": [
    {
      "name": "...",
      "normalized_name": "...",
      "confidence": 0.0,
      "mention_count": 1,
      "dishes": ["..."],
      "location_hints": ["..."],
      "sentiment": "positive|neutral|negative",
      "evidence": ["..."]
    }
  ],
  "dish": ["..."],
  "location_hint": ["..."],
  "sentiment": "positive|neutral|negative"
}

Rules:
- Extract only restaurants or food businesses that are explicitly mentioned.
- Do not output neighborhoods, landmarks, institutions, tools, or generic descriptive phrases as restaurants.
- Use short direct evidence snippets copied from the input.
- If a dish is attached to a restaurant name, separate it into the dish fields.
- If nothing reliable is present, return empty arrays and neutral sentiment.

Example 1
Input:
Title: Best vegetarian ramen?
Comments:
- Slurping Turtle Red Curry Ramen is the best veg option imo!
- I love mama satto’s shoyu ramen.

Output:
{
  "restaurant_name": ["Slurping Turtle", "Mama Satto"],
  "restaurant_candidates": [
    {
      "name": "Slurping Turtle",
      "normalized_name": "slurpingturtle",
      "confidence": 0.93,
      "mention_count": 1,
      "dishes": ["red curry ramen"],
      "location_hints": [],
      "sentiment": "positive",
      "evidence": ["Slurping Turtle Red Curry Ramen is the best veg option imo!"]
    },
    {
      "name": "Mama Satto",
      "normalized_name": "mamasatto",
      "confidence": 0.84,
      "mention_count": 1,
      "dishes": ["shoyu ramen"],
      "location_hints": [],
      "sentiment": "positive",
      "evidence": ["I love mama satto’s shoyu ramen."]
    }
  ],
  "dish": ["red curry ramen", "shoyu ramen", "ramen"],
  "location_hint": [],
  "sentiment": "positive"
}

Example 2
Input:
Title: Pho in southeast side
Comments:
- While not the best broth, Pho 21 on the backside of NASA JSC has the best soft tendon of anywhere in Houston.

Output:
{
  "restaurant_name": ["Pho 21"],
  "restaurant_candidates": [
    {
      "name": "Pho 21",
      "normalized_name": "pho21",
      "confidence": 0.9,
      "mention_count": 1,
      "dishes": ["pho"],
      "location_hints": ["Houston"],
      "sentiment": "positive",
      "evidence": ["Pho 21 on the backside of NASA JSC has the best soft tendon of anywhere in Houston."]
    }
  ],
  "dish": ["pho"],
  "location_hint": ["Houston"],
  "sentiment": "positive"
}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,
        input=user_input,
        max_output_tokens=1200,
        temperature=0,
    )

    text = response.output_text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {text}") from exc

    return _normalize_result(payload)
