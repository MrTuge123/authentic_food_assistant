# Authentic Local Restaurant Discovery App

This project is a backend prototype for a restaurant recommendation system that prioritizes community knowledge over ad-driven ranking.

Instead of relying on traditional review platforms, the system searches local discussion content, extracts restaurant mentions, and returns evidence-backed recommendations. The goal is to help users discover places that locals genuinely talk about, not just places that rank well commercially.

## Goal

Build a web or mobile app that can answer queries such as:

`authentic ramen near downtown Seattle`

The intended pipeline is:

1. Accept a natural-language food query.
2. Search community discussions for relevant posts.
3. Extract restaurant names, dishes, and location clues from those discussions.
4. Aggregate and rank restaurant candidates.
5. Eventually resolve them to real places and show them on a map.

## What Works Today

The current repository contains a FastAPI backend that already does the following:

- accepts restaurant-related queries through `/search`
- searches Reddit using Reddit's public JSON endpoints
- fetches top-level comments for matched posts
- filters posts by food and location relevance
- extracts restaurant candidates from titles and comments
- supports both:
  - a heuristic extractor
  - an optional GPT-based extractor using `gpt-4.1-mini`
- returns:
  - restaurant candidates
  - supporting evidence snippets
  - dish mentions
  - location hints
  - raw Reddit posts for debugging

This is still a prototype. It is not yet a full recommendation product.

## Current Architecture

`FastAPI -> Reddit search -> Comment fetch -> Extraction -> Aggregation -> JSON response`

Core files:

- [`backend/app/main.py`](/Users/tiger.c/Desktop/food_planning_chatbot/backend/app/main.py)
- [`backend/app/reddit_client.py`](/Users/tiger.c/Desktop/food_planning_chatbot/backend/app/reddit_client.py)
- [`backend/app/extraction.py`](/Users/tiger.c/Desktop/food_planning_chatbot/backend/app/extraction.py)
- [`backend/app/llm_extraction.py`](/Users/tiger.c/Desktop/food_planning_chatbot/backend/app/llm_extraction.py)

## API

### Health Check

`GET /`

Returns a simple status message.

### Search

`GET /search?query=best+pho+houston`

Supported query params:

- `query`: natural-language food query
- `limit`: number of Reddit search hits to fetch before filtering, default `15`
- `debug`: include filter/debug metadata, default `false`
- `use_llm`: use GPT-based extraction instead of heuristic extraction, default `false`

Example URLs:

- [http://127.0.0.1:8000/search?query=authentic+ramen+in+Seattle](http://127.0.0.1:8000/search?query=authentic+ramen+in+Seattle)
- [http://127.0.0.1:8000/search?query=authentic+ramen+in+Seattle&debug=true](http://127.0.0.1:8000/search?query=authentic+ramen+in+Seattle&debug=true)
- [http://127.0.0.1:8000/search?query=authentic+ramen+in+Seattle&use_llm=true](http://127.0.0.1:8000/search?query=authentic+ramen+in+Seattle&use_llm=true)

### Current Response Shape

The backend currently returns JSON like:

- `query`
- `use_llm`
- `restaurants`
- `reddit_posts`
- `debug_filters` when `debug=true`

Each restaurant item may include:

- `name`
- `score`
- `mention_count`
- `post_count`
- `sentiment`
- `dishes`
- `location_hints`
- `subreddits`
- `evidence`
- `source_urls`

## Setup

### Prerequisites

- Python 3.12 recommended

### Install

```bash
cd /Users/tiger.c/Desktop/food_planning_chatbot/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="放你的open api key"
python3 -m nltk.downloader vader_lexicon
```

### Run the API

```bash
cd "删掉引号，放你到/backend 这个folder的path"
source venv/bin/activate
export OPENAI_API_KEY="不删引号，放你的open api key"
uvicorn app.main:app --reload

```

Then open:

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- [http://127.0.0.1:8000/search?query=best+sushi+ann+arbor](http://127.0.0.1:8000/search?query=best+sushi+ann+arbor)

## Optional LLM Extraction

The project includes an alternative extractor in [`backend/app/llm_extraction.py`](/Users/tiger.c/Desktop/food_planning_chatbot/backend/app/llm_extraction.py) that uses `gpt-4.1-mini`.

To enable it:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Then call:

```text
/search?query=authentic+ramen+in+Seattle&use_llm=true
```

If the LLM path fails, the backend currently falls back to the heuristic extractor.

## Tests

Run the regression tests with:

```bash
cd /Users/tiger.c/Desktop/food_planning_chatbot
python3 -m unittest discover backend/tests
```

The tests currently cover:

- extraction false positives
- short-name recall
- acronym and location filtering
- Reddit retrieval heuristics
- LLM fallback behavior

## What Is Still Missing

The current backend is good enough to prove the idea, but the full app still needs:

- stronger restaurant ranking
- better duplicate/entity resolution
- real place resolution through a maps/places API
- richer recommendation summaries
- more sources beyond Reddit
- frontend or mobile UI

## Recommended Next Steps

The highest-value next steps are:

1. Improve ranking so the best restaurants consistently appear first.
2. Build a small evaluation dataset of real user queries and expected good outcomes.
3. Resolve extracted restaurant names to real businesses.
4. Add recommendation summaries once ranking is stable.

## License

This repository is licensed under the terms in [`LICENSE`](/Users/tiger.c/Desktop/food_planning_chatbot/LICENSE).
