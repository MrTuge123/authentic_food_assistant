# Authentic Local Restaurant Discovery App

This project is an early backend prototype for an authenticity-first restaurant recommendation app.

Instead of relying on ad-driven ranking from review platforms, the goal is to find restaurants that locals genuinely recommend in community discussions. A user should be able to ask for something like `authentic ramen near downtown Seattle`, and the system should retrieve relevant discussions, extract restaurant signals, and return clear recommendations with supporting evidence.

## Project Goal

Build a mobile or web application that helps users discover authentic local restaurants by mining community discussions rather than sponsored listings.

Core workflow:

1. Accept a natural-language restaurant query.
2. Search community sources such as Reddit for relevant local discussions.
3. Extract restaurant names, dishes, and location hints from posts and comments.
4. Map restaurant mentions to real places.
5. Rank results using authenticity and relevance signals.
6. Generate concise summaries explaining why each place is recommended.

## Current Status

The repository currently contains a small FastAPI backend MVP with:

- A root health endpoint.
- A `/search` endpoint that accepts a query string.
- Reddit search using Reddit's public JSON endpoints.
- Comment fetching for matching Reddit posts.
- Lightweight rule-based extraction for restaurant names, dishes, location hints, and coarse sentiment.

Current backend files:

- [`backend/app/main.py`](/Users/tiger.c/Desktop/food_planning_chatbot/backend/app/main.py)
- [`backend/app/reddit_client.py`](/Users/tiger.c/Desktop/food_planning_chatbot/backend/app/reddit_client.py)
- [`backend/app/extraction.py`](/Users/tiger.c/Desktop/food_planning_chatbot/backend/app/extraction.py)

## Current Architecture

`FastAPI endpoint -> Reddit search -> Comment fetch -> Structured extraction -> JSON response`

Today, the system returns Reddit posts and extracted fields. It does not yet perform place resolution, ranking, vector retrieval, Xiaohongshu ingestion, or LLM summarization.

## Example API Behavior

`GET /search?query=authentic+ramen+in+Seattle`

The API currently returns JSON with:

- The original query
- Matching Reddit posts
- Post metadata such as subreddit and score
- A small set of fetched comments
- Extracted structured signals

Example extracted fields:

- `restaurant_name`
- `dish`
- `location_hint`
- `sentiment`

## Local Setup

### Prerequisites

- Python 3.12 recommended

### Install dependencies

```bash
cd /Users/tiger.c/Desktop/food_planning_chatbot/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m nltk.downloader vader_lexicon
```

### Run the API

```bash
cd /Users/tiger.c/Desktop/food_planning_chatbot/backend
uvicorn app.main:app --reload
```

Then open:

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- [http://127.0.0.1:8000/search?query=authentic+ramen+in+Seattle](http://127.0.0.1:8000/search?query=authentic+ramen+in+Seattle)
- [http://127.0.0.1:8000/search?query=authentic+ramen+in+Seattle&use_llm=true](http://127.0.0.1:8000/search?query=authentic+ramen+in+Seattle&use_llm=true)

### LLM Extraction Setup

The repository also includes [`backend/app/llm_extraction.py`](/Users/tiger.c/Desktop/food_planning_chatbot/backend/app/llm_extraction.py), which uses `gpt-4.1-mini` for extraction.

Set your API key as an environment variable before using it:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Tests

Run the regression tests with:

```bash
cd /Users/tiger.c/Desktop/food_planning_chatbot
python3 -m unittest discover backend/tests
```

## Near-Term Roadmap

Suggested next steps for the MVP:

1. Add request validation and better error handling.
2. Normalize and deduplicate extracted restaurant names across posts.
3. Resolve restaurant entities with a maps or places API.
4. Add a scoring layer based on mention frequency, unique users, recency, and relevance.
5. Introduce LLM-based extraction and recommendation summarization.
6. Add persistence, caching, and tests.
7. Build a simple web or mobile UI.

## Research Direction

This project can also support a thesis or research direction around authenticity-aware recommendation from social discussion platforms:

- How to identify trustworthy local recommendations in noisy community data
- How to rank authenticity without relying on sponsored content
- How LLMs can summarize evidence-backed recommendations transparently

## License

This repository is licensed under the terms in [`LICENSE`](/Users/tiger.c/Desktop/food_planning_chatbot/LICENSE).
