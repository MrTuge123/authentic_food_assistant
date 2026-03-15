from fastapi import FastAPI
from app.reddit_client import search_reddit

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Authentic Food Assistant API is running"}

@app.get("/search")
def search_restaurants(query: str, debug: bool = False, limit: int = 15):
    search_results = search_reddit(query, limit=limit, debug=debug)

    response = {
        "query": query,
        "restaurants": search_results["restaurants"],
        "reddit_posts": search_results["posts"],
    }
    if debug:
        response["debug_filters"] = search_results.get("debug_filters", [])
    return response
