from fastapi import FastAPI
from app.reddit_client import search_reddit

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Authentic Food Assistant API is running"}

@app.get("/search")
def search_restaurants(query: str):
    posts = search_reddit(query)

    return {
        "query": query,
        "reddit_posts": posts
    }