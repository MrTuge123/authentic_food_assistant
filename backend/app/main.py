from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
    return {"message": "Authentic Food Assistant API is running"}

@app.get("/search")
def search_restaurants(query: str):
    return{
        "query": query,
        "results": [
            {
                "name": "Palace Tang",
                "score": 2.5,
                "reason": "Frequently recommended by locals"
            },
            {
                "name": "Slurping Turtle",
                "score": 1,
                "reason": "Commonly mentioned in community discussions"
            }
        ]
    }