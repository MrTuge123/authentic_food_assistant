import requests

HEADERS = {"User-Agent": "authentic-food-assistant"}

def search_reddit(query, limit = 5):
    url = f"https://www.reddit.com/search.json?q={query}&limit={limit}"

    response = requests.get(url,headers = HEADERS)
    data = response.json()

    posts = []

    for post in data["data"]["children"]:
        p = post["data"]
        posts.append({
            "title": p["title"],
            "subreddit": p["subreddit"],
            "score": p["score"],
            "text": p.get("selftxt",""),
            "url": p["url"]
        })
    
    return posts

def get_comments(post_url):
    url = post_url + ".json"

    response = requests.get(url, headers = HEADERS)
    data = response.json()

    comments = []

    for comment in data[1]["data"]["children"]:
        if comment["kind"]== "t1":
            body = comment["data"]["body"]
    