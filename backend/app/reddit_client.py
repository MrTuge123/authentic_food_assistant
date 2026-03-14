import requests

HEADERS = {"User-Agent": "authentic-food-assistant"}

def get_comments(post_url):
    url = post_url + ".json"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []
    
    comments = []

    for comment in data[1]["data"]["children"]:
        if comment["kind"]== "t1":
            body = comment["data"]["body"]
            comments.append(body)
        
    return comments

def search_reddit(query, limit = 5):
    url = f"https://www.reddit.com/search.json?q={query}&limit={limit}"

    response = requests.get(url,headers = HEADERS)
    data = response.json()

    posts = []

    for post in data["data"]["children"]:
        p = post["data"]

        comments = get_comments(p["url"])

        posts.append({
            "title": p["title"],
            "subreddit": p["subreddit"],
            "score": p["score"],
            "comments": comments[:5],
            "url": f"https://www.reddit.com{p['permalink']}"
        })
    
    return posts

