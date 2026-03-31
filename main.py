from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="News RSS Deduplicator")

class Article(BaseModel):
    title: str
    content: Optional[str] = ""
    contentSnippet: Optional[str] = ""
    link: Optional[str] = ""
    source: Optional[str] = ""

class Payload(BaseModel):
    articles: List[Article]
    threshold: float = 0.75

@app.post("/deduplicate")
def deduplicate_batch(payload: Payload):
    if not payload.articles:
        return {"unique_articles": []}

    # --- STEP 1: URL-based exact duplicate removal first ---
    seen_links = set()
    url_filtered = []
    for article in payload.articles:
        url = (article.link or "").strip().lower()
        if url and url in seen_links:
            continue
        if url:
            seen_links.add(url)
        url_filtered.append(article)

    if len(url_filtered) == 1:
        return {
            "original_count": len(payload.articles),
            "unique_count": 1,
            "unique_articles": url_filtered
        }

    # --- STEP 2: Semantic deduplication via TF-IDF ---
    # Prefer contentSnippet for matching (clean text), fall back to content
    texts = [
        f"{a.title} {(a.contentSnippet or a.content or '').strip()}"
        for a in url_filtered
    ]

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # All texts were empty or only stop words — return url-filtered result
        return {
            "original_count": len(payload.articles),
            "unique_count": len(url_filtered),
            "unique_articles": url_filtered
        }

    similarity_matrix = cosine_similarity(tfidf_matrix)

    unique_articles = []
    seen_indices = set()

    for i in range(len(url_filtered)):
        if i in seen_indices:
            continue
        unique_articles.append(url_filtered[i])
        for j in range(i + 1, len(url_filtered)):
            if similarity_matrix[i][j] >= payload.threshold:
                seen_indices.add(j)

    return {
        "original_count": len(payload.articles),
        "unique_count": len(unique_articles),
        "unique_articles": unique_articles
    }

@app.get("/")
def health_check():
    return {"status": "API is alive!", "message": "Send a POST request to /deduplicate"}
