
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="News RSS Deduplicator")

# Define the structure of your incoming news articles
class Article(BaseModel):
    title: str
    link: str
    description: Optional[str] = ""
    source: Optional[str] = "" 

class Payload(BaseModel):
    articles: List[Article]
    # 0.75 is usually the sweet spot for catching duplicate news events 
    threshold: float = 0.75 

@app.post("/deduplicate")
def deduplicate_batch(payload: Payload):
    if not payload.articles:
        return {"unique_articles": []}

    # Combine title and description for richer semantic matching
    texts = [f"{a.title} {a.description}" for a in payload.articles]

    # Convert the text into a mathematical matrix of features
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Failsafe for batches containing only empty or stop-word text
        return {"unique_articles": payload.articles}

    # Calculate how similar every article is to every other article
    similarity_matrix = cosine_similarity(tfidf_matrix)

    unique_articles = []
    seen_indices = set()

    for i in range(len(payload.articles)):
        if i in seen_indices:
            continue

        # Keep the first high-quality instance of the story
        unique_articles.append(payload.articles[i])

        # Scan ahead: if any other article has a similarity score above 
        # the threshold, flag it as a duplicate and ignore it
        for j in range(i + 1, len(payload.articles)):
            if similarity_matrix[i][j] >= payload.threshold:
                seen_indices.add(j)

    return {
        "original_count": len(payload.articles),
        "unique_count": len(unique_articles),
        "unique_articles": unique_articles
    }
