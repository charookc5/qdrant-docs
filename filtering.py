from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding

# ================================
# 1. Setup client + embedder
# ================================
client = QdrantClient(":memory:")  # In-memory DB (use host="localhost", port=6333 for real server)
embedder = TextEmbedding()

# ================================
# 2. Create collection
# ================================
client.create_collection(
    collection_name="articles",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# ================================
# 3. Insert documents with metadata
# ================================
docs = [
    {"id": 1, "text": "Eiffel Tower is in Paris", "category": "travel"},
    {"id": 2, "text": "Python is a programming language", "category": "tech"},
    {"id": 3, "text": "Berlin is the capital of Germany", "category": "travel"},
    {"id": 4, "text": "AI is transforming healthcare", "category": "tech"},
]

# Embed all texts at once
vectors = list(embedder.embed([d["text"] for d in docs]))

points = [
    PointStruct(id=doc["id"], vector=vec, payload=doc)
    for doc, vec in zip(docs, vectors)
]

client.upsert(collection_name="articles", points=points)

# ================================
# 4. Similarity search
# ================================
query = "capital of France"
query_vec = list(embedder.embed([query]))[0]

results = client.query_points(
    collection_name="articles",
    query=query_vec,
    limit=3,
    with_payload=True,
).points

print("\n🔎 Similarity search results for:", query)
for r in results:
    print("-", r.payload["text"], "| Score:", round(r.score, 3))

# ================================
# 5. Similarity search + filtering
# ================================
results_filtered = client.query_points(
    collection_name="articles",
    query=query_vec,
    limit=3,
    with_payload=True,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="travel")
            )
        ]
    )
).points

print("\n🎯 Filtered results (category=travel):")
for r in results_filtered:
    print("-", r.payload["text"], "| Score:", round(r.score, 3))
