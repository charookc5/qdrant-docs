from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    HnswConfig
)
from fastembed import TextEmbedding

# ================================
# 1. Connect to Qdrant
# ================================
client = QdrantClient(":memory:")  # in-memory DB for testing
embedder = TextEmbedding()          # local embedding model

# ================================
# 2. Create collection with HNSW index
# ================================
from qdrant_client.models import HnswConfig

client.create_collection(
    collection_name="articles",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    hnsw_config=HnswConfig(
        m=16,
        ef_construct=200,
        full_scan_threshold=10000
    )
)
print("✅ Collection 'articles' created with HNSW indexing")


# ================================
# 3. Insert documents
# ================================
docs = [
    {"id": 1, "text": "Eiffel Tower is in Paris", "category": "travel"},
    {"id": 2, "text": "Python is a programming language", "category": "tech"},
    {"id": 3, "text": "Berlin is the capital of Germany", "category": "travel"},
    {"id": 4, "text": "AI is transforming healthcare", "category": "tech"},
]

vectors = list(embedder.embed([d["text"] for d in docs]))

points = [
    PointStruct(id=doc["id"], vector=vec, payload=doc)
    for doc, vec in zip(docs, vectors)
]

client.upsert(collection_name="articles", points=points)
print("✅ Inserted documents into collection")

# ================================
# 4. Similarity search
# ================================
query = "capital of France"
query_vec = list(embedder.embed([query]))[0]

results = client.query_points(
    collection_name="articles",
    query=query_vec,
    limit=3,
    with_payload=True
).points

print("\n🔎 Similarity search results for:", query)
for r in results:
    print("-", r.payload["text"], "| Score:", round(r.score, 3))

# ================================
# 5. Similarity search with filtering
# ================================
filter_condition = Filter(
    must=[FieldCondition(key="category", match=MatchValue(value="travel"))]
)

results_filtered = client.query_points(
    collection_name="articles",
    query=query_vec,
    limit=3,
    with_payload=True,
    query_filter=filter_condition
).points

print("\n🎯 Filtered results (category=travel):")
for r in results_filtered:
    print("-", r.payload["text"], "| Score:", round(r.score, 3))

# ================================
# 6. Update a point
# ================================
client.upsert(
    collection_name="articles",
    points=[PointStruct(id=2, vector=vectors[1], payload={"text":"Python programming", "category":"tech"})]
)
print("\n✏️ Updated point ID=2")

# ================================
# 7. Delete a point
# ================================
client.delete(collection_name="articles", points_selector=[4])
print("🗑️ Deleted point ID=4")
