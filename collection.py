from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

# Setup Qdrant
client = QdrantClient(":memory:")

# Use FastEmbed (local, free)
embedder = TextEmbedding()

# Create collection (384-dimensional vectors for this model)
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Insert texts
texts = [
    "The Eiffel Tower is in Paris",
    "Berlin is the capital of Germany",
    "New York is known as the Big Apple",
]
points = [
    PointStruct(id=i, vector=vec, payload={"text": txt})
    for i, (txt, vec) in enumerate(zip(texts, embedder.embed(texts)), start=1)
]
client.upsert(collection_name="documents", points=points)

# Query
query = "capital of France"
query_vec = list(embedder.embed([query]))[0]

results = client.query_points(
    collection_name="documents",
    query=query_vec,
    limit=2,
    with_payload=True,
).points

print("\n🔎 Results for:", query)
for r in results:
    print(r.payload["text"], "=> score:", round(r.score, 3))
