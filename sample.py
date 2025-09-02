from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(":memory:")

#collection creation
client.create_collection(
                         collection_name="test_collection",
                         vectors_config=VectorParams(size=4, distance=Distance.COSINE))
print("Collection created")

#list existing collections
print(client.get_collections())

#list collection details
print(client.get_collection("test_collection"))

#update a collection
client.update_collection(
    collection_name="test_collection",
    optimizers_config={"default_segment_number": 2}
)

#delete a collection
client.delete_collection("test_collection")
print("Collection deleted!")


print(client.get_collections())

client.create_collection(
    collection_name="cities",
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
)
print("✅ Collection 'cities' ready")

#insert points
#Each point = {id, vector, payload}
#id = unique identifier
#vector = embedding (list of floats)
#payload = metadata (dict, e.g., city name, country)

client.upsert(
    collection_name="cities",
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin", "country": "Germany"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London", "country": "UK"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow", "country": "Russia"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York", "country": "USA"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing", "country": "China"}),
    ],
)
print("✅ Inserted points")

#query points(similarity search)
#find the top-3 most similar cities
result = client.query_points(
    collection_name="cities",
    query=[0.2, 0.1, 0.9, 0.7],
    limit=3,
    with_payload=True,   # include metadata
).points

print("\n🔎 Search Results:")
for p in result:
    print(f"ID: {p.id}, Score: {p.score:.3f}, Payload: {p.payload}")

#Filtering (filter results by metadata)
#find only cities in Europe

from qdrant_client.models import Filter, FieldCondition, MatchValue

europe_filter = Filter(
    must=[FieldCondition(key="country", match=MatchValue(value="UK"))]
)

result = client.query_points(
    collection_name="cities",
    query=[0.2, 0.1, 0.9, 0.7],
    limit=3,
    with_payload=True,
    query_filter=europe_filter,
).points

print("\n🔎 Filtered Search (only UK):")
for p in result:
    print(f"ID: {p.id}, Payload: {p.payload}")

#Update or Delete Points

client.upsert(
    collection_name="cities",
    points=[PointStruct(id=2, vector=[0.5, 0.5, 0.5, 0.5], payload={"city": "London", "country": "UK"})],
)

client.delete(collection_name="cities", points_selector=[2])
print("🗑️ Deleted point with ID=2")

