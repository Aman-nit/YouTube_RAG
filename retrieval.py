import os
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from embeddings import get_embedding_model



# Load environment variables from the .env file
load_dotenv()

# Always use the intended collection
COLLECTION_NAME = "youtube_transcripts"
# Read Qdrant credentials
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Stop early if credentials are missing
if not qdrant_url or not qdrant_api_key:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing in .env")

# Connect to Qdrant
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)


# Check that the collection exists
if not client.collection_exists(COLLECTION_NAME):
    raise ValueError(
        f"Qdrant collection '{COLLECTION_NAME}' does not exist."
    )

# Filtering by video_id requires a payload index in Qdrant.
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="video_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
    wait=True,
)


#Embedding query of users
def embed_user_query(query: str) -> list[float]:
    vector = get_embedding_model().encode(
        query,
        normalize_embeddings=True,
    )
    return vector.tolist()
    

#retrieval chunks from Qdrant
def retrieve_chunks(
    query: str,
    video_id: str,
    k: int = 5
) -> list:
    """Retrieve the most relevant chunks for a video question."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # Convert the question into a vector
    query_vector = embed_user_query(query)

    # Search only within the selected video's chunks
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id)
                )
            ]
        ),
        limit=k,
        with_payload=True
    )

    return response.points




if __name__ == "__main__":
    results = retrieve_chunks(
        query="What is this video about?",
        video_id="DNz-aCUWyHQ",
        k=5
    )

    for point in results:
        print("\nScore:", point.score)
        print("Chunk:", point.payload["text"])