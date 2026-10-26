"""
ingestion.py

Handles:
1. YouTube URL validation and video ID extraction
2. Transcript language detection and downloading
3. Splitting transcripts into chunks
4. Creating embeddings and storing them in Qdrant
5. Deleting a video's stored vectors
"""

import os
import re
from uuid import uuid5, NAMESPACE_URL
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)
from youtube_transcript_api._errors import IpBlocked
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from embeddings import get_embedding_model
from qdrant_client import QdrantClient, models


# Load environment variables
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url or not qdrant_api_key:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing in .env")


# Connect to Qdrant
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

COLLECTION_NAME = "youtube_transcripts"

# Validate collection
if not client.collection_exists(COLLECTION_NAME):
    raise ValueError(
        f"Qdrant collection '{COLLECTION_NAME}' does not exist."
    )

collection_info = client.get_collection(COLLECTION_NAME)
vectors_config = collection_info.config.params.vectors

if not isinstance(vectors_config, models.VectorParams):
    raise ValueError(
        "Expected a collection with one unnamed vector per point."
    )

if vectors_config.size != 384:
    raise ValueError(
        f"Expected 384-dimensional vectors, found {vectors_config.size}."
    )

# Filtering by video_id requires a payload index in Qdrant.
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="video_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
    wait=True,
)


# Split transcripts into overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    keep_separator="end",
    strip_whitespace=True,
)


def get_video_id(url: str) -> str:
    """Extract the 11-character video ID from a YouTube URL."""

    # Also allow a video ID directly
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]

    elif hostname in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    ):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        else:
            match = re.match(
                r"^/(?:shorts|embed|live)/([^/?]+)",
                parsed.path,
            )
            video_id = match.group(1) if match else None
    else:
        video_id = None

    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Invalid YouTube URL or video ID.")

    return video_id


def get_available_languages(video_id: str) -> list[str]:
    """Return the available transcript language codes."""

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    return [
        transcript.language_code
        for transcript in transcript_list
    ]


def get_transcript(video_id: str, language_code: str = "en") -> str:
    """Download a transcript and return it as plain text."""

    if language_code != "en":
        raise ValueError(
            "Only English transcripts are supported currently."
        )

    api = YouTubeTranscriptApi()
    transcript = api.fetch(
        video_id,
        languages=[language_code],
    )

    return " ".join(
        segment.text for segment in transcript
    ).strip()


def chunk_transcript(
    transcript: str,
    video_id: str,
    language_code: str = "en",
) -> list[Document]:
    """Split transcript text and attach metadata."""

    if not transcript.strip():
        return []

    chunks = splitter.split_text(transcript)

    return [
        Document(
            page_content=chunk,
            metadata={
                "video_id": video_id,
                "chunk_index": index,
                "language_code": language_code,
            },
        )
        for index, chunk in enumerate(chunks)
    ]


def store_chunks(
    documents: list[Document],
    video_id: str,
) -> int:
    """Embed transcript chunks and store them in Qdrant."""

    if not documents:
        print("No chunks to store.")
        return 0

    texts = [doc.page_content for doc in documents]

    # Load the embedding model lazily and reuse the cached instance
    vectors = get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    points = []

    for doc, vector in zip(documents, vectors):
        chunk_index = doc.metadata["chunk_index"]

        points.append(
            models.PointStruct(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{video_id}:{chunk_index}",
                    )
                ),
                vector=vector,
                payload={
                    "text": doc.page_content,
                    **doc.metadata,
                    "video_id": video_id,
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    print(f"Stored {len(points)} chunks successfully.")
    return len(points)


def ingest_video(url: str) -> int:
    """Run the complete ingestion pipeline."""

    # Step 1: Extract video ID
    video_id = get_video_id(url)
    print("Video ID:", video_id)

    # Step 2: Check available languages
    languages = get_available_languages(video_id)
    print("Available languages:", languages)

    if "en" not in languages:
        raise ValueError(
            "No English transcript is available for this video."
        )

    # Step 3: Download transcript with useful error logging
    try:
        transcript = get_transcript(video_id, "en")

    except (
        TranscriptsDisabled,
        NoTranscriptFound,
        IpBlocked,
        CouldNotRetrieveTranscript,
    ) as error:
        print(
            f"Transcript download failed: "
            f"{type(error).__name__}: {error}"
        )
        raise

    if not transcript:
        raise ValueError("The retrieved transcript is empty.")

    # Step 4: Create chunks
    documents = chunk_transcript(
        transcript,
        video_id,
        "en",
    )
    print("Chunks created:", len(documents))

    # Step 5: Generate embeddings and store in Qdrant
    return store_chunks(documents, video_id)


def delete_video_vectors(video_id: str) -> None:
    """Delete all stored chunks belonging to a video."""

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id",
                        match=models.MatchValue(value=video_id),
                    )
                ]
            )
        ),
        wait=True,
    )

    print(f"Deleted vectors for video: {video_id}")


def video_vectors_exist(video_id: str) -> bool:
    """Return whether Qdrant already contains points for a video."""

    result = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id),
                )
            ]
        ),
        exact=True,
    )

    return result.count > 0


# Run the pipeline only when this file is executed directly
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=DNz-aCUWyHQ"

    try:
        total = ingest_video(video_url)
        print("Ingestion completed. Chunks stored:", total)

    except Exception as error:
        print(
            f"Ingestion failed: {type(error).__name__}: {error}"
        )
