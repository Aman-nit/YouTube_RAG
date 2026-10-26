
import re
import streamlit as st

from ingestion import (
    delete_video_vectors,
    ingest_video,
    video_vectors_exist,
)
from retrieval import retrieve_chunks
from final_output import generate_answer


st.set_page_config(
    page_title="YouTube AI Chat",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 YouTube AI Chat")
st.caption("Chat with a video using LangChain, Groq and Qdrant.")

# Session state
if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"youtube\.com/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


with st.sidebar:
    st.header("Video setup")
    video_url = st.text_input("YouTube URL")

    if st.button("Process video", use_container_width=True):
        video_id = extract_video_id(video_url.strip())

        if not video_id:
            st.error("Please enter a valid YouTube URL.")
        else:
            try:
                with st.spinner("Checking stored vectors..."):
                    already_ingested = video_vectors_exist(video_id)

                if already_ingested:
                    st.session_state.video_id = video_id
                    st.session_state.chat_history = []
                    st.success(
                        "This video is already stored. "
                        "Skipped ingestion and embedding."
                    )
                else:
                    with st.spinner("Processing video transcript..."):
                        ingest_video(video_url)

                    st.session_state.video_id = video_id
                    st.session_state.chat_history = []
                    st.success("Video processing completed!")

            except Exception as exc:
                st.error(f"Could not process video: {exc}")

    if st.session_state.video_id:
        st.info(f"Active video: {st.session_state.video_id}")

        if st.button(
            "Delete stored vectors",
            use_container_width=True,
            type="secondary",
        ):
            try:
                with st.spinner("Deleting stored vectors..."):
                    delete_video_vectors(st.session_state.video_id)

                st.session_state.video_id = None
                st.session_state.chat_history = []
                st.success("Stored vectors deleted successfully.")
                st.rerun()

            except Exception as exc:
                st.error(f"Could not delete stored vectors: {exc}")

 
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# Display previous conversation
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input("Ask a question about your video...")

if question:
    if not st.session_state.video_id:
        st.warning("Please process a YouTube video first.")
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding relevant information..."):
                    chunks = retrieve_chunks(
                        query=question,
                        video_id=st.session_state.video_id,
                        k=5,
                    )

                    answer = generate_answer(question, chunks)

                st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                })

            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
