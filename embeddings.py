
import streamlit as st

@st.cache_resource
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model for the first time...")

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )