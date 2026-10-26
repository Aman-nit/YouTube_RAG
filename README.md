# YouTube AI Chatbot (RAG)

**YouTube AI Chatbot** — [GitHub] [Live] — Python, RAG, LangChain, Streamlit, Qdrant, Sentence Transformers, Groq

– Built a retrieval-augmented generation (RAG) application that lets users process an English YouTube video and ask questions about its transcript through a conversational Streamlit interface.

– Implemented an end-to-end ingestion pipeline using `youtube-transcript-api` and LangChain text splitting, dividing transcripts into overlapping 1,000-character chunks with 200-character overlap and attaching video metadata for filtered retrieval.

– Generated normalized 384-dimensional embeddings with `sentence-transformers/all-MiniLM-L6-v2`, stored transcript vectors in Qdrant Cloud with deterministic point IDs, and avoided duplicate ingestion by checking whether a video's vectors already exist.

– Built semantic search over the active video's transcript to retrieve the five most relevant chunks, then used LangChain and Groq's `openai/gpt-oss-120b` model to produce context-grounded answers while preventing unsupported responses.

– Added production-oriented controls for YouTube URL validation, transcript availability checks, vector deletion, chat-history reset, environment-based credential loading, and user-facing error handling.