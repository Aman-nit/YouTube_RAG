
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful YouTube video assistant.
Answer the user's question using only the provided
video context.

If the context does not contain the answer, say that
the information was not found in the video context.
Do not make up facts."""
    ),
    (
        "human",
        """Video context:
{context}

User question:
{question}

Answer:"""
    )
])

# Create the LangChain runnable
chain = prompt | llm


def generate_answer(question: str, retrieved_chunks: list) -> str:
    """Generate an answer from retrieved Qdrant chunks."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return "I couldn't find relevant information in the video."

    # Convert Qdrant points into readable text
    context = "\n\n".join(
        point.payload["text"]
        for point in retrieved_chunks
        if point.payload and point.payload.get("text")
    )

    if not context.strip():
        return "No usable transcript context was retrieved."

    # Run the LangChain pipeline
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content


