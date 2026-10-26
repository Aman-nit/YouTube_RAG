
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

try:
    response = llm.invoke("Explain RAG in two simple sentences.")

    print("\nLLM RESPONSE:")
    print(response.content)

    print("\nLLM connection test successful!")

except Exception as e:
    print("\nLLM test failed:")
    print(f"{type(e).__name__}: {e}")
