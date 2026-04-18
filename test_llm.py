import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("Initializing LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,
    max_output_tokens=2048,
)
print("Testing LLM...")
res = llm.invoke("Hello, say 'API is working'")
print(f"Response: {res.content}")
