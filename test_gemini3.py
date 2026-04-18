import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model_name = "gemini-3-flash-preview"
print(f"Testing {model_name}...")
llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,
)
res = llm.invoke("Hello, say 'Gemini 3 is working'")
print(f"Response: {res.content}")
