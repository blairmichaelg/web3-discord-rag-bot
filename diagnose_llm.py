import os
import traceback
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def test_llm(model_name):
    print(f"Testing model: {model_name}")
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
        )
        messages = [
            ("system", "You are a helpful assistant."),
            ("user", "Hello, are you working?"),
        ]
        response = llm.invoke(messages)
        print(f"Success! Response: {response.content}")
        return True
    except Exception as e:
        print(f"FAILED: {model_name}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    current_model = "gemini-2.0-flash-exp"
    candidates = ["gemini-2.0-flash-001", "gemini-1.5-flash", "gemini-1.5-flash-latest"]
    
    print("Checking current model...")
    if not test_llm(current_model):
        print("\nChecking candidates...")
        for candidate in candidates:
            if test_llm(candidate):
                print(f"\nSUCCESSFOUND: {candidate}")
                break
