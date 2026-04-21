import os
import time
import asyncio
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bot import safe_collection_count, build_llm_with_fallback, invoke_with_retry, split_for_discord, extract_text_from_gemini, PROMPTS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

async def simulate_query():
    mode = "origami"
    query = "what is origami finance?"
    print("Initializing...")
    PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings, collection_name="origami_ecosystem_v1")
    llm = build_llm_with_fallback()

    print("Running query simulation...")
    start_time = time.time()
    
    scored_docs = await asyncio.to_thread(vectorstore.similarity_search_with_score, query, k=15)
    docs = [doc for doc, score in scored_docs if score < 1.2]
    if len(docs) < 5:
        top = [doc for doc, score in scored_docs[:5]]
        seen = {id(d) for d in docs}
        for d in top:
            if id(d) not in seen:
                docs.append(d)
                seen.add(id(d))
                
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", doc.metadata.get("url", "unknown"))
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    raw_context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"CONTEXT:\n{raw_context}\n\nQUESTION: {query}"
    messages = [("system", PROMPTS[mode]), ("user", prompt)]
    
    response = await asyncio.to_thread(invoke_with_retry, llm, messages)
    raw_answer = extract_text_from_gemini(response)
    
    elapsed = time.time() - start_time
    print(f"\n--- RESULTS ---\nTime Taken: {elapsed:.2f} seconds")
    print(f"Response:\n{raw_answer}")

if __name__ == "__main__":
    asyncio.run(simulate_query())
