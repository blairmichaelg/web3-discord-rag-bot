import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "origami_ecosystem_v1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print("Initializing Chroma...")
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)
print("Testing search...")
q = "What is Origami?"
results = vectorstore.similarity_search(q, k=1)
print(f"Result: {len(results)} docs found.")
print(results[0].page_content[:100])
