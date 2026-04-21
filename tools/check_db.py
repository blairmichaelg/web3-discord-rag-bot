import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

collections = ["berachain_ecosystem_v1", "infrared_ecosystem_v1", "dolomite_ecosystem_v1", "origami_ecosystem_v1"]

print(f"Checking ChromaDB at {PERSIST_DIR}...")
for coll in collections:
    try:
        vs = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=coll,
        )
        count = vs._collection.count()
        print(f"Collection: {coll} | Count: {count}")
    except Exception as e:
        print(f"Collection: {coll} | Error: {e}")
