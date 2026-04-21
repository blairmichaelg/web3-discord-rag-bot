import os
import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Check ChromaDB collection counts")
    parser.add_argument("--collection", type=str, help="Specific collection to check", default=None)
    args = parser.parse_args()

    PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    print(f"Checking ChromaDB at {PERSIST_DIR}...")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if args.collection:
        collections_to_check = [args.collection]
    else:
        try:
            client = chromadb.PersistentClient(path=PERSIST_DIR)
            collections_to_check = [c.name for c in client.list_collections()]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return

    for coll in collections_to_check:
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

if __name__ == "__main__":
    main()
