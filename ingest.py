"""
Multi-Target Ecosystem Documentation Ingestion Pipeline
Supports: berachain | infrared | dolomite | origami | ion | euler | silo
"""

import os
import sys
import time
import shutil
import argparse
import warnings
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_community.document_loaders import RecursiveUrlLoader, GitbookLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
load_dotenv()

def safe_collection_count(vs: Chroma) -> int:
    """Safely get the document count of a collection."""
    try:
        if vs and hasattr(vs, "_collection") and vs._collection:
            return vs._collection.count()
    except Exception:
        pass
    return 0

# ── Configuration ────────────────────────────────────────────────────────────
PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TARGETS = {
    "berachain": {
        "collection_name": "berachain_ecosystem_v1",
        "allowed_domains": {"docs.berachain.com"},
        "blocked_paths": {"/changelog/", "/blog/", "/careers/"},
        "sources": [
            {
                "url": "https://docs.berachain.com/general/proof-of-liquidity/overview",
                "label": "PoL Mechanics",
                "loader": "recursive",
            },
            {
                "url": "https://docs.berachain.com/general/tokens/bgt",
                "label": "BGT Mechanics",
                "loader": "recursive",
            },
            {
                "url": "https://docs.berachain.com/general/introduction/what-is-berachain",
                "label": "Core Concepts",
                "loader": "recursive",
            },
            {
                "url": "https://docs.berachain.com/general/help/glossary",
                "label": "Glossary",
                "loader": "recursive",
            },
            {
                "url": "https://docs.berachain.com/build/getting-started/overview",
                "label": "Dev Onboarding",
                "loader": "recursive",
            },
            {
                "url": "https://docs.berachain.com",
                "label": "Full Berachain Docs",
                "loader": "recursive",
            },
        ]
    },
    "infrared": {
        "collection_name": "infrared_ecosystem_v1",
        "allowed_domains": {"infrared.finance"},
        "blocked_paths": {"/blog/", "/changelog/", "/careers/"},
        "sources": [
            {
                "url": "https://infrared.finance/docs",
                "label": "Full Protocol Overview",
                "loader": "recursive",
            },
            {
                "url": "https://infrared.finance/education/ibgt",
                "label": "iBGT Mechanics",
                "loader": "recursive",
            },
            {
                "url": "https://infrared.finance/education/ibera",
                "label": "iBERA Mechanics",
                "loader": "recursive",
            },
            {
                "url": "https://infrared.finance/docs/tokens",
                "label": "Tokens & Contracts",
                "loader": "recursive",
            },
            {
                "url": "https://infrared.finance/docs/ibgt-rewards",
                "label": "Reward APR Mechanics",
                "loader": "recursive",
            },
            {
                "url": "https://infrared.finance/docs/berachain",
                "label": "Berachain Context",
                "loader": "recursive",
            },
            {
                "url": "https://infrared.finance/docs/developers/contract-deployments",
                "label": "Contract Addresses",
                "loader": "recursive",
            },
        ]
    },
    "dolomite": {
        "collection_name": "dolomite_ecosystem_v1",
        "allowed_domains": {"docs.dolomite.io"},
        "blocked_paths": {"/blog/", "/changelog/", "/careers/"},
        "sources": [
            {
                "url": "https://docs.dolomite.io",
                "label": "Full Protocol Overview",
                "loader": "gitbook",
            },
            {
                "url": "https://docs.dolomite.io/dolomite-governance",
                "label": "veDOLO Governance Mechanics",
                "loader": "recursive",
            },
            {
                "url": "https://docs.dolomite.io/developer-documentation/zapping-assets",
                "label": "Asset Zapping Routing",
                "loader": "recursive",
            },
            {
                "url": "https://docs.dolomite.io/admin-privileges",
                "label": "Admin Immutable Layer",
                "loader": "recursive",
            },
            {
                "url": "https://docs.dolomite.io/integrations/berachain-proof-of-liquidity",
                "label": "Berachain PoL Integration",
                "loader": "recursive",
            },
        ]
    },
    "origami": {
        "collection_name": "origami_ecosystem_v1",
        "allowed_domains": {"docs.origami.finance"},
        "blocked_paths": {"/blog/", "/changelog/", "/careers/"},
        "sources": [
            {
                "url": "https://docs.origami.finance",
                "label": "Full Origami docs",
                "loader": "recursive",
            },
            {
                "url": "https://docs.origami.finance/boyco",
                "label": "Origami Boyco USDC vault",
                "loader": "recursive",
            },
        ]
    },
    "ion": {
        "collection_name": "ion_ecosystem_v1",
        "allowed_domains": {"docs.ionprotocol.io"},
        "blocked_paths": {"/blog/", "/press/", "/careers/", "/changelog/"},
        "sources": [
            {
                "url": "https://docs.ionprotocol.io",
                "label": "Ion Docs Root",
                "loader": "recursive",
            },
            {
                "url": "https://docs.ionprotocol.io/overview/welcome-to-ion-protocol",
                "label": "Protocol Overview",
                "loader": "recursive",
            },
            {
                "url": "https://docs.ionprotocol.io/overview/faq",
                "label": "FAQ & Risk Disclosures",
                "loader": "recursive",
            },
            {
                "url": "https://docs.ionprotocol.io/ion-protocol/how-ion-works",
                "label": "How Ion Works (Architecture)",
                "loader": "recursive",
            },
            {
                "url": "https://docs.ionprotocol.io/supported-collateral/lsts",
                "label": "Supported Collateral: LSTs",
                "loader": "recursive",
            },
            {
                "url": "https://docs.ionprotocol.io/security/security-reviews",
                "label": "Security Reviews & Audits",
                "loader": "recursive",
            },
            {
                "url": "https://docs.ionprotocol.io/overview/deprecation-guide",
                "label": "Deprecation Guide",
                "loader": "recursive",
            },
        ]
    },
    "euler": {
        "collection_name": "euler_ecosystem_v1",
        "allowed_domains": {"docs.euler.finance"},
        "blocked_paths": {"/blog/", "/changelog/", "/careers/"},
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "sources": [
            {
                "url": "https://docs.euler.finance/llms-full.txt",
                "label": "Full LLM-Ready Doc Dump",
                "loader": "direct",
            },
            {
                "url": "https://docs.euler.finance/overview/introduction",
                "label": "Protocol Introduction",
                "loader": "recursive",
            },
            {
                "url": "https://docs.euler.finance/user-guide/euler-swap",
                "label": "EulerSwap User Guide",
                "loader": "recursive",
            },
            {
                "url": "https://docs.euler.finance/developers/euler-swap/how-it-works",
                "label": "EulerSwap Developer Mechanics",
                "loader": "recursive",
            },
            {
                "url": "https://docs.euler.finance/developers/data-querying",
                "label": "Data Querying & Indexing",
                "loader": "recursive",
            },
        ]
    },
    "silo": {
        "collection_name": "silo_ecosystem_v1",
        "allowed_domains": {"docs.silo.finance", "silodocs2.netlify.app", "devdocs.silo.finance"},
        "blocked_paths": {"/blog/", "/changelog/", "/careers/"},
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "sources": [
            {
                "url": "https://docs.silo.finance/docs/users/intro",
                "label": "Silo User Docs (Users)",
                "loader": "recursive",
                "max_depth": 4,
            },
            {
                "url": "https://docs.silo.finance/docs/vaults/intro",
                "label": "Silo User Docs (Vaults)",
                "loader": "recursive",
                "max_depth": 4,
            },
            {
                "url": "https://docs.silo.finance/docs/category/protocol-overview",
                "label": "Silo User Docs (Developers)",
                "loader": "recursive",
                "max_depth": 4,
            },
            {
                "url": "https://docs.silo.finance/docs/audits",
                "label": "Silo Audits",
                "loader": "recursive",
            },
            {
                "url": "https://silodocs2.netlify.app",
                "label": "Silo V3 Architecture Docs",
                "loader": "recursive",
                "max_depth": 4,
            },
            {
                "url": "https://devdocs.silo.finance",
                "label": "Silo Developer Docs",
                "loader": "recursive",
                "max_depth": 3,
            },
        ]
    },
}

def domain_ok(url: str, allowed: set) -> bool:
    try:
        host = urlparse(url).hostname or ""
        return any(host == d or host.endswith("." + d) for d in allowed)
    except Exception:
        return False

def path_ok(url: str, blocked: set) -> bool:
    path = urlparse(url).path.lower()
    return not any(seg in path for seg in blocked)

SLOW_PARSE = os.getenv("SLOW_PARSE", "0") == "1"

def bs4_extractor(html: str) -> str:
    if SLOW_PARSE:
        time.sleep(0.5)
    soup = BeautifulSoup(html, "html.parser")
    # Strip all noise elements
    for tag in soup(["script", "style", "nav", "footer", "header", "svg", "button",
                      "aside", "form", "[role='navigation']", "[role='complementary']"]):
        tag.decompose()
    # Try to isolate main content — GitBook uses <article> or <main>
    main = soup.find("article") or soup.find("main") or soup.find("div", {"class": lambda c: c and "content" in c.lower()})
    target = main if main else soup
    return target.get_text(separator="\n", strip=True)

def load_source(source: dict, allowed_domains: set, blocked_paths: set) -> list:
    url = source["url"]
    label = source["label"]
    loader_type = source.get("loader", "recursive")
    print(f"       Loading [{loader_type}]: {url} ({label})...")

    if loader_type == "direct":
        # Direct HTTP fetch for plain-text URLs (e.g., llms-full.txt)
        from langchain_core.documents import Document
        resp = requests.get(url, timeout=60, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })
        resp.raise_for_status()
        docs = [Document(page_content=resp.text, metadata={"source": url})]
    elif loader_type == "gitbook":
        loader = GitbookLoader(
            web_page=url,
            load_all_paths=True
        )
        docs = loader.load()
    else:
        depth = source.get("max_depth", 3)
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=depth,
            extractor=bs4_extractor,
            prevent_outside=True,
            check_response_status=True,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            },
        )
        docs = loader.load()

    rejected = []
    filtered = []
    for doc in docs:
        doc_url = doc.metadata.get("source", doc.metadata.get("url", ""))
        if not domain_ok(doc_url, allowed_domains):
            # print(f"DEBUG: domain_reject {doc_url} (not in {allowed_domains})")
            rejected.append(("domain_reject", doc_url))
        elif not path_ok(doc_url, blocked_paths):
            # print(f"DEBUG: path_reject {doc_url}")
            rejected.append(("path_reject", doc_url))
        else:
            doc.metadata["ecosystem_source"] = label
            filtered.append(doc)

    if not filtered and docs:
        print(f"       !!! ALL {len(docs)} PAGES REJECTED. First Doc Source: {docs[0].metadata.get('source')}")
        print(f"       ALLOWED: {allowed_domains}")

    return filtered, rejected

def main():
    parser = argparse.ArgumentParser(description="Multi-Target Ecosystem Ingestion")
    parser.add_argument("--target", required=True, choices=["berachain", "infrared", "dolomite", "origami", "ion", "euler", "silo"], help="Target ecosystem to ingest")
    args = parser.parse_args()
    
    target_config = TARGETS[args.target]
    collection_name = target_config["collection_name"]
    allowed_domains = target_config["allowed_domains"]
    blocked_paths = target_config["blocked_paths"]
    sources = target_config["sources"]

    # Delete existing collection if any
    print(f"[0/5] Initiating localized DB flush for {collection_name} if exists (we keep others intact)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    try:
        # Instead of blowing away PERSIST_DIR, we initialize VectorStore and delete the collection
        _temp_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings, collection_name=collection_name)
        _temp_db.delete_collection()
    except Exception:
        pass

    print(f"[1/5] Crawling {len(sources)} {args.target} documentation sources...")
    all_docs = []
    all_rejected = []
    source_stats = {}
    seen_urls = set()

    for source in sources:
        try:
            docs, rejected = load_source(source, allowed_domains, blocked_paths)
            
            unique_docs = []
            for doc in docs:
                url = doc.metadata.get("source", doc.metadata.get("url", ""))
                # simple URL normalization
                url = url.rstrip("/")
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_docs.append(doc)
            
            label = source["label"]
            source_stats[label] = {"pages": len(unique_docs), "rejected": len(rejected)}
            all_docs.extend(unique_docs)
            all_rejected.extend(rejected)

            print(f"       [OK] {label}: {len(unique_docs)} unique pages loaded, {len(rejected)} URLs rejected")
        except Exception as e:
            label = source["label"]
            print(f"       [ERROR] {label}: FAILED — {e}")
            source_stats[label] = {"pages": 0, "rejected": 0, "error": str(e)}

    print(f"       Total: {len(all_docs)} pages across all sources.")

    if not all_docs:
        print("ERROR: No documents loaded from any source. Check URLs and network.")
        sys.exit(1)

    target_chunk_size = target_config.get("chunk_size", CHUNK_SIZE)
    target_chunk_overlap = target_config.get("chunk_overlap", CHUNK_OVERLAP)
    print(f"[2/5] Splitting into chunks (size={target_chunk_size}, overlap={target_chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_chunk_size,
        chunk_overlap=target_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"       Generated {len(chunks)} chunks.")

    chunk_stats = {}
    for chunk in chunks:
        label = chunk.metadata.get("ecosystem_source", "unknown")
        chunk_stats[label] = chunk_stats.get(label, 0) + 1
    for label, count in chunk_stats.items():
        print(f"       -> {label}: {count} chunks")

    print(f"[3/5] Embedding with local HuggingFace {EMBEDDING_MODEL} into collection '{collection_name}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
    )

    print(f"\n[4/5] Ingestion complete.")
    print(f"=" * 60)
    print(f"INGESTION REPORT - {collection_name}")
    print(f"=" * 60)

    for label, stats in source_stats.items():
        c_count = chunk_stats.get(label, 0)
        print(f"  {label}:")
        print(f"    Pages loaded : {stats['pages']}")
        print(f"    URLs rejected: {stats['rejected']}")
        print(f"    Chunks stored: {c_count}")

    final_count = safe_collection_count(vectorstore) if vectorstore else 0
    print(f"\n  ChromaDB collection '{collection_name}': {final_count} total documents")

    if all_rejected:
        print(f"\n  Rejected URLs (count: {len(all_rejected)}). Showing up to 20:")
        for reason, url in all_rejected[:20]:
            print(f"    [{reason}] {url}")
        if len(all_rejected) > 20:
            print(f"    ... and {len(all_rejected) - 20} more")

    print(f"\n[5/5] Vector store ready for bot.py (--mode {args.target}).")

if __name__ == "__main__":
    main()
