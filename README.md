# Web3 Discord RAG Support Bot

A privately deployable, multi-protocol AI support bot for Web3
Discord communities. Trained strictly on each protocol's official
documentation — zero hallucination, zero data leaving your pipeline.

---

## Demo Videos

| Protocol | Mechanics Demonstrated | Demo |
|---|---|---|
| **Pendle Finance** | YT mechanics, dynamic pricing, Real vs. Implied Yield math | [Watch](https://screenapp.io/app/v/OIAKpA2aHZ) |
| **Mantle Network** | L2 MNT gas requirements, mETH→cmETH bridging, MI4 guardrails | [Watch](https://screenapp.io/app/v/8Z3QAet38v) |
| **Infrared Finance** | iBGT vs BGT soulbound mechanics, iBGT redemption limits, HONEY earning flow | [Watch](https://screenapp.io/app/v/6vNqxF92kZ) |
| **Dolomite** | Core vs Modular architecture, Isolation Mode constraints | [Watch](https://screenapp.io/app/v/vHM9IyL1mR) |

---

## How It Works

1. `ingest.py --target <protocol>` crawls the protocol's official
   docs and embeds them into a local ChromaDB vector store using
   `all-MiniLM-L6-v2` local HuggingFace embeddings — no API quota,
   no external data transmission
2. `bot.py --mode <protocol>` connects to Discord and listens for
   mentions — on trigger, it retrieves the top relevant doc chunks
   and generates a sourced, structured answer
3. The LLM is strictly prompted to answer only from retrieved
   context — if it can't find the answer, it says so explicitly
   and redirects to the official docs or support ticket

---

## Supported Protocols

| Protocol | Chain(s) | Collection | Chunks | Status |
|---|---|---|---|---|
| Berachain | Berachain L1 | `berachain_ecosystem_v1` | 1,098 | ✅ Live |
| Infrared Finance | Berachain | `infrared_ecosystem_v1` | 533 | ✅ Live |
| Dolomite | Arbitrum / Berachain / Mantle | `dolomite_ecosystem_v1` | 606 | ✅ Live |
| Origami Finance | Ethereum / Berachain | `origami_ecosystem_v1` | 316 | ✅ Live |
| Pendle Finance | Ethereum / Arbitrum | `pendle_ecosystem_v1` | Archive | 📁 Demo Only |
| Mantle Network | Ethereum L2 | `mantle_ecosystem_v1` | Archive | 📁 Demo Only |

---

## Stack

| Component | Tool |
|---|---|
| Discord integration | `discord.py` |
| Retrieval chain | `LangChain` |
| Vector store | `ChromaDB` (local) |
| Embeddings | `HuggingFace all-MiniLM-L6-v2` (local, zero quota) |
| LLM | `Google Gemini 3 Flash Preview` via `langchain-google-genai` |
| Doc ingestion | `RecursiveUrlLoader` + `GitbookLoader` |
| Multi-protocol CLI | `argparse` — `--target` / `--mode` flags |

---

## Usage

```bash
# Ingest a protocol's docs into its own isolated collection
python ingest.py --target berachain
python ingest.py --target infrared
python ingest.py --target dolomite
python ingest.py --target origami

# Run the bot for a specific protocol
python bot.py --mode berachain
python bot.py --mode infrared
python bot.py --mode dolomite
python bot.py --mode origami
```

---

## Setup

```bash
git clone https://github.com/[your-username]/web3-discord-rag-bot
cd web3-discord-rag-bot
pip install -r requirements.txt
cp .env.example .env
# Fill in DISCORD_TOKEN and GOOGLE_API_KEY in .env
python ingest.py --target <protocol>
python bot.py --mode <protocol>
```

---

## Adding a New Protocol (Under 1 Hour)

1. Add a `--target <protocol>` block in `ingest.py` with the docs URL and crawl config
2. Add a `--mode <protocol>` system prompt in `bot.py` tuned to that protocol's common confusion points
3. Run `python ingest.py --target <protocol>`
4. Done — new protocol live with zero changes to existing collections

---

## Design Principles

- **Zero hallucination policy** — bot explicitly refuses out-of-scope questions and redirects to official docs
- **Privacy-first architecture** — all embeddings run locally, no user queries leave the pipeline
- **Protocol-specific prompting** — each mode has a hand-tuned system prompt targeting that protocol's known confusion points
- **Modular by design** — new protocols added via CLI flags, existing collections are never modified during pivots
- **Honest failure mode** — when context is insufficient, the bot says so clearly rather than guessing
