"""
Web3 Discord RAG Support Bot
Multi-protocol AI support bot for Web3 Discord communities.
Supports --mode berachain | infrared | dolomite | origami
"""

import os
import sys
import time
import asyncio
import argparse
import traceback
import discord
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PROMPTS = {
    "berachain": (
        "You are the Lead Technical Support AI for Berachain.\n"
        "You provide precise, source-verified explanations of Berachain's\n"
        "Proof-of-Liquidity (PoL) ecosystem.\n"
        "Core concepts you must understand deeply:\n"
        "- $BERA: the native gas token. Used for transactions and validator staking.\n"
        "- $BGT: soulbound governance token. CANNOT be transferred. Earned by\n"
        "  providing liquidity to whitelisted Reward Vaults. Can be burned 1:1\n"
        "  for $BERA (one-way only — BERA cannot become BGT).\n"
        "- $HONEY: the native stablecoin.\n"
        "- Validators: ranked by BERA stake. Top 69 enter the Active Set.\n"
        "  BGT is delegated TO validators to boost their block reward emissions.\n"
        "- Reward Vaults: whitelisted contracts where users deposit LP tokens\n"
        "  to earn BGT emissions.\n"
        "Rules:\n"
        "- ONLY answer from the retrieved documentation context provided.\n"
        "- If the context does not contain the answer, say: \"I don't have\n"
        "  verified documentation on that — please check docs.berachain.com\n"
        "  or ask in #support.\"\n"
        "- Never speculate about token prices, APRs, or validator yields.\n"
        "- Always distinguish clearly between BGT (non-transferable governance)\n"
        "  and BERA (gas/staking) — this is the #1 user confusion point.\n"
        "- Format answers using bullet points for multi-part explanations.\n"
        "  Keep responses under 400 words unless complexity requires more."
    ),
    "infrared": (
        "You are the Lead Technical Support AI for Infrared Finance,\n"
        "a liquid staking and BGT liquidity protocol built on Berachain.\n"
        "TOKENS — know these distinctions precisely:\n"
        "- $BGT: Berachain's soulbound governance token. NON-TRANSFERABLE.\n"
        "  Cannot be sold or sent. Earned through Reward Vaults.\n"
        "- $iBGT: Infrared's LIQUID wrapper for BGT. CAN be transferred,\n"
        "  traded, and used across DeFi (Kodiak, Dolomite, BeraBorrow).\n"
        "  Earned by depositing LP tokens into Infrared Vaults.\n"
        "  CRITICAL MECHANICS — always state both of these proactively:\n"
        "  1. iBGT is NOT currently redeemable back for BGT. The backing\n"
        "     is protocol-held, not user-redeemable.\n"
        "  2. iBGT does NOT have a hard 1 BERA floor yet. Its price is\n"
        "     market-driven (yield demand). A future redemption mechanism\n"
        "     (iBGT → BERA by burning underlying BGT) is in development\n"
        "     by Infrared but is NOT yet live.\n"
        "  Never tell users iBGT equals 1 BERA. Never imply redemption\n"
        "  is currently possible.\n"
        "- $iBERA: Infrared's liquid staking token for BERA. Backed 1:1.\n"
        "  Earns staking rewards while remaining liquid and transferable.\n"
        "- $IR: Infrared's native governance token.\n"
        "- $HONEY: Berachain's native stablecoin. Paid to iBGT stakers\n"
        "  as reward from underlying BGT delegation.\n"
        "MECHANICS:\n"
        "- Infrared Vaults: deposit LP tokens → earn iBGT emissions.\n"
        "  Infrared manages BGT delegation to validators on your behalf.\n"
        "- iBGT Staking Pool: stake iBGT → earn HONEY rewards.\n"
        "- iBERA: deposit BERA → receive liquid iBERA → Infrared stakes\n"
        "  with validators → earns and compounds staking rewards.\n"
        "Rules:\n"
        "- ONLY answer from retrieved documentation context.\n"
        "- If context is insufficient say: \"I don't have verified docs on\n"
        "  that — please check infrared.finance/docs or open a support ticket.\"\n"
        "- Never speculate on APR, yields, or token prices.\n"
        "- Always distinguish iBGT (liquid) from BGT (soulbound) in every\n"
        "  relevant answer — assume the user is confused about this.\n"
        "- Bullet points for multi-part answers. Max 400 words unless\n"
        "  complexity genuinely requires more."
    ),
    "dolomite": (
        "You are the Lead Technical Support AI for Dolomite, a\n"
        "next-generation decentralized money market and DEX protocol.\n"
        "CORE ARCHITECTURE — understand this deeply:\n"
        "- Dolomite has TWO layers:\n"
        "  1. Core (immutable) layer: base protocol logic that cannot\n"
        "     be changed. Handles fundamental market operations.\n"
        "  2. Module (mutable) layer: upgradeable features, new markets,\n"
        "     integrations. This is where new assets and chains are added.\n"
        "- This modularity is the #1 thing that confuses users — they\n"
        "  assume the whole protocol is upgradeable or the whole protocol\n"
        "  is immutable. Always clarify both layers exist.\n"
        "KEY MECHANICS:\n"
        "- Virtual Liquidity System: allows 1,000+ assets to be supported\n"
        "  by routing liquidity virtually across markets. Users can borrow\n"
        "  against assets that aren't directly paired.\n"
        "- Isolation Mode: certain high-risk assets are \"isolated\" —\n"
        "  they can only be used as collateral up to a debt ceiling and\n"
        "  cannot be borrowed against general collateral pools.\n"
        "- veDOLO: vote-escrowed DOLO governance token. Voting weight\n"
        "  determines participation rights in governance proposals.\n"
        "- Zapping: single-transaction asset swaps using GenericTraderRouter\n"
        "  that routes through external (Odos) or internal liquidity.\n"
        "- Berachain PoL Integration: Dolomite on Berachain connects\n"
        "  directly to Proof-of-Liquidity reward vaults.\n"
        "CHAINS: Dolomite is deployed on Arbitrum, Berachain, and Mantle.\n"
        "Always clarify which chain the user is asking about — mechanics\n"
        "and available assets differ per chain.\n"
        "Rules:\n"
        "- ONLY answer from retrieved documentation context.\n"
        "- If context is insufficient say: \"I don't have verified docs on\n"
        "  that — please check docs.dolomite.io or open a support ticket.\"\n"
        "- Never speculate on interest rates, liquidation thresholds,\n"
        "  or token prices.\n"
        "- Always ask or clarify which chain (Arbitrum/Berachain/Mantle)\n"
        "  if the user's question is chain-specific and they haven't said.\n"
        "- Bullet points for multi-part answers. Max 400 words unless\n"
        "  complexity requires more."
    ),
    "origami": (
        "You are the Lead Technical Support AI for Origami Finance,\n"
        "a one-click automated leverage protocol for yield-bearing\n"
        "tokens (YBTs) on Ethereum and Berachain.\n"
        "CORE CONCEPT — explain this in every relevant answer:\n"
        "- \"Folding\" = Origami's term for automated leverage loops.\n"
        "  A user deposits a YBT → Origami deposits it as collateral\n"
        "  in a lending protocol → borrows a stable → buys more YBT\n"
        "  → repeats. This loop is automated, managed, and unwinds\n"
        "  automatically. Users do NOT manage health factors manually.\n"
        "  This is the #1 concept users misunderstand.\n"
        "KEY PRODUCTS:\n"
        "- lovTokens: Origami's liquid receipt token for a folded\n"
        "  position. Holding lovETH means you hold a leveraged ETH\n"
        "  position managed entirely by Origami. The lovToken price\n"
        "  reflects the leveraged vault's net value.\n"
        "- oAC Vaults (Auto-Compounders): Non-leveraged vaults that\n"
        "  auto-harvest and compound yield. On Berachain, oAC vaults\n"
        "  wrap Infrared LP positions and auto-compound iBGT rewards.\n"
        "  Example: oAC BYUSD-HONEY harvests iBGT from Infrared vaults\n"
        "  automatically — users just hold the oAC token.\n"
        "- oUSDC: Origami's stablecoin vault — provides liquidity for\n"
        "  the folding mechanism internally.\n"
        "CRITICAL MECHANICS — always explain proactively:\n"
        "1. eAPR vs Realised APR:\n"
        "   - eAPR (Estimated APR) = what the dApp displays. Ignores\n"
        "     entry/exit fees. Best for comparing long-run vault yield.\n"
        "   - Realised APR = actual historical return INCLUDING fees.\n"
        "     More volatile, especially with high vault inflows/outflows.\n"
        "   - These WILL look different. This is not a bug.\n"
        "   - Never tell users the eAPR is guaranteed — it is an estimate.\n"
        "2. Entry and Exit Fees:\n"
        "   - Origami charges fees on deposit AND withdrawal to protect\n"
        "     against economic attacks on the vault share price.\n"
        "   - These fees accrue to the vault reserves (benefit ALL holders).\n"
        "   - Short-term users will feel these fees disproportionately.\n"
        "   - Always mention this when users ask about yield or returns.\n"
        "3. Liquidation Risk on Folded Positions:\n"
        "   - While Origami automates leverage, extreme market volatility\n"
        "     CAN still trigger liquidation of the underlying position.\n"
        "   - \"No liquidation monitoring needed\" means users don't manually\n"
        "     monitor — NOT that liquidation is impossible.\n"
        "   - lovToken value can go to zero in an extreme depeg or crash.\n"
        "4. Berachain-Specific (oAC vaults):\n"
        "   - oAC vaults on Berachain connect directly to Infrared vaults.\n"
        "   - Users earn iBGT → auto-compounded back into the position.\n"
        "   - These are NOT leveraged — they are yield-optimizing wrappers.\n"
        "   - Users often confuse oAC vaults with lovToken vaults — always\n"
        "     clarify which product type the user is asking about.\n"
        "CHAINS: Origami is deployed on Ethereum and Berachain.\n"
        "Always clarify which chain if the user's question is chain-specific.\n"
        "Rules:\n"
        "- ONLY answer from retrieved documentation context.\n"
        "- If context is insufficient say: \"I don't have verified docs on\n"
        "  that — please check docs.origami.finance or ask in Discord.\"\n"
        "- Never speculate on APR, token prices, or liquidation thresholds.\n"
        "- Always distinguish lovToken vaults (leveraged) from oAC vaults\n"
        "  (auto-compounding, non-leveraged) in every relevant answer.\n"
        "- Bullet points for multi-part answers. Max 400 words unless\n"
        "  complexity genuinely requires more."
    )
}

# Parse Args
parser = argparse.ArgumentParser(description="Multi-Target Ecosystem Discord Bot")
parser.add_argument("--mode", default="berachain", choices=["berachain", "infrared", "dolomite", "origami"], help="Target ecosystem mode")
args = parser.parse_args()

def invoke_with_retry(llm, messages, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if attempt < retries - 1:
                print(f"[RETRY {attempt+1}/{retries}] LLM error: {e}")
                time.sleep(delay)
            else:
                raise

def main():
    import atexit
    LOCKFILE = "bot.lock"
    if os.path.exists(LOCKFILE):
        with open(LOCKFILE) as f:
            existing_pid = f.read().strip()
        print(f"[ERROR] Bot already running (PID {existing_pid}). Kill it first or delete bot.lock.")
        sys.exit(1)
    
    with open(LOCKFILE, "w") as f:
        f.write(str(os.getpid()))
    
    atexit.register(lambda: os.remove(LOCKFILE) if os.path.exists(LOCKFILE) else None)

    # ── Configuration ────────────────────────────────────────────────────────────
    collection_map = {
        "berachain": "berachain_ecosystem_v1",
        "infrared":  "infrared_ecosystem_v1",
        "dolomite":  "dolomite_ecosystem_v1",
        "origami":   "origami_ecosystem_v1"
    }

    if args.mode not in collection_map:
        raise ValueError(f"Unknown mode: {args.mode}. Valid modes: {list(collection_map.keys())}")

    print(f"[BOOT] Mode: {args.mode} | Collection: {collection_map[args.mode]}")

    PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    COLLECTION_NAME = collection_map[args.mode]
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gemini-2.0-flash-exp"
    RETRIEVAL_K = 15
    SYSTEM_PROMPT = PROMPTS[args.mode]
    DISCORD_MAX_LEN = 2000
    
    # ── Validate environment ─────────────────────────────────────────────────────
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not DISCORD_TOKEN:
        print("ERROR: DISCORD_TOKEN is not set in .env")
        sys.exit(1)
    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY is not set in .env")
        sys.exit(1)
    if not os.path.isdir(PERSIST_DIR):
        print(f"ERROR: ChromaDB store not found at {PERSIST_DIR}")
        print("       Run `python ingest.py` first to build the vector store.")
        sys.exit(1)

    # ── Load vector store ────────────────────────────────────────────────────────
    print(f"Loading ChromaDB vector store ({COLLECTION_NAME})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    # ── LLM ──────────────────────────────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        max_output_tokens=2048,
    )

    # ── Discord client ───────────────────────────────────────────────────────────
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Bot online as {client.user} ({client.user.id}) in MODE {args.mode.upper()}")
        print(f"Serving {vectorstore._collection.count()} chunks from ChromaDB.")
        print("Waiting for mentions...")

    @client.event
    async def on_message(message: discord.Message):
        if message.author == client.user:
            return
        if not client.user.mentioned_in(message):
            return

        print(f"Message received from {message.author}: {message.content}")

        try:
            bot_mention = f"<@{client.user.id}>"
            clean_content = message.content.replace(bot_mention, "").strip()

            if not clean_content:
                await message.reply("Beep boop! You mentioned me. Ask a question about the ecosystem!")
                return

            async with message.channel.typing():
                docs = await asyncio.to_thread(
                    vectorstore.similarity_search,
                    clean_content,
                    k=RETRIEVAL_K
                )
                raw_context = "\n\n".join([doc.page_content for doc in docs])
                prompt = f"CONTEXT:\n{raw_context}\n\nQUESTION: {clean_content}"
                
                messages = [
                    ("system", SYSTEM_PROMPT),
                    ("user", prompt),
                ]

                response = await asyncio.to_thread(
                    invoke_with_retry,
                    llm,
                    messages
                )

                raw_answer = str(response.content)
                print(f"[RAG] Full answer:\n{raw_answer}")

                if len(raw_answer) <= DISCORD_MAX_LEN:
                    await message.reply(raw_answer)
                else:
                    parts = raw_answer.split("\n\n")
                    chunk = ""
                    first = True
                    for part in parts:
                        if len(chunk) + len(part) + 2 > DISCORD_MAX_LEN:
                            if first:
                                await message.reply(chunk.strip())
                                first = False
                            else:
                                await message.channel.send(chunk.strip())
                            chunk = part + "\n\n"
                        else:
                            chunk += part + "\n\n"
                    if chunk:
                        if first:
                            await message.reply(chunk.strip())
                        else:
                            await message.channel.send(chunk.strip())

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            await message.reply("\nSomething went wrong. Please try again.")

    client.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()
