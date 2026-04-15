import os
import time
import asyncio
import traceback
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

SYSTEM_PROMPT = (
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
    "   - These ARE NOT leveraged — they are yield-optimizing wrappers.\n"
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

def invoke_with_retry(llm, messages, retries=3):
    for attempt in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err = str(e).lower()
            is_quota = "quota" in err or "429" in err or "resource_exhausted" in err
            is_overload = "503" in err or "unavailable" in err or "overloaded" in err
            if is_quota:
                wait = 65
                label = "Quota hit — waiting 65s"
            elif is_overload:
                wait = 30
                label = f"Server overloaded (503) — waiting 30s"
            else:
                wait = 3
                label = f"LLM error: {e}"
            if attempt < retries - 1:
                print(f"[RETRY {attempt+1}/{retries}] {label}")
                time.sleep(wait)
            else:
                raise

async def main():
    PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    COLLECTION_NAME = "origami_ecosystem_v1"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gemini-2.5-flash"
    
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,
        max_output_tokens=2048,
    )
    
    questions = [
        "Is the BYUSD-HONEY vault on Berachain a leveraged product or an auto-compounder, and what's the difference between eAPR and realised APR?",
        "If I deposit into a lovToken vault and the underlying collateral drops 60% in a single day, what happens to my position? Can I lose everything?",
        "What are entry and exit fees on Origami vaults, and why would a user who deposits and withdraws within the same week see a much lower return than the displayed eAPR?",
        "How does the Infrared integration work inside Origami's Berachain vaults — specifically where do the iBGT rewards come from and what happens to them?",
        "What is oUSDC and how does it relate to the folding mechanism — is it something users deposit into directly?"
    ]
    
    results = []
    
    for i, q in enumerate(questions):
        print(f"\nProcessing Q{i+1}...")
        try:
            scored_docs = await asyncio.to_thread(
                vectorstore.similarity_search_with_score,
                q,
                k=15
            )
            docs = [doc for doc, score in scored_docs if score < 1.2]
            if len(docs) < 5:
                # Enforce minimum floor — take top N by score even if above threshold
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
            
            prompt = f"CONTEXT:\n{raw_context}\n\nQUESTION: {q}"
            
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", prompt),
            ]
            
            response = await asyncio.to_thread(invoke_with_retry, llm, messages)
            answer = response.content
            
            print("=" * 40)
            print(f"Q{i+1}: {q}")
            print(f"Chunks retrieved: {len(docs)} (after filtering)")
            print("-" * 40)
            print(answer)
            print("=" * 40)
            
            results.append({"q": q, "success": True, "answer": answer})
            
        except Exception as e:
            print(f"ERROR on Q{i+1}: {e}")
            results.append({"q": q, "success": False, "error": str(e)})
            
        if i < len(questions) - 1:
            print("Waiting 20s...")
            time.sleep(20)
            
    print("\nVERIFICATION COMPLETE")
    print(f"Questions answered: {len([r for r in results if r['success']])}/5")
    errors = [f"Q{i+1}: {r['error']}" for i, r in enumerate(results) if not r['success']]
    print(f"Any empty/error responses: {', '.join(errors) if errors else 'None'}")

if __name__ == "__main__":
    asyncio.run(main())
