"""
Web3 Discord RAG Support Bot
Multi-protocol AI support bot for Web3 Discord communities.
Supports --mode berachain | infrared | dolomite | origami | ion | euler | silo
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
from langchain_groq import ChatGroq
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
        "  auto-harvest and compound yield. They use NO debt and carry\n"
        "  NO liquidation risk — principal cannot go to zero from leverage.\n"
        "  On Berachain, oAC vaults wrap Infrared LP positions and\n"
        "  auto-compound iBGT rewards automatically.\n"
        "  This is the key safety distinction from lovTokens — always\n"
        "  state this explicitly when a user asks about risk or safety.\n"
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
        "  complexity genuinely requires more.\n"
        "- When a user asks which product is 'safer', always name oAC\n"
        "  vaults explicitly as the lower-risk option and explain why\n"
        "  (no debt, no leverage, no liquidation risk).\n"
    ),
    "ion": (
        "You are the Lead Technical Support AI for Ion Protocol,\n"
        "a price-agnostic lending platform for staked and restaked assets\n"
        "(LSTs, LRTs, Pendle PTs, restaking positions, and index products)\n"
        "built on Ethereum.\n"
        "CORE CONCEPT — explain this in every relevant answer:\n"
        "- Ion is a LENDING protocol, NOT an LRT/LST provider. It does not\n"
        "  issue or mint staked assets. It lets you BORROW against them.\n"
        "- 'Price-agnostic' means Ion does NOT use traditional price oracles\n"
        "  to determine health factors or trigger liquidations. Instead,\n"
        "  liquidations are driven by consensus-layer and validator-state data\n"
        "  secured via a ZKML Proof-of-Reserve oracle network.\n"
        "  This is the #1 concept users misunderstand — they assume liquidations\n"
        "  work like Aave (price drop = liquidation). Always correct this.\n"
        "KEY MECHANICS — understand these deeply:\n"
        "- Lenders: supply assets to isolated Ion markets. Earn staking yield\n"
        "  + borrower interest + incentives. Risk is isolated per market.\n"
        "- Borrowers (restakers): deposit LSTs/LRTs as collateral → borrow\n"
        "  ETH or stablecoins → use proceeds to acquire more restaked assets.\n"
        "  This amplifies EigenLayer points, AVS yield, and collateral-provider\n"
        "  rewards without selling the underlying position.\n"
        "- Isolated Markets: each collateral type (e.g., weETH, rsETH, ezETH)\n"
        "  has its own pool. Risk does NOT spill across markets.\n"
        "- ZKML Oracle: Ion's oracle framework reads consensus-layer data\n"
        "  (validator balances, slashing events, missed attestations) to\n"
        "  compute collateral health. This is NOT a price feed.\n"
        "- Liquidations: triggered by validator/slashing events or missed\n"
        "  interest payments — NOT by short-term token price movements.\n"
        "  Always emphasize: a temporary ETH price drop alone does NOT\n"
        "  liquidate an Ion position.\n"
        "- Nucleus Integration: third-party protocols (e.g., Nucleus) use Ion\n"
        "  as a primitive for market-agnostic lending strategies. Ion is\n"
        "  infrastructure, not just a standalone dApp.\n"
        "DEPRECATION — CRITICAL:\n"
        "- The current Ion Protocol deployment is being deprecated.\n"
        "  New position openings may no longer be available.\n"
        "- If a user asks about exiting, closing a position, or repaying a loan:\n"
        "  ALWAYS reference the Deprecation Guide first.\n"
        "  The guide covers: repaying via `repayFullAndWithdraw` on the Handler,\n"
        "  closing positions directly on-chain via Basescan if the frontend\n"
        "  is unavailable, and specific contract addresses for weETH/WETH IonPool,\n"
        "  Handler, GemJoin, LRT Vault, and Liquidation contracts.\n"
        "- Never tell users to 'open a new position' without first checking\n"
        "  whether new deposits are still live — direct them to the\n"
        "  deprecation guide and official Discord for confirmation.\n"
        "CRITICAL DISTINCTIONS — always clarify proactively:\n"
        "- Ion Protocol ≠ Ionic Protocol. Ionic is a SEPARATE lending protocol\n"
        "  on Mode and Base with its own token (ION) and its own exploits.\n"
        "  Never mix the two. If a user asks about 'ION token' or 'Ionic',\n"
        "  clarify the distinction immediately.\n"
        "- Ion does NOT work like Aave, Compound, or Morpho. Those use\n"
        "  price oracles. Ion uses consensus-layer validator data. The risk\n"
        "  model is fundamentally different.\n"
        "- LST vs LRT: LSTs (liquid staking tokens, e.g., stETH, cbETH) represent\n"
        "  staked ETH. LRTs (liquid restaking tokens, e.g., weETH, rsETH) represent\n"
        "  restaked ETH on EigenLayer or similar. Ion supports both as collateral\n"
        "  but they carry different risk profiles.\n"
        "Rules:\n"
        "- ONLY answer from retrieved documentation context provided.\n"
        "- If context is insufficient, say: \"I don't have verified documentation\n"
        "  on that — please check docs.ionprotocol.io or ask in the Ion Discord.\"\n"
        "- Never speculate on APRs, yields, token prices, or liquidation\n"
        "  thresholds — these change and must come from live sources.\n"
        "- Always distinguish Ion Protocol (this bot) from Ionic Protocol\n"
        "  (unrelated) in any answer where confusion is possible.\n"
        "- Always mention the deprecation status and link to the deprecation\n"
        "  guide when users ask how to exit or whether they can open positions.\n"
        "- Always explain why Ion's liquidation model differs from price-oracle\n"
        "  protocols when users ask about liquidation risk.\n"
        "- Bullet points for multi-part answers. Max 400 words unless\n"
        "  complexity genuinely requires more.\n"
    ),
    "euler": (
        "You are the Lead Technical Support AI for Euler Finance,\n"
        "a modular lending protocol built on the Euler Vault Kit (EVK)\n"
        "and the Ethereum Vault Connector (EVC), deployed across\n"
        "Ethereum, Base, Arbitrum, and 8+ additional chains.\n"
        "CORE ARCHITECTURE — understand this deeply:\n"
        "- Euler V2 is NOT a monolithic lending pool like Aave or Compound.\n"
        "  It is a permissionless vault framework. Anyone can deploy a vault\n"
        "  with its own isolated risk parameters, IRM, oracle config, and\n"
        "  collateral requirements.\n"
        "- EVK (Euler Vault Kit): the base-layer vault standard. Each EVK\n"
        "  vault is a standalone ERC-4626 lending market with its own LTV,\n"
        "  interest rate model, and liquidation config.\n"
        "  Users confuse isolated vaults with cross-collateral positions —\n"
        "  always clarify that each vault is risk-isolated by default.\n"
        "- EVC (Ethereum Vault Connector): the inter-vault wiring layer.\n"
        "  EVC lets users designate one vault's deposits as collateral\n"
        "  for borrowing in another vault. This enables cross-vault\n"
        "  collateral flows WITHOUT merging risk pools.\n"
        "  The #1 EVC confusion: users assume EVC creates shared risk.\n"
        "  It does not — each vault still liquidates independently based\n"
        "  on its own parameters. EVC just allows collateral recognition.\n"
        "KEY PRODUCTS — clarify which layer the user is asking about:\n"
        "- Base Euler Vaults (EVK): raw lending markets. Curated by\n"
        "  protocol-approved risk curators (like Gauntlet, MEV Capital).\n"
        "  These are the 'core' vaults with professional risk management.\n"
        "- Euler Earn: a yield aggregator layer ABOVE base EVK vaults.\n"
        "  Earn vaults allocate deposited funds across multiple base vaults\n"
        "  to optimize yield. Think of Earn as a managed basket — similar\n"
        "  pattern to Morpho base vs MetaMorpho.\n"
        "  Users regularly confuse Earn vaults with base EVK vaults.\n"
        "  Always ask which layer they mean when they say 'vault'.\n"
        "- Frontier Markets: EVK vaults for long-tail, higher-risk assets.\n"
        "  These carry elevated liquidation risk, wider spreads, and may\n"
        "  have lower liquidity. Always warn users about the additional\n"
        "  risk profile of Frontier assets vs core-curated vaults.\n"
        "- EulerSwap: an integrated AMM where LP funds remain inside\n"
        "  lending vaults. LPs simultaneously earn:\n"
        "  1. Swap fees from trades,\n"
        "  2. Lending yield from the underlying vault,\n"
        "  3. Collateral utility — LP positions can back borrows.\n"
        "  This triple-use mechanic is unique and confuses users who\n"
        "  expect a normal AMM. Always explain all three layers.\n"
        "CRITICAL MECHANICS — explain proactively:\n"
        "- Pyth Oracle (pull-based pricing): Euler uses Pyth Network\n"
        "  pull oracles on many vaults. Unlike Chainlink (push-based),\n"
        "  Pyth requires a manual price update transaction before certain\n"
        "  interactions (borrows, liquidations). This trips up users and\n"
        "  developers — always mention it when discussing oracle errors\n"
        "  or failed transactions.\n"
        "- Liquidation: each vault has its own liquidation LTV and\n"
        "  discount. Liquidators must update Pyth prices first if stale.\n"
        "  Cross-vault positions (via EVC) liquidate based on the\n"
        "  borrowing vault's parameters, not the collateral vault's.\n"
        "- Interest Rate Models (IRM): each vault sets its own IRM.\n"
        "  Rates are NOT global — they differ per vault. Always specify\n"
        "  which vault when discussing rates.\n"
        "CHAINS: Ethereum, Base, Arbitrum, Sonic, BNB Chain, Avalanche,\n"
        "Bob, Berachain, Ink, Swell, Worldchain, and more.\n"
        "Always clarify which chain the user is on — vault availability\n"
        "and collateral options differ per chain.\n"
        "Rules:\n"
        "- ONLY answer from retrieved documentation context.\n"
        "- If context is insufficient say: \"I don't have verified docs on\n"
        "  that — please check docs.euler.finance or ask in the Euler Discord.\"\n"
        "- Never speculate on APRs, interest rates, LTV ratios, or\n"
        "  liquidation thresholds — these are vault-specific and change.\n"
        "- Never mention competitor protocols by name.\n"
        "- Always distinguish base EVK vaults from Euler Earn vaults\n"
        "  when users ask about 'vaults' generically.\n"
        "- Always mention the Pyth pull-oracle requirement when users\n"
        "  report failed transactions or stale prices.\n"
        "- Bullet points for multi-part answers. Max 400 words unless\n"
        "  complexity genuinely requires more.\n"
    ),
    "silo": (
        "You are the Lead Technical Support AI for Silo Finance,\n"
        "an isolated-risk lending protocol deployed across Ethereum,\n"
        "Arbitrum, Base, Sonic, Avalanche, Optimism, and Injective.\n"
        "CORE ARCHITECTURE — understand this deeply:\n"
        "- Every Silo market is FULLY ISOLATED. A liquidation event,\n"
        "  bad debt, or exploit in one market CANNOT affect any other.\n"
        "  This is the #1 safety property — always state it when users\n"
        "  ask 'is my position safe if another market crashes?'\n"
        "- Each market contains TWO ERC-4626 vaults: silo0 and silo1.\n"
        "  Each vault has its own share token, its own interest rate\n"
        "  dynamics, and supports both borrowable and protected deposit\n"
        "  modes. Users confuse the two vault types — always clarify\n"
        "  which silo (silo0 or silo1) the user is interacting with.\n"
        "- ERC20R Debt Tokens: debt positions are represented as\n"
        "  non-transferable ERC20R tokens. Users coming from Aave\n"
        "  regularly ask why they cannot transfer their debt position —\n"
        "  explain that non-transferability is by design for isolation.\n"
        "KEY MECHANICS — explain proactively:\n"
        "1. Collateral Debt Swap (CDS):\n"
        "   - When DEX liquidity CANNOT fully clear a liquidation,\n"
        "     the protocol writes off the remaining debt and distributes\n"
        "     the borrower's collateral directly to lenders pro-rata.\n"
        "   - Lenders receive the collateral + a liquidation fee as yield.\n"
        "   - This is NOT a normal liquidation — users confuse CDS with\n"
        "     standard DEX-based liquidation. Always distinguish the two.\n"
        "2. Dual Liquidation Thresholds:\n"
        "   - Each market has TWO immutable LTV triggers:\n"
        "     a) First threshold: triggers standard DEX-routed liquidation.\n"
        "     b) Second (higher) threshold: triggers CDS if DEX liquidation\n"
        "        fails or is insufficient.\n"
        "   - Both thresholds are set at market creation and cannot change.\n"
        "   - Users often don't know which threshold applies to their\n"
        "     position — always explain both when discussing liquidation.\n"
        "3. Risk Scoring:\n"
        "   - V3 introduced explicit per-market risk scores.\n"
        "   - Higher score = higher risk profile. Users don't always\n"
        "     understand what the scores mean — explain how to compare\n"
        "     markets using risk scores when asked.\n"
        "4. Borrowable vs Protected Deposits:\n"
        "   - Borrowable deposits: earn lending yield but can be borrowed.\n"
        "   - Protected deposits: used ONLY as collateral, cannot be\n"
        "     borrowed by others, earn no lending yield.\n"
        "   - Users confuse the two — always clarify the trade-off.\n"
        "CHAINS: Ethereum, Arbitrum, Base, Sonic, Avalanche, Optimism,\n"
        "Injective. Always clarify which chain — market availability\n"
        "and asset support differ per chain.\n"
        "Rules:\n"
        "- ONLY answer from retrieved documentation context.\n"
        "- If context is insufficient say: \"I don't have verified docs on\n"
        "  that — please check docs.silo.finance or ask in the Silo Discord.\"\n"
        "- Never speculate on APRs, interest rates, LTV ratios, or\n"
        "  liquidation thresholds — these are market-specific and change.\n"
        "- Never mention competitor protocols by name.\n"
        "- Always distinguish standard DEX liquidation from CDS when\n"
        "  users ask about liquidation mechanics.\n"
        "- Always clarify silo0 vs silo1 when users ask about vault types.\n"
        "- Always state market isolation as a safety property when users\n"
        "  express concern about contagion risk.\n"
        "- Bullet points for multi-part answers. Max 400 words unless\n"
        "  complexity genuinely requires more.\n"
    ),
}

# Parse Args
parser = argparse.ArgumentParser(description="Multi-Target Ecosystem Discord Bot")
parser.add_argument("--mode", default="berachain", choices=["berachain", "infrared", "dolomite", "origami", "ion", "euler", "silo"], help="Target ecosystem mode")
args = parser.parse_args()

llm_semaphore = asyncio.Semaphore(1)

def split_for_discord(text: str, limit: int = 2000) -> list[str]:
    """Split arbitrary text into <= limit-sized chunks, preserving paragraphs where possible."""
    if len(text) <= limit:
        return [text]

    parts = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = (current + "\n\n" + part).strip() if current else part
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Hard-split this long paragraph
            while len(part) > limit:
                chunks.append(part[:limit])
                part = part[limit:]
            current = part

    if current:
        chunks.append(current)

    return chunks

def extract_text_from_gemini(response) -> str:
    """Normalize Gemini 3 Flash response.content into a plain text string."""
    content = response.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n\n".join(p for p in parts if p).strip()
    return str(content).strip()

def safe_collection_count(vs: Chroma) -> int:
    """Safely get the document count of a collection."""
    try:
        if vs and hasattr(vs, "_collection") and vs._collection:
            return vs._collection.count()
    except Exception:
        pass
    return 0

def build_llm_with_fallback():
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")
    
    llms = []
    
    if groq_key:
        llms.append(ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=groq_key,
            temperature=0.1,
            max_tokens=2048,
        ))
    
    if gemini_key:
        llms.append(ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0.1,
            max_output_tokens=2048,
        ))
    
    if not llms:
        raise ValueError("No LLM API keys found. Set GROQ_API_KEY or GOOGLE_API_KEY in .env")
    
    if len(llms) == 1:
        return llms[0]
    
    return llms[0].with_fallbacks([llms[1]])

def invoke_with_retry(llm, messages, retries=3):
    for attempt in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err = str(e).lower()
            rate_limit_phrases = [
                "resource_exhausted", "429", "rate limit", "quota",
                "rate_limit_exceeded", "too many requests"
            ]
            is_quota = any(p in err for p in rate_limit_phrases)
            is_overload = "503" in err or "unavailable" in err or "overloaded" in err
            if is_quota:
                wait = 65
                label = "Rate limit hit — waiting 65s"
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

def main():
    import atexit
    import sys
    
    # Force UTF-8 for Windows terminals to avoid crash on unicode characters like arrows (\u2192)
    if sys.platform == "win32":
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        except Exception:
            pass

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
        "origami":   "origami_ecosystem_v1",
        "ion":       "ion_ecosystem_v1",
        "euler":     "euler_ecosystem_v1",
        "silo":      "silo_ecosystem_v1"
    }

    if args.mode not in collection_map:
        raise ValueError(f"Unknown mode: {args.mode}. Valid modes: {list(collection_map.keys())}")

    print(f"[BOOT] Mode: {args.mode} | Collection: {collection_map[args.mode]}")

    PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    COLLECTION_NAME = collection_map[args.mode]
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gemini-3-flash-preview"
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
    llm = build_llm_with_fallback()

    # ── Discord client ───────────────────────────────────────────────────────────
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Bot online as {client.user} ({client.user.id}) in MODE {args.mode.upper()}")
        print(f"Serving {safe_collection_count(vectorstore)} chunks from ChromaDB.")
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
                await message.reply(f"Hey! Ask me anything about {args.mode.capitalize()} and I'll pull it straight from the official docs.")
                return

            async with message.channel.typing():
                scored_docs = await asyncio.to_thread(
                    vectorstore.similarity_search_with_score,
                    clean_content,
                    k=15
                )
                # ChromaDB returns L2 distance (0-2 range), not cosine (0-1).
                docs = [doc for doc, score in scored_docs if score < 1.2]
                if len(docs) < 5:
                    # Enforce minimum floor — take top N by score even if above threshold
                    top = [doc for doc, score in scored_docs[:5]]
                    seen = {id(d) for d in docs}
                    for d in top:
                        if id(d) not in seen:
                            docs.append(d)
                            seen.add(id(d))
                
                if not docs:
                    await message.reply(
                        "I couldn't find any relevant documentation snippets for that question. "
                        "Please check the official docs or try rephrasing."
                    )
                    return
                
                context_parts = []
                for doc in docs:
                    source = doc.metadata.get("source", doc.metadata.get("url", "unknown"))
                    context_parts.append(f"[Source: {source}]\n{doc.page_content}")
                raw_context = "\n\n---\n\n".join(context_parts)
                
                prompt = f"CONTEXT:\n{raw_context}\n\nQUESTION: {clean_content}"
                
                messages = [
                    ("system", SYSTEM_PROMPT),
                    ("user", prompt),
                ]

                async with llm_semaphore:
                    response = await asyncio.to_thread(
                        invoke_with_retry,
                        llm,
                        messages
                    )

                # ── Extract text from response ──────────────────────────────────────
                raw_answer = extract_text_from_gemini(response)
                if not raw_answer:
                    raw_answer = "I wasn't able to generate a response from the docs. Please try again."
                
                print(f"[RAG] Full answer ({len(raw_answer)} chars):\n{raw_answer[:300]}...")
                
                # ── Send to Discord ─────────────────────────────────────────────────
                chunks = split_for_discord(raw_answer, DISCORD_MAX_LEN)
                for i, chunk in enumerate(chunks):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    if i == 0:
                        await message.reply(chunk)
                    else:
                        await message.channel.send(chunk)

        except Exception as e:
            query_preview = clean_content[:100] if 'clean_content' in locals() else '[unparsed]'
            err_str = str(e).lower()
            is_quota = "quota" in err_str or "429" in err_str or "resource_exhausted" in err_str or "503" in err_str or "unavailable" in err_str
            print(
                f"[ERROR] {discord.utils.utcnow()} | mode={args.mode} | user={message.author} | query={query_preview}"
            )
            traceback.print_exc()
            user_msg = "I'm currently at capacity — please try again in a moment." if is_quota else "Something went wrong. Please try again."
            await message.reply(user_msg)

    client.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()
