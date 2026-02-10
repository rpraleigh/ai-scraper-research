import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, LLMConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
import chromadb

# --- 1. Environment & Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("❌ GOOGLE_API_KEY not found in .env. Please add it to continue.")
    st.stop()

# Initialize Local Vector Database
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="site_research")

class PageContent(BaseModel):
    title: str = Field(..., description="The title of the page.")
    text: str = Field(..., description="The full main text content.")

# --- 2. Asynchronous Scraper Logic ---
async def crawl_task(semaphore, crawler, url, instruction):
    """Worker function to crawl a single page with concurrency control."""
    async with semaphore:
        llm_config = LLMConfig(
            provider="gemini/gemini-2.5-pro", 
            api_token=GOOGLE_API_KEY
        )
        llm_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            schema=PageContent.model_json_schema(),
            instruction=instruction
        )
        run_cfg = CrawlerRunConfig(
            extraction_strategy=llm_strategy, 
            cache_mode=CacheMode.BYPASS, 
            magic=True,
            excluded_tags=[]  # Prevent pruning of menus and links
        )
        result = await crawler.arun(url=url, config=run_cfg)
        return url, result

async def run_batch_crawl(urls, instruction):
    """Orchestrates parallel crawling for the 'ALL' option."""
    semaphore = asyncio.Semaphore(2) # Safe batch size for Gemini 2.5 Pro
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        tasks = [crawl_task(semaphore, crawler, url, instruction) for url in urls]
        return await asyncio.gather(*tasks)

# --- 3. Streamlit UI Layout ---
st.set_page_config(page_title="Gemini RAG Researcher", layout="wide")
st.title("🕵️‍♂️ Gemini 2.5 Pro Deep Researcher")

# Initialize Session State
if "found_links" not in st.session_state:
    st.session_state.found_links = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- STEP 1: Site Mapping ---
st.header("Step 1: Map the Site")
col_url, col_btn = st.columns([3, 1])
with col_url:
    start_url = st.text_input("Entry URL:", "https://")
with col_btn:
    if st.button("Discover Links"):
        with st.spinner("Mapping site structure..."):
            async def get_links():
                async with AsyncWebCrawler() as c:
                    return await c.arun(url=start_url)
            res = asyncio.run(get_links())
            if res.success:
                st.session_state.found_links = [l['href'] for l in res.links.get("internal", []) if l.get('href')]
                st.success(f"Discovered {len(st.session_state.found_links)} internal links.")

# --- STEP 2: Deep Scrape & RAG Storage ---
if st.session_state.found_links:
    st.divider()
    st.header("Step 2: Deep Read & Archive")
    
    select_all = st.checkbox("Select ALL discovered links")
    selected_pages = st.session_state.found_links if select_all else st.multiselect("Pick pages to read:", st.session_state.found_links)

    if st.button("💾 Scrape & Save to Local RAG"):
        if selected_pages:
            with st.spinner(f"Deep scraping {len(selected_pages)} pages..."):
                results = asyncio.run(run_batch_crawl(selected_pages, "Extract all biography, organizational text, and roles."))
                
                for url, res in results:
                    if res.success:
                        # Chunking logic for RAG
                        chunks = [c.strip() for c in res.markdown.split("\n\n") if len(c) > 50]
                        if chunks:
                            collection.add(
                                documents=chunks,
                                metadatas=[{"source": url}] * len(chunks),
                                ids=[f"{url}_{i}" for i in range(len(chunks))]
                            )
                st.success(f"Archived content to local database in ./chroma_data/")
        else:
            st.warning("Please select at least one page.")

# --- STEP 3: Semantic Research Chat ---
# Check if database exists to enable chat
if os.path.exists("./chroma_data"):
    st.divider()
    st.header("Step 3: Interactive Research Chat")
    st.info("Querying local vector database via Gemini 2.5 Pro.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question about the archived site data..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # 1. RAG Retrieval: Search the local vector DB
            search_results = collection.query(query_texts=[query], n_results=5)
            context_text = "\n\n".join(search_results['documents'][0])
            
            # 2. Synthesis: Send retrieved context to Gemini
            model = genai.GenerativeModel('gemini-2.5-pro')
            rag_prompt = f"""
            You are a professional researcher. Use the following snippets extracted from a website 
            to answer the user's question. If the answer is not in the context, say so.
            
            CONTEXT:
            {context_text}
            
            USER QUESTION: {query}
            """
            
            response = model.generate_content(rag_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    if st.button("Clear Session"):
        st.session_state.found_links = []
        st.session_state.messages = []
        st.rerun()