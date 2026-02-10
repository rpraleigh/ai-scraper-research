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
    st.error("❌ GOOGLE_API_KEY not found in .env.")
    st.stop()

# Initialize Local Vector Database
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="site_research")

class PageContent(BaseModel):
    title: str = Field(..., description="The title of the page.")
    text: str = Field(..., description="The full main text content.")

# --- 2. Sidebar Logic (Mission Control) ---
with st.sidebar:
    st.header("🗄️ Knowledge Base")
    
    # Check if DB has data
    try:
        all_data = collection.get(include=['metadatas'])
        if all_data['ids']:
            # Metrics
            sources = [meta.get('source', 'Unknown') for meta in all_data['metadatas']]
            unique_sources = list(set(sources))
            
            col_a, col_b = st.columns(2)
            col_a.metric("Pages", len(unique_sources))
            col_b.metric("Chunks", len(all_data['ids']))
            
            st.divider()
            
            # --- NEW: Chunk Inspector ---
            st.subheader("🔍 Inspector")
            selected_source = st.selectbox("Select a page to inspect:", unique_sources)
            
            if selected_source:
                # Query DB for just this specific URL
                source_data = collection.get(where={"source": selected_source})
                st.caption(f"Stored {len(source_data['ids'])} snippets from this page.")
                
                # Scrollable expanders for each chunk
                for i, text in enumerate(source_data['documents']):
                    with st.expander(f"Snippet {i+1}", expanded=False):
                        st.text(text)
            
            st.divider()
            
            # Clear Database Button
            if st.button("🗑️ Wipe Database", type="primary"):
                chroma_client.delete_collection("site_research")
                # Re-create immediately so app doesn't crash
                collection = chroma_client.get_or_create_collection(name="site_research")
                st.rerun()
                
        else:
            st.info("Database is empty. Start crawling!")
            
    except Exception as e:
        st.error(f"DB Error: {e}")

# --- 3. Asynchronous Scraper Logic ---
async def crawl_task(semaphore, crawler, url, instruction):
    async with semaphore:
        llm_config = LLMConfig(provider="gemini/gemini-2.5-pro", api_token=GOOGLE_API_KEY)
        llm_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            schema=PageContent.model_json_schema(),
            instruction=instruction
        )
        run_cfg = CrawlerRunConfig(
            extraction_strategy=llm_strategy, 
            cache_mode=CacheMode.BYPASS, 
            magic=True,
            excluded_tags=[]
        )
        result = await crawler.arun(url=url, config=run_cfg)
        return url, result

async def run_batch_crawl(urls, instruction):
    semaphore = asyncio.Semaphore(2)
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        tasks = [crawl_task(semaphore, crawler, url, instruction) for url in urls]
        return await asyncio.gather(*tasks)

# --- 4. Main UI Layout ---
st.title("🕵️‍♂️ Gemini 2.5 Pro Deep Researcher")

if "found_links" not in st.session_state: st.session_state.found_links = []
if "messages" not in st.session_state: st.session_state.messages = []

# --- STEP 1: Site Mapping ---
st.header("Step 1: Map the Site")
col_url, col_btn = st.columns([3, 1])
with col_url:
    start_url = st.text_input("Entry URL:", "https://")
with col_btn:
    if st.button("Discover Links"):
        with st.spinner("Mapping site structure..."):
            async def get_links():
                async with AsyncWebCrawler() as c: return await c.arun(url=start_url)
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
                        chunks = [c.strip() for c in res.markdown.split("\n\n") if len(c) > 50]
                        if chunks:
                            collection.add(
                                documents=chunks,
                                metadatas=[{"source": url}] * len(chunks),
                                ids=[f"{url}_{i}" for i in range(len(chunks))]
                            )
                st.success(f"Archived content to local database.")
                st.rerun()
        else:
            st.warning("Please select at least one page.")

# --- STEP 3: Semantic Research Chat ---
if os.path.exists("./chroma_data"):
    st.divider()
    st.header("Step 3: Interactive Research Chat")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question about the archived site data..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # RAG Retrieval
            search_results = collection.query(query_texts=[query], n_results=5)
            context_text = "\n\n".join(search_results['documents'][0])
            
            # Synthesis
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

    if st.button("Clear Chat Session"):
        st.session_state.found_links = []
        st.session_state.messages = []
        st.rerun()