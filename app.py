import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, LLMConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

# --- Initialize Environment ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("❌ GOOGLE_API_KEY not found in .env. Please add it to continue.")
    st.stop()

class PageContent(BaseModel):
    title: str = Field(..., description="The title of the page.")
    text: str = Field(..., description="The full main text content.")

# --- Asynchronous Logic ---
async def crawl_task(semaphore, crawler, url, instruction):
    async with semaphore:
        # Switching Crawl4AI to Gemini 2.5 Pro for deep extraction
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
            excluded_tags=[]
        )
        result = await crawler.arun(url=url, config=run_cfg)
        return url, result

async def run_batch_crawl(urls, instruction):
    # Pro models have lower rate limits than Flash; 
    # Semaphore(2) is safer for a 2.5 Pro batch crawl on a personal API key.
    semaphore = asyncio.Semaphore(2) 
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        tasks = [crawl_task(semaphore, crawler, url, instruction) for url in urls]
        return await asyncio.gather(*tasks)

# --- Streamlit UI ---
st.set_page_config(page_title="Gemini 2.5 Researcher", layout="wide")
st.title("♊ Gemini 2.5 Pro Site Researcher")

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = ""
if "found_links" not in st.session_state:
    st.session_state.found_links = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- STEP 1: Discovery ---
st.header("Step 1: Map the Site")
col1, col2 = st.columns([2, 1])
with col1:
    start_url = st.text_input("Entry URL:", "https://")
with col2:
    if st.button("Find Links"):
        with st.spinner("Mapping with Gemini 2.5 Pro..."):
            async def get_links():
                async with AsyncWebCrawler() as crawler:
                    return await crawler.arun(url=start_url)
            res = asyncio.run(get_links())
            if res.success:
                st.session_state.found_links = [l['href'] for l in res.links.get("internal", []) if l.get('href')]
                st.success(f"Discovered {len(st.session_state.found_links)} internal links.")

# --- STEP 2: Deep Read ---
if st.session_state.found_links:
    st.divider()
    st.header("Step 2: Deep Read (Recursive Scrape)")
    select_all = st.checkbox("Select ALL links")
    selected_pages = st.session_state.found_links if select_all else st.multiselect("Pick pages:", st.session_state.found_links)

    if st.button("Execute Deep Scrape"):
        if selected_pages:
            progress_bar = st.progress(0)
            results = asyncio.run(run_batch_crawl(selected_pages, "Extract all biography and organizational details."))
            for i, (url, res) in enumerate(results):
                if res.success:
                    st.session_state.knowledge_base += f"\nSOURCE URL: {url}\nCONTENT:\n{res.markdown}\n---\n"
                progress_bar.progress((i + 1) / len(results))
            st.success("Deep Read Complete!")

# --- STEP 3: Chat with Gemini 2.5 Pro ---
if st.session_state.knowledge_base:
    st.divider()
    st.header("Step 3: Interactive Chat")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if chat_input := st.chat_input("Ask about the site content..."):
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)

        with st.chat_message("assistant"):
            # Specifically initializing the 2.5 Pro model
            model = genai.GenerativeModel('gemini-2.5-pro')
            
            # Formulating the context
            full_context = f"Context from Site Crawl:\n{st.session_state.knowledge_base}"
            
            # Map history for Gemini's native chat object
            history = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in st.session_state.messages[:-1]]
            
            chat = model.start_chat(history=history)
            
            # Combine context with the latest question
            response = chat.send_message(f"{full_context}\n\nQuestion: {chat_input}")
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    if st.button("Reset Knowledge Base"):
        st.session_state.knowledge_base = ""
        st.session_state.messages = []
        st.rerun()