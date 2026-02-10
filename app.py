import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, LLMConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
import chromadb # <--- New Import

# --- Initialize Environment ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("❌ GOOGLE_API_KEY not found in .env.")
    st.stop()

# Initialize ChromaDB (Local Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="site_research")

class PageContent(BaseModel):
    title: str = Field(..., description="The title of the page.")
    text: str = Field(..., description="The full main text content.")

# ... [Previous crawl_task and run_batch_crawl functions remain the same] ...

# --- NEW: Function to Save to Vector Store ---
def save_to_rag(content_str, url):
    """Chunks content and saves it to ChromaDB."""
    # Simple chunking: split by double newlines (paragraphs)
    chunks = [c.strip() for c in content_str.split("\n\n") if len(c.strip()) > 50]
    
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": url}],
            ids=[f"{url}_{i}"]
        )

# --- Streamlit UI ---
st.set_page_config(page_title="Gemini RAG Scraper", layout="wide")
st.title("♊ Gemini 2.5 RAG Researcher")

# ... [Step 1: Map the Site remains the same] ...

# --- STEP 2: Deep Read & Save ---
if st.session_state.found_links:
    st.divider()
    st.header("Step 2: Deep Read & Save to RAG")
    select_all = st.checkbox("Select ALL links")
    selected_pages = st.session_state.found_links if select_all else st.multiselect("Pick pages:", st.session_state.found_links)

    col_btn1, col_btn2 = st.columns(2)
    
    if col_btn1.button("Execute Deep Scrape"):
        if selected_pages:
            results = asyncio.run(run_batch_crawl(selected_pages, "Extract all biography and organizational details."))
            for url, res in results:
                if res.success:
                    st.session_state.knowledge_base += f"\nSOURCE: {url}\n{res.markdown}\n---\n"
            st.success("Deep Read Complete!")

    if col_btn2.button("💾 Save to Vector Store (RAG)"):
        if st.session_state.knowledge_base:
            with st.spinner("Embedding and saving to local database..."):
                save_to_rag(st.session_state.knowledge_base, "multi_page_crawl")
                st.success("Data successfully stored in /chroma_data!")
        else:
            st.warning("Scrape some data first!")

# --- STEP 3: Chat (RAG-Enabled) ---
if st.session_state.knowledge_base:
    st.divider()
    st.header("Step 3: RAG-Powered Chat")
    
    chat_mode = st.radio("Search Mode:", ["Full Context (All Data)", "RAG (Semantic Search)"])

    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("assistant"):
            if chat_mode == "RAG (Semantic Search)":
                # Search the local vector DB for the top 5 most relevant snippets
                results = collection.query(query_texts=[prompt], n_results=5)
                context = "\n".join(results['documents'][0])
                final_prompt = f"Using ONLY the following snippets, answer the question: {prompt}\n\nSNIPPETS:\n{context}"
            else:
                final_prompt = f"Knowledge Base:\n{st.session_state.knowledge_base}\n\nQuestion: {prompt}"

            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(final_prompt)
            st.markdown(response.text)