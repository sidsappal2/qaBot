import os
import time
import json
import sqlite3  # For lightweight caching
import threading
import streamlit as st
import logging
import datetime
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from createvector import VectorStore
from utils import Cleaner

# Load environment variables (e.g., API keys)
load_dotenv()

# Setup logging
logging.basicConfig(filename="qa_bot.log", level=logging.INFO)

# Initialize a global lock for SQLite operations
db_lock = threading.Lock()

def initialize_vector_store():
    """Initialize and return the VectorStore instance."""
    return VectorStore()

def initialize_llm():
    """Initialize and return the ChatGroq language model."""
    return ChatGroq(
        temperature=1,
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama-3.1-70b-versatile"
    )

def initialize_cache():
    """Initialize SQLite database for caching queries."""
    conn = sqlite3.connect("cache.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            question TEXT PRIMARY KEY,
            answer TEXT,
            sources TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn

def log_query(query, answer, sources, response_time):
    """Log query details for evaluation purposes."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "sources": sources,
        "response_time": response_time,
    }
    logging.info(json.dumps(log_entry))

def get_cached_answer(conn, question):
    """Retrieve an answer from the cache."""
    with db_lock:
        cursor = conn.cursor()
        cursor.execute("SELECT answer, sources FROM cache WHERE question = ?", (question,))
        row = cursor.fetchone()
        if row:
            return {"answer": row[0], "sources": row[1]}
    return None

def cache_answer(conn, question, answer, sources):
    """Store an answer in the cache."""
    with db_lock:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO cache (question, answer, sources, timestamp) 
            VALUES (?, ?, ?, ?)
        """, (question, answer, sources, time.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()

def save_to_history(question, answer, sources):
    """Save question, answer, and sources to a history.json file."""
    history_file = "history.json"

    # Load existing history if it exists
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            history = json.load(file)
    else:
        history = []
    # Check if the question already exists in history
    existing_questions = {entry["question"] for entry in history}
    if question in existing_questions:
        return  # Skip saving if the question already exists    

    # Append the new question and answer
    history.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources
    })

    # Write back the updated history
    with open(history_file, "w") as file:
        json.dump(history, file, indent=4)
        
def update_vectorstore(urls, vectorstore, placeholder):
    """Process URLs and update the vector store."""
    loader = WebBaseLoader(urls)
    placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=100
    )
    placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    time.sleep(2)

    # Generate embeddings and update VectorStore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L-12-v3")
    placeholder.text("Building Embedding Vectors...✅✅✅")
    vectorstore = vectorstore.loadVectorStore(docs, embeddings)

    if vectorstore is None:
        st.error("Error while loading VectorStore. Please process URLs again.")
    else:
        placeholder.text("Ready to answer...✅✅✅")
    return vectorstore

def load_history():
    """Load the history from the history.json file."""
    history_file = "history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            return json.load(file)
    return []

def display_history():
    """Display the history in a collapsible format."""
    
    history = load_history()
    if history:
        for entry in history:
            with st.sidebar.expander(entry["question"]):
                st.write(entry['timestamp'])
                st.write(f"**Answer:** {entry['answer']}")
                st.write(f"**Sources:** {entry['sources']}")
    else:
        st.sidebar.write("No history available.")
        
def generate_answer(query, vectorstore, llm, cache_db, placeholder):
    """Generate an answer using the LLM."""
    placeholder.text("Generating answer...✅✅✅")
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever()
    )
    result = chain.invoke({"question": query}, return_only_outputs=True)

    # Display the answer and sources
    placeholder.text("Here's your answer...✅✅✅")
    st.write("Answer:")
    st.write(result["answer"])
    st.subheader("Sources:")
    sources = result.get("sources", "")
    st.write(sources)
    response_time = time.time() - st.session_state.start_time
    log_query(query, result["answer"], sources, response_time)
    save_to_history(query, result["answer"], sources)

    # Cache the new answer
    cache_answer(cache_db,query, result["answer"], sources)

        
def ask_question(query, vectorstore, llm, cache_db, placeholder):
    """Handle user queries, leveraging cache and generating responses if not cached."""
    st.session_state.start_time = time.time()

    # Check if the query is already cached
    cached_response = get_cached_answer(cache_db, query)
    if cached_response:
        # Ensure cached response contains expected keys
        if "answer" in cached_response and "sources" in cached_response:
            placeholder.text("Here's your answer...✅✅✅")
            st.write("Answer (from cache):")
            st.write(cached_response["answer"])
            st.subheader("Sources (from cache):")
            st.write(cached_response["sources"])
            regenerate=st.button("Regenerate")
            #To regenerate from LLM
            if regenerate:
                placeholder.text("Regenerate...✅✅✅")
                generate_answer(query, vectorstore, llm, cache_db, placeholder)
                
            response_time = time.time() - st.session_state.start_time
            log_query(query, cached_response["answer"], cached_response["sources"], response_time)
            save_to_history(query, cached_response["answer"], cached_response["sources"])
            return
        else:
            st.error("Cached response is incomplete. Generating a new response...")

    # Ensure vectorstore is initialized
    if vectorstore is None:
        st.error("VectorStore not initialized. Please process URLs first.")
    else:
        generate_answer(query, vectorstore, llm, cache_db, placeholder)


def main():
    """Main function to run the Streamlit app."""
    # Initialize components
    vectorstore = initialize_vector_store()
    llm = initialize_llm()
    cache_conn = initialize_cache()

    # Streamlit setup
    st.title("QA Chat Bot with Caching")
    st.sidebar.title("Source URLs")

    # Collect URLs from user input
    urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(2)]
    process_url_clicked = st.sidebar.button("Process URLs")
    main_placeholder = st.empty()

    # Initialize session state variables
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "cache_db" not in st.session_state:
        st.session_state.cache_db = cache_conn

    if process_url_clicked:
        st.session_state.vectorstore = update_vectorstore(urls, vectorstore, main_placeholder)

    # Handle user queries
    query = st.text_input("Ask a question:")
    query = Cleaner.clean_sentence(query)#clean_sentence function is used to avoid new call to llm for smae ques with different case
    if st.button("Ask the Bot"):
        ask_question(query, st.session_state.vectorstore, llm, st.session_state.cache_db, main_placeholder)
        # Add a History button to toggle the sidebar
    if st.sidebar.button("History", key="history_toggle"):
        st.sidebar.write("")  # Force open the sidebar
        display_history()

if __name__ == "__main__":
    main()