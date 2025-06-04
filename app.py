import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import time
import hashlib
import json
from pathlib import Path

# Set Gemini API key from Streamlit secrets
genai.configure(api_key=st.secrets["API_KEY"]) 

# Create cache directory
CACHE_DIR = Path("embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(text):
    """Get cache file path for text."""
    hash_object = hashlib.md5(text.encode())
    return CACHE_DIR / f"{hash_object.hexdigest()}.json"

def get_cached_embedding(text):
    """Get cached embedding if exists."""
    cache_path = get_cache_path(text)
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None

def save_embedding(text, embedding):
    """Save embedding to cache."""
    cache_path = get_cache_path(text)
    with open(cache_path, 'w') as f:
        json.dump(embedding, f)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_vectorstore(text):
    """Create vector store from text."""
    # Split into very small chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # Very small chunks
        chunk_overlap=10,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        max_retries=2,
        timeout=20
    )
    
    all_embeddings = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        try:
            status_text.text(f"Processing chunk {i+1} of {len(chunks)}")
            
            # Try to get from cache first
            cached_embedding = get_cached_embedding(chunk)
            if cached_embedding:
                embedding = cached_embedding
            else:
                # If not in cache, get from API
                embedding = embeddings.embed_query(chunk)
                save_embedding(chunk, embedding)
                time.sleep(2)  # Longer delay between API calls
            
            all_embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(chunks))
            
        except Exception as e:
            st.error(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_embeddings:
        st.error("Failed to generate embeddings. Please try again with a smaller PDF.")
        return None
        
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(chunks, all_embeddings)),
        embedding=embeddings,
        metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))]
    )
    return vectorstore

def create_conversation_chain(vectorstore):
    """Create conversation chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        convert_system_message_to_human=True,
        max_retries=3,
        timeout=60
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        max_tokens_limit=4000
    )
    return conversation_chain

# Main UI
st.title("📚 PDF Chatbot")
st.write("Upload a PDF and ask questions about its content!")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        vectorstore = create_vectorstore(text)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation = create_conversation_chain(st.session_state.vectorstore)
            st.success("PDF processed successfully! You can now ask questions.")
        else:
            st.error("Failed to process PDF. Please try again with a smaller file.")

# Chat interface
if st.session_state.conversation is not None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask a question about your PDF"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation({"question": prompt})
                    st.write(response["answer"])
                    st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}. Please try again.")
