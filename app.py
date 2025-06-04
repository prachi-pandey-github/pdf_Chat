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
from tenacity import retry, stop_after_attempt, wait_exponential

# Set Gemini API key from Streamlit secrets
genai.configure(api_key=st.secrets["API_KEY"]) 

# Set page config
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📚",
    layout="wide"
)
# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def embed_with_retry(embeddings, text):
    """Embed text with retry logic."""
    return embeddings.embed_query(text)

def create_vectorstore(text):
    """Create vector store from text."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Further reduced chunk size
        chunk_overlap=20,  # Further reduced overlap
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        max_retries=3,
        timeout=30,
        request_timeout=30
    )
    
    # Process chunks in very small batches
    batch_size = 3  # Reduced batch size
    all_embeddings = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            status_text.text(f"Processing batch {i//batch_size + 1} of {(len(chunks) + batch_size - 1)//batch_size}")
            batch_embeddings = []
            for chunk in batch:
                embedding = embed_with_retry(embeddings, chunk)
                batch_embeddings.append(embedding)
                time.sleep(1)  # Add delay between requests
            all_embeddings.extend(batch_embeddings)
            progress_bar.progress(min((i + batch_size) / len(chunks), 1.0))
        except Exception as e:
            st.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
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
