import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def create_vectorstore(text):
    """Create vector store from text."""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def create_conversation_chain(vectorstore):
    """Create conversation chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp-01-21",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Main UI
st.title("📚 PDF Chatbot")
st.write("Upload a PDF and ask questions about its content!")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Process PDF
    with st.spinner("Processing PDF..."):
        # Extract text
        text = extract_text_from_pdf(uploaded_file)
        
        # Create vector store
        st.session_state.vectorstore = create_vectorstore(text)
        
        # Create conversation chain
        st.session_state.conversation = create_conversation_chain(st.session_state.vectorstore)
        
    st.success("PDF processed successfully! You can now ask questions.")

# Chat interface
if st.session_state.conversation is not None:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": prompt})
                st.write(response["answer"])
                
        # Add AI response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]}) 