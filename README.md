# PDF Chatbot

A powerful chatbot that allows you to upload PDF documents and ask questions about their content. Built with LangChain, Google Gemini, Streamlit, and PyMuPDF.

## Features

- PDF document upload and processing
- Interactive chat interface
- Context-aware responses based on PDF content
- Conversation history
- Modern and user-friendly UI

## Prerequisites

- Python 3.8 or higher
- Google API key (for Gemini)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd pdf-chatbot
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your-api-key-here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload a PDF file using the file uploader

4. Once the PDF is processed, you can start asking questions about its content in the chat interface

## How it Works

1. The application uses PyMuPDF to extract text from the uploaded PDF
2. The text is split into chunks and converted into embeddings using Google's Gemini embedding model
3. A vector store (FAISS) is created to enable efficient similarity search
4. When you ask a question, the application:
   - Finds the most relevant text chunks from the PDF
   - Uses these chunks as context for the Gemini model
   - Generates a response based on the context and your question

## Note

Make sure you have a valid Google API key with access to the Gemini API. You can get one from the Google AI Studio (https://makersuite.google.com/app/apikey). 