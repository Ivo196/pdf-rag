"""
PDF RAG Assistant - A Streamlit application for interacting with PDF documents and web content
using Retrieval Augmented Generation (RAG) technology.
"""

import streamlit as st
from chain import chat_ai
from data import get_document_from_web, splitter, get_document_from_pdf
from vector_db import vectorStore
from urllib.parse import urlparse
import re

def is_valid_url(url: str) -> bool:
    """
    Validates if a given string is a valid URL.
    
    Args:
        url (str): The URL string to validate
        
    Returns:
        bool: True if the URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        # Check if the URL has a scheme (http or https) and a netloc (domain)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="PDF RAG Assistant 🤖",
    page_icon="📚",
    # layout="wide"
)

# Title and description
st.title("PDF RAG Assistant 🤖")
st.markdown("""
    Upload a PDF or enter a URL to ask questions about its content.
    The assistant will use RAG (Retrieval Augmented Generation) to provide accurate answers.
""")

# Sidebar configuration
st.sidebar.title("Upload your PDF file")

# URL input section
url = st.sidebar.text_input('Enter an URL', type='default', placeholder='https://es.wikipedia.org/wiki/Wifi')
if url:
    if not is_valid_url(url):
        st.sidebar.error('Please enter a valid URL (must start with http:// or https://)')
    else:
        # Button to download and process URL content
        if st.sidebar.button('Download data from URL', type='primary'):
            try:
                docs = get_document_from_web(url=url)
                split_docs = splitter(docs)
                vectorStore = vectorStore().add_documents(split_docs)
                st.sidebar.markdown(f'{len(split_docs)} chunks created')
                st.sidebar.success('Data downloaded successfully')
            except Exception as e:
                st.sidebar.error(f'Error processing URL: {str(e)}')

# PDF file upload section
pdf_file = st.sidebar.file_uploader('Upload a PDF file', type='pdf')

if pdf_file:
    # Button to process uploaded PDF
    if st.sidebar.button('Upload PDF file', type='primary'):
        full_text = get_document_from_pdf(pdf_file)
        split_docs = splitter(full_text)
        vectorStore = vectorStore().add_documents(split_docs)
        st.sidebar.markdown(f'{len(split_docs)} chunks created')
        st.sidebar.success('Data uploaded successfully')

# Chat interface section
st.header('Chat with your docs')
with st.chat_message('assistant'):
    st.markdown('Hello! I am the PDF RAG Assistant. How can I help you today?')

# Initialize chat session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat input and response handling
user_input = st.chat_input('Enter your question here...')

if user_input:
    with st.chat_message('user'):
        st.markdown(user_input)
    
    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    with st.chat_message('assistant'):
        thinking_message = st.empty()
        thinking_message.markdown("🤔 Thinking...")

    # Generate response using RAG
    response = chat_ai(user_input)

    # Display assistant response
    thinking_message.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({'role':'assistant', 'content': response})

# Clear chat button
if st.sidebar.button('Clear chat', type='secondary'):
    st.session_state.messages = []
    st.rerun()