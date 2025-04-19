import streamlit as st
from chain import chat_ai
from data import get_document_from_web, splitter, get_document_from_pdf
from vector_db import vectorStore
from urllib.parse import urlparse
import re

def is_valid_url(url):
    try:
        result = urlparse(url)
        # Check if the URL has a scheme (http or https) and a netloc (domain)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

st.set_page_config(
    page_title="PDF RAG Assistant 🤖",
    page_icon="📚",
    # layout="wide"
)

#Title and description
st.title("PDF RAG Assistant 🤖")
st.markdown("""
    Upload a PDF or enter a URL to ask questions about its content.
    The assistant will use RAG (Retrieval Augmented Generation) to provide accurate answers.
""")

#Sidebar 
st.sidebar.title("Upload your PDF file")


#URL input 
url = st.sidebar.text_input('Enter an URL', type='default', placeholder='https://es.wikipedia.org/wiki/Wifi')
if url:
    if not is_valid_url(url):
        st.sidebar.error('Please enter a valid URL (must start with http:// or https://)')
    else:
        #add button to download URL data
        if st.sidebar.button('Download data from URL', type='primary'):
            try:
                docs = get_document_from_web(url=url)
                split_docs = splitter(docs)
                vectorStore = vectorStore().add_documents(split_docs)
                st.sidebar.markdown(f'{len(split_docs)} chunks created')
                st.sidebar.success('Data downloaded successfully')
            except Exception as e:
                st.sidebar.error(f'Error processing URL: {str(e)}')

#PDF Input
pdf_file = st.sidebar.file_uploader('Upload a PDF file', type='pdf')

if pdf_file:
    #add button to upload PDF file
    if st.sidebar.button('Upload PDF file', type='primary'):
        full_text = get_document_from_pdf(pdf_file)
        split_docs = splitter(full_text)
        vectorStore = vectorStore().add_documents(split_docs)
        st.sidebar.markdown(f'{len(split_docs)} chunks created')
        st.sidebar.success('Data uploaded successfully')

#Chat interface
st.header('Chat with your docs')
with st.chat_message('assistant'):
    st.markdown('Hello! I am the PDF RAG Assistant. How can I help you today?')

#chat session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

#display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

user_input = st.chat_input('Enter your question here...')

if user_input:
    with st.chat_message('user'):
        st.markdown(user_input)
    
    #add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    with st.chat_message('assistant'):
        thinking_message = st.empty()
        thinking_message.markdown("🤔 Thinking...")

    #generate response
    response = chat_ai(user_input)

    #display assistan response
    thinking_message.markdown(response)
    
    #add assistant response to chat history
    st.session_state.messages.append({'role':'assistant', 'content': response})
    print(st.session_state.messages)
#Clear chat button 
if st.sidebar.button('Clear chat', type='secondary'):
    st.session_state.messages = []
    st.rerun()