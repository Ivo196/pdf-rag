import streamlit as st
import ollama
from app import chat_ai




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
url = st.sidebar.text_input('Enter an URL', type='default')

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
    thinking_message.markdown(response['message']['content'])
    
    #add assistant response to chat history
    st.session_state.messages.append({'role':'assistant', 'content': response['message']['content']})
    print(st.session_state.messages)

