"""
Chain module for implementing the Retrieval Augmented Generation (RAG) functionality.
This module handles the creation of the RAG chain and processing of user queries.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector_db import vectorStore
import os

def create_chain():
    """
    Creates and configures the RAG chain with the following components:
    1. OpenAI language model
    2. Custom prompt template
    3. Document chain
    4. Retrieval chain
    
    Returns:
        RetrievalChain: A configured RAG chain ready for processing queries
        
    Raises:
        ValueError: If required environment variables are not set
    """
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Initialize the OpenAI language model
    model = ChatOpenAI(
        model="gpt-4-turbo-preview",  # Using GPT-4 Turbo for better performance
        temperature=0.4,
    )
    
    # Create a custom prompt template for the RAG system
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant that answers questions based on the provided context.
    
    Context: {context}
    
    Instructions for using the context:
    1. Carefully analyze the provided context to find relevant information
    2. If the context contains the answer, use it directly
    3. If the context doesn't contain enough information, say so clearly
    4. If the context contains multiple relevant pieces, combine them coherently
    5. Always cite specific parts of the context when possible
    6. If the question is unclear or ambiguous, ask for clarification
    
    Question: {input}
    
    Please provide a clear, concise, and accurate answer based on the context above.
    If you're unsure about any part of the answer, acknowledge the uncertainty.
    """)

    # Create the document chain that processes the context
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
    )

    # Configure the retriever to fetch relevant documents
    retriever = vectorStore().as_retriever(search_kwargs={'k': 3})
    
    # Combine the retriever and document chain into a retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

def chat_ai(message: str) -> str:
    """
    Processes a user message through the RAG system and returns a response.
    
    Args:
        message (str): The user's question or input
        
    Returns:
        str: The AI's response based on the retrieved context
        
    Raises:
        Exception: If there's an error processing the message
    """
    try:
        # Create the RAG chain
        chain = create_chain()
        
        # Process the message and get the response
        response = chain.invoke({
            'input': message
        })
        
        return response['answer']
    except Exception as e:
        return f"Error processing your question: {str(e)}"
