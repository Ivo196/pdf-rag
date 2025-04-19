"""
Vector database module for handling document embeddings and storage.
This module provides functionality for creating and managing the vector store.
"""

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Initialize Pinecone client and index
pc = Pinecone()
index = pc.Index('langchain')

def vectorStore():
    """
    Creates and returns a configured Pinecone vector store instance.
    The vector store is used for storing and retrieving document embeddings.
    
    Returns:
        PineconeVectorStore: A configured vector store instance
        
    Raises:
        ValueError: If required environment variables are not set
        Exception: If there's an error creating the vector store
    """
    # Initialize OpenAI embeddings model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Using the small embedding model for efficiency
    )
    
    # Create and return the vector store
    vectorStore = PineconeVectorStore(
        embedding=embeddings,
        index=index
    )
    #vectorStore.add_documents(docs)
    return vectorStore

