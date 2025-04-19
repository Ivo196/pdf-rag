"""
Data processing module for handling document loading and text splitting.
This module provides functions for processing PDF files and web content.
"""

from dotenv import load_dotenv
import os 
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader

def get_document_from_pdf(pdf_file) -> list[Document]:
    """
    Extracts text content from a PDF file and converts it into a Document object.
    
    Args:
        pdf_file: The uploaded PDF file object
        
    Returns:
        list[Document]: A list containing a single Document with the PDF's content
        
    Raises:
        Exception: If there's an error reading the PDF file
    """
   
    reader = PdfReader(pdf_file)
    full_text = ""
    # Extract text from each page
    for page in reader.pages: 
        text = page.extract_text()
        full_text += text 
        # Wrap the text into a Document object
        return [Document(page_content=full_text)]
    

def get_document_from_web(url: str) -> list[Document]:
    """
    Loads content from a web URL and converts it into Document objects.
    
    Args:
        url (str): The URL to load content from
        
    Returns:
        list[Document]: A list of Documents containing the web content
        
    Raises:
        Exception: If there's an error loading the web content
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def splitter(docs: list[Document]) -> list[Document]:
    """
    Splits documents into smaller chunks for better processing and retrieval.
    
    Args:
        docs (list[Document]): List of documents to split
        
    Returns:
        list[Document]: List of split document chunks
        
    Raises:
        Exception: If there's an error splitting the documents
    """
    # Configure the text splitter with appropriate chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # Number of characters per chunk
        chunk_overlap=20     # Number of characters to overlap between chunks
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

